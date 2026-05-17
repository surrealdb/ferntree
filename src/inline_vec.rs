//! # Inline-only fixed-capacity vector
//!
//! [`InlineVec<T, N>`] is a `Vec`-like container that always stores its
//! elements inline in the struct itself — no heap fallback. Capacity is
//! fixed at compile time by the const generic `N`; pushing past it is a
//! contract violation (debug-asserts; in release builds the tree never
//! does this because nodes split at the limit).
//!
//! ## Why ferntree uses this instead of [`smallvec::SmallVec`]
//!
//! The optimistic read fast path reads `len` and the data pointer of the
//! storage held inside a B+ tree leaf or internal node without acquiring a
//! lock. The version-recheck protocol on the surrounding [`crate::latch::HybridLatch`]
//! catches torn reads at validate time. For this to be sound under
//! Tree Borrows, the read must NOT create an `&` reference to the storage:
//! a `&` retag would race with a concurrent writer's non-atomic write
//! (Tree Borrows permits the compiler to speculate a load through the
//! retag, regardless of whether the source code actually reads from it).
//!
//! `smallvec::SmallVec` only exposes its length and data through `&self`
//! methods, so it cannot be used safely on the optimistic fast path.
//! `InlineVec` is `#[repr(C)]` with explicit `len` / `data` fields, and
//! exposes [`InlineVec::raw_len`] / [`InlineVec::raw_data_ptr`] which
//! project from a `*const Self` without any reborrow.
//!
//! ## Layout
//!
//! ```text
//! offset 0: len: u16
//! offset 2: padding (alignof T)
//! offset _: data: [MaybeUninit<T>; N]
//! ```
//!
//! `#[repr(C)]` guarantees the field order. `len` is `u16` to match the
//! existing `u16` length encoding used throughout `LeafNode` /
//! `InternalNode`.

use core::cmp::Ordering;
use core::fmt;
use core::iter::FusedIterator;
use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut, RangeBounds};
use core::ptr;
use core::slice;

/// A fixed-capacity, inline-storage `Vec`-like container.
///
/// See the [module-level docs](self) for the motivation.
///
/// # Invariants
///
/// - `len <= N`.
/// - The first `len` slots of `data` are initialised.
/// - `len` is the only mutable counter; reads from concurrent threads see
///   either the pre- or post-mutation value (single aligned `u16` store).
#[repr(C)]
pub(crate) struct InlineVec<T, const N: usize> {
	/// Number of initialised elements at the front of `data`.
	len: u16,
	/// Inline storage. The first `len` slots are initialised; the rest are
	/// `MaybeUninit::uninit()`.
	data: [MaybeUninit<T>; N],
}

// SAFETY: `InlineVec<T, N>` owns its elements by value. Sending it to
// another thread moves ownership of the data, so `T: Send` is sufficient.
unsafe impl<T: Send, const N: usize> Send for InlineVec<T, N> {}

// SAFETY: To share `&InlineVec<T, N>` between threads, two threads need
// concurrent access to the underlying `T`s. The slice projection
// (`as_slice`) yields `&[T]`, which requires `T: Sync`.
unsafe impl<T: Sync, const N: usize> Sync for InlineVec<T, N> {}

impl<T, const N: usize> InlineVec<T, N> {
	/// Creates a new empty `InlineVec`.
	#[inline]
	pub(crate) const fn new() -> Self {
		// Capacity must fit in u16 — checked at compile time via the
		// `len` field's type.
		const {
			assert!(N <= u16::MAX as usize, "InlineVec capacity exceeds u16::MAX");
		}
		Self {
			len: 0,
			// SAFETY: `[MaybeUninit<T>; N]` is itself `MaybeUninit`-safe.
			data: [const { MaybeUninit::uninit() }; N],
		}
	}

	/// Returns the number of initialised elements.
	#[inline]
	pub(crate) fn len(&self) -> usize {
		self.len as usize
	}

	/// Returns `true` if there are no initialised elements.
	#[inline]
	pub(crate) fn is_empty(&self) -> bool {
		self.len == 0
	}

	/// Returns the compile-time capacity.
	#[inline]
	pub(crate) fn capacity(&self) -> usize {
		N
	}

	/// Returns a slice over the initialised prefix.
	#[inline]
	pub(crate) fn as_slice(&self) -> &[T] {
		// SAFETY: the first `len` slots are initialised by the type
		// invariant; reading them as `[T]` is sound.
		unsafe { slice::from_raw_parts(self.data.as_ptr() as *const T, self.len as usize) }
	}

	/// Returns a mutable slice over the initialised prefix.
	#[inline]
	pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
		// SAFETY: see `as_slice`; we hold `&mut self`, so no aliasing.
		unsafe { slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len as usize) }
	}

	/// Returns a raw pointer to the first element.
	#[inline]
	pub(crate) fn as_ptr(&self) -> *const T {
		self.data.as_ptr() as *const T
	}

	/// Returns a mutable raw pointer to the first element.
	#[inline]
	#[allow(dead_code)] // surfaced for parity with SmallVec; reserved for future callers
	pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
		self.data.as_mut_ptr() as *mut T
	}

	/// Returns a reference to the element at `idx`, or `None` if out of bounds.
	#[inline]
	pub(crate) fn get(&self, idx: usize) -> Option<&T> {
		self.as_slice().get(idx)
	}

	/// Returns a reference to the element at `idx`, without bounds checking.
	///
	/// # Safety
	///
	/// `idx` must be less than [`len`](Self::len).
	#[inline]
	pub(crate) unsafe fn get_unchecked(&self, idx: usize) -> &T {
		// SAFETY: caller upholds `idx < len`.
		unsafe { &*self.data.as_ptr().add(idx).cast::<T>() }
	}

	/// Returns a mutable reference to the element at `idx`, without bounds checking.
	///
	/// # Safety
	///
	/// `idx` must be less than [`len`](Self::len).
	#[inline]
	pub(crate) unsafe fn get_unchecked_mut(&mut self, idx: usize) -> &mut T {
		// SAFETY: caller upholds `idx < len`.
		unsafe { &mut *self.data.as_mut_ptr().add(idx).cast::<T>() }
	}

	/// Appends an element to the back.
	///
	/// # Panics (debug)
	///
	/// Debug-asserts that `len < N`. In release this is UB by contract —
	/// callers (B+ tree nodes) split before reaching capacity.
	#[inline]
	pub(crate) fn push(&mut self, value: T) {
		debug_assert!((self.len as usize) < N, "InlineVec capacity exceeded");
		// SAFETY: `len < N` per debug_assert; the slot is uninitialised.
		unsafe {
			self.data.as_mut_ptr().add(self.len as usize).write(MaybeUninit::new(value));
		}
		self.len += 1;
	}

	/// Removes the last element and returns it, or `None` if empty.
	#[inline]
	pub(crate) fn pop(&mut self) -> Option<T> {
		if self.len == 0 {
			return None;
		}
		self.len -= 1;
		// SAFETY: just decremented; slot at new `len` is initialised.
		Some(unsafe { self.data.as_ptr().add(self.len as usize).read().assume_init() })
	}

	/// Inserts `value` at position `pos`, shifting later elements right.
	///
	/// # Panics
	///
	/// Panics if `pos > len`. Debug-asserts that there is capacity.
	pub(crate) fn insert(&mut self, pos: usize, value: T) {
		let len = self.len as usize;
		assert!(pos <= len, "InlineVec::insert index out of bounds");
		debug_assert!(len < N, "InlineVec capacity exceeded");
		// SAFETY: pos <= len <= N - 1 (per debug_assert). Shift elements
		// [pos..len) right by one, then write into the freed slot.
		unsafe {
			let base = self.data.as_mut_ptr();
			ptr::copy(base.add(pos), base.add(pos + 1), len - pos);
			base.add(pos).write(MaybeUninit::new(value));
		}
		self.len += 1;
	}

	/// Removes and returns the element at `pos`, shifting later elements left.
	///
	/// # Panics
	///
	/// Panics if `pos >= len`.
	pub(crate) fn remove(&mut self, pos: usize) -> T {
		let len = self.len as usize;
		assert!(pos < len, "InlineVec::remove index out of bounds");
		// SAFETY: pos < len, so the slot is initialised. Shift elements
		// [pos+1..len) left by one, then decrement len.
		unsafe {
			let base = self.data.as_mut_ptr();
			let value = base.add(pos).read().assume_init();
			ptr::copy(base.add(pos + 1), base.add(pos), len - pos - 1);
			self.len -= 1;
			value
		}
	}

	/// Extends with items from the iterator.
	///
	/// # Panics (debug)
	///
	/// Debug-asserts on capacity overflow.
	pub(crate) fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
		for value in iter {
			self.push(value);
		}
	}

	/// Returns an iterator over the elements.
	#[inline]
	pub(crate) fn iter(&self) -> slice::Iter<'_, T> {
		self.as_slice().iter()
	}

	/// Returns a mutable iterator over the elements.
	#[inline]
	#[allow(dead_code)] // surfaced for parity with SmallVec; reserved for future callers
	pub(crate) fn iter_mut(&mut self) -> slice::IterMut<'_, T> {
		self.as_mut_slice().iter_mut()
	}

	/// Returns a draining iterator that removes the specified range.
	///
	/// On drop, any non-iterated elements are dropped, and the elements
	/// after the range shift down to fill the gap.
	///
	/// # Panics
	///
	/// Panics if `range` is out of bounds for `len`.
	pub(crate) fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> Drain<'_, T, N> {
		use core::ops::Bound::*;
		let len = self.len as usize;
		let start = match range.start_bound() {
			Included(&n) => n,
			Excluded(&n) => n + 1,
			Unbounded => 0,
		};
		let end = match range.end_bound() {
			Included(&n) => n + 1,
			Excluded(&n) => n,
			Unbounded => len,
		};
		assert!(start <= end, "InlineVec::drain start > end");
		assert!(end <= len, "InlineVec::drain end > len");
		// Pre-shrink the logical length to `start` so a panicking drop in
		// a yielded element doesn't leave the vec in an inconsistent state
		// (matches Vec::drain's discipline).
		self.len = start as u16;
		Drain {
			vec: self,
			start,
			next_idx: start,
			end,
			original_len: len,
		}
	}

	// ---- Raw-pointer projection helpers (optimistic-read fast path) ----

	/// Reads `len` via raw pointer projection, without creating an `&Self`
	/// reborrow.
	///
	/// # Safety
	///
	/// `this` must be a valid (aligned, dereferenceable) pointer to an
	/// `InlineVec<T, N>`. The caller must validate consistency of the read
	/// against the surrounding [`crate::latch::HybridLatch`] version
	/// before acting on the result.
	#[inline]
	#[allow(dead_code)] // used in commit 2 by the raw-projection descent
	pub(crate) unsafe fn raw_len(this: *const Self) -> u16 {
		// SAFETY: `len` is the first field of a `#[repr(C)]` struct, so
		// `addr_of!((*this).len)` is a valid aligned pointer to a u16.
		// `ptr::read` performs an unsynchronised aligned load — no retag.
		unsafe { ptr::read(ptr::addr_of!((*this).len)) }
	}

	/// Returns a raw pointer to the first element, without creating an
	/// `&Self` reborrow.
	///
	/// # Safety
	///
	/// `this` must be a valid pointer to an `InlineVec<T, N>`. The returned
	/// pointer is valid for at least `N` elements but only the first `len`
	/// are initialised; the caller must validate the index it accesses
	/// against `raw_len` and against the surrounding version recheck before
	/// reading.
	#[inline]
	#[allow(dead_code)] // used in commit 2 by the raw-projection descent
	pub(crate) unsafe fn raw_data_ptr(this: *const Self) -> *const T {
		// SAFETY: `data` is the second field of a `#[repr(C)]` struct;
		// `addr_of!((*this).data)` is a valid pointer to the array. The
		// cast to `*const T` is sound because `[MaybeUninit<T>; N]` has
		// the same layout as `[T; N]`.
		unsafe { ptr::addr_of!((*this).data) as *const T }
	}
}

impl<T, const N: usize> Default for InlineVec<T, N> {
	#[inline]
	fn default() -> Self {
		Self::new()
	}
}

impl<T, const N: usize> Drop for InlineVec<T, N> {
	fn drop(&mut self) {
		// SAFETY: the first `len` slots are initialised.
		unsafe {
			let slice =
				slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len as usize);
			ptr::drop_in_place(slice);
		}
	}
}

impl<T, const N: usize> Deref for InlineVec<T, N> {
	type Target = [T];
	#[inline]
	fn deref(&self) -> &[T] {
		self.as_slice()
	}
}

impl<T, const N: usize> DerefMut for InlineVec<T, N> {
	#[inline]
	fn deref_mut(&mut self) -> &mut [T] {
		self.as_mut_slice()
	}
}

impl<T, const N: usize, I: slice::SliceIndex<[T]>> core::ops::Index<I> for InlineVec<T, N> {
	type Output = I::Output;
	#[inline]
	fn index(&self, idx: I) -> &Self::Output {
		core::ops::Index::index(self.as_slice(), idx)
	}
}

impl<T, const N: usize, I: slice::SliceIndex<[T]>> core::ops::IndexMut<I> for InlineVec<T, N> {
	#[inline]
	fn index_mut(&mut self, idx: I) -> &mut Self::Output {
		core::ops::IndexMut::index_mut(self.as_mut_slice(), idx)
	}
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for InlineVec<T, N> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_list().entries(self.iter()).finish()
	}
}

impl<T: Clone, const N: usize> Clone for InlineVec<T, N> {
	fn clone(&self) -> Self {
		let mut new = Self::new();
		for v in self.iter() {
			new.push(v.clone());
		}
		new
	}
}

impl<T: PartialEq, const N: usize> PartialEq for InlineVec<T, N> {
	fn eq(&self, other: &Self) -> bool {
		self.as_slice() == other.as_slice()
	}
}

impl<T: Eq, const N: usize> Eq for InlineVec<T, N> {}

impl<T: PartialOrd, const N: usize> PartialOrd for InlineVec<T, N> {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		self.as_slice().partial_cmp(other.as_slice())
	}
}

impl<T: Ord, const N: usize> Ord for InlineVec<T, N> {
	fn cmp(&self, other: &Self) -> Ordering {
		self.as_slice().cmp(other.as_slice())
	}
}

impl<'a, T, const N: usize> IntoIterator for &'a InlineVec<T, N> {
	type Item = &'a T;
	type IntoIter = slice::Iter<'a, T>;
	fn into_iter(self) -> Self::IntoIter {
		self.iter()
	}
}

impl<'a, T, const N: usize> IntoIterator for &'a mut InlineVec<T, N> {
	type Item = &'a mut T;
	type IntoIter = slice::IterMut<'a, T>;
	fn into_iter(self) -> Self::IntoIter {
		self.iter_mut()
	}
}

// ============================================================================
// Drain
// ============================================================================

/// Draining iterator returned by [`InlineVec::drain`].
pub(crate) struct Drain<'a, T, const N: usize> {
	/// Mutable borrow of the source vec. On drop we restore `len` based
	/// on the elements left at the tail.
	vec: &'a mut InlineVec<T, N>,
	/// Index where the drain range begins (where remaining tail elements
	/// will be shifted to on drop).
	start: usize,
	/// Index of the next element to yield.
	next_idx: usize,
	/// One past the last index in the drain range.
	end: usize,
	/// Original `len` before the drain — used on drop to compute the tail.
	original_len: usize,
}

impl<'a, T, const N: usize> Iterator for Drain<'a, T, N> {
	type Item = T;

	fn next(&mut self) -> Option<T> {
		if self.next_idx >= self.end {
			return None;
		}
		let idx = self.next_idx;
		self.next_idx += 1;
		// SAFETY: idx in [start, end), the slot is initialised and has
		// not been yielded yet.
		Some(unsafe { self.vec.data.as_ptr().add(idx).read().assume_init() })
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		let rem = self.end - self.next_idx;
		(rem, Some(rem))
	}
}

impl<'a, T, const N: usize> ExactSizeIterator for Drain<'a, T, N> {}
impl<'a, T, const N: usize> FusedIterator for Drain<'a, T, N> {}

impl<'a, T, const N: usize> Drop for Drain<'a, T, N> {
	fn drop(&mut self) {
		// Drop any drained-but-not-yielded elements.
		for i in self.next_idx..self.end {
			// SAFETY: `i` is in the drain range, so the slot is initialised
			// and has not been yielded.
			unsafe {
				let _ = self.vec.data.as_ptr().add(i).read().assume_init();
			}
		}
		// Shift tail elements [end..original_len) down to [start..).
		let tail_len = self.original_len - self.end;
		if tail_len > 0 && self.start != self.end {
			// SAFETY: tail elements at [end..original_len) are initialised
			// and not aliased — we hold `&mut self.vec`.
			unsafe {
				let base = self.vec.data.as_mut_ptr();
				ptr::copy(base.add(self.end), base.add(self.start), tail_len);
			}
		}
		// Restore the vec's length to reflect what's left.
		self.vec.len = (self.start + tail_len) as u16;
	}
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn push_pop_len() {
		let mut v: InlineVec<u32, 4> = InlineVec::new();
		assert_eq!(v.len(), 0);
		assert!(v.is_empty());
		v.push(1);
		v.push(2);
		v.push(3);
		assert_eq!(v.len(), 3);
		assert_eq!(v.as_slice(), &[1, 2, 3]);
		assert_eq!(v.pop(), Some(3));
		assert_eq!(v.pop(), Some(2));
		assert_eq!(v.pop(), Some(1));
		assert_eq!(v.pop(), None);
	}

	#[test]
	fn insert_remove() {
		let mut v: InlineVec<u32, 8> = InlineVec::new();
		v.extend([10, 20, 30, 40]);
		v.insert(2, 25);
		assert_eq!(v.as_slice(), &[10, 20, 25, 30, 40]);
		assert_eq!(v.remove(1), 20);
		assert_eq!(v.as_slice(), &[10, 25, 30, 40]);
		assert_eq!(v.remove(0), 10);
		assert_eq!(v.as_slice(), &[25, 30, 40]);
	}

	#[test]
	fn insert_at_end() {
		let mut v: InlineVec<u32, 4> = InlineVec::new();
		v.push(1);
		v.insert(1, 2);
		assert_eq!(v.as_slice(), &[1, 2]);
	}

	#[test]
	fn drain_middle() {
		let mut v: InlineVec<u32, 8> = InlineVec::new();
		v.extend([1, 2, 3, 4, 5]);
		let drained: Vec<u32> = v.drain(1..4).collect();
		assert_eq!(drained, vec![2, 3, 4]);
		assert_eq!(v.as_slice(), &[1, 5]);
	}

	#[test]
	fn drain_full() {
		let mut v: InlineVec<u32, 8> = InlineVec::new();
		v.extend([1, 2, 3, 4, 5]);
		let drained: Vec<u32> = v.drain(..).collect();
		assert_eq!(drained, vec![1, 2, 3, 4, 5]);
		assert!(v.is_empty());
	}

	#[test]
	fn drain_unconsumed_drops_elements_and_shifts() {
		use std::sync::atomic::{AtomicUsize, Ordering};
		static DROPPED: AtomicUsize = AtomicUsize::new(0);

		struct DropCounter(u32);
		impl Drop for DropCounter {
			fn drop(&mut self) {
				DROPPED.fetch_add(1, Ordering::Relaxed);
			}
		}

		DROPPED.store(0, Ordering::Relaxed);
		let mut v: InlineVec<DropCounter, 8> = InlineVec::new();
		v.push(DropCounter(1));
		v.push(DropCounter(2));
		v.push(DropCounter(3));
		v.push(DropCounter(4));
		{
			let mut iter = v.drain(1..3);
			let _ = iter.next(); // yields DropCounter(2), drops it after caller drops
		} // iter dropped here, must drop DropCounter(3) and shift DropCounter(4) -> idx 1
		assert_eq!(v.len(), 2);
		assert_eq!(v.as_slice()[0].0, 1);
		assert_eq!(v.as_slice()[1].0, 4);
		// Drops so far: DropCounter(2) (yielded then dropped) + DropCounter(3) (in drain's drop)
		assert_eq!(DROPPED.load(Ordering::Relaxed), 2);
		drop(v);
		// Now DropCounter(1) and DropCounter(4) get dropped.
		assert_eq!(DROPPED.load(Ordering::Relaxed), 4);
	}

	#[test]
	fn deref_slice_apis() {
		let mut v: InlineVec<u32, 8> = InlineVec::new();
		v.extend([5, 3, 1, 4, 2]);
		assert_eq!(v.iter().copied().min(), Some(1));
		assert_eq!(v[0], 5);
		v[0] = 9;
		assert_eq!(v.as_slice(), &[9, 3, 1, 4, 2]);
	}

	#[test]
	fn drop_runs_for_initialised_elements_only() {
		use std::sync::atomic::{AtomicUsize, Ordering};
		static DROPPED: AtomicUsize = AtomicUsize::new(0);

		struct DropCounter;
		impl Drop for DropCounter {
			fn drop(&mut self) {
				DROPPED.fetch_add(1, Ordering::Relaxed);
			}
		}

		DROPPED.store(0, Ordering::Relaxed);
		{
			let mut v: InlineVec<DropCounter, 16> = InlineVec::new();
			v.push(DropCounter);
			v.push(DropCounter);
			v.push(DropCounter);
		}
		assert_eq!(DROPPED.load(Ordering::Relaxed), 3);
	}

	#[test]
	fn raw_accessors_match_safe_accessors() {
		let mut v: InlineVec<u32, 8> = InlineVec::new();
		v.extend([10, 20, 30]);

		// SAFETY: pointer derived from a valid InlineVec; not aliased.
		let raw_len = unsafe { InlineVec::raw_len(&v as *const _) };
		assert_eq!(raw_len, 3);

		// SAFETY: same as above; we read three initialised u32s.
		unsafe {
			let data: *const u32 = InlineVec::raw_data_ptr(&v as *const _);
			assert_eq!(ptr::read(data.add(0)), 10);
			assert_eq!(ptr::read(data.add(1)), 20);
			assert_eq!(ptr::read(data.add(2)), 30);
		}
	}
}
