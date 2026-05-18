//! # Per-type atomic storage for optimistic-read fields
//!
//! This module provides the storage primitives used by leaf and internal
//! nodes to hold their `K` and `V` arrays in a form that satisfies Miri's
//! data-race detector under the optimistic-read protocol.
//!
//! The optimistic fast path skips the leaf's shared lock and validates
//! reads via the [`crate::latch::HybridLatch`] version-recheck protocol.
//! The version recheck catches *application-level* inconsistency, but the
//! C/Rust memory model requires that the underlying loads / stores
//! themselves be properly synchronised. This module supplies that bottom
//! layer: every slot is backed by an atomic that the writer mutates with
//! `Release` / `AcqRel` ordering and the reader observes with `Acquire`.
//!
//! ## Two storage strategies
//!
//! Each type that opts into optimistic reads picks one of:
//!
//! - [`InlineSlot<T>`] — for `T: AtomicLoadable` (primitive integers /
//!   floats whose size and alignment match one of `AtomicU{8,16,32,64}`).
//!   The value lives inline; loads use `AtomicXX::load(Acquire)` and
//!   transmute the bits to `T`. Zero allocation overhead.
//!
//! - [`BoxedSlot<T>`] — for `T: Clone + Send + 'static`. The value lives
//!   in a `Box<T>` referenced via `AtomicPtr<T>`. Loads do
//!   `AtomicPtr::load(Acquire)` + clone through the pointer. Writes
//!   allocate a fresh `Box<T>`, swap the pointer in atomically, and the
//!   displaced pointer is routed through the epoch GC so a concurrent
//!   reader's `&T` borrow stays valid until no reader could still hold it.
//!
//! The choice is encoded as the `Slot` associated type on
//! [`crate::optimistic::OptimisticRead`].
//!
//! ## Slot lifecycle
//!
//! Each slot has two logical states: **empty** (no value) and **init**
//! (holds a value). The surrounding leaf / internal node tracks which
//! slots are init via its `len` field. The slot itself does not track its
//! own state — the operations are split into:
//!
//! - [`OptimisticSlot::store_into_empty`]: write a value into an empty
//!   slot, transitioning it to init.
//! - [`OptimisticSlot::swap_init`]: replace the value in an init slot,
//!   returning the old.
//! - [`OptimisticSlot::take_init`]: read out the value of an init slot,
//!   transitioning it to empty.
//! - [`OptimisticSlot::load`]: read a copy of the value out of an init
//!   slot (does not transition state).
//! - [`OptimisticSlot::drop_in_place`]: drop the contents during the
//!   parent's destructor, given exclusive ownership.
//!
//! All read paths use `load`. Writers under exclusive lock use
//! `store_into_empty` / `swap_init` / `take_init` depending on whether
//! the target slot is currently init.

use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::ptr;
use core::sync::atomic::{
	AtomicI16, AtomicI32, AtomicI64, AtomicI8, AtomicPtr, AtomicU16, AtomicU32, AtomicU64,
	AtomicU8, Ordering,
};

// ---------------------------------------------------------------------------
// Sealing
// ---------------------------------------------------------------------------

mod seal {
	pub trait Sealed {}
}

// ---------------------------------------------------------------------------
// AtomicLoadable: bridge from a `Copy` primitive type to its stdlib atomic
// ---------------------------------------------------------------------------

/// A `Copy` type whose size and alignment match a stdlib atomic, enabling
/// inline atomic storage.
///
/// Sealed: implemented only for the stdlib primitives we know to be sound
/// to load atomically and reinterpret as `Self`. In particular, every bit
/// pattern of the underlying integer storage must produce a valid
/// `Self` — which rules out types like `bool` (only `0` and `1` are
/// valid) and `char`.
///
/// # Safety
///
/// Implementors must guarantee:
///
/// - `Self` and `Atomic` have the same size and alignment.
/// - Every bit pattern of `Atomic`'s integer storage is a valid `Self`.
/// - `Atomic`'s atomic loads / stores commute with concurrent atomic ops
///   from any thread (true for all stdlib atomics).
pub unsafe trait AtomicLoadable:
	Copy + Send + Sync + 'static + Sized + seal::Sealed
{
	/// The stdlib atomic type whose loads and stores back this `Self`.
	type Atomic: Send + Sync + 'static;

	/// Construct a fresh atomic holding the bit-pattern of `value`.
	fn new_atomic(value: Self) -> Self::Atomic;

	/// `Acquire`-load the atomic and reinterpret as `Self`.
	fn load_acquire(atomic: &Self::Atomic) -> Self;

	/// `Release`-store `value` into the atomic.
	fn store_release(atomic: &Self::Atomic, value: Self);

	/// `AcqRel`-swap `value` into the atomic, returning the previous value
	/// reinterpreted as `Self`.
	fn swap_acqrel(atomic: &Self::Atomic, value: Self) -> Self;
}

macro_rules! impl_atomic_loadable_int {
	($($prim:ty => $atomic:ty),* $(,)?) => {
		$(
			impl seal::Sealed for $prim {}
			// SAFETY: matched sizes/aligns and all-bit-patterns-valid for
			// every primitive integer.
			unsafe impl AtomicLoadable for $prim {
				type Atomic = $atomic;
				#[inline]
				fn new_atomic(value: Self) -> Self::Atomic { <$atomic>::new(value as _) }
				#[inline]
				fn load_acquire(a: &Self::Atomic) -> Self { a.load(Ordering::Acquire) as _ }
				#[inline]
				fn store_release(a: &Self::Atomic, value: Self) { a.store(value as _, Ordering::Release) }
				#[inline]
				fn swap_acqrel(a: &Self::Atomic, value: Self) -> Self {
					a.swap(value as _, Ordering::AcqRel) as _
				}
			}
		)*
	};
}

impl_atomic_loadable_int!(
	u8  => AtomicU8,
	i8  => AtomicI8,
	u16 => AtomicU16,
	i16 => AtomicI16,
	u32 => AtomicU32,
	i32 => AtomicI32,
	u64 => AtomicU64,
	i64 => AtomicI64,
);

// usize / isize on 64-bit platforms — gated to avoid mismatched widths.
#[cfg(target_pointer_width = "64")]
mod size_atomics {
	use super::*;
	impl seal::Sealed for usize {}
	// SAFETY: 64-bit pointer width: usize == u64 in size and alignment.
	unsafe impl AtomicLoadable for usize {
		type Atomic = AtomicU64;
		#[inline]
		fn new_atomic(value: Self) -> Self::Atomic {
			AtomicU64::new(value as u64)
		}
		#[inline]
		fn load_acquire(a: &Self::Atomic) -> Self {
			a.load(Ordering::Acquire) as usize
		}
		#[inline]
		fn store_release(a: &Self::Atomic, value: Self) {
			a.store(value as u64, Ordering::Release)
		}
		#[inline]
		fn swap_acqrel(a: &Self::Atomic, value: Self) -> Self {
			a.swap(value as u64, Ordering::AcqRel) as usize
		}
	}
	impl seal::Sealed for isize {}
	// SAFETY: 64-bit pointer width: isize == i64 in size and alignment.
	unsafe impl AtomicLoadable for isize {
		type Atomic = AtomicI64;
		#[inline]
		fn new_atomic(value: Self) -> Self::Atomic {
			AtomicI64::new(value as i64)
		}
		#[inline]
		fn load_acquire(a: &Self::Atomic) -> Self {
			a.load(Ordering::Acquire) as isize
		}
		#[inline]
		fn store_release(a: &Self::Atomic, value: Self) {
			a.store(value as i64, Ordering::Release)
		}
		#[inline]
		fn swap_acqrel(a: &Self::Atomic, value: Self) -> Self {
			a.swap(value as i64, Ordering::AcqRel) as isize
		}
	}
}

#[cfg(target_pointer_width = "32")]
mod size_atomics {
	use super::*;
	impl seal::Sealed for usize {}
	unsafe impl AtomicLoadable for usize {
		type Atomic = AtomicU32;
		#[inline]
		fn new_atomic(value: Self) -> Self::Atomic {
			AtomicU32::new(value as u32)
		}
		#[inline]
		fn load_acquire(a: &Self::Atomic) -> Self {
			a.load(Ordering::Acquire) as usize
		}
		#[inline]
		fn store_release(a: &Self::Atomic, value: Self) {
			a.store(value as u32, Ordering::Release)
		}
		#[inline]
		fn swap_acqrel(a: &Self::Atomic, value: Self) -> Self {
			a.swap(value as u32, Ordering::AcqRel) as usize
		}
	}
	impl seal::Sealed for isize {}
	unsafe impl AtomicLoadable for isize {
		type Atomic = AtomicI32;
		#[inline]
		fn new_atomic(value: Self) -> Self::Atomic {
			AtomicI32::new(value as i32)
		}
		#[inline]
		fn load_acquire(a: &Self::Atomic) -> Self {
			a.load(Ordering::Acquire) as isize
		}
		#[inline]
		fn store_release(a: &Self::Atomic, value: Self) {
			a.store(value as i32, Ordering::Release)
		}
		#[inline]
		fn swap_acqrel(a: &Self::Atomic, value: Self) -> Self {
			a.swap(value as i32, Ordering::AcqRel) as isize
		}
	}
}

// ---------------------------------------------------------------------------
// OptimisticSlot trait
// ---------------------------------------------------------------------------

/// Per-slot atomic storage for an optimistically-readable type.
///
/// Each slot has two logical states — **empty** and **init** — tracked by
/// the surrounding leaf via `len`. See module docs for the lifecycle.
///
/// # Safety
///
/// Implementors must ensure that:
///
/// - [`load`](Self::load) and [`swap_init`](Self::swap_init) /
///   [`store_into_empty`](Self::store_into_empty) are properly
///   synchronised so that concurrent loads observe well-formed values.
/// - The state-transition contract (empty ↔ init) is upheld: methods that
///   require an empty slot must not leak a previous occupant, and methods
///   that require an init slot must not drop an uninitialised one.
pub unsafe trait OptimisticSlot: Default + Send + Sync + Sized {
	/// The value type held in this slot.
	type Value;

	/// Attempt an atomic load that tolerates a concurrently emptied
	/// slot. For inline storage this is equivalent to
	/// [`load`](Self::load) wrapped in `Some`. For boxed storage it
	/// returns `None` if the underlying `AtomicPtr` is null (i.e. the
	/// slot was emptied between the caller observing `len` and the
	/// load), letting the caller treat it as a recheck failure rather
	/// than dereferencing a null pointer.
	///
	/// # Safety
	///
	/// - The caller must not assume the slot is init (the whole point
	///   of this method is to tolerate concurrent emptying).
	/// - For inline storage the load returns the current bit-pattern
	///   regardless of init state; the caller's outer recheck disambiguates.
	unsafe fn try_load(&self) -> Option<Self::Value>;

	/// Same as [`try_load`](Self::try_load) but through a raw pointer,
	/// without ever creating an `&Self` reborrow. Used by the optimistic-
	/// read fast path under raw-pointer projection so that the read does
	/// not conflict with the writer's protected `&mut` tag on the
	/// surrounding node under Tree Borrows.
	///
	/// The trait's default implementation reborrows via `&*this`, which
	/// is fine for slot types that consist solely of interior-mutable
	/// atomic primitives. Custom slots that wrap non-atomic state must
	/// override this method to project directly to their inner atomic.
	///
	/// # Safety
	///
	/// - `this` must be a valid pointer to `Self`.
	/// - Caller must validate via the surrounding latch's version recheck.
	#[inline]
	unsafe fn try_load_raw_ptr(this: *const Self) -> Option<Self::Value> {
		// SAFETY: see the function-level safety contract.
		unsafe { (*this).try_load() }
	}

	/// Raw-pointer variant of [`store_into_empty`](Self::store_into_empty).
	/// Bypasses the `&Self` reborrow so writer-side updates do not
	/// conflict with a concurrent reader's foreign tag under Tree Borrows.
	///
	/// # Safety
	///
	/// See [`store_into_empty`](Self::store_into_empty); the same
	/// invariants apply, with `this` standing in for `&self`.
	#[inline]
	unsafe fn store_into_empty_raw_ptr(this: *const Self, value: Self::Value) {
		// SAFETY: see the function-level safety contract.
		unsafe { (*this).store_into_empty(value) }
	}

	/// Raw-pointer variant of [`swap_init`](Self::swap_init).
	///
	/// # Safety
	///
	/// See [`swap_init`](Self::swap_init); the same invariants apply,
	/// with `this` standing in for `&self`.
	#[inline]
	unsafe fn swap_init_raw_ptr(this: *const Self, value: Self::Value) -> Self::Displaced {
		// SAFETY: see the function-level safety contract.
		unsafe { (*this).swap_init(value) }
	}

	/// Raw-pointer variant of [`take_init`](Self::take_init).
	///
	/// # Safety
	///
	/// See [`take_init`](Self::take_init); the same invariants apply,
	/// with `this` standing in for `&self`.
	#[inline]
	unsafe fn take_init_raw_ptr(this: *const Self) -> Self::Displaced {
		// SAFETY: see the function-level safety contract.
		unsafe { (*this).take_init() }
	}

	/// Raw-pointer variant of [`move_init_to_empty`](Self::move_init_to_empty).
	///
	/// # Safety
	///
	/// See [`move_init_to_empty`](Self::move_init_to_empty); the same
	/// invariants apply, with `src` / `dst` standing in for `&self` /
	/// `&dst`.
	#[inline]
	unsafe fn move_init_to_empty_raw_ptr(src: *const Self, dst: *const Self) {
		// SAFETY: see the function-level safety contract.
		unsafe { (*src).move_init_to_empty(&*dst) }
	}

	/// The displaced-owner type returned when a value leaves the slot
	/// (via [`swap_init`](Self::swap_init) or [`take_init`](Self::take_init)).
	///
	/// For inline storage, `Displaced` is `Value` itself: a `Copy` type
	/// whose drop is a no-op, so the caller can discard it inline.
	///
	/// For boxed storage, `Displaced` is a `Box<Value>` whose
	/// deallocation must be deferred until no concurrent optimistic
	/// reader could still be cloning through the displaced raw pointer.
	/// The caller routes it through the epoch GC.
	type Displaced: Send + 'static;

	/// Atomic-load a copy of the value at this slot.
	///
	/// # Safety
	///
	/// - The slot must be currently **init**.
	/// - The caller must validate the returned value via the surrounding
	///   [`crate::latch::HybridLatch`] version `recheck()` before acting
	///   on it.
	unsafe fn load(&self) -> Self::Value;

	/// Atomic-store `value` into an empty slot, transitioning it to init.
	///
	/// # Safety
	///
	/// - The slot must be currently **empty**.
	/// - The caller must hold the exclusive lock on the surrounding node.
	unsafe fn store_into_empty(&self, value: Self::Value);

	/// Atomic-swap `value` into an init slot, returning the previous
	/// value as a [`Displaced`](Self::Displaced) owner.
	///
	/// # Safety
	///
	/// - The slot must be currently **init**.
	/// - The caller must hold the exclusive lock on the surrounding node.
	/// - For boxed storage, the caller MUST route the returned Box
	///   through the epoch GC if concurrent optimistic readers may still
	///   be cloning through the displaced pointer.
	unsafe fn swap_init(&self, value: Self::Value) -> Self::Displaced;

	/// Atomic-take the value out of an init slot, transitioning it to
	/// empty, returning it as a [`Displaced`](Self::Displaced) owner.
	///
	/// # Safety
	///
	/// - The slot must be currently **init**.
	/// - The caller must hold the exclusive lock on the surrounding node.
	/// - Same epoch-defer requirement as
	///   [`swap_init`](Self::swap_init) applies for boxed storage.
	unsafe fn take_init(&self) -> Self::Displaced;

	/// Move the value from this (init) slot to `dst` (empty) atomically,
	/// leaving `self` empty.
	///
	/// Used by [`SlotArray`] for `shift_insert` / `shift_remove`. The
	/// value is moved between slots without ever materialising as a
	/// `Self::Value`, so the displaced raw pointer for boxed storage
	/// stays alive — no epoch defer is needed for these intra-array
	/// moves.
	///
	/// # Safety
	///
	/// - `self` must be currently **init**.
	/// - `dst` must be currently **empty**.
	/// - The caller must hold the exclusive lock on the surrounding node.
	unsafe fn move_init_to_empty(&self, dst: &Self);

	/// Drop the slot's contents during the parent's destructor.
	///
	/// # Safety
	///
	/// - `was_init` must accurately reflect the slot's state.
	/// - The caller must have exclusive ownership of the slot (e.g., this
	///   is the parent's `Drop` impl).
	unsafe fn drop_in_place(&mut self, was_init: bool);

	/// Load the slot's value into a caller-provided buffer for
	/// materialisation, returning a borrow valid for the lifetime of
	/// the buffer / surrounding lock.
	///
	/// - For [`InlineSlot`], the buffer is written with the loaded
	///   value's bits and the returned borrow points into the buffer.
	/// - For [`BoxedSlot`], the buffer is left untouched and the
	///   returned borrow points into the `Box<T>` directly. Valid for
	///   the duration of the surrounding shared / exclusive lock,
	///   which prevents concurrent `swap_init` / `take_init`.
	/// - For [`UnitSlot`], the buffer is written with `()` and the
	///   returned borrow is `&()`.
	///
	/// This unifies the "read a borrow without taking ownership" API
	/// across both inline and boxed storage, allowing iterators and
	/// closure-based read methods to expose `&K` / `&V` semantics over
	/// atomic storage.
	///
	/// # Safety
	///
	/// - The slot must be currently **init**.
	/// - The caller must hold a shared or exclusive lock on the
	///   surrounding node so the underlying data is not freed.
	unsafe fn load_into<'a>(&'a self, buf: &'a mut MaybeUninit<Self::Value>) -> &'a Self::Value;
}

// ---------------------------------------------------------------------------
// InlineSlot: T: AtomicLoadable
// ---------------------------------------------------------------------------

/// Inline atomic storage for `T: AtomicLoadable`.
///
/// Holds `T`'s bits in a stdlib atomic of matching size and alignment.
/// Zero allocation overhead.
#[repr(transparent)]
pub struct InlineSlot<T: AtomicLoadable> {
	inner: T::Atomic,
}

impl<T: AtomicLoadable + Default> Default for InlineSlot<T> {
	#[inline]
	fn default() -> Self {
		Self {
			inner: T::new_atomic(T::default()),
		}
	}
}

// Some primitives (e.g. raw integers) don't implement Default with the
// "zero" value automatically — but they all do via their stdlib Default
// impl, which yields `0` for ints and `0.0` for floats. Both are
// well-formed Self values for AtomicLoadable types.

// SAFETY: `InlineSlot` holds only `T::Atomic`, which is `Send + Sync` per
// `AtomicLoadable`'s contract.
unsafe impl<T: AtomicLoadable + Default> OptimisticSlot for InlineSlot<T> {
	type Value = T;
	// Inline storage: T is Copy. Dropping it is a no-op, so a "displaced"
	// inline value is just the value itself.
	type Displaced = T;

	#[inline]
	unsafe fn load(&self) -> T {
		T::load_acquire(&self.inner)
	}

	#[inline]
	unsafe fn try_load(&self) -> Option<T> {
		// Inline storage always holds a valid bit-pattern (zero default
		// for empty slots); the caller's outer recheck distinguishes.
		Some(T::load_acquire(&self.inner))
	}

	#[inline]
	unsafe fn try_load_raw_ptr(this: *const Self) -> Option<T> {
		// Project directly to the inner `T::Atomic` without an
		// `&InlineSlot` reborrow. `&T::Atomic` is interior-mutable
		// (UnsafeCell underneath in stdlib atomics) and is Tree-Borrows-
		// safe to reborrow concurrently with a writer's `&mut` on the
		// surrounding node.
		// SAFETY: `this` is a valid pointer; `inner` is at a known
		// offset.
		// SAFETY: see the function-level safety contract.
		let atomic_ptr: *const T::Atomic = unsafe { ptr::addr_of!((*this).inner) };
		// SAFETY: see the function-level safety contract.
		Some(T::load_acquire(unsafe { &*atomic_ptr }))
	}

	#[inline]
	unsafe fn store_into_empty_raw_ptr(this: *const Self, value: T) {
		// SAFETY: see the function-level safety contract.
		let atomic_ptr: *const T::Atomic = unsafe { ptr::addr_of!((*this).inner) };
		// SAFETY: see the function-level safety contract.
		T::store_release(unsafe { &*atomic_ptr }, value)
	}

	#[inline]
	unsafe fn swap_init_raw_ptr(this: *const Self, value: T) -> T {
		// SAFETY: see the function-level safety contract.
		let atomic_ptr: *const T::Atomic = unsafe { ptr::addr_of!((*this).inner) };
		// SAFETY: see the function-level safety contract.
		T::swap_acqrel(unsafe { &*atomic_ptr }, value)
	}

	#[inline]
	unsafe fn take_init_raw_ptr(this: *const Self) -> T {
		// SAFETY: see the function-level safety contract.
		let atomic_ptr: *const T::Atomic = unsafe { ptr::addr_of!((*this).inner) };
		// SAFETY: see the function-level safety contract.
		T::swap_acqrel(unsafe { &*atomic_ptr }, T::default())
	}

	#[inline]
	unsafe fn move_init_to_empty_raw_ptr(src: *const Self, dst: *const Self) {
		// SAFETY: see the function-level safety contract.
		let src_atomic: *const T::Atomic = unsafe { ptr::addr_of!((*src).inner) };
		// SAFETY: see the function-level safety contract.
		let dst_atomic: *const T::Atomic = unsafe { ptr::addr_of!((*dst).inner) };
		// SAFETY: see the function-level safety contract.
		let bits = T::load_acquire(unsafe { &*src_atomic });
		// SAFETY: see the function-level safety contract.
		T::store_release(unsafe { &*dst_atomic }, bits);
		// Clear src so the slot is logically empty.
		// SAFETY: see the function-level safety contract.
		T::store_release(unsafe { &*src_atomic }, T::default());
	}

	#[inline]
	unsafe fn store_into_empty(&self, value: T) {
		// For inline storage, empty / init are tracked by the leaf's
		// `len`; the underlying atomic always holds a valid T bit-pattern
		// (the zero value for empty slots). A `Release` store overwrites
		// it.
		T::store_release(&self.inner, value)
	}

	#[inline]
	unsafe fn swap_init(&self, value: T) -> T {
		T::swap_acqrel(&self.inner, value)
	}

	#[inline]
	unsafe fn take_init(&self) -> T {
		// Replace with zero pattern; the leaf will not read this slot
		// again until it is restored via `store_into_empty`.
		T::swap_acqrel(&self.inner, T::default())
	}

	#[inline]
	unsafe fn move_init_to_empty(&self, dst: &Self) {
		let bits = T::load_acquire(&self.inner);
		T::store_release(&dst.inner, bits);
		// Clear `self` so the slot is logically empty.
		T::store_release(&self.inner, T::default());
	}

	#[inline]
	unsafe fn drop_in_place(&mut self, _was_init: bool) {
		// AtomicLoadable types are `Copy`; their `Drop` is a no-op.
	}

	#[inline]
	unsafe fn load_into<'a>(&'a self, buf: &'a mut MaybeUninit<T>) -> &'a T {
		// Materialise the loaded value into the caller's buffer.
		let v = T::load_acquire(&self.inner);
		buf.write(v)
	}
}

// ---------------------------------------------------------------------------
// BoxedSlot: T (with EPOCH_DEFERRED_DROP)
// ---------------------------------------------------------------------------

/// Boxed atomic storage for any `T: Send + 'static`.
///
/// Holds an `AtomicPtr<T>` which is null for empty slots and points at a
/// heap-allocated `Box<T>` for init slots. Loads do an `Acquire` load and
/// clone through the pointer; writes allocate a fresh `Box<T>` and swap
/// the pointer in.
///
/// The displaced pointer from a write is **not** freed inline by this
/// type — the surrounding tree must route it through the epoch GC so
/// that a concurrent optimistic reader's `&T` borrow remains valid until
/// no reader could still hold it. See
/// [`crate::optimistic::OptimisticRead::EPOCH_DEFERRED_DROP`].
pub struct BoxedSlot<T> {
	inner: AtomicPtr<T>,
	_marker: PhantomData<Box<T>>,
}

impl<T> Default for BoxedSlot<T> {
	#[inline]
	fn default() -> Self {
		Self {
			inner: AtomicPtr::new(ptr::null_mut()),
			_marker: PhantomData,
		}
	}
}

// SAFETY: `BoxedSlot<T>` owns at most one `Box<T>`. Sending the slot to
// another thread moves that `Box`. `T: Send` is required.
unsafe impl<T: Send> Send for BoxedSlot<T> {}
// SAFETY: All shared access is through atomic ops on the `AtomicPtr`;
// concurrent loads observe consistent values. `T: Sync` is required so
// that concurrent `&T` borrows from validated snapshots are sound.
unsafe impl<T: Sync> Sync for BoxedSlot<T> {}

// SAFETY: `BoxedSlot` uses atomic ops for all shared access. The
// `Clone + Send + 'static` bounds on T are enforced at the slot's use
// sites in the tree (the trait below is unbounded so this type can also
// be used in test contexts).
unsafe impl<T: Send + Sync + Clone + 'static> OptimisticSlot for BoxedSlot<T> {
	type Value = T;
	// Boxed storage: the displaced owner is the entire `Box<T>` so the
	// caller can route it through the epoch GC for deferred deallocation.
	type Displaced = Box<T>;

	#[inline]
	unsafe fn load(&self) -> T {
		let raw = self.inner.load(Ordering::Acquire);
		// SAFETY: Caller asserts the slot is init, so `raw` is non-null
		// and points at a valid `T` whose lifetime is at least as long as
		// our read. We do NOT take ownership — clone through the pointer.
		// SAFETY: see the function-level safety contract.
		unsafe { (*raw).clone() }
	}

	#[inline]
	unsafe fn try_load(&self) -> Option<T> {
		let raw = self.inner.load(Ordering::Acquire);
		if raw.is_null() {
			return None;
		}
		// SAFETY: raw is non-null and points at a `T` whose lifetime is
		// at least as long as the surrounding epoch guard the caller
		// holds (writer routes displaced Boxes through epoch defer).
		// SAFETY: see the function-level safety contract.
		Some(unsafe { (*raw).clone() })
	}

	#[inline]
	unsafe fn try_load_raw_ptr(this: *const Self) -> Option<T> {
		// Project directly to the inner `AtomicPtr<T>` without going
		// through an `&BoxedSlot` reborrow. `&AtomicPtr` is
		// interior-mutable (UnsafeCell underneath) and is Tree-Borrows-
		// safe to reborrow concurrently with a writer's `&mut` on the
		// surrounding node; `&BoxedSlot` is not.
		// SAFETY: `this` is a valid pointer to `BoxedSlot<T>` (caller's
		// invariant); `inner` is at a known field offset.
		// SAFETY: see the function-level safety contract.
		let atomic_ptr: *const AtomicPtr<T> = unsafe { ptr::addr_of!((*this).inner) };
		// SAFETY: see the function-level safety contract.
		let raw = unsafe { (*atomic_ptr).load(Ordering::Acquire) };
		if raw.is_null() {
			return None;
		}
		// SAFETY: see the function-level safety contract.
		Some(unsafe { (*raw).clone() })
	}

	#[inline]
	unsafe fn store_into_empty_raw_ptr(this: *const Self, value: T) {
		// SAFETY: see the function-level safety contract.
		let atomic_ptr: *const AtomicPtr<T> = unsafe { ptr::addr_of!((*this).inner) };
		let raw = Box::into_raw(Box::new(value));
		// SAFETY: see the function-level safety contract.
		debug_assert!(unsafe { (*atomic_ptr).load(Ordering::Relaxed).is_null() });
		// SAFETY: see the function-level safety contract.
		unsafe { (*atomic_ptr).store(raw, Ordering::Release) };
	}

	#[inline]
	unsafe fn swap_init_raw_ptr(this: *const Self, value: T) -> Box<T> {
		// SAFETY: see the function-level safety contract.
		let atomic_ptr: *const AtomicPtr<T> = unsafe { ptr::addr_of!((*this).inner) };
		let raw_new = Box::into_raw(Box::new(value));
		// SAFETY: see the function-level safety contract.
		let raw_old = unsafe { (*atomic_ptr).swap(raw_new, Ordering::AcqRel) };
		debug_assert!(!raw_old.is_null());
		// SAFETY: see the function-level safety contract.
		unsafe { Box::from_raw(raw_old) }
	}

	#[inline]
	unsafe fn take_init_raw_ptr(this: *const Self) -> Box<T> {
		// SAFETY: see the function-level safety contract.
		let atomic_ptr: *const AtomicPtr<T> = unsafe { ptr::addr_of!((*this).inner) };
		// SAFETY: see the function-level safety contract.
		let raw_old = unsafe { (*atomic_ptr).swap(ptr::null_mut(), Ordering::AcqRel) };
		debug_assert!(!raw_old.is_null());
		// SAFETY: see the function-level safety contract.
		unsafe { Box::from_raw(raw_old) }
	}

	#[inline]
	unsafe fn move_init_to_empty_raw_ptr(src: *const Self, dst: *const Self) {
		// SAFETY: see the function-level safety contract.
		let src_atomic: *const AtomicPtr<T> = unsafe { ptr::addr_of!((*src).inner) };
		// SAFETY: see the function-level safety contract.
		let dst_atomic: *const AtomicPtr<T> = unsafe { ptr::addr_of!((*dst).inner) };
		// SAFETY: see the function-level safety contract.
		let raw = unsafe { (*src_atomic).swap(ptr::null_mut(), Ordering::AcqRel) };
		debug_assert!(!raw.is_null());
		// SAFETY: see the function-level safety contract.
		debug_assert!(unsafe { (*dst_atomic).load(Ordering::Relaxed).is_null() });
		// SAFETY: see the function-level safety contract.
		unsafe { (*dst_atomic).store(raw, Ordering::Release) };
	}

	#[inline]
	unsafe fn store_into_empty(&self, value: T) {
		let raw = Box::into_raw(Box::new(value));
		// `Release` ordering pairs with `Acquire` loads on readers.
		// SAFETY: prior state is empty (null), so we do not leak.
		debug_assert!(self.inner.load(Ordering::Relaxed).is_null());
		self.inner.store(raw, Ordering::Release);
	}

	#[inline]
	unsafe fn swap_init(&self, value: T) -> Box<T> {
		let raw_new = Box::into_raw(Box::new(value));
		let raw_old = self.inner.swap(raw_new, Ordering::AcqRel);
		debug_assert!(!raw_old.is_null());
		// SAFETY: prior state was init, so `raw_old` is a valid heap
		// allocation produced by `Box::into_raw`. Caller takes ownership
		// of the Box and is responsible for arranging epoch-deferred
		// drop if a concurrent optimistic reader could still be cloning
		// through the displaced pointer.
		// SAFETY: see the function-level safety contract.
		unsafe { Box::from_raw(raw_old) }
	}

	#[inline]
	unsafe fn take_init(&self) -> Box<T> {
		let raw_old = self.inner.swap(ptr::null_mut(), Ordering::AcqRel);
		debug_assert!(!raw_old.is_null());
		// SAFETY: as for `swap_init`.
		unsafe { Box::from_raw(raw_old) }
	}

	#[inline]
	unsafe fn move_init_to_empty(&self, dst: &Self) {
		// Move the raw pointer between slots. No allocation, no clone,
		// no drop — the same heap object lives in `dst` after this.
		let raw = self.inner.swap(ptr::null_mut(), Ordering::AcqRel);
		debug_assert!(!raw.is_null());
		debug_assert!(dst.inner.load(Ordering::Relaxed).is_null());
		dst.inner.store(raw, Ordering::Release);
	}

	#[inline]
	unsafe fn drop_in_place(&mut self, was_init: bool) {
		if was_init {
			// `&mut self` => exclusive ownership; no synchronisation
			// needed.
			let raw = *self.inner.get_mut();
			debug_assert!(!raw.is_null());
			// SAFETY: `was_init == true` => raw is a valid Box allocation.
			drop(unsafe { Box::from_raw(raw) });
			*self.inner.get_mut() = ptr::null_mut();
		} else {
			debug_assert!(self.inner.load(Ordering::Relaxed).is_null());
		}
	}

	#[inline]
	unsafe fn load_into<'a>(&'a self, _buf: &'a mut MaybeUninit<T>) -> &'a T {
		// For boxed storage, the borrow points into the `Box<T>` rather
		// than the caller's buffer. The lock the caller holds keeps the
		// Box alive (`swap_init` / `take_init` require exclusive lock).
		let raw = self.inner.load(Ordering::Acquire);
		// SAFETY: slot is init (caller's invariant), so `raw` is a
		// valid `*const T` whose pointee outlives the surrounding lock.
		// SAFETY: see the function-level safety contract.
		unsafe { &*raw }
	}
}

// ---------------------------------------------------------------------------
// UnitSlot: ZST storage for `T = ()`
// ---------------------------------------------------------------------------

/// Zero-sized slot for `T = ()`. All operations are no-ops.
#[derive(Default)]
pub struct UnitSlot;

// SAFETY: ZST with no fields; trivially Send + Sync.
unsafe impl OptimisticSlot for UnitSlot {
	type Value = ();
	type Displaced = ();
	#[inline]
	unsafe fn load(&self) {}
	#[inline]
	unsafe fn store_into_empty(&self, _value: ()) {}
	#[inline]
	unsafe fn swap_init(&self, _value: ()) {}
	#[inline]
	unsafe fn take_init(&self) {}
	#[inline]
	unsafe fn move_init_to_empty(&self, _dst: &Self) {}
	#[inline]
	unsafe fn drop_in_place(&mut self, _was_init: bool) {}
	#[inline]
	unsafe fn load_into<'a>(&'a self, buf: &'a mut MaybeUninit<()>) -> &'a () {
		buf.write(())
	}
	#[inline]
	unsafe fn try_load(&self) -> Option<()> {
		Some(())
	}
}

// ---------------------------------------------------------------------------
// SlotArray<T, N>: array of slots
// ---------------------------------------------------------------------------

/// Fixed-capacity array of [`OptimisticSlot`]s.
///
/// Used inside [`crate::lib::LeafNode`] and `InternalNode` to hold key /
/// value arrays in an optimistically-readable form. The surrounding node
/// tracks how many slots at the front of the array are init via its
/// `len` field.
///
/// The inner array is wrapped in [`core::cell::UnsafeCell`] and the
/// outer struct is `#[repr(transparent)]` so that — at the type level —
/// `&SlotArray<S, N>` has the same layout and interior-mutability
/// semantics as `&UnsafeCell<[S; N]>`. This means reborrows of
/// `&SlotArray` from a thread that does not own the surrounding node's
/// exclusive lock do not conflict, under Tree Borrows, with a concurrent
/// writer holding `&mut LeafNode` on the surrounding node. All
/// synchronisation happens at the inner atomic (`AtomicPtr` /
/// `AtomicU{8,16,32,64}`) level via [`OptimisticSlot::try_load_raw_ptr`]
/// and friends.
#[repr(transparent)]
pub struct SlotArray<S, const N: usize> {
	slots: core::cell::UnsafeCell<[S; N]>,
}

// SAFETY: SlotArray's inner array is accessed through the slot trait's
// atomic ops which provide the synchronisation. Send / Sync flow from
// the slot type's own Send / Sync.
unsafe impl<S: Send, const N: usize> Send for SlotArray<S, N> {}
// SAFETY: see the matching `Send` impl above.
unsafe impl<S: Sync, const N: usize> Sync for SlotArray<S, N> {}

impl<S: Default, const N: usize> Default for SlotArray<S, N> {
	fn default() -> Self {
		Self {
			slots: core::cell::UnsafeCell::new(core::array::from_fn(|_| S::default())),
		}
	}
}

impl<S, const N: usize> SlotArray<S, N> {
	/// Construct an all-empty slot array.
	#[inline]
	pub fn new() -> Self
	where
		S: Default,
	{
		Self::default()
	}

	/// Returns a reference to the slot at `pos`.
	#[inline]
	pub fn slot(&self, pos: usize) -> &S {
		// SAFETY: the inner array is wrapped in UnsafeCell purely to
		// opt out of Tree-Borrows protection. Slot access itself goes
		// through interior-mutable atomics, so producing an `&S`
		// reference is sound.
		// SAFETY: see the function-level safety contract.
		unsafe { &(*self.slots.get())[pos] }
	}
}

impl<S, const N: usize> SlotArray<S, N> {
	/// Returns a raw pointer to the first slot, without creating an
	/// `&SlotArray` reborrow.
	///
	/// # Safety
	///
	/// `this` must be a valid pointer to a `SlotArray<S, N>`.
	#[inline]
	pub unsafe fn slots_ptr(this: *const Self) -> *const S {
		// `UnsafeCell::get` returns `*mut T` from `&UnsafeCell<T>`;
		// here we go through the raw pointer projection to avoid
		// creating any `&SlotArray` reborrow.
		let cell_ptr: *const core::cell::UnsafeCell<[S; N]> =
			// SAFETY: see the function-level safety contract.
			unsafe { ptr::addr_of!((*this).slots) };
		core::cell::UnsafeCell::raw_get(cell_ptr) as *const S
	}
}

impl<S, const N: usize> SlotArray<S, N>
where
	S: OptimisticSlot,
{
	/// Atomic-load the value at `pos`. See [`OptimisticSlot::load`].
	///
	/// # Safety
	///
	/// - `pos < N`.
	/// - Slot at `pos` must be currently init.
	/// - Caller must validate via the surrounding latch's version recheck.
	#[inline]
	pub unsafe fn load(&self, pos: usize) -> S::Value {
		debug_assert!(pos < N);
		// SAFETY: bounds asserted by caller.
		unsafe { (*self.slots.get()).get_unchecked(pos).load() }
	}

	/// Atomic-load the value at `pos` through a raw pointer, without
	/// creating an `&SlotArray` reborrow.
	///
	/// # Safety
	///
	/// - `this` must be a valid pointer to a `SlotArray<S, N>`.
	/// - `pos < N`.
	/// - Slot at `pos` must be currently init.
	/// - Caller must validate via the surrounding latch's version recheck.
	#[inline]
	pub unsafe fn load_raw(this: *const Self, pos: usize) -> S::Value {
		debug_assert!(pos < N);
		// SAFETY: `this` is valid; `slots_ptr` returns a `*const S` for
		// at least `N` elements. The reborrow of `&S` is sound because
		// `S: OptimisticSlot` has interior mutability (atomic types).
		// SAFETY: see the function-level safety contract.
		let slot = unsafe { &*Self::slots_ptr(this).add(pos) };
		// SAFETY: see the function-level safety contract.
		unsafe { slot.load() }
	}

	/// Attempt an atomic load that tolerates a concurrently emptied
	/// slot. See [`OptimisticSlot::try_load`]. Uses raw-pointer
	/// projection through the slot's inner atomic (via
	/// [`OptimisticSlot::try_load_raw_ptr`]) so no `&S` reborrow is
	/// created — making this safe under Tree Borrows even when a
	/// writer concurrently holds `&mut` on the surrounding node.
	///
	/// # Safety
	///
	/// - `this` must be a valid pointer to a `SlotArray<S, N>`.
	/// - `pos < N`.
	/// - Caller must validate via the surrounding latch's version recheck.
	#[inline]
	pub unsafe fn try_load_raw(this: *const Self, pos: usize) -> Option<S::Value> {
		debug_assert!(pos < N);
		// SAFETY: see the function-level safety contract.
		let slot_ptr = unsafe { Self::slots_ptr(this).add(pos) };
		// SAFETY: see the function-level safety contract.
		unsafe { S::try_load_raw_ptr(slot_ptr) }
	}

	/// Raw-pointer variant of [`shift_insert`](Self::shift_insert).
	/// Uses per-slot raw-pointer atomic ops to avoid materialising any
	/// `&S` reborrow under the writer's `&mut LeafNode` tag tree.
	///
	/// # Safety
	///
	/// - `this` must be a valid pointer to a `SlotArray<S, N>`.
	/// - `pos <= len < N`.
	/// - Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn shift_insert_raw(this: *const Self, len: usize, pos: usize, value: S::Value) {
		debug_assert!(pos <= len);
		debug_assert!(len < N);
		// SAFETY: see the function-level safety contract.
		let base = unsafe { Self::slots_ptr(this) };
		for i in (pos..len).rev() {
			// SAFETY: see the function-level safety contract.
			let src_ptr = unsafe { base.add(i) };
			// SAFETY: see the function-level safety contract.
			let dst_ptr = unsafe { base.add(i + 1) };
			// SAFETY: see the function-level safety contract.
			unsafe { S::move_init_to_empty_raw_ptr(src_ptr, dst_ptr) };
		}
		// SAFETY: see the function-level safety contract.
		let slot_ptr = unsafe { base.add(pos) };
		// SAFETY: see the function-level safety contract.
		unsafe { S::store_into_empty_raw_ptr(slot_ptr, value) };
	}

	/// Raw-pointer variant of [`shift_remove`](Self::shift_remove).
	///
	/// # Safety
	///
	/// - `this` must be a valid pointer to a `SlotArray<S, N>`.
	/// - `pos < len <= N`.
	/// - Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn shift_remove_raw(this: *const Self, len: usize, pos: usize) -> S::Displaced {
		debug_assert!(pos < len);
		debug_assert!(len <= N);
		// SAFETY: see the function-level safety contract.
		let base = unsafe { Self::slots_ptr(this) };
		// SAFETY: see the function-level safety contract.
		let removed = unsafe { S::take_init_raw_ptr(base.add(pos)) };
		for i in pos..(len - 1) {
			// SAFETY: see the function-level safety contract.
			let src_ptr = unsafe { base.add(i + 1) };
			// SAFETY: see the function-level safety contract.
			let dst_ptr = unsafe { base.add(i) };
			// SAFETY: see the function-level safety contract.
			unsafe { S::move_init_to_empty_raw_ptr(src_ptr, dst_ptr) };
		}
		removed
	}

	/// Raw-pointer variant of [`swap_init`](Self::swap_init).
	///
	/// # Safety
	///
	/// - `this` must be a valid pointer to a `SlotArray<S, N>`.
	/// - `pos < N`.
	/// - Slot at `pos` must be currently init.
	/// - Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn swap_init_raw(this: *const Self, pos: usize, value: S::Value) -> S::Displaced {
		debug_assert!(pos < N);
		// SAFETY: see the function-level safety contract.
		let slot_ptr = unsafe { Self::slots_ptr(this).add(pos) };
		// SAFETY: see the function-level safety contract.
		unsafe { S::swap_init_raw_ptr(slot_ptr, value) }
	}

	/// Raw-pointer move between two SlotArrays. Used by `split` and
	/// `merge` to migrate slots between leaves.
	///
	/// # Safety
	///
	/// - Both pointers must be valid.
	/// - `src_pos < N_src` and `dst_pos < N_dst` (compile-time const).
	/// - Source slot must be init; destination must be empty.
	/// - Caller must hold exclusive locks on both surrounding nodes.
	#[inline]
	pub unsafe fn move_raw<const M: usize>(
		src: *const Self,
		src_pos: usize,
		dst: *const SlotArray<S, M>,
		dst_pos: usize,
	) {
		debug_assert!(src_pos < N);
		debug_assert!(dst_pos < M);
		// SAFETY: see the function-level safety contract.
		let src_ptr = unsafe { Self::slots_ptr(src).add(src_pos) };
		// SAFETY: see the function-level safety contract.
		let dst_ptr = unsafe { SlotArray::<S, M>::slots_ptr(dst).add(dst_pos) };
		// SAFETY: see the function-level safety contract.
		unsafe { S::move_init_to_empty_raw_ptr(src_ptr, dst_ptr) };
	}

	/// Load the value at `pos` into a caller-provided buffer for
	/// materialisation. See [`OptimisticSlot::load_into`].
	///
	/// # Safety
	///
	/// - `pos < N`.
	/// - Slot at `pos` must be currently init.
	/// - Caller must hold a shared or exclusive lock on the surrounding
	///   node.
	#[inline]
	pub unsafe fn load_into<'a>(
		&'a self,
		pos: usize,
		buf: &'a mut MaybeUninit<S::Value>,
	) -> &'a S::Value {
		debug_assert!(pos < N);
		// SAFETY: bounds asserted by caller.
		unsafe { (*self.slots.get()).get_unchecked(pos).load_into(buf) }
	}

	/// Store `value` into the slot at `pos`, which must currently be empty.
	///
	/// # Safety
	///
	/// - `pos < N`.
	/// - Slot at `pos` must be currently empty.
	/// - Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn store_into_empty(&self, pos: usize, value: S::Value) {
		debug_assert!(pos < N);
		// SAFETY: see the function-level safety contract.
		unsafe {
			(*self.slots.get()).get_unchecked(pos).store_into_empty(value);
		}
	}

	/// Swap `value` into the slot at `pos`, which must currently be init,
	/// returning the previous value as a
	/// [`Displaced`](OptimisticSlot::Displaced) owner.
	///
	/// # Safety
	///
	/// - `pos < N`.
	/// - Slot at `pos` must be currently init.
	/// - Caller must hold the exclusive lock on the surrounding node.
	/// - For boxed storage, the returned `Box` must be routed through
	///   the epoch GC if concurrent optimistic readers may still be
	///   cloning through the displaced pointer.
	#[inline]
	pub unsafe fn swap_init(&self, pos: usize, value: S::Value) -> S::Displaced {
		debug_assert!(pos < N);
		// SAFETY: see the function-level safety contract.
		unsafe { (*self.slots.get()).get_unchecked(pos).swap_init(value) }
	}

	/// Take the value out of the slot at `pos`, leaving it empty,
	/// returning it as a [`Displaced`](OptimisticSlot::Displaced) owner.
	///
	/// # Safety
	///
	/// - `pos < N`.
	/// - Slot at `pos` must be currently init.
	/// - Caller must hold the exclusive lock on the surrounding node.
	/// - Same epoch-defer requirement as
	///   [`swap_init`](Self::swap_init).
	#[inline]
	pub unsafe fn take_init(&self, pos: usize) -> S::Displaced {
		debug_assert!(pos < N);
		// SAFETY: see the function-level safety contract.
		unsafe { (*self.slots.get()).get_unchecked(pos).take_init() }
	}

	/// Insert `value` at position `pos`, shifting init slots in
	/// `[pos..len)` one position to the right.
	///
	/// On entry, slots `[0..len)` are init and `[len..N)` are empty;
	/// `pos <= len < N`. On exit, slots `[0..len+1)` are init.
	///
	/// Intra-array moves use [`OptimisticSlot::move_init_to_empty`], which
	/// for boxed storage just shuffles raw pointers — no allocation
	/// and no defer-drop required.
	///
	/// # Safety
	///
	/// - `pos <= len < N`.
	/// - Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn shift_insert(&self, len: usize, pos: usize, value: S::Value) {
		debug_assert!(pos <= len);
		debug_assert!(len < N);
		// Walk right-to-left, moving each init slot one slot to the right
		// (which is empty). Each move is an atomic-pointer shuffle.
		for i in (pos..len).rev() {
			// SAFETY: `i < len < N`; src is init, dst (i+1) was empty
			// before this iteration (and the right-to-left walk keeps it
			// empty just before we store).
			// SAFETY: see the function-level safety contract.
			let src = unsafe { (*self.slots.get()).get_unchecked(i) };
			// SAFETY: see the function-level safety contract.
			let dst = unsafe { (*self.slots.get()).get_unchecked(i + 1) };
			// SAFETY: see the function-level safety contract.
			unsafe { src.move_init_to_empty(dst) };
		}
		// SAFETY: slot at `pos` is now empty.
		unsafe {
			(*self.slots.get()).get_unchecked(pos).store_into_empty(value);
		}
	}

	/// Remove the value at position `pos`, shifting init slots in
	/// `(pos..len)` one position to the left, and return the removed value
	/// as a [`Displaced`](OptimisticSlot::Displaced) owner.
	///
	/// On entry, slots `[0..len)` are init; `pos < len <= N`. On exit,
	/// slots `[0..len-1)` are init.
	///
	/// # Safety
	///
	/// - `pos < len <= N`.
	/// - Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn shift_remove(&self, len: usize, pos: usize) -> S::Displaced {
		debug_assert!(pos < len);
		debug_assert!(len <= N);
		// Pull out the value at `pos`, leaving it empty.
		// SAFETY: bounds checked above.
		let removed = unsafe { (*self.slots.get()).get_unchecked(pos).take_init() };
		// Walk left-to-right, moving each init slot one slot to the left.
		for i in pos..(len - 1) {
			// SAFETY: see the function-level safety contract.
			let src = unsafe { (*self.slots.get()).get_unchecked(i + 1) };
			// SAFETY: see the function-level safety contract.
			let dst = unsafe { (*self.slots.get()).get_unchecked(i) };
			// SAFETY: see the function-level safety contract.
			unsafe { src.move_init_to_empty(dst) };
		}
		removed
	}

	/// Drop the first `len` slots (which must be init) during the parent's
	/// destructor.
	///
	/// # Safety
	///
	/// - `len <= N`.
	/// - Slots `[0..len)` must be init; slots `[len..N)` must be empty.
	/// - Caller must have exclusive ownership.
	#[inline]
	pub unsafe fn drop_initialized(&mut self, len: usize) {
		debug_assert!(len <= N);
		for i in 0..len {
			// SAFETY: bounds checked; was_init=true asserted by caller.
			unsafe { (*self.slots.get())[i].drop_in_place(true) };
		}
		for i in len..N {
			// SAFETY: was_init=false asserted by caller.
			unsafe { (*self.slots.get())[i].drop_in_place(false) };
		}
	}
}

// ---------------------------------------------------------------------------
// AtomicLen: AtomicU16 with raw-projection access
// ---------------------------------------------------------------------------

/// `AtomicU16` length field for leaf / internal nodes.
///
/// The optimistic-read fast path reads this without taking the leaf's
/// shared lock; the shared-lock read paths also pass through here during
/// internal-node descent. Both load with `Acquire`; writers under
/// exclusive lock store with `Release`.
#[repr(transparent)]
pub struct AtomicLen(AtomicU16);

impl AtomicLen {
	#[inline]
	pub const fn new(value: u16) -> Self {
		Self(AtomicU16::new(value))
	}

	#[inline]
	pub fn load(&self) -> u16 {
		self.0.load(Ordering::Acquire)
	}

	/// `Relaxed` load — used on the exclusive-lock write side where the
	/// surrounding lock acquire already supplies the necessary ordering.
	#[inline]
	pub fn load_relaxed(&self) -> u16 {
		self.0.load(Ordering::Relaxed)
	}

	#[inline]
	pub fn store(&self, value: u16) {
		self.0.store(value, Ordering::Release);
	}

	#[inline]
	pub fn fetch_add(&self, delta: u16) -> u16 {
		self.0.fetch_add(delta, Ordering::AcqRel)
	}

	#[inline]
	pub fn fetch_sub(&self, delta: u16) -> u16 {
		self.0.fetch_sub(delta, Ordering::AcqRel)
	}

	/// `Relaxed` increment — for use under the node's exclusive lock,
	/// where the lock release supplies the necessary Release fence and
	/// the writer is the sole observer of the new value until release.
	///
	/// # Safety
	///
	/// Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn fetch_add_relaxed(&self, delta: u16) -> u16 {
		self.0.fetch_add(delta, Ordering::Relaxed)
	}

	/// `Relaxed` decrement. See [`Self::fetch_add_relaxed`] for the safety
	/// contract.
	///
	/// # Safety
	///
	/// Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn fetch_sub_relaxed(&self, delta: u16) -> u16 {
		self.0.fetch_sub(delta, Ordering::Relaxed)
	}

	/// `Relaxed` store. See [`Self::fetch_add_relaxed`] for the safety
	/// contract.
	///
	/// # Safety
	///
	/// Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn store_relaxed(&self, value: u16) {
		self.0.store(value, Ordering::Relaxed);
	}

	/// Raw-projection read of the length without an `&Self` reborrow.
	/// Used by the optimistic-read raw-pointer descent.
	///
	/// # Safety
	///
	/// `this` must be a valid pointer to an `AtomicLen`.
	#[inline]
	pub unsafe fn load_raw(this: *const Self) -> u16 {
		// SAFETY: `AtomicLen` is `#[repr(transparent)]` over `AtomicU16`,
		// so the raw pointer dereferences to the same memory. The reborrow
		// of `&AtomicU16` is sound because `AtomicU16` has interior
		// mutability (`UnsafeCell<u16>` underneath); the `Acquire` load
		// is an atomic op.
		// SAFETY: see the function-level safety contract.
		let atomic = unsafe { &*(this as *const AtomicU16) };
		atomic.load(Ordering::Acquire)
	}

	/// `Relaxed` variant of [`Self::load_raw`]. Used on writer paths that
	/// already run under the exclusive lock (which provides the Release
	/// fence on unlock).
	///
	/// # Safety
	///
	/// `this` must be a valid pointer to an `AtomicLen`. Caller must hold
	/// the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn load_raw_relaxed(this: *const Self) -> u16 {
		// SAFETY: see `Self::load_raw` for the layout justification.
		let atomic = unsafe { &*(this as *const AtomicU16) };
		atomic.load(Ordering::Relaxed)
	}
}

impl Default for AtomicLen {
	#[inline]
	fn default() -> Self {
		Self::new(0)
	}
}

impl core::fmt::Debug for AtomicLen {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		write!(f, "AtomicLen({})", self.load_relaxed())
	}
}

// ---------------------------------------------------------------------------
// OptimisticOption<T>: atomically-loadable `Option<T>`
// ---------------------------------------------------------------------------

/// Atomically-loadable `Option<T>` for fence keys and `sample_key`.
///
/// Stores a `T` slot plus a one-bit `present` flag. The reader observes
/// either consistent `Some(T)` or `None` (with version recheck catching
/// transient inconsistency). Writers under exclusive lock atomically
/// transition between states.
///
/// Implemented in terms of [`OptimisticSlot`]: the inner slot is
/// `T::Slot` (the same slot machinery used by leaf entries).
pub struct OptimisticOption<S> {
	/// 0 = empty (None), 1 = init (Some).
	present: AtomicU8,
	slot: S,
}

impl<S: Default> Default for OptimisticOption<S> {
	fn default() -> Self {
		Self {
			present: AtomicU8::new(0),
			slot: S::default(),
		}
	}
}

impl<S: OptimisticSlot> OptimisticOption<S> {
	/// Atomically load the inner `Option<T>`.
	///
	/// # Safety
	///
	/// Caller must validate the returned value via the surrounding latch's
	/// version recheck. A failed recheck may have observed an
	/// inconsistent `(present, slot)` pair.
	#[inline]
	pub unsafe fn load(&self) -> Option<S::Value> {
		// Load `present` first; if it's 0, treat the slot as empty.
		// If it's 1, atomic-load the slot.
		// Ordering: Acquire on present; the slot's own load is also
		// Acquire. The version recheck on the surrounding latch
		// disambiguates torn observations.
		let present = self.present.load(Ordering::Acquire);
		if present == 0 {
			None
		} else {
			// SAFETY: present == 1 indicates init at the moment of load;
			// version recheck validates that the slot was actually init
			// when we read.
			// SAFETY: see the function-level safety contract.
			Some(unsafe { self.slot.load() })
		}
	}

	/// Raw-projection load without `&Self` reborrow.
	///
	/// # Safety
	///
	/// `this` must be a valid pointer to an `OptimisticOption<S>`.
	#[inline]
	pub unsafe fn load_raw(this: *const Self) -> Option<S::Value> {
		// SAFETY: `present` and `slot` are atomic / have interior
		// mutability; the `&AtomicU8` and `&S` reborrows are sound.
		// SAFETY: see the function-level safety contract.
		let present_ptr = unsafe { ptr::addr_of!((*this).present) };
		// SAFETY: see the function-level safety contract.
		let slot_ptr = unsafe { ptr::addr_of!((*this).slot) };
		// SAFETY: see the function-level safety contract.
		let present = unsafe { (*present_ptr).load(Ordering::Acquire) };
		if present == 0 {
			None
		} else {
			// SAFETY: see the function-level safety contract.
			Some(unsafe { (*slot_ptr).load() })
		}
	}

	/// Store `Some(value)` into an empty option, transitioning to init.
	///
	/// # Safety
	///
	/// - Option must currently be `None`.
	/// - Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn store_some_into_empty(&self, value: S::Value) {
		debug_assert_eq!(self.present.load(Ordering::Relaxed), 0);
		// SAFETY: slot is empty.
		unsafe { self.slot.store_into_empty(value) };
		// Publish presence with Release, paired with the load's Acquire.
		self.present.store(1, Ordering::Release);
	}

	/// Replace the option's value, transitioning to (or staying at) init.
	///
	/// Returns the previous value (if `Some`) as a
	/// [`Displaced`](OptimisticSlot::Displaced) owner. Boxed-storage
	/// callers must route the returned Box through the epoch GC if
	/// concurrent optimistic readers could still be cloning through it.
	///
	/// # Safety
	///
	/// - Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn replace(&self, value: S::Value) -> Option<S::Displaced> {
		let was_present = self.present.load(Ordering::Relaxed) != 0;
		if was_present {
			// SAFETY: slot is init.
			let old = unsafe { self.slot.swap_init(value) };
			Some(old)
		} else {
			// SAFETY: slot is empty.
			unsafe { self.slot.store_into_empty(value) };
			self.present.store(1, Ordering::Release);
			None
		}
	}

	/// Take the option's value, transitioning to empty.
	///
	/// Returns the value as a [`Displaced`](OptimisticSlot::Displaced)
	/// owner. Same epoch-defer caveat as
	/// [`replace`](Self::replace).
	///
	/// # Safety
	///
	/// - Caller must hold the exclusive lock on the surrounding node.
	#[inline]
	pub unsafe fn take(&self) -> Option<S::Displaced> {
		let was_present = self.present.load(Ordering::Relaxed) != 0;
		if was_present {
			// Clear `present` first so a concurrent reader sees `None`
			// before observing the slot in flux.
			self.present.store(0, Ordering::Release);
			// SAFETY: slot was init.
			Some(unsafe { self.slot.take_init() })
		} else {
			None
		}
	}

	/// Drop the option's contents during the parent's destructor.
	///
	/// # Safety
	///
	/// - Caller must have exclusive ownership.
	#[inline]
	pub unsafe fn drop_in_place(&mut self) {
		let was_present = *self.present.get_mut() != 0;
		// SAFETY: was_init reflects state; exclusive ownership.
		unsafe { self.slot.drop_in_place(was_present) };
		*self.present.get_mut() = 0;
	}
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use std::sync::Arc;

	#[test]
	fn inline_slot_u64_roundtrip() {
		let s: InlineSlot<u64> = InlineSlot::default();
		// Initially zero.
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(s.load(), 0u64) };
		// SAFETY: see the function-level safety contract.
		unsafe { s.store_into_empty(42) };
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(s.load(), 42) };
		// SAFETY: see the function-level safety contract.
		let old = unsafe { s.swap_init(7) };
		assert_eq!(old, 42);
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(s.load(), 7) };
		// SAFETY: see the function-level safety contract.
		let taken = unsafe { s.take_init() };
		assert_eq!(taken, 7);
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(s.load(), 0) };
	}

	#[test]
	fn inline_slot_i32_negative() {
		let s: InlineSlot<i32> = InlineSlot::default();
		// SAFETY: see the function-level safety contract.
		unsafe { s.store_into_empty(-12345) };
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(s.load(), -12345) };
	}

	#[test]
	fn boxed_slot_arc_roundtrip() {
		let s: BoxedSlot<Arc<String>> = BoxedSlot::default();
		let v = Arc::new("hello".to_string());
		// SAFETY: see the function-level safety contract.
		unsafe { s.store_into_empty(v.clone()) };
		// SAFETY: see the function-level safety contract.
		unsafe {
			let read: Arc<String> = s.load();
			assert_eq!(*read, "hello");
		}
		let new = Arc::new("world".to_string());
		// SAFETY: see the function-level safety contract.
		let old: Box<Arc<String>> = unsafe { s.swap_init(new) };
		assert_eq!(**old, "hello");
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(*s.load(), "world") };
		// SAFETY: see the function-level safety contract.
		let taken: Box<Arc<String>> = unsafe { s.take_init() };
		assert_eq!(**taken, "world");
	}

	#[test]
	fn boxed_slot_drops_on_destruction() {
		let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
		struct DropCounter(Arc<std::sync::atomic::AtomicUsize>);
		impl Clone for DropCounter {
			fn clone(&self) -> Self {
				Self(self.0.clone())
			}
		}
		impl Drop for DropCounter {
			fn drop(&mut self) {
				self.0.fetch_add(1, Ordering::SeqCst);
			}
		}
		{
			let mut s: BoxedSlot<DropCounter> = BoxedSlot::default();
			// SAFETY: see the function-level safety contract.
			unsafe { s.store_into_empty(DropCounter(counter.clone())) };
			// Manually invoke drop_in_place since we own the slot here.
			// SAFETY: see the function-level safety contract.
			unsafe { s.drop_in_place(true) };
			// Slot is now empty.
		}
		// Expect at least one drop (the boxed DropCounter freed on
		// drop_in_place).
		assert!(counter.load(Ordering::SeqCst) >= 1);
	}

	#[test]
	fn slot_array_shift_insert_remove() {
		let arr: SlotArray<InlineSlot<u64>, 8> = SlotArray::default();
		// SAFETY: see the function-level safety contract.
		unsafe {
			arr.store_into_empty(0, 10);
			arr.store_into_empty(1, 20);
			arr.store_into_empty(2, 30);
		}
		// Insert 15 at pos 1: [10, 15, 20, 30]
		// SAFETY: see the function-level safety contract.
		unsafe { arr.shift_insert(3, 1, 15) };
		// SAFETY: see the function-level safety contract.
		unsafe {
			assert_eq!(arr.load(0), 10);
			assert_eq!(arr.load(1), 15);
			assert_eq!(arr.load(2), 20);
			assert_eq!(arr.load(3), 30);
		}
		// Remove pos 2 (the 20): [10, 15, 30]
		// SAFETY: see the function-level safety contract.
		let removed = unsafe { arr.shift_remove(4, 2) };
		assert_eq!(removed, 20);
		// SAFETY: see the function-level safety contract.
		unsafe {
			assert_eq!(arr.load(0), 10);
			assert_eq!(arr.load(1), 15);
			assert_eq!(arr.load(2), 30);
		}
	}

	#[test]
	fn inline_slot_load_into_returns_buffer_borrow() {
		use core::mem::MaybeUninit;
		let s: InlineSlot<u64> = InlineSlot::default();
		// SAFETY: see the function-level safety contract.
		unsafe { s.store_into_empty(0x123456789abcdef0) };
		let mut buf = MaybeUninit::uninit();
		// SAFETY: slot is init.
		let v: &u64 = unsafe { s.load_into(&mut buf) };
		assert_eq!(*v, 0x123456789abcdef0);
	}

	#[test]
	fn boxed_slot_load_into_returns_box_borrow() {
		use core::mem::MaybeUninit;
		let s: BoxedSlot<String> = BoxedSlot::default();
		// SAFETY: see the function-level safety contract.
		unsafe { s.store_into_empty(String::from("hello")) };
		let mut buf = MaybeUninit::uninit();
		// SAFETY: slot is init; we hold exclusive ownership in this test.
		let v: &String = unsafe { s.load_into(&mut buf) };
		assert_eq!(v.as_str(), "hello");
		// Tear down before drop.
		// SAFETY: slot is init.
		let _ = unsafe { s.take_init() };
	}

	#[test]
	fn unit_slot_load_into_returns_unit() {
		use core::mem::MaybeUninit;
		let s = UnitSlot;
		let mut buf = MaybeUninit::uninit();
		// SAFETY: UnitSlot is always "init" (no state).
		let v: &() = unsafe { s.load_into(&mut buf) };
		assert_eq!(*v, ());
	}

	#[test]
	fn slot_array_raw_projection() {
		let arr: SlotArray<InlineSlot<u64>, 4> = SlotArray::default();
		// SAFETY: see the function-level safety contract.
		unsafe {
			arr.store_into_empty(0, 100);
			arr.store_into_empty(1, 200);
		}
		let raw: *const SlotArray<InlineSlot<u64>, 4> = &arr;
		// SAFETY: see the function-level safety contract.
		unsafe {
			assert_eq!(SlotArray::<InlineSlot<u64>, 4>::load_raw(raw, 0), 100);
			assert_eq!(SlotArray::<InlineSlot<u64>, 4>::load_raw(raw, 1), 200);
		}
	}

	#[test]
	fn atomic_len_basics() {
		let l = AtomicLen::new(0);
		assert_eq!(l.load(), 0);
		l.store(5);
		assert_eq!(l.load(), 5);
		assert_eq!(l.fetch_add(2), 5);
		assert_eq!(l.load(), 7);
		assert_eq!(l.fetch_sub(3), 7);
		assert_eq!(l.load(), 4);
		let raw: *const AtomicLen = &l;
		// SAFETY: see the function-level safety contract.
		assert_eq!(unsafe { AtomicLen::load_raw(raw) }, 4);
	}

	#[test]
	fn optimistic_option_lifecycle() {
		let opt: OptimisticOption<InlineSlot<u32>> = OptimisticOption::default();
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(opt.load(), None) };
		// SAFETY: see the function-level safety contract.
		unsafe { opt.store_some_into_empty(42) };
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(opt.load(), Some(42)) };
		// SAFETY: see the function-level safety contract.
		let prev: Option<u32> = unsafe { opt.replace(100) };
		assert_eq!(prev, Some(42));
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(opt.load(), Some(100)) };
		// SAFETY: see the function-level safety contract.
		let taken: Option<u32> = unsafe { opt.take() };
		assert_eq!(taken, Some(100));
		// SAFETY: see the function-level safety contract.
		unsafe { assert_eq!(opt.load(), None) };
	}

	#[test]
	fn optimistic_option_raw_load() {
		let opt: OptimisticOption<InlineSlot<u32>> = OptimisticOption::default();
		// SAFETY: see the function-level safety contract.
		unsafe { opt.store_some_into_empty(7) };
		let raw: *const OptimisticOption<InlineSlot<u32>> = &opt;
		// SAFETY: see the function-level safety contract.
		unsafe {
			assert_eq!(OptimisticOption::<InlineSlot<u32>>::load_raw(raw), Some(7));
		}
	}

	// -----------------------------------------------------------------------
	// Concurrent tests
	// -----------------------------------------------------------------------
	//
	// These tests interleave writes (under a simulated "exclusive lock", i.e.
	// a `Mutex`) with concurrent unsynchronised loads from multiple reader
	// threads. They prove that the storage primitives' loads and stores
	// commute under Miri's data-race detector — the foundation Phase 2 will
	// build on for the leaf and internal nodes.
	//
	// Under `-Zmiri-tree-borrows` with the data-race detector enabled, an
	// implementation that used `ptr::read` / `mem::replace` instead of the
	// atomic ops in this module would fail these tests. Passing them shows
	// the primitives themselves do not race.

	use std::sync::Mutex;
	use std::thread;

	/// `InlineSlot<u64>`: writer mutates with `swap_init`, readers see
	/// well-formed `u64` values under Acquire/Release synchronisation.
	#[test]
	fn inline_slot_concurrent_swap_vs_load() {
		let slot: Arc<InlineSlot<u64>> = Arc::new(InlineSlot::default());
		// Seed with an initial value so swap_init is valid.
		// SAFETY: see the function-level safety contract.
		unsafe { slot.store_into_empty(1) };

		let writer_iters = if cfg!(miri) {
			32
		} else {
			4096
		};
		let reader_iters = if cfg!(miri) {
			64
		} else {
			8192
		};

		let writer_lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
		let writer = {
			let slot = slot.clone();
			let writer_lock = writer_lock.clone();
			thread::spawn(move || {
				for i in 1u64..=writer_iters {
					let _g = writer_lock.lock().unwrap();
					// SAFETY: we hold the "exclusive lock" (the Mutex); slot is init.
					// InlineSlot's Displaced = u64 (Copy); discard freely.
					// SAFETY: see the function-level safety contract.
					let _: u64 = unsafe { slot.swap_init(i + 1) };
				}
			})
		};

		let readers: Vec<_> = (0..3)
			.map(|_| {
				let slot = slot.clone();
				thread::spawn(move || {
					for _ in 0..reader_iters {
						// SAFETY: slot has always been init since the seed.
						let v = unsafe { slot.load() };
						// Every observed value must be one we could have stored.
						assert!(v >= 1 && v <= writer_iters + 1);
					}
				})
			})
			.collect();

		writer.join().unwrap();
		for r in readers {
			r.join().unwrap();
		}
	}

	/// `BoxedSlot<Arc<String>>`: writer stores once, then readers concurrently
	/// load. Validates Acquire/Release ordering on the AtomicPtr.
	#[test]
	fn boxed_slot_concurrent_load_after_store() {
		let slot: Arc<BoxedSlot<Arc<String>>> = Arc::new(BoxedSlot::default());
		let value = Arc::new("hello".to_string());
		// SAFETY: slot is empty.
		unsafe { slot.store_into_empty(value.clone()) };

		let reader_iters = if cfg!(miri) {
			32
		} else {
			4096
		};
		let readers: Vec<_> = (0..4)
			.map(|_| {
				let slot = slot.clone();
				thread::spawn(move || {
					for _ in 0..reader_iters {
						// SAFETY: slot is init.
						let v: Arc<String> = unsafe { slot.load() };
						assert_eq!(*v, "hello");
					}
				})
			})
			.collect();
		for r in readers {
			r.join().unwrap();
		}
		// Tear down: take the value to leave the slot empty before Drop.
		// SAFETY: slot is init; readers have joined.
		let _displaced: Box<Arc<String>> = unsafe { slot.take_init() };
	}

	/// `BoxedSlot<Arc<String>>`: writer swaps repeatedly (allocating new Box
	/// each time) while readers load. The writer must NOT free old boxes
	/// inline because readers might still be cloning through them. This test
	/// stashes the displaced values into a "deferred drop bag" and drops
	/// them all after readers terminate — modelling what
	/// [`crate::sync::epoch`] does in the tree.
	#[test]
	fn boxed_slot_concurrent_swap_with_deferred_drop() {
		let slot: Arc<BoxedSlot<Arc<u64>>> = Arc::new(BoxedSlot::default());
		// SAFETY: slot is empty.
		unsafe { slot.store_into_empty(Arc::new(0)) };

		let writer_iters = if cfg!(miri) {
			16
		} else {
			1024
		};
		let reader_iters = if cfg!(miri) {
			32
		} else {
			4096
		};

		// The deferred bag holds Box<Arc<u64>> values — the displaced
		// owners returned from swap_init. Keeping them alive until
		// readers join models what epoch GC does in the tree.
		let deferred: Arc<Mutex<Vec<Box<Arc<u64>>>>> = Arc::new(Mutex::new(Vec::new()));
		let writer_lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));

		let writer = {
			let slot = slot.clone();
			let writer_lock = writer_lock.clone();
			let deferred = deferred.clone();
			thread::spawn(move || {
				for i in 1u64..=writer_iters {
					let _g = writer_lock.lock().unwrap();
					// SAFETY: slot is init; we hold the exclusive lock.
					let old: Box<Arc<u64>> = unsafe { slot.swap_init(Arc::new(i)) };
					// Defer-drop: hold the displaced Box until readers stop.
					deferred.lock().unwrap().push(old);
				}
			})
		};

		let readers: Vec<_> = (0..3)
			.map(|_| {
				let slot = slot.clone();
				thread::spawn(move || {
					for _ in 0..reader_iters {
						// SAFETY: slot is init throughout this test.
						let v: Arc<u64> = unsafe { slot.load() };
						assert!(*v <= writer_iters);
					}
				})
			})
			.collect();

		writer.join().unwrap();
		for r in readers {
			r.join().unwrap();
		}
		// All concurrent activity has ended; safe to drop the deferred bag.
		drop(deferred);
	}

	/// `SlotArray<InlineSlot<u64>, _>`: writer mutates via
	/// `shift_insert` / `shift_remove` under an exclusive-lock Mutex;
	/// readers load arbitrary positions via `load_raw` (the raw-projection
	/// API the tree's optimistic descent will use). Every observed value
	/// must be in the set of values the writer ever stored.
	#[test]
	fn slot_array_concurrent_shift_vs_load_raw() {
		const CAP: usize = 16;
		let arr: Arc<SlotArray<InlineSlot<u64>, CAP>> = Arc::new(SlotArray::default());
		// Seed with [1, 2, 3, 4].
		for (i, v) in [1u64, 2, 3, 4].iter().enumerate() {
			// SAFETY: see the function-level safety contract.
			unsafe { arr.store_into_empty(i, *v) };
		}
		let initial_len = Arc::new(std::sync::atomic::AtomicUsize::new(4));

		let writer_iters = if cfg!(miri) {
			8
		} else {
			512
		};
		let reader_iters = if cfg!(miri) {
			32
		} else {
			4096
		};
		let writer_lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));

		let writer = {
			let arr = arr.clone();
			let writer_lock = writer_lock.clone();
			let initial_len = initial_len.clone();
			thread::spawn(move || {
				for i in 0..writer_iters {
					let _g = writer_lock.lock().unwrap();
					let len = initial_len.load(std::sync::atomic::Ordering::Relaxed);
					if i % 2 == 0 && len < CAP {
						// SAFETY: exclusive lock; pos==1 <= len; len < CAP.
						unsafe { arr.shift_insert(len, 1, 100 + i as u64) };
						initial_len.store(len + 1, std::sync::atomic::Ordering::Relaxed);
					} else if len > 1 {
						// SAFETY: exclusive lock; pos==1 < len. The
						// removed value is a Copy u64; discard freely.
						// SAFETY: see the function-level safety contract.
						let _removed: u64 = unsafe { arr.shift_remove(len, 1) };
						initial_len.store(len - 1, std::sync::atomic::Ordering::Relaxed);
					}
				}
			})
		};

		let arr_for_readers = arr.clone();
		let raw_ptr_addr = Arc::as_ptr(&arr_for_readers) as usize;
		let readers: Vec<_> = (0..3)
			.map(|_| {
				let _slot = arr_for_readers.clone();
				let raw = raw_ptr_addr;
				thread::spawn(move || {
					let this: *const SlotArray<InlineSlot<u64>, CAP> = raw as *const _;
					for _ in 0..reader_iters {
						// Read position 0 — always seeded with 1 and never
						// shifted. Atomic load synchronises with any
						// concurrent writer store.
						// SAFETY: slot 0 is always init throughout this test.
						let v = unsafe { SlotArray::load_raw(this, 0) };
						assert_eq!(v, 1);
					}
				})
			})
			.collect();

		writer.join().unwrap();
		for r in readers {
			r.join().unwrap();
		}
	}

	/// `AtomicLen` concurrent fetch_add + load + load_raw.
	#[test]
	fn atomic_len_concurrent_writers_readers() {
		let len: Arc<AtomicLen> = Arc::new(AtomicLen::new(0));
		let writers_n = 2;
		let writer_iters = if cfg!(miri) {
			32
		} else {
			4096
		};
		let reader_iters = if cfg!(miri) {
			32
		} else {
			4096
		};

		let writers: Vec<_> = (0..writers_n)
			.map(|_| {
				let len = len.clone();
				thread::spawn(move || {
					for _ in 0..writer_iters {
						len.fetch_add(1);
					}
				})
			})
			.collect();
		let len_raw = Arc::as_ptr(&len) as usize;
		let readers: Vec<_> = (0..2)
			.map(|_| {
				let len = len.clone();
				let raw = len_raw;
				thread::spawn(move || {
					let this: *const AtomicLen = raw as *const _;
					for _ in 0..reader_iters {
						let a = len.load();
						// SAFETY: AtomicLen lives for the join below.
						let b = unsafe { AtomicLen::load_raw(this) };
						let expected_max = (writers_n * writer_iters) as u16;
						assert!(a <= expected_max);
						assert!(b <= expected_max);
					}
				})
			})
			.collect();

		for w in writers {
			w.join().unwrap();
		}
		for r in readers {
			r.join().unwrap();
		}
		assert_eq!(len.load(), (writers_n * writer_iters) as u16);
	}

	/// `OptimisticOption` concurrent replace + load.
	#[test]
	fn optimistic_option_concurrent_replace_vs_load() {
		let opt: Arc<OptimisticOption<InlineSlot<u32>>> = Arc::new(OptimisticOption::default());
		// Seed with Some(0).
		// SAFETY: opt is empty.
		unsafe { opt.store_some_into_empty(0) };

		let writer_iters = if cfg!(miri) {
			32
		} else {
			4096
		};
		let reader_iters = if cfg!(miri) {
			32
		} else {
			4096
		};

		let writer_lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
		let writer = {
			let opt = opt.clone();
			let writer_lock = writer_lock.clone();
			thread::spawn(move || {
				for i in 1u32..=writer_iters {
					let _g = writer_lock.lock().unwrap();
					// SAFETY: writer holds the exclusive lock simulator.
					// Displaced<u32> = u32 (Copy); discard freely.
					// SAFETY: see the function-level safety contract.
					let _old: Option<u32> = unsafe { opt.replace(i) };
				}
			})
		};

		let readers: Vec<_> = (0..3)
			.map(|_| {
				let opt = opt.clone();
				thread::spawn(move || {
					for _ in 0..reader_iters {
						// SAFETY: caller doesn't need to validate via recheck
						// here because the test guarantees no concurrent
						// take() leaves the option None mid-read; the only
						// transitions are Some(x) -> Some(y).
						// SAFETY: see the function-level safety contract.
						let v = unsafe { opt.load() };
						let n = v.expect("option always Some in this test");
						assert!(n <= writer_iters);
					}
				})
			})
			.collect();

		writer.join().unwrap();
		for r in readers {
			r.join().unwrap();
		}
	}
}
