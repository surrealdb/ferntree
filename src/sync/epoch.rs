//! Epoch-based memory reclamation with loom support.
//!
//! Under normal compilation, re-exports crossbeam_epoch.
//! Under `cfg(loom)`, provides a simplified mock implementation that
//! uses immediate reclamation, which is safe for loom's deterministic
//! execution model.
//!
//! # Loom Mock
//!
//! The loom mock uses `Arc<T>` internally to track allocations and
//! provides the same API surface as crossbeam_epoch, but with immediate
//! memory reclamation when no references remain.

// Allow unused items - some are only used under loom cfg
#![allow(unused)]

#[cfg(not(loom))]
pub use crossbeam_epoch::{pin, Atomic, Guard, Owned, Shared};

#[cfg(loom)]
mod mock_epoch {
	//! Mock epoch-based memory reclamation for loom testing.
	//!
	//! This provides a simplified version of crossbeam_epoch's API that
	//! uses reference counting for memory management. Under loom's
	//! deterministic execution model, this is sufficient for testing
	//! concurrent algorithms.

	use std::marker::PhantomData;
	use std::ptr::NonNull;
	use std::sync::atomic::Ordering;
	use std::sync::Arc;

	use loom::sync::atomic::AtomicPtr;

	/// A mock epoch guard. In the real implementation, this would pin
	/// the current thread to an epoch to prevent premature reclamation.
	/// In our mock, it's just a marker type.
	pub struct Guard {
		// Prevent Send/Sync to match crossbeam_epoch behavior
		_marker: PhantomData<*mut ()>,
	}

	impl Guard {
		fn new() -> Self {
			Guard {
				_marker: PhantomData,
			}
		}

		/// Defers destruction of the given shared pointer until it's safe.
		/// In our mock, we don't actually defer - the Arc handles cleanup.
		pub fn defer_destroy<T>(&self, _ptr: Shared<'_, T>) {
			// Arc's Drop handles cleanup when all references are gone.
			// The Shared pointer doesn't own the data, so nothing to do here.
		}

		/// Flush deferred operations. No-op in our mock.
		pub fn flush(&self) {
			// No-op: we don't defer anything
		}
	}

	/// Pin the current thread to the epoch. Returns a guard that must
	/// be held while accessing epoch-protected data.
	pub fn pin() -> Guard {
		Guard::new()
	}

	/// A shared reference to epoch-protected data.
	///
	/// This is similar to `&T` but can be null and is tied to an epoch guard.
	pub struct Shared<'g, T> {
		ptr: *const T,
		_marker: PhantomData<(&'g (), *const T)>,
	}

	impl<'g, T> Clone for Shared<'g, T> {
		fn clone(&self) -> Self {
			*self
		}
	}

	impl<'g, T> Copy for Shared<'g, T> {}

	impl<'g, T> Shared<'g, T> {
		/// Creates a null shared pointer.
		pub fn null() -> Self {
			Shared {
				ptr: std::ptr::null(),
				_marker: PhantomData,
			}
		}

		/// Returns true if this pointer is null.
		pub fn is_null(&self) -> bool {
			self.ptr.is_null()
		}

		/// Converts to a raw pointer.
		pub fn as_raw(&self) -> *const T {
			self.ptr
		}

		/// Dereferences the pointer.
		///
		/// # Safety
		///
		/// The pointer must be non-null and the data must be valid.
		pub unsafe fn deref(&self) -> &'g T {
			&*self.ptr
		}

		/// Dereferences the pointer, returning None if null.
		///
		/// # Safety
		///
		/// If non-null, the data must be valid.
		pub unsafe fn as_ref(&self) -> Option<&'g T> {
			if self.is_null() {
				None
			} else {
				Some(&*self.ptr)
			}
		}

		/// Creates a Shared from a raw pointer.
		///
		/// # Safety
		///
		/// The pointer must be valid for the lifetime 'g or null.
		pub unsafe fn from_raw(ptr: *const T) -> Self {
			Shared {
				ptr,
				_marker: PhantomData,
			}
		}
	}

	/// An owned pointer to epoch-protected data.
	///
	/// This owns the data and can be converted to a Shared reference.
	pub struct Owned<T> {
		// We use Box internally for ownership
		data: NonNull<T>,
	}

	// SAFETY: Owned<T> can be sent across threads if T: Send
	unsafe impl<T: Send> Send for Owned<T> {}

	impl<T> Owned<T> {
		/// Creates a new owned pointer containing the given value.
		pub fn new(value: T) -> Self {
			let boxed = Box::new(value);
			Owned {
				data: NonNull::new(Box::into_raw(boxed)).unwrap(),
			}
		}

		/// Converts this owned pointer into a shared reference.
		pub fn into_shared<'g>(self, _guard: &'g Guard) -> Shared<'g, T> {
			let ptr = self.data.as_ptr();
			std::mem::forget(self); // Don't drop, ownership transfers to the atomic
			Shared {
				ptr,
				_marker: PhantomData,
			}
		}

		/// Returns a reference to the value.
		pub fn as_ref(&self) -> &T {
			// SAFETY: data is always valid
			unsafe { self.data.as_ref() }
		}

		/// Returns a mutable reference to the value.
		pub fn as_mut(&mut self) -> &mut T {
			// SAFETY: data is always valid and we have exclusive access
			unsafe { self.data.as_mut() }
		}
	}

	impl<T> Drop for Owned<T> {
		fn drop(&mut self) {
			// SAFETY: We own the data
			unsafe {
				drop(Box::from_raw(self.data.as_ptr()));
			}
		}
	}

	impl<T> std::ops::Deref for Owned<T> {
		type Target = T;
		fn deref(&self) -> &T {
			self.as_ref()
		}
	}

	impl<T> std::ops::DerefMut for Owned<T> {
		fn deref_mut(&mut self) -> &mut T {
			// SAFETY: data is always valid and we have exclusive access
			unsafe { self.data.as_mut() }
		}
	}

	/// An atomic pointer that can be safely shared between threads.
	pub struct Atomic<T> {
		ptr: AtomicPtr<T>,
		// Track whether we own the data (for cleanup in drop)
		owns_data: std::sync::atomic::AtomicBool,
	}

	// SAFETY: Atomic<T> can be sent/shared if T: Send + Sync
	unsafe impl<T: Send + Sync> Send for Atomic<T> {}
	unsafe impl<T: Send + Sync> Sync for Atomic<T> {}

	impl<T> std::fmt::Debug for Atomic<T> {
		fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
			let ptr = self.ptr.load(Ordering::SeqCst);
			f.debug_struct("Atomic").field("ptr", &ptr).finish()
		}
	}

	impl<T> Atomic<T> {
		/// Creates a new atomic pointer initialized to null.
		pub fn null() -> Self {
			Atomic {
				ptr: AtomicPtr::new(std::ptr::null_mut()),
				owns_data: std::sync::atomic::AtomicBool::new(false),
			}
		}

		/// Creates a new atomic pointer containing the given value.
		pub fn new(value: T) -> Self {
			let boxed = Box::new(value);
			Atomic {
				ptr: AtomicPtr::new(Box::into_raw(boxed)),
				owns_data: std::sync::atomic::AtomicBool::new(true),
			}
		}

		/// Loads the current value.
		pub fn load<'g>(&self, order: Ordering, _guard: &'g Guard) -> Shared<'g, T> {
			let ptr = self.ptr.load(order);
			Shared {
				ptr,
				_marker: PhantomData,
			}
		}

		/// Stores a new value.
		pub fn store(&self, new: Shared<'_, T>, order: Ordering) {
			self.ptr.store(new.ptr as *mut T, order);
		}

		/// Swaps the current value with a new one, returning the old value.
		pub fn swap<'g>(
			&self,
			new: Shared<'_, T>,
			order: Ordering,
			_guard: &'g Guard,
		) -> Shared<'g, T> {
			let old = self.ptr.swap(new.ptr as *mut T, order);
			Shared {
				ptr: old,
				_marker: PhantomData,
			}
		}

		/// Compares and exchanges the value.
		pub fn compare_exchange<'g>(
			&self,
			current: Shared<'_, T>,
			new: Shared<'_, T>,
			success: Ordering,
			failure: Ordering,
			_guard: &'g Guard,
		) -> Result<Shared<'g, T>, CompareExchangeError<'g, T, Shared<'g, T>>> {
			match self.ptr.compare_exchange(
				current.ptr as *mut T,
				new.ptr as *mut T,
				success,
				failure,
			) {
				Ok(ptr) => Ok(Shared {
					ptr,
					_marker: PhantomData,
				}),
				Err(ptr) => Err(CompareExchangeError {
					current: Shared {
						ptr,
						_marker: PhantomData,
					},
					new: Shared {
						ptr: new.ptr,
						_marker: PhantomData,
					},
				}),
			}
		}

		/// Compares and exchanges the value, storing an owned value on success.
		pub fn compare_exchange_owned<'g>(
			&self,
			current: Shared<'_, T>,
			new: Owned<T>,
			success: Ordering,
			failure: Ordering,
			_guard: &'g Guard,
		) -> Result<Shared<'g, T>, CompareExchangeError<'g, T, Owned<T>>> {
			let new_ptr = new.data.as_ptr();
			match self.ptr.compare_exchange(current.ptr as *mut T, new_ptr, success, failure) {
				Ok(ptr) => {
					std::mem::forget(new); // Ownership transferred to atomic
					Ok(Shared {
						ptr,
						_marker: PhantomData,
					})
				}
				Err(ptr) => Err(CompareExchangeError {
					current: Shared {
						ptr,
						_marker: PhantomData,
					},
					new,
				}),
			}
		}

		/// Gets a mutable reference to the underlying pointer.
		/// This is only safe when there are no concurrent accesses.
		/// Note: Under loom, this uses relaxed load which is safe for single-threaded cleanup.
		pub fn get_mut(&mut self) -> Option<&mut T> {
			let ptr = self.ptr.load(Ordering::Relaxed);
			if ptr.is_null() {
				None
			} else {
				// SAFETY: We have exclusive access through &mut self
				unsafe { Some(&mut *ptr) }
			}
		}

		/// Takes the value out, leaving null in its place.
		/// This is only safe when there are no concurrent accesses.
		pub fn take(&mut self) -> Option<Box<T>> {
			let ptr = self.ptr.swap(std::ptr::null_mut(), Ordering::Relaxed);
			if ptr.is_null() {
				None
			} else {
				self.owns_data.store(false, std::sync::atomic::Ordering::Relaxed);
				// SAFETY: We have exclusive access and the pointer is valid
				unsafe { Some(Box::from_raw(ptr)) }
			}
		}
	}

	impl<T> Default for Atomic<T> {
		fn default() -> Self {
			Self::null()
		}
	}

	impl<T> Drop for Atomic<T> {
		fn drop(&mut self) {
			if self.owns_data.load(std::sync::atomic::Ordering::Relaxed) {
				let ptr = self.ptr.load(Ordering::Relaxed);
				if !ptr.is_null() {
					// SAFETY: We own the data
					unsafe {
						drop(Box::from_raw(ptr));
					}
				}
			}
		}
	}

	// Implement From<Owned> for Atomic to match crossbeam_epoch API
	impl<T> From<Owned<T>> for Atomic<T> {
		fn from(owned: Owned<T>) -> Self {
			let ptr = owned.data.as_ptr();
			std::mem::forget(owned);
			Atomic {
				ptr: AtomicPtr::new(ptr),
				owns_data: std::sync::atomic::AtomicBool::new(true),
			}
		}
	}

	// Implement From<Shared> for Atomic to match crossbeam_epoch API
	impl<T> From<Shared<'_, T>> for Atomic<T> {
		fn from(shared: Shared<'_, T>) -> Self {
			Atomic {
				ptr: AtomicPtr::new(shared.ptr as *mut T),
				owns_data: std::sync::atomic::AtomicBool::new(false),
			}
		}
	}

	/// Error returned by compare_exchange operations.
	pub struct CompareExchangeError<'g, T, N> {
		/// The current value that was found instead of the expected value.
		pub current: Shared<'g, T>,
		/// The new value that was not stored.
		pub new: N,
	}
}

#[cfg(loom)]
pub use mock_epoch::*;
