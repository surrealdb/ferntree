//! Synchronization primitives with loom support.
//!
//! Under normal compilation, re-exports from std/parking_lot/crossbeam.
//! Under `cfg(loom)`, uses loom's equivalents for deterministic testing.
//!
//! # Usage
//!
//! Instead of importing directly from `std::sync::atomic` or `parking_lot`,
//! import from this module:
//!
//! ```ignore
//! use crate::sync::{AtomicUsize, Ordering, RwLock};
//! use crate::sync::epoch::{self as epoch, Atomic, Owned};
//! ```
//!
//! # Loom Integration
//!
//! When compiled with `--cfg loom`, this module provides loom-compatible
//! implementations that allow deterministic testing of concurrent code
//! by exploring all possible thread interleavings.

// Allow unused items - some are only used under loom cfg
#![allow(unused)]

pub mod epoch;

// ===========================================================================
// Atomic Primitives
// ===========================================================================

#[cfg(not(loom))]
pub use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(loom)]
pub use loom::sync::atomic::{AtomicUsize, Ordering};

// ===========================================================================
// RwLock
// ===========================================================================

// Note: parking_lot::RwLock and loom::sync::RwLock have different APIs.
// parking_lot uses RAII guards without Result wrapping, while loom uses
// Result-returning methods. We need wrapper types to unify them.

#[cfg(not(loom))]
pub use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(loom)]
mod loom_rwlock {
	//! Wrapper types for loom's RwLock to match parking_lot's API.

	use loom::sync::{
		RwLock as LoomRwLock, RwLockReadGuard as LoomReadGuard, RwLockWriteGuard as LoomWriteGuard,
	};

	/// A wrapper around loom's RwLock that provides a parking_lot-compatible API.
	pub struct RwLock<T>(LoomRwLock<T>);

	impl<T> RwLock<T> {
		/// Creates a new RwLock.
		pub fn new(value: T) -> Self {
			RwLock(LoomRwLock::new(value))
		}

		/// Acquires a read lock, blocking until available.
		pub fn read(&self) -> RwLockReadGuard<'_, T> {
			RwLockReadGuard(self.0.read().unwrap())
		}

		/// Acquires a write lock, blocking until available.
		pub fn write(&self) -> RwLockWriteGuard<'_, T> {
			RwLockWriteGuard(self.0.write().unwrap())
		}

		/// Attempts to acquire a read lock without blocking.
		pub fn try_read(&self) -> Option<RwLockReadGuard<'_, T>> {
			self.0.try_read().ok().map(RwLockReadGuard)
		}

		/// Attempts to acquire a write lock without blocking.
		pub fn try_write(&self) -> Option<RwLockWriteGuard<'_, T>> {
			self.0.try_write().ok().map(RwLockWriteGuard)
		}
	}

	/// Wrapper around loom's read guard.
	pub struct RwLockReadGuard<'a, T>(LoomReadGuard<'a, T>);

	impl<'a, T> std::ops::Deref for RwLockReadGuard<'a, T> {
		type Target = T;
		fn deref(&self) -> &T {
			&self.0
		}
	}

	/// Wrapper around loom's write guard.
	pub struct RwLockWriteGuard<'a, T>(LoomWriteGuard<'a, T>);

	impl<'a, T> std::ops::Deref for RwLockWriteGuard<'a, T> {
		type Target = T;
		fn deref(&self) -> &T {
			&self.0
		}
	}

	impl<'a, T> std::ops::DerefMut for RwLockWriteGuard<'a, T> {
		fn deref_mut(&mut self) -> &mut T {
			&mut self.0
		}
	}
}

#[cfg(loom)]
pub use loom_rwlock::{RwLock, RwLockReadGuard, RwLockWriteGuard};

// ===========================================================================
// UnsafeCell
// ===========================================================================

// UnsafeCell is from std in both cases, but loom provides its own UnsafeCell
// that integrates with loom's model checker. We need a wrapper for loom
// because loom's UnsafeCell::get() returns ConstPtr<T> instead of *const T.

#[cfg(not(loom))]
pub use std::cell::UnsafeCell;

#[cfg(loom)]
mod loom_unsafe_cell {
	//! Wrapper around loom's UnsafeCell to provide std-compatible API.

	use loom::cell::UnsafeCell as LoomUnsafeCell;

	/// A wrapper around loom's UnsafeCell that provides std's API.
	pub struct UnsafeCell<T>(LoomUnsafeCell<T>);

	impl<T> UnsafeCell<T> {
		/// Creates a new UnsafeCell.
		pub fn new(value: T) -> Self {
			UnsafeCell(LoomUnsafeCell::new(value))
		}

		/// Gets a raw pointer to the underlying data.
		///
		/// # Safety Note
		/// Under loom, this extracts the raw pointer from loom's tracked ConstPtr.
		/// The caller must ensure proper synchronization.
		pub fn get(&self) -> *mut T {
			// SAFETY: We're converting loom's tracked pointer to a raw pointer.
			// The caller is responsible for ensuring this is used correctly.
			unsafe { self.0.get().deref() as *const T as *mut T }
		}

		/// Unwraps the value.
		pub fn into_inner(self) -> T {
			self.0.into_inner()
		}
	}
}

#[cfg(loom)]
pub use loom_unsafe_cell::UnsafeCell;
