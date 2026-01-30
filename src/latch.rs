//! Implementation of a hybrid latch based on the LeanStore paper.
//!
//! The key difference from a standard `RwLock` is the ability of acquiring optimistic read access
//! without performing any writes to memory. This mode of access is called optimistic because it allows reads
//! to the underlying data even though writers may be able to acquire exclusive access without
//! being blocked and perform writes while optimistic access is still in place.
//!
//! Those reads would normally result in undefined behavior, but can be made safe by correctly validating
//! each optimistic access before allowing any side effects to happen. The validation is performed through
//! the [`OptimisticGuard::recheck`] method that returns an [`error::Error::Unwind`] if any writes could have taken place since
//! the acquisition of the optimistic access.
//!
//! We refer to unwinding as the premature return from a function that performed invalid accesses with the
//! error variant [`error::Error::Unwind`].

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error;

/// Simple spin wait implementation
struct SpinWait {
	counter: u32,
}

impl SpinWait {
	fn new() -> Self {
		SpinWait {
			counter: 0,
		}
	}

	fn spin(&mut self) -> bool {
		if self.counter < 10 {
			self.counter += 1;
			std::hint::spin_loop();
			true
		} else if self.counter < 20 {
			self.counter += 1;
			std::thread::yield_now();
			true
		} else {
			std::thread::yield_now();
			false
		}
	}

	fn reset(&mut self) {
		self.counter = 0;
	}
}

/// A hybrid latch that uses versioning to enable optimistic, shared or exclusive access to the
/// underlying data
pub struct HybridLatch<T> {
	version: AtomicUsize,
	lock: RwLock<()>,
	data: UnsafeCell<T>,
}

unsafe impl<T: Send> Send for HybridLatch<T> {}
unsafe impl<T: Send + Sync> Sync for HybridLatch<T> {}

impl<T> HybridLatch<T> {
	/// Creates a new instance of a `HybridLatch<T>` which is unlocked.
	#[inline]
	pub fn new(data: T) -> HybridLatch<T> {
		HybridLatch {
			version: AtomicUsize::new(0),
			data: UnsafeCell::new(data),
			lock: RwLock::new(()),
		}
	}

	/// Locks this `HybridLatch` with exclusive write access, blocking the thread until it can be
	/// acquired.
	///
	/// Returns an RAII guard which will release the exclusive access when dropped
	#[inline]
	pub fn exclusive(&self) -> ExclusiveGuard<'_, T> {
		let guard = self.lock.write();
		let version = self.version.load(Ordering::Relaxed) + 1;
		self.version.store(version, Ordering::Release);
		ExclusiveGuard {
			latch: self,
			guard,
			data: self.data.get(),
			version,
		}
	}

	/// Locks this `HybridLatch` with shared read access, blocking the thread until it can be
	/// acquired.
	///
	/// Returns an RAII guard which will release the shared access when dropped
	#[inline]
	pub fn shared(&self) -> SharedGuard<'_, T> {
		let guard = self.lock.read();
		let version = self.version.load(Ordering::Relaxed);
		SharedGuard {
			latch: self,
			guard,
			data: self.data.get(),
			version,
		}
	}

	/// Acquires optimistic read access from this `HybridLatch`, spinning until it can be acquired.
	///
	/// Optimistic access must be validated before performing any action based on a read of the
	/// underlying data. See [`OptimisticGuard::recheck`] for the details.
	///
	/// Returns an RAII guard which will NOT validate any accesses when dropped.
	#[inline(never)]
	pub fn optimistic_or_spin(&self) -> OptimisticGuard<'_, T> {
		let mut version = self.version.load(Ordering::Acquire);
		if (version & 1) == 1 {
			let mut spinwait = SpinWait::new();
			loop {
				version = self.version.load(Ordering::Acquire);
				if (version & 1) == 1 {
					let result = spinwait.spin();
					if !result {
						spinwait.reset();
					}
					continue;
				} else {
					break;
				}
			}
		}

		OptimisticGuard {
			latch: self,
			data: self.data.get(),
			version,
		}
	}

	/// Tries to acquire optimistic read access from this `HybridLatch`, unwinding on contention.
	///
	/// Optimistic access must be validated before performing any action based on a read of the
	/// underlying data. See [`OptimisticGuard::recheck`] for the details.
	///
	/// Returns an RAII guard which will NOT validate any accesses when dropped.
	#[inline]
	#[allow(dead_code)]
	pub fn optimistic_or_unwind(&self) -> OptimisticOrShared<'_, T> {
		let version = self.version.load(Ordering::Acquire);
		if (version & 1) == 1 {
			let guard = self.lock.read();
			let version = self.version.load(Ordering::Relaxed);
			OptimisticOrShared::Shared(SharedGuard {
				latch: self,
				guard,
				data: self.data.get(),
				version,
			})
		} else {
			OptimisticOrShared::Optimistic(OptimisticGuard {
				latch: self,
				data: self.data.get(),
				version,
			})
		}
	}

	/// Tries to acquire optimistic read access from this `HybridLatch`, falling back to exclusive
	/// access on contention.
	///
	/// Optimistic access must be validated before performing any action based on a read of the
	/// underlying data. See [`OptimisticGuard::recheck`] for the details.
	///
	/// Acquiring exclusive access will may block the current thread. Reads or writes from exclusive access do not
	/// need to be validated.
	///
	/// Returns either an [`OptimisticGuard`] or an [`ExclusiveGuard`] through the [`OptimisticOrExclusive`] enum.
	#[inline]
	#[allow(dead_code)]
	pub fn optimistic_or_exclusive(&self) -> OptimisticOrExclusive<'_, T> {
		let version = self.version.load(Ordering::Acquire);
		if (version & 1) == 1 {
			let guard = self.lock.write();
			let version = self.version.load(Ordering::Relaxed) + 1;
			self.version.store(version, Ordering::Release);
			OptimisticOrExclusive::Exclusive(ExclusiveGuard {
				latch: self,
				guard,
				data: self.data.get(),
				version,
			})
		} else {
			OptimisticOrExclusive::Optimistic(OptimisticGuard {
				latch: self,
				data: self.data.get(),
				version,
			})
		}
	}
}

impl<T> std::convert::AsMut<T> for HybridLatch<T> {
	#[inline]
	fn as_mut(&mut self) -> &mut T {
		unsafe { &mut *self.data.get() }
	}
}

/// Trait to allow using any guard when only read access is needed.
pub trait HybridGuard<T> {
	/// Allows read access to the underlying data, which must be validated before any side effects
	fn inner(&self) -> &T;

	/// Validates any accesses performed.
	///
	/// The user of a `HybridGuard` must validate all accesses because there is no guarantee of which
	/// mode the accesses are being performed.
	///
	/// If validation fails it returns [`error::Error::Unwind`].
	fn recheck(&self) -> error::Result<()>;

	/// Returns a reference to the original `HybridLatch` struct
	fn latch(&self) -> &HybridLatch<T>;
}

/// Structure used to perform optimistic accesses and validation.
pub struct OptimisticGuard<'a, T> {
	latch: &'a HybridLatch<T>,
	data: *const T,
	version: usize,
}

unsafe impl<'a, T: Sync> Sync for OptimisticGuard<'a, T> {}

impl<'a, T> OptimisticGuard<'a, T> {
	/// Validates all previous optimistic accesses since the creation of the guard,
	/// if validation fails an [`error::Error::Unwind`] is returned to signal that the
	/// stack should be unwinded (by conditional returns) to a safe state.
	#[inline]
	pub fn recheck(&self) -> error::Result<()> {
		if self.version != self.latch.version.load(Ordering::Acquire) {
			return Err(error::Error::Unwind);
		}
		Ok(())
	}

	/// Tries to acquire shared access after validation of all previous optimistic accesses on
	/// this guard.
	///
	/// If validation fails it returns [`error::Error::Unwind`].
	#[inline]
	pub fn to_shared(self) -> error::Result<SharedGuard<'a, T>> {
		if let Some(guard) = self.latch.lock.try_read() {
			if self.version != self.latch.version.load(Ordering::Relaxed) {
				return Err(error::Error::Unwind);
			}

			Ok(SharedGuard {
				latch: self.latch,
				guard,
				data: self.data,
				version: self.version,
			})
		} else {
			Err(error::Error::Unwind)
		}
	}

	/// Tries to acquire exclusive access after validation.
	///
	/// If validation fails it returns [`error::Error::Unwind`].
	#[inline]
	pub fn to_exclusive(self) -> error::Result<ExclusiveGuard<'a, T>> {
		if let Some(guard) = self.latch.lock.try_write() {
			if self.version != self.latch.version.load(Ordering::Relaxed) {
				return Err(error::Error::Unwind);
			}

			let version = self.version + 1;
			self.latch.version.store(version, Ordering::Release);

			Ok(ExclusiveGuard {
				latch: self.latch,
				guard,
				data: self.data as *mut T,
				version,
			})
		} else {
			Err(error::Error::Unwind)
		}
	}

	/// Returns a reference to the original `HybridLatch` struct
	pub fn latch(&self) -> &'a HybridLatch<T> {
		self.latch
	}
}

impl<'a, T> std::ops::Deref for OptimisticGuard<'a, T> {
	type Target = T;

	fn deref(&self) -> &T {
		unsafe { &*self.data }
	}
}

impl<'a, T> HybridGuard<T> for OptimisticGuard<'a, T> {
	fn inner(&self) -> &T {
		self
	}
	fn recheck(&self) -> error::Result<()> {
		self.recheck()
	}
	fn latch(&self) -> &HybridLatch<T> {
		self.latch()
	}
}

/// RAII structure used to release the exclusive write access of a latch when dropped.
pub struct ExclusiveGuard<'a, T> {
	latch: &'a HybridLatch<T>,
	#[allow(dead_code)]
	guard: RwLockWriteGuard<'a, ()>,
	data: *mut T,
	version: usize,
}

unsafe impl<'a, T: Sync> Sync for ExclusiveGuard<'a, T> {}

impl<'a, T> ExclusiveGuard<'a, T> {
	/// A sanity assertion, exclusive guards do not need to be validated
	#[inline]
	pub fn recheck(&self) {
		assert!(self.version == self.latch.version.load(Ordering::Relaxed));
	}

	/// Unlocks the `HybridLatch` returning a [`OptimisticGuard`] in the current version
	#[inline]
	pub fn unlock(self) -> OptimisticGuard<'a, T> {
		let new_version = self.version + 1;
		let latch = self.latch;
		let data = self.data;
		// The version is incremented in drop
		drop(self);
		OptimisticGuard {
			latch,
			data,
			version: new_version,
		}
	}

	/// Returns a reference to the original `HybridLatch` struct
	pub fn latch(&self) -> &'a HybridLatch<T> {
		self.latch
	}
}

impl<'a, T> Drop for ExclusiveGuard<'a, T> {
	#[inline]
	fn drop(&mut self) {
		let new_version = self.version + 1;
		self.latch.version.store(new_version, Ordering::Release);
	}
}

impl<'a, T> std::ops::Deref for ExclusiveGuard<'a, T> {
	type Target = T;

	#[inline]
	fn deref(&self) -> &T {
		unsafe { &*self.data }
	}
}

impl<'a, T> std::ops::DerefMut for ExclusiveGuard<'a, T> {
	#[inline]
	fn deref_mut(&mut self) -> &mut T {
		unsafe { &mut *self.data }
	}
}

impl<'a, T> std::convert::AsMut<T> for ExclusiveGuard<'a, T> {
	#[inline]
	fn as_mut(&mut self) -> &mut T {
		unsafe { &mut *self.data }
	}
}

impl<'a, T> HybridGuard<T> for ExclusiveGuard<'a, T> {
	fn inner(&self) -> &T {
		self
	}
	fn recheck(&self) -> error::Result<()> {
		self.recheck();
		Ok(())
	}
	fn latch(&self) -> &HybridLatch<T> {
		self.latch()
	}
}

/// RAII structure used to release the shared read access of a latch when dropped.
pub struct SharedGuard<'a, T> {
	latch: &'a HybridLatch<T>,
	#[allow(dead_code)]
	guard: RwLockReadGuard<'a, ()>,
	data: *const T,
	version: usize,
}

unsafe impl<'a, T: Sync> Sync for SharedGuard<'a, T> {}

impl<'a, T> SharedGuard<'a, T> {
	/// A sanity assertion, shared guards do not need to be validated
	#[inline]
	pub fn recheck(&self) {
		assert!(self.version == self.latch.version.load(Ordering::Relaxed));
	}

	/// Unlocks the `HybridLatch` returning a [`OptimisticGuard`] in the current version
	#[inline]
	pub fn unlock(self) -> OptimisticGuard<'a, T> {
		OptimisticGuard {
			latch: self.latch,
			data: self.data,
			version: self.version,
		}
	}

	/// Returns a [`OptimisticGuard`] in the current version without consuming the original `SharedGuard`
	#[inline]
	#[allow(dead_code)]
	pub fn as_optimistic(&self) -> OptimisticGuard<'_, T> {
		OptimisticGuard {
			latch: self.latch,
			data: self.data,
			version: self.version,
		}
	}

	/// Returns a reference to the original `HybridLatch` struct
	pub fn latch(&self) -> &'a HybridLatch<T> {
		self.latch
	}
}

impl<'a, T> std::ops::Deref for SharedGuard<'a, T> {
	type Target = T;

	#[inline]
	fn deref(&self) -> &T {
		unsafe { &*self.data }
	}
}

impl<'a, T> HybridGuard<T> for SharedGuard<'a, T> {
	fn inner(&self) -> &T {
		self
	}
	fn recheck(&self) -> error::Result<()> {
		self.recheck();
		Ok(())
	}
	fn latch(&self) -> &HybridLatch<T> {
		self.latch()
	}
}

/// Either an `OptimisticGuard` or a `SharedGuard`.
#[allow(dead_code)]
pub enum OptimisticOrShared<'a, T> {
	Optimistic(OptimisticGuard<'a, T>),
	Shared(SharedGuard<'a, T>),
}

#[allow(dead_code)]
impl<'a, T> OptimisticOrShared<'a, T> {
	#[inline]
	pub fn recheck(&self) -> error::Result<()> {
		match self {
			OptimisticOrShared::Optimistic(g) => g.recheck(),
			OptimisticOrShared::Shared(g) => {
				g.recheck();
				Ok(())
			}
		}
	}
}

/// Either an `OptimisticGuard` or an `ExclusiveGuard`.
#[allow(dead_code)]
pub enum OptimisticOrExclusive<'a, T> {
	Optimistic(OptimisticGuard<'a, T>),
	Exclusive(ExclusiveGuard<'a, T>),
}

#[allow(dead_code)]
impl<'a, T> OptimisticOrExclusive<'a, T> {
	#[inline]
	pub fn recheck(&self) -> error::Result<()> {
		match self {
			OptimisticOrExclusive::Optimistic(g) => g.recheck(),
			OptimisticOrExclusive::Exclusive(g) => {
				g.recheck();
				Ok(())
			}
		}
	}
}
