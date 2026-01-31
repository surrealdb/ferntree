//! # Hybrid Latch Implementation
//!
//! This module provides the [`HybridLatch`], a concurrency primitive based on the
//! [LeanStore paper](https://dbis1.github.io/leanstore.html) that enables optimistic
//! lock-free reading for high-performance concurrent data structures.
//!
//! ## Overview
//!
//! A `HybridLatch` is like a `RwLock` but with an additional "optimistic" access mode
//! that doesn't acquire any lock at all. This is the key innovation that enables
//! high-throughput concurrent B+ tree operations.
//!
//! ## Access Modes
//!
//! | Mode       | Blocking? | Prevents Writers? | Validates Reads? |
//! |------------|-----------|-------------------|------------------|
//! | Optimistic | No        | No                | Yes (required)   |
//! | Shared     | Yes       | Yes               | No               |
//! | Exclusive  | Yes       | Yes               | No               |
//!
//! ### Optimistic Access
//!
//! Optimistic access is the most efficient read mode. It captures a version number
//! and allows the caller to read data without blocking. However:
//!
//! - Writers can acquire the latch and modify data concurrently
//! - All reads MUST be validated before trusting the data
//! - If validation fails, the operation must be retried
//!
//! ### Shared Access
//!
//! Traditional read lock. Blocks until no writer holds the latch, then blocks
//! subsequent writers. Use when you need guaranteed consistent reads.
//!
//! ### Exclusive Access
//!
//! Traditional write lock. Blocks until no other readers or writers hold the latch.
//! Increments the version number to invalidate optimistic readers.
//!
//! ## Version Encoding
//!
//! The latch uses a version number (`AtomicUsize`) with special encoding:
//!
//! ```text
//! Version bits: [.......X]
//!                       ^
//!                       └── Odd (1) = Write lock held
//!                           Even (0) = No write lock
//!
//! Sequence:
//!   0 (unlocked) -> 1 (write locked) -> 2 (unlocked) -> 3 (write locked) -> ...
//! ```
//!
//! - When a writer acquires the latch: version becomes odd (v + 1)
//! - When a writer releases the latch: version becomes even (v + 1 again)
//! - Optimistic readers check: has the version changed since we started?
//!
//! ## Validation Pattern
//!
//! ```ignore
//! let guard = latch.optimistic_or_spin();
//!
//! // Read some data (might be inconsistent if concurrent write!)
//! let value = guard.some_field;
//!
//! // CRITICAL: Validate before trusting the read
//! guard.recheck()?;  // Returns Err(Unwind) if version changed
//!
//! // Now safe to use `value`
//! ```
//!
//! ## Unwinding
//!
//! When optimistic validation fails, we say the operation must "unwind". This means:
//! - Discard any data read (it may be garbage)
//! - Return `Err(error::Error::Unwind)` to the caller
//! - Caller typically retries the entire operation
//!
//! This is safe because no side effects occurred before validation.

use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error;

// ===========================================================================
// Spin Wait Helper
// ===========================================================================

/// A simple spin-wait implementation with exponential backoff.
///
/// Used when acquiring optimistic access while a write is in progress.
/// Instead of immediately blocking, we spin briefly hoping the writer finishes.
///
/// # Backoff Strategy
///
/// 1. First 10 iterations: CPU spin loop (`spin_loop()`)
/// 2. Next 10 iterations: Thread yield (`yield_now()`)
/// 3. After 20 iterations: Returns `false` to signal "give up spinning"
///
/// This balances low-latency (spinning is faster than sleeping) with
/// fairness (eventually yielding to let other threads run).
struct SpinWait {
	/// Number of spin iterations performed.
	counter: u32,
}

impl SpinWait {
	/// Creates a new `SpinWait` with counter at 0.
	fn new() -> Self {
		SpinWait {
			counter: 0,
		}
	}

	/// Performs one spin iteration.
	///
	/// # Returns
	///
	/// - `true`: Keep spinning (more attempts available)
	/// - `false`: Stop spinning (reached iteration limit)
	fn spin(&mut self) -> bool {
		if self.counter < 10 {
			// Phase 1: CPU spin loop (fastest, no OS involvement)
			self.counter += 1;
			std::hint::spin_loop();
			true
		} else if self.counter < 20 {
			// Phase 2: Yield to OS scheduler (slightly slower, fairer)
			self.counter += 1;
			std::thread::yield_now();
			true
		} else {
			// Phase 3: Exhausted patience, but still yield once
			std::thread::yield_now();
			false
		}
	}

	/// Resets the counter to start fresh spin-waiting.
	fn reset(&mut self) {
		self.counter = 0;
	}
}

// ===========================================================================
// HybridLatch
// ===========================================================================

/// A hybrid latch enabling optimistic, shared, or exclusive access to data.
///
/// This is the core concurrency primitive for the B+ tree. It combines:
/// - A version counter for optimistic locking
/// - A `RwLock` for blocking shared/exclusive access
/// - The actual data protected by the latch
///
/// # Version Encoding
///
/// ```text
/// version = 0: Unlocked (even)
/// version = 1: Write-locked (odd)
/// version = 2: Unlocked (even), one write completed
/// version = 3: Write-locked (odd)
/// ...
/// ```
///
/// The least significant bit indicates lock state:
/// - Even (bit 0 = 0): No exclusive lock held
/// - Odd (bit 0 = 1): Exclusive lock is held
///
/// # Thread Safety
///
/// The latch is `Send` if `T: Send` and `Sync` if `T: Send + Sync`.
pub struct HybridLatch<T> {
	/// Version counter for optimistic validation.
	/// - Odd values indicate write lock is held
	/// - Each exclusive lock acquire/release increments by 1
	version: AtomicUsize,

	/// Traditional RwLock for shared and exclusive access.
	/// Note: The lock protects nothing directly (unit type) - the version
	/// number and UnsafeCell provide the actual synchronization.
	lock: RwLock<()>,

	/// The protected data. Access is coordinated through the version and lock.
	data: UnsafeCell<T>,
}

// SAFETY: HybridLatch can be sent between threads if T can be sent.
unsafe impl<T: Send> Send for HybridLatch<T> {}

// SAFETY: HybridLatch can be shared between threads if T is Send+Sync.
// The latch provides its own synchronization.
unsafe impl<T: Send + Sync> Sync for HybridLatch<T> {}

impl<T> HybridLatch<T> {
	/// Creates a new, unlocked `HybridLatch` containing the given data.
	///
	/// # Example
	///
	/// ```ignore
	/// let latch = HybridLatch::new(42);
	/// ```
	#[inline]
	pub fn new(data: T) -> HybridLatch<T> {
		HybridLatch {
			version: AtomicUsize::new(0), // Start at 0 (even = unlocked)
			data: UnsafeCell::new(data),
			lock: RwLock::new(()),
		}
	}

	// -----------------------------------------------------------------------
	// Lock Acquisition Methods
	// -----------------------------------------------------------------------

	/// Acquires exclusive (write) access, blocking until available.
	///
	/// The returned guard provides mutable access to the data. When dropped,
	/// the lock is released and the version is incremented (invalidating
	/// optimistic readers).
	///
	/// # Version Sequence
	///
	/// ```text
	/// Before: version = 2n (even, unlocked)
	/// After acquire: version = 2n+1 (odd, locked)
	/// After drop: version = 2n+2 (even, unlocked)
	/// ```
	#[inline]
	pub fn exclusive(&self) -> ExclusiveGuard<'_, T> {
		// Acquire the write lock (blocks if held)
		let guard = self.lock.write();

		// Increment version to odd (signals "write in progress")
		// This invalidates any optimistic readers who started before us
		let version = self.version.load(Ordering::Relaxed) + 1;
		self.version.store(version, Ordering::Release);

		ExclusiveGuard {
			latch: self,
			guard,
			data: self.data.get(),
			version,
		}
	}

	/// Acquires shared (read) access, blocking until available.
	///
	/// The returned guard provides immutable access to the data. Multiple
	/// shared guards can exist simultaneously, but they block exclusive access.
	///
	/// Unlike optimistic access, shared access doesn't need validation - the
	/// lock guarantees no concurrent writes.
	#[inline]
	pub fn shared(&self) -> SharedGuard<'_, T> {
		// Acquire the read lock (blocks if write lock held)
		let guard = self.lock.read();

		// Capture current version (for debugging/assertions, not validation)
		let version = self.version.load(Ordering::Relaxed);

		SharedGuard {
			latch: self,
			guard,
			data: self.data.get(),
			version,
		}
	}

	/// Acquires optimistic (non-blocking) read access, spinning if write-locked.
	///
	/// This is the primary method for reading in the B+ tree. It captures the
	/// current version and returns immediately if no writer holds the lock.
	/// If a writer is active, it spins briefly waiting for the write to complete.
	///
	/// # Important
	///
	/// **All reads through the returned guard MUST be validated** with
	/// [`OptimisticGuard::recheck`] before trusting the data. Failure to validate
	/// can result in using garbage data.
	///
	/// # Spin Behavior
	///
	/// If the latch is write-locked (version is odd), this method spins:
	/// 1. First 10 iterations: CPU spin loop
	/// 2. Next 10 iterations: Thread yield
	/// 3. After that: Keeps yielding and retrying
	///
	/// The spin is designed to be brief, as writes in the B+ tree are typically fast.
	#[inline(never)]
	pub fn optimistic_or_spin(&self) -> OptimisticGuard<'_, T> {
		// Load current version
		let mut version = self.version.load(Ordering::Acquire);

		// Check if write-locked (odd version)
		if (version & 1) == 1 {
			// Write in progress - spin until it completes
			let mut spinwait = SpinWait::new();
			loop {
				version = self.version.load(Ordering::Acquire);
				if (version & 1) == 1 {
					// Still locked - continue spinning
					let result = spinwait.spin();
					if !result {
						// Exhausted spin budget - reset and keep trying
						// (We don't give up because we need to acquire access)
						spinwait.reset();
					}
					continue;
				} else {
					// Write completed - version is now even
					break;
				}
			}
		}

		// Return optimistic guard with captured version
		OptimisticGuard {
			latch: self,
			data: self.data.get(),
			version,
		}
	}

	/// Tries to acquire optimistic access, falling back to shared on contention.
	///
	/// This is an alternative to `optimistic_or_spin` that uses blocking shared
	/// access instead of spinning when a write is in progress.
	///
	/// # When to Use
	///
	/// Use when you prefer blocking over spinning, or when writes are expected
	/// to be long-running.
	#[inline]
	#[allow(dead_code)]
	pub fn optimistic_or_unwind(&self) -> OptimisticOrShared<'_, T> {
		let version = self.version.load(Ordering::Acquire);
		if (version & 1) == 1 {
			// Write in progress - fall back to shared (blocking) access
			let guard = self.lock.read();
			let version = self.version.load(Ordering::Relaxed);
			OptimisticOrShared::Shared(SharedGuard {
				latch: self,
				guard,
				data: self.data.get(),
				version,
			})
		} else {
			// Not write-locked - use optimistic access
			OptimisticOrShared::Optimistic(OptimisticGuard {
				latch: self,
				data: self.data.get(),
				version,
			})
		}
	}

	/// Tries to acquire optimistic access, falling back to exclusive on contention.
	///
	/// This is useful when you need to read data but might need to modify it
	/// based on what you read. If a write is in progress, you get exclusive
	/// access and can proceed without re-reading.
	#[inline]
	#[allow(dead_code)]
	pub fn optimistic_or_exclusive(&self) -> OptimisticOrExclusive<'_, T> {
		let version = self.version.load(Ordering::Acquire);
		if (version & 1) == 1 {
			// Write in progress - fall back to exclusive access
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
			// Not write-locked - use optimistic access
			OptimisticOrExclusive::Optimistic(OptimisticGuard {
				latch: self,
				data: self.data.get(),
				version,
			})
		}
	}
}

impl<T> std::convert::AsMut<T> for HybridLatch<T> {
	/// Returns mutable access to the data without any synchronization.
	///
	/// # Safety
	///
	/// This is only safe when you have `&mut self`, which guarantees
	/// no other references exist.
	#[inline]
	fn as_mut(&mut self) -> &mut T {
		// SAFETY: We have &mut self, so no other references exist
		unsafe { &mut *self.data.get() }
	}
}

// ===========================================================================
// HybridGuard Trait
// ===========================================================================

/// A unified interface for all guard types.
///
/// This trait allows code to work with any guard type when it only needs
/// read access. The key methods are:
/// - `inner()`: Get a reference to the protected data
/// - `recheck()`: Validate optimistic reads (may be a no-op for blocking guards)
/// - `latch()`: Get a reference to the underlying latch
///
/// This is particularly useful in the B+ tree for `find_parent()` which
/// accepts any guard type.
pub trait HybridGuard<T> {
	/// Returns a reference to the underlying data.
	///
	/// **Warning**: For optimistic guards, this data may be inconsistent.
	/// Always call `recheck()` before trusting the data.
	fn inner(&self) -> &T;

	/// Validates all reads performed through this guard.
	///
	/// - For `OptimisticGuard`: Checks if version changed (may return `Err(Unwind)`)
	/// - For `SharedGuard`/`ExclusiveGuard`: Always succeeds (blocking guarantees consistency)
	///
	/// # Errors
	///
	/// Returns `Err(error::Error::Unwind)` if validation fails, indicating
	/// the caller should discard all read data and retry.
	fn recheck(&self) -> error::Result<()>;

	/// Returns a reference to the `HybridLatch` this guard is associated with.
	///
	/// Useful for operations like comparing latch pointers (identity checks)
	/// or acquiring different lock types on the same latch.
	fn latch(&self) -> &HybridLatch<T>;
}

// ===========================================================================
// OptimisticGuard
// ===========================================================================

/// A guard for optimistic (non-blocking) read access.
///
/// This guard captures a version number when created and allows reading
/// the protected data without blocking. However, because no lock is held,
/// concurrent writers can modify the data at any time.
///
/// # Critical: Validation Required
///
/// **You MUST call `recheck()` before trusting any data read through this guard.**
///
/// The typical pattern is:
///
/// ```ignore
/// let guard = latch.optimistic_or_spin();
/// let value = guard.field;  // Read data (may be garbage!)
/// guard.recheck()?;          // Validate the read
/// // Now `value` is safe to use
/// ```
///
/// # What Happens Without Validation
///
/// If you skip validation and a write occurred:
/// - You might read partially-written data (torn reads)
/// - You might read data that no longer exists
/// - Your logic might behave incorrectly based on stale data
/// - **This is undefined behavior** in the formal sense
///
/// # Version Check
///
/// `recheck()` compares the version captured at guard creation with the
/// current version. If they differ, a write occurred (or is in progress),
/// and the guard is invalid.
pub struct OptimisticGuard<'a, T> {
	/// Reference to the latch for version checking.
	latch: &'a HybridLatch<T>,
	/// Raw pointer to the data.
	data: *const T,
	/// Version captured at guard creation.
	version: usize,
}

// SAFETY: OptimisticGuard can be shared if T is Sync.
// Even though it holds a raw pointer, access is read-only and
// coordinated through version checking.
unsafe impl<'a, T: Sync> Sync for OptimisticGuard<'a, T> {}

impl<'a, T> OptimisticGuard<'a, T> {
	/// Validates all reads performed through this guard.
	///
	/// This is the **critical** method that makes optimistic reads safe.
	/// It checks if the latch's version has changed since this guard was created.
	///
	/// # Returns
	///
	/// - `Ok(())`: Version unchanged - all reads are valid
	/// - `Err(Error::Unwind)`: Version changed - discard all reads and retry
	///
	/// # Example
	///
	/// ```ignore
	/// let guard = latch.optimistic_or_spin();
	/// let data = guard.some_field;  // Potentially invalid read
	/// guard.recheck()?;              // Validate
	/// process(data);                 // Safe to use data now
	/// ```
	#[inline]
	pub fn recheck(&self) -> error::Result<()> {
		// Compare current version with captured version
		// If they differ, a write occurred and our reads are invalid
		if self.version != self.latch.version.load(Ordering::Acquire) {
			return Err(error::Error::Unwind);
		}
		Ok(())
	}

	/// Upgrades to a shared (blocking read) guard after validation.
	///
	/// This is useful when you want to "lock in" your position after
	/// optimistically finding what you're looking for.
	///
	/// # Algorithm
	///
	/// 1. Try to acquire read lock (non-blocking try_read)
	/// 2. Re-validate version hasn't changed
	/// 3. Return SharedGuard if both succeed
	///
	/// # Returns
	///
	/// - `Ok(SharedGuard)`: Successfully upgraded
	/// - `Err(Unwind)`: Lock couldn't be acquired or version changed
	#[inline]
	pub fn to_shared(self) -> error::Result<SharedGuard<'a, T>> {
		// Try to acquire read lock without blocking
		if let Some(guard) = self.latch.lock.try_read() {
			// Double-check version hasn't changed while we acquired the lock
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
			// Lock is held by a writer - can't upgrade
			Err(error::Error::Unwind)
		}
	}

	/// Upgrades to an exclusive (blocking write) guard after validation.
	///
	/// Similar to `to_shared`, but acquires write access. This increments
	/// the version number, invalidating other optimistic readers.
	///
	/// # Returns
	///
	/// - `Ok(ExclusiveGuard)`: Successfully upgraded
	/// - `Err(Unwind)`: Lock couldn't be acquired or version changed
	#[inline]
	pub fn to_exclusive(self) -> error::Result<ExclusiveGuard<'a, T>> {
		// Try to acquire write lock without blocking
		if let Some(guard) = self.latch.lock.try_write() {
			// Validate version hasn't changed
			if self.version != self.latch.version.load(Ordering::Relaxed) {
				return Err(error::Error::Unwind);
			}

			// Increment version to odd (write in progress)
			let version = self.version + 1;
			self.latch.version.store(version, Ordering::Release);

			Ok(ExclusiveGuard {
				latch: self.latch,
				guard,
				data: self.data as *mut T,
				version,
			})
		} else {
			// Lock is held - can't upgrade
			Err(error::Error::Unwind)
		}
	}

	/// Returns a reference to the underlying latch.
	pub fn latch(&self) -> &'a HybridLatch<T> {
		self.latch
	}
}

impl<'a, T> std::ops::Deref for OptimisticGuard<'a, T> {
	type Target = T;

	/// Dereferences to the underlying data.
	///
	/// **Warning**: The data may be inconsistent! Always `recheck()` after reading.
	fn deref(&self) -> &T {
		// SAFETY: The pointer is valid for the latch's lifetime.
		// However, the DATA may be inconsistent - caller must validate.
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

// ===========================================================================
// ExclusiveGuard
// ===========================================================================

/// RAII guard for exclusive (write) access to a latch.
///
/// This guard provides mutable access to the protected data. While held:
/// - No other readers or writers can access the data
/// - The version number is odd, invalidating optimistic readers
///
/// When dropped, the version is incremented again (becoming even), and
/// the write lock is released.
///
/// # Version Lifecycle
///
/// ```text
/// Acquire: version 2n -> 2n+1 (odd, write in progress)
/// Drop:    version 2n+1 -> 2n+2 (even, unlocked)
/// ```
pub struct ExclusiveGuard<'a, T> {
	/// Reference to the latch.
	latch: &'a HybridLatch<T>,
	/// The actual RwLock write guard (ensures mutual exclusion).
	#[allow(dead_code)]
	guard: RwLockWriteGuard<'a, ()>,
	/// Mutable pointer to the data.
	data: *mut T,
	/// Version at acquisition (odd).
	version: usize,
}

// SAFETY: ExclusiveGuard can be shared if T is Sync (though mutable ops need &mut self)
unsafe impl<'a, T: Sync> Sync for ExclusiveGuard<'a, T> {}

impl<'a, T> ExclusiveGuard<'a, T> {
	/// Debug assertion that we still own the lock.
	///
	/// This is a sanity check - exclusive guards don't need validation
	/// because the lock guarantees exclusive access.
	#[inline]
	pub fn recheck(&self) {
		assert!(self.version == self.latch.version.load(Ordering::Relaxed));
	}

	/// Releases the write lock and returns an optimistic guard.
	///
	/// This is useful when you've finished writing and want to continue
	/// reading without blocking other readers. The returned optimistic
	/// guard has the new (post-write) version.
	///
	/// # Version Sequence
	///
	/// ```text
	/// Before unlock: version = 2n+1 (odd, locked)
	/// After unlock:  version = 2n+2 (even, unlocked)
	/// Returned guard has version 2n+2
	/// ```
	#[inline]
	pub fn unlock(self) -> OptimisticGuard<'a, T> {
		// The version will become self.version + 1 after drop
		let new_version = self.version + 1;
		let latch = self.latch;
		let data = self.data;

		// Drop self, which increments version and releases lock
		drop(self);

		// Return optimistic guard with the new version
		OptimisticGuard {
			latch,
			data,
			version: new_version,
		}
	}

	/// Returns a reference to the underlying latch.
	pub fn latch(&self) -> &'a HybridLatch<T> {
		self.latch
	}
}

impl<'a, T> Drop for ExclusiveGuard<'a, T> {
	/// Releases the exclusive lock and increments the version.
	///
	/// The version increment is critical - it signals to any optimistic
	/// readers that started during our write that their reads are invalid.
	#[inline]
	fn drop(&mut self) {
		// Increment version to even (write complete)
		let new_version = self.version + 1;
		self.latch.version.store(new_version, Ordering::Release);
		// RwLockWriteGuard is dropped automatically, releasing the lock
	}
}

impl<'a, T> std::ops::Deref for ExclusiveGuard<'a, T> {
	type Target = T;

	/// Returns an immutable reference to the data.
	#[inline]
	fn deref(&self) -> &T {
		// SAFETY: We hold the exclusive lock, so no concurrent access
		unsafe { &*self.data }
	}
}

impl<'a, T> std::ops::DerefMut for ExclusiveGuard<'a, T> {
	/// Returns a mutable reference to the data.
	#[inline]
	fn deref_mut(&mut self) -> &mut T {
		// SAFETY: We hold the exclusive lock, so no concurrent access
		unsafe { &mut *self.data }
	}
}

impl<'a, T> std::convert::AsMut<T> for ExclusiveGuard<'a, T> {
	#[inline]
	fn as_mut(&mut self) -> &mut T {
		// SAFETY: Same as DerefMut
		unsafe { &mut *self.data }
	}
}

impl<'a, T> HybridGuard<T> for ExclusiveGuard<'a, T> {
	fn inner(&self) -> &T {
		self
	}
	fn recheck(&self) -> error::Result<()> {
		// Exclusive guards never fail validation
		self.recheck();
		Ok(())
	}
	fn latch(&self) -> &HybridLatch<T> {
		self.latch()
	}
}

// ===========================================================================
// SharedGuard
// ===========================================================================

/// RAII guard for shared (read) access to a latch.
///
/// This guard provides immutable access to the protected data. While held:
/// - Other readers can access the data (shared access)
/// - Writers are blocked
/// - The version number remains stable (no writes occurring)
///
/// Unlike optimistic guards, shared guards don't need validation - the
/// lock guarantees no concurrent writes.
pub struct SharedGuard<'a, T> {
	/// Reference to the latch.
	latch: &'a HybridLatch<T>,
	/// The actual RwLock read guard (blocks writers).
	#[allow(dead_code)]
	guard: RwLockReadGuard<'a, ()>,
	/// Immutable pointer to the data.
	data: *const T,
	/// Version at acquisition (should be even, no write in progress).
	version: usize,
}

// SAFETY: SharedGuard can be shared if T is Sync
unsafe impl<'a, T: Sync> Sync for SharedGuard<'a, T> {}

impl<'a, T> SharedGuard<'a, T> {
	/// Debug assertion that the version hasn't changed.
	///
	/// This should never fail for a shared guard (we block writers),
	/// but it's useful as a sanity check.
	#[inline]
	pub fn recheck(&self) {
		assert!(self.version == self.latch.version.load(Ordering::Relaxed));
	}

	/// Releases the shared lock and returns an optimistic guard.
	///
	/// The returned guard has the same version as this guard (no writes
	/// occurred while we held the lock).
	#[inline]
	pub fn unlock(self) -> OptimisticGuard<'a, T> {
		OptimisticGuard {
			latch: self.latch,
			data: self.data,
			version: self.version,
		}
	}

	/// Creates an optimistic guard without releasing this shared guard.
	///
	/// This is useful when you want to pass an optimistic guard to a function
	/// but continue holding the shared lock.
	#[inline]
	#[allow(dead_code)]
	pub fn as_optimistic(&self) -> OptimisticGuard<'_, T> {
		OptimisticGuard {
			latch: self.latch,
			data: self.data,
			version: self.version,
		}
	}

	/// Returns a reference to the underlying latch.
	pub fn latch(&self) -> &'a HybridLatch<T> {
		self.latch
	}
}

impl<'a, T> std::ops::Deref for SharedGuard<'a, T> {
	type Target = T;

	/// Returns an immutable reference to the data.
	#[inline]
	fn deref(&self) -> &T {
		// SAFETY: We hold the read lock, blocking writers
		unsafe { &*self.data }
	}
}

impl<'a, T> HybridGuard<T> for SharedGuard<'a, T> {
	fn inner(&self) -> &T {
		self
	}
	fn recheck(&self) -> error::Result<()> {
		// Shared guards never fail validation (we block writers)
		self.recheck();
		Ok(())
	}
	fn latch(&self) -> &HybridLatch<T> {
		self.latch()
	}
}

// ===========================================================================
// Fallback Enums
// ===========================================================================

/// Result of `optimistic_or_unwind()` - either optimistic or shared access.
///
/// This enum is returned when you want optimistic access but are willing to
/// fall back to shared (blocking) access if contention is detected.
///
/// Use the unified `recheck()` method to validate regardless of which
/// variant you have.
#[allow(dead_code)]
pub enum OptimisticOrShared<'a, T> {
	/// Got optimistic access (no blocking).
	Optimistic(OptimisticGuard<'a, T>),
	/// Fell back to shared access (blocking, but read-only).
	Shared(SharedGuard<'a, T>),
}

#[allow(dead_code)]
impl<'a, T> OptimisticOrShared<'a, T> {
	/// Validates reads through this guard.
	///
	/// - For `Optimistic`: Actually validates (may return `Err(Unwind)`)
	/// - For `Shared`: Always succeeds (blocking guarantees consistency)
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

/// Result of `optimistic_or_exclusive()` - either optimistic or exclusive access.
///
/// This enum is returned when you want optimistic access but are willing to
/// fall back to exclusive (blocking write) access if contention is detected.
///
/// This is useful when you might need to write based on what you read - if
/// you get exclusive access, you can proceed without re-reading.
#[allow(dead_code)]
pub enum OptimisticOrExclusive<'a, T> {
	/// Got optimistic access (no blocking).
	Optimistic(OptimisticGuard<'a, T>),
	/// Fell back to exclusive access (blocking write access).
	Exclusive(ExclusiveGuard<'a, T>),
}

#[allow(dead_code)]
impl<'a, T> OptimisticOrExclusive<'a, T> {
	/// Validates reads through this guard.
	///
	/// - For `Optimistic`: Actually validates (may return `Err(Unwind)`)
	/// - For `Exclusive`: Always succeeds (we have write access)
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
	use super::*;

	// -----------------------------------------------------------------------
	// Version Encoding Tests
	// -----------------------------------------------------------------------

	#[test]
	fn version_starts_at_zero() {
		let latch = HybridLatch::new(42);
		// New latch should have version 0 (even = unlocked)
		assert_eq!(latch.version.load(Ordering::Relaxed), 0);
	}

	#[test]
	fn exclusive_lock_makes_version_odd() {
		let latch = HybridLatch::new(42);
		let guard = latch.exclusive();
		// While exclusive lock is held, version should be odd
		assert_eq!(latch.version.load(Ordering::Relaxed) & 1, 1);
		drop(guard);
	}

	#[test]
	fn exclusive_unlock_makes_version_even() {
		let latch = HybridLatch::new(42);
		{
			let _guard = latch.exclusive();
		}
		// After exclusive lock is released, version should be even (and incremented)
		let version = latch.version.load(Ordering::Relaxed);
		assert_eq!(version & 1, 0);
		assert_eq!(version, 2); // 0 -> 1 (lock) -> 2 (unlock)
	}

	#[test]
	fn multiple_exclusive_locks_increment_version() {
		let latch = HybridLatch::new(42);

		for i in 0..5 {
			{
				let _guard = latch.exclusive();
				// During lock: version should be 2*i + 1
				assert_eq!(latch.version.load(Ordering::Relaxed), 2 * i + 1);
			}
			// After unlock: version should be 2*(i+1)
			assert_eq!(latch.version.load(Ordering::Relaxed), 2 * (i + 1));
		}
	}

	// -----------------------------------------------------------------------
	// Optimistic Validation Tests
	// -----------------------------------------------------------------------

	#[test]
	fn optimistic_recheck_succeeds_without_write() {
		let latch = HybridLatch::new(42);
		let guard = latch.optimistic_or_spin();

		// Read the value
		let value = *guard;
		assert_eq!(value, 42);

		// Recheck should succeed - no concurrent write
		assert!(guard.recheck().is_ok());
	}

	#[test]
	fn optimistic_recheck_fails_after_write() {
		let latch = HybridLatch::new(42);
		let opt = latch.optimistic_or_spin();

		// Concurrent write happens
		{
			let mut exc = latch.exclusive();
			*exc = 100;
		}

		// Recheck should fail - version changed
		assert!(opt.recheck().is_err());
	}

	#[test]
	fn optimistic_recheck_fails_during_write() {
		let latch = HybridLatch::new(42);

		// First, acquire exclusive to set version to odd
		let exc = latch.exclusive();

		// Another thread (simulated) would see odd version and spin
		// We can't easily test the spinning behavior, but we can verify
		// that the version is odd during exclusive lock
		assert_eq!(latch.version.load(Ordering::Relaxed) & 1, 1);

		drop(exc);
	}

	#[test]
	fn optimistic_captures_correct_version() {
		let latch = HybridLatch::new(42);

		// Do some writes to increment version
		for _ in 0..3 {
			let _guard = latch.exclusive();
		}

		let opt = latch.optimistic_or_spin();
		// Version should be 6 after 3 exclusive locks
		assert_eq!(opt.version, 6);
		assert!(opt.recheck().is_ok());
	}

	// -----------------------------------------------------------------------
	// Lock Upgrade Tests
	// -----------------------------------------------------------------------

	#[test]
	fn to_shared_succeeds_without_contention() {
		let latch = HybridLatch::new(42);
		let opt = latch.optimistic_or_spin();

		let shared = opt.to_shared();
		assert!(shared.is_ok());

		let shared = shared.unwrap();
		assert_eq!(*shared, 42);
	}

	#[test]
	fn to_shared_fails_after_write() {
		let latch = HybridLatch::new(42);
		let opt = latch.optimistic_or_spin();

		// Concurrent write
		{
			let mut exc = latch.exclusive();
			*exc = 100;
		}

		// Upgrade should fail due to version change
		assert!(opt.to_shared().is_err());
	}

	#[test]
	fn to_exclusive_succeeds_without_contention() {
		let latch = HybridLatch::new(42);
		let opt = latch.optimistic_or_spin();

		let exc = opt.to_exclusive();
		assert!(exc.is_ok());

		let mut exc = exc.unwrap();
		*exc = 100;
		drop(exc);

		// Verify the write took effect
		let opt2 = latch.optimistic_or_spin();
		assert_eq!(*opt2, 100);
	}

	#[test]
	fn to_exclusive_fails_after_write() {
		let latch = HybridLatch::new(42);
		let opt = latch.optimistic_or_spin();

		// Concurrent write
		{
			let mut exc = latch.exclusive();
			*exc = 100;
		}

		// Upgrade should fail due to version change
		assert!(opt.to_exclusive().is_err());
	}

	#[test]
	fn to_exclusive_increments_version() {
		let latch = HybridLatch::new(42);
		let opt = latch.optimistic_or_spin();
		let initial_version = opt.version;

		let exc = opt.to_exclusive().unwrap();
		// Version should be odd (initial + 1)
		assert_eq!(latch.version.load(Ordering::Relaxed), initial_version + 1);

		drop(exc);
		// Version should be even (initial + 2)
		assert_eq!(latch.version.load(Ordering::Relaxed), initial_version + 2);
	}

	// -----------------------------------------------------------------------
	// Guard Unlock Tests
	// -----------------------------------------------------------------------

	#[test]
	fn exclusive_unlock_returns_optimistic_with_new_version() {
		let latch = HybridLatch::new(42);
		let exc = latch.exclusive();
		let version_during_lock = exc.version;

		let opt = exc.unlock();

		// Returned optimistic guard should have version = version_during_lock + 1
		assert_eq!(opt.version, version_during_lock + 1);
		// And recheck should succeed (we just unlocked, no one else modified)
		assert!(opt.recheck().is_ok());
	}

	#[test]
	fn shared_unlock_returns_optimistic_with_same_version() {
		let latch = HybridLatch::new(42);
		let shared = latch.shared();
		let version = shared.version;

		let opt = shared.unlock();

		// Shared unlock doesn't change version
		assert_eq!(opt.version, version);
		assert!(opt.recheck().is_ok());
	}

	// -----------------------------------------------------------------------
	// Shared Lock Tests
	// -----------------------------------------------------------------------

	#[test]
	fn shared_lock_provides_read_access() {
		let latch = HybridLatch::new(42);
		let shared = latch.shared();
		assert_eq!(*shared, 42);
	}

	#[test]
	fn multiple_shared_locks_allowed() {
		let latch = HybridLatch::new(42);
		let shared1 = latch.shared();
		let shared2 = latch.shared();

		assert_eq!(*shared1, 42);
		assert_eq!(*shared2, 42);
	}

	#[test]
	fn shared_recheck_always_succeeds() {
		let latch = HybridLatch::new(42);
		let shared = latch.shared();

		// Shared guards don't need validation - they block writers
		// The recheck is just an assertion that should pass
		shared.recheck();
	}

	// -----------------------------------------------------------------------
	// Exclusive Lock Tests
	// -----------------------------------------------------------------------

	#[test]
	fn exclusive_lock_provides_write_access() {
		let latch = HybridLatch::new(42);
		{
			let mut exc = latch.exclusive();
			*exc = 100;
		}

		let opt = latch.optimistic_or_spin();
		assert_eq!(*opt, 100);
	}

	#[test]
	fn exclusive_deref_mut_works() {
		let latch = HybridLatch::new(vec![1, 2, 3]);
		{
			let mut exc = latch.exclusive();
			exc.push(4);
		}

		let opt = latch.optimistic_or_spin();
		assert_eq!(*opt, vec![1, 2, 3, 4]);
	}

	// -----------------------------------------------------------------------
	// SpinWait Tests
	// -----------------------------------------------------------------------

	#[test]
	fn spinwait_returns_true_initially() {
		let mut sw = SpinWait::new();
		// First 20 spins should return true
		for _ in 0..20 {
			assert!(sw.spin());
		}
	}

	#[test]
	fn spinwait_returns_false_after_exhaustion() {
		let mut sw = SpinWait::new();
		// Exhaust the spin budget
		for _ in 0..20 {
			sw.spin();
		}
		// After 20 spins, spin() returns false
		assert!(!sw.spin());
	}

	#[test]
	fn spinwait_reset_restarts_counter() {
		let mut sw = SpinWait::new();
		// Exhaust budget
		for _ in 0..20 {
			sw.spin();
		}
		assert!(!sw.spin());

		// Reset
		sw.reset();

		// Should return true again
		assert!(sw.spin());
	}

	// -----------------------------------------------------------------------
	// HybridGuard Trait Tests
	// -----------------------------------------------------------------------

	#[test]
	fn hybrid_guard_trait_works_for_optimistic() {
		let latch = HybridLatch::new(42);
		let guard = latch.optimistic_or_spin();

		fn use_guard<T>(guard: &impl HybridGuard<T>) -> &T {
			guard.inner()
		}

		assert_eq!(*use_guard(&guard), 42);
		assert!(guard.recheck().is_ok());
	}

	#[test]
	fn hybrid_guard_trait_works_for_shared() {
		let latch = HybridLatch::new(42);
		let guard = latch.shared();

		fn use_guard<T>(guard: &impl HybridGuard<T>) -> &T {
			guard.inner()
		}

		assert_eq!(*use_guard(&guard), 42);
		// SharedGuard::recheck() is an assertion, not fallible
		guard.recheck();
	}

	#[test]
	fn hybrid_guard_trait_works_for_exclusive() {
		let latch = HybridLatch::new(42);
		let guard = latch.exclusive();

		fn use_guard<T>(guard: &impl HybridGuard<T>) -> &T {
			guard.inner()
		}

		assert_eq!(*use_guard(&guard), 42);
		// ExclusiveGuard::recheck() is an assertion, not fallible
		guard.recheck();
	}

	// -----------------------------------------------------------------------
	// OptimisticOrShared / OptimisticOrExclusive Tests
	// -----------------------------------------------------------------------

	#[test]
	fn optimistic_or_unwind_returns_optimistic_when_unlocked() {
		let latch = HybridLatch::new(42);
		let guard = latch.optimistic_or_unwind();

		match guard {
			OptimisticOrShared::Optimistic(_) => {}
			OptimisticOrShared::Shared(_) => panic!("expected optimistic"),
		}
	}

	#[test]
	fn optimistic_or_exclusive_returns_optimistic_when_unlocked() {
		let latch = HybridLatch::new(42);
		let guard = latch.optimistic_or_exclusive();

		match guard {
			OptimisticOrExclusive::Optimistic(_) => {}
			OptimisticOrExclusive::Exclusive(_) => panic!("expected optimistic"),
		}
	}

	// -----------------------------------------------------------------------
	// Edge Cases
	// -----------------------------------------------------------------------

	#[test]
	fn latch_with_zero_sized_type() {
		let latch = HybridLatch::new(());
		let opt = latch.optimistic_or_spin();
		assert!(opt.recheck().is_ok());
	}

	#[test]
	fn latch_with_complex_type() {
		#[derive(Debug, PartialEq)]
		struct Complex {
			a: i32,
			b: String,
			c: Vec<u8>,
		}

		let latch = HybridLatch::new(Complex {
			a: 42,
			b: "hello".to_string(),
			c: vec![1, 2, 3],
		});

		let opt = latch.optimistic_or_spin();
		assert_eq!(opt.a, 42);
		assert_eq!(opt.b, "hello");
		assert_eq!(opt.c, vec![1, 2, 3]);
		assert!(opt.recheck().is_ok());
	}

	#[test]
	fn as_mut_provides_direct_access() {
		let mut latch = HybridLatch::new(42);
		*latch.as_mut() = 100;

		let opt = latch.optimistic_or_spin();
		assert_eq!(*opt, 100);
	}
}
