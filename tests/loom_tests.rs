//! Loom-based concurrency tests for ferntree.
//!
//! These tests use loom to systematically explore all possible thread
//! interleavings, catching race conditions and deadlocks that might not
//! manifest in regular concurrent testing.
//!
//! # Running Loom Tests
//!
//! Loom tests must be run with the `loom` cfg flag:
//!
//! ```bash
//! RUSTFLAGS="--cfg loom" cargo test -p ferntree --features loom --release -- --test-threads=1
//! ```
//!
//! # Test Design
//!
//! Loom tests should be kept small (2-4 threads, few operations) because
//! the number of possible interleavings grows exponentially. Focus on
//! testing critical sections and edge cases.
//!
//! # Limitations
//!
//! The ferntree uses optimistic lock coupling with spin-retry loops.
//! This creates unbounded interleavings that loom cannot exhaustively explore.
//! Therefore, full tree operation tests are not feasible with loom.
//! Instead, we focus on testing the underlying HybridLatch primitive which
//! has bounded behavior under loom.

#![cfg(loom)]

use loom::sync::Arc;
use loom::thread;

// ===========================================================================
// HybridLatch Tests Under Loom
// ===========================================================================
//
// These tests focus on the HybridLatch primitive which is the foundation
// of ferntree's concurrency. The latch operations have bounded behavior
// that loom can effectively model.

/// Test HybridLatch shared read with concurrent write.
/// Shared readers should block writers and vice versa.
#[test]
fn loom_latch_shared_blocking() {
	use ferntree::latch::HybridLatch;

	loom::model(|| {
		let latch = Arc::new(HybridLatch::new(0i32));

		// Start with shared access
		let t1 = {
			let latch = Arc::clone(&latch);
			thread::spawn(move || {
				let guard = latch.shared();
				let val = *guard;
				// Hold the guard briefly
				drop(guard);
				val
			})
		};

		let t2 = {
			let latch = Arc::clone(&latch);
			thread::spawn(move || {
				let mut guard = latch.exclusive();
				*guard = 42;
			})
		};

		let v1 = t1.join().unwrap();
		t2.join().unwrap();

		// Reader should see either 0 (read before write) or 42 (read after write)
		assert!(v1 == 0 || v1 == 42);

		// Final value should be 42
		let final_val = *latch.shared();
		assert_eq!(final_val, 42);
	});
}

/// Test multiple exclusive accesses are properly serialized.
#[test]
fn loom_latch_exclusive_serialization() {
	use ferntree::latch::HybridLatch;

	loom::model(|| {
		let latch = Arc::new(HybridLatch::new(0i32));

		let t1 = {
			let latch = Arc::clone(&latch);
			thread::spawn(move || {
				let mut guard = latch.exclusive();
				*guard += 1;
			})
		};

		let t2 = {
			let latch = Arc::clone(&latch);
			thread::spawn(move || {
				let mut guard = latch.exclusive();
				*guard += 1;
			})
		};

		t1.join().unwrap();
		t2.join().unwrap();

		// Both increments should be visible
		let final_val = *latch.shared();
		assert_eq!(final_val, 2);
	});
}

/// Test multiple shared readers can access concurrently.
#[test]
fn loom_latch_concurrent_shared() {
	use ferntree::latch::HybridLatch;

	loom::model(|| {
		let latch = Arc::new(HybridLatch::new(42i32));

		let t1 = {
			let latch = Arc::clone(&latch);
			thread::spawn(move || {
				let guard = latch.shared();
				*guard
			})
		};

		let t2 = {
			let latch = Arc::clone(&latch);
			thread::spawn(move || {
				let guard = latch.shared();
				*guard
			})
		};

		let v1 = t1.join().unwrap();
		let v2 = t2.join().unwrap();

		// Both readers should see the same value
		assert_eq!(v1, 42);
		assert_eq!(v2, 42);
	});
}

/// Test that exclusive access is truly exclusive.
#[test]
fn loom_latch_exclusive_mutual_exclusion() {
	use ferntree::latch::HybridLatch;
	use loom::sync::atomic::{AtomicUsize, Ordering};

	loom::model(|| {
		let latch = Arc::new(HybridLatch::new(0i32));
		let counter = Arc::new(AtomicUsize::new(0));

		let t1 = {
			let latch = Arc::clone(&latch);
			let counter = Arc::clone(&counter);
			thread::spawn(move || {
				let mut guard = latch.exclusive();
				// Increment counter while holding exclusive lock
				let prev = counter.fetch_add(1, Ordering::SeqCst);
				// No other thread should have incremented while we hold the lock
				assert_eq!(prev, 0);
				*guard = 1;
				counter.fetch_sub(1, Ordering::SeqCst);
			})
		};

		let t2 = {
			let latch = Arc::clone(&latch);
			let counter = Arc::clone(&counter);
			thread::spawn(move || {
				let mut guard = latch.exclusive();
				let prev = counter.fetch_add(1, Ordering::SeqCst);
				assert_eq!(prev, 0);
				*guard = 2;
				counter.fetch_sub(1, Ordering::SeqCst);
			})
		};

		t1.join().unwrap();
		t2.join().unwrap();

		// Final value should be from one of the writers
		let final_val = *latch.shared();
		assert!(final_val == 1 || final_val == 2);
	});
}

/// Test shared followed by exclusive.
#[test]
fn loom_latch_shared_then_exclusive() {
	use ferntree::latch::HybridLatch;

	loom::model(|| {
		let latch = Arc::new(HybridLatch::new(0i32));

		// First take shared
		{
			let guard = latch.shared();
			assert_eq!(*guard, 0);
		}

		// Then take exclusive
		{
			let mut guard = latch.exclusive();
			*guard = 42;
		}

		// Verify the write
		assert_eq!(*latch.shared(), 42);
	});
}

/// Test exclusive followed by shared.
#[test]
fn loom_latch_exclusive_then_shared() {
	use ferntree::latch::HybridLatch;

	loom::model(|| {
		let latch = Arc::new(HybridLatch::new(0i32));

		// First take exclusive
		{
			let mut guard = latch.exclusive();
			*guard = 42;
		}

		// Then take shared
		{
			let guard = latch.shared();
			assert_eq!(*guard, 42);
		}
	});
}
