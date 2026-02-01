//! Allocation tracking for memory leak detection.
//!
//! This module provides a custom global allocator that tracks allocation
//! counts and bytes allocated. It's designed for use in tests to verify
//! that memory is properly reclaimed.
//!
//! # Usage
//!
//! In test binaries that want to track allocations, use:
//!
//! ```ignore
//! use ferntree::alloc::TrackingAllocator;
//!
//! #[global_allocator]
//! static ALLOC: TrackingAllocator = TrackingAllocator;
//!
//! #[test]
//! fn test_no_leaks() {
//!     ferntree::alloc::reset_counters();
//!     
//!     // ... test code ...
//!     
//!     // Force cleanup
//!     drop(tree);
//!     
//!     ferntree::alloc::check_no_leaks();
//! }
//! ```
//!
//! # Caveats
//!
//! - The tracking allocator adds overhead to every allocation
//! - Counters are global, so tests must run single-threaded to get accurate counts
//! - Some allocations from the test harness itself may be counted

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

/// Global allocation counter - total number of allocations.
pub static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Global deallocation counter - total number of deallocations.
pub static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Current bytes allocated (can be negative due to race conditions in tests).
pub static BYTES_ALLOCATED: AtomicIsize = AtomicIsize::new(0);

/// Peak bytes allocated since last reset.
pub static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// A tracking allocator that counts allocations and deallocations.
///
/// This wraps the system allocator and increments/decrements counters
/// on each allocation operation.
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
		let size = layout.size() as isize;
		let current = BYTES_ALLOCATED.fetch_add(size, Ordering::Relaxed) + size;

		// Update peak (not atomic with the add, but close enough for testing)
		let peak = PEAK_BYTES.load(Ordering::Relaxed);
		if current as usize > peak {
			PEAK_BYTES.store(current as usize, Ordering::Relaxed);
		}

		System.alloc(layout)
	}

	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		DEALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
		BYTES_ALLOCATED.fetch_sub(layout.size() as isize, Ordering::Relaxed);
		System.dealloc(ptr, layout)
	}

	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
		let size = layout.size() as isize;
		let current = BYTES_ALLOCATED.fetch_add(size, Ordering::Relaxed) + size;

		let peak = PEAK_BYTES.load(Ordering::Relaxed);
		if current as usize > peak {
			PEAK_BYTES.store(current as usize, Ordering::Relaxed);
		}

		System.alloc_zeroed(layout)
	}

	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		let old_size = layout.size() as isize;
		let diff = new_size as isize - old_size;
		BYTES_ALLOCATED.fetch_add(diff, Ordering::Relaxed);

		if diff > 0 {
			let current = BYTES_ALLOCATED.load(Ordering::Relaxed);
			let peak = PEAK_BYTES.load(Ordering::Relaxed);
			if current as usize > peak {
				PEAK_BYTES.store(current as usize, Ordering::Relaxed);
			}
		}

		System.realloc(ptr, layout, new_size)
	}
}

/// Resets all allocation counters to zero.
///
/// Call this at the beginning of a test to start fresh counting.
pub fn reset_counters() {
	ALLOC_COUNT.store(0, Ordering::SeqCst);
	DEALLOC_COUNT.store(0, Ordering::SeqCst);
	BYTES_ALLOCATED.store(0, Ordering::SeqCst);
	PEAK_BYTES.store(0, Ordering::SeqCst);
}

/// Returns the current allocation statistics.
pub fn get_stats() -> AllocationStats {
	AllocationStats {
		alloc_count: ALLOC_COUNT.load(Ordering::SeqCst),
		dealloc_count: DEALLOC_COUNT.load(Ordering::SeqCst),
		bytes_allocated: BYTES_ALLOCATED.load(Ordering::SeqCst),
		peak_bytes: PEAK_BYTES.load(Ordering::SeqCst),
	}
}

/// Allocation statistics snapshot.
#[derive(Debug, Clone, Copy)]
pub struct AllocationStats {
	/// Total number of allocations since reset.
	pub alloc_count: usize,
	/// Total number of deallocations since reset.
	pub dealloc_count: usize,
	/// Current bytes allocated (may be negative due to races).
	pub bytes_allocated: isize,
	/// Peak bytes allocated since reset.
	pub peak_bytes: usize,
}

/// Checks that no memory has been leaked.
///
/// This asserts that the number of allocations equals the number of
/// deallocations and that no bytes are currently allocated.
///
/// # Panics
///
/// Panics if there are unmatched allocations or bytes still allocated.
///
/// # Note
///
/// Due to deferred reclamation in crossbeam_epoch, you may need to
/// call `crossbeam_epoch::pin()` multiple times before calling this
/// to trigger garbage collection.
pub fn check_no_leaks() {
	let stats = get_stats();

	// Allow small discrepancy due to test harness allocations
	// that may have happened before reset_counters() was called
	let diff = stats.alloc_count as isize - stats.dealloc_count as isize;

	if diff != 0 {
		panic!(
			"Memory leak detected!\n\
             Allocations: {}\n\
             Deallocations: {}\n\
             Difference: {}\n\
             Bytes still allocated: {}",
			stats.alloc_count, stats.dealloc_count, diff, stats.bytes_allocated
		);
	}

	if stats.bytes_allocated != 0 {
		panic!(
			"Memory leak detected!\n\
             Bytes still allocated: {}\n\
             (alloc_count == dealloc_count but bytes != 0, possible size mismatch)",
			stats.bytes_allocated
		);
	}
}

/// Checks that allocation counts are balanced, allowing for a tolerance.
///
/// This is useful when tests cannot guarantee exact cleanup due to
/// test harness overhead or timing issues.
pub fn check_balanced_with_tolerance(tolerance: usize) {
	let stats = get_stats();
	let diff = (stats.alloc_count as isize - stats.dealloc_count as isize).unsigned_abs();

	if diff > tolerance {
		panic!(
			"Memory leak detected (beyond tolerance of {})!\n\
             Allocations: {}\n\
             Deallocations: {}\n\
             Difference: {}",
			tolerance, stats.alloc_count, stats.dealloc_count, diff
		);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	// Note: These tests require TrackingAllocator to be set as the global allocator.
	// They are ignored by default because the test harness uses the system allocator.
	// To run these tests, create a separate test binary with:
	//   #[global_allocator]
	//   static ALLOC: TrackingAllocator = TrackingAllocator;

	#[test]
	#[ignore = "requires TrackingAllocator to be set as global allocator"]
	fn test_tracking_allocator_basic() {
		reset_counters();

		// Make some allocations
		let v1: Vec<u8> = vec![1, 2, 3, 4];
		let v2: Vec<u8> = vec![5, 6, 7, 8];

		let stats = get_stats();
		assert!(stats.alloc_count > 0);
		assert!(stats.bytes_allocated > 0);

		// Drop them
		drop(v1);
		drop(v2);

		// Stats should show balanced alloc/dealloc
		let stats = get_stats();
		assert_eq!(stats.alloc_count, stats.dealloc_count);
	}

	#[test]
	#[ignore = "requires TrackingAllocator to be set as global allocator"]
	fn test_reset_counters() {
		// Make some allocations
		let _v: Vec<u8> = vec![1, 2, 3, 4];

		reset_counters();

		let stats = get_stats();
		assert_eq!(stats.alloc_count, 0);
		assert_eq!(stats.dealloc_count, 0);
	}
}
