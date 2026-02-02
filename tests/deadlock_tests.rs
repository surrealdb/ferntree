//! # Deadlock, Timeout, and Starvation Tests for Ferntree B+ Tree
//!
//! This module contains tests specifically designed to detect:
//! - Deadlocks in concurrent tree operations
//! - Same-thread mutation conflicts with active iterators
//! - Iterator behavior during concurrent deletions
//! - Writer/reader starvation under contention
//!
//! ## Test Strategy
//!
//! Since loom cannot handle the unbounded interleavings from optimistic lock
//! coupling, these tests use timeout-based detection. If operations don't
//! complete within expected time, the test fails (indicating potential deadlock).
//!
//! ## Running Tests
//!
//! ```bash
//! cargo test -p ferntree deadlock_tests
//! ```
//!
//! For longer stress tests:
//! ```bash
//! cargo test -p ferntree deadlock_tests -- --ignored
//! ```

use ferntree::Tree;
use rand::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// ===========================================================================
// Timeout Helper
// ===========================================================================

/// Runs a closure with a timeout, panicking if the operation doesn't complete
/// within the specified duration.
///
/// This is the primary mechanism for detecting deadlocks in tests. If a test
/// hangs due to a deadlock, the timeout will trigger and fail the test with
/// a descriptive message.
///
/// # Arguments
///
/// * `timeout` - Maximum duration to wait for the operation
/// * `name` - Descriptive name for the operation (used in panic message)
/// * `f` - The closure to execute
///
/// # Panics
///
/// Panics if the operation doesn't complete within the timeout, or if the
/// spawned thread panics.
fn run_with_timeout<F, R>(timeout: Duration, name: &str, f: F) -> R
where
	F: FnOnce() -> R + Send + 'static,
	R: Send + 'static,
{
	let (tx, rx) = channel();
	let name = name.to_string();

	let handle = thread::spawn(move || {
		let result = f();
		let _ = tx.send(result);
	});

	match rx.recv_timeout(timeout) {
		Ok(result) => {
			// Join the thread to ensure clean shutdown
			handle.join().expect("Thread panicked");
			result
		}
		Err(RecvTimeoutError::Timeout) => {
			panic!(
				"TIMEOUT: '{}' did not complete within {:?} - potential deadlock detected",
				name, timeout
			);
		}
		Err(RecvTimeoutError::Disconnected) => {
			// Thread terminated without sending - likely panicked
			handle.join().expect("Thread panicked without sending result");
			panic!("Thread terminated unexpectedly without completing");
		}
	}
}

/// Runs multiple operations concurrently with a shared timeout.
///
/// All operations must complete within the timeout for the test to pass.
/// This is useful for testing concurrent scenarios where we want to ensure
/// no combination of operations causes a deadlock.
#[allow(dead_code)]
fn run_concurrent_with_timeout<F>(timeout: Duration, name: &str, num_threads: usize, f: F)
where
	F: Fn(usize) + Send + Sync + 'static,
{
	let f = Arc::new(f);
	let (tx, rx) = channel();
	let name = name.to_string();

	let handles: Vec<_> = (0..num_threads)
		.map(|i| {
			let f = Arc::clone(&f);
			let tx = tx.clone();
			thread::spawn(move || {
				f(i);
				let _ = tx.send(i);
			})
		})
		.collect();

	// Drop our sender so rx knows when all threads are done
	drop(tx);

	let start = Instant::now();
	let mut completed = 0;

	while completed < num_threads {
		let remaining = timeout.saturating_sub(start.elapsed());
		if remaining.is_zero() {
			panic!(
				"TIMEOUT: '{}' - only {}/{} threads completed within {:?} - potential deadlock",
				name, completed, num_threads, timeout
			);
		}

		match rx.recv_timeout(remaining) {
			Ok(_) => completed += 1,
			Err(RecvTimeoutError::Timeout) => {
				panic!(
					"TIMEOUT: '{}' - only {}/{} threads completed within {:?} - potential deadlock",
					name, completed, num_threads, timeout
				);
			}
			Err(RecvTimeoutError::Disconnected) => {
				// All senders dropped - check if we got all results
				break;
			}
		}
	}

	// Join all threads
	for handle in handles {
		handle.join().expect("Thread panicked");
	}
}

// ===========================================================================
// Full Tree Deadlock Tests
// ===========================================================================

/// Test that multiple threads acquiring exclusive locks in different key orders
/// don't deadlock.
///
/// This tests the lock acquisition order when multiple writers target
/// overlapping key ranges.
#[test]
fn deadlock_multiple_writers_different_orders() {
	run_with_timeout(Duration::from_secs(10), "multiple_writers_different_orders", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate the tree
		for i in 0..100 {
			tree.insert(i, i);
		}

		let num_threads = 4;
		let iterations = 100;

		let handles: Vec<_> = (0..num_threads)
			.map(|t| {
				let tree = Arc::clone(&tree);
				thread::spawn(move || {
					let mut rng = rand::rng();
					for _ in 0..iterations {
						// Each thread inserts in a random order
						let keys: Vec<i32> = (0..10).map(|_| rng.random_range(0..100)).collect();
						for key in keys {
							tree.insert(key, t);
						}
					}
				})
			})
			.collect();

		for h in handles {
			h.join().unwrap();
		}

		tree.assert_invariants();
	});
}

/// Test reader-writer interleaving at high contention doesn't deadlock.
///
/// Multiple readers and writers competing for the same key range should
/// make progress without getting stuck.
#[test]
fn deadlock_reader_writer_interleaving() {
	run_with_timeout(Duration::from_secs(10), "reader_writer_interleaving", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate
		for i in 0..100 {
			tree.insert(i, i);
		}

		let running = Arc::new(AtomicBool::new(true));
		let num_readers = 4;
		let num_writers = 2;

		// Spawn readers that continuously iterate
		let reader_handles: Vec<_> = (0..num_readers)
			.map(|_| {
				let tree = Arc::clone(&tree);
				let running = Arc::clone(&running);
				thread::spawn(move || {
					let mut count = 0u64;
					while running.load(Ordering::Relaxed) {
						let mut iter = tree.raw_iter();
						iter.seek_to_first();
						while iter.next().is_some() {
							count += 1;
						}
					}
					count
				})
			})
			.collect();

		// Spawn writers that continuously modify
		let writer_handles: Vec<_> = (0..num_writers)
			.map(|t| {
				let tree = Arc::clone(&tree);
				let running = Arc::clone(&running);
				thread::spawn(move || {
					let mut rng = rand::rng();
					let mut count = 0u64;
					while running.load(Ordering::Relaxed) {
						let key: i32 = rng.random_range(0..100);
						if rng.random_bool(0.5) {
							tree.insert(key, t);
						} else {
							tree.remove(&key);
						}
						count += 1;
					}
					count
				})
			})
			.collect();

		// Let them run for a bit
		thread::sleep(Duration::from_millis(500));
		running.store(false, Ordering::Relaxed);

		// Collect results
		let reader_ops: u64 = reader_handles.into_iter().map(|h| h.join().unwrap()).sum();
		let writer_ops: u64 = writer_handles.into_iter().map(|h| h.join().unwrap()).sum();

		// Both readers and writers should have made progress
		assert!(reader_ops > 0, "Readers made no progress");
		assert!(writer_ops > 0, "Writers made no progress");

		tree.assert_invariants();
	});
}

/// Test concurrent splits don't cause deadlocks.
///
/// When multiple threads cause splits simultaneously, they need to acquire
/// parent locks. This tests that the lock acquisition doesn't deadlock.
#[test]
fn deadlock_concurrent_splits() {
	run_with_timeout(Duration::from_secs(15), "concurrent_splits", || {
		let tree = Arc::new(Tree::<i32, i32>::new());
		let num_threads = 8;
		let entries_per_thread = 500;

		let handles: Vec<_> = (0..num_threads)
			.map(|t| {
				let tree = Arc::clone(&tree);
				thread::spawn(move || {
					// Each thread inserts sequential keys to trigger splits
					for i in 0..entries_per_thread {
						let key = t * entries_per_thread + i;
						tree.insert(key, key * 10);
					}
				})
			})
			.collect();

		for h in handles {
			h.join().unwrap();
		}

		tree.assert_invariants();
		assert_eq!(tree.len(), (num_threads * entries_per_thread) as usize);
		assert!(tree.height() > 1, "Tree should have split");
	});
}

/// Test concurrent merges during deletions don't deadlock.
///
/// When multiple threads delete entries causing merges, they need to
/// coordinate with siblings and parents.
#[test]
fn deadlock_concurrent_merges() {
	run_with_timeout(Duration::from_secs(15), "concurrent_merges", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Build a large tree
		for i in 0..2000 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		let num_threads = 4;
		let entries_per_thread = 500;

		let handles: Vec<_> = (0..num_threads)
			.map(|t| {
				let tree = Arc::clone(&tree);
				thread::spawn(move || {
					// Each thread removes its range
					for i in 0..entries_per_thread {
						let key = t * entries_per_thread + i;
						tree.remove(&key);
					}
				})
			})
			.collect();

		for h in handles {
			h.join().unwrap();
		}

		tree.assert_invariants();
		assert!(tree.is_empty(), "All entries should be removed");
	});
}

/// Test interleaved splits and merges don't deadlock.
///
/// This creates a scenario where some threads are causing splits while
/// others are causing merges, testing the full range of structural changes.
#[test]
fn deadlock_interleaved_splits_and_merges() {
	run_with_timeout(Duration::from_secs(20), "interleaved_splits_and_merges", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate with some entries
		for i in 0..500 {
			tree.insert(i, i);
		}

		let running = Arc::new(AtomicBool::new(true));
		let num_threads = 6;

		let handles: Vec<_> = (0..num_threads)
			.map(|t| {
				let tree = Arc::clone(&tree);
				let running = Arc::clone(&running);
				thread::spawn(move || {
					let mut rng = rand::rng();
					let mut ops = 0u64;

					while running.load(Ordering::Relaxed) {
						let key: i32 = rng.random_range(0..1000);

						// Mix of inserts (causing splits) and removes (causing merges)
						if t % 2 == 0 {
							// Predominantly insert
							if rng.random_bool(0.7) {
								tree.insert(key, t);
							} else {
								tree.remove(&key);
							}
						} else {
							// Predominantly remove
							if rng.random_bool(0.7) {
								tree.remove(&key);
							} else {
								tree.insert(key, t);
							}
						}
						ops += 1;
					}
					ops
				})
			})
			.collect();

		// Let it run
		thread::sleep(Duration::from_millis(1000));
		running.store(false, Ordering::Relaxed);

		let total_ops: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();

		tree.assert_invariants();
		assert!(total_ops > 100, "Should have performed many operations");
	});
}

// ===========================================================================
// Same-Thread Mutation Outside Iterator Tests
// ===========================================================================

/// Test that inserting to a different leaf while holding a shared iterator
/// succeeds (different leaves use different locks).
#[test]
fn same_thread_insert_different_leaf() {
	run_with_timeout(Duration::from_secs(5), "insert_different_leaf", || {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to create multiple leaves
		for i in 0..200 {
			tree.insert(i, i);
		}

		tree.assert_invariants();
		assert!(tree.height() > 1, "Need multiple leaves for this test");

		// Position iterator at the first leaf (keys near 0)
		let mut iter = tree.raw_iter();
		iter.seek(&0);

		// Read from the iterator (holds shared lock on first leaf)
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 0);

		// Insert to a key that's likely in a different leaf (high keys)
		// This should succeed because it's a different leaf
		tree.insert(1000, 1000);

		// Iterator should still work
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 1);

		tree.assert_invariants();
	});
}

/// Test that removing from a different leaf while holding a shared iterator
/// succeeds.
#[test]
fn same_thread_remove_different_leaf() {
	run_with_timeout(Duration::from_secs(5), "remove_different_leaf", || {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to create multiple leaves
		for i in 0..200 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		// Also insert a key far away
		tree.insert(1000, 1000);

		// Position iterator at the first leaf
		let mut iter = tree.raw_iter();
		iter.seek(&0);

		// Read from the iterator
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 0);

		// Remove a key from a different leaf
		let removed = tree.remove(&1000);
		assert_eq!(removed, Some(1000));

		// Iterator should still work
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 1);

		tree.assert_invariants();
	});
}

/// Test behavior when iterator moves through leaves while modifications happen.
///
/// This documents the expected behavior: as the iterator moves to new leaves,
/// it releases locks on old leaves, allowing modifications to those leaves.
#[test]
fn same_thread_iterator_releases_locks_on_leaf_change() {
	run_with_timeout(Duration::from_secs(5), "iterator_releases_locks", || {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to create multiple leaves
		for i in 0..200 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		// Iterate through all entries to move through all leaves
		let mut count = 0;
		while iter.next().is_some() {
			count += 1;
		}
		assert_eq!(count, 200);

		// After iteration is complete (or iterator moved past first leaf),
		// we should be able to modify any leaf
		tree.insert(0, 999);
		assert_eq!(tree.lookup(&0, |v| *v), Some(999));

		tree.assert_invariants();
	});
}

/// Test exclusive iterator blocks all other access to its current leaf.
///
/// This test documents the behavior: exclusive iterator holds exclusive lock,
/// so same-thread modifications to different leaves should work.
#[test]
fn same_thread_exclusive_iter_different_leaf() {
	run_with_timeout(Duration::from_secs(5), "exclusive_iter_different_leaf", || {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to create multiple leaves
		for i in 0..200 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		// Position exclusive iterator at first leaf
		let mut iter = tree.raw_iter_mut();
		iter.seek(&0);

		// Modify through the iterator
		if let Some((_, v)) = iter.next() {
			*v = 999;
		}

		// After iterator moves or is dropped, modifications should work
		drop(iter);

		// Now we can modify anywhere
		tree.insert(1000, 1000);
		tree.assert_invariants();
	});
}

// ===========================================================================
// Concurrent Delete + Iterate Tests
// ===========================================================================

/// Test forward iteration while another thread removes entries ahead.
///
/// The iterator should handle concurrent structural changes gracefully,
/// either seeing the entry or not, but never crashing or returning garbage.
#[test]
fn concurrent_delete_ahead_of_iterator() {
	run_with_timeout(Duration::from_secs(10), "delete_ahead_of_iterator", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate
		for i in 0..500 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		let tree_reader = Arc::clone(&tree);
		let tree_deleter = Arc::clone(&tree);

		let reader = thread::spawn(move || {
			let mut iter = tree_reader.raw_iter();
			iter.seek_to_first();

			let mut prev = -1i32;
			let mut count = 0;

			while let Some((k, v)) = iter.next() {
				// Keys should still be in sorted order
				assert!(*k > prev, "Order violation: {} not > {} at count {}", k, prev, count);
				// Value should match key (we never modified values)
				assert_eq!(*k, *v, "Value mismatch at key {}", k);
				prev = *k;
				count += 1;
			}

			count
		});

		let deleter = thread::spawn(move || {
			// Delete entries in the upper range while reader iterates
			for i in (250..500).rev() {
				tree_deleter.remove(&i);
			}
		});

		deleter.join().unwrap();
		let count = reader.join().unwrap();

		// Reader should have seen some entries (at least the ones before deletes started)
		assert!(count > 0, "Reader saw no entries");

		tree.assert_invariants();
	});
}

/// Test forward iteration while another thread removes entries behind cursor.
///
/// Deletions behind the cursor shouldn't affect the iterator's forward progress.
#[test]
fn concurrent_delete_behind_iterator() {
	run_with_timeout(Duration::from_secs(10), "delete_behind_iterator", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate
		for i in 0..500 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		let tree_reader = Arc::clone(&tree);
		let tree_deleter = Arc::clone(&tree);

		let reader = thread::spawn(move || {
			let mut iter = tree_reader.raw_iter();
			iter.seek(&250); // Start in the middle

			let mut prev = 249i32;
			let mut count = 0;

			while let Some((k, _)) = iter.next() {
				assert!(*k > prev, "Order violation");
				prev = *k;
				count += 1;
			}

			count
		});

		let deleter = thread::spawn(move || {
			// Delete entries in the lower range while reader iterates forward
			for i in 0..250 {
				tree_deleter.remove(&i);
			}
		});

		deleter.join().unwrap();
		let count = reader.join().unwrap();

		// Reader should have seen the upper half
		assert!(count > 0, "Reader saw no entries");

		tree.assert_invariants();
	});
}

/// Test reverse iteration with concurrent removes.
#[test]
fn concurrent_delete_during_reverse_iteration() {
	run_with_timeout(Duration::from_secs(10), "delete_during_reverse_iteration", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate
		for i in 0..500 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		let tree_reader = Arc::clone(&tree);
		let tree_deleter = Arc::clone(&tree);

		let reader = thread::spawn(move || {
			let mut iter = tree_reader.raw_iter();
			iter.seek_to_last();

			let mut prev = 500i32;
			let mut count = 0;

			while let Some((k, _)) = iter.prev() {
				assert!(*k < prev, "Reverse order violation: {} not < {}", k, prev);
				prev = *k;
				count += 1;
			}

			count
		});

		let deleter = thread::spawn(move || {
			let mut rng = rand::rng();
			// Randomly delete entries
			for _ in 0..200 {
				let key: i32 = rng.random_range(0..500);
				tree_deleter.remove(&key);
			}
		});

		deleter.join().unwrap();
		let count = reader.join().unwrap();

		assert!(count > 0, "Reader saw no entries");

		tree.assert_invariants();
	});
}

/// Test iterator recovery after concurrent structural changes.
///
/// This tests the anchor-based recovery mechanism when the tree structure
/// changes during iteration.
#[test]
fn concurrent_structural_changes_during_iteration() {
	run_with_timeout(Duration::from_secs(15), "structural_changes_during_iteration", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate
		for i in 0..200 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		let running = Arc::new(AtomicBool::new(true));
		let tree_reader = Arc::clone(&tree);
		let tree_modifier = Arc::clone(&tree);
		let running_reader = Arc::clone(&running);
		let running_modifier = Arc::clone(&running);

		// Reader that continuously iterates
		let reader = thread::spawn(move || {
			let mut iterations = 0u64;

			while running_reader.load(Ordering::Relaxed) {
				let mut iter = tree_reader.raw_iter();
				iter.seek_to_first();

				let mut prev = -1i32;
				while let Some((k, _)) = iter.next() {
					// Verify sorted order is maintained despite concurrent changes
					assert!(*k > prev, "Order violation during iteration {}", iterations);
					prev = *k;
				}
				iterations += 1;
			}

			iterations
		});

		// Modifier that causes splits and merges
		let modifier = thread::spawn(move || {
			let mut rng = rand::rng();
			let mut ops = 0u64;

			while running_modifier.load(Ordering::Relaxed) {
				let key: i32 = rng.random_range(0..1000);
				if rng.random_bool(0.5) {
					tree_modifier.insert(key, key);
				} else {
					tree_modifier.remove(&key);
				}
				ops += 1;
			}

			ops
		});

		// Let them run
		thread::sleep(Duration::from_millis(1000));
		running.store(false, Ordering::Relaxed);

		let read_ops = reader.join().unwrap();
		let modify_ops = modifier.join().unwrap();

		assert!(read_ops > 0, "Reader made no progress");
		assert!(modify_ops > 0, "Modifier made no progress");

		tree.assert_invariants();
	});
}

/// Test that iteration maintains sorted order despite concurrent modifications.
#[test]
fn concurrent_iteration_maintains_sorted_order() {
	run_with_timeout(Duration::from_secs(10), "iteration_maintains_sorted_order", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate
		for i in 0..200 {
			tree.insert(i, i);
		}

		let tree_reader = Arc::clone(&tree);
		let tree_writer = Arc::clone(&tree);

		let reader = thread::spawn(move || {
			for _ in 0..10 {
				let mut iter = tree_reader.raw_iter();
				iter.seek_to_first();

				let mut keys = Vec::new();
				while let Some((k, _)) = iter.next() {
					keys.push(*k);
				}

				// Verify sorted
				for window in keys.windows(2) {
					assert!(
						window[0] < window[1],
						"Order violation: {} >= {}",
						window[0],
						window[1]
					);
				}
			}
		});

		let writer = thread::spawn(move || {
			let mut rng = rand::rng();
			for _ in 0..100 {
				let key: i32 = rng.random_range(0..500);
				if rng.random_bool(0.5) {
					tree_writer.insert(key, key);
				} else {
					tree_writer.remove(&key);
				}
			}
		});

		reader.join().unwrap();
		writer.join().unwrap();

		tree.assert_invariants();
	});
}

// ===========================================================================
// Timeout-Based Deadlock Detection for Stress Tests
// ===========================================================================

/// Stress test with timeout: concurrent mixed operations under high contention.
#[test]
fn stress_timeout_concurrent_mixed_high_contention() {
	run_with_timeout(Duration::from_secs(30), "stress_mixed_high_contention", || {
		let tree = Arc::new(Tree::<i32, i32>::new());
		let num_threads = 8;
		let key_range = 100; // Very small range = high contention
		let ops_per_thread = 2000;

		// Pre-populate
		for i in 0..key_range {
			tree.insert(i, i);
		}

		let handles: Vec<_> = (0..num_threads)
			.map(|t| {
				let tree = Arc::clone(&tree);
				thread::spawn(move || {
					let mut rng = rand::rng();
					for _ in 0..ops_per_thread {
						let key: i32 = rng.random_range(0..key_range);
						match rng.random_range(0..4) {
							0 => {
								tree.insert(key, t);
							}
							1 => {
								tree.remove(&key);
							}
							2 => {
								tree.lookup(&key, |v| *v);
							}
							3 => {
								// Full scan
								let mut iter = tree.raw_iter();
								iter.seek_to_first();
								while iter.next().is_some() {}
							}
							_ => unreachable!(),
						}
					}
				})
			})
			.collect();

		for h in handles {
			h.join().unwrap();
		}

		tree.assert_invariants();
	});
}

/// Stress test with timeout: single key extreme contention.
#[test]
fn stress_timeout_single_key_contention() {
	run_with_timeout(Duration::from_secs(15), "stress_single_key_contention", || {
		let tree = Arc::new(Tree::<i32, i32>::new());
		let num_threads = 8;
		let iterations = 5000;

		tree.insert(42, 0);

		let handles: Vec<_> = (0..num_threads)
			.map(|t| {
				let tree = Arc::clone(&tree);
				thread::spawn(move || {
					for i in 0..iterations {
						match i % 3 {
							0 => {
								tree.insert(42, t);
							}
							1 => {
								tree.lookup(&42, |v| *v);
							}
							2 => {
								tree.remove(&42);
								tree.insert(42, t);
							}
							_ => unreachable!(),
						}
					}
				})
			})
			.collect();

		for h in handles {
			h.join().unwrap();
		}

		tree.assert_invariants();
	});
}

/// Stress test with timeout: rapid split/merge cycles.
#[test]
fn stress_timeout_rapid_split_merge_cycles() {
	run_with_timeout(Duration::from_secs(30), "stress_rapid_split_merge", || {
		let tree = Arc::new(Tree::<i32, i32>::new());
		let num_threads = 4;
		let cycles = 10;

		for cycle in 0..cycles {
			let handles: Vec<_> = (0..num_threads)
				.map(|t| {
					let tree = Arc::clone(&tree);
					thread::spawn(move || {
						let base = t * 200;
						// Insert phase (causes splits)
						for i in 0..200 {
							tree.insert(base + i, cycle);
						}
					})
				})
				.collect();

			for h in handles {
				h.join().unwrap();
			}

			// Delete phase (causes merges)
			let handles: Vec<_> = (0..num_threads)
				.map(|t| {
					let tree = Arc::clone(&tree);
					thread::spawn(move || {
						let base = t * 200;
						for i in 0..200 {
							tree.remove(&(base + i));
						}
					})
				})
				.collect();

			for h in handles {
				h.join().unwrap();
			}

			tree.assert_invariants();
		}
	});
}

// ===========================================================================
// Starvation Tests
// ===========================================================================

/// Test that writers don't starve under heavy read load.
///
/// This test spawns many reader threads that continuously iterate, and one
/// writer thread that tries to complete insertions. The writer should be able
/// to make progress within a reasonable time.
#[test]
fn starvation_writer_under_heavy_reads() {
	run_with_timeout(Duration::from_secs(30), "writer_starvation", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate with data for readers to iterate over
		for i in 0..500 {
			tree.insert(i, i);
		}

		let num_readers = 8;
		let writer_target_ops = 100;
		let running = Arc::new(AtomicBool::new(true));
		let writer_ops = Arc::new(AtomicU64::new(0));
		let reader_ops = Arc::new(AtomicU64::new(0));

		// Spawn readers
		let reader_handles: Vec<_> = (0..num_readers)
			.map(|_| {
				let tree = Arc::clone(&tree);
				let running = Arc::clone(&running);
				let ops = Arc::clone(&reader_ops);
				thread::spawn(move || {
					while running.load(Ordering::Relaxed) {
						let mut iter = tree.raw_iter();
						iter.seek_to_first();
						while iter.next().is_some() {
							ops.fetch_add(1, Ordering::Relaxed);
						}
					}
				})
			})
			.collect();

		// Give readers a head start
		thread::sleep(Duration::from_millis(50));

		// Spawn writer and measure time to complete
		let tree_writer = Arc::clone(&tree);
		let writer_ops_clone = Arc::clone(&writer_ops);
		let start = Instant::now();

		let writer = thread::spawn(move || {
			for i in 0..writer_target_ops {
				let key = 1000 + i;
				tree_writer.insert(key, i);
				writer_ops_clone.fetch_add(1, Ordering::Relaxed);
			}
		});

		writer.join().unwrap();
		let write_duration = start.elapsed();

		// Stop readers
		running.store(false, Ordering::Relaxed);
		for h in reader_handles {
			h.join().unwrap();
		}

		let final_writer_ops = writer_ops.load(Ordering::Relaxed);
		let final_reader_ops = reader_ops.load(Ordering::Relaxed);

		// Writer should have completed all operations
		assert_eq!(
			final_writer_ops, writer_target_ops as u64,
			"Writer didn't complete all operations"
		);

		// Writer shouldn't take too long (allow generous time due to contention)
		assert!(
			write_duration < Duration::from_secs(10),
			"Writer took too long ({:?}) - possible starvation",
			write_duration
		);

		// Readers should have made progress too
		assert!(final_reader_ops > 0, "Readers made no progress");

		tree.assert_invariants();

		// Verify writer's inserts are present
		for i in 0..writer_target_ops {
			let key = 1000 + i;
			assert!(tree.contains_key(&key), "Writer's key {} not found", key);
		}
	});
}

/// Test that readers don't starve under heavy write load.
///
/// This test spawns many writer threads doing rapid modifications, and one
/// reader thread that tries to complete full iterations. The reader should
/// be able to make progress.
#[test]
fn starvation_reader_under_heavy_writes() {
	run_with_timeout(Duration::from_secs(30), "reader_starvation", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate
		for i in 0..200 {
			tree.insert(i, i);
		}

		let num_writers = 8;
		let reader_target_iterations = 10;
		let running = Arc::new(AtomicBool::new(true));
		let reader_iterations = Arc::new(AtomicU64::new(0));
		let writer_ops = Arc::new(AtomicU64::new(0));

		// Spawn writers
		let writer_handles: Vec<_> = (0..num_writers)
			.map(|t| {
				let tree = Arc::clone(&tree);
				let running = Arc::clone(&running);
				let ops = Arc::clone(&writer_ops);
				thread::spawn(move || {
					let mut rng = rand::rng();
					while running.load(Ordering::Relaxed) {
						let key: i32 = rng.random_range(0..500);
						if rng.random_bool(0.5) {
							tree.insert(key, t);
						} else {
							tree.remove(&key);
						}
						ops.fetch_add(1, Ordering::Relaxed);
					}
				})
			})
			.collect();

		// Give writers a head start
		thread::sleep(Duration::from_millis(50));

		// Spawn reader and measure iterations
		let tree_reader = Arc::clone(&tree);
		let reader_iterations_clone = Arc::clone(&reader_iterations);
		let start = Instant::now();

		let reader = thread::spawn(move || {
			for _ in 0..reader_target_iterations {
				let mut iter = tree_reader.raw_iter();
				iter.seek_to_first();

				// Verify sorted order during iteration
				let mut prev = -1i32;
				while let Some((k, _)) = iter.next() {
					assert!(*k > prev, "Order violation during read under write pressure");
					prev = *k;
				}

				reader_iterations_clone.fetch_add(1, Ordering::Relaxed);
			}
		});

		reader.join().unwrap();
		let read_duration = start.elapsed();

		// Stop writers
		running.store(false, Ordering::Relaxed);
		for h in writer_handles {
			h.join().unwrap();
		}

		let final_reader_iters = reader_iterations.load(Ordering::Relaxed);
		let final_writer_ops = writer_ops.load(Ordering::Relaxed);

		// Reader should have completed all iterations
		assert_eq!(
			final_reader_iters, reader_target_iterations as u64,
			"Reader didn't complete all iterations"
		);

		// Reader shouldn't take too long
		assert!(
			read_duration < Duration::from_secs(15),
			"Reader took too long ({:?}) - possible starvation",
			read_duration
		);

		// Writers should have made progress
		assert!(final_writer_ops > 0, "Writers made no progress");

		tree.assert_invariants();
	});
}

/// Test fairness under mixed workload.
///
/// This tests that both readers and writers make reasonable progress when
/// competing for access.
#[test]
fn starvation_fairness_mixed_workload() {
	run_with_timeout(Duration::from_secs(30), "fairness_mixed_workload", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		// Pre-populate
		for i in 0..200 {
			tree.insert(i, i);
		}

		let num_readers = 4;
		let num_writers = 4;
		let test_duration = Duration::from_secs(3);
		let running = Arc::new(AtomicBool::new(true));
		let reader_ops = Arc::new(AtomicU64::new(0));
		let writer_ops = Arc::new(AtomicU64::new(0));

		// Spawn readers
		let reader_handles: Vec<_> = (0..num_readers)
			.map(|_| {
				let tree = Arc::clone(&tree);
				let running = Arc::clone(&running);
				let ops = Arc::clone(&reader_ops);
				thread::spawn(move || {
					while running.load(Ordering::Relaxed) {
						let mut iter = tree.raw_iter();
						iter.seek_to_first();
						while iter.next().is_some() {
							ops.fetch_add(1, Ordering::Relaxed);
						}
					}
				})
			})
			.collect();

		// Spawn writers
		let writer_handles: Vec<_> = (0..num_writers)
			.map(|t| {
				let tree = Arc::clone(&tree);
				let running = Arc::clone(&running);
				let ops = Arc::clone(&writer_ops);
				thread::spawn(move || {
					let mut rng = rand::rng();
					while running.load(Ordering::Relaxed) {
						let key: i32 = rng.random_range(0..500);
						if rng.random_bool(0.5) {
							tree.insert(key, t);
						} else {
							tree.remove(&key);
						}
						ops.fetch_add(1, Ordering::Relaxed);
					}
				})
			})
			.collect();

		// Run for the specified duration
		thread::sleep(test_duration);
		running.store(false, Ordering::Relaxed);

		// Wait for all threads
		for h in reader_handles {
			h.join().unwrap();
		}
		for h in writer_handles {
			h.join().unwrap();
		}

		let final_reader_ops = reader_ops.load(Ordering::Relaxed);
		let final_writer_ops = writer_ops.load(Ordering::Relaxed);

		// Both should have made significant progress
		assert!(
			final_reader_ops > 100,
			"Readers made insufficient progress: {} ops",
			final_reader_ops
		);
		assert!(
			final_writer_ops > 100,
			"Writers made insufficient progress: {} ops",
			final_writer_ops
		);

		// Check fairness ratio (neither should dominate completely)
		let ratio = if final_reader_ops > final_writer_ops {
			final_reader_ops as f64 / final_writer_ops as f64
		} else {
			final_writer_ops as f64 / final_reader_ops as f64
		};

		// Allow up to 100:1 ratio (readers naturally do more ops since lookups are cheaper)
		assert!(
			ratio < 100.0,
			"Unfair workload distribution: reader_ops={}, writer_ops={}, ratio={}",
			final_reader_ops,
			final_writer_ops,
			ratio
		);

		tree.assert_invariants();
	});
}

/// Long-running starvation test (ignored by default).
///
/// This runs for a longer duration to catch subtle starvation issues that
/// might not manifest in shorter tests.
#[test]
#[ignore]
fn starvation_long_running_fairness() {
	run_with_timeout(Duration::from_secs(120), "long_running_fairness", || {
		let tree = Arc::new(Tree::<i32, i32>::new());

		for i in 0..1000 {
			tree.insert(i, i);
		}

		let num_readers = 8;
		let num_writers = 4;
		let test_duration = Duration::from_secs(60);
		let running = Arc::new(AtomicBool::new(true));

		// Track operations per second for each type
		let reader_ops = Arc::new(AtomicU64::new(0));
		let writer_ops = Arc::new(AtomicU64::new(0));

		let reader_handles: Vec<_> = (0..num_readers)
			.map(|_| {
				let tree = Arc::clone(&tree);
				let running = Arc::clone(&running);
				let ops = Arc::clone(&reader_ops);
				thread::spawn(move || {
					while running.load(Ordering::Relaxed) {
						let mut iter = tree.raw_iter();
						iter.seek_to_first();
						while iter.next().is_some() {}
						ops.fetch_add(1, Ordering::Relaxed);
					}
				})
			})
			.collect();

		let writer_handles: Vec<_> = (0..num_writers)
			.map(|t| {
				let tree = Arc::clone(&tree);
				let running = Arc::clone(&running);
				let ops = Arc::clone(&writer_ops);
				thread::spawn(move || {
					let mut rng = rand::rng();
					while running.load(Ordering::Relaxed) {
						let key: i32 = rng.random_range(0..2000);
						if rng.random_bool(0.5) {
							tree.insert(key, t);
						} else {
							tree.remove(&key);
						}
						ops.fetch_add(1, Ordering::Relaxed);
					}
				})
			})
			.collect();

		thread::sleep(test_duration);
		running.store(false, Ordering::Relaxed);

		for h in reader_handles {
			h.join().unwrap();
		}
		for h in writer_handles {
			h.join().unwrap();
		}

		let final_reader_ops = reader_ops.load(Ordering::Relaxed);
		let final_writer_ops = writer_ops.load(Ordering::Relaxed);

		tree.assert_invariants();

		// Both must have made substantial progress over 60 seconds
		assert!(
			final_reader_ops > 1000,
			"Readers potentially starved: only {} iterations",
			final_reader_ops
		);
		assert!(
			final_writer_ops > 10000,
			"Writers potentially starved: only {} ops",
			final_writer_ops
		);
	});
}
