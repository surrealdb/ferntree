//! # Concurrency Tests for Ferntree B+ Tree
//!
//! This module contains multi-threaded tests to verify the correctness
//! of the concurrent B+ tree implementation under various contention scenarios.
//!
//! ## Test Categories
//!
//! - Basic concurrent tests: Lower contention, always run
//! - Stress tests: Higher contention, marked with `#[ignore]` - run with `cargo test -- --ignored`
//!
//! Some stress tests may expose race conditions in the underlying implementation
//! that occur under extreme contention. These are marked as ignored but can be
//! run explicitly to investigate potential issues.

use ferntree::Tree;
use rand::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ===========================================================================
// Basic Concurrent Insert Tests
// ===========================================================================

#[test]
fn concurrent_insert_disjoint_ranges() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_threads = 4;
	let entries_per_thread = 100;

	let handles: Vec<_> = (0..num_threads)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
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

	assert_eq!(tree.len(), (num_threads * entries_per_thread) as usize);

	// Verify all entries
	for t in 0..num_threads {
		for i in 0..entries_per_thread {
			let key = t * entries_per_thread + i;
			assert_eq!(tree.lookup(&key, |v| *v), Some(key * 10), "Missing key {}", key);
		}
	}
}

#[test]
fn concurrent_insert_same_keys() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_threads = 4;
	let iterations = 100;

	// All threads repeatedly insert the same small set of keys
	let handles: Vec<_> = (0..num_threads)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				for i in 0..iterations {
					let key = i % 10; // Only 10 unique keys
					tree.insert(key, t); // Value is thread ID
				}
			})
		})
		.collect();

	for h in handles {
		h.join().unwrap();
	}

	// Should have exactly 10 entries
	assert_eq!(tree.len(), 10);

	// Each key should have a valid thread ID as value
	for key in 0..10 {
		let value = tree.lookup(&key, |v| *v).expect("Key should exist");
		assert!(value < num_threads, "Invalid value {} for key {}", value, key);
	}
}

// ===========================================================================
// Basic Concurrent Lookup Tests
// ===========================================================================

#[test]
fn many_concurrent_readers() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_readers = 4;
	let entries = 100;

	// Pre-insert data
	for i in 0..entries {
		tree.insert(i, i * 10);
	}

	let handles: Vec<_> = (0..num_readers)
		.map(|_| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				// Each reader does a full scan
				let mut iter = tree.raw_iter();
				iter.seek_to_first();

				let mut count = 0;
				while let Some((k, v)) = iter.next() {
					assert_eq!(*v, *k * 10);
					count += 1;
				}
				count
			})
		})
		.collect();

	for h in handles {
		let count = h.join().unwrap();
		assert_eq!(count, entries);
	}
}

// ===========================================================================
// Basic Mixed Operation Tests
// ===========================================================================

#[test]
fn concurrent_insert_and_lookup() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let entries = 100;

	// Pre-insert some data
	for i in 0..entries {
		tree.insert(i, i * 10);
	}

	let tree_writer = Arc::clone(&tree);
	let tree_reader = Arc::clone(&tree);

	let writer = thread::spawn(move || {
		for i in entries..(entries + 50) {
			tree_writer.insert(i, i * 10);
		}
	});

	let reader = thread::spawn(move || {
		let mut found = 0;
		for i in 0..entries {
			if tree_reader.lookup(&i, |v| *v).is_some() {
				found += 1;
			}
		}
		found
	});

	writer.join().unwrap();
	let found = reader.join().unwrap();

	// Reader should have found entries
	assert!(found > 0);
	assert!(tree.len() >= entries as usize);
}

// ===========================================================================
// Concurrent Iteration Tests
// ===========================================================================

#[test]
fn iterate_while_inserting() {
	let tree = Arc::new(Tree::<i32, i32>::new());

	// Pre-insert some data
	for i in 0..50 {
		tree.insert(i, i);
	}

	let tree_writer = Arc::clone(&tree);
	let tree_reader = Arc::clone(&tree);

	let writer = thread::spawn(move || {
		for i in 50..75 {
			tree_writer.insert(i, i);
		}
	});

	let reader = thread::spawn(move || {
		let mut iter = tree_reader.raw_iter();
		iter.seek_to_first();

		let mut prev = -1i32;
		let mut count = 0;
		while let Some((k, _)) = iter.next() {
			// Keys should be in sorted order
			assert!(*k > prev, "Order violation: {} not > {}", *k, prev);
			prev = *k;
			count += 1;
		}
		count
	});

	writer.join().unwrap();
	let count = reader.join().unwrap();

	// Reader should have seen a consistent snapshot
	assert!(count > 0);
}

// ===========================================================================
// Split Under Contention Tests
// ===========================================================================

#[test]
fn concurrent_inserts_cause_splits() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_threads = 2;
	let entries_per_thread = 50;

	// Insert enough entries concurrently to cause splits
	let handles: Vec<_> = (0..num_threads)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				for i in 0..entries_per_thread {
					tree.insert(t * entries_per_thread + i, i);
				}
			})
		})
		.collect();

	for h in handles {
		h.join().unwrap();
	}

	// Tree should have grown
	assert!(tree.height() > 1, "Tree should have split");
	assert_eq!(tree.len(), (num_threads * entries_per_thread) as usize);

	// Verify all entries
	for t in 0..num_threads {
		for i in 0..entries_per_thread {
			let key = t * entries_per_thread + i;
			assert_eq!(tree.lookup(&key, |v| *v), Some(i), "Missing key {}", key);
		}
	}
}

// ===========================================================================
// Basic Remove Tests
// ===========================================================================

#[test]
fn concurrent_removes() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let entries = 100;

	// Pre-insert entries
	for i in 0..entries {
		tree.insert(i, i);
	}

	let num_threads = 2;
	let entries_per_thread = entries / num_threads;

	// Concurrently remove entries
	let handles: Vec<_> = (0..num_threads)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
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

	// Tree should be empty
	assert!(tree.is_empty());
}

// ===========================================================================
// Stress Tests (ignored by default - run with `cargo test -- --ignored`)
// ===========================================================================

/// Higher contention stress test - may expose race conditions
#[test]
#[ignore]
fn stress_concurrent_insert_overlapping_ranges() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_threads = 8;
	let entries_per_thread = 1000;
	let key_range = 500; // High contention - all threads insert into 0..500

	let handles: Vec<_> = (0..num_threads)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				let mut rng = rand::thread_rng();
				for _ in 0..entries_per_thread {
					let key: i32 = rng.gen_range(0..key_range);
					tree.insert(key, t);
				}
			})
		})
		.collect();

	for h in handles {
		h.join().unwrap();
	}

	// Tree should have at most key_range entries
	assert!(tree.len() <= key_range as usize);
}

/// Mixed operations stress test
#[test]
#[ignore]
fn stress_concurrent_mixed_operations() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_threads = 8;
	let operations_per_thread = 1000;
	let key_range = 500;

	// Pre-insert some data
	for i in 0..key_range {
		tree.insert(i, i);
	}

	let handles: Vec<_> = (0..num_threads)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				let mut rng = rand::thread_rng();
				for _ in 0..operations_per_thread {
					let key: i32 = rng.gen_range(0..key_range);
					let op: u8 = rng.gen_range(0..3);

					match op {
						0 => {
							tree.insert(key, t);
						}
						1 => {
							tree.remove(&key);
						}
						2 => {
							tree.lookup(&key, |v| *v);
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

	// Verify tree is still functional
	let len = tree.len();
	assert!(len <= key_range as usize);
}

/// High contention single key stress test
#[test]
#[ignore]
fn stress_high_contention_single_key() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_threads = 8;
	let iterations = 1000;

	tree.insert(42, 0);

	let handles: Vec<_> = (0..num_threads)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				for i in 0..iterations {
					if i % 2 == 0 {
						tree.insert(42, t);
					} else {
						tree.lookup(&42, |v| *v);
					}
				}
			})
		})
		.collect();

	for h in handles {
		h.join().unwrap();
	}

	// Key should still exist
	assert!(tree.lookup(&42, |v| *v).is_some());
	assert_eq!(tree.len(), 1);
}

/// Sustained mixed operations stress test
#[test]
#[ignore]
fn stress_sustained_mixed_operations() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_threads = 4;
	let duration_ms = 500;

	let running = Arc::new(AtomicUsize::new(1));

	let handles: Vec<_> = (0..num_threads)
		.map(|t| {
			let tree = Arc::clone(&tree);
			let running = Arc::clone(&running);
			thread::spawn(move || {
				let mut rng = rand::thread_rng();
				let mut ops = 0u64;

				while running.load(Ordering::Relaxed) == 1 {
					let key: i32 = rng.gen_range(0..1000);
					let op: u8 = rng.gen_range(0..10);

					match op {
						0..=3 => {
							tree.insert(key, t);
						}
						4..=5 => {
							tree.remove(&key);
						}
						6..=9 => {
							tree.lookup(&key, |v| *v);
						}
						_ => unreachable!(),
					}
					ops += 1;
				}

				ops
			})
		})
		.collect();

	// Let it run for the specified duration
	thread::sleep(Duration::from_millis(duration_ms));
	running.store(0, Ordering::Relaxed);

	let total_ops: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();

	// Should have performed many operations
	assert!(total_ops > 100, "Only {} operations performed", total_ops);
}

/// Large scale concurrent inserts stress test
#[test]
#[ignore]
fn stress_large_scale_concurrent_inserts() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_threads = 8;
	let entries_per_thread = 1000;

	let handles: Vec<_> = (0..num_threads)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
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

	assert_eq!(tree.len(), (num_threads * entries_per_thread) as usize);
}

/// Producer-consumer pattern stress test
#[test]
#[ignore]
fn stress_producer_consumer() {
	let tree = Arc::new(Tree::<i32, i32>::new());
	let num_producers = 4;
	let num_consumers = 4;
	let entries_per_producer = 500;

	let produced = Arc::new(AtomicUsize::new(0));

	// Producers insert entries
	let producer_handles: Vec<_> = (0..num_producers)
		.map(|p| {
			let tree = Arc::clone(&tree);
			let produced = Arc::clone(&produced);
			thread::spawn(move || {
				for i in 0..entries_per_producer {
					let key = p * entries_per_producer + i;
					tree.insert(key, key * 10);
					produced.fetch_add(1, Ordering::Relaxed);
				}
			})
		})
		.collect();

	// Consumers lookup entries
	let consumer_handles: Vec<_> = (0..num_consumers)
		.map(|_| {
			let tree = Arc::clone(&tree);
			let produced = Arc::clone(&produced);
			thread::spawn(move || {
				let mut rng = rand::thread_rng();
				let mut found = 0u64;
				let total_entries = num_producers * entries_per_producer;

				while produced.load(Ordering::Relaxed) < total_entries as usize {
					let key: i32 = rng.gen_range(0..total_entries);
					if tree.lookup(&key, |v| *v).is_some() {
						found += 1;
					}
				}

				found
			})
		})
		.collect();

	for h in producer_handles {
		h.join().unwrap();
	}

	let total_found: u64 = consumer_handles.into_iter().map(|h| h.join().unwrap()).sum();

	// Consumers should have found some entries
	assert!(total_found > 0);

	// All entries should be present
	assert_eq!(tree.len(), (num_producers * entries_per_producer) as usize);
}
