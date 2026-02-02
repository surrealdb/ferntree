// Explicit drops are used for clarity in memory leak tests, even when the type
// doesn't implement Drop. This documents the point at which reclamation should occur.
#![allow(clippy::drop_non_drop)]

//! Memory leak detection tests for ferntree.
//!
//! These tests verify that memory is properly reclaimed after tree operations.
//! They are designed to catch memory leaks in the epoch-based reclamation system.
//!
//! # Running Memory Tests
//!
//! These tests can be run normally:
//!
//! ```bash
//! cargo test -p ferntree memory_tests
//! ```
//!
//! For more thorough leak detection, run under AddressSanitizer or LeakSanitizer:
//!
//! ```bash
//! RUSTFLAGS="-Zsanitizer=leak" cargo +nightly test -p ferntree --target x86_64-unknown-linux-gnu
//! ```
//!
//! # Test Design
//!
//! Each test follows this pattern:
//! 1. Perform tree operations that allocate memory
//! 2. Drop the tree to trigger cleanup
//! 3. Force epoch advancement to trigger deferred reclamation
//! 4. Verify memory is reclaimed (when using tracking allocator or sanitizers)

use ferntree::Tree;
use std::sync::Arc;
use std::thread;

// ===========================================================================
// Helper Functions
// ===========================================================================

/// Force epoch advancement to trigger garbage collection of deferred items.
///
/// crossbeam_epoch uses deferred reclamation, so dropped items aren't
/// immediately freed. Calling pin() multiple times helps advance the
/// epoch and trigger cleanup.
fn force_epoch_advancement() {
	for _ in 0..10 {
		let _guard = crossbeam_epoch::pin();
		// Small sleep to allow GC threads to run
		std::thread::yield_now();
	}
}

// ===========================================================================
// Basic Memory Tests
// ===========================================================================

/// Verify nodes are reclaimed after inserting and removing all entries.
#[test]
fn no_leak_after_insert_remove_all() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert entries
	for i in 0..1000 {
		tree.insert(i, i);
	}

	// Remove all entries
	for i in 0..1000 {
		tree.remove(&i);
	}

	assert!(tree.is_empty());

	// Drop tree
	drop(tree);

	// Force epoch advancement for deferred reclamation
	force_epoch_advancement();
}

/// Verify nodes are reclaimed after clear().
#[test]
fn no_leak_after_clear() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert entries to create multiple nodes
	for i in 0..1000 {
		tree.insert(i, i);
	}

	assert!(tree.height() > 1);

	// Clear the tree
	tree.clear();

	assert!(tree.is_empty());
	assert_eq!(tree.height(), 1);

	// Drop tree
	drop(tree);

	force_epoch_advancement();
}

/// Verify memory is reclaimed with many updates to the same key.
#[test]
fn no_leak_repeated_updates() {
	let tree: Tree<i32, String> = Tree::new();

	// Repeatedly update the same key (creates garbage values)
	for i in 0..1000 {
		tree.insert(1, format!("value_{}", i));
	}

	assert_eq!(tree.len(), 1);

	drop(tree);
	force_epoch_advancement();
}

/// Verify memory is reclaimed after tree splits.
#[test]
fn no_leak_after_splits() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert enough to trigger multiple splits
	for i in 0..5000 {
		tree.insert(i, i);
	}

	assert!(tree.height() >= 3);

	drop(tree);
	force_epoch_advancement();
}

/// Verify memory is reclaimed after tree merges.
#[test]
fn no_leak_after_merges() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert entries
	for i in 0..1000 {
		tree.insert(i, i);
	}

	// Remove half to trigger merges
	for i in (0..1000).step_by(2) {
		tree.remove(&i);
	}

	drop(tree);
	force_epoch_advancement();
}

// ===========================================================================
// Concurrent Memory Tests
// ===========================================================================

/// Verify no leaks with concurrent insert operations.
#[test]
fn no_leak_concurrent_inserts() {
	let tree = Arc::new(Tree::<i32, i32>::new());

	let handles: Vec<_> = (0..4)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				for i in 0..250 {
					let key = t * 250 + i;
					tree.insert(key, key);
				}
			})
		})
		.collect();

	for h in handles {
		h.join().unwrap();
	}

	assert_eq!(tree.len(), 1000);

	// Drop the Arc (last reference)
	drop(tree);

	force_epoch_advancement();
}

/// Verify no leaks with concurrent insert and remove operations.
#[test]
fn no_leak_concurrent_insert_remove() {
	let tree = Arc::new(Tree::<i32, i32>::new());

	let handles: Vec<_> = (0..4)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				for i in 0..250 {
					let key = t * 250 + i;
					tree.insert(key, key);
				}
				for i in 0..250 {
					let key = t * 250 + i;
					tree.remove(&key);
				}
			})
		})
		.collect();

	for h in handles {
		h.join().unwrap();
	}

	assert!(tree.is_empty());

	drop(tree);
	force_epoch_advancement();
}

/// Verify no leaks with concurrent updates to same keys.
#[test]
fn no_leak_concurrent_updates() {
	let tree = Arc::new(Tree::<i32, i32>::new());

	// Pre-insert keys
	for i in 0..100 {
		tree.insert(i, 0);
	}

	let handles: Vec<_> = (0..4)
		.map(|t| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				for iter in 0..100 {
					for key in 0..100 {
						tree.insert(key, t * 100 + iter);
					}
				}
			})
		})
		.collect();

	for h in handles {
		h.join().unwrap();
	}

	assert_eq!(tree.len(), 100);

	drop(tree);
	force_epoch_advancement();
}

// ===========================================================================
// Iterator Memory Tests
// ===========================================================================

/// Verify iterator doesn't leak when dropped mid-iteration.
#[test]
fn no_leak_iterator_early_drop() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..100 {
		tree.insert(i, i);
	}

	// Create and drop iterator mid-way
	{
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		for _ in 0..50 {
			let _ = iter.next();
		}
		// Iterator dropped here without completing
	}

	// Tree should still work
	assert_eq!(tree.len(), 100);

	drop(tree);
	force_epoch_advancement();
}

/// Verify multiple iterators don't leak.
#[test]
fn no_leak_multiple_iterators() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..100 {
		tree.insert(i, i);
	}

	// Create multiple iterators
	for _ in 0..10 {
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		while iter.next().is_some() {}
	}

	drop(tree);
	force_epoch_advancement();
}

/// Verify mutable iterator doesn't leak.
#[test]
fn no_leak_mutable_iterator() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..100 {
		tree.insert(i, i);
	}

	// Use mutable iterator
	{
		let mut iter = tree.raw_iter_mut();
		iter.seek_to_first();
		while let Some((_, v)) = iter.next() {
			*v *= 2;
		}
	}

	drop(tree);
	force_epoch_advancement();
}

// ===========================================================================
// Edge Case Memory Tests
// ===========================================================================

/// Verify empty tree has minimal memory footprint.
#[test]
fn no_leak_empty_tree() {
	let tree: Tree<i32, i32> = Tree::new();
	drop(tree);
	force_epoch_advancement();
}

/// Verify single element tree properly cleans up.
#[test]
fn no_leak_single_element() {
	let tree: Tree<i32, i32> = Tree::new();
	tree.insert(1, 1);
	drop(tree);
	force_epoch_advancement();
}

/// Verify tree with large values doesn't leak.
#[test]
fn no_leak_large_values() {
	let tree: Tree<i32, Vec<u8>> = Tree::new();

	for i in 0..100 {
		// Each value is 1KB
		tree.insert(i, vec![0u8; 1024]);
	}

	drop(tree);
	force_epoch_advancement();
}

/// Verify tree with string keys/values doesn't leak.
#[test]
fn no_leak_string_tree() {
	let tree: Tree<String, String> = Tree::new();

	for i in 0..100 {
		tree.insert(format!("key_{}", i), format!("value_{}", i));
	}

	for i in 0..100 {
		tree.remove(&format!("key_{}", i));
	}

	drop(tree);
	force_epoch_advancement();
}

/// Verify pop_first doesn't leak.
#[test]
fn no_leak_pop_first() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..100 {
		tree.insert(i, i);
	}

	while tree.pop_first().is_some() {}

	assert!(tree.is_empty());

	drop(tree);
	force_epoch_advancement();
}

/// Verify pop_last doesn't leak.
#[test]
fn no_leak_pop_last() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..100 {
		tree.insert(i, i);
	}

	while tree.pop_last().is_some() {}

	assert!(tree.is_empty());

	drop(tree);
	force_epoch_advancement();
}

// ===========================================================================
// Stress Tests
// ===========================================================================

/// Stress test with many small operations.
#[test]
fn no_leak_stress_small_ops() {
	let tree: Tree<i32, i32> = Tree::new();

	for round in 0..10 {
		// Insert batch
		for i in 0..100 {
			tree.insert(round * 100 + i, i);
		}

		// Remove half
		for i in (0..100).step_by(2) {
			tree.remove(&(round * 100 + i));
		}
	}

	drop(tree);
	force_epoch_advancement();
}

/// Stress test with tree rebuilding.
#[test]
fn no_leak_stress_rebuild() {
	for _ in 0..5 {
		let tree: Tree<i32, i32> = Tree::new();

		for i in 0..500 {
			tree.insert(i, i);
		}

		for i in 0..500 {
			tree.remove(&i);
		}

		drop(tree);
	}

	force_epoch_advancement();
}
