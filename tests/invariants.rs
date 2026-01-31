//! # Invariant Testing for Ferntree B+ Tree
//!
//! This module contains tests specifically designed to validate tree invariants
//! and ensure unreachable code paths are never reached. It focuses on:
//!
//! - Boundary conditions for splits and merges
//! - Randomized operations with invariant validation
//! - Edge cases in tree structure modifications

use ferntree::Tree;
use rand::prelude::*;

// ===========================================================================
// Split Boundary Tests
// ===========================================================================

/// Test split at exact leaf capacity.
/// Inserts exactly enough items to fill a leaf, then one more to trigger split.
#[test]
fn split_at_exact_leaf_capacity() {
	// Using default Tree which has LEAF_CAPACITY = 64
	let tree: Tree<i32, i32> = Tree::new();

	// Insert exactly 64 items (leaf capacity)
	for i in 0..64 {
		tree.insert(i, i * 10);
	}

	tree.assert_invariants();
	assert_eq!(tree.len(), 64);

	// Insert one more to trigger a split
	tree.insert(64, 640);

	tree.assert_invariants();
	assert_eq!(tree.len(), 65);
	assert!(tree.height() >= 2, "Expected height >= 2 after split, got {}", tree.height());

	// Verify all entries are still accessible
	for i in 0..=64 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i * 10), "Key {} not found after split", i);
	}
}

/// Test split at exact internal node capacity by filling multiple leaf nodes.
#[test]
fn split_at_exact_internal_capacity() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert enough to fill an internal node's worth of children
	// Internal capacity is 64, so we need 64+ children
	// Each leaf holds ~64 entries, so we need 64 * 64 = 4096+ entries to get
	// enough leaf nodes to potentially split an internal node
	for i in 0..5000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();
	assert_eq!(tree.len(), 5000);
	assert!(tree.height() >= 2, "Expected height >= 2");

	// Verify all entries
	for i in 0..5000 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i), "Key {} not found", i);
	}
}

/// Test the transition from a single-leaf root to an internal root with two leaf children.
#[test]
fn root_split_leaf_to_internal() {
	let tree: Tree<i32, i32> = Tree::new();

	// Start with height 1 (single leaf root)
	assert_eq!(tree.height(), 1);

	// Insert until we force a root split
	for i in 0..100 {
		tree.insert(i, i);
		tree.assert_invariants();
	}

	// After split, height should increase
	assert!(tree.height() >= 2, "Root should have split to create internal node");

	// Verify structure integrity after the split
	tree.assert_invariants();

	// Verify all entries
	for i in 0..100 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i));
	}
}

/// Test cascading splits by forcing multiple levels of splits.
#[test]
fn cascading_splits() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert a large number of entries to force multiple levels
	for i in 0..10_000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();
	assert!(tree.height() >= 3, "Expected height >= 3 for cascading splits, got {}", tree.height());

	// Verify all entries
	for i in 0..10_000 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i), "Key {} not found", i);
	}
}

/// Test splits with reverse-order insertions.
#[test]
fn splits_with_reverse_order() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert in reverse order to stress different split scenarios
	for i in (0..1000).rev() {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Verify order is maintained
	let mut prev = -1;
	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	while let Some((k, _)) = iter.next() {
		assert!(*k > prev, "Keys not in sorted order");
		prev = *k;
	}
}

// ===========================================================================
// Merge Boundary Tests
// ===========================================================================

/// Test merge when nodes are at minimum occupancy (40% threshold).
#[test]
fn merge_at_minimum_occupancy() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert entries to create multiple nodes
	for i in 0..200 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Delete entries to bring nodes to underfull threshold
	// Delete strategically to force merges
	for i in 0..150 {
		tree.remove(&i);
		tree.assert_invariants();
	}

	assert_eq!(tree.len(), 50);

	// Verify remaining entries
	for i in 150..200 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i));
	}
}

/// Test cascading merges by deleting in a pattern that forces merge propagation.
#[test]
fn cascading_merges() {
	let tree: Tree<i32, i32> = Tree::new();

	// Build up a large tree
	for i in 0..5000 {
		tree.insert(i, i);
	}

	let initial_height = tree.height();
	tree.assert_invariants();

	// Delete most entries to force cascading merges and height reduction
	for i in 0..4900 {
		tree.remove(&i);
	}

	tree.assert_invariants();
	assert_eq!(tree.len(), 100);

	// Height should have decreased or remained the same (depending on tree structure)
	// The important thing is that invariants are maintained
	assert!(
		tree.height() <= initial_height,
		"Height should not increase after deletions, got {} (was {})",
		tree.height(),
		initial_height
	);

	// Verify remaining entries
	for i in 4900..5000 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i), "Key {} not found", i);
	}
}

/// Stress test: alternating merges and splits on the same region.
#[test]
fn merge_then_split_same_region() {
	let tree: Tree<i32, i32> = Tree::new();

	// Initial population
	for i in 0..100 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Alternate between heavy deletes (causing merges) and inserts (causing splits)
	for round in 0..5 {
		// Delete phase - remove many entries
		for i in 0..50 {
			tree.remove(&(i + round * 100));
		}
		tree.assert_invariants();

		// Insert phase - add new entries
		for i in 0..100 {
			tree.insert(i + (round + 1) * 100, i);
		}
		tree.assert_invariants();
	}
}

// ===========================================================================
// Iterator Boundary Tests
// ===========================================================================

/// Test iteration when tree structure changes during lifetime.
#[test]
fn iterator_with_structure_changes() {
	let tree: Tree<i32, i32> = Tree::new();

	// Build initial tree
	for i in 0..500 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Perform some modifications
	for i in 0..100 {
		tree.remove(&i);
	}
	for i in 500..600 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Verify iteration still works correctly
	let mut count = 0;
	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	while iter.next().is_some() {
		count += 1;
	}

	assert_eq!(count, tree.len());
}

/// Test seeking to various positions in a tree with multiple levels.
#[test]
fn seek_across_levels() {
	let tree: Tree<i32, i32> = Tree::new();

	// Build a multi-level tree
	for i in 0..2000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	let mut iter = tree.raw_iter();

	// Seek to beginning
	iter.seek_to_first();
	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, 0);

	// Seek to middle
	iter.seek(&1000);
	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, 1000);

	// Seek to end
	iter.seek_to_last();
	let (k, _) = iter.prev().unwrap();
	assert_eq!(*k, 1999);

	// Seek to non-existent key in the middle
	iter.seek(&1500);
	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, 1500);
}

// ===========================================================================
// Randomized Invariant Tests
// ===========================================================================

/// Randomized operations with periodic invariant validation.
#[test]
fn random_operations_with_invariant_checks() {
	let tree: Tree<i32, i32> = Tree::new();
	let mut rng = rand::rng();
	let mut expected: std::collections::BTreeMap<i32, i32> = std::collections::BTreeMap::new();

	for op in 0..10_000 {
		let key: i32 = rng.random_range(0..1000);

		match rng.random_range(0..3) {
			0 => {
				// Insert
				let value = key * 10;
				tree.insert(key, value);
				expected.insert(key, value);
			}
			1 => {
				// Remove
				let tree_result = tree.remove(&key);
				let expected_result = expected.remove(&key);
				assert_eq!(tree_result, expected_result);
			}
			2 => {
				// Lookup
				let tree_result = tree.lookup(&key, |v| *v);
				let expected_result = expected.get(&key).copied();
				assert_eq!(tree_result, expected_result);
			}
			_ => unreachable!(),
		}

		// Validate every 100 operations
		if op % 100 == 0 {
			tree.assert_invariants();
			assert_eq!(tree.len(), expected.len());
		}
	}

	// Final validation
	tree.assert_invariants();
	assert_eq!(tree.len(), expected.len());

	// Verify all expected entries are present
	for (k, v) in &expected {
		assert_eq!(tree.lookup(k, |val| *val), Some(*v), "Key {} not found", k);
	}
}

/// Heavy random workload with validation checkpoints.
#[test]
fn stress_random_workload() {
	let tree: Tree<i32, i32> = Tree::new();
	let mut rng = rand::rng();

	// Phase 1: Heavy insertions
	for _ in 0..5000 {
		let key: i32 = rng.random_range(0..10_000);
		tree.insert(key, key);
	}
	tree.assert_invariants();

	// Phase 2: Mixed operations
	for _ in 0..5000 {
		let key: i32 = rng.random_range(0..10_000);
		if rng.random_bool(0.5) {
			tree.insert(key, key);
		} else {
			tree.remove(&key);
		}
	}
	tree.assert_invariants();

	// Phase 3: Heavy deletions
	for _ in 0..3000 {
		let key: i32 = rng.random_range(0..10_000);
		tree.remove(&key);
	}
	tree.assert_invariants();

	// Verify iteration matches length
	let mut count = 0;
	let mut iter = tree.raw_iter();
	iter.seek_to_first();
	while iter.next().is_some() {
		count += 1;
	}
	assert_eq!(count, tree.len());
}

// ===========================================================================
// Edge Case Tests
// ===========================================================================

/// Test with sequential inserts and random deletes.
#[test]
fn sequential_insert_random_delete() {
	let tree: Tree<i32, i32> = Tree::new();
	let mut rng = rand::rng();

	// Sequential inserts
	for i in 0..1000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Random deletes
	let mut keys: Vec<i32> = (0..1000).collect();
	keys.shuffle(&mut rng);

	for key in keys.iter().take(500) {
		tree.remove(key);
	}

	tree.assert_invariants();

	// Verify remaining
	for key in keys.iter().skip(500) {
		assert_eq!(tree.lookup(key, |v| *v), Some(*key));
	}
}

/// Test with all same keys (updates only).
#[test]
fn repeated_same_key_updates() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert and update the same key many times
	for i in 0..1000 {
		tree.insert(42, i);
	}

	tree.assert_invariants();
	assert_eq!(tree.len(), 1);
	assert_eq!(tree.lookup(&42, |v| *v), Some(999));
}

/// Test boundary with i32::MIN and i32::MAX keys.
#[test]
fn boundary_key_values() {
	let tree: Tree<i32, i32> = Tree::new();

	tree.insert(i32::MIN, 1);
	tree.insert(i32::MAX, 2);
	tree.insert(0, 3);

	tree.assert_invariants();

	assert_eq!(tree.lookup(&i32::MIN, |v| *v), Some(1));
	assert_eq!(tree.lookup(&i32::MAX, |v| *v), Some(2));
	assert_eq!(tree.lookup(&0, |v| *v), Some(3));

	// Check iteration order
	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, i32::MIN);

	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, 0);

	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, i32::MAX);
}

/// Test empty tree operations.
#[test]
fn empty_tree_invariants() {
	let tree: Tree<i32, i32> = Tree::new();

	tree.assert_invariants();
	assert!(tree.is_empty());
	assert_eq!(tree.height(), 1);

	// Operations on empty tree
	assert_eq!(tree.remove(&1), None);
	assert_eq!(tree.lookup(&1, |v| *v), None);

	tree.assert_invariants();
}

/// Test tree after removing all entries.
#[test]
fn tree_after_clearing_all() {
	let tree: Tree<i32, i32> = Tree::new();

	// Build up
	for i in 0..500 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Remove all
	for i in 0..500 {
		tree.remove(&i);
	}

	tree.assert_invariants();
	assert!(tree.is_empty());

	// Can still insert after clearing
	tree.insert(1, 10);
	tree.assert_invariants();
	assert_eq!(tree.len(), 1);
}
