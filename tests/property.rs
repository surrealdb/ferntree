//! # Property-Based Tests for Ferntree B+ Tree
//!
//! This module contains property-based tests using proptest to systematically
//! discover edge cases through randomized testing. These tests verify that
//! tree invariants hold across thousands of random inputs.
//!
//! ## Test Properties
//!
//! - Insert-then-lookup: All inserted keys must be retrievable
//! - Remove-then-lookup: Removed keys must not be found
//! - Ordering: Iteration always yields sorted keys
//! - Length consistency: Tree length matches expected count
//! - Bidirectional iteration: Forward and reverse yield same elements
//! - Oracle comparison: Behavior matches BTreeMap reference

use ferntree::Tree;
use proptest::prelude::*;
use std::collections::BTreeMap;

// ===========================================================================
// Strategy Helpers
// ===========================================================================

/// Generate a vector of unique keys for testing
fn unique_keys(max_len: usize) -> impl Strategy<Value = Vec<i32>> {
	prop::collection::hash_set(any::<i32>(), 0..max_len).prop_map(|s| s.into_iter().collect())
}

/// Generate a vector of key-value pairs
fn key_value_pairs(max_len: usize) -> impl Strategy<Value = Vec<(i32, i32)>> {
	prop::collection::vec((any::<i32>(), any::<i32>()), 0..max_len)
}

/// Operations that can be performed on the tree
#[derive(Debug, Clone)]
enum Op {
	Insert(i32, i32),
	Remove(i32),
	Lookup(i32),
}

/// Generate a sequence of random operations
fn operations(max_ops: usize) -> impl Strategy<Value = Vec<Op>> {
	prop::collection::vec(
		prop_oneof![
			(any::<i32>(), any::<i32>()).prop_map(|(k, v)| Op::Insert(k, v)),
			any::<i32>().prop_map(Op::Remove),
			any::<i32>().prop_map(Op::Lookup),
		],
		0..max_ops,
	)
}

// ===========================================================================
// Insert-Then-Lookup Property
// ===========================================================================

proptest! {
	/// Property: After inserting a key-value pair, lookup returns that value
	#[test]
	fn insert_then_lookup(entries in key_value_pairs(500)) {
		let tree: Tree<i32, i32> = Tree::new();
		let mut expected: BTreeMap<i32, i32> = BTreeMap::new();

		// Insert all entries (last value wins for duplicates)
		for (k, v) in &entries {
			tree.insert(*k, *v);
			expected.insert(*k, *v);
		}

		tree.assert_invariants();

		// Verify all expected entries are present
		for (k, v) in &expected {
			let result = tree.lookup(k, |val| *val);
			prop_assert_eq!(result, Some(*v), "Key {} should have value {}", k, v);
		}

		// Verify length matches
		prop_assert_eq!(tree.len(), expected.len());
	}

	/// Property: All inserted keys must be retrievable
	#[test]
	fn all_inserted_keys_exist(keys in unique_keys(500)) {
		let tree: Tree<i32, i32> = Tree::new();

		for k in &keys {
			tree.insert(*k, k.wrapping_mul(10));
		}

		tree.assert_invariants();

		for k in &keys {
			prop_assert!(
				tree.contains_key(k),
				"Key {} should exist after insertion", k
			);
		}
	}
}

// ===========================================================================
// Remove-Then-Lookup Property
// ===========================================================================

proptest! {
	/// Property: After removing a key, lookup returns None
	#[test]
	fn remove_then_lookup(keys in unique_keys(200)) {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert all keys
		for k in &keys {
			tree.insert(*k, *k);
		}

		tree.assert_invariants();

		// Remove all keys and verify they're gone
		for k in &keys {
			let removed = tree.remove(k);
			prop_assert_eq!(removed, Some(*k), "Remove should return the value");
			prop_assert_eq!(tree.lookup(k, |v| *v), None, "Key {} should not exist after removal", k);
		}

		tree.assert_invariants();
		prop_assert!(tree.is_empty(), "Tree should be empty after removing all keys");
	}

	/// Property: Removing a non-existent key returns None
	#[test]
	fn remove_nonexistent_returns_none(
		existing in unique_keys(100),
		nonexistent in unique_keys(100)
	) {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert existing keys
		for k in &existing {
			tree.insert(*k, *k);
		}

		tree.assert_invariants();

		// Try to remove keys that might not exist
		for k in &nonexistent {
			if !existing.contains(k) {
				let removed = tree.remove(k);
				prop_assert_eq!(removed, None, "Removing non-existent key {} should return None", k);
			}
		}

		tree.assert_invariants();
	}
}

// ===========================================================================
// Ordering Property
// ===========================================================================

proptest! {
	/// Property: Forward iteration always yields keys in sorted order
	#[test]
	fn iteration_is_sorted(entries in key_value_pairs(500)) {
		let tree: Tree<i32, i32> = Tree::new();

		for (k, v) in &entries {
			tree.insert(*k, *v);
		}

		tree.assert_invariants();

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let mut prev: Option<i32> = None;
		while let Some((k, _)) = iter.next() {
			if let Some(p) = prev {
				prop_assert!(
					*k > p,
					"Keys should be in ascending order: {} should be > {}", k, p
				);
			}
			prev = Some(*k);
		}
	}

	/// Property: Reverse iteration yields keys in descending order
	#[test]
	fn reverse_iteration_is_sorted(entries in key_value_pairs(500)) {
		let tree: Tree<i32, i32> = Tree::new();

		for (k, v) in &entries {
			tree.insert(*k, *v);
		}

		tree.assert_invariants();

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		let mut prev: Option<i32> = None;
		while let Some((k, _)) = iter.prev() {
			if let Some(p) = prev {
				prop_assert!(
					*k < p,
					"Keys should be in descending order: {} should be < {}", k, p
				);
			}
			prev = Some(*k);
		}
	}
}

// ===========================================================================
// Bidirectional Iteration Property
// ===========================================================================

proptest! {
	/// Property: Forward then reverse iteration visits the same elements
	#[test]
	fn bidirectional_iteration_consistency(entries in key_value_pairs(200)) {
		let tree: Tree<i32, i32> = Tree::new();

		for (k, v) in &entries {
			tree.insert(*k, *v);
		}

		tree.assert_invariants();

		// Collect keys via forward iteration
		let mut forward_keys = Vec::new();
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		while let Some((k, _)) = iter.next() {
			forward_keys.push(*k);
		}

		// Collect keys via reverse iteration
		let mut reverse_keys = Vec::new();
		iter.seek_to_last();
		while let Some((k, _)) = iter.prev() {
			reverse_keys.push(*k);
		}

		// Reverse should be the opposite of forward
		reverse_keys.reverse();
		prop_assert_eq!(forward_keys, reverse_keys, "Forward and reverse iteration should yield same keys");
	}
}

// ===========================================================================
// Length Consistency Property
// ===========================================================================

proptest! {
	/// Property: Tree length equals number of unique keys
	#[test]
	fn length_matches_unique_keys(entries in key_value_pairs(500)) {
		let tree: Tree<i32, i32> = Tree::new();
		let mut expected: BTreeMap<i32, i32> = BTreeMap::new();

		for (k, v) in &entries {
			tree.insert(*k, *v);
			expected.insert(*k, *v);
		}

		tree.assert_invariants();
		prop_assert_eq!(tree.len(), expected.len(), "Length should match unique key count");
	}

	/// Property: Length updates correctly with inserts and removes
	#[test]
	fn length_tracks_operations(ops in operations(300)) {
		let tree: Tree<i32, i32> = Tree::new();
		let mut expected: BTreeMap<i32, i32> = BTreeMap::new();

		for op in &ops {
			match op {
				Op::Insert(k, v) => {
					tree.insert(*k, *v);
					expected.insert(*k, *v);
				}
				Op::Remove(k) => {
					tree.remove(k);
					expected.remove(k);
				}
				Op::Lookup(_) => {
					// Lookups don't change length
				}
			}
		}

		tree.assert_invariants();
		prop_assert_eq!(tree.len(), expected.len(), "Length should match after operations");
	}
}

// ===========================================================================
// Idempotent Update Property
// ===========================================================================

proptest! {
	/// Property: Inserting same key twice returns old value and updates to new
	#[test]
	fn update_returns_old_value(
		key in any::<i32>(),
		value1 in any::<i32>(),
		value2 in any::<i32>()
	) {
		let tree: Tree<i32, i32> = Tree::new();

		// First insert returns None
		let result1 = tree.insert(key, value1);
		prop_assert_eq!(result1, None, "First insert should return None");

		// Second insert returns old value
		let result2 = tree.insert(key, value2);
		prop_assert_eq!(result2, Some(value1), "Second insert should return old value");

		// Lookup returns new value
		let current = tree.lookup(&key, |v| *v);
		prop_assert_eq!(current, Some(value2), "Lookup should return new value");

		tree.assert_invariants();
	}
}

// ===========================================================================
// Oracle (BTreeMap) Comparison Property
// ===========================================================================

proptest! {
	/// Property: Tree behavior matches BTreeMap for all operation sequences
	#[test]
	fn matches_btreemap_oracle(ops in operations(500)) {
		let tree: Tree<i32, i32> = Tree::new();
		let mut oracle: BTreeMap<i32, i32> = BTreeMap::new();

		for op in &ops {
			match op {
				Op::Insert(k, v) => {
					let tree_result = tree.insert(*k, *v);
					let oracle_result = oracle.insert(*k, *v);
					prop_assert_eq!(
						tree_result, oracle_result,
						"Insert({}, {}) mismatch", k, v
					);
				}
				Op::Remove(k) => {
					let tree_result = tree.remove(k);
					let oracle_result = oracle.remove(k);
					prop_assert_eq!(
						tree_result, oracle_result,
						"Remove({}) mismatch", k
					);
				}
				Op::Lookup(k) => {
					let tree_result = tree.lookup(k, |v| *v);
					let oracle_result = oracle.get(k).copied();
					prop_assert_eq!(
						tree_result, oracle_result,
						"Lookup({}) mismatch", k
					);
				}
			}
		}

		tree.assert_invariants();

		// Final state should match
		prop_assert_eq!(tree.len(), oracle.len(), "Final length mismatch");

		// All oracle entries should be in tree
		for (k, v) in &oracle {
			let tree_val = tree.lookup(k, |val| *val);
			prop_assert_eq!(tree_val, Some(*v), "Final state mismatch for key {}", k);
		}

		// Iteration order should match
		let mut tree_iter = tree.raw_iter();
		tree_iter.seek_to_first();

		for (oracle_k, oracle_v) in &oracle {
			let (tree_k, tree_v) = tree_iter.next().expect("Tree should have same entries as oracle");
			prop_assert_eq!(tree_k, oracle_k, "Key mismatch during iteration");
			prop_assert_eq!(tree_v, oracle_v, "Value mismatch during iteration");
		}
	}
}

// ===========================================================================
// Edge Case Properties
// ===========================================================================

proptest! {
	/// Property: Empty tree operations are safe
	#[test]
	fn empty_tree_operations(keys in unique_keys(50)) {
		let tree: Tree<i32, i32> = Tree::new();

		prop_assert!(tree.is_empty());
		prop_assert_eq!(tree.len(), 0);
		prop_assert_eq!(tree.height(), 1);

		// Lookups and removes on empty tree return None
		for k in &keys {
			prop_assert_eq!(tree.lookup(k, |v| *v), None);
			prop_assert_eq!(tree.remove(k), None);
		}

		tree.assert_invariants();
	}

	/// Property: Single element operations work correctly
	#[test]
	fn single_element_operations(key in any::<i32>(), value in any::<i32>()) {
		let tree: Tree<i32, i32> = Tree::new();

		tree.insert(key, value);

		prop_assert!(!tree.is_empty());
		prop_assert_eq!(tree.len(), 1);
		prop_assert_eq!(tree.lookup(&key, |v| *v), Some(value));

		tree.assert_invariants();

		// Remove and verify empty
		let removed = tree.remove(&key);
		prop_assert_eq!(removed, Some(value));
		prop_assert!(tree.is_empty());

		tree.assert_invariants();
	}

	/// Property: Boundary keys (MIN/MAX) work correctly
	#[test]
	fn boundary_keys_work(value in any::<i32>()) {
		let tree: Tree<i32, i32> = Tree::new();

		tree.insert(i32::MIN, value);
		tree.insert(i32::MAX, value);
		tree.insert(0, value);

		tree.assert_invariants();

		prop_assert_eq!(tree.lookup(&i32::MIN, |v| *v), Some(value));
		prop_assert_eq!(tree.lookup(&i32::MAX, |v| *v), Some(value));
		prop_assert_eq!(tree.lookup(&0, |v| *v), Some(value));

		// Verify order
		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let k1 = *iter.next().unwrap().0;
		let k2 = *iter.next().unwrap().0;
		let k3 = *iter.next().unwrap().0;

		prop_assert_eq!(k1, i32::MIN);
		prop_assert_eq!(k2, 0);
		prop_assert_eq!(k3, i32::MAX);
	}
}

// ===========================================================================
// Seek Properties
// ===========================================================================

proptest! {
	/// Property: Seek positions correctly for existing keys
	#[test]
	fn seek_finds_existing_keys(entries in key_value_pairs(200)) {
		let tree: Tree<i32, i32> = Tree::new();
		let mut expected: BTreeMap<i32, i32> = BTreeMap::new();

		for (k, v) in &entries {
			tree.insert(*k, *v);
			expected.insert(*k, *v);
		}

		tree.assert_invariants();

		// Seek to each existing key
		for (k, v) in &expected {
			let mut iter = tree.raw_iter();
			iter.seek(k);

			let result = iter.next();
			prop_assert!(result.is_some(), "Seek to existing key {} should find entry", k);

			let (found_k, found_v) = result.unwrap();
			prop_assert_eq!(found_k, k, "Seek should find exact key");
			prop_assert_eq!(found_v, v, "Seek should find correct value");
		}
	}

	/// Property: seek_exact returns correct boolean
	#[test]
	fn seek_exact_correctness(
		existing in unique_keys(100),
		queries in unique_keys(100)
	) {
		let tree: Tree<i32, i32> = Tree::new();

		for k in &existing {
			tree.insert(*k, *k);
		}

		tree.assert_invariants();

		for k in &queries {
			let mut iter = tree.raw_iter();
			let found = iter.seek_exact(k);

			if existing.contains(k) {
				prop_assert!(found, "seek_exact should return true for existing key {}", k);
			} else {
				prop_assert!(!found, "seek_exact should return false for non-existing key {}", k);
			}
		}
	}
}
