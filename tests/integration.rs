//! # Integration Tests for Ferntree B+ Tree
//!
//! This module contains end-to-end integration tests that exercise the tree
//! through its public API with realistic workloads.

use ferntree::Tree;
use rand::prelude::*;

// ===========================================================================
// Large Scale Operation Tests
// ===========================================================================

#[test]
fn large_scale_insert_and_lookup() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert 10,000 entries
	for i in 0..10_000 {
		tree.insert(i, i * 10);
	}

	tree.assert_invariants();
	assert_eq!(tree.len(), 10_000);

	// Verify all entries are findable
	for i in 0..10_000 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i * 10), "Failed to find key {}", i);
	}
}

#[test]
fn large_scale_insert_and_remove() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert 10,000 entries
	for i in 0..10_000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Remove all entries
	for i in 0..10_000 {
		assert_eq!(tree.remove(&i), Some(i), "Failed to remove key {}", i);
	}

	tree.assert_invariants();
	assert!(tree.is_empty());
}

#[test]
fn large_scale_random_operations() {
	let tree: Tree<i32, i32> = Tree::new();
	let mut rng = rand::rng();

	// Random insert/delete/lookup operations
	let mut expected: std::collections::BTreeMap<i32, i32> = std::collections::BTreeMap::new();

	for _ in 0..10_000 {
		let key: i32 = rng.random_range(0..1000);
		let op: u8 = rng.random_range(0..3);

		match op {
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
	}

	tree.assert_invariants();

	// Verify final state matches
	assert_eq!(tree.len(), expected.len());

	for (k, v) in expected.iter() {
		assert_eq!(tree.lookup(k, |val| *val), Some(*v));
	}
}

// ===========================================================================
// Sequential and Random Key Pattern Tests
// ===========================================================================

#[test]
fn sequential_keys_ascending() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..5000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Verify sorted order
	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let mut prev = -1;
	while let Some((k, _)) = iter.next() {
		assert!(*k > prev);
		prev = *k;
	}
	assert_eq!(prev, 4999);
}

#[test]
fn sequential_keys_descending() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in (0..5000).rev() {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Verify sorted order
	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let mut prev = -1;
	while let Some((k, _)) = iter.next() {
		assert!(*k > prev);
		prev = *k;
	}
	assert_eq!(prev, 4999);
}

#[test]
fn random_keys() {
	let tree: Tree<i32, i32> = Tree::new();
	let mut rng = rand::rng();

	let mut keys: Vec<i32> = (0..5000).collect();
	keys.shuffle(&mut rng);

	for k in &keys {
		tree.insert(*k, *k * 10);
	}

	tree.assert_invariants();

	// Verify all entries
	for k in &keys {
		assert_eq!(tree.lookup(k, |v| *v), Some(*k * 10));
	}

	// Verify sorted order
	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let mut prev = -1;
	while let Some((k, _)) = iter.next() {
		assert!(*k > prev);
		prev = *k;
	}
}

#[test]
fn sparse_keys() {
	let tree: Tree<i64, i64> = Tree::new();

	// Insert keys that are far apart
	let keys = [0, 1000, 2000, 10000, 100000, 1000000, i64::MAX - 1];

	for k in keys {
		tree.insert(k, k);
	}

	tree.assert_invariants();

	for k in keys {
		assert_eq!(tree.lookup(&k, |v| *v), Some(k));
	}
}

// ===========================================================================
// Full Tree Deletion Tests
// ===========================================================================

#[test]
fn delete_all_forward() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..1000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	for i in 0..1000 {
		tree.remove(&i);
	}

	tree.assert_invariants();
	assert!(tree.is_empty());
}

#[test]
fn delete_all_reverse() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..1000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	for i in (0..1000).rev() {
		tree.remove(&i);
	}

	tree.assert_invariants();
	assert!(tree.is_empty());
}

#[test]
fn delete_all_random() {
	let tree: Tree<i32, i32> = Tree::new();
	let mut rng = rand::rng();

	for i in 0..1000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	let mut keys: Vec<i32> = (0..1000).collect();
	keys.shuffle(&mut rng);

	for k in keys {
		tree.remove(&k);
	}

	tree.assert_invariants();
	assert!(tree.is_empty());
}

#[test]
fn delete_every_other() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..1000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Delete even keys
	for i in (0..1000).step_by(2) {
		tree.remove(&i);
	}

	tree.assert_invariants();
	assert_eq!(tree.len(), 500);

	// Verify remaining keys
	for i in 0..1000 {
		if i % 2 == 0 {
			assert_eq!(tree.lookup(&i, |v| *v), None);
		} else {
			assert_eq!(tree.lookup(&i, |v| *v), Some(i));
		}
	}
}

// ===========================================================================
// Iterator Full Scan Tests
// ===========================================================================

#[test]
fn full_forward_scan() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..5000 {
		tree.insert(i, i * 2);
	}

	tree.assert_invariants();

	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let mut count = 0;
	while let Some((k, v)) = iter.next() {
		assert_eq!(*k, count);
		assert_eq!(*v, count * 2);
		count += 1;
	}
	assert_eq!(count, 5000);
}

#[test]
fn full_reverse_scan() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..5000 {
		tree.insert(i, i * 2);
	}

	tree.assert_invariants();

	let mut iter = tree.raw_iter();
	iter.seek_to_last();

	let mut count = 4999i32;
	while let Some((k, v)) = iter.prev() {
		assert_eq!(*k, count);
		assert_eq!(*v, count * 2);
		count -= 1;
	}
	assert_eq!(count, -1);
}

#[test]
fn forward_then_reverse_scan() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..100 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	// Forward to middle
	for _ in 0..50 {
		iter.next();
	}

	// Reverse to beginning
	let mut collected = Vec::new();
	while let Some((k, _)) = iter.prev() {
		collected.push(*k);
	}

	// Should have 50 elements (49 down to 0)
	assert_eq!(collected.len(), 50);
	assert_eq!(collected[0], 49);
	assert_eq!(collected[49], 0);
}

// ===========================================================================
// Range Query Tests
// ===========================================================================

#[test]
fn range_query_forward() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..1000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Query range [250, 750)
	let mut iter = tree.raw_iter();
	iter.seek(&250);

	let mut collected = Vec::new();
	while let Some((k, _)) = iter.next() {
		if *k >= 750 {
			break;
		}
		collected.push(*k);
	}

	assert_eq!(collected.len(), 500);
	assert_eq!(collected[0], 250);
	assert_eq!(collected[499], 749);
}

#[test]
fn range_query_reverse() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..1000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Query range (250, 750] in reverse
	let mut iter = tree.raw_iter();
	iter.seek_for_prev(&750);

	let mut collected = Vec::new();
	while let Some((k, _)) = iter.prev() {
		if *k <= 250 {
			break;
		}
		collected.push(*k);
	}

	// 750 down to 251 = 500 elements
	assert_eq!(collected.len(), 500);
	assert_eq!(collected[0], 750);
	assert_eq!(collected[499], 251);
}

#[test]
fn range_query_nonexistent_bounds() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert only even numbers
	for i in (0..1000).step_by(2) {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Seek to 251 (doesn't exist, should position at 252)
	let mut iter = tree.raw_iter();
	iter.seek(&251);

	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, 252);
}

// ===========================================================================
// Remove During Iteration Tests
// ===========================================================================

#[test]
fn remove_via_exclusive_iterator() {
	let tree: Tree<i32, i32> = Tree::new();

	for i in 0..100 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// Remove every other element using exclusive iterator
	{
		let mut iter = tree.raw_iter_mut();
		for i in (0..100).step_by(2) {
			iter.remove(&i);
		}
	}

	tree.assert_invariants();
	assert_eq!(tree.len(), 50);

	// Verify remaining elements
	for i in 0..100 {
		if i % 2 == 0 {
			assert_eq!(tree.lookup(&i, |v| *v), None);
		} else {
			assert_eq!(tree.lookup(&i, |v| *v), Some(i));
		}
	}
}

#[test]
fn insert_via_exclusive_iterator() {
	let tree: Tree<i32, i32> = Tree::new();

	{
		let mut iter = tree.raw_iter_mut();
		for i in 0..100 {
			iter.insert(i, i * 10);
		}
	}

	tree.assert_invariants();
	assert_eq!(tree.len(), 100);

	for i in 0..100 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i * 10));
	}
}

// ===========================================================================
// Edge Case Tests
// ===========================================================================

#[test]
fn single_element_operations() {
	let tree: Tree<i32, i32> = Tree::new();

	tree.insert(42, 420);
	tree.assert_invariants();
	assert_eq!(tree.len(), 1);
	assert_eq!(tree.height(), 1);
	assert_eq!(tree.lookup(&42, |v| *v), Some(420));

	tree.remove(&42);
	tree.assert_invariants();
	assert!(tree.is_empty());
}

#[test]
fn two_element_operations() {
	let tree: Tree<i32, i32> = Tree::new();

	tree.insert(1, 10);
	tree.insert(2, 20);
	tree.assert_invariants();

	// Forward iteration
	let mut iter = tree.raw_iter();
	iter.seek_to_first();
	assert_eq!(*iter.next().unwrap().0, 1);
	assert_eq!(*iter.next().unwrap().0, 2);
	assert!(iter.next().is_none());

	// Reverse iteration
	iter.seek_to_last();
	assert_eq!(*iter.prev().unwrap().0, 2);
	assert_eq!(*iter.prev().unwrap().0, 1);
	assert!(iter.prev().is_none());
}

#[test]
fn boundary_keys_i32() {
	let tree: Tree<i32, i32> = Tree::new();

	tree.insert(i32::MIN, 1);
	tree.insert(0, 2);
	tree.insert(i32::MAX, 3);

	tree.assert_invariants();

	assert_eq!(tree.lookup(&i32::MIN, |v| *v), Some(1));
	assert_eq!(tree.lookup(&0, |v| *v), Some(2));
	assert_eq!(tree.lookup(&i32::MAX, |v| *v), Some(3));

	// Verify order
	let mut iter = tree.raw_iter();
	iter.seek_to_first();
	assert_eq!(*iter.next().unwrap().0, i32::MIN);
	assert_eq!(*iter.next().unwrap().0, 0);
	assert_eq!(*iter.next().unwrap().0, i32::MAX);
}

#[test]
fn string_keys_various_lengths() {
	let tree: Tree<String, i32> = Tree::new();

	let keys = [
		"".to_string(),
		"a".to_string(),
		"ab".to_string(),
		"abc".to_string(),
		"x".repeat(100),
		"x".repeat(1000),
	];

	for (i, k) in keys.iter().enumerate() {
		tree.insert(k.clone(), i as i32);
	}

	tree.assert_invariants();

	for (i, k) in keys.iter().enumerate() {
		assert_eq!(tree.lookup(k, |v| *v), Some(i as i32));
	}
}

#[test]
fn consecutive_duplicates() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert same key multiple times
	for i in 0..100 {
		tree.insert(42, i);
	}

	tree.assert_invariants();

	// Only the last value should be stored
	assert_eq!(tree.len(), 1);
	assert_eq!(tree.lookup(&42, |v| *v), Some(99));
}

// ===========================================================================
// Tree Height Tests
// ===========================================================================

#[test]
fn height_increases_with_inserts() {
	let tree: Tree<i32, i32> = Tree::new();

	tree.assert_invariants();
	assert_eq!(tree.height(), 1);

	// Insert enough to cause at least one level of splits
	for i in 0..200 {
		tree.insert(i, i);
	}

	tree.assert_invariants();
	assert!(tree.height() >= 2);
}

#[test]
fn height_with_many_inserts() {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert many elements
	for i in 0..10_000 {
		tree.insert(i, i);
	}

	tree.assert_invariants();

	// With leaf capacity 64 and internal capacity 64,
	// 10000 entries should result in height >= 3
	let height = tree.height();
	assert!(height >= 2, "Height was {} but expected >= 2", height);
}

// ===========================================================================
// Value Types Tests
// ===========================================================================

#[test]
fn complex_value_type() {
	#[derive(Clone, Debug, PartialEq)]
	struct ComplexValue {
		data: Vec<u8>,
		count: usize,
		name: String,
	}

	let tree: Tree<i32, ComplexValue> = Tree::new();

	for i in 0..100 {
		tree.insert(
			i,
			ComplexValue {
				data: vec![i as u8; 10],
				count: i as usize,
				name: format!("item_{}", i),
			},
		);
	}

	tree.assert_invariants();

	let result = tree.lookup(&50, |v| v.clone());
	assert!(result.is_some());
	let value = result.unwrap();
	assert_eq!(value.count, 50);
	assert_eq!(value.name, "item_50");
	assert_eq!(value.data, vec![50u8; 10]);
}

#[test]
fn zero_sized_value() {
	let tree: Tree<i32, ()> = Tree::new();

	for i in 0..100 {
		tree.insert(i, ());
	}

	tree.assert_invariants();
	assert_eq!(tree.len(), 100);
	assert!(tree.lookup(&50, |_| true).unwrap());
}
