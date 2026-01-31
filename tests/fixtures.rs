//! # Fixture-Based Tests for Ferntree B+ Tree
//!
//! This module contains tests that verify tree behavior with pre-defined
//! structures similar to what JSON fixtures would provide.
//!
//! Since the `sample_tree` utility is only available in the crate's internal
//! tests, these tests create equivalent tree structures through the public API.

use ferntree::Tree;

// ===========================================================================
// Tests Mirroring sample.json Fixture Structure
// ===========================================================================

/// Creates a tree structure similar to fixtures/sample.json
///
/// The sample.json represents a tree with height 3:
/// - Root internal node with separator "0003"
/// - Two internal children
/// - Four leaf nodes with keys: "0002", "0003", "0005", and empty leaf
fn create_sample_tree_structure() -> Tree<String, u64> {
	let tree: Tree<String, u64> = Tree::new();

	// Insert keys in the order that would create a similar structure
	tree.insert("0002".to_string(), 2);
	tree.insert("0003".to_string(), 3);
	tree.insert("0005".to_string(), 5);

	tree
}

#[test]
fn sample_tree_lookup() {
	let tree = create_sample_tree_structure();

	// Verify the entries from sample.json
	assert_eq!(tree.lookup(&"0002".to_string(), |v| *v), Some(2));
	assert_eq!(tree.lookup(&"0003".to_string(), |v| *v), Some(3));
	assert_eq!(tree.lookup(&"0005".to_string(), |v| *v), Some(5));

	// Verify non-existent keys
	assert_eq!(tree.lookup(&"0001".to_string(), |v| *v), None);
	assert_eq!(tree.lookup(&"0004".to_string(), |v| *v), None);
	assert_eq!(tree.lookup(&"0006".to_string(), |v| *v), None);
}

#[test]
fn sample_tree_iteration() {
	let tree = create_sample_tree_structure();

	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let (k, v) = iter.next().unwrap();
	assert_eq!(k, "0002");
	assert_eq!(*v, 2);

	let (k, v) = iter.next().unwrap();
	assert_eq!(k, "0003");
	assert_eq!(*v, 3);

	let (k, v) = iter.next().unwrap();
	assert_eq!(k, "0005");
	assert_eq!(*v, 5);

	assert!(iter.next().is_none());
}

#[test]
fn sample_tree_insert_new_key() {
	let tree = create_sample_tree_structure();

	// Insert a new key
	tree.insert("0004".to_string(), 4);

	// Verify all keys including the new one
	assert_eq!(tree.lookup(&"0002".to_string(), |v| *v), Some(2));
	assert_eq!(tree.lookup(&"0003".to_string(), |v| *v), Some(3));
	assert_eq!(tree.lookup(&"0004".to_string(), |v| *v), Some(4));
	assert_eq!(tree.lookup(&"0005".to_string(), |v| *v), Some(5));

	// Verify order
	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let keys: Vec<String> = std::iter::from_fn(|| iter.next().map(|(k, _)| k.clone())).collect();
	assert_eq!(keys, vec!["0002", "0003", "0004", "0005"]);
}

#[test]
fn sample_tree_remove_key() {
	let tree = create_sample_tree_structure();

	// Remove a key
	let removed = tree.remove(&"0003".to_string());
	assert_eq!(removed, Some(3));

	// Verify remaining keys
	assert_eq!(tree.lookup(&"0002".to_string(), |v| *v), Some(2));
	assert_eq!(tree.lookup(&"0003".to_string(), |v| *v), None);
	assert_eq!(tree.lookup(&"0005".to_string(), |v| *v), Some(5));
}

// ===========================================================================
// Multi-Level Tree Fixture Tests
// ===========================================================================

/// Creates a multi-level tree with a predictable structure
fn create_multilevel_tree() -> Tree<i32, i32> {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert enough entries to create multiple levels
	// With default capacity 64, we need >64 entries for level 2
	for i in 0..200 {
		tree.insert(i, i * 10);
	}

	tree
}

#[test]
fn multilevel_tree_has_correct_height() {
	let tree = create_multilevel_tree();
	assert!(tree.height() >= 2, "Tree should have at least 2 levels");
}

#[test]
fn multilevel_tree_all_entries_accessible() {
	let tree = create_multilevel_tree();

	for i in 0..200 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i * 10), "Entry {} should be accessible", i);
	}
}

#[test]
fn multilevel_tree_iteration_order() {
	let tree = create_multilevel_tree();

	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let mut prev = -1;
	let mut count = 0;

	while let Some((k, v)) = iter.next() {
		assert!(*k > prev, "Keys should be in ascending order");
		assert_eq!(*v, *k * 10, "Value should be key * 10");
		prev = *k;
		count += 1;
	}

	assert_eq!(count, 200);
}

#[test]
fn multilevel_tree_seek_operations() {
	let tree = create_multilevel_tree();

	// Seek to various positions
	let mut iter = tree.raw_iter();

	iter.seek(&50);
	assert_eq!(*iter.next().unwrap().0, 50);

	iter.seek(&100);
	assert_eq!(*iter.next().unwrap().0, 100);

	iter.seek(&150);
	assert_eq!(*iter.next().unwrap().0, 150);

	// Seek to non-existent key (between 100 and 101)
	iter.seek(&100);
	iter.next(); // consume 100
	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, 101);
}

// ===========================================================================
// Sparse Tree Fixture Tests
// ===========================================================================

/// Creates a tree with sparse key distribution
fn create_sparse_tree() -> Tree<i32, String> {
	let tree: Tree<i32, String> = Tree::new();

	// Insert keys with large gaps
	let sparse_keys = [0, 100, 200, 500, 1000, 2000, 5000, 10000];

	for k in sparse_keys {
		tree.insert(k, format!("value_{}", k));
	}

	tree
}

#[test]
fn sparse_tree_lookup() {
	let tree = create_sparse_tree();

	assert_eq!(tree.lookup(&0, |v| v.clone()), Some("value_0".to_string()));
	assert_eq!(tree.lookup(&1000, |v| v.clone()), Some("value_1000".to_string()));
	assert_eq!(tree.lookup(&10000, |v| v.clone()), Some("value_10000".to_string()));

	// Non-existent keys in gaps
	assert_eq!(tree.lookup(&50, |v| v.clone()), None);
	assert_eq!(tree.lookup(&750, |v| v.clone()), None);
}

#[test]
fn sparse_tree_seek() {
	let tree = create_sparse_tree();

	let mut iter = tree.raw_iter();

	// Seek to key in gap - should position at next key
	iter.seek(&50);
	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, 100);

	// Seek to key in gap
	iter.seek(&750);
	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, 1000);
}

#[test]
fn sparse_tree_range_query() {
	let tree = create_sparse_tree();

	let mut iter = tree.raw_iter();
	iter.seek(&100);

	let mut collected = Vec::new();
	while let Some((k, _)) = iter.next() {
		if *k > 2000 {
			break;
		}
		collected.push(*k);
	}

	assert_eq!(collected, vec![100, 200, 500, 1000, 2000]);
}

// ===========================================================================
// Deep Tree Fixture Tests
// ===========================================================================

/// Creates a deep tree by inserting many sequential keys
fn create_deep_tree() -> Tree<i32, i32> {
	let tree: Tree<i32, i32> = Tree::new();

	// Insert many entries to create a deeper tree
	for i in 0..1000 {
		tree.insert(i, i);
	}

	tree
}

#[test]
fn deep_tree_height() {
	let tree = create_deep_tree();
	let height = tree.height();

	// With 1000 entries and capacity 64, we expect at least 2 levels
	assert!(height >= 2, "Height {} should be >= 2", height);
}

#[test]
fn deep_tree_first_and_last() {
	let tree = create_deep_tree();

	let mut iter = tree.raw_iter();

	iter.seek_to_first();
	let (k, _) = iter.next().unwrap();
	assert_eq!(*k, 0);

	iter.seek_to_last();
	let (k, _) = iter.prev().unwrap();
	assert_eq!(*k, 999);
}

#[test]
fn deep_tree_middle_access() {
	let tree = create_deep_tree();

	// Access keys in the middle of the tree
	for i in (400..600).step_by(10) {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i));
	}
}

// ===========================================================================
// String Key Fixture Tests
// ===========================================================================

/// Creates a tree with string keys (similar to sample.json format)
fn create_string_key_tree() -> Tree<String, u64> {
	let tree: Tree<String, u64> = Tree::new();

	// Insert keys with zero-padded format like sample.json
	for i in 0..100u64 {
		let key = format!("{:04}", i);
		tree.insert(key, i);
	}

	tree
}

#[test]
fn string_key_tree_lookup() {
	let tree = create_string_key_tree();

	assert_eq!(tree.lookup(&"0000".to_string(), |v| *v), Some(0));
	assert_eq!(tree.lookup(&"0050".to_string(), |v| *v), Some(50));
	assert_eq!(tree.lookup(&"0099".to_string(), |v| *v), Some(99));
}

#[test]
fn string_key_tree_lexicographic_order() {
	let tree = create_string_key_tree();

	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let mut prev = String::new();
	while let Some((k, _)) = iter.next() {
		if !prev.is_empty() {
			assert!(k > &prev, "Keys should be in lexicographic order: {} > {}", k, prev);
		}
		prev = k.clone();
	}
}

#[test]
fn string_key_seek_with_str_ref() {
	let tree = create_string_key_tree();

	let mut iter = tree.raw_iter();

	// Seek using &str reference (not String)
	iter.seek("0050");
	let (k, v) = iter.next().unwrap();
	assert_eq!(k, "0050");
	assert_eq!(*v, 50);
}

// ===========================================================================
// Edge Case Fixture Tests
// ===========================================================================

/// Creates an empty tree (minimal fixture)
fn create_empty_tree() -> Tree<i32, i32> {
	Tree::new()
}

#[test]
fn empty_tree_operations() {
	let tree = create_empty_tree();

	assert!(tree.is_empty());
	assert_eq!(tree.len(), 0);
	assert_eq!(tree.height(), 1);
	assert_eq!(tree.lookup(&42, |v| *v), None);

	let mut iter = tree.raw_iter();
	iter.seek_to_first();
	assert!(iter.next().is_none());
}

/// Creates a single-entry tree
fn create_single_entry_tree() -> Tree<i32, i32> {
	let tree: Tree<i32, i32> = Tree::new();
	tree.insert(42, 420);
	tree
}

#[test]
fn single_entry_tree_operations() {
	let tree = create_single_entry_tree();

	assert!(!tree.is_empty());
	assert_eq!(tree.len(), 1);
	assert_eq!(tree.height(), 1);
	assert_eq!(tree.lookup(&42, |v| *v), Some(420));

	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let (k, v) = iter.next().unwrap();
	assert_eq!(*k, 42);
	assert_eq!(*v, 420);
	assert!(iter.next().is_none());
}

// ===========================================================================
// Boundary Key Fixture Tests
// ===========================================================================

/// Creates a tree with boundary keys
fn create_boundary_tree() -> Tree<i32, i32> {
	let tree: Tree<i32, i32> = Tree::new();

	tree.insert(i32::MIN, 1);
	tree.insert(-1, 2);
	tree.insert(0, 3);
	tree.insert(1, 4);
	tree.insert(i32::MAX, 5);

	tree
}

#[test]
fn boundary_tree_lookup() {
	let tree = create_boundary_tree();

	assert_eq!(tree.lookup(&i32::MIN, |v| *v), Some(1));
	assert_eq!(tree.lookup(&-1, |v| *v), Some(2));
	assert_eq!(tree.lookup(&0, |v| *v), Some(3));
	assert_eq!(tree.lookup(&1, |v| *v), Some(4));
	assert_eq!(tree.lookup(&i32::MAX, |v| *v), Some(5));
}

#[test]
fn boundary_tree_order() {
	let tree = create_boundary_tree();

	let mut iter = tree.raw_iter();
	iter.seek_to_first();

	let expected = [i32::MIN, -1, 0, 1, i32::MAX];
	for expected_key in expected {
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, expected_key);
	}
	assert!(iter.next().is_none());
}
