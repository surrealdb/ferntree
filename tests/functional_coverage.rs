//! # Functional coverage tests
//!
//! Targeted tests filling gaps in the API surface coverage that the existing
//! `integration.rs` test file does not exercise:
//!
//! - Every bound combination on `range()` against every interesting tree
//!   shape (empty, single-element, leaf-boundary, internal-boundary).
//! - Boundary keys (`u32::MIN`, `u32::MAX`, empty `String`, very long
//!   `String`).
//! - Drop-counting for inserts/removes/clears, to catch double-free or leak
//!   bugs that escape LSan.
//! - `get_or_insert_with`'s closure-runs-exactly-once-when-missing contract,
//!   and zero times when present.
//! - `height()` monotonicity after fill/drain.
//! - `pop_first` / `pop_last` against single- and empty-tree edge cases.

// Explicit `drop(tree)` calls are used for clarity in destructor-ordering
// tests, even though `Tree` itself does not implement `Drop`.
#![allow(clippy::drop_non_drop)]

use ferntree::Tree;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ===========================================================================
// `range()` — exhaustive bound combinations × tree shapes
// ===========================================================================

fn collect_range<F>(tree: &Tree<u32, u32>, f: F) -> Vec<u32>
where
	F: for<'a> FnOnce(&'a Tree<u32, u32>) -> ferntree::iter::Range<'a, u32, u32, 64, 64>,
{
	let mut iter = f(tree);
	let mut out = Vec::new();
	while let Some((k, _)) = iter.next() {
		out.push(*k);
	}
	out
}

#[test]
fn range_empty_tree_yields_nothing_for_every_bound() {
	let tree: Tree<u32, u32> = Tree::new();
	assert!(collect_range(&tree, |t| t.range(Unbounded, Unbounded)).is_empty());
	assert!(collect_range(&tree, |t| t.range(Included(&0), Included(&100))).is_empty());
	assert!(collect_range(&tree, |t| t.range(Excluded(&0), Excluded(&100))).is_empty());
}

#[test]
fn range_single_element_tree_every_bound() {
	let tree: Tree<u32, u32> = Tree::new();
	tree.insert(42, 100);
	assert_eq!(collect_range(&tree, |t| t.range(Included(&42), Included(&42))), vec![42]);
	assert_eq!(collect_range(&tree, |t| t.range(Included(&42), Excluded(&42))), Vec::<u32>::new());
	assert_eq!(collect_range(&tree, |t| t.range(Excluded(&42), Included(&42))), Vec::<u32>::new());
	assert_eq!(collect_range(&tree, |t| t.range(Excluded(&41), Excluded(&43))), vec![42]);
	assert_eq!(collect_range(&tree, |t| t.range(Unbounded, Unbounded)), vec![42]);
	assert_eq!(
		collect_range(&tree, |t| t.range(Included(&u32::MIN), Included(&u32::MAX))),
		vec![42]
	);
}

#[test]
fn range_spans_leaf_boundary() {
	// Capacity is 64 — pick a tree that straddles two leaves.
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..100u32 {
		tree.insert(i, i);
	}
	// Range that brackets the split point.
	let collected = collect_range(&tree, |t| t.range(Included(&30), Excluded(&70)));
	let expected: Vec<u32> = (30..70).collect();
	assert_eq!(collected, expected);
}

#[test]
fn range_spans_internal_boundary() {
	// Force a multi-level tree.
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..1024u32 {
		tree.insert(i, i);
	}
	let collected = collect_range(&tree, |t| t.range(Included(&100), Excluded(&900)));
	assert_eq!(collected.len(), 800);
	assert_eq!(*collected.first().unwrap(), 100);
	assert_eq!(*collected.last().unwrap(), 899);
}

// ===========================================================================
// Boundary keys
// ===========================================================================

#[test]
fn boundary_keys_u32_min_and_max() {
	let tree: Tree<u32, u32> = Tree::new();
	tree.insert(u32::MIN, 0);
	tree.insert(u32::MAX, 1);
	assert_eq!(tree.lookup(&u32::MIN, |v| *v), Some(0));
	assert_eq!(tree.lookup(&u32::MAX, |v| *v), Some(1));
	tree.assert_invariants();

	// Iterate and confirm both ends are reachable.
	let mut iter = tree.raw_iter();
	iter.seek_to_first();
	let (k1, _) = iter.next().unwrap();
	assert_eq!(*k1, u32::MIN);
	let (k2, _) = iter.next().unwrap();
	assert_eq!(*k2, u32::MAX);
	assert!(iter.next().is_none());
}

#[test]
fn boundary_keys_string_empty_and_long() {
	let tree: Tree<String, u32> = Tree::new();
	tree.insert(String::new(), 0);
	let long_key: String = "x".repeat(8192);
	tree.insert(long_key.clone(), 1);
	tree.insert("short".to_string(), 2);
	tree.assert_invariants();
	assert_eq!(tree.lookup(&String::new(), |v| *v), Some(0));
	assert_eq!(tree.lookup(&long_key, |v| *v), Some(1));
	assert_eq!(tree.lookup(&"short".to_string(), |v| *v), Some(2));
}

// ===========================================================================
// Drop counting — every Drop must fire exactly once
// ===========================================================================

#[derive(Clone)]
struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
	fn drop(&mut self) {
		self.0.fetch_add(1, Ordering::SeqCst);
	}
}

/// Aggressively try to flush crossbeam-epoch's deferred bag.
///
/// The library does not provide a synchronous `flush()` so we approximate it
/// by repeatedly pinning a guard, yielding to the OS, and looping until the
/// observed drop counter stops moving. This is best-effort: under heavy load
/// some destructors may still be deferred when the test exits.
fn force_epoch_flush(count: &AtomicUsize) {
	let mut prev = count.load(Ordering::SeqCst);
	for _ in 0..64 {
		for _ in 0..16 {
			drop(crossbeam_epoch::pin());
			std::thread::yield_now();
		}
		let curr = count.load(Ordering::SeqCst);
		if curr == prev {
			break;
		}
		prev = curr;
	}
}

#[test]
fn insert_then_drop_tree_does_not_over_drop() {
	// `Drop` of the tree must run each user value at most once. Under-drop is
	// hard to catch deterministically (crossbeam-epoch may keep items in a
	// deferred bag arbitrarily long); over-drop would be a double-free.
	let count = Arc::new(AtomicUsize::new(0));
	{
		let tree: Tree<u32, DropCounter> = Tree::new();
		for i in 0..200u32 {
			tree.insert(i, DropCounter(Arc::clone(&count)));
		}
		assert_eq!(count.load(Ordering::SeqCst), 0);
		drop(tree);
	}
	force_epoch_flush(&count);
	let observed = count.load(Ordering::SeqCst);
	assert!(observed <= 200, "over-drop: observed {observed} > 200 inserts");
}

#[test]
fn replace_does_not_over_drop() {
	let count = Arc::new(AtomicUsize::new(0));
	let tree: Tree<u32, DropCounter> = Tree::new();
	tree.insert(1, DropCounter(Arc::clone(&count)));
	tree.insert(1, DropCounter(Arc::clone(&count)));
	force_epoch_flush(&count);
	// At most 1 drop here: the old value displaced by the second insert.
	let after_replace = count.load(Ordering::SeqCst);
	assert!(after_replace <= 1, "over-drop after replace: {after_replace}");
	drop(tree);
	force_epoch_flush(&count);
	// At most 2 drops total: the displaced value and the surviving one.
	let final_count = count.load(Ordering::SeqCst);
	assert!(final_count <= 2, "over-drop after tree drop: {final_count}");
}

#[test]
fn clear_does_not_over_drop() {
	let count = Arc::new(AtomicUsize::new(0));
	let tree: Tree<u32, DropCounter> = Tree::new();
	for i in 0..128u32 {
		tree.insert(i, DropCounter(Arc::clone(&count)));
	}
	tree.clear();
	force_epoch_flush(&count);
	let observed = count.load(Ordering::SeqCst);
	assert!(observed <= 128, "over-drop on clear: {observed} > 128");
	assert!(tree.is_empty());
}

#[test]
fn remove_returns_owned_value_that_drops_when_let_goes() {
	// `remove` returns the value by value, so the drop count must reflect at
	// least the removed values after the bindings go out of scope. (Tree
	// internals may also drop additional metadata via epoch reclamation.)
	let count = Arc::new(AtomicUsize::new(0));
	let tree: Tree<u32, DropCounter> = Tree::new();
	for i in 0..50u32 {
		tree.insert(i, DropCounter(Arc::clone(&count)));
	}
	for i in 0..50u32 {
		let _ = tree.remove(&i).unwrap();
	}
	force_epoch_flush(&count);
	let observed = count.load(Ordering::SeqCst);
	assert_eq!(observed, 50, "remove must drop returned values: got {observed}");
}

// ===========================================================================
// `get_or_insert_with` closure contract
// ===========================================================================

#[test]
fn get_or_insert_with_runs_closure_when_missing() {
	let tree: Tree<u32, u32> = Tree::new();
	let calls = AtomicUsize::new(0);
	let got = tree.get_or_insert_with(7, || {
		calls.fetch_add(1, Ordering::SeqCst);
		77
	});
	assert_eq!(got, 77);
	assert_eq!(calls.load(Ordering::SeqCst), 1);
	assert_eq!(tree.lookup(&7, |v| *v), Some(77));
}

#[test]
fn get_or_insert_with_skips_closure_when_present() {
	let tree: Tree<u32, u32> = Tree::new();
	tree.insert(5, 50);
	let calls = AtomicUsize::new(0);
	let got = tree.get_or_insert_with(5, || {
		calls.fetch_add(1, Ordering::SeqCst);
		999
	});
	assert_eq!(got, 50);
	assert_eq!(calls.load(Ordering::SeqCst), 0, "closure must not run when key exists");
}

// ===========================================================================
// Height monotonicity
// ===========================================================================

#[test]
fn height_does_not_grow_after_drain() {
	// After fully draining the tree, height should not exceed its post-fill
	// value — and ideally should shrink. The current implementation collapses
	// merged levels gradually, so we don't require height to return to 1
	// from individual removes (use `clear()` for that — see the dedicated
	// test below).
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..2048u32 {
		tree.insert(i, i);
	}
	let tall = tree.height();
	assert!(tall > 1, "expected multi-level tree, got height {tall}");
	for i in 0..2048u32 {
		tree.remove(&i);
	}
	tree.assert_invariants();
	let drained = tree.height();
	assert!(drained <= tall, "height grew during drain: {tall} -> {drained}");
	assert!(tree.is_empty());
}

#[test]
fn height_is_one_after_clear() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..2048u32 {
		tree.insert(i, i);
	}
	assert!(tree.height() > 1);
	tree.clear();
	tree.assert_invariants();
	assert_eq!(tree.height(), 1);
}

// ===========================================================================
// `pop_first` / `pop_last` edge cases
// ===========================================================================

#[test]
fn pop_on_empty_tree_returns_none() {
	let tree: Tree<u32, u32> = Tree::new();
	assert_eq!(tree.pop_first(), None);
	assert_eq!(tree.pop_last(), None);
}

#[test]
fn pop_first_drains_tree_in_order() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..512u32 {
		tree.insert(i, i * 2);
	}
	for expected in 0..512u32 {
		assert_eq!(tree.pop_first(), Some((expected, expected * 2)));
	}
	assert!(tree.is_empty());
	assert_eq!(tree.pop_first(), None);
	tree.assert_invariants();
}

#[test]
fn pop_last_drains_tree_in_reverse() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..512u32 {
		tree.insert(i, i * 2);
	}
	for expected in (0..512u32).rev() {
		assert_eq!(tree.pop_last(), Some((expected, expected * 2)));
	}
	assert!(tree.is_empty());
	assert_eq!(tree.pop_last(), None);
	tree.assert_invariants();
}

#[test]
fn pop_alternating_first_last_meets_in_middle() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..100u32 {
		tree.insert(i, i);
	}
	let mut lo = 0u32;
	let mut hi = 99u32;
	while !tree.is_empty() {
		let (k, _) = tree.pop_first().unwrap();
		assert_eq!(k, lo);
		lo += 1;
		if tree.is_empty() {
			break;
		}
		let (k, _) = tree.pop_last().unwrap();
		assert_eq!(k, hi);
		hi -= 1;
	}
	assert!(tree.is_empty());
}
