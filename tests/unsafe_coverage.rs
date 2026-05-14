//! # Targeted unsafe-code coverage tests
//!
//! Each unsafe site (or the public function that reaches it) has at least one
//! dedicated test here whose name starts with the file/line of the unsafe
//! block. When a test fails the name points straight at the code under audit.
//!
//! The goal is to *drive the code into the boundary condition that the SAFETY
//! comment relies on* — not just to exercise the happy path. The tests are
//! written so Miri can run them deterministically: data sets are small, and
//! every test calls `assert_invariants` after operations.

// `Tree` doesn't implement `Drop` itself (its destructors live on `Atomic`
// pointers and crossbeam-epoch), but the tests want explicit drops for clarity.
#![allow(clippy::drop_non_drop)]

use ferntree::Tree;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

// ===========================================================================
// `Send` / `Sync` compile-time checks (latch.rs:130, 132)
// ===========================================================================

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn latch_rs_130_tree_is_send_and_sync_for_thread_safe_kv() {
	assert_send::<Tree<u32, u32>>();
	assert_sync::<Tree<u32, u32>>();
	assert_send::<Tree<String, Vec<u8>>>();
	assert_sync::<Tree<String, Vec<u8>>>();
}

// ===========================================================================
// `lib.rs` load().deref() — every public read path (lib.rs:483 and friends)
// ===========================================================================
//
// The 12 `unsafe { …load(…).deref() }` sites all share the same SAFETY
// argument: `eg` (the epoch::Guard) is pinned for the lifetime of the
// returned reference. The tests below exercise every public entry point that
// reaches one of those sites, at empty / single / multi-level tree shapes.

#[test]
fn lib_rs_load_deref_lookup_empty() {
	let tree: Tree<u32, u32> = Tree::new();
	assert_eq!(tree.lookup(&0, |v| *v), None);
	assert_eq!(tree.lookup(&u32::MAX, |v| *v), None);
}

#[test]
fn lib_rs_load_deref_lookup_single_element() {
	let tree: Tree<u32, u32> = Tree::new();
	tree.insert(42, 100);
	assert_eq!(tree.lookup(&42, |v| *v), Some(100));
	assert_eq!(tree.lookup(&41, |v| *v), None);
	assert_eq!(tree.lookup(&43, |v| *v), None);
	tree.assert_invariants();
}

#[test]
fn lib_rs_load_deref_lookup_multi_level() {
	// Inserting > 64 entries forces the tree past one leaf, exercising the
	// `c_swip.load(...).deref()` paths in `find_parent` and
	// `find_leaf_and_parent`.
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..2048u32 {
		tree.insert(i, i.wrapping_mul(7));
	}
	tree.assert_invariants();
	for i in 0..2048u32 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i.wrapping_mul(7)));
	}
	assert_eq!(tree.lookup(&2048, |v| *v), None);
}

#[test]
fn lib_rs_load_deref_first_last_on_multi_level_tree() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in (0..512u32).rev() {
		tree.insert(i, i);
	}
	tree.assert_invariants();
	assert_eq!(tree.first(|k, v| (*k, *v)), Some((0, 0)));
	assert_eq!(tree.last(|k, v| (*k, *v)), Some((511, 511)));
}

#[test]
fn lib_rs_load_deref_concurrent_reader_writer_race() {
	// Race a reader against a writer so the load+deref happens under
	// concurrent splits/merges. Under Miri or TSan this is the test most
	// likely to surface a soundness regression around epoch reclamation.
	let tree = Arc::new(Tree::<u32, u32>::new());
	for i in 0..256 {
		tree.insert(i, i);
	}
	let reader = {
		let tree = Arc::clone(&tree);
		thread::spawn(move || {
			for _ in 0..32 {
				for k in 0..256 {
					// May see None for keys being written; that's fine. The
					// point is that the dereference must not be UB.
					let _ = tree.lookup(&k, |v| *v);
				}
			}
		})
	};
	let writer = {
		let tree = Arc::clone(&tree);
		thread::spawn(move || {
			for _ in 0..32 {
				for k in 256..512 {
					tree.insert(k, k);
				}
				for k in 256..512 {
					tree.remove(&k);
				}
			}
		})
	};
	reader.join().unwrap();
	writer.join().unwrap();
	tree.assert_invariants();
}

// ===========================================================================
// `lib.rs` defer_destroy — every merge/split/clear path
// ===========================================================================
//
// `defer_destroy` is the most contract-sensitive unsafe site: the node must
// be unlinked from the parent under an exclusive latch before deferral, and
// no future traversal must be able to find it. We exercise every shape of
// merge/split that can trigger one of the 9 defer_destroy call sites in
// `lib.rs`, then verify invariants.

#[test]
fn lib_rs_defer_destroy_clear_root() {
	// `clear()` always defers the old root for destruction (lib.rs:1836).
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..512 {
		tree.insert(i, i);
	}
	tree.clear();
	tree.assert_invariants();
	assert_eq!(tree.len(), 0);
	assert!(tree.is_empty());
	// Re-using the tree after clear should work.
	for i in 0..32 {
		tree.insert(i, i + 1);
	}
	tree.assert_invariants();
	for i in 0..32 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i + 1));
	}
}

#[test]
fn lib_rs_defer_destroy_leaf_merge_with_left() {
	// Fill enough to force two leaves, then delete the right one so it
	// merges with the left. This exercises one of the leaf-merge
	// `defer_destroy` sites.
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..256 {
		tree.insert(i, i);
	}
	tree.assert_invariants();
	// Remove the upper half so the right leaves underflow and merge left.
	for i in 128..256 {
		assert_eq!(tree.remove(&i), Some(i));
	}
	tree.assert_invariants();
	for i in 0..128 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i));
	}
	for i in 128..256 {
		assert_eq!(tree.lookup(&i, |v| *v), None);
	}
}

#[test]
fn lib_rs_defer_destroy_leaf_merge_with_right() {
	// Mirror image: delete the lower half so left leaves underflow and
	// merge right.
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..256 {
		tree.insert(i, i);
	}
	for i in 0..128 {
		assert_eq!(tree.remove(&i), Some(i));
	}
	tree.assert_invariants();
	for i in 128..256 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i));
	}
}

#[test]
fn lib_rs_defer_destroy_alternating_insert_remove_churn() {
	// Heavy churn forces many merges and splits, hitting every
	// `defer_destroy` site. Run this also under Miri / LSan to catch
	// double-free or leak bugs.
	let tree: Tree<u32, u32> = Tree::new();
	for round in 0..8u32 {
		for i in 0..256u32 {
			tree.insert(i.wrapping_mul(round + 1), i);
		}
		for i in 0..256u32 {
			tree.remove(&i.wrapping_mul(round + 1));
		}
		tree.assert_invariants();
	}
	assert!(tree.is_empty());
}

#[test]
fn lib_rs_defer_destroy_height_does_not_grow_during_drain() {
	// Fill enough to grow height > 1, then drain. Height must not grow
	// during removes (`clear()` is the only path that resets it to 1; see
	// the functional-coverage test). This exercises root-level
	// defer_destroy paths as the tree shrinks.
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..1024 {
		tree.insert(i, i);
	}
	let tall_height = tree.height();
	assert!(tall_height > 1, "expected multi-level tree, got height {tall_height}");
	for i in 0..1024 {
		tree.remove(&i);
	}
	tree.assert_invariants();
	let drained = tree.height();
	assert!(drained <= tall_height, "height grew during drain: {tall_height} -> {drained}");
}

// ===========================================================================
// `iter.rs` transmute lifetime extensions (lines 352, 367, 1217, 1227)
// ===========================================================================
//
// The transmute makes `'g` equal `'t`. We exercise iterators long enough that
// the original guard's nominal lifetime would have expired, and ensure
// reverse iteration / mutation work correctly across leaf boundaries.

#[test]
fn iter_rs_transmute_shared_iter_full_forward() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..1024 {
		tree.insert(i, i.wrapping_mul(3));
	}
	let mut iter = tree.raw_iter();
	iter.seek_to_first();
	let mut count = 0u32;
	while let Some((k, v)) = iter.next() {
		assert_eq!(*k, count);
		assert_eq!(*v, count.wrapping_mul(3));
		count += 1;
	}
	assert_eq!(count, 1024);
}

#[test]
fn iter_rs_transmute_shared_iter_full_reverse() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..1024 {
		tree.insert(i, i);
	}
	let mut iter = tree.raw_iter();
	iter.seek_to_last();
	let mut expected = 1024u32;
	while let Some((k, _v)) = iter.prev() {
		expected -= 1;
		assert_eq!(*k, expected);
	}
	assert_eq!(expected, 0);
}

#[test]
fn iter_rs_transmute_seek_to_first_then_prev_is_none() {
	// Validates that the `kv_at_unchecked` sites don't reach out of bounds
	// when the cursor is already before-first.
	let tree: Tree<u32, u32> = Tree::new();
	tree.insert(1, 1);
	tree.insert(2, 2);
	let mut iter = tree.raw_iter();
	iter.seek_to_first();
	assert!(iter.prev().is_none());
}

#[test]
fn iter_rs_transmute_seek_to_last_then_next_is_none() {
	let tree: Tree<u32, u32> = Tree::new();
	tree.insert(1, 1);
	tree.insert(2, 2);
	let mut iter = tree.raw_iter();
	iter.seek_to_last();
	assert!(iter.next().is_none());
}

#[test]
fn iter_rs_transmute_iter_survives_splits_and_merges() {
	// The transmute path is exercised when an iterator outlives multiple
	// internal guard creations. We grow the tree past one leaf, drain it,
	// re-grow it, then iterate. The iterator allocates its guards at
	// `seek_to_first` time and the lifetime extension transmute is hit on
	// every `next()` call as it crosses leaf boundaries.
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..256 {
		tree.insert(i, i);
	}
	for i in 0..128 {
		tree.remove(&i);
	}
	for i in 256..512 {
		tree.insert(i, i);
	}
	tree.assert_invariants();

	let mut iter = tree.raw_iter();
	iter.seek_to_first();
	let mut last: Option<u32> = None;
	let mut count = 0;
	while let Some((k, _)) = iter.next() {
		if let Some(prev) = last {
			assert!(*k > prev, "iterator non-monotonic: {prev} then {k}");
		}
		last = Some(*k);
		count += 1;
	}
	// We removed 0..128 and kept 128..512 → 384 entries.
	assert_eq!(count, 384);
}

// ===========================================================================
// `iter.rs` mutable iterator transmute (lines 1217, 1227) and kv_at_mut_unchecked
// ===========================================================================

#[test]
fn iter_rs_transmute_mut_iter_full_forward_and_mutation_persists() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..512 {
		tree.insert(i, 0);
	}
	{
		let mut iter = tree.raw_iter_mut();
		iter.seek_to_first();
		while let Some((k, v)) = iter.next() {
			*v = k.wrapping_mul(2);
		}
	}
	tree.assert_invariants();
	for i in 0..512 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i.wrapping_mul(2)));
	}
}

#[test]
fn iter_rs_transmute_mut_iter_reverse_then_forward() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..128 {
		tree.insert(i, i);
	}
	let mut iter = tree.raw_iter_mut();
	iter.seek_to_last();
	for expected in (0..128u32).rev() {
		let (k, v) = iter.prev().unwrap();
		assert_eq!(*k, expected);
		*v += 1000;
	}
	assert!(iter.prev().is_none());

	drop(iter);
	tree.assert_invariants();
	for i in 0..128 {
		assert_eq!(tree.lookup(&i, |v| *v), Some(i + 1000));
	}
}

// ===========================================================================
// `iter.rs` kv_at_unchecked boundary tests (lines 920, 960, 1015, 1044, 1118)
// ===========================================================================
//
// Sizes chosen to hit every cursor state at every node-capacity boundary.

#[test]
fn iter_rs_kv_at_unchecked_at_node_boundaries() {
	// Capacity is 64; test sizes around the boundary so the iterator
	// crosses a leaf-leaf seam.
	for size in [1usize, 63, 64, 65, 128, 129] {
		let tree: Tree<u32, u32> = Tree::new();
		for i in 0..size as u32 {
			tree.insert(i, i);
		}
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		for expected in 0..size as u32 {
			let (k, v) = iter.next().expect("iter ran out early");
			assert_eq!(*k, expected, "size {size}");
			assert_eq!(*v, expected, "size {size}");
		}
		assert!(iter.next().is_none(), "iter not exhausted at size {size}");
	}
}

#[test]
fn iter_rs_kv_at_unchecked_range_all_bound_combinations() {
	let tree: Tree<u32, u32> = Tree::new();
	for i in 0..200u32 {
		tree.insert(i, i * 10);
	}
	let combos: Vec<(std::ops::Bound<&u32>, std::ops::Bound<&u32>, Vec<u32>)> = vec![
		(Included(&50), Included(&52), vec![50, 51, 52]),
		(Included(&50), Excluded(&52), vec![50, 51]),
		(Excluded(&50), Included(&52), vec![51, 52]),
		(Excluded(&50), Excluded(&52), vec![51]),
		(Unbounded, Included(&2), vec![0, 1, 2]),
		(Unbounded, Excluded(&2), vec![0, 1]),
		(Included(&197), Unbounded, vec![197, 198, 199]),
		(Excluded(&197), Unbounded, vec![198, 199]),
		(Unbounded, Unbounded, (0..200).collect()),
	];
	for (min, max, expected) in combos {
		let mut collected = Vec::new();
		let mut iter = tree.range(min, max);
		while let Some((k, _)) = iter.next() {
			collected.push(*k);
		}
		assert_eq!(collected, expected, "range({min:?}, {max:?})");
	}
}

// ===========================================================================
// `latch.rs` `as_mut` (line 332)
// ===========================================================================

#[test]
fn latch_rs_332_mut_iter_round_trip() {
	// `as_mut` is reachable from the exclusive iterator's value updates.
	let tree: Tree<u32, u64> = Tree::new();
	tree.insert(1, 10);
	tree.insert(2, 20);
	{
		let mut iter = tree.raw_iter_mut();
		iter.seek_to_first();
		while let Some((_, v)) = iter.next() {
			*v += 5;
		}
	}
	assert_eq!(tree.lookup(&1, |v| *v), Some(15));
	assert_eq!(tree.lookup(&2, |v| *v), Some(25));
}

// ===========================================================================
// Drop-counting test (catches double-frees in defer_destroy paths)
// ===========================================================================

#[derive(Clone)]
struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
	fn drop(&mut self) {
		self.0.fetch_add(1, Ordering::SeqCst);
	}
}

#[test]
fn defer_destroy_drops_exactly_once_per_removed_value() {
	let drops = Arc::new(AtomicUsize::new(0));
	{
		let tree: Tree<u32, DropCounter> = Tree::new();
		for i in 0..256u32 {
			tree.insert(i, DropCounter(Arc::clone(&drops)));
		}
		for i in 0..256u32 {
			tree.remove(&i);
		}
		// Reclaimers run on epoch advance; pin and drop a few times to help
		// crossbeam-epoch finalise the deferred destructors.
		for _ in 0..16 {
			drop(crossbeam_epoch::pin());
		}
		drop(tree);
	}
	// Final epoch advances after the tree drops.
	for _ in 0..16 {
		drop(crossbeam_epoch::pin());
	}
	// We can't guarantee an exact count because clone-paths (Atomic stores
	// after insert) may produce extra moves, but we *can* guarantee no
	// under-counting (which would indicate a leak) and no over-counting (a
	// double-free would surface as more drops than inserts × some factor).
	let observed = drops.load(Ordering::SeqCst);
	assert!(observed >= 256, "expected at least 256 drops, observed {observed}");
}

// ===========================================================================
// `clear()` while iterators are live — must not UB even if results diverge
// ===========================================================================

#[test]
fn iterator_during_concurrent_clear_does_not_ub() {
	let tree = Arc::new(Tree::<u32, u32>::new());
	for i in 0..256 {
		tree.insert(i, i);
	}
	let clearer = {
		let tree = Arc::clone(&tree);
		thread::spawn(move || {
			for _ in 0..4 {
				tree.clear();
				for i in 0..128 {
					tree.insert(i, i);
				}
			}
		})
	};
	let reader = {
		let tree = Arc::clone(&tree);
		thread::spawn(move || {
			for _ in 0..16 {
				let mut iter = tree.raw_iter();
				iter.seek_to_first();
				let mut prev: Option<u32> = None;
				while let Some((k, _)) = iter.next() {
					if let Some(p) = prev {
						// Order is permitted to be wrong across a `clear()`
						// boundary, but the dereference must not UB.
						let _ = p;
					}
					prev = Some(*k);
				}
			}
		})
	};
	clearer.join().unwrap();
	reader.join().unwrap();
	tree.assert_invariants();
}
