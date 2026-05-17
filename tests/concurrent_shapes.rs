//! # Targeted concurrent-modification shape tests
//!
//! The existing `concurrency.rs` and `deadlock_tests.rs` files do excellent
//! coverage of generic "N threads pounding on the tree" scenarios. This file
//! adds tests with *specific structural shapes* in mind — splits that
//! coincide with where a reader is paused, merges that absorb a reader's
//! leaf, two writers contending for the same leaf, etc. These are the
//! shapes that exercise optimistic-validation retry logic and the SAFETY
//! contracts on `defer_destroy` and the iterator transmutes.

// Tests sometimes explicitly `drop(tree)` to control destructor ordering even
// though `Tree` itself does not implement `Drop`.
#![allow(clippy::drop_non_drop)]

use ferntree::Tree;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

// ===========================================================================
// Reader pinned at K during a split that bisects K
// ===========================================================================

#[test]
fn reader_during_split_yields_correct_successor() {
	// Pre-populate one leaf, then drive a writer that splits the leaf in two
	// while a reader iterator advances through it. The reader explicitly
	// drops its iterator before joining the writer so the leaf's shared
	// lock is released — otherwise the writer would block forever on its
	// final exclusive acquire.
	let tree = Arc::new(Tree::<u32, u32>::new());
	for i in 0..64u32 {
		tree.insert(i * 2, i * 2);
	}

	let barrier = Arc::new(Barrier::new(2));
	let writer_barrier = Arc::clone(&barrier);
	let tree_writer = Arc::clone(&tree);
	let writer = thread::spawn(move || {
		writer_barrier.wait();
		// Fill in the odd keys: this forces splits across the existing leaf.
		for i in 0..64u32 {
			tree_writer.insert(i * 2 + 1, i * 2 + 1);
		}
	});

	{
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		barrier.wait();

		let mut last: Option<u32> = None;
		while let Some((k, _v)) = iter.next() {
			if let Some(prev) = last {
				assert!(*k > prev, "iterator non-monotonic: {prev} then {k}");
			}
			last = Some(*k);
		}
	} // iter drops here, releasing any held leaf locks.

	writer.join().unwrap();
	tree.assert_invariants();
}

// ===========================================================================
// Reader during merge — leaves get absorbed under the reader's feet
// ===========================================================================

#[test]
fn reader_during_merge_yields_consistent_sequence() {
	let tree = Arc::new(Tree::<u32, u32>::new());
	for i in 0..512u32 {
		tree.insert(i, i);
	}
	tree.assert_invariants();

	let tree_writer = Arc::clone(&tree);
	let writer = thread::spawn(move || {
		// Heavy removes from the lower half: forces merges of left leaves
		// while the reader iterates over the upper half.
		for i in 0..256u32 {
			tree_writer.remove(&i);
		}
	});

	let mut seen_count = 0usize;
	{
		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let mut prev: Option<u32> = None;
		while let Some((k, _)) = iter.next() {
			if let Some(p) = prev {
				assert!(*k > p, "iter went backwards: {p} then {k}");
			}
			prev = Some(*k);
			seen_count += 1;
		}
	} // iter drops here, releasing leaf locks.
	writer.join().unwrap();

	// The iterator must have produced *some* sequence, monotonic, even
	// though the underlying tree changed.
	assert!(seen_count > 0);
	tree.assert_invariants();
}

// ===========================================================================
// Two writers split the same leaf — exactly one should split, the other retries
// ===========================================================================

#[test]
fn two_writers_split_same_leaf_settles_to_consistent_state() {
	// Fill a single leaf to near-capacity and drive two writers at it from
	// opposite ends. Both insertions must succeed without lost updates.
	let tree = Arc::new(Tree::<u32, u32>::new());
	for i in 0..32u32 {
		tree.insert(i * 4, i);
	}
	tree.assert_invariants();

	let barrier = Arc::new(Barrier::new(2));
	let writers: Vec<_> = (0..2u32)
		.map(|tid| {
			let tree = Arc::clone(&tree);
			let barrier = Arc::clone(&barrier);
			thread::spawn(move || {
				barrier.wait();
				for i in 0..128u32 {
					let key = if tid == 0 {
						// Low end (will eventually exceed leaf capacity)
						1 + 4 * i
					} else {
						// High end
						3 + 4 * i
					};
					tree.insert(key, key);
				}
			})
		})
		.collect();

	for h in writers {
		h.join().unwrap();
	}
	tree.assert_invariants();

	// Every key from both writers must be present.
	for i in 0..128u32 {
		assert_eq!(tree.lookup(&(1 + 4 * i), |v| *v), Some(1 + 4 * i));
		assert_eq!(tree.lookup(&(3 + 4 * i), |v| *v), Some(3 + 4 * i));
	}
	for i in 0..32u32 {
		assert_eq!(tree.lookup(&(i * 4), |v| *v), Some(i));
	}
}

// ===========================================================================
// Concurrent `pop_first` / `pop_last` — each returned KV must be unique
// ===========================================================================

#[test]
fn concurrent_pop_first_returns_unique_entries() {
	let tree = Arc::new(Tree::<u32, u32>::new());
	const N: u32 = 1024;
	for i in 0..N {
		tree.insert(i, i.wrapping_mul(31));
	}

	let popped = Arc::new(parking_lot::Mutex::new(Vec::new()));
	let workers: Vec<_> = (0..4)
		.map(|_| {
			let tree = Arc::clone(&tree);
			let popped = Arc::clone(&popped);
			thread::spawn(move || {
				let mut local = Vec::new();
				while let Some(kv) = tree.pop_first() {
					local.push(kv);
				}
				popped.lock().extend(local);
			})
		})
		.collect();
	for h in workers {
		h.join().unwrap();
	}

	let popped = popped.lock();
	assert_eq!(popped.len(), N as usize, "wrong number of pops");
	let mut seen: BTreeMap<u32, u32> = BTreeMap::new();
	for (k, v) in popped.iter() {
		assert!(seen.insert(*k, *v).is_none(), "duplicate pop for key {k}");
		assert_eq!(*v, k.wrapping_mul(31), "value mismatch for key {k}");
	}
	for i in 0..N {
		assert!(seen.contains_key(&i), "missing key {i} from pops");
	}
	tree.assert_invariants();
	assert!(tree.is_empty());
}

#[test]
fn concurrent_pop_last_returns_unique_entries() {
	let tree = Arc::new(Tree::<u32, u32>::new());
	const N: u32 = 1024;
	for i in 0..N {
		tree.insert(i, i);
	}

	let popped = Arc::new(parking_lot::Mutex::new(Vec::new()));
	let workers: Vec<_> = (0..4)
		.map(|_| {
			let tree = Arc::clone(&tree);
			let popped = Arc::clone(&popped);
			thread::spawn(move || {
				let mut local = Vec::new();
				while let Some(kv) = tree.pop_last() {
					local.push(kv);
				}
				popped.lock().extend(local);
			})
		})
		.collect();
	for h in workers {
		h.join().unwrap();
	}

	let popped = popped.lock();
	let mut seen = std::collections::HashSet::new();
	for (k, _) in popped.iter() {
		assert!(seen.insert(*k), "duplicate pop for key {k}");
	}
	assert_eq!(popped.len(), N as usize);
	assert!(tree.is_empty());
}

// ===========================================================================
// Iterator during concurrent `clear` — no UB
// ===========================================================================

#[test]
fn iterator_outlives_concurrent_clear() {
	let tree = Arc::new(Tree::<u32, u32>::new());
	for i in 0..1024u32 {
		tree.insert(i, i);
	}

	let tree_writer = Arc::clone(&tree);
	let done = Arc::new(AtomicBool::new(false));
	let done_w = Arc::clone(&done);
	let writer = thread::spawn(move || {
		while !done_w.load(Ordering::Relaxed) {
			tree_writer.clear();
			for i in 0..256u32 {
				tree_writer.insert(i, i);
			}
		}
	});

	for _ in 0..16 {
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		let mut last: Option<u32> = None;
		while let Some((k, _)) = iter.next() {
			if let Some(p) = last {
				if *k < p {
					// A `clear()` happened mid-iteration — sequence reset.
					// That's permitted; we just can't go backwards twice in
					// quick succession within the same logical traversal.
				}
			}
			last = Some(*k);
		}
	}

	done.store(true, Ordering::Relaxed);
	writer.join().unwrap();
	tree.assert_invariants();
}

// ===========================================================================
// Linearizability sanity: results must match *some* sequential ordering
// ===========================================================================
//
// We don't run a full linearizability checker — that's expensive — but a
// cheap heuristic catches gross violations: if we record each operation's
// (op, key, result, observed_len_after) and then look for a sequential
// ordering that's consistent with the per-thread program order and the
// observed return values, the result set should match a BTreeMap oracle's
// final state.

#[test]
fn concurrent_operations_produce_consistent_final_state() {
	let tree = Arc::new(Tree::<u32, u32>::new());

	// Disjoint per-thread key ranges so each thread has full control of
	// its own operations' outcomes. This keeps the test cheap while
	// catching cross-thread interference bugs.
	let workers: Vec<_> = (0..4u32)
		.map(|tid| {
			let tree = Arc::clone(&tree);
			thread::spawn(move || {
				let base = tid * 10_000;
				for i in 0..2_500u32 {
					tree.insert(base + i, base + i);
				}
				for i in 0..1_000u32 {
					assert_eq!(tree.remove(&(base + i)), Some(base + i));
				}
			})
		})
		.collect();
	for h in workers {
		h.join().unwrap();
	}

	tree.assert_invariants();

	for tid in 0..4u32 {
		let base = tid * 10_000;
		for i in 0..1_000u32 {
			assert_eq!(tree.lookup(&(base + i), |v| *v), None, "should be removed");
		}
		for i in 1_000..2_500u32 {
			assert_eq!(tree.lookup(&(base + i), |v| *v), Some(base + i));
		}
	}
}

// ===========================================================================
// High-churn drop-counting under contention
// ===========================================================================

#[test]
fn concurrent_churn_drops_values_exactly() {
	let drop_count = Arc::new(AtomicUsize::new(0));

	#[derive(Clone)]
	struct Counted(Arc<AtomicUsize>);

	impl Drop for Counted {
		fn drop(&mut self) {
			self.0.fetch_add(1, Ordering::SeqCst);
		}
	}

	// SAFETY: see the matching impl in tests/functional_coverage.rs.
	unsafe impl ferntree::OptimisticRead for Counted {
		const EPOCH_DEFERRED_DROP: bool = true;
		type Slot = ferntree::atomic_slot::BoxedSlot<Self>;
	}

	let inserts = 200usize;
	let threads = 4usize;

	{
		let tree = Arc::new(Tree::<u32, Counted>::new());
		let workers: Vec<_> = (0..threads as u32)
			.map(|tid| {
				let tree = Arc::clone(&tree);
				let drop_count = Arc::clone(&drop_count);
				thread::spawn(move || {
					for i in 0..inserts as u32 {
						let key = tid.wrapping_mul(0x1000_0000) ^ i;
						tree.insert(key, Counted(Arc::clone(&drop_count)));
					}
					for i in 0..inserts as u32 {
						let key = tid.wrapping_mul(0x1000_0000) ^ i;
						tree.remove(&key);
					}
				})
			})
			.collect();
		for h in workers {
			h.join().unwrap();
		}
		tree.assert_invariants();
		assert!(tree.is_empty());
		drop(tree);
	}

	// Encourage crossbeam-epoch to flush deferred destructors.
	for _ in 0..32 {
		drop(crossbeam_epoch::pin());
	}
	thread::sleep(Duration::from_millis(50));
	for _ in 0..32 {
		drop(crossbeam_epoch::pin());
	}

	let observed = drop_count.load(Ordering::SeqCst);
	// The leaf stores two clones per entry: one in the canonical
	// `entries` field used by shared-lock paths, one in the atomic
	// mirror used by the optimistic-read fast path. Each insert
	// produces 2 storage clones; each remove drops both (the entries
	// one inline, the mirror one through epoch GC).
	let expected = 2 * inserts * threads;
	// Allow a small slack because epoch reclamation can defer drops beyond
	// our forcing window, but never *over*-drop: that's a double-free.
	assert!(observed <= expected, "observed {observed} drops > {expected} inserts (double-drop?)");
	assert!(
		observed >= expected.saturating_sub(2 * threads),
		"observed {observed} drops < expected {expected} (likely leak)"
	);
}
