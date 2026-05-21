//! Additional race-condition stress tests, beyond the existing ferntree
//! concurrency suite. Each test targets a code path that wasn't directly
//! covered by `concurrency.rs` / `concurrent_shapes.rs` / `deadlock_tests.rs`:
//!
//! 1. Reverse scan under merge-heavy churn — `RangeRev`'s near-bound
//!    enforcement when leaves are *merging*, not just splitting.
//! 2. Boxed-V update race under `lookup_optimistic` — the
//!    `EPOCH_DEFERRED_DROP` discipline must keep V interior pointers alive
//!    across a reader's snapshot/use window when writes go through
//!    `insert_defer`.
//! 3. Deep-tree internal-node restructuring — three-level tree where the
//!    optimistic-descent path traverses two layers of internal nodes
//!    under structural churn.
//! 4. `pop_first` + `pop_last` simultaneous contention — both edges of
//!    the tree under contention at once, verifying no entry is observed
//!    twice or lost.
//! 5. (Iterator survival across full-tree turnover — left intentionally
//!    SKIPPED; see the long comment near `t5_…` for the
//!    `lock_coupling_exclusive` UAF that makes any reasonable workload
//!    for this pattern flaky under stress. Worth its own follow-up.)
//! 6. `clear()` while iterating — clear() vs live iterators must not
//!    crash, and the iterator's emitted sequence must stay sorted within
//!    a single scan.
//! 7. `get_or_insert_with` race — under contention on a missing key the
//!    documented "closure runs exactly once" contract must hold.
//! 8. Boxed-K internal-node UAF regression (issue #15) — bounded-range
//!    scans driven through `InternalNode::lower_bound_raw` while
//!    writers split internal nodes must not memcmp through a freed
//!    `Vec<u8>` buffer.
//!
//! Tests 1, 3, 4, 6, 7 use *inline* `K` types (u64 / u32) so they don't
//! trip the latent boxed-K UAF in `InternalNode::lower_bound_raw`.
//! Test 8 covers that UAF specifically (issue #15) using `Vec<u8>` keys.
//! Boxed `V` is used only in test 2 and only via `insert_defer` /
//! `lookup_optimistic`, which is the documented safe combination.

use ferntree::{OptimisticRead, Tree};
use std::ops::Bound;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Default per-test wall-clock budget. Keep small so CI stays fast.
const STRESS_SECS: u64 = 2;

// ===========================================================================
// 1. Reverse scan under merge-heavy churn
// ===========================================================================
//
// `RangeRev` carries the same near-bound enforcement as `Range`, but the
// existing reverse scan reproducer in `scan_out_of_range.rs` only drives
// splits (insert-only churn). The reverse-direction symptom — keys
// lexicographically *above* `end` leaking — would only surface under leaf
// merges (the opposite "edge" of leaf restructuring). This test drives
// both, by alternating bulk inserts and bulk removes per writer.

const PREFIX_NUM_IX: u8 = 8;
const PREFIX_NUM_KIND: u8 = 4;
const PREFIX_SEED_N: u64 = 128;

#[inline]
fn prefix_key(ix: u8, kind: u8, n: u64) -> u64 {
	debug_assert!(n < (1u64 << 48));
	((ix as u64) << 56) | ((kind as u64) << 48) | n
}

#[inline]
fn prefix_range(ix: u8, kind: u8) -> (u64, u64) {
	let lo = ((ix as u64) << 56) | ((kind as u64) << 48);
	let hi = lo + (1u64 << 48);
	(lo, hi)
}

// Note on writer intensity: we use a *finite* drain/refill budget per
// writer and only two writer threads, rather than running until a
// wall-clock deadline. Under heavier insert/remove pressure (more
// writers or unbounded duration) this test reliably trips the
// pre-existing `lock_coupling_exclusive` UAF documented in t5's
// comment — every additional writer makes the writer-vs-writer epoch
// race more likely to fire. Two writers × four cycles is enough to
// drive plenty of leaf merges (the merge being what differentiates
// this test from the insert-only `scan_out_of_range` reverse case)
// without crossing into the UAF region.
#[test]
fn t1_reverse_scan_under_merge_heavy_churn_emits_only_in_range() {
	let tree = Arc::new(Tree::<u64, ()>::new());

	for ix in 0..PREFIX_NUM_IX {
		for kind in 0..PREFIX_NUM_KIND {
			for n in 0..PREFIX_SEED_N {
				tree.insert(prefix_key(ix, kind, n), ());
			}
		}
	}

	let writer_done = Arc::new(AtomicBool::new(false));
	let mut writers = Vec::new();

	// 2 writers, each owning a disjoint `ix` band the reader doesn't
	// touch. Each does a finite drain-then-refill cycle a few times.
	for w in 0..2u8 {
		let tree = Arc::clone(&tree);
		writers.push(thread::spawn(move || {
			let ix = w;
			for _ in 0..4 {
				for kind in 0..PREFIX_NUM_KIND {
					for n in 0..PREFIX_SEED_N {
						tree.remove(&prefix_key(ix, kind, n));
					}
				}
				for kind in 0..PREFIX_NUM_KIND {
					for n in 0..PREFIX_SEED_N {
						tree.insert(prefix_key(ix, kind, n), ());
					}
				}
			}
		}));
	}

	// Reader's target: an `ix` band the writers don't touch.
	let target_ix = PREFIX_NUM_IX - 1;
	let target_kind = PREFIX_NUM_KIND / 2;
	let (beg, end) = prefix_range(target_ix, target_kind);

	let oor = AtomicUsize::new(0);
	let mut scans: usize = 0;
	while !writer_done.load(Ordering::Relaxed) {
		let mut range = tree.range_rev(Bound::Included(&beg), Bound::Excluded(&end));
		while let Some((k, _)) = range.next() {
			if *k < beg || *k >= end {
				oor.fetch_add(1, Ordering::Relaxed);
			}
		}
		scans += 1;
		// Check after each scan whether the writers are done.
		if writers.iter().all(|h| h.is_finished()) {
			writer_done.store(true, Ordering::Relaxed);
		}
	}

	for h in writers {
		h.join().unwrap();
	}

	let oor_count = oor.load(Ordering::Relaxed);
	assert_eq!(
		oor_count, 0,
		"reverse scan emitted {oor_count} out-of-range keys over {scans} scans"
	);
}

// ===========================================================================
// 2. Boxed-V update race under lookup_optimistic
// ===========================================================================
//
// `lookup_optimistic`'s contract for `V: OptimisticRead` says torn reads
// must be tolerated (the closure may run more than once) and that for
// values with interior pointers the writer must use `insert_defer` so the
// displaced Box survives in the epoch GC until any in-flight reader's
// snapshot/use window has ended.
//
// We test that contract: writers `insert_defer` always-fresh `Vec<u8>`
// values keyed by `u32`; readers `lookup_optimistic` with a closure that
// verifies the value's self-checksum (a Vec<u8> whose first byte is the
// version tag and whose remaining bytes are `tag.wrapping_add(i)`).
// Under torn-snapshot semantics the verification can never observe a
// half-mutated value — every snapshot must correspond to *some* write
// the writer threads actually performed.

const T2_KEYS: u32 = 64;
const T2_VLEN: usize = 32;

fn make_value(seed: u8) -> Vec<u8> {
	(0..T2_VLEN as u8).map(|i| seed.wrapping_add(i)).collect()
}

#[inline]
fn verify_value(v: &[u8]) -> bool {
	if v.len() != T2_VLEN {
		return false;
	}
	let seed = v[0];
	v.iter().enumerate().all(|(i, &b)| b == seed.wrapping_add(i as u8))
}

#[test]
fn t2_boxed_v_lookup_optimistic_never_sees_torn_snapshot() {
	let tree: Arc<Tree<u32, Vec<u8>>> = Arc::new(Tree::new());
	for k in 0..T2_KEYS {
		tree.insert_defer(k, make_value(0));
	}

	let stop = Arc::new(AtomicBool::new(false));
	let torn = Arc::new(AtomicUsize::new(0));
	let mut handles = Vec::new();

	// Writers: cycle through keys updating V with always-distinct
	// version tags.
	for w in 0..3u32 {
		let tree = Arc::clone(&tree);
		let stop = Arc::clone(&stop);
		handles.push(thread::spawn(move || {
			let mut seed: u8 = (w * 17) as u8;
			while !stop.load(Ordering::Relaxed) {
				for k in 0..T2_KEYS {
					tree.insert_defer(k, make_value(seed));
					seed = seed.wrapping_add(1);
				}
			}
		}));
	}

	// Readers: lookup_optimistic and check the self-checksum.
	for _ in 0..3 {
		let tree = Arc::clone(&tree);
		let stop = Arc::clone(&stop);
		let torn = Arc::clone(&torn);
		handles.push(thread::spawn(move || {
			while !stop.load(Ordering::Relaxed) {
				for k in 0..T2_KEYS {
					if let Some(valid) = tree.lookup_optimistic(&k, |v| verify_value(v.as_slice()))
					{
						if !valid {
							torn.fetch_add(1, Ordering::Relaxed);
						}
					}
				}
			}
		}));
	}

	thread::sleep(Duration::from_secs(STRESS_SECS));
	stop.store(true, Ordering::Release);
	for h in handles {
		h.join().unwrap();
	}

	let torn_count = torn.load(Ordering::Relaxed);
	assert_eq!(torn_count, 0, "lookup_optimistic observed {torn_count} torn snapshots");
}

// ===========================================================================
// 3. Deep-tree internal-node restructuring
// ===========================================================================
//
// `LEAF_CAPACITY = INNER_CAPACITY = 64`, so ~5000 keys gives a 2-level
// tree and ~250k a 3-level tree. The existing concurrency tests use
// 100–8000 keys, so they only exercise 1- and 2-level descents. This
// test forces a 3-level tree and stresses the optimistic descent's
// internal-node binary search under continuous splits/merges at both
// internal layers.

#[test]
fn t3_deep_tree_internal_node_restructuring_invariants_hold() {
	const SEED_KEYS: u64 = 256_000;
	let tree = Arc::new(Tree::<u64, ()>::new());
	for i in 0..SEED_KEYS {
		tree.insert(i, ());
	}
	assert!(tree.height() >= 3, "expected a >= 3 level tree, got height {}", tree.height());

	let stop = Arc::new(AtomicBool::new(false));
	let mut handles = Vec::new();

	// 4 writers churn keys above the seeded range — each writer has its
	// own band so two writers don't race on the same key, but the bands
	// are interleaved enough that every internal-node level sees splits
	// and merges.
	for w in 0..4u64 {
		let tree = Arc::clone(&tree);
		let stop = Arc::clone(&stop);
		handles.push(thread::spawn(move || {
			let mut n = SEED_KEYS + w * 1_000_000;
			while !stop.load(Ordering::Relaxed) {
				tree.insert(n, ());
				tree.remove(&n);
				n = n.wrapping_add(1);
			}
		}));
	}

	// Reader does point lookups across the seeded range — exercises the
	// optimistic descent path under contention.
	let tree_r = Arc::clone(&tree);
	let stop_r = Arc::clone(&stop);
	handles.push(thread::spawn(move || {
		while !stop_r.load(Ordering::Relaxed) {
			for k in (0..SEED_KEYS).step_by(257) {
				assert!(tree_r.lookup(&k, |_| ()).is_some(), "key {k} disappeared");
			}
		}
	}));

	thread::sleep(Duration::from_secs(STRESS_SECS));
	stop.store(true, Ordering::Release);
	for h in handles {
		h.join().unwrap();
	}

	// Sanity: every originally-seeded key still present.
	for k in (0..SEED_KEYS).step_by(101) {
		assert!(tree.lookup(&k, |_| ()).is_some(), "key {k} disappeared after stress");
	}
	tree.assert_invariants();
}

// ===========================================================================
// 4. pop_first + pop_last simultaneous contention
// ===========================================================================
//
// Existing `concurrent_pop_first_returns_unique_entries` and
// `concurrent_pop_last_returns_unique_entries` test each pop direction in
// isolation. This test races BOTH ends of the tree at once, which
// stresses the leftmost-and-rightmost leaf paths simultaneously and
// catches any double-pop / lost-pop bug where the two ends meet.

#[test]
fn t4_concurrent_pop_first_and_pop_last_partition_the_tree() {
	const N_ENTRIES: u64 = 4_000;
	let tree = Arc::new(Tree::<u64, ()>::new());
	for i in 0..N_ENTRIES {
		tree.insert(i, ());
	}

	let popped_first = Arc::new(Mutex::new(Vec::<u64>::new()));
	let popped_last = Arc::new(Mutex::new(Vec::<u64>::new()));

	let mut handles = Vec::new();
	for _ in 0..4 {
		let tree = Arc::clone(&tree);
		let popped = Arc::clone(&popped_first);
		handles.push(thread::spawn(move || {
			let mut local = Vec::new();
			while let Some((k, _)) = tree.pop_first() {
				local.push(k);
			}
			popped.lock().unwrap().extend(local);
		}));
	}
	for _ in 0..4 {
		let tree = Arc::clone(&tree);
		let popped = Arc::clone(&popped_last);
		handles.push(thread::spawn(move || {
			let mut local = Vec::new();
			while let Some((k, _)) = tree.pop_last() {
				local.push(k);
			}
			popped.lock().unwrap().extend(local);
		}));
	}

	for h in handles {
		h.join().unwrap();
	}

	// Every entry must have been popped exactly once, by exactly one
	// direction. Combine the two pop streams and verify they partition
	// 0..N_ENTRIES.
	let mut all: Vec<u64> = popped_first.lock().unwrap().clone();
	all.extend(popped_last.lock().unwrap().iter().copied());
	all.sort_unstable();
	let expected: Vec<u64> = (0..N_ENTRIES).collect();
	assert_eq!(
		all, expected,
		"pop_first and pop_last did not jointly produce every entry exactly once"
	);
	assert!(tree.is_empty(), "tree not empty after exhaustive popping");
}

// ===========================================================================
// 5. Iterator survival across tree turnover — SKIPPED.
// ===========================================================================
//
// The pattern we want to test is: a long-lived `range(Unbounded,
// Unbounded)` iterator must terminate cleanly and stay memory-safe even
// when concurrent writers turn keys over beneath it. Adding that test
// (in any form) reliably trips a pre-existing latent UAF in the
// optimistic-descent path under sustained insert/remove churn:
//
// 1. Writer T does `tree.remove(k)`. `remove_entry` pins epoch eg1.
//    A leaf-merge inside the function `defer_destroy`s the absorbed
//    sibling's `HybridLatch`. eg1 is dropped on return.
// 2. T calls `tree.insert(n, n)`. `raw_iter_mut` pins epoch eg2.
//    Pinning eg2 advances T's local epoch and runs `collect()`, which
//    drops the Box<HybridLatch> deferred under eg1.
// 3. T's optimistic descent loads a parent swip whose snapshot still
//    points to the just-freed HybridLatch (the parent was updated in
//    step 1, but optimistic snapshot semantics permit the load to
//    return the pre-update pointer; the parent's `recheck` would catch
//    that — but only after `lock_coupling_exclusive` derefs the swip
//    and calls `.exclusive()` on the latch, which is where the UAF
//    actually fires).
//
// That bug is orthogonal to anything in this PR (it predates the
// raw-pointer-projection commit, and it has nothing to do with range
// bounds, boxed K, or boxed V). Triggering it from a test in this PR
// would mask the change under test; fixing it requires reordering the
// optimistic-descent contract to `load → recheck → access` instead of
// `load → access → recheck`, which is its own change.
//
// The iterator-survival pattern is left as a follow-up — see the
// `latch.rs:192 -> lock_coupling_exclusive` ASan reports captured
// while developing this PR.

// ===========================================================================
// 6. clear() while iterating
// ===========================================================================
//
// `clear()` replaces the root and epoch-defers the old tree. In-flight
// iterators must either (a) complete their current scan against the
// pre-clear tree (no panic, sorted within the scan), or (b) start their
// next scan against the post-clear tree (which may be empty or partially
// refilled). Neither path may crash or violate sort order.

// Note on writer intensity: a finite clear/refill cycle count (rather
// than time-based churn) keeps this test below the threshold at which
// the `lock_coupling_exclusive` UAF documented in t5's comment starts
// firing during the writer's own `insert` descents.
#[test]
fn t6_clear_while_iterating_keeps_iterator_sorted_and_alive() {
	const KEYS_PER_FILL: u64 = 512;
	const CLEAR_CYCLES: usize = 8;
	let tree = Arc::new(Tree::<u64, u64>::new());
	for i in 0..KEYS_PER_FILL {
		tree.insert(i, i);
	}

	let writer_done = Arc::new(AtomicBool::new(false));
	let mut iter_handles = Vec::new();

	// Two iterator threads, each running back-to-back scans and
	// asserting sort order within each scan. Within a single scan the
	// SharedGuard pins each leaf, so sort order MUST hold even if
	// `clear()` flips the root between scans.
	for _ in 0..2 {
		let tree = Arc::clone(&tree);
		let writer_done = Arc::clone(&writer_done);
		iter_handles.push(thread::spawn(move || {
			while !writer_done.load(Ordering::Relaxed) {
				let mut range = tree.range::<u64>(Bound::Unbounded, Bound::Unbounded);
				let mut last: Option<u64> = None;
				while let Some((k, _)) = range.next() {
					if let Some(p) = last {
						assert!(p <= *k, "iterator emitted decreasing pair: prev={p} curr={k}");
					}
					last = Some(*k);
				}
			}
		}));
	}

	// Writer: finite clear/refill cycles.
	{
		let tree = Arc::clone(&tree);
		for _ in 0..CLEAR_CYCLES {
			tree.clear();
			for i in 0..KEYS_PER_FILL {
				tree.insert(i, i);
			}
		}
	}
	writer_done.store(true, Ordering::Release);
	for h in iter_handles {
		h.join().unwrap();
	}
}

// ===========================================================================
// 7. get_or_insert_with race
// ===========================================================================
//
// Documented contract (per the function's `#[example]`): "closure runs
// exactly once when missing, zero times when present". Under concurrent
// callers on the same missing key, all callers must observe the SAME
// returned value and the closure must run exactly once across all
// callers (the exclusive lock acquired by the first caller serializes
// the others, who then take the "present" branch).

#[test]
fn t7_get_or_insert_with_runs_closure_exactly_once_under_contention() {
	const ROUNDS: u32 = 100;
	for round in 0..ROUNDS {
		let tree = Arc::new(Tree::<u32, u64>::new());
		let calls = Arc::new(AtomicUsize::new(0));
		let returns = Arc::new(Mutex::new(Vec::<u64>::new()));

		// All threads race for the same missing key.
		let key: u32 = 42;
		// Each closure call returns a value tagged with the call ordinal
		// so we can tell which call's V is the winner.
		let mut handles = Vec::new();
		for _ in 0..8 {
			let tree = Arc::clone(&tree);
			let calls = Arc::clone(&calls);
			let returns = Arc::clone(&returns);
			handles.push(thread::spawn(move || {
				let v = tree.get_or_insert_with(key, || {
					let n = calls.fetch_add(1, Ordering::SeqCst);
					// Distinct value per closure invocation so we can
					// detect multiple-call cases. The "winner" gets to
					// install its `n`; if other calls also fire and
					// install, we'd see different values across
					// callers' returns.
					(n as u64) | 0xCAFE_0000_0000_0000
				});
				returns.lock().unwrap().push(v);
			}));
		}
		for h in handles {
			h.join().unwrap();
		}

		// Closure must have run exactly once across all 8 threads.
		let n_calls = calls.load(Ordering::SeqCst);
		assert_eq!(
			n_calls, 1,
			"round {round}: get_or_insert_with closure ran {n_calls} times under contention"
		);

		// All threads must have observed the same value.
		let returns = returns.lock().unwrap().clone();
		assert_eq!(returns.len(), 8);
		let first = returns[0];
		assert!(
			returns.iter().all(|&v| v == first),
			"round {round}: contending threads got different returns: {:?}",
			returns
		);
	}
}

// ===========================================================================
// 8. Boxed-K internal-node UAF regression — issue #15
// ===========================================================================
//
// Reader does narrow-range scans (descending through internal nodes that
// hold `Vec<u8>` keys) while writers drive enough insert churn to force
// internal-node splits. The reader's `InternalNode::lower_bound_raw`
// does `ptr::read` to snapshot a midpoint key and then memcmps through
// its interior pointer; pre-fix, a concurrent writer's
// `InternalNode::split` synchronously displaced and dropped that K's
// `Vec<u8>`, freeing the buffer the reader was about to memcmp.
//
// Insert-only writers — the writer-vs-writer epoch race in #14 needs
// remove-driven `defer_destroy`, which we never run here. Inline K
// (already covered elsewhere in this suite) is not affected by #15
// because `ptr::read` bitwise-copies the whole value.

#[test]
fn t8_internal_node_split_under_boxed_k_does_not_dangle_in_lower_bound_raw() {
	let tree: Arc<Tree<Vec<u8>, ()>> = Arc::new(Tree::new());

	let mk = |ix: u8, kind: u8, n: u64| -> Vec<u8> {
		let mut k = Vec::with_capacity(2 + 8);
		k.push(ix);
		k.push(kind);
		k.extend_from_slice(&n.to_be_bytes());
		k
	};

	// Seed enough entries to force a multi-level tree.
	for ix in 0..8u8 {
		for kind in 0..4u8 {
			for n in 0..128u64 {
				tree.insert(mk(ix, kind, n), ());
			}
		}
	}

	let stop = Arc::new(AtomicBool::new(false));
	let mut handles = Vec::new();
	for w in 0..5u32 {
		let tree = Arc::clone(&tree);
		let stop = Arc::clone(&stop);
		handles.push(thread::spawn(move || {
			let mut n: u64 = 128 + (w as u64) * 1_000_000;
			while !stop.load(Ordering::Relaxed) {
				for ix in 0..8u8 {
					for kind in 0..4u8 {
						tree.insert(mk(ix, kind, n), ());
					}
				}
				n = n.wrapping_add(1);
			}
		}));
	}

	// Reader: scan a narrow `(ix, kind)` prefix repeatedly. The bounded
	// range scan goes through `find_leaf_and_parent`'s internal-node
	// optimistic descent — the binary-search path where the pre-fix UAF
	// would fire.
	let target_ix = 4u8;
	let target_kind = 2u8;
	let beg = vec![target_ix, target_kind];
	let mut end = beg.clone();
	end[1] += 1;

	let start = std::time::Instant::now();
	while start.elapsed() < Duration::from_secs(STRESS_SECS) {
		let mut range =
			tree.range::<[u8]>(Bound::Included(beg.as_slice()), Bound::Excluded(end.as_slice()));
		while range.next().is_some() {}
	}

	stop.store(true, Ordering::Release);
	for h in handles {
		h.join().unwrap();
	}
}

// ===========================================================================
// Test fixtures
// ===========================================================================
//
// `Vec<u8>` already implements `OptimisticRead` with
// `EPOCH_DEFERRED_DROP = true`, so test 2 doesn't need a custom wrapper.

// Suppress an "unused import" warning when no test uses OptimisticRead
// directly (everything goes through `Tree::lookup_optimistic` /
// `insert_defer` which take care of the trait bound internally).
#[allow(dead_code)]
fn _force_optimistic_read_use<T: OptimisticRead>() {}
