//! Concurrency regression: bounded `range` / `range_rev` scans must never
//! emit keys outside the requested half-open range, even when leaves are
//! restructuring beneath the iterator due to concurrent inserts and removes.
//!
//! Before the fix, `Range::peek` / `Range::next` (and the reverse
//! equivalents) only re-checked the *far* bound on each emit. The *near*
//! bound was enforced solely by the initial seek, so a sufficiently busy
//! commit path could move the raw iterator's leaf cursor across that bound
//! and leak keys from a completely different prefix.
//!
//! Keys are `Vec<u8>` shaped after surrealdb's index-key layout — a fixed
//! prefix followed by an `ix` byte and a `kind` byte, then a per-prefix
//! counter. This mirrors the failure mode from the parent project
//! (`multi_index_concurrent_test_create_update_delete`), where the
//! out-of-range bytes had completely different `ix` *and* `kind` bytes.

use ferntree::Tree;
use std::ops::Bound;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const TEST_DURATION_SECS: u64 = 2;
const NUM_INDEXES: u8 = 8;
const NUM_KINDS: u8 = 4;
/// Seed each (ix, kind) prefix with enough entries to span multiple leaves
/// (LEAF_CAPACITY is 64), so concurrent inserts/removes drive real splits
/// and merges rather than purely in-leaf shuffles.
const SEED_PER_PREFIX: u64 = 128;
const NUM_WRITERS: u32 = 5;

fn make_key(ix: u8, kind: u8, n: u64) -> Vec<u8> {
	// Layout: 4-byte fixed prefix + ix + kind + 8-byte counter, so the
	// natural lexicographic order matches `(ix, kind, n)` tuple order.
	let mut k = Vec::with_capacity(4 + 1 + 1 + 8);
	k.extend_from_slice(b"\0\0\0\0");
	k.push(ix);
	k.push(kind);
	k.extend_from_slice(&n.to_be_bytes());
	k
}

fn prefix_range(ix: u8, kind: u8) -> (Vec<u8>, Vec<u8>) {
	let mut beg = Vec::with_capacity(6);
	beg.extend_from_slice(b"\0\0\0\0");
	beg.push(ix);
	beg.push(kind);
	let mut end = beg.clone();
	// Next prefix start: bump `kind` (no carry needed in the test
	// range, which keeps `kind < NUM_KINDS`).
	*end.last_mut().unwrap() += 1;
	(beg, end)
}

fn seed(tree: &Tree<Vec<u8>, ()>) {
	for ix in 0..NUM_INDEXES {
		for kind in 0..NUM_KINDS {
			for n in 0..SEED_PER_PREFIX {
				tree.insert(make_key(ix, kind, n), ());
			}
		}
	}
}

fn spawn_writers(
	tree: Arc<Tree<Vec<u8>, ()>>,
	stop: Arc<AtomicBool>,
) -> Vec<thread::JoinHandle<()>> {
	(0..NUM_WRITERS)
		.map(|w| {
			let tree = Arc::clone(&tree);
			let stop = Arc::clone(&stop);
			thread::spawn(move || {
				// Each writer cycles through every (ix, kind) prefix and
				// inserts always-fresh keys (counter `n` keeps growing).
				// Insert-only churn — matching the surrealdb full-text
				// commit pattern — keeps leaves filling and splitting,
				// which is what drives the cursor across its near bound.
				let mut n: u64 = SEED_PER_PREFIX + (w as u64) * 1_000_000;
				while !stop.load(Ordering::Relaxed) {
					for ix in 0..NUM_INDEXES {
						for kind in 0..NUM_KINDS {
							tree.insert(make_key(ix, kind, n), ());
						}
					}
					n = n.wrapping_add(1);
				}
			})
		})
		.collect()
}

/// Target a narrow `(ix, kind)` prefix in the middle of the keyspace, so any
/// cursor that drifts backward across the near bound under concurrent
/// commits will surface keys from a *different* prefix entirely (different
/// `ix` *and* different `kind` byte) — i.e. clearly lexicographically
/// before `beg`.
fn target_prefix() -> (Vec<u8>, Vec<u8>) {
	prefix_range(NUM_INDEXES / 2, NUM_KINDS / 2)
}

#[test]
fn forward_scan_never_emits_keys_outside_requested_range_under_concurrent_commits() {
	let tree = Arc::new(Tree::<Vec<u8>, ()>::new());
	seed(&tree);

	let stop = Arc::new(AtomicBool::new(false));
	let writers = spawn_writers(Arc::clone(&tree), Arc::clone(&stop));

	let (beg, end) = target_prefix();
	let oor = AtomicUsize::new(0);
	let mut scans: usize = 0;
	let start = Instant::now();
	while start.elapsed() < Duration::from_secs(TEST_DURATION_SECS) {
		let mut range = tree.range::<[u8]>(
			Bound::Included(beg.as_slice()),
			Bound::Excluded(end.as_slice()),
		);
		while let Some((k, _)) = range.next() {
			if k.as_slice() < beg.as_slice() || k.as_slice() >= end.as_slice() {
				oor.fetch_add(1, Ordering::Relaxed);
			}
		}
		scans += 1;
	}

	stop.store(true, Ordering::Release);
	for h in writers {
		h.join().unwrap();
	}

	let oor_count = oor.load(Ordering::Relaxed);
	assert_eq!(
		oor_count, 0,
		"forward scan emitted {oor_count} out-of-range keys over {scans} scans"
	);
}

/// Stress the optimistic `remove` descent under concurrent `shift_remove`
/// on the same leaf. Before the fix this would intermittently SEGV from
/// `BoxedSlot::load_into` dereferencing a transiently null slot inside
/// `LeafNode::lower_bound`, because the unsafe `&self` binary search did
/// not handle the writer-vs-optimistic-reader race that
/// `lower_bound_raw` is designed for.
#[test]
fn concurrent_inserts_and_removes_do_not_segv_on_boxed_keys() {
	let tree = Arc::new(Tree::<Vec<u8>, ()>::new());
	seed(&tree);

	let stop = Arc::new(AtomicBool::new(false));
	let mut handles = Vec::new();
	for w in 0..NUM_WRITERS {
		let tree = Arc::clone(&tree);
		let stop = Arc::clone(&stop);
		handles.push(thread::spawn(move || {
			// Disjoint `n` band per writer — every writer's `remove`
			// targets a key its own thread inserted, but the shifts
			// inside the leaf hit shared slots, so two writers
			// triggering `find_exact_exclusive_leaf_and_optimistic_parent`
			// race on the same leaf's contents.
			let mut n: u64 = SEED_PER_PREFIX + (w as u64) * 1_000_000;
			while !stop.load(Ordering::Relaxed) {
				for ix in 0..NUM_INDEXES {
					for kind in 0..NUM_KINDS {
						let k = make_key(ix, kind, n);
						tree.insert(k.clone(), ());
						let _ = tree.remove(&k);
					}
				}
				n = n.wrapping_add(1);
			}
		}));
	}

	thread::sleep(Duration::from_secs(TEST_DURATION_SECS));
	stop.store(true, Ordering::Release);
	for h in handles {
		h.join().unwrap();
	}
}

#[test]
fn reverse_scan_never_emits_keys_outside_requested_range_under_concurrent_commits() {
	let tree = Arc::new(Tree::<Vec<u8>, ()>::new());
	seed(&tree);

	let stop = Arc::new(AtomicBool::new(false));
	let writers = spawn_writers(Arc::clone(&tree), Arc::clone(&stop));

	let (beg, end) = target_prefix();
	let oor = AtomicUsize::new(0);
	let mut scans: usize = 0;
	let start = Instant::now();
	while start.elapsed() < Duration::from_secs(TEST_DURATION_SECS) {
		let mut range = tree.range_rev::<[u8]>(
			Bound::Included(beg.as_slice()),
			Bound::Excluded(end.as_slice()),
		);
		while let Some((k, _)) = range.next() {
			if k.as_slice() < beg.as_slice() || k.as_slice() >= end.as_slice() {
				oor.fetch_add(1, Ordering::Relaxed);
			}
		}
		scans += 1;
	}

	stop.store(true, Ordering::Release);
	for h in writers {
		h.join().unwrap();
	}

	let oor_count = oor.load(Ordering::Relaxed);
	assert_eq!(
		oor_count, 0,
		"reverse scan emitted {oor_count} out-of-range keys over {scans} scans"
	);
}
