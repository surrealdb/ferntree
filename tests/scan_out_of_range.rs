//! Concurrency regression: bounded `range` / `range_rev` scans must never
//! emit keys outside the requested half-open range, even when leaves are
//! restructuring beneath the iterator due to concurrent inserts.
//!
//! Before the fix, `Range::peek` / `Range::next` (and the reverse
//! equivalents) only re-checked the *far* bound on each emit. The *near*
//! bound was enforced solely by the initial seek, so a sufficiently busy
//! commit path could move the raw iterator's leaf cursor across that bound
//! and leak keys from a completely different prefix.
//!
//! Keys are packed `u64`s of the form `(ix, kind, n)`:
//!
//! ```text
//! bits 56..64  -- ix    (8 bits, the "index" byte)
//! bits 48..56  -- kind  (8 bits, the "category" byte)
//! bits  0..48  -- n     (48 bits, the per-prefix counter)
//! ```
//!
//! The natural `u64` order matches the lexicographic byte order, so prefix
//! ranges `[(ix,kind,0), (ix,kind+1,0))` are contiguous in the key space.
//! Inline `u64` storage also keeps the test away from a separate, latent
//! pre-existing issue: `InternalNode::lower_bound_raw` does a `ptr::read`
//! of `K` for its binary-search snapshot, which is sound for `Copy` types
//! but reads a stale interior pointer for boxed `K` (e.g. `Vec<u8>`) when
//! an internal-node `remove_at` / `split` / `merge` concurrently drops
//! the displaced `K` without routing it through the epoch GC. That's
//! orthogonal to the bound-emission bug under test here and is worth its
//! own follow-up.

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

#[inline]
fn key(ix: u8, kind: u8, n: u64) -> u64 {
	debug_assert!(n < (1u64 << 48));
	((ix as u64) << 56) | ((kind as u64) << 48) | n
}

#[inline]
fn prefix_lo(ix: u8, kind: u8) -> u64 {
	key(ix, kind, 0)
}

#[inline]
fn prefix_hi(ix: u8, kind: u8) -> u64 {
	// Next-prefix start: (ix, kind+1, 0) with carry into ix if kind == 0xff.
	let raised = ((ix as u64) << 56) | ((kind as u64) << 48);
	raised + (1u64 << 48)
}

fn seed(tree: &Tree<u64, ()>) {
	for ix in 0..NUM_INDEXES {
		for kind in 0..NUM_KINDS {
			for n in 0..SEED_PER_PREFIX {
				tree.insert(key(ix, kind, n), ());
			}
		}
	}
}

fn spawn_writers(tree: Arc<Tree<u64, ()>>, stop: Arc<AtomicBool>) -> Vec<thread::JoinHandle<()>> {
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
							tree.insert(key(ix, kind, n), ());
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
fn target_prefix() -> (u64, u64) {
	let ix = NUM_INDEXES / 2;
	let kind = NUM_KINDS / 2;
	(prefix_lo(ix, kind), prefix_hi(ix, kind))
}

#[test]
fn forward_scan_never_emits_keys_outside_requested_range_under_concurrent_commits() {
	let tree = Arc::new(Tree::<u64, ()>::new());
	seed(&tree);

	let stop = Arc::new(AtomicBool::new(false));
	let writers = spawn_writers(Arc::clone(&tree), Arc::clone(&stop));

	let (beg, end) = target_prefix();
	let oor = AtomicUsize::new(0);
	let mut scans: usize = 0;
	let start = Instant::now();
	while start.elapsed() < Duration::from_secs(TEST_DURATION_SECS) {
		let mut range = tree.range(Bound::Included(&beg), Bound::Excluded(&end));
		while let Some((k, _)) = range.next() {
			if *k < beg || *k >= end {
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

#[test]
fn reverse_scan_never_emits_keys_outside_requested_range_under_concurrent_commits() {
	let tree = Arc::new(Tree::<u64, ()>::new());
	seed(&tree);

	let stop = Arc::new(AtomicBool::new(false));
	let writers = spawn_writers(Arc::clone(&tree), Arc::clone(&stop));

	let (beg, end) = target_prefix();
	let oor = AtomicUsize::new(0);
	let mut scans: usize = 0;
	let start = Instant::now();
	while start.elapsed() < Duration::from_secs(TEST_DURATION_SECS) {
		let mut range = tree.range_rev(Bound::Included(&beg), Bound::Excluded(&end));
		while let Some((k, _)) = range.next() {
			if *k < beg || *k >= end {
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
