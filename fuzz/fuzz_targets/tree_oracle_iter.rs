#![no_main]
//! Iterator-focused fuzz target: builds a tree, then iterates forward, in
//! reverse, and over random ranges, comparing each yielded entry against a
//! `BTreeMap` oracle.
//!
//! This exercises the `unsafe { std::mem::transmute(guard) }` and
//! `kv_at_unchecked` paths in `src/iter.rs` across every node-boundary and
//! cursor state.

use arbitrary::{Arbitrary, Unstructured};
use ferntree::Tree;
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;
use std::ops::Bound;

#[derive(Debug, Arbitrary)]
struct Input {
	entries: Vec<(u32, u32)>,
	removes: Vec<u32>,
	range_min: Option<(bool, u32)>,
	range_max: Option<(bool, u32)>,
}

fn bound((included, key): (bool, u32)) -> Bound<u32> {
	if included {
		Bound::Included(key)
	} else {
		Bound::Excluded(key)
	}
}

fn as_ref(b: &Bound<u32>) -> Bound<&u32> {
	match b {
		Bound::Included(k) => Bound::Included(k),
		Bound::Excluded(k) => Bound::Excluded(k),
		Bound::Unbounded => Bound::Unbounded,
	}
}

fuzz_target!(|input: Input| {
	let Input { entries, removes, range_min, range_max } = input;

	let tree: Tree<u32, u32> = Tree::new();
	let mut oracle: BTreeMap<u32, u32> = BTreeMap::new();

	// Build up the tree. Cap to keep runs bounded.
	for (k, v) in entries.into_iter().take(2048) {
		tree.insert(k, v);
		oracle.insert(k, v);
	}
	for k in removes.into_iter().take(2048) {
		assert_eq!(tree.remove(&k), oracle.remove(&k));
	}

	tree.assert_invariants();

	// Forward iteration matches oracle order.
	{
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		let mut oracle_iter = oracle.iter();
		while let Some((tk, tv)) = iter.next() {
			let (ok, ov) = oracle_iter.next().expect("tree had more entries than oracle");
			assert_eq!(tk, ok, "forward iter key");
			assert_eq!(tv, ov, "forward iter value");
		}
		assert!(oracle_iter.next().is_none(), "oracle had more entries than tree");
	}

	// Reverse iteration matches oracle order.
	{
		let mut iter = tree.raw_iter();
		iter.seek_to_last();
		let mut oracle_iter = oracle.iter().rev();
		while let Some((tk, tv)) = iter.prev() {
			let (ok, ov) = oracle_iter.next().expect("tree had more entries than oracle (rev)");
			assert_eq!(tk, ok, "reverse iter key");
			assert_eq!(tv, ov, "reverse iter value");
		}
		assert!(oracle_iter.next().is_none(), "oracle had more entries than tree (rev)");
	}

	// Range iteration matches oracle.
	let min = range_min.map(bound).unwrap_or(Bound::Unbounded);
	let max = range_max.map(bound).unwrap_or(Bound::Unbounded);
	let valid = match (min, max) {
		(Bound::Included(a), Bound::Included(b)) => a <= b,
		(Bound::Included(a), Bound::Excluded(b))
		| (Bound::Excluded(a), Bound::Included(b)) => a < b,
		(Bound::Excluded(a), Bound::Excluded(b)) => a < b,
		_ => true,
	};
	if valid {
		let mut range_iter = tree.range(as_ref(&min), as_ref(&max));
		let mut oracle_range = oracle.range((min, max));
		while let Some((tk, tv)) = range_iter.next() {
			let (ok, ov) = oracle_range.next().expect("range: tree had extra entries");
			assert_eq!(tk, ok, "range key");
			assert_eq!(tv, ov, "range value");
		}
		assert!(oracle_range.next().is_none(), "range: oracle had extra entries");
	}
});
