#![no_main]
//! Differential fuzz target: apply a random sequence of operations to a
//! ferntree `Tree<u32, u32>` and a `BTreeMap<u32, u32>` oracle, asserting
//! return-value equivalence after every operation and full snapshot equality
//! at the end.
//!
//! The aim is to push the tree into states that the proptest oracle does not
//! reach as quickly — splits, merges, root replacements, repeated re-inserts,
//! and random remove patterns. Any divergence between ferntree and BTreeMap
//! is a correctness bug.

use arbitrary::{Arbitrary, Unstructured};
use ferntree::Tree;
use libfuzzer_sys::fuzz_target;
use std::collections::BTreeMap;

#[derive(Debug, Arbitrary)]
enum Op {
	Insert(u32, u32),
	Remove(u32),
	Lookup(u32),
	ContainsKey(u32),
	First,
	Last,
	PopFirst,
	PopLast,
	GetOrInsert(u32, u32),
	Len,
	Clear,
}

fuzz_target!(|data: &[u8]| {
	let mut u = Unstructured::new(data);
	let Ok(ops) = <Vec<Op> as Arbitrary>::arbitrary(&mut u) else {
		return;
	};
	// Cap the op count so each iteration stays in a reasonable wall-time
	// budget. Long sequences still get exercised across many runs.
	let ops: Vec<Op> = ops.into_iter().take(1024).collect();

	let tree: Tree<u32, u32> = Tree::new();
	let mut oracle: BTreeMap<u32, u32> = BTreeMap::new();

	for op in ops {
		match op {
			Op::Insert(k, v) => {
				let got = tree.insert(k, v);
				let expected = oracle.insert(k, v);
				assert_eq!(got, expected, "insert({k}, {v})");
			}
			Op::Remove(k) => {
				let got = tree.remove(&k);
				let expected = oracle.remove(&k);
				assert_eq!(got, expected, "remove({k})");
			}
			Op::Lookup(k) => {
				let got = tree.lookup(&k, |v| *v);
				let expected = oracle.get(&k).copied();
				assert_eq!(got, expected, "lookup({k})");
			}
			Op::ContainsKey(k) => {
				let got = tree.contains_key(&k);
				let expected = oracle.contains_key(&k);
				assert_eq!(got, expected, "contains_key({k})");
			}
			Op::First => {
				let got = tree.first(|k, v| (*k, *v));
				let expected = oracle.iter().next().map(|(k, v)| (*k, *v));
				assert_eq!(got, expected, "first()");
			}
			Op::Last => {
				let got = tree.last(|k, v| (*k, *v));
				let expected = oracle.iter().next_back().map(|(k, v)| (*k, *v));
				assert_eq!(got, expected, "last()");
			}
			Op::PopFirst => {
				let got = tree.pop_first();
				let expected_key = oracle.keys().next().copied();
				let expected = expected_key.map(|k| (k, oracle.remove(&k).unwrap()));
				assert_eq!(got, expected, "pop_first()");
			}
			Op::PopLast => {
				let got = tree.pop_last();
				let expected_key = oracle.keys().next_back().copied();
				let expected = expected_key.map(|k| (k, oracle.remove(&k).unwrap()));
				assert_eq!(got, expected, "pop_last()");
			}
			Op::GetOrInsert(k, v) => {
				let got = tree.get_or_insert(k, v);
				let expected = *oracle.entry(k).or_insert(v);
				assert_eq!(got, expected, "get_or_insert({k}, {v})");
			}
			Op::Len => {
				assert_eq!(tree.len(), oracle.len(), "len()");
			}
			Op::Clear => {
				tree.clear();
				oracle.clear();
				assert_eq!(tree.len(), 0, "clear()");
				assert!(tree.is_empty(), "clear()");
			}
		}
	}

	// Final snapshot equivalence: every key in the oracle must be in the
	// tree with the same value, and the tree must report the same length.
	assert_eq!(tree.len(), oracle.len(), "final len");
	for (k, v) in &oracle {
		assert_eq!(tree.lookup(k, |val| *val), Some(*v), "final lookup({k})");
	}

	// Structural invariants must hold after any operation sequence.
	tree.assert_invariants();
});
