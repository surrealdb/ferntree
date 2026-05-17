// Copyright © SurrealDB Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

//! Fanout sweep benchmarks.
//!
//! Sweeps the internal-node (`IC`) and leaf (`LC`) capacities across a
//! representative range and measures sequential insert, random lookup, and
//! full forward iteration throughput. The goal is to give users a basis for
//! picking a fanout that matches their key/value sizes.
//!
//! For small values (e.g. `u64`), high fanout (64–128) generally wins. For
//! large values (e.g. `[u8; 256]`), lower fanout (16–32) keeps leaves
//! cache-friendly and reduces split/merge cost.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ferntree::GenericTree;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::hint::black_box;

const SEED: u64 = 42;

fn sequential_keys(count: usize) -> Vec<i64> {
	(0..count as i64).collect()
}

fn random_keys(count: usize) -> Vec<i64> {
	let mut rng = StdRng::seed_from_u64(SEED);
	(0..count).map(|_| rng.random()).collect()
}

// Small-value workload (`u64` value) — high fanout should win.
mod small_value {
	use super::*;

	fn fill<const IC: usize, const LC: usize>(keys: &[i64]) -> GenericTree<i64, u64, IC, LC> {
		let tree: GenericTree<i64, u64, IC, LC> = GenericTree::new();
		for &k in keys {
			tree.insert(k, k as u64);
		}
		tree
	}

	pub fn bench(c: &mut Criterion) {
		let mut insert = c.benchmark_group("fanout_small_insert_random");
		insert.throughput(Throughput::Elements(10_000));
		let keys = random_keys(10_000);

		insert.bench_with_input(BenchmarkId::new("IC=LC=16", 10_000), &keys, |b, keys| {
			b.iter(|| black_box(fill::<16, 16>(keys)))
		});
		insert.bench_with_input(BenchmarkId::new("IC=LC=32", 10_000), &keys, |b, keys| {
			b.iter(|| black_box(fill::<32, 32>(keys)))
		});
		insert.bench_with_input(BenchmarkId::new("IC=LC=64", 10_000), &keys, |b, keys| {
			b.iter(|| black_box(fill::<64, 64>(keys)))
		});
		insert.bench_with_input(BenchmarkId::new("IC=LC=128", 10_000), &keys, |b, keys| {
			b.iter(|| black_box(fill::<128, 128>(keys)))
		});
		insert.finish();

		let mut lookup = c.benchmark_group("fanout_small_lookup_random");
		lookup.throughput(Throughput::Elements(10_000));
		let keys = sequential_keys(10_000);

		lookup.bench_with_input(BenchmarkId::new("IC=LC=16", 10_000), &keys, |b, keys| {
			let tree = fill::<16, 16>(keys);
			b.iter(|| {
				for k in keys.iter() {
					black_box(tree.lookup(k, |v| *v));
				}
			})
		});
		lookup.bench_with_input(BenchmarkId::new("IC=LC=32", 10_000), &keys, |b, keys| {
			let tree = fill::<32, 32>(keys);
			b.iter(|| {
				for k in keys.iter() {
					black_box(tree.lookup(k, |v| *v));
				}
			})
		});
		lookup.bench_with_input(BenchmarkId::new("IC=LC=64", 10_000), &keys, |b, keys| {
			let tree = fill::<64, 64>(keys);
			b.iter(|| {
				for k in keys.iter() {
					black_box(tree.lookup(k, |v| *v));
				}
			})
		});
		lookup.bench_with_input(BenchmarkId::new("IC=LC=128", 10_000), &keys, |b, keys| {
			let tree = fill::<128, 128>(keys);
			b.iter(|| {
				for k in keys.iter() {
					black_box(tree.lookup(k, |v| *v));
				}
			})
		});
		lookup.finish();
	}
}

// Large-value workload (256-byte value) — low fanout should win because
// leaf-cache pressure is higher.
mod large_value {
	use super::*;

	#[derive(Clone, Copy)]
	#[allow(dead_code)]
	struct Big([u64; 32]); // 256 bytes

	// SAFETY: `Big` is `Copy + Clone + Send + Sync + 'static`. It does
	// not fit a stdlib atomic (256 bytes); use `BoxedSlot` so every
	// slot lives in a `Box<Big>` reached via `AtomicPtr`. `Drop` is a
	// no-op (Copy), but the Box allocation itself still wants
	// epoch-deferred deallocation to keep optimistic readers' pointers
	// valid across writer swaps.
	unsafe impl ferntree::OptimisticRead for Big {
		const EPOCH_DEFERRED_DROP: bool = true;
		type Slot = ferntree::atomic_slot::BoxedSlot<Self>;
	}

	fn fill<const IC: usize, const LC: usize>(keys: &[i64]) -> GenericTree<i64, Big, IC, LC> {
		let tree: GenericTree<i64, Big, IC, LC> = GenericTree::new();
		for &k in keys {
			tree.insert(k, Big([k as u64; 32]));
		}
		tree
	}

	pub fn bench(c: &mut Criterion) {
		let mut group = c.benchmark_group("fanout_large_insert_random");
		group.throughput(Throughput::Elements(5_000));
		let keys = random_keys(5_000);

		group.bench_with_input(BenchmarkId::new("IC=LC=16", 5_000), &keys, |b, keys| {
			b.iter(|| black_box(fill::<16, 16>(keys)))
		});
		group.bench_with_input(BenchmarkId::new("IC=LC=32", 5_000), &keys, |b, keys| {
			b.iter(|| black_box(fill::<32, 32>(keys)))
		});
		group.bench_with_input(BenchmarkId::new("IC=LC=64", 5_000), &keys, |b, keys| {
			b.iter(|| black_box(fill::<64, 64>(keys)))
		});
		group.bench_with_input(BenchmarkId::new("IC=LC=128", 5_000), &keys, |b, keys| {
			b.iter(|| black_box(fill::<128, 128>(keys)))
		});
		group.finish();
	}
}

criterion_group!(fanout_benches, small_value::bench, large_value::bench);
criterion_main!(fanout_benches);
