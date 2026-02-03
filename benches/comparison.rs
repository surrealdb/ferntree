// Copyright © SurrealDB Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Criterion benchmarks comparing FernTree against other map implementations.
//!
//! This benchmark suite compares:
//! - `ferntree::Tree` - Concurrent B+ tree with optimistic lock coupling
//! - `crossbeam_skiplist::SkipMap` - Lock-free concurrent skip list
//! - `std::collections::BTreeMap` - Standard library B-tree (single-threaded)
//! - `std::collections::HashMap` - Standard library hash map (single-threaded)
//!
//! Single-threaded benchmarks test raw performance without synchronization overhead.
//! Concurrent benchmarks wrap BTreeMap/HashMap in `parking_lot::RwLock`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use crossbeam_skiplist::SkipMap;
use ferntree::Tree;
use parking_lot::RwLock;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;
use std::ops::Bound;
use std::sync::Arc;
use std::thread;

const SEED: u64 = 42;

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate sequential keys from 0 to count-1
fn sequential_keys(count: usize) -> Vec<i64> {
	(0..count as i64).collect()
}

/// Generate random keys using a seeded RNG
fn random_keys(count: usize) -> Vec<i64> {
	let mut rng = StdRng::seed_from_u64(SEED);
	(0..count).map(|_| rng.random()).collect()
}

/// Generate keys that don't exist in a sequential key set
fn missing_keys(count: usize) -> Vec<i64> {
	// Use negative numbers which won't be in sequential 0..N set
	(0..count as i64).map(|i| -(i + 1)).collect()
}

// ============================================================================
// Single-Threaded Insert Benchmarks
// ============================================================================

fn bench_insert_sequential(c: &mut Criterion) {
	let mut group = c.benchmark_group("insert_sequential");

	for count in [1_000, 10_000, 100_000] {
		let keys = sequential_keys(count);
		group.throughput(Throughput::Elements(count as u64));

		// FernTree
		group.bench_with_input(BenchmarkId::new("ferntree", count), &keys, |b, keys| {
			b.iter_batched(
				Tree::new,
				|tree| {
					for &k in keys {
						black_box(tree.insert(k, k));
					}
					tree
				},
				criterion::BatchSize::SmallInput,
			)
		});

		// SkipMap
		group.bench_with_input(BenchmarkId::new("skipmap", count), &keys, |b, keys| {
			b.iter_batched(
				SkipMap::new,
				|map| {
					for &k in keys {
						black_box(map.insert(k, k));
					}
					map
				},
				criterion::BatchSize::SmallInput,
			)
		});

		// BTreeMap
		group.bench_with_input(BenchmarkId::new("btreemap", count), &keys, |b, keys| {
			b.iter_batched(
				BTreeMap::new,
				|mut map| {
					for &k in keys {
						black_box(map.insert(k, k));
					}
					map
				},
				criterion::BatchSize::SmallInput,
			)
		});

		// HashMap
		group.bench_with_input(BenchmarkId::new("hashmap", count), &keys, |b, keys| {
			b.iter_batched(
				HashMap::new,
				|mut map| {
					for &k in keys {
						black_box(map.insert(k, k));
					}
					map
				},
				criterion::BatchSize::SmallInput,
			)
		});
	}
	group.finish();
}

fn bench_insert_random(c: &mut Criterion) {
	let mut group = c.benchmark_group("insert_random");

	for count in [1_000, 10_000, 100_000] {
		let keys = random_keys(count);
		group.throughput(Throughput::Elements(count as u64));

		// FernTree
		group.bench_with_input(BenchmarkId::new("ferntree", count), &keys, |b, keys| {
			b.iter_batched(
				Tree::new,
				|tree| {
					for &k in keys {
						black_box(tree.insert(k, k));
					}
					tree
				},
				criterion::BatchSize::SmallInput,
			)
		});

		// SkipMap
		group.bench_with_input(BenchmarkId::new("skipmap", count), &keys, |b, keys| {
			b.iter_batched(
				SkipMap::new,
				|map| {
					for &k in keys {
						black_box(map.insert(k, k));
					}
					map
				},
				criterion::BatchSize::SmallInput,
			)
		});

		// BTreeMap
		group.bench_with_input(BenchmarkId::new("btreemap", count), &keys, |b, keys| {
			b.iter_batched(
				BTreeMap::new,
				|mut map| {
					for &k in keys {
						black_box(map.insert(k, k));
					}
					map
				},
				criterion::BatchSize::SmallInput,
			)
		});

		// HashMap
		group.bench_with_input(BenchmarkId::new("hashmap", count), &keys, |b, keys| {
			b.iter_batched(
				HashMap::new,
				|mut map| {
					for &k in keys {
						black_box(map.insert(k, k));
					}
					map
				},
				criterion::BatchSize::SmallInput,
			)
		});
	}
	group.finish();
}

// ============================================================================
// Single-Threaded Lookup Benchmarks
// ============================================================================

fn bench_lookup_hit(c: &mut Criterion) {
	let mut group = c.benchmark_group("lookup_hit");

	for count in [1_000, 10_000, 100_000] {
		let keys = sequential_keys(count);
		let lookup_count = 1000.min(count);
		let lookup_keys: Vec<i64> = keys[..lookup_count].to_vec();

		// Pre-populate data structures
		let ferntree: Tree<i64, i64> = Tree::new();
		let skipmap: SkipMap<i64, i64> = SkipMap::new();
		let mut btreemap: BTreeMap<i64, i64> = BTreeMap::new();
		let mut hashmap: HashMap<i64, i64> = HashMap::new();

		for &k in &keys {
			ferntree.insert(k, k);
			skipmap.insert(k, k);
			btreemap.insert(k, k);
			hashmap.insert(k, k);
		}

		group.throughput(Throughput::Elements(lookup_count as u64));

		// FernTree
		group.bench_with_input(BenchmarkId::new("ferntree", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(ferntree.get(&k));
				}
			})
		});

		// SkipMap
		group.bench_with_input(BenchmarkId::new("skipmap", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(skipmap.get(&k).map(|e| *e.value()));
				}
			})
		});

		// BTreeMap
		group.bench_with_input(BenchmarkId::new("btreemap", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(btreemap.get(&k));
				}
			})
		});

		// HashMap
		group.bench_with_input(BenchmarkId::new("hashmap", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(hashmap.get(&k));
				}
			})
		});
	}
	group.finish();
}

fn bench_lookup_miss(c: &mut Criterion) {
	let mut group = c.benchmark_group("lookup_miss");

	for count in [1_000, 10_000, 100_000] {
		let keys = sequential_keys(count);
		let missing = missing_keys(1000);

		// Pre-populate data structures
		let ferntree: Tree<i64, i64> = Tree::new();
		let skipmap: SkipMap<i64, i64> = SkipMap::new();
		let mut btreemap: BTreeMap<i64, i64> = BTreeMap::new();
		let mut hashmap: HashMap<i64, i64> = HashMap::new();

		for &k in &keys {
			ferntree.insert(k, k);
			skipmap.insert(k, k);
			btreemap.insert(k, k);
			hashmap.insert(k, k);
		}

		group.throughput(Throughput::Elements(missing.len() as u64));

		// FernTree
		group.bench_with_input(BenchmarkId::new("ferntree", count), &missing, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(ferntree.get(&k));
				}
			})
		});

		// SkipMap
		group.bench_with_input(BenchmarkId::new("skipmap", count), &missing, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(skipmap.get(&k).map(|e| *e.value()));
				}
			})
		});

		// BTreeMap
		group.bench_with_input(BenchmarkId::new("btreemap", count), &missing, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(btreemap.get(&k));
				}
			})
		});

		// HashMap
		group.bench_with_input(BenchmarkId::new("hashmap", count), &missing, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(hashmap.get(&k));
				}
			})
		});
	}
	group.finish();
}

// ============================================================================
// Single-Threaded Remove Benchmarks
// ============================================================================

fn bench_remove(c: &mut Criterion) {
	let mut group = c.benchmark_group("remove");

	for count in [1_000, 10_000, 100_000] {
		let keys = sequential_keys(count);
		let remove_count = count / 10; // Remove 10% of entries
		let remove_keys: Vec<i64> = keys[..remove_count].to_vec();

		group.throughput(Throughput::Elements(remove_count as u64));

		// FernTree
		group.bench_with_input(
			BenchmarkId::new("ferntree", count),
			&remove_keys,
			|b, remove_keys| {
				b.iter_batched(
					|| {
						let tree = Tree::new();
						for &k in &keys {
							tree.insert(k, k);
						}
						tree
					},
					|tree| {
						for &k in remove_keys {
							black_box(tree.remove(&k));
						}
						tree
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);

		// SkipMap
		group.bench_with_input(
			BenchmarkId::new("skipmap", count),
			&remove_keys,
			|b, remove_keys| {
				b.iter_batched(
					|| {
						let map = SkipMap::new();
						for &k in &keys {
							map.insert(k, k);
						}
						map
					},
					|map| {
						for &k in remove_keys {
							black_box(map.remove(&k));
						}
						map
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);

		// BTreeMap
		group.bench_with_input(
			BenchmarkId::new("btreemap", count),
			&remove_keys,
			|b, remove_keys| {
				b.iter_batched(
					|| {
						let mut map = BTreeMap::new();
						for &k in &keys {
							map.insert(k, k);
						}
						map
					},
					|mut map| {
						for &k in remove_keys {
							black_box(map.remove(&k));
						}
						map
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);

		// HashMap
		group.bench_with_input(
			BenchmarkId::new("hashmap", count),
			&remove_keys,
			|b, remove_keys| {
				b.iter_batched(
					|| {
						let mut map = HashMap::new();
						for &k in &keys {
							map.insert(k, k);
						}
						map
					},
					|mut map| {
						for &k in remove_keys {
							black_box(map.remove(&k));
						}
						map
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);
	}
	group.finish();
}

// ============================================================================
// Single-Threaded Range Benchmarks (ordered maps only)
// ============================================================================

fn bench_range(c: &mut Criterion) {
	let mut group = c.benchmark_group("range");

	for count in [1_000, 10_000, 100_000, 1_000_000] {
		let keys = sequential_keys(count);

		// Pre-populate data structures
		let ferntree: Tree<i64, i64> = Tree::new();
		let skipmap: SkipMap<i64, i64> = SkipMap::new();
		let mut btreemap: BTreeMap<i64, i64> = BTreeMap::new();

		for &k in &keys {
			ferntree.insert(k, k);
			skipmap.insert(k, k);
			btreemap.insert(k, k);
		}

		// Range covers 10% of entries in the middle
		let range_size = count / 10;
		let start = (count / 2 - range_size / 2) as i64;
		let end = start + range_size as i64;

		group.throughput(Throughput::Elements(range_size as u64));

		// FernTree (uses manual next() method, not Iterator trait)
		group.bench_function(BenchmarkId::new("ferntree", count), |b| {
			b.iter(|| {
				let mut sum = 0i64;
				let mut range = ferntree.range(Bound::Included(&start), Bound::Excluded(&end));
				while let Some((k, v)) = range.next() {
					sum = sum.wrapping_add(*k).wrapping_add(*v);
				}
				black_box(sum)
			})
		});

		// SkipMap
		group.bench_function(BenchmarkId::new("skipmap", count), |b| {
			b.iter(|| {
				let mut sum = 0i64;
				for entry in skipmap.range(start..end) {
					sum = sum.wrapping_add(*entry.key()).wrapping_add(*entry.value());
				}
				black_box(sum)
			})
		});

		// BTreeMap
		group.bench_function(BenchmarkId::new("btreemap", count), |b| {
			b.iter(|| {
				let mut sum = 0i64;
				for (&k, &v) in btreemap.range(start..end) {
					sum = sum.wrapping_add(k).wrapping_add(v);
				}
				black_box(sum)
			})
		});

		// Note: HashMap does not support range iteration (unordered)
	}
	group.finish();
}

fn bench_raw_iter(c: &mut Criterion) {
	let mut group = c.benchmark_group("iterator");

	for count in [1_000, 10_000, 100_000, 1_000_000] {
		let keys = sequential_keys(count);

		// Pre-populate data structures
		let ferntree: Tree<i64, i64> = Tree::new();
		let skipmap: SkipMap<i64, i64> = SkipMap::new();
		let mut btreemap: BTreeMap<i64, i64> = BTreeMap::new();

		for &k in &keys {
			ferntree.insert(k, k);
			skipmap.insert(k, k);
			btreemap.insert(k, k);
		}

		group.throughput(Throughput::Elements(count as u64));

		// FernTree raw_iter
		group.bench_function(BenchmarkId::new("ferntree", count), |b| {
			b.iter(|| {
				let mut sum = 0i64;
				let mut iter = ferntree.raw_iter();
				iter.seek_to_first();
				while let Some((k, v)) = iter.next() {
					sum = sum.wrapping_add(*k).wrapping_add(*v);
				}
				black_box(sum)
			})
		});

		// SkipMap iterator
		group.bench_function(BenchmarkId::new("skipmap", count), |b| {
			b.iter(|| {
				let mut sum = 0i64;
				for entry in skipmap.iter() {
					sum = sum.wrapping_add(*entry.key()).wrapping_add(*entry.value());
				}
				black_box(sum)
			})
		});

		// BTreeMap iterator
		group.bench_function(BenchmarkId::new("btreemap", count), |b| {
			b.iter(|| {
				let mut sum = 0i64;
				for (&k, &v) in btreemap.iter() {
					sum = sum.wrapping_add(k).wrapping_add(v);
				}
				black_box(sum)
			})
		});

		// Note: HashMap does not support ordered iteration
	}
	group.finish();
}

// ============================================================================
// Concurrent Benchmarks
// ============================================================================

fn bench_concurrent_readers(c: &mut Criterion) {
	let mut group = c.benchmark_group("concurrent_readers");

	let cpu_cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
	let thread_counts = [1, 4, cpu_cores];

	for count in [10_000, 100_000] {
		let keys = sequential_keys(count);
		let lookup_count = 1000;
		let lookup_keys: Vec<i64> = keys[..lookup_count].to_vec();

		// Pre-populate data structures
		let ferntree: Arc<Tree<i64, i64>> = Arc::new(Tree::new());
		let skipmap: Arc<SkipMap<i64, i64>> = Arc::new(SkipMap::new());
		let btreemap: Arc<RwLock<BTreeMap<i64, i64>>> = Arc::new(RwLock::new(BTreeMap::new()));
		let hashmap: Arc<RwLock<HashMap<i64, i64>>> = Arc::new(RwLock::new(HashMap::new()));

		for &k in &keys {
			ferntree.insert(k, k);
			skipmap.insert(k, k);
			btreemap.write().insert(k, k);
			hashmap.write().insert(k, k);
		}

		for &num_threads in &thread_counts {
			let total_ops = lookup_count * num_threads;
			group.throughput(Throughput::Elements(total_ops as u64));

			// FernTree
			group.bench_with_input(
				BenchmarkId::new(format!("ferntree/{}t", num_threads), count),
				&lookup_keys,
				|b, keys| {
					b.iter(|| {
						let handles: Vec<_> = (0..num_threads)
							.map(|_| {
								let tree = Arc::clone(&ferntree);
								let keys = keys.clone();
								thread::spawn(move || {
									for &k in &keys {
										black_box(tree.get(&k));
									}
								})
							})
							.collect();
						for h in handles {
							h.join().unwrap();
						}
					})
				},
			);

			// SkipMap
			group.bench_with_input(
				BenchmarkId::new(format!("skipmap/{}t", num_threads), count),
				&lookup_keys,
				|b, keys| {
					b.iter(|| {
						let handles: Vec<_> = (0..num_threads)
							.map(|_| {
								let map = Arc::clone(&skipmap);
								let keys = keys.clone();
								thread::spawn(move || {
									for &k in &keys {
										black_box(map.get(&k).map(|e| *e.value()));
									}
								})
							})
							.collect();
						for h in handles {
							h.join().unwrap();
						}
					})
				},
			);

			// BTreeMap with RwLock
			group.bench_with_input(
				BenchmarkId::new(format!("btreemap_rwlock/{}t", num_threads), count),
				&lookup_keys,
				|b, keys| {
					b.iter(|| {
						let handles: Vec<_> = (0..num_threads)
							.map(|_| {
								let map = Arc::clone(&btreemap);
								let keys = keys.clone();
								thread::spawn(move || {
									for &k in &keys {
										let guard = map.read();
										black_box(guard.get(&k).copied());
									}
								})
							})
							.collect();
						for h in handles {
							h.join().unwrap();
						}
					})
				},
			);

			// HashMap with RwLock
			group.bench_with_input(
				BenchmarkId::new(format!("hashmap_rwlock/{}t", num_threads), count),
				&lookup_keys,
				|b, keys| {
					b.iter(|| {
						let handles: Vec<_> = (0..num_threads)
							.map(|_| {
								let map = Arc::clone(&hashmap);
								let keys = keys.clone();
								thread::spawn(move || {
									for &k in &keys {
										let guard = map.read();
										black_box(guard.get(&k).copied());
									}
								})
							})
							.collect();
						for h in handles {
							h.join().unwrap();
						}
					})
				},
			);
		}
	}
	group.finish();
}

fn bench_concurrent_writers(c: &mut Criterion) {
	let mut group = c.benchmark_group("concurrent_writers");

	let cpu_cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
	let thread_counts = [1, 4, cpu_cores];

	for &num_threads in &thread_counts {
		let ops_per_thread = 1000;
		let total_ops = ops_per_thread * num_threads;
		group.throughput(Throughput::Elements(total_ops as u64));

		// Pre-generate unique keys per thread to avoid conflicts
		let thread_keys: Vec<Vec<i64>> = (0..num_threads)
			.map(|t| (0..ops_per_thread).map(|i| (t * ops_per_thread + i) as i64).collect())
			.collect();

		// FernTree
		group.bench_with_input(
			BenchmarkId::new("ferntree", format!("{}t", num_threads)),
			&thread_keys,
			|b, thread_keys| {
				b.iter_batched(
					|| Arc::new(Tree::new()),
					|tree| {
						let handles: Vec<_> = thread_keys
							.iter()
							.map(|keys| {
								let tree = Arc::clone(&tree);
								let keys = keys.clone();
								thread::spawn(move || {
									for k in keys {
										black_box(tree.insert(k, k));
									}
								})
							})
							.collect();
						for h in handles {
							h.join().unwrap();
						}
						tree
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);

		// SkipMap
		group.bench_with_input(
			BenchmarkId::new("skipmap", format!("{}t", num_threads)),
			&thread_keys,
			|b, thread_keys| {
				b.iter_batched(
					|| Arc::new(SkipMap::new()),
					|map| {
						let handles: Vec<_> = thread_keys
							.iter()
							.map(|keys| {
								let map = Arc::clone(&map);
								let keys = keys.clone();
								thread::spawn(move || {
									for k in keys {
										black_box(map.insert(k, k));
									}
								})
							})
							.collect();
						for h in handles {
							h.join().unwrap();
						}
						map
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);

		// BTreeMap with RwLock
		group.bench_with_input(
			BenchmarkId::new("btreemap_rwlock", format!("{}t", num_threads)),
			&thread_keys,
			|b, thread_keys| {
				b.iter_batched(
					|| Arc::new(RwLock::new(BTreeMap::new())),
					|map| {
						let handles: Vec<_> = thread_keys
							.iter()
							.map(|keys| {
								let map = Arc::clone(&map);
								let keys = keys.clone();
								thread::spawn(move || {
									for k in keys {
										let mut guard = map.write();
										black_box(guard.insert(k, k));
									}
								})
							})
							.collect();
						for h in handles {
							h.join().unwrap();
						}
						map
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);

		// HashMap with RwLock
		group.bench_with_input(
			BenchmarkId::new("hashmap_rwlock", format!("{}t", num_threads)),
			&thread_keys,
			|b, thread_keys| {
				b.iter_batched(
					|| Arc::new(RwLock::new(HashMap::new())),
					|map| {
						let handles: Vec<_> = thread_keys
							.iter()
							.map(|keys| {
								let map = Arc::clone(&map);
								let keys = keys.clone();
								thread::spawn(move || {
									for k in keys {
										let mut guard = map.write();
										black_box(guard.insert(k, k));
									}
								})
							})
							.collect();
						for h in handles {
							h.join().unwrap();
						}
						map
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);
	}
	group.finish();
}

fn bench_concurrent_mixed(c: &mut Criterion) {
	let mut group = c.benchmark_group("concurrent_mixed");

	let cpu_cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
	// Use half readers, half writers
	let configs = [
		("2r_2w", 2, 2),
		("4r_4w", 4, 4),
		(&format!("{}r_{}w", cpu_cores / 2, cpu_cores / 2), cpu_cores / 2, cpu_cores / 2),
	];

	for count in [10_000, 100_000] {
		let keys = sequential_keys(count);
		let ops_per_thread = 500;

		for (config_name, num_readers, num_writers) in &configs {
			if *num_readers == 0 || *num_writers == 0 {
				continue;
			}

			let total_ops = ops_per_thread * (num_readers + num_writers);
			group.throughput(Throughput::Elements(total_ops as u64));

			// Pre-generate read keys and write keys
			let read_keys: Vec<i64> = keys[..ops_per_thread].to_vec();
			let write_keys: Vec<Vec<i64>> = (0..*num_writers)
				.map(|w| {
					(0..ops_per_thread)
						.map(|i| (count as i64) + (w * ops_per_thread + i) as i64)
						.collect()
				})
				.collect();

			// FernTree
			group.bench_function(
				BenchmarkId::new(format!("ferntree/{}", config_name), count),
				|b| {
					b.iter_batched(
						|| {
							let tree = Arc::new(Tree::new());
							for &k in &keys {
								tree.insert(k, k);
							}
							tree
						},
						|tree| {
							let mut handles = Vec::new();

							// Spawn reader threads
							for _ in 0..*num_readers {
								let tree = Arc::clone(&tree);
								let keys = read_keys.clone();
								handles.push(thread::spawn(move || {
									for &k in &keys {
										black_box(tree.get(&k));
									}
								}));
							}

							// Spawn writer threads
							for keys in write_keys.iter().take(*num_writers) {
								let tree = Arc::clone(&tree);
								let keys = keys.clone();
								handles.push(thread::spawn(move || {
									for k in keys {
										black_box(tree.insert(k, k));
									}
								}));
							}

							for h in handles {
								h.join().unwrap();
							}
							tree
						},
						criterion::BatchSize::SmallInput,
					)
				},
			);

			// SkipMap
			group.bench_function(
				BenchmarkId::new(format!("skipmap/{}", config_name), count),
				|b| {
					b.iter_batched(
						|| {
							let map = Arc::new(SkipMap::new());
							for &k in &keys {
								map.insert(k, k);
							}
							map
						},
						|map| {
							let mut handles = Vec::new();

							// Spawn reader threads
							for _ in 0..*num_readers {
								let map = Arc::clone(&map);
								let keys = read_keys.clone();
								handles.push(thread::spawn(move || {
									for &k in &keys {
										black_box(map.get(&k).map(|e| *e.value()));
									}
								}));
							}

							// Spawn writer threads
							for keys in write_keys.iter().take(*num_writers) {
								let map = Arc::clone(&map);
								let keys = keys.clone();
								handles.push(thread::spawn(move || {
									for k in keys {
										black_box(map.insert(k, k));
									}
								}));
							}

							for h in handles {
								h.join().unwrap();
							}
							map
						},
						criterion::BatchSize::SmallInput,
					)
				},
			);

			// BTreeMap with RwLock
			group.bench_function(
				BenchmarkId::new(format!("btreemap_rwlock/{}", config_name), count),
				|b| {
					b.iter_batched(
						|| {
							let map = Arc::new(RwLock::new(BTreeMap::new()));
							{
								let mut guard = map.write();
								for &k in &keys {
									guard.insert(k, k);
								}
							}
							map
						},
						|map| {
							let mut handles = Vec::new();

							// Spawn reader threads
							for _ in 0..*num_readers {
								let map = Arc::clone(&map);
								let keys = read_keys.clone();
								handles.push(thread::spawn(move || {
									for &k in &keys {
										let guard = map.read();
										black_box(guard.get(&k).copied());
									}
								}));
							}

							// Spawn writer threads
							for keys in write_keys.iter().take(*num_writers) {
								let map = Arc::clone(&map);
								let keys = keys.clone();
								handles.push(thread::spawn(move || {
									for k in keys {
										let mut guard = map.write();
										black_box(guard.insert(k, k));
									}
								}));
							}

							for h in handles {
								h.join().unwrap();
							}
							map
						},
						criterion::BatchSize::SmallInput,
					)
				},
			);

			// HashMap with RwLock
			group.bench_function(
				BenchmarkId::new(format!("hashmap_rwlock/{}", config_name), count),
				|b| {
					b.iter_batched(
						|| {
							let map = Arc::new(RwLock::new(HashMap::new()));
							{
								let mut guard = map.write();
								for &k in &keys {
									guard.insert(k, k);
								}
							}
							map
						},
						|map| {
							let mut handles = Vec::new();

							// Spawn reader threads
							for _ in 0..*num_readers {
								let map = Arc::clone(&map);
								let keys = read_keys.clone();
								handles.push(thread::spawn(move || {
									for &k in &keys {
										let guard = map.read();
										black_box(guard.get(&k).copied());
									}
								}));
							}

							// Spawn writer threads
							for keys in write_keys.iter().take(*num_writers) {
								let map = Arc::clone(&map);
								let keys = keys.clone();
								handles.push(thread::spawn(move || {
									for k in keys {
										let mut guard = map.write();
										black_box(guard.insert(k, k));
									}
								}));
							}

							for h in handles {
								h.join().unwrap();
							}
							map
						},
						criterion::BatchSize::SmallInput,
					)
				},
			);
		}
	}
	group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
	single_threaded_benches,
	bench_insert_sequential,
	bench_insert_random,
	bench_lookup_hit,
	bench_lookup_miss,
	bench_remove,
	bench_range,
	bench_raw_iter,
);

criterion_group!(
	concurrent_benches,
	bench_concurrent_readers,
	bench_concurrent_writers,
	bench_concurrent_mixed,
);

criterion_main!(single_threaded_benches, concurrent_benches);
