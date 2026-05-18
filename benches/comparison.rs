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

use bytes::Bytes;
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

// `bytes::Bytes` is foreign, so we wrap it in a local newtype (orphan
// rules). `BytesBlob` is a zero-cost wrapper: same size, same Clone cost
// (Arc-bump), same Ord (lex byte order). `EPOCH_DEFERRED_DROP = true`
// keeps the inner buffer alive across a reader's borrow window.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[repr(transparent)]
struct BytesBlob(Bytes);

impl Ord for BytesBlob {
	#[inline]
	fn cmp(&self, other: &Self) -> std::cmp::Ordering {
		self.0.as_ref().cmp(other.0.as_ref())
	}
}
impl PartialOrd for BytesBlob {
	#[inline]
	fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
		Some(self.cmp(other))
	}
}

// SAFETY: `BytesBlob` is `Send + Sync + Clone + 'static`. Wrapped
// `bytes::Bytes` is itself refcounted with atomic Clone semantics, so
// the boxed-slot atomic-load + Clone pair is sound.
unsafe impl ferntree::OptimisticRead for BytesBlob {
	const EPOCH_DEFERRED_DROP: bool = true;
	type Slot = ferntree::atomic_slot::BoxedSlot<Self>;
}

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

/// Sequential `String` keys, zero-padded so lex order == numeric order.
fn sequential_string_keys(count: usize) -> Vec<String> {
	(0..count).map(|i| format!("k{:010}", i)).collect()
}

/// Random `String` keys (seeded RNG → 10-digit zero-padded for stable width).
fn random_string_keys(count: usize) -> Vec<String> {
	let mut rng = StdRng::seed_from_u64(SEED);
	(0..count).map(|_| format!("k{:010}", rng.random::<u64>())).collect()
}

/// Sequential `Vec<u8>` keys (10-byte big-endian-encoded u64 prefix).
fn sequential_bytes_keys(count: usize) -> Vec<Vec<u8>> {
	(0..count as u64)
		.map(|i| {
			let mut v = Vec::with_capacity(10);
			v.extend_from_slice(b"k");
			v.extend_from_slice(&i.to_be_bytes());
			v.push(0);
			v
		})
		.collect()
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

		// FernTree (shared-lock path)
		group.bench_with_input(BenchmarkId::new("ferntree", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(ferntree.get(&k));
				}
			})
		});

		// FernTree (optimistic fast path)
		group.bench_with_input(
			BenchmarkId::new("ferntree_optimistic", count),
			&lookup_keys,
			|b, keys| {
				b.iter(|| {
					for &k in keys {
						black_box(ferntree.get_optimistic(&k));
					}
				})
			},
		);

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

		// FernTree (shared-lock path)
		group.bench_with_input(BenchmarkId::new("ferntree", count), &missing, |b, keys| {
			b.iter(|| {
				for &k in keys {
					black_box(ferntree.get(&k));
				}
			})
		});

		// FernTree (optimistic fast path)
		group.bench_with_input(
			BenchmarkId::new("ferntree_optimistic", count),
			&missing,
			|b, keys| {
				b.iter(|| {
					for &k in keys {
						black_box(ferntree.get_optimistic(&k));
					}
				})
			},
		);

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
// Single-Threaded String / Vec<u8> Benchmarks (BoxedSlot path)
// ============================================================================
//
// These exercise the `BoxedSlot<T>` storage path that primitives like
// `i64` never hit. They measure the cost of:
//   - heap-allocated K/V load_into (boxed atomic pointer + clone)
//   - String / Vec<u8> comparison during binary search
// and are the baseline against which the relaxed-atomic and load_into
// changes are measured.

fn bench_insert_random_string(c: &mut Criterion) {
	let mut group = c.benchmark_group("insert_random_string");

	for count in [1_000, 10_000] {
		let keys = random_string_keys(count);
		group.throughput(Throughput::Elements(count as u64));

		group.bench_with_input(BenchmarkId::new("ferntree", count), &keys, |b, keys| {
			b.iter_batched(
				Tree::<String, String>::new,
				|tree| {
					for k in keys {
						black_box(tree.insert(k.clone(), k.clone()));
					}
					tree
				},
				criterion::BatchSize::SmallInput,
			)
		});

		group.bench_with_input(BenchmarkId::new("skipmap", count), &keys, |b, keys| {
			b.iter_batched(
				SkipMap::<String, String>::new,
				|map| {
					for k in keys {
						black_box(map.insert(k.clone(), k.clone()));
					}
					map
				},
				criterion::BatchSize::SmallInput,
			)
		});

		group.bench_with_input(BenchmarkId::new("btreemap", count), &keys, |b, keys| {
			b.iter_batched(
				BTreeMap::<String, String>::new,
				|mut map| {
					for k in keys {
						black_box(map.insert(k.clone(), k.clone()));
					}
					map
				},
				criterion::BatchSize::SmallInput,
			)
		});

		group.bench_with_input(BenchmarkId::new("hashmap", count), &keys, |b, keys| {
			b.iter_batched(
				HashMap::<String, String>::new,
				|mut map| {
					for k in keys {
						black_box(map.insert(k.clone(), k.clone()));
					}
					map
				},
				criterion::BatchSize::SmallInput,
			)
		});
	}
	group.finish();
}

fn bench_lookup_hit_string(c: &mut Criterion) {
	let mut group = c.benchmark_group("lookup_hit_string");

	for count in [1_000, 10_000, 100_000] {
		let keys = sequential_string_keys(count);
		let lookup_count = 1000.min(count);
		let lookup_keys: Vec<String> = keys[..lookup_count].to_vec();

		let ferntree: Tree<String, String> = Tree::new();
		let skipmap: SkipMap<String, String> = SkipMap::new();
		let mut btreemap: BTreeMap<String, String> = BTreeMap::new();
		let mut hashmap: HashMap<String, String> = HashMap::new();

		for k in &keys {
			ferntree.insert(k.clone(), k.clone());
			skipmap.insert(k.clone(), k.clone());
			btreemap.insert(k.clone(), k.clone());
			hashmap.insert(k.clone(), k.clone());
		}

		group.throughput(Throughput::Elements(lookup_count as u64));

		group.bench_with_input(BenchmarkId::new("ferntree", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(ferntree.lookup(k, |v| v.len()));
				}
			})
		});

		group.bench_with_input(
			BenchmarkId::new("ferntree_optimistic", count),
			&lookup_keys,
			|b, keys| {
				b.iter(|| {
					for k in keys {
						black_box(ferntree.lookup_optimistic(k, |v| v.len()));
					}
				})
			},
		);

		group.bench_with_input(BenchmarkId::new("skipmap", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(skipmap.get(k).map(|e| e.value().len()));
				}
			})
		});

		group.bench_with_input(BenchmarkId::new("btreemap", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(btreemap.get(k).map(|v| v.len()));
				}
			})
		});

		group.bench_with_input(BenchmarkId::new("hashmap", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(hashmap.get(k).map(|v| v.len()));
				}
			})
		});
	}
	group.finish();
}

fn bench_lookup_hit_bytes(c: &mut Criterion) {
	let mut group = c.benchmark_group("lookup_hit_bytes");

	for count in [1_000, 10_000] {
		let keys = sequential_bytes_keys(count);
		let lookup_count = 1000.min(count);
		let lookup_keys: Vec<Vec<u8>> = keys[..lookup_count].to_vec();

		let ferntree: Tree<Vec<u8>, Vec<u8>> = Tree::new();
		let skipmap: SkipMap<Vec<u8>, Vec<u8>> = SkipMap::new();
		let mut btreemap: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();
		let mut hashmap: HashMap<Vec<u8>, Vec<u8>> = HashMap::new();

		for k in &keys {
			ferntree.insert(k.clone(), k.clone());
			skipmap.insert(k.clone(), k.clone());
			btreemap.insert(k.clone(), k.clone());
			hashmap.insert(k.clone(), k.clone());
		}

		group.throughput(Throughput::Elements(lookup_count as u64));

		group.bench_with_input(BenchmarkId::new("ferntree", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(ferntree.lookup(k.as_slice(), |v| v.len()));
				}
			})
		});

		group.bench_with_input(
			BenchmarkId::new("ferntree_optimistic", count),
			&lookup_keys,
			|b, keys| {
				b.iter(|| {
					for k in keys {
						black_box(ferntree.lookup_optimistic(k.as_slice(), |v| v.len()));
					}
				})
			},
		);

		group.bench_with_input(BenchmarkId::new("skipmap", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(skipmap.get(k.as_slice()).map(|e| e.value().len()));
				}
			})
		});

		group.bench_with_input(BenchmarkId::new("btreemap", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(btreemap.get(k.as_slice()).map(|v| v.len()));
				}
			})
		});

		group.bench_with_input(BenchmarkId::new("hashmap", count), &lookup_keys, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(hashmap.get(k.as_slice()).map(|v| v.len()));
				}
			})
		});
	}
	group.finish();
}

// ============================================================================
// Vec<u8> vs bytes::Bytes (side-by-side)
// ============================================================================
//
// Compares the boxed-K/V storage cost for two refcount profiles:
//   - `Vec<u8>`  — Clone allocates + memcpy of the inner buffer
//   - `Bytes`    — Clone bumps an Arc refcount (no buffer copy)
// Same 10-byte payload in both cases; only the Clone shape differs.

fn bench_vec_vs_bytes_lookup_hit(c: &mut Criterion) {
	let mut group = c.benchmark_group("vec_vs_bytes_lookup_hit");

	for count in [1_000, 10_000] {
		let vec_keys = sequential_bytes_keys(count);
		let bytes_keys: Vec<BytesBlob> =
			vec_keys.iter().map(|v| BytesBlob(Bytes::copy_from_slice(v))).collect();
		let lookup_count = 1000.min(count);

		let tree_vec: Tree<Vec<u8>, Vec<u8>> = Tree::new();
		let tree_bytes: Tree<BytesBlob, BytesBlob> = Tree::new();

		for (vk, bk) in vec_keys.iter().zip(bytes_keys.iter()) {
			tree_vec.insert(vk.clone(), vk.clone());
			tree_bytes.insert(bk.clone(), bk.clone());
		}

		let vec_lookup: Vec<Vec<u8>> = vec_keys[..lookup_count].to_vec();
		let bytes_lookup: Vec<BytesBlob> = bytes_keys[..lookup_count].to_vec();

		group.throughput(Throughput::Elements(lookup_count as u64));

		group.bench_with_input(BenchmarkId::new("vec_u8/ferntree", count), &vec_lookup, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(tree_vec.lookup(k.as_slice(), |v| v.len()));
				}
			})
		});
		group.bench_with_input(
			BenchmarkId::new("vec_u8/ferntree_optimistic", count),
			&vec_lookup,
			|b, keys| {
				b.iter(|| {
					for k in keys {
						black_box(tree_vec.lookup_optimistic(k.as_slice(), |v| v.len()));
					}
				})
			},
		);
		group.bench_with_input(BenchmarkId::new("bytes/ferntree", count), &bytes_lookup, |b, keys| {
			b.iter(|| {
				for k in keys {
					black_box(tree_bytes.lookup(k, |v| v.0.len()));
				}
			})
		});
		group.bench_with_input(
			BenchmarkId::new("bytes/ferntree_optimistic", count),
			&bytes_lookup,
			|b, keys| {
				b.iter(|| {
					for k in keys {
						black_box(tree_bytes.lookup_optimistic(k, |v| v.0.len()));
					}
				})
			},
		);
	}
	group.finish();
}

fn bench_vec_vs_bytes_insert_random(c: &mut Criterion) {
	let mut group = c.benchmark_group("vec_vs_bytes_insert_random");

	for count in [1_000, 10_000] {
		// Random 10-byte payloads from the same seeded RNG.
		let mut rng = StdRng::seed_from_u64(SEED);
		let payloads: Vec<[u8; 10]> = (0..count)
			.map(|_| {
				let mut p = [0u8; 10];
				rng.fill(&mut p);
				p
			})
			.collect();
		let vec_keys: Vec<Vec<u8>> = payloads.iter().map(|p| p.to_vec()).collect();
		let bytes_keys: Vec<BytesBlob> =
			payloads.iter().map(|p| BytesBlob(Bytes::copy_from_slice(p))).collect();

		group.throughput(Throughput::Elements(count as u64));

		group.bench_with_input(
			BenchmarkId::new("vec_u8/ferntree", count),
			&vec_keys,
			|b, keys| {
				b.iter_batched(
					Tree::<Vec<u8>, Vec<u8>>::new,
					|tree| {
						for k in keys {
							black_box(tree.insert(k.clone(), k.clone()));
						}
						tree
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);

		group.bench_with_input(
			BenchmarkId::new("bytes/ferntree", count),
			&bytes_keys,
			|b, keys| {
				b.iter_batched(
					Tree::<BytesBlob, BytesBlob>::new,
					|tree| {
						for k in keys {
							black_box(tree.insert(k.clone(), k.clone()));
						}
						tree
					},
					criterion::BatchSize::SmallInput,
				)
			},
		);
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

		// FernTree raw_iter with batch leaf iteration
		group.bench_function(BenchmarkId::new("ferntree-batch", count), |b| {
			b.iter(|| {
				let mut sum = 0i64;
				let mut iter = ferntree.raw_iter();
				iter.seek_to_first();
				loop {
					let has_more = iter.for_each_in_leaf(|k, v| {
						sum = sum.wrapping_add(*k).wrapping_add(*v);
					});
					if !has_more {
						break;
					}
					// Move to next leaf - next() returns first entry of new leaf
					if let Some((k, v)) = iter.next() {
						sum = sum.wrapping_add(*k).wrapping_add(*v);
					} else {
						break;
					}
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

			// FernTree (shared-lock path)
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

			// FernTree (optimistic fast path)
			group.bench_with_input(
				BenchmarkId::new(format!("ferntree_optimistic/{}t", num_threads), count),
				&lookup_keys,
				|b, keys| {
					b.iter(|| {
						let handles: Vec<_> = (0..num_threads)
							.map(|_| {
								let tree = Arc::clone(&ferntree);
								let keys = keys.clone();
								thread::spawn(move || {
									for &k in &keys {
										black_box(tree.get_optimistic(&k));
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

fn bench_concurrent_mixed_string(c: &mut Criterion) {
	let mut group = c.benchmark_group("concurrent_mixed_string");

	let cpu_cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
	let configs = [
		("2r_2w", 2, 2),
		("4r_4w", 4, 4),
		(&format!("{}r_{}w", cpu_cores / 2, cpu_cores / 2), cpu_cores / 2, cpu_cores / 2),
	];

	for count in [10_000] {
		let keys = sequential_string_keys(count);
		let ops_per_thread = 500;

		for (config_name, num_readers, num_writers) in &configs {
			if *num_readers == 0 || *num_writers == 0 {
				continue;
			}

			let total_ops = ops_per_thread * (num_readers + num_writers);
			group.throughput(Throughput::Elements(total_ops as u64));

			let read_keys: Vec<String> = keys[..ops_per_thread].to_vec();
			let write_keys: Vec<Vec<String>> = (0..*num_writers)
				.map(|w| {
					(0..ops_per_thread)
						.map(|i| format!("w{:010}", w * ops_per_thread + i + count))
						.collect()
				})
				.collect();

			group.bench_function(
				BenchmarkId::new(format!("ferntree/{}", config_name), count),
				|b| {
					b.iter_batched(
						|| {
							let tree = Arc::new(Tree::<String, String>::new());
							for k in &keys {
								tree.insert(k.clone(), k.clone());
							}
							tree
						},
						|tree| {
							let mut handles = Vec::new();
							for _ in 0..*num_readers {
								let tree = Arc::clone(&tree);
								let keys = read_keys.clone();
								handles.push(thread::spawn(move || {
									for k in &keys {
										black_box(tree.lookup(k, |v| v.len()));
									}
								}));
							}
							for keys in write_keys.iter().take(*num_writers) {
								let tree = Arc::clone(&tree);
								let keys = keys.clone();
								handles.push(thread::spawn(move || {
									for k in keys {
										black_box(tree.insert(k.clone(), k));
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
		}
	}
	group.finish();
}

// ============================================================================
// Refcounted-value benchmarks (Phase 3: epoch-deferred drop path)
// ============================================================================

#[derive(Clone)]
struct ArcBlob(#[allow(dead_code)] Arc<[u8; 32]>);

// SAFETY: `ArcBlob` wraps `Arc<[u8; 32]>`. Bitwise snapshot + recheck yields
// a valid `Arc`. `EPOCH_DEFERRED_DROP = true` is paired with
// `insert_defer` / `remove_defer` for all writes, keeping the buffer alive
// across the reader's snapshot/use window.
unsafe impl ferntree::OptimisticRead for ArcBlob {
	const EPOCH_DEFERRED_DROP: bool = true;
	type Slot = ferntree::atomic_slot::BoxedSlot<Self>;
}

fn bench_refcounted_lookup(c: &mut Criterion) {
	let mut group = c.benchmark_group("refcounted_lookup");

	for count in [10_000usize, 100_000] {
		let tree: Tree<i64, ArcBlob> = Tree::new();
		for k in 0..count as i64 {
			tree.insert_defer(k, ArcBlob(Arc::new([(k & 0xff) as u8; 32])));
		}
		let lookup_keys: Vec<i64> = (0..1000i64).collect();
		group.throughput(Throughput::Elements(lookup_keys.len() as u64));

		// Cloning shared-lock path
		group.bench_with_input(
			BenchmarkId::new("shared_get_clone", count),
			&lookup_keys,
			|b, keys| {
				b.iter(|| {
					for &k in keys {
						black_box(tree.get(&k));
					}
				})
			},
		);

		// Cloning optimistic path
		group.bench_with_input(
			BenchmarkId::new("optimistic_get_clone", count),
			&lookup_keys,
			|b, keys| {
				b.iter(|| {
					for &k in keys {
						black_box(tree.get_optimistic(&k));
					}
				})
			},
		);
	}
	group.finish();
}

fn bench_refcounted_writes(c: &mut Criterion) {
	let mut group = c.benchmark_group("refcounted_writes");

	{
		let count: usize = 10_000;
		group.throughput(Throughput::Elements(count as u64));

		// insert (synchronous drop of displaced value)
		group.bench_function(BenchmarkId::new("insert", count), |b| {
			b.iter_with_setup(
				|| {
					let tree: Tree<i64, ArcBlob> = Tree::new();
					for k in 0..count as i64 {
						tree.insert(k, ArcBlob(Arc::new([0u8; 32])));
					}
					tree
				},
				|tree| {
					for k in 0..count as i64 {
						tree.insert(k, ArcBlob(Arc::new([1u8; 32])));
					}
					black_box(tree);
				},
			)
		});

		// insert_defer (deferred drop of displaced value)
		group.bench_function(BenchmarkId::new("insert_defer", count), |b| {
			b.iter_with_setup(
				|| {
					let tree: Tree<i64, ArcBlob> = Tree::new();
					for k in 0..count as i64 {
						tree.insert_defer(k, ArcBlob(Arc::new([0u8; 32])));
					}
					tree
				},
				|tree| {
					for k in 0..count as i64 {
						tree.insert_defer(k, ArcBlob(Arc::new([1u8; 32])));
					}
					black_box(tree);
				},
			)
		});
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
	bench_insert_random_string,
	bench_lookup_hit_string,
	bench_lookup_hit_bytes,
	bench_vec_vs_bytes_lookup_hit,
	bench_vec_vs_bytes_insert_random,
);

criterion_group!(
	concurrent_benches,
	bench_concurrent_readers,
	bench_concurrent_writers,
	bench_concurrent_mixed,
	bench_concurrent_mixed_string,
);

criterion_group!(refcounted_benches, bench_refcounted_lookup, bench_refcounted_writes,);

criterion_main!(single_threaded_benches, concurrent_benches, refcounted_benches);
