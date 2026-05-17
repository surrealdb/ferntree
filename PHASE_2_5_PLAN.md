# Phase 2-5 implementation guide

Phase 1 (atomic_slot.rs) is complete. This document captures the design
decisions and step-by-step refactor plan for finishing #7.

## API decision: `OptimisticRead` gains `type Slot`

```rust
pub unsafe trait OptimisticRead: Sized + Send + 'static {
    const EPOCH_DEFERRED_DROP: bool = false;
    type Slot: OptimisticSlot<Value = Self>;
}
```

The existing blanket `unsafe impl<T: Copy> OptimisticRead for T {}` is
**dropped**. Replaced with enumerated stdlib impls and an
`impl_optimistic_read_inline!` / `impl_optimistic_read_boxed!` macro
for user types.

Built-in impls to add in `src/optimistic.rs`:

```rust
macro_rules! impl_inline {
    ($($t:ty),*) => {
        $(
            unsafe impl OptimisticRead for $t {
                type Slot = crate::atomic_slot::InlineSlot<Self>;
            }
        )*
    };
}
impl_inline!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize);

macro_rules! impl_boxed {
    ($($t:ty),*) => {
        $(
            unsafe impl OptimisticRead for $t {
                const EPOCH_DEFERRED_DROP: bool = true;
                type Slot = crate::atomic_slot::BoxedSlot<Self>;
            }
        )*
    };
}
impl_boxed!(String, Vec<u8>);

unsafe impl<T: Send + Sync + Clone + 'static> OptimisticRead for Vec<T> {
    const EPOCH_DEFERRED_DROP: bool = true;
    type Slot = BoxedSlot<Self>;
}
unsafe impl<T: Send + Sync + 'static + ?Sized> OptimisticRead for std::sync::Arc<T>
where
    Self: Clone,
{
    const EPOCH_DEFERRED_DROP: bool = true;
    type Slot = BoxedSlot<Self>;
}

// Unit type — special-case ZST slot.
pub struct UnitSlot;
impl Default for UnitSlot { fn default() -> Self { Self } }
unsafe impl OptimisticSlot for UnitSlot {
    type Value = ();
    type Displaced = ();
    unsafe fn load(&self) -> () {}
    unsafe fn store_into_empty(&self, _: ()) {}
    unsafe fn swap_init(&self, _: ()) -> () {}
    unsafe fn take_init(&self) -> () {}
    unsafe fn move_init_to_empty(&self, _: &Self) {}
    unsafe fn drop_in_place(&mut self, _: bool) {}
}
unsafe impl OptimisticRead for () { type Slot = UnitSlot; }
```

For `&'static str`: it is `Copy` but not `AtomicLoadable` (16 bytes).
Use `BoxedSlot`:
```rust
unsafe impl OptimisticRead for &'static str {
    const EPOCH_DEFERRED_DROP: bool = false;  // Drop is no-op
    type Slot = BoxedSlot<Self>;
}
```

## LeafNode struct change

```rust
pub(crate) struct LeafNode<K: OptimisticRead, V: OptimisticRead, const LC: usize> {
    pub(crate) len: crate::atomic_slot::AtomicLen,
    pub(crate) keys:   crate::atomic_slot::SlotArray<K::Slot, LC>,
    pub(crate) values: crate::atomic_slot::SlotArray<V::Slot, LC>,
    pub(crate) lower_fence: crate::atomic_slot::OptimisticOption<K::Slot>,
    pub(crate) upper_fence: crate::atomic_slot::OptimisticOption<K::Slot>,
    pub(crate) sample_key:  crate::atomic_slot::OptimisticOption<K::Slot>,
}
```

## InternalNode struct change

```rust
pub(crate) struct InternalNode<K: OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize> {
    pub(crate) len: AtomicLen,
    pub(crate) keys: SlotArray<K::Slot, IC>,
    pub(crate) edges: InlineVec<Atomic<HybridLatch<Node<K, V, IC, LC>>>, IC>,
    pub(crate) upper_edge: Atomic<HybridLatch<Node<K, V, IC, LC>>>,
    pub(crate) lower_fence: OptimisticOption<K::Slot>,
    pub(crate) upper_fence: OptimisticOption<K::Slot>,
    pub(crate) sample_key:  OptimisticOption<K::Slot>,
}
```

`edges`' inner `Atomic<...>` is already atomic; its `InlineVec` wrapper
needs `len: AtomicU16`. The existing `InlineVec` impl in
[src/inline_vec.rs](src/inline_vec.rs) can be migrated to atomic `len`
with `Release` stores under exclusive lock and `Acquire` loads on
readers — straightforward change.

## Method rewrites per node type

For LeafNode (in src/lib.rs, lines ~3595-4015):

| Method | New impl |
|---|---|
| `new()` | `keys: SlotArray::new(), values: SlotArray::new(), len: AtomicLen::new(0), lower_fence/upper_fence/sample_key: OptimisticOption::default()` |
| `lower_bound<Q>(&self, key)` | Use `keys.load(pos)` instead of `entries.get(pos)`; binary search on owned K snapshots |
| `lower_bound_raw<Q>` (existing) | Use `SlotArray::load_raw` instead of `ptr::read` |
| `value_at(pos)` | `unsafe { self.values.load(pos) }` returns owned V |
| `key_at(pos)` | `unsafe { self.keys.load(pos) }` returns owned K |
| `kv_at(pos)` | Load K and V separately, build the tuple |
| `insert_at(pos, k, v)` | `keys.shift_insert(len, pos, k); values.shift_insert(len, pos, v); len.fetch_add(1)` |
| `remove_at(pos)` | Symmetric with `shift_remove`; returns `Displaced` (Box for boxed storage) |
| `split` | Move entries via `take_init` from self, `store_into_empty` on `right`; or migrate via raw shuffle |
| `merge` | Symmetric |
| `borrow_from_*` | Symmetric |

Caller-side `lookup` closure now receives owned V (via load). To
preserve the existing `F: FnMut(&V) -> R` API, materialise the loaded
V on the stack:
```rust
let v: V = unsafe { self.values.load(pos) };
let result = f(&v);
// For Copy V, drop is no-op. For boxed V, dropping v frees the local
// clone; the underlying Box stored in the slot is untouched.
result
```

For BoxedSlot::load, the load clones through the raw pointer. The
clone is the V we hand to the user; dropping it drops only the clone.
The slot's actual Box stays alive until a swap displaces it (and that
displaced Box is epoch-deferred).

## `find_shared_leaf_and_optimistic_parent` migration (Phase 4)

Today (lines 1041-1147), the descent reborrows `&InternalNode`:
```rust
let (c_swip, pos) = match *target_guard {
    Node::Internal(ref internal) => {
        let (pos, _) = internal.lower_bound(key);
        let swip = internal.edge_at(pos)?;
        (swip, pos)
    }
    ...
};
```

After Part A's atomic-K storage, change to raw projection mirroring
`find_optimistic_leaf` (lines 1190-1217):
```rust
let target_ptr = target_guard.as_ptr();
let kind = unsafe { Node::variant_raw(target_ptr) };
let c_swip_ptr = match kind {
    NodeKindRaw::Internal(internal_ptr) => {
        let (pos, _) = unsafe { InternalNode::lower_bound_raw(internal_ptr, key) };
        unsafe { InternalNode::edge_at_raw(internal_ptr, pos)? }
    }
    NodeKindRaw::Leaf(_) => break /* leaf reached */
};
```

The two functions converge on a shared descent body that returns either
an optimistic or shared leaf guard at the bottom.

## Test fixture additions

Add `unsafe impl OptimisticRead` blocks in these files:

- `tests/integration.rs`: `ComplexValue { ... type Slot = BoxedSlot<Self>; const EPOCH_DEFERRED_DROP: bool = true; }`
- `tests/functional_coverage.rs`, `tests/unsafe_coverage.rs`: `DropCounter`
- `tests/memory_tests.rs`: `Versions`
- `tests/property.rs`: any custom proptest types

## Moving quarantined tests into `--lib`

`tests/concurrency.rs` contains two tests at lines 713-815. After Phase 2,
move them into a new `#[cfg(test)] mod concurrent_optimistic` module
in `src/lib.rs` (or a dedicated `src/tests/concurrent.rs` test module
included via `#[path]`). They must run under `cargo miri test --lib`.

Wrap iteration counts in `cfg!(miri)` checks to keep Miri runtime
bounded:
```rust
let writer_iters = if cfg!(miri) { 10 } else { 50 };
let reader_iters = if cfg!(miri) { 100 } else { 200_000 };
```

## New concurrent splits-during-reads test (Phase 4)

Add a `--lib` test that:
1. Prefills the tree to a height that ensures multiple internal-node
   splits during the test (e.g., 100 keys with `LEAF_CAPACITY = 8`).
2. Spawns a writer thread that inserts / removes keys in a range that
   triggers splits and merges.
3. Spawns 2-3 reader threads using `Tree::lookup` / `Tree::range`
   concurrently.
4. Runs for `cfg!(miri) { 50 } else { 50_000 }` iterations.
5. Joins and validates: no reader observed a corrupted value; final
   tree state matches a reference `BTreeMap`.

This is the test the issue's first acceptance criterion calls for.

## Phased commit plan

1. **Storage trait scaffolding** — DONE (commit c6d97b7, src/atomic_slot.rs,
   15 Miri-clean tests including 6 concurrent ones).
2. **OptimisticRead trait change + built-in impls** — DONE (commit c6d97b7).
   `type Slot` associated type added; Copy blanket dropped; stdlib impls
   provided for integers, String, Vec<u8>, Arc<T>, &'static str, ();
   user test fixtures (RefcountedBlob, RcKey, ComplexValue, DropCounter×2,
   Counted) updated.
3. **AtomicLen for node `len` fields** — DONE (commit 2f79c3a). Both
   LeafNode and InternalNode `len` are now `AtomicLen`; ~40 access sites
   migrated. `len_raw` uses `AtomicLen::load_raw` (Acquire-load via raw
   projection).
4. **Add atomic mirror fields to LeafNode + propagate K/V OptimisticRead
   bounds** — DONE (commit 1b0c873 + 8c68aaa). `LeafNode` has
   `atomic_keys: SlotArray<K::Slot, LC>` and
   `atomic_values: SlotArray<V::Slot, LC>` fields, allocated empty.
   `GenericTree<K, V, IC, LC>` now bounds K, V on `OptimisticRead`.
5. **Write-path mirror maintenance** — ATTEMPTED, NOT VIABLE (stashed).
   Tried updating `insert_at`, `remove_at`, `split`, `merge` to maintain
   both `entries` and the mirror. Three blockers surfaced:
   - `Arc::strong_count` test assertions break because the mirror's
     clone doubles the held refcount per entry.
   - The overwrite path in `iter::insert` does `std::mem::replace` on
     `entries` without touching the mirror — mirror entries become
     stale after overwrite.
   - The displaced mirror entries from `remove_at` require
     `eg: &epoch::Guard` plumbing through 20+ call sites; tractable
     but doubles write cost for no semantic gain.

   **Conclusion:** the dual-storage approach is unworkable. The right
   path is **full migration**: drop `entries` and use `atomic_keys` +
   `atomic_values` as the sole storage.

6. **Full LeafNode storage migration** — TODO. The substantive remaining
   work. Drop the `entries` field. Rewrite every method on LeafNode to
   use the atomic storage. For closure-based reads (`Tree::lookup`),
   load V atomically into a stack-local and pass `&V` to the closure
   (one extra clone per lookup for boxed V; zero overhead for Copy V).
   For iterators (`RawSharedIter::next`, `Range::next`, etc.), add an
   internal buffer field; return `&'_ K, &'_ V` borrows from the buffer
   with lifetime tied to `&mut self`. For mut iterators, write back via
   `swap_init` on the slot, route the displaced through `eg.defer`.

7. **InternalNode K migration** — TODO. Replace
   `keys: InlineVec<K, IC>` with `keys: SlotArray<K::Slot, IC>`.
   `lower_bound_raw` switches from `ptr::read(K)` to atomic load via
   `SlotArray::load_raw`.

8. **`find_shared_leaf_and_optimistic_parent` raw-projection migration**
   — TODO. Replace the `match *target_guard` (which reborrows
   `&Node` / `&InternalNode`) with `Node::variant_raw` +
   `InternalNode::lower_bound_raw` + `InternalNode::edge_at_raw`.

9. **Move quarantined tests into `--lib`. Add new concurrent splits test.**
   TODO. Validates the fix under Miri.

10. **Update src/optimistic.rs module docs.** TODO. Remove "out of scope"
    carve-outs; document the fix.

## Estimated effort

Honest assessment based on the per-step blast radius:

- Step 2: ~2-3 hours (trait change + impls + test-fixture impls).
- Step 3: ~6-10 hours (LeafNode is the bulk of the work).
- Step 4: ~4-6 hours.
- Step 5: ~2-4 hours.
- Step 6: ~3-5 hours (new test design + Miri tuning).
- Step 7: ~1 hour.

Total: 18-29 hours of focused engineering. This is a multi-session PR.

## Risks not yet resolved

- **`&'static str` V:** boxing it adds an allocation. Most existing test
  cases use `&str` only for literals; revisit if benchmarks complain.
- **AtomicLoadable for `bool`:** invalid bit patterns make a torn load
  potentially UB even atomically. Recommendation: omit from inline list;
  if needed by users, box it.
- **`InlineVec` len atomicisation:** simplest path is to add a
  `raw_len_atomic` returning `AtomicU16::load(Acquire)`, deprecate the
  non-atomic `raw_len`, and migrate callers.
- **Generic Tree height (`AtomicUsize`):** already atomic; no change.
