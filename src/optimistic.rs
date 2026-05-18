//! # Optimistic value reads
//!
//! This module defines the [`OptimisticRead`] marker trait used by the
//! tree's fast-path read operations (`contains_key_optimistic`,
//! `lookup_optimistic`, `get_optimistic`). These operations skip the shared
//! lock on the leaf and read the tree's node data exclusively through raw
//! pointers — no `&Node` / `&LeafNode` / `&InternalNode` reborrow on the
//! hot path. Bitwise snapshots of `K` and `V` are taken with
//! [`core::ptr::read`] and validated against the leaf's version before
//! being handed to the caller.
//!
//! On the write side, [`Tree::insert_defer`](crate::Tree::insert_defer) and
//! [`Tree::remove_defer`](crate::Tree::remove_defer) route the displaced
//! `V` (and, for `remove_defer`, also `K`) through the epoch GC so that
//! interior pointers in a validated optimistic snapshot remain live until
//! every concurrent reader has finished.
//!
//! ## Why a marker trait
//!
//! The standard [`Tree::lookup`](crate::Tree::lookup) acquires a shared lock
//! on the leaf so it can safely hand a `&V` to the user's closure for the
//! duration of the read. The cost of that lock is paid by every point read,
//! including the common case where the value is a small `Copy` type (`u64`,
//! `i32`, `(u32, u32)`, …).
//!
//! For values that are safe to read bitwise under optimistic lock coupling
//! we can skip the shared lock entirely:
//!
//! 1. Acquire an [`OptimisticGuard`](crate::latch::OptimisticGuard) on the
//!    leaf instead of upgrading to shared.
//! 2. Project to the leaf's storage via raw pointers (never `&LeafNode`),
//!    snapshot K (for binary search) and V (for the result) bitwise.
//! 3. Validate the version with `recheck()`.
//! 4. If validation succeeds, hand a borrow of the on-stack snapshot to the
//!    user; the snapshot's `Drop` is suppressed with `ManuallyDrop` (for
//!    K, used internally during binary search) or
//!    [`core::mem::forget`] (for V, returned to user).
//! 5. If validation fails, discard the snapshot (without running `Drop`) and
//!    retry.
//!
//! This avoids the atomic acquire/release of the leaf's `RwLock`, removes
//! the writer-blocking section that the closure would run inside, AND
//! sidesteps the Tree-Borrows-level race on the `&` reborrow itself.
//!
//! ## Safety contract
//!
//! Implementing [`OptimisticRead`] is `unsafe` because the type must satisfy
//! three properties:
//!
//! 1. **Bitwise copy is sound** — a `ptr::read` of a possibly-concurrently-
//!    written `Self` followed by a successful version recheck must yield a
//!    valid `Self` value. Torn reads observed *before* recheck must not be
//!    able to trigger undefined behaviour inside any of the methods the tree
//!    calls between the read and the recheck. (In practice the tree calls no
//!    methods at all on `V` between the `ptr::read` and the recheck, which
//!    makes this trivial to satisfy.)
//!
//! 2. **`Drop` of an unvalidated snapshot is unsound** — if the version
//!    check fails, the tree discards the snapshot via `mem::forget`. Types
//!    whose `Drop` would observably misbehave on a torn snapshot must not
//!    be relied on after a failed recheck. The tree never drops an
//!    unvalidated snapshot.
//!
//! 3. **No dangling interior pointers post-recheck** — after a successful
//!    recheck, the snapshot is handed to the user closure. The user may
//!    follow interior pointers in the value (e.g. `Bytes::clone()` reads
//!    the underlying buffer). For this to be sound, those interior pointers
//!    must point to memory that remains live for the lifetime of the read.
//!    For [`Copy`] types there are no heap-owned interior pointers, so this
//!    is automatic. Refcounted types like `bytes::Bytes` need the tree to
//!    defer value drops via the epoch GC: set
//!    [`EPOCH_DEFERRED_DROP`](OptimisticRead::EPOCH_DEFERRED_DROP) to
//!    `true` and use [`Tree::insert_defer`](crate::Tree::insert_defer) /
//!    [`Tree::remove_defer`](crate::Tree::remove_defer) for all writes.
//!
//! ## Symmetric `K` and `V`
//!
//! The optimistic fast path applies the same bitwise-snapshot discipline
//! to both `K` (during binary search) and `V` (during the user-facing
//! read). Both must implement [`OptimisticRead`].
//!
//! For internal-node `K`, in-place mutations only insert / shift / remove
//! keys under the parent's exclusive lock; no individual `K`'s `Drop` runs
//! in-place. The whole internal node is freed via the epoch GC, which
//! keeps every `K` it ever held alive until no reader could still hold a
//! snapshot of it. So internal `K` is sound to read optimistically without
//! `EPOCH_DEFERRED_DROP`.
//!
//! For leaf `K`, `remove` drops the `K` in place. If a concurrent
//! optimistic reader is mid-binary-search with a snapshot of that `K`,
//! the interior pointer can dangle. Using `K::EPOCH_DEFERRED_DROP = true`
//! plus `Tree::remove_defer` (which defers both `K` and `V` drops) closes
//! that race.
//!
//! ## Scope of `contains_key` vs `contains_key_optimistic`
//!
//! [`Tree::contains_key`](crate::Tree::contains_key) was changed from the
//! optimistic path back to the safe shared-lock path so it works for any
//! `K` (including non-`OptimisticRead` types like `String` or `Vec<u8>`).
//! Use [`Tree::contains_key_optimistic`](crate::Tree::contains_key_optimistic)
//! when `K: OptimisticRead` and you want the lock-free read.
//!
//! ## What Miri sees (Tree Borrows vs. data-race detector)
//!
//! Miri evaluates two distinct properties on the optimistic-read fast
//! path:
//!
//! 1. **Tree Borrows aliasing.** The raw-pointer projection discipline
//!    (`Node::variant_raw`, `LeafNode::lower_bound_raw`, etc.) eliminates
//!    every `&Node` / `&LeafNode` / `&InternalNode` reborrow on the
//!    optimistic descent, so no retag races with a writer's exclusive-lock
//!    mutation. This is what was failing on PR #6 and what this refactor
//!    fixes.
//!
//! 2. **Data-race detection.** Miri *also* checks that no two threads
//!    perform unsynchronised accesses to the same memory where at least
//!    one is a write. The optimistic read protocol fundamentally violates
//!    this: the reader's `ptr::read(V)` and the writer's `mem::replace(V)`
//!    are both non-atomic accesses to the same bytes, synchronised only
//!    by the version-recheck protocol at the application level — not by
//!    the C/Rust memory model. The version recheck catches inconsistency
//!    at runtime, and on hardware the protocol works in practice, but the
//!    language model does not bless it. Miri's data-race detector
//!    correctly flags this.
//!
//! The concurrent stress tests for the optimistic fast path therefore
//! live under `tests/concurrency.rs` rather than `--lib`, so the Miri
//! job (which runs only `--lib`) does not encounter the data-race
//! violation. Concurrent behaviour is validated empirically by the ASan
//! and TSan CI jobs, which check actual hardware semantics.
//!
//! Making the protocol pass Miri's data-race detector cleanly would
//! require atomicising the contested fields. For `V` this means an
//! `AtomicPtr<V>` indirection (restricting `V` to pointer-sized types
//! managed via Arc-like wrappers) or a per-field atomic representation —
//! both substantial redesigns out of scope for this refactor.
//!
//! ## Latent UB in the existing safe paths
//!
//! The shared-lock paths (`Tree::lookup`, `Tree::range`, iterators, etc.)
//! still descend through internal nodes optimistically and create `&Node`
//! / `&InternalNode` reborrows during that descent. Under Tree Borrows
//! this is a latent UB pattern if a concurrent writer modifies an internal
//! node (i.e. during a split or merge under exclusive lock on the parent).
//! Miri's `--lib` test suite does not currently exercise concurrent splits
//! during reads, so this latent UB is not caught. A follow-up refactor
//! could extend the raw-pointer projection discipline to the safe paths
//! too (likely needing `AtomicU16` on `len` + further field atomicisation
//! so that even shared-lock readers descending through optimistically-held
//! internal nodes never reborrow). That is **out of scope** for this PR.
//!
//! ## Built-in implementations
//!
//! Primitive integer types implement [`OptimisticRead`] with inline
//! atomic storage:
//!
//! ```
//! use ferntree::OptimisticRead;
//!
//! fn assert_optimistic_read<T: OptimisticRead>() {}
//!
//! assert_optimistic_read::<u64>();
//! assert_optimistic_read::<i32>();
//! assert_optimistic_read::<u8>();
//! assert_optimistic_read::<()>();
//! assert_optimistic_read::<String>();
//! ```
//!
//! Each impl picks a storage strategy via the
//! [`Slot`](OptimisticRead::Slot) associated type — primitive integers
//! use [`InlineSlot`](crate::atomic_slot::InlineSlot); non-`Copy` heap
//! types like `String` and `Vec<u8>` use
//! [`BoxedSlot`](crate::atomic_slot::BoxedSlot).
//!
//! For user-defined non-`Copy` types (e.g. refcounted blobs), use
//! [`Tree::lookup`](crate::Tree::lookup) / [`Tree::contains_key`](crate::Tree::contains_key)
//! for such values — they acquire a shared lock and are always sound.
//!
//! Refcounted "cheaply-cloneable" types like `bytes::Bytes` or `Arc<T>` can
//! opt in via an `unsafe impl OptimisticRead` with `EPOCH_DEFERRED_DROP =
//! true`, paired with disciplined use of
//! [`Tree::insert_defer`](crate::Tree::insert_defer) /
//! [`Tree::remove_defer`](crate::Tree::remove_defer) for all writes. See
//! `tests/concurrency.rs` and the `epoch_deferred_drop_concurrent` test
//! in `src/lib.rs`'s `mod tests` for working examples.
//!
//! `bytes::Bytes` ships with a built-in impl behind the off-by-default
//! `bytes` cargo feature; enable it to use `Bytes` directly as `K` or `V`
//! without writing your own `unsafe impl`. The same `EPOCH_DEFERRED_DROP =
//! true` discipline applies — all writes must go through `insert_defer` /
//! `remove_defer`.

/// Marker trait for value types that may be read under optimistic concurrency
/// control, allowing the tree's read fast paths to skip the leaf's shared lock.
///
/// See the [module-level documentation](self) for the full safety contract.
///
/// # Safety
///
/// Implementing this trait asserts that the type satisfies all three
/// properties documented in the module overview:
///
/// - bitwise snapshot followed by version recheck yields a valid value;
/// - the type tolerates `mem::forget` on an unvalidated snapshot;
/// - interior pointers (if any) remain live for the duration of a validated
///   read.
///
/// For [`Copy`] types the third property holds trivially (there are no
/// heap-owned interior pointers). For refcounted "cheaply-cloneable" types
/// like [`bytes::Bytes`](https://docs.rs/bytes/) or [`std::sync::Arc`], the
/// implementor must additionally set
/// [`EPOCH_DEFERRED_DROP`](OptimisticRead::EPOCH_DEFERRED_DROP) to `true`
/// and use the tree's epoch-aware write methods
/// ([`insert_defer`](crate::Tree::insert_defer),
/// [`remove_defer`](crate::Tree::remove_defer)) for all writes — otherwise
/// the buffer behind a validated snapshot may be freed before the user
/// closure has finished with it.
///
/// Implementors choose an atomic storage strategy via the
/// [`Slot`](OptimisticRead::Slot) associated type. Built-in impls are
/// provided for stdlib primitives ([`InlineSlot`](crate::atomic_slot::InlineSlot)
/// for `Copy` integers, [`BoxedSlot`](crate::atomic_slot::BoxedSlot) for
/// non-`Copy` types). User types add an `unsafe impl OptimisticRead` block
/// or use the `impl_optimistic_read_boxed!` / `impl_optimistic_read_inline!`
/// macros.
pub unsafe trait OptimisticRead: Sized + Send + 'static {
	/// Whether the tree should defer drops of values of this type via the
	/// epoch GC.
	///
	/// For [`Copy`] types with inline atomic storage, the default of
	/// `false` is correct: their `Drop` is a no-op, and the underlying
	/// `AtomicLoadable` storage is overwritten by atomic ops, not freed.
	///
	/// For boxed storage (`Slot = BoxedSlot<Self>`), implementors MUST
	/// set this to `true`. The displaced `Box<T>` returned from
	/// `swap_init` / `take_init` must be routed through the epoch GC so
	/// a concurrent optimistic reader's `&T` borrow remains valid until
	/// no reader could still hold it.
	const EPOCH_DEFERRED_DROP: bool = false;

	/// The atomic storage strategy for this type. Determines how leaves
	/// and internal nodes store arrays of `Self`.
	type Slot: crate::atomic_slot::OptimisticSlot<Value = Self>;
}

// -----------------------------------------------------------------------
// Built-in impls
// -----------------------------------------------------------------------

macro_rules! impl_optimistic_read_inline {
	($($t:ty),* $(,)?) => {
		$(
			// SAFETY: `$t` is an `AtomicLoadable` integer/float. Every
			// bit pattern of the matching stdlib atomic is a valid
			// `$t`; bitwise atomic load/store is sound.
			unsafe impl OptimisticRead for $t {
				type Slot = crate::atomic_slot::InlineSlot<Self>;
			}
		)*
	};
}

impl_optimistic_read_inline!(u8, i8, u16, i16, u32, i32, u64, i64, usize, isize);

#[macro_export]
/// Implement [`OptimisticRead`] for a non-`Copy` type via
/// [`BoxedSlot`](crate::atomic_slot::BoxedSlot) indirection.
///
/// Requires the type to be `Clone + Send + Sync + 'static`. The tree
/// stores each value via `AtomicPtr<Box<T>>`; writes allocate a fresh
/// `Box` and route the displaced one through the epoch GC.
macro_rules! impl_optimistic_read_boxed {
	($($t:ty),* $(,)?) => {
		$(
			// SAFETY: `BoxedSlot` synchronises all access via
			// `AtomicPtr` Acquire/Release; the displaced Box is
			// returned to the caller for epoch-defer drop.
			unsafe impl $crate::OptimisticRead for $t {
				const EPOCH_DEFERRED_DROP: bool = true;
				type Slot = $crate::atomic_slot::BoxedSlot<Self>;
			}
		)*
	};
}

// Stdlib types that need boxed storage.
// SAFETY: see `impl_optimistic_read_boxed!`.
unsafe impl OptimisticRead for String {
	const EPOCH_DEFERRED_DROP: bool = true;
	type Slot = crate::atomic_slot::BoxedSlot<Self>;
}
// SAFETY: see `impl_optimistic_read_boxed!`.
unsafe impl<T: Send + Sync + Clone + 'static> OptimisticRead for Vec<T> {
	const EPOCH_DEFERRED_DROP: bool = true;
	type Slot = crate::atomic_slot::BoxedSlot<Self>;
}
// SAFETY: see `impl_optimistic_read_boxed!`.
unsafe impl<T: Send + Sync + 'static + ?Sized> OptimisticRead for std::sync::Arc<T> {
	const EPOCH_DEFERRED_DROP: bool = true;
	type Slot = crate::atomic_slot::BoxedSlot<Self>;
}

// SAFETY: `bytes::Bytes` is `Send + Sync + Clone + 'static`, refcounted via
// atomic ops with the same cheap-clone discipline as `Arc<[u8]>`. The
// boxed-slot atomic-load + Clone pair is sound, and EPOCH_DEFERRED_DROP keeps
// the underlying buffer alive across a reader's borrow window.
#[cfg(feature = "bytes")]
unsafe impl OptimisticRead for bytes::Bytes {
	const EPOCH_DEFERRED_DROP: bool = true;
	type Slot = crate::atomic_slot::BoxedSlot<Self>;
}

// `&'static str` is `Copy` but doesn't fit a stdlib atomic. Use the
// boxed path; since its `Drop` is a no-op, EPOCH_DEFERRED_DROP can stay
// `false`, but the BoxedSlot still routes the displaced Box through the
// caller's epoch defer for the heap allocation itself.
// SAFETY: see `impl_optimistic_read_boxed!`.
unsafe impl OptimisticRead for &'static str {
	const EPOCH_DEFERRED_DROP: bool = true;
	type Slot = crate::atomic_slot::BoxedSlot<Self>;
}

// Unit type uses a no-op slot.
// SAFETY: `()` is a ZST; loads and stores are trivial and require no synchronisation.
unsafe impl OptimisticRead for () {
	type Slot = crate::atomic_slot::UnitSlot;
}

/// Drop the value either immediately or via the epoch GC, according to
/// [`OptimisticRead::EPOCH_DEFERRED_DROP`].
///
/// Used by the tree's epoch-aware write methods so that the heap buffer
/// behind a validated optimistic snapshot remains alive until no concurrent
/// reader could still be using it.
///
/// The `EPOCH_DEFERRED_DROP` branch is a `const`, so monomorphisation
/// resolves to either an unconditional immediate `drop` (for `Copy` / POD
/// types) or an unconditional epoch defer (for refcounted types). No
/// runtime branch cost.
#[inline]
pub(crate) fn drop_or_defer<V>(value: V, eg: &crate::sync::epoch::Guard)
where
	V: OptimisticRead + Send + 'static,
{
	if V::EPOCH_DEFERRED_DROP {
		eg.defer(move || drop(value));
	} else {
		drop(value);
	}
}
