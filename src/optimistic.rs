//! # Optimistic value reads
//!
//! This module defines the [`OptimisticRead`] marker trait used by the
//! tree's fast-path read operations (`contains_key`, `lookup_optimistic`,
//! `get_optimistic`). These operations skip the shared lock on the leaf and
//! instead snapshot the value bitwise, validating against the leaf's version
//! before handing the snapshot to the caller.
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
//! 2. Locate the entry and snapshot the value with [`core::ptr::read`].
//! 3. Validate the version with `recheck()`.
//! 4. If validation succeeds, hand a borrow of the on-stack snapshot to the
//!    user; the snapshot's `Drop` is suppressed with [`core::mem::forget`].
//! 5. If validation fails, discard the snapshot (without running `Drop`) and
//!    retry.
//!
//! This avoids the atomic acquire/release of the leaf's `RwLock` and removes
//! the writer-blocking section that the closure runs inside.
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
//!    defer value drops via the epoch GC; this is a Phase 3 extension.
//!
//! ## Blanket implementations
//!
//! Every [`Copy`] type is automatically [`OptimisticRead`]:
//!
//! ```
//! use ferntree::OptimisticRead;
//!
//! fn assert_optimistic_read<T: OptimisticRead>() {}
//!
//! assert_optimistic_read::<u64>();
//! assert_optimistic_read::<i32>();
//! assert_optimistic_read::<(u32, u32)>();
//! assert_optimistic_read::<[u8; 16]>();
//! ```
//!
//! Types that are not `Copy` (e.g. `String`, `Vec<u8>`, `SmallVec<…>`) are
//! deliberately *not* `OptimisticRead`: their `Drop` frees heap memory, and
//! a torn snapshot of their layout has a pointer/length pair that does not
//! describe any real allocation. Use [`Tree::lookup`](crate::Tree::lookup)
//! for such values — it acquires a shared lock and is always sound.

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
/// All [`Copy`] types satisfy the contract trivially and have a blanket impl.
pub unsafe trait OptimisticRead: Sized {
	/// Whether the tree should defer drops of values of this type via the
	/// epoch GC.
	///
	/// For [`Copy`] types the default of `false` is correct: their `Drop`
	/// is a no-op, so deferring would only add allocation overhead.
	///
	/// For refcounted "cheaply-cloneable" types whose `Drop` may free a
	/// shared heap buffer (e.g. `bytes::Bytes`, `Arc<T>`), implementors
	/// MUST set this to `true`. Combined with using the tree's epoch-aware
	/// write methods, this guarantees that the shared buffer behind a
	/// validated optimistic snapshot remains live until the epoch GC
	/// reclaims it — by which point no concurrent reader could still be
	/// using a snapshot from before the write.
	///
	/// This is a `const` so the branch is resolved at monomorphisation
	/// time: for `EPOCH_DEFERRED_DROP = false` types the defer path is
	/// eliminated entirely by the optimiser.
	const EPOCH_DEFERRED_DROP: bool = false;
}

// SAFETY: `Copy` types have no `Drop`, so a torn snapshot is discarded
// safely with no side effect. They contain no heap-owned interior pointers
// whose target memory can be freed by a concurrent writer.
unsafe impl<T: Copy> OptimisticRead for T {}

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
