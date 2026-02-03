//! # Iterator Module for the Concurrent B+ Tree
//!
//! This module provides two iterator types for traversing the B+ tree:
//!
//! - [`RawSharedIter`]: Read-only iteration with shared (blocking) locks on leaves
//! - [`RawExclusiveIter`]: Read-write iteration with exclusive locks on leaves
//!
//! ## Cursor Model
//!
//! The iterators use a cursor-based model where the cursor position is tracked
//! relative to entries in the current leaf:
//!
//! ```text
//! Leaf: [K0, K1, K2, K3, K4]
//!        ^
//!        └── Cursor::Before(0) - next() returns K0
//!
//! Leaf: [K0, K1, K2, K3, K4]
//!                    ^
//!                    └── Cursor::After(2) - next() returns K3, prev() returns K2
//! ```
//!
//! The `Cursor` enum distinguishes:
//! - `Before(n)`: Next entry is at position `n`
//! - `After(n)`: Just visited position `n`, next is `n+1`, prev is `n`
//!
//! ## Anchor Recovery
//!
//! Because the tree uses optimistic concurrency, concurrent modifications can
//! invalidate our position. The anchor mechanism captures a logical position
//! that can survive structural changes:
//!
//! ```text
//! 1. Iterator at position [Before(K5)]
//! 2. Concurrent split happens, K5 moves to new leaf
//! 3. Optimistic validation fails
//! 4. Create anchor: Anchor::Before("K5")
//! 5. Seek to anchor position in new tree structure
//! 6. Continue iteration from recovered position
//! ```
//!
//! ## Lock Holding
//!
//! The iterators hold locks on the current leaf node:
//! - `RawSharedIter`: Holds a `SharedGuard` - blocks writers on current leaf
//! - `RawExclusiveIter`: Holds an `ExclusiveGuard` - blocks all access to current leaf
//!
//! Both iterators also keep an optimistic guard on the parent node for efficient
//! sibling leaf traversal.
//!
//! ## Epoch Guard
//!
//! Each iterator pins an epoch (`epoch::Guard`) for the lifetime of iteration.
//! This ensures nodes aren't reclaimed while we hold references to them.

use crate::error;
use crate::latch::{ExclusiveGuard, OptimisticGuard, SharedGuard};
use crate::sync::epoch::{self as epoch};
use crate::{Direction, GenericTree, Node};
use std::borrow::Borrow;
use std::ops::Bound;

// ===========================================================================
// Helper Enums
// ===========================================================================

/// A recoverable position marker for handling concurrent modifications.
///
/// When optimistic validation fails (the tree was modified during our operation),
/// we need to "recover" our position. The `Anchor` captures a logical position
/// that can be converted back to a physical position by re-traversing the tree.
///
/// # Variants
///
/// - `Start`: Position before the first entry in the tree
/// - `End`: Position after the last entry in the tree
/// - `Before(key)`: Position immediately before the given key
/// - `After(key)`: Position immediately after the given key
///
/// # Usage
///
/// 1. Before an operation that might fail validation, create an anchor
/// 2. Attempt the operation
/// 3. If validation fails, restore from anchor by seeking to its position
#[derive(Debug, PartialEq, Copy, Clone)]
enum Anchor<T> {
	/// Anchored at the very beginning of the tree.
	Start,
	/// Anchored immediately after this key.
	After(T),
	/// Anchored immediately before this key.
	Before(T),
	/// Anchored at the very end of the tree.
	End,
}

/// Tracks the cursor position within a leaf node.
///
/// The cursor indicates where we are in the iteration relative to the current
/// entry. This distinction is important for bidirectional iteration.
///
/// # Position Semantics
///
/// - `Before(n)`: We haven't returned entry `n` yet. `next()` returns `n`.
/// - `After(n)`: We've returned entry `n`. `next()` returns `n+1`, `prev()` returns `n`.
///
/// # Examples
///
/// ```text
/// Leaf with entries: [a, b, c, d]
///                     0  1  2  3
///
/// Cursor::Before(0) -> next() = a, cursor becomes Before(1)
/// Cursor::Before(2) -> next() = c, cursor becomes Before(3)
/// Cursor::After(2)  -> next() = d, cursor becomes Before(4)
/// Cursor::After(2)  -> prev() = c, cursor becomes Before(2) or After(1)
/// ```
#[derive(Debug, PartialEq, Copy, Clone)]
enum Cursor {
	/// Cursor is positioned immediately after index `n`.
	/// The entry at `n` has been returned. `prev()` would return `n`.
	After(u16),
	/// Cursor is positioned immediately before index `n`.
	/// `next()` would return the entry at `n`.
	Before(u16),
}

impl Cursor {
	/// Compute the entry to return for `prev()` and the new cursor state.
	///
	/// Returns `Some((position, new_cursor))` if there's an entry to return,
	/// or `None` if at the beginning of the leaf (should try prev_leaf).
	///
	/// # State Transitions
	///
	/// | Starting State | Returns | New State |
	/// |---------------|---------|-----------|
	/// | `After(n)` where n > 0 | `n` | `After(n-1)` |
	/// | `After(0)` | `0` | `Before(0)` |
	/// | `Before(n)` where n > 0 | `n-1` | `After(n-2)` or `Before(0)` |
	/// | `Before(0)` | None | - |
	#[inline]
	fn prev_entry(self) -> Option<(u16, Cursor)> {
		match self {
			Cursor::After(pos) => {
				// After(pos): return entry at pos, move cursor backward
				let new = if pos == 0 {
					Cursor::Before(0) // Can't go further back in this leaf
				} else {
					Cursor::After(pos - 1)
				};
				Some((pos, new))
			}
			Cursor::Before(0) => {
				// At the very beginning - nothing to return from this leaf
				None
			}
			Cursor::Before(pos) => {
				// Before(pos) with pos > 0: return entry at pos-1
				let curr = pos - 1;
				let new = if curr == 0 {
					Cursor::Before(0) // No more entries before in this leaf
				} else {
					Cursor::After(curr - 1)
				};
				Some((curr, new))
			}
		}
	}

	/// Compute the entry to return for `next()` and the new cursor state.
	///
	/// Returns `Some((position, new_cursor))` if there's an entry to return,
	/// or `None` if past the end of the leaf (should try next_leaf).
	///
	/// # Parameters
	///
	/// - `leaf_len`: The number of entries in the current leaf
	///
	/// # State Transitions
	///
	/// | Starting State | Returns | New State |
	/// |---------------|---------|-----------|
	/// | `Before(n)` where n < len | `n` | `Before(n+1)` |
	/// | `Before(n)` where n >= len | None | - |
	/// | `After(n)` where n+1 < len | `n+1` | `Before(n+2)` |
	/// | `After(n)` where n+1 >= len | None | - |
	#[inline]
	fn next_entry(self, leaf_len: u16) -> Option<(u16, Cursor)> {
		match self {
			Cursor::Before(pos) if pos < leaf_len => {
				// Before(pos): return entry at pos
				Some((pos, Cursor::Before(pos + 1)))
			}
			Cursor::After(pos) => {
				// After(pos): return entry at pos+1 if it exists
				let next = pos + 1;
				if next < leaf_len {
					Some((next, Cursor::Before(next + 1)))
				} else {
					None // Past end of leaf
				}
			}
			_ => None, // Before(pos) where pos >= leaf_len
		}
	}

	/// Returns the position for peek() without computing new cursor state.
	///
	/// This is used by `peek()` to inspect the next entry without advancing.
	/// Returns `Some(position)` if there's an entry available, `None` if at end of leaf.
	#[inline]
	fn peek_next_pos(self, leaf_len: u16) -> Option<u16> {
		match self {
			Cursor::Before(pos) if pos < leaf_len => Some(pos),
			Cursor::After(pos) if (pos + 1) < leaf_len => Some(pos + 1),
			_ => None,
		}
	}

	/// Returns the position for peek_prev() without computing new cursor state.
	///
	/// This is used by `peek_prev()` to inspect the previous entry without moving backward.
	/// Returns `Some(position)` if there's an entry available, `None` if at start of leaf.
	#[inline]
	fn peek_prev_pos(self) -> Option<u16> {
		match self {
			Cursor::After(pos) => Some(pos),
			Cursor::Before(pos) if pos > 0 => Some(pos - 1),
			_ => None,
		}
	}
}

/// Result of attempting to move to the next/previous leaf.
///
/// Used by `next_leaf()` and `prev_leaf()` to signal the outcome.
#[derive(Debug, PartialEq, Copy, Clone)]
enum LeafResult {
	/// Successfully moved to a new leaf.
	Ok,
	/// Reached the end/beginning of the tree.
	End,
	/// Current leaf still has entries in the iteration direction.
	/// Caller should continue iterating within current leaf.
	Retry,
}

/// Result of an optimistic sibling leaf jump.
///
/// Used by `optimistic_jump()` to signal the outcome.
#[derive(Debug)]
enum JumpResult {
	/// Successfully acquired the sibling leaf.
	Ok,
	/// No sibling exists (at tree boundary).
	End,
	/// Optimistic validation failed - must restore from anchor and retry.
	Err,
}

// ===========================================================================
// RawSharedIter - Read-Only Iterator
// ===========================================================================

/// A read-only iterator over the entries of a B+ tree.
///
/// This iterator acquires shared (read) locks on leaf nodes, which:
/// - Allows concurrent readers on the same leaf
/// - Blocks writers on the current leaf
/// - Uses optimistic locks on parent nodes for efficient traversal
///
/// # Usage
///
/// ```
/// use ferntree::Tree;
///
/// let tree: Tree<i32, &str> = Tree::new();
/// tree.insert(1, "one");
/// tree.insert(2, "two");
/// tree.insert(3, "three");
///
/// let mut iter = tree.raw_iter();
/// iter.seek_to_first();
///
/// while let Some((k, v)) = iter.next() {
///     println!("{}: {}", k, v);
/// }
/// ```
///
/// # Memory Safety
///
/// The iterator pins an epoch guard for its entire lifetime, ensuring that
/// node memory isn't reclaimed while references to entries are outstanding.
pub struct RawSharedIter<'t, K, V, const IC: usize, const LC: usize> {
	/// Reference to the tree being iterated.
	tree: &'t GenericTree<K, V, IC, LC>,
	/// Epoch guard - pinned for the iterator's lifetime.
	/// Prevents garbage collection of nodes we might be accessing.
	eg: epoch::Guard,
	/// Optimistic guard on the parent of the current leaf.
	/// Used for efficient sibling traversal. Contains (guard, position_in_parent).
	parent: Option<(OptimisticGuard<'t, Node<K, V, IC, LC>>, u16)>,
	/// Shared guard on the current leaf and our cursor position.
	/// The shared lock blocks writers but allows concurrent readers.
	leaf: Option<(SharedGuard<'t, Node<K, V, IC, LC>>, Cursor)>,
}

impl<'t, K: Clone + Ord, V, const IC: usize, const LC: usize> RawSharedIter<'t, K, V, IC, LC> {
	/// Creates a new iterator, pinning a fresh epoch guard.
	///
	/// The iterator starts in an unpositioned state;
	/// call `seek*` methods to position it before iterating.
	pub(crate) fn new(tree: &'t GenericTree<K, V, IC, LC>) -> RawSharedIter<'t, K, V, IC, LC> {
		RawSharedIter {
			tree,
			// Pin the epoch immediately - this guard lives for the iterator's lifetime
			eg: epoch::pin(),
			parent: None,
			leaf: None,
		}
	}

	// -----------------------------------------------------------------------
	// Lifetime Extension Helpers
	// -----------------------------------------------------------------------
	//
	// These functions use `std::mem::transmute` to extend the lifetime of
	// guards from 'g to 't. This is safe because:
	//
	// 1. The epoch guard (`eg`) is held for the entire iterator lifetime ('t)
	// 2. The epoch guard ensures the underlying memory isn't reclaimed
	// 3. The guard's actual lifetime 'g is bounded by the epoch anyway
	//
	// Without this, we'd have borrowing issues because the tree methods
	// return guards with shorter lifetimes tied to the epoch::Guard borrow.

	/// Extends the lifetime of a leaf guard from 'g to 't.
	///
	/// # Safety
	///
	/// This is safe because we hold `self.eg` (epoch guard) for the entire
	/// iterator lifetime 't. The epoch guard guarantees the underlying node
	/// memory won't be reclaimed, so extending the lifetime is sound.
	#[inline]
	fn leaf_lt<'g>(
		guard: SharedGuard<'g, Node<K, V, IC, LC>>,
	) -> SharedGuard<'t, Node<K, V, IC, LC>> {
		// SAFETY: We hold the epoch guard at all times so 'g should equal 't.
		// The guard's actual lifetime is bounded by the epoch, and we maintain
		// the epoch pin for our entire lifetime.
		unsafe { std::mem::transmute(guard) }
	}

	/// Extends the lifetime of a parent guard from 'g to 't.
	///
	/// # Safety
	///
	/// Same justification as `leaf_lt` - epoch guard ensures memory safety.
	#[inline]
	fn parent_lt<'g>(
		guard: OptimisticGuard<'g, Node<K, V, IC, LC>>,
	) -> OptimisticGuard<'t, Node<K, V, IC, LC>> {
		// SAFETY: We hold the epoch guard at all times so 'g should equal 't.
		// The guard's actual lifetime is bounded by the epoch, and we maintain
		// the epoch pin for our entire lifetime.
		unsafe { std::mem::transmute(guard) }
	}

	// -----------------------------------------------------------------------
	// Anchor Management
	// -----------------------------------------------------------------------

	/// Creates an anchor representing our current logical position.
	///
	/// The anchor captures enough information to restore our position after
	/// the tree structure changes. It uses key values rather than positions
	/// so it survives splits and merges.
	///
	/// # Returns
	///
	/// - `Some(anchor)` if we're positioned on a leaf
	/// - `None` if we're not positioned
	fn current_anchor(&self) -> Option<Anchor<K>> {
		if let Some((guard, cursor)) = self.leaf.as_ref() {
			let leaf = guard.as_leaf();

			let anchor = match *cursor {
				Cursor::Before(pos) => {
					// We're before position `pos`
					if pos >= leaf.len {
						// Past the end of this leaf
						if leaf.len > 0 {
							// Anchor after the last key in this leaf
							Anchor::After(
								leaf.key_at(leaf.len - 1)
									.expect("leaf.len > 0 implies key exists at len-1")
									.clone(),
							)
						} else if let Some(k) = &leaf.lower_fence {
							// Empty leaf - anchor at its lower bound
							Anchor::After(k.clone())
						} else {
							// Empty leaf at start of tree
							Anchor::Start
						}
					} else {
						// Normal case - anchor before the key at this position
						Anchor::Before(
							leaf.key_at(pos)
								.expect("pos < leaf.len implies key exists at pos")
								.clone(),
						)
					}
				}
				Cursor::After(pos) => {
					// We're after position `pos`
					if pos >= leaf.len {
						// Position is past leaf bounds
						if leaf.len > 0 {
							Anchor::After(
								leaf.key_at(leaf.len - 1)
									.expect("leaf.len > 0 implies key exists at len-1")
									.clone(),
							)
						} else if let Some(k) = &leaf.upper_fence {
							Anchor::After(k.clone())
						} else {
							Anchor::End
						}
					} else {
						// Normal case - anchor after the key at this position
						Anchor::After(
							leaf.key_at(pos)
								.expect("pos < leaf.len implies key exists at pos")
								.clone(),
						)
					}
				}
			};

			Some(anchor)
		} else {
			None
		}
	}

	/// Restores the iterator position from an anchor.
	///
	/// This is called when optimistic validation fails. It releases all locks
	/// and re-seeks to the anchored position in the (possibly restructured) tree.
	fn restore_from_anchor(&mut self, anchor: &Anchor<K>) {
		// Release all locks before re-seeking
		self.parent.take();
		self.leaf.take(); // Make sure there are no locks held

		// Re-seek based on anchor type
		match anchor {
			Anchor::Start => self.seek_to_first(),
			Anchor::End => self.seek_to_last(),
			Anchor::Before(key) => self.seek(key),
			Anchor::After(key) => self.seek_for_prev(key),
		}
	}

	// -----------------------------------------------------------------------
	// Leaf Traversal
	// -----------------------------------------------------------------------

	/// Attempts to move to the next (rightward) leaf.
	///
	/// This is called when we've exhausted all entries in the current leaf
	/// during forward iteration. It handles the complexity of sibling
	/// acquisition with optimistic concurrency.
	///
	/// # Algorithm
	///
	/// 1. Check if we actually need to move (maybe current leaf still has entries)
	/// 2. Check if we're at the rightmost leaf (no upper_fence means no right sibling)
	/// 3. Save an anchor in case we need to recover
	/// 4. Attempt optimistic jump to sibling
	/// 5. If validation fails, restore from anchor and retry
	fn next_leaf(&mut self) -> LeafResult {
		if self.leaf.is_none() {
			return LeafResult::End;
		}

		// Save anchor before attempting the jump
		let anchor = self.current_anchor().expect("leaf exists");

		loop {
			// Check if we actually need to leave this leaf
			{
				let (guard, cursor) = self.leaf.as_ref().unwrap();
				let leaf = guard.as_leaf();

				match *cursor {
					// If before a valid position, we have more entries in this leaf
					Cursor::Before(pos) if pos < leaf.len => {
						return LeafResult::Retry;
					}
					// If after a position and there are more entries after
					Cursor::After(pos) if (pos + 1) < leaf.len => {
						return LeafResult::Retry;
					}
					_ => {}
				}

				// Check if we're at the rightmost leaf
				// (no upper_fence means this is the last leaf)
				if leaf.upper_fence.is_none() || self.parent.is_none() {
					return LeafResult::End;
				}
			}

			// Release current leaf lock before jumping
			let _ = self.leaf.take();

			// Attempt to acquire the next leaf
			match self.optimistic_jump(Direction::Forward) {
				JumpResult::Ok => {
					return LeafResult::Ok;
				}
				JumpResult::End => {
					return LeafResult::End;
				}
				JumpResult::Err => {
					// Validation failed - restore from anchor and retry
					self.restore_from_anchor(&anchor);
					continue;
				}
			}
		}
	}

	/// Attempts to move to the previous (leftward) leaf.
	///
	/// This is the reverse direction equivalent of `next_leaf()`.
	fn prev_leaf(&mut self) -> LeafResult {
		if self.leaf.is_none() {
			return LeafResult::End;
		}

		let anchor = self.current_anchor().expect("leaf exists");

		loop {
			// Check if we actually need to leave this leaf
			{
				let (guard, cursor) = self.leaf.as_ref().unwrap();
				let leaf = guard.as_leaf();

				match *cursor {
					// If before position > 0, we have entries before us
					Cursor::Before(pos) if pos > 0 => {
						return LeafResult::Retry;
					}
					// If after any position, we can still go backwards
					Cursor::After(_pos) => {
						return LeafResult::Retry;
					}
					_ => {}
				}

				// Check if we're at the leftmost leaf
				if leaf.lower_fence.is_none() || self.parent.is_none() {
					return LeafResult::End;
				}
			}

			// Release current leaf and attempt jump
			let _ = self.leaf.take();

			match self.optimistic_jump(Direction::Reverse) {
				JumpResult::Ok => {
					return LeafResult::Ok;
				}
				JumpResult::End => {
					return LeafResult::End;
				}
				JumpResult::Err => {
					self.restore_from_anchor(&anchor);
					continue;
				}
			}
		}
	}

	/// Performs an optimistic jump to a sibling leaf.
	///
	/// This is the core mechanism for moving between leaves during iteration.
	/// It tries two strategies:
	///
	/// 1. **Fast path**: If the sibling is in the same parent, use the cached
	///    parent guard to directly access it
	/// 2. **Slow path**: If we need to traverse up/down the tree to find the
	///    sibling, use `find_nearest_leaf` (which may traverse multiple levels)
	///
	/// # Parameters
	///
	/// - `direction`: Which sibling to jump to (Forward = right, Reverse = left)
	///
	/// # Returns
	///
	/// - `JumpResult::Ok`: Successfully acquired the sibling
	/// - `JumpResult::End`: No sibling exists
	/// - `JumpResult::Err`: Optimistic validation failed, need to recover
	fn optimistic_jump(&mut self, direction: Direction) -> JumpResult {
		// Internal enum to capture the different outcomes
		enum Outcome<L, P> {
			/// Got leaf from same parent (fast path)
			Leaf(L, u16),
			/// Got leaf with new parent (slow path)
			LeafAndParent(L, P, u16),
			/// No sibling exists
			End,
		}

		// The actual optimistic operation - may fail validation
		let optimistic_perform = || {
			if let Some((parent_guard, p_cursor)) = self.parent.as_ref() {
				// Check if sibling is within the same parent
				let bounded_pos = match direction {
					Direction::Forward if *p_cursor < parent_guard.as_internal().len => {
						Some(*p_cursor + 1)
					}
					Direction::Reverse if *p_cursor > 0 => Some(*p_cursor - 1),
					_ => None,
				};

				if let Some(pos) = bounded_pos {
					// Fast path: sibling is in the same parent
					let guard = {
						let swip = parent_guard.as_internal().edge_at(pos)?;
						GenericTree::lock_coupling(parent_guard, swip, &self.eg)?
					};

					assert!(guard.is_leaf());

					// Upgrade to shared lock
					error::Result::Ok(Outcome::Leaf(Self::leaf_lt(guard.to_shared()?), pos))
				} else {
					// Slow path: need to traverse up/down to find sibling
					let opt =
						match self.tree.find_nearest_leaf(parent_guard, direction, &self.eg)? {
							Some((guard, (parent, pos))) => Outcome::LeafAndParent(
								Self::leaf_lt(guard.to_shared()?),
								Self::parent_lt(parent),
								pos,
							),
							None => Outcome::End,
						};
					error::Result::Ok(opt)
				}
			} else {
				// No parent means we can't traverse
				error::Result::Ok(Outcome::End)
			}
		};

		// Execute and handle the result
		match optimistic_perform() {
			Ok(Outcome::Leaf(leaf_guard, p_cursor)) => {
				// Set cursor to start (Forward) or end (Reverse) of new leaf
				let l_cursor = match direction {
					Direction::Forward => Cursor::Before(0),
					Direction::Reverse => Cursor::After(leaf_guard.as_leaf().len - 1),
				};

				self.leaf = Some((leaf_guard, l_cursor));
				// Update parent cursor position
				if let Some((_, p_c)) = self.parent.as_mut() {
					*p_c = p_cursor;
				}

				JumpResult::Ok
			}
			Ok(Outcome::LeafAndParent(leaf_guard, parent_guard, p_cursor)) => {
				let l_cursor = match direction {
					Direction::Forward => Cursor::Before(0),
					Direction::Reverse => Cursor::After(leaf_guard.as_leaf().len - 1),
				};

				self.leaf = Some((leaf_guard, l_cursor));
				self.parent = Some((parent_guard, p_cursor));

				JumpResult::Ok
			}
			Ok(Outcome::End) => JumpResult::End,
			Err(_) => JumpResult::Err,
		}
	}

	// -----------------------------------------------------------------------
	// Seek Methods
	// -----------------------------------------------------------------------

	/// Positions the cursor immediately before the given key.
	///
	/// After this call, `next()` will return the entry at or after `key`,
	/// and `prev()` will return the entry before `key`.
	///
	/// # Optimization
	///
	/// If we're already positioned on a leaf that contains the target key
	/// (checked via `within_bounds`), we reuse the current leaf instead of
	/// re-traversing the tree.
	pub fn seek<Q>(&mut self, key: &Q)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		// Try to reuse current leaf if the key is within its bounds
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().within_bounds(key) => (guard, self.parent.take()),
			_ => {
				// Need to find the correct leaf from scratch
				let (guard, parent_opt) =
					self.tree.find_shared_leaf_and_optimistic_parent(key, &self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		// Find position within the leaf
		let leaf = guard.as_leaf();
		let leaf_len = leaf.len;
		let (pos, _) = leaf.lower_bound(key);

		// Set cursor before the found position (or end if past bounds)
		if pos >= leaf_len {
			self.leaf = Some((guard, Cursor::Before(leaf_len)));
			self.parent = parent_opt;
		} else {
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		}
	}

	/// Positions the cursor so that `prev()` returns the largest key <= the given key.
	///
	/// This follows RocksDB-style `SeekForPrev` semantics:
	/// - If the key exists, `prev()` will return that key
	/// - If the key doesn't exist, `prev()` will return the largest key less than it
	/// - If no key <= the given key exists, `prev()` will return `None`
	///
	/// This is useful for reverse iteration starting from a specific key.
	pub fn seek_for_prev<Q>(&mut self, key: &Q)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().within_bounds(key) => (guard, self.parent.take()),
			_ => {
				let (guard, parent_opt) =
					self.tree.find_shared_leaf_and_optimistic_parent(key, &self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		let leaf = guard.as_leaf();
		let leaf_len = leaf.len;
		let (pos, exact) = leaf.lower_bound(key);

		// Position cursor based on whether we found an exact match
		if exact {
			// Key found - position after it so prev() returns it
			self.leaf = Some((guard, Cursor::After(pos)));
			self.parent = parent_opt;
		} else if pos == 0 {
			// Key would be before all entries - position at start
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		} else if pos >= leaf_len {
			// Key would be after all entries - position after last entry so prev() returns it
			self.leaf = Some((guard, Cursor::After(leaf_len - 1)));
			self.parent = parent_opt;
		} else {
			// Key not found but in range - position after the previous entry so prev() returns largest key < seek_key
			self.leaf = Some((guard, Cursor::After(pos - 1)));
			self.parent = parent_opt;
		}
	}

	/// Positions the cursor immediately before the given key and returns whether
	/// the key exists.
	///
	/// After this call, if `true` is returned, `next()` will return the entry
	/// with exactly this key.
	///
	/// # Returns
	///
	/// - `true` if the key exists in the tree
	/// - `false` if the key doesn't exist (cursor still positioned for insertion point)
	pub fn seek_exact<Q>(&mut self, key: &Q) -> bool
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().within_bounds(key) => (guard, self.parent.take()),
			_ => {
				let (guard, parent_opt) =
					self.tree.find_shared_leaf_and_optimistic_parent(key, &self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		let leaf = guard.as_leaf();
		let leaf_len = leaf.len;
		let (pos, exact) = leaf.lower_bound(key);

		// Always position before the found position
		if pos >= leaf_len {
			self.leaf = Some((guard, Cursor::Before(leaf_len)));
			self.parent = parent_opt;
		} else {
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		}

		exact
	}

	/// Positions the cursor at the very beginning of the tree.
	///
	/// After this call, `next()` will return the first (smallest key) entry,
	/// and `prev()` will return `None`.
	pub fn seek_to_first(&mut self) {
		// Check if current leaf is already the first leaf (lower_fence = None)
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().lower_fence.is_none() => {
				(guard, self.parent.take())
			}
			_ => {
				self.parent.take();
				let (guard, parent_opt) =
					self.tree.find_first_shared_leaf_and_optimistic_parent(&self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		// Position before the first entry
		self.leaf = Some((guard, Cursor::Before(0)));
		self.parent = parent_opt;
	}

	/// Positions the cursor at the very end of the tree.
	///
	/// After this call, `next()` will return `None`, and `prev()` will return
	/// the last (largest key) entry.
	pub fn seek_to_last(&mut self) {
		// Check if current leaf is already the last leaf (upper_fence = None)
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().upper_fence.is_none() => {
				(guard, self.parent.take())
			}
			_ => {
				self.parent.take();
				let (guard, parent_opt) =
					self.tree.find_last_shared_leaf_and_optimistic_parent(&self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		// Position after the last entry (before position = len)
		let leaf_len = guard.as_leaf().len;
		self.leaf = Some((guard, Cursor::Before(leaf_len)));
		self.parent = parent_opt;
	}

	// -----------------------------------------------------------------------
	// Iteration Methods
	// -----------------------------------------------------------------------

	/// Returns the next entry in key order.
	///
	/// Advances the cursor forward and returns the next key-value pair.
	/// Returns `None` when there are no more entries.
	///
	/// # Cursor Behavior
	///
	/// - If `Cursor::Before(n)`: Returns entry at `n`, cursor becomes `Before(n+1)`
	/// - If `Cursor::After(n)`: Returns entry at `n+1`, cursor becomes `Before(n+2)`
	/// - If at end of leaf: Moves to next leaf and continues
	///
	/// Note: This method intentionally doesn't implement `Iterator` because the
	/// iterator holds locks and has complex retry semantics incompatible with
	/// the standard trait.
	#[inline]
	#[allow(clippy::should_implement_trait)]
	pub fn next(&mut self) -> Option<(&K, &V)> {
		loop {
			// Determine what to return based on current cursor position
			let opt = match self.leaf.as_ref() {
				Some((guard, cursor)) => cursor.next_entry(guard.as_leaf().len),
				None => return None,
			};

			if let Some((curr_pos, new_cursor)) = opt {
				// We have an entry to return from current leaf
				let (guard, cursor) = self.leaf.as_mut().unwrap();
				let leaf = guard.as_leaf();
				*cursor = new_cursor;
				// SAFETY: cursor.next_entry() already validated that curr_pos < leaf.len
				return Some(unsafe { leaf.kv_at_unchecked(curr_pos) });
			} else {
				// Current leaf exhausted - try to move to next leaf
				match self.next_leaf() {
					LeafResult::Ok | LeafResult::Retry => {
						// Got new leaf or current leaf has entries - continue loop
						continue;
					}
					LeafResult::End => {
						// No more leaves - iteration complete
						return None;
					}
				}
			}
		}
	}

	/// Returns the previous entry in key order.
	///
	/// Moves the cursor backward and returns the previous key-value pair.
	/// Returns `None` when there are no more entries.
	///
	/// # Cursor Behavior
	///
	/// - If `Cursor::After(n)`: Returns entry at `n`, cursor becomes `After(n-1)` or `Before(0)`
	/// - If `Cursor::Before(n)`: Returns entry at `n-1` (if n > 0)
	/// - If at start of leaf: Moves to previous leaf and continues
	#[inline]
	pub fn prev(&mut self) -> Option<(&K, &V)> {
		loop {
			let opt = match self.leaf.as_ref() {
				Some((_guard, cursor)) => cursor.prev_entry(),
				None => return None,
			};

			if let Some((curr_pos, new_cursor)) = opt {
				let (guard, cursor) = self.leaf.as_mut().unwrap();
				let leaf = guard.as_leaf();
				*cursor = new_cursor;
				// SAFETY: cursor.prev_entry() already validated that curr_pos is valid
				return Some(unsafe { leaf.kv_at_unchecked(curr_pos) });
			} else {
				// Current leaf exhausted - try to move to previous leaf
				match self.prev_leaf() {
					LeafResult::Ok | LeafResult::Retry => {
						continue;
					}
					LeafResult::End => {
						return None;
					}
				}
			}
		}
	}

	/// Returns the next entry without advancing the cursor.
	///
	/// Repeated calls return the same entry until `next()` or `prev()` is called.
	/// If at a leaf boundary, may advance to the next leaf (but not the cursor
	/// position within that leaf).
	///
	/// # Usage
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// tree.insert(1, "one");
	/// tree.insert(2, "two");
	///
	/// let mut iter = tree.raw_iter();
	/// iter.seek_to_first();
	///
	/// // peek() returns the same value repeatedly
	/// assert_eq!(iter.peek(), Some((&1, &"one")));
	/// assert_eq!(iter.peek(), Some((&1, &"one")));
	///
	/// // next() returns the same value and advances
	/// assert_eq!(iter.next(), Some((&1, &"one")));
	/// assert_eq!(iter.peek(), Some((&2, &"two")));
	/// ```
	#[inline]
	pub fn peek(&mut self) -> Option<(&K, &V)> {
		loop {
			// Get position to return (or None if at end of leaf)
			// This captures the position as a value, ending the borrow
			let pos_opt = match self.leaf.as_ref() {
				Some((guard, cursor)) => cursor.peek_next_pos(guard.as_leaf().len),
				None => return None,
			};

			if let Some(pos) = pos_opt {
				// Return entry at position (new borrow)
				let (guard, _) = self.leaf.as_ref().unwrap();
				// SAFETY: peek_next_pos() already validated that pos < leaf.len
				return Some(unsafe { guard.as_leaf().kv_at_unchecked(pos) });
			} else {
				// At end of current leaf - try to move to next leaf
				match self.next_leaf() {
					LeafResult::Ok | LeafResult::Retry => continue,
					LeafResult::End => return None,
				}
			}
		}
	}

	/// Returns the previous entry without moving the cursor backward.
	///
	/// Repeated calls return the same entry until `next()` or `prev()` is called.
	/// If at a leaf boundary, may move to the previous leaf (but not the cursor
	/// position within that leaf).
	#[inline]
	pub fn peek_prev(&mut self) -> Option<(&K, &V)> {
		loop {
			// Get position to return (or None if at start of leaf)
			let pos_opt = match self.leaf.as_ref() {
				Some((_, cursor)) => cursor.peek_prev_pos(),
				None => return None,
			};

			if let Some(pos) = pos_opt {
				// Return entry at position (new borrow)
				let (guard, _) = self.leaf.as_ref().unwrap();
				// SAFETY: peek_prev_pos() already validated that pos is valid
				return Some(unsafe { guard.as_leaf().kv_at_unchecked(pos) });
			} else {
				// At start of current leaf - try to move to previous leaf
				match self.prev_leaf() {
					LeafResult::Ok | LeafResult::Retry => continue,
					LeafResult::End => return None,
				}
			}
		}
	}

	/// Process all remaining entries in the current leaf with a closure.
	///
	/// This method iterates through all entries from the current cursor position
	/// to the end of the current leaf, calling the provided closure for each entry.
	/// After exhausting the leaf, the cursor is positioned at the end.
	///
	/// Returns `true` if more leaves may exist (for continued iteration),
	/// `false` if this was the last leaf.
	///
	/// This is more efficient than repeated `next()` calls when processing
	/// many sequential entries, as it avoids per-entry cursor bookkeeping.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, i32> = Tree::new();
	/// for i in 0..100 {
	///     tree.insert(i, i * 10);
	/// }
	///
	/// let mut iter = tree.raw_iter();
	/// iter.seek_to_first();
	///
	/// let mut sum = 0i64;
	/// loop {
	///     let has_more = iter.for_each_in_leaf(|k, v| {
	///         sum += (*k as i64) + (*v as i64);
	///     });
	///     if !has_more {
	///         break;
	///     }
	///     // Move to next leaf if exists
	///     if iter.next().is_none() {
	///         break;
	///     }
	/// }
	/// ```
	#[inline]
	pub fn for_each_in_leaf<F>(&mut self, mut f: F) -> bool
	where
		F: FnMut(&K, &V),
	{
		let Some((guard, cursor)) = self.leaf.as_mut() else {
			return false;
		};

		let leaf = guard.as_leaf();
		let leaf_len = leaf.len;

		// Determine starting position based on cursor state
		let start = match *cursor {
			Cursor::Before(pos) => pos,
			Cursor::After(pos) => pos.saturating_add(1),
		};

		// Process all remaining entries in this leaf
		// SAFETY: i is bounded by start..leaf_len where:
		// - start comes from cursor state which tracks valid positions
		// - leaf_len is the actual length of the leaf
		// - the loop only iterates while i < leaf_len
		for i in start..leaf_len {
			let (k, v) = unsafe { leaf.kv_at_unchecked(i) };
			f(k, v);
		}

		// Mark cursor as exhausted (positioned at end of leaf)
		*cursor = Cursor::Before(leaf_len);

		// Return whether more leaves may exist
		leaf.upper_fence.is_some()
	}
}

// ===========================================================================
// RawExclusiveIter - Read-Write Iterator
// ===========================================================================

/// A read-write iterator over the entries of a B+ tree.
///
/// This iterator acquires exclusive (write) locks on leaf nodes, which:
/// - Blocks all other readers and writers on the current leaf
/// - Allows in-place modification of values
/// - Enables insert and remove operations during iteration
///
/// The exclusive iterator is used internally by `tree.insert()` to perform
/// insertions with potential node splits.
///
/// # Usage
///
/// ```
/// use ferntree::Tree;
///
/// let tree: Tree<i32, i32> = Tree::new();
/// tree.insert(1, 10);
/// tree.insert(2, 20);
///
/// let mut iter = tree.raw_iter_mut();
/// iter.seek_to_first();
///
/// // Modify values in place
/// while let Some((k, v)) = iter.next() {
///     *v *= 2;
/// }
/// ```
///
/// # Insert and Split
///
/// The `insert()` method handles the case where a leaf becomes full:
/// 1. Release the leaf lock
/// 2. Trigger a split operation
/// 3. Re-seek to the correct position
/// 4. Retry the insertion
pub struct RawExclusiveIter<'t, K, V, const IC: usize, const LC: usize> {
	/// Reference to the tree being iterated.
	tree: &'t GenericTree<K, V, IC, LC>,
	/// Epoch guard - pinned for the iterator's lifetime.
	eg: epoch::Guard,
	/// Optimistic guard on the parent of the current leaf.
	parent: Option<(OptimisticGuard<'t, Node<K, V, IC, LC>>, u16)>,
	/// Exclusive guard on the current leaf and our cursor position.
	/// The exclusive lock blocks all other access to this leaf.
	leaf: Option<(ExclusiveGuard<'t, Node<K, V, IC, LC>>, Cursor)>,
}

impl<'t, K: Clone + Ord, V, const IC: usize, const LC: usize> RawExclusiveIter<'t, K, V, IC, LC> {
	/// Creates a new exclusive iterator, pinning a fresh epoch guard.
	///
	/// Call `seek*` methods to position before iterating.
	pub(crate) fn new(tree: &'t GenericTree<K, V, IC, LC>) -> RawExclusiveIter<'t, K, V, IC, LC> {
		RawExclusiveIter {
			tree,
			eg: epoch::pin(),
			parent: None,
			leaf: None,
		}
	}

	// -----------------------------------------------------------------------
	// Lifetime Extension Helpers
	// -----------------------------------------------------------------------
	// Same safety rationale as RawSharedIter - the epoch guard ensures memory
	// remains valid for the iterator's lifetime.

	/// Extends the lifetime of a leaf guard from 'g to 't.
	///
	/// # Safety
	///
	/// Safe because we maintain the epoch guard for our entire lifetime.
	#[inline]
	fn leaf_lt<'g>(
		guard: ExclusiveGuard<'g, Node<K, V, IC, LC>>,
	) -> ExclusiveGuard<'t, Node<K, V, IC, LC>> {
		// SAFETY: Epoch guard protects the memory
		unsafe { std::mem::transmute(guard) }
	}

	/// Extends the lifetime of a parent guard from 'g to 't.
	#[inline]
	fn parent_lt<'g>(
		guard: OptimisticGuard<'g, Node<K, V, IC, LC>>,
	) -> OptimisticGuard<'t, Node<K, V, IC, LC>> {
		// SAFETY: Epoch guard protects the memory
		unsafe { std::mem::transmute(guard) }
	}

	// -----------------------------------------------------------------------
	// Anchor Management (same as RawSharedIter)
	// -----------------------------------------------------------------------

	/// Creates an anchor representing our current logical position.
	/// See `RawSharedIter::current_anchor` for detailed documentation.
	fn current_anchor(&self) -> Option<Anchor<K>> {
		if let Some((guard, cursor)) = self.leaf.as_ref() {
			let leaf = guard.as_leaf();
			let anchor = match *cursor {
				Cursor::Before(pos) => {
					if pos >= leaf.len {
						if leaf.len > 0 {
							Anchor::After(
								leaf.key_at(leaf.len - 1)
									.expect("leaf.len > 0 implies key exists at len-1")
									.clone(),
							)
						} else if let Some(k) = &leaf.lower_fence {
							Anchor::After(k.clone())
						} else {
							Anchor::Start
						}
					} else {
						Anchor::Before(
							leaf.key_at(pos)
								.expect("pos < leaf.len implies key exists at pos")
								.clone(),
						)
					}
				}
				Cursor::After(pos) => {
					if pos >= leaf.len {
						if leaf.len > 0 {
							Anchor::After(
								leaf.key_at(leaf.len - 1)
									.expect("leaf.len > 0 implies key exists at len-1")
									.clone(),
							)
						} else if let Some(k) = &leaf.upper_fence {
							Anchor::After(k.clone())
						} else {
							Anchor::End
						}
					} else {
						Anchor::After(
							leaf.key_at(pos)
								.expect("pos < leaf.len implies key exists at pos")
								.clone(),
						)
					}
				}
			};

			Some(anchor)
		} else {
			None
		}
	}

	/// Restores the iterator position from an anchor.
	fn restore_from_anchor(&mut self, anchor: &Anchor<K>) {
		self.parent.take();
		self.leaf.take();

		match anchor {
			Anchor::Start => self.seek_to_first(),
			Anchor::End => self.seek_to_last(),
			Anchor::Before(key) => self.seek(key),
			Anchor::After(key) => self.seek_for_prev(key),
		}
	}

	// -----------------------------------------------------------------------
	// Leaf Traversal (same structure as RawSharedIter, but with exclusive locks)
	// -----------------------------------------------------------------------

	/// Attempts to move to the next (rightward) leaf.
	/// See `RawSharedIter::next_leaf` for detailed documentation.
	fn next_leaf(&mut self) -> LeafResult {
		if self.leaf.is_none() {
			return LeafResult::End;
		}

		let anchor = self.current_anchor().expect("leaf exists");

		loop {
			{
				let (guard, cursor) = self.leaf.as_ref().unwrap();
				let leaf = guard.as_leaf();
				match *cursor {
					Cursor::Before(pos) if pos < leaf.len => {
						return LeafResult::Retry;
					}
					Cursor::After(pos) if (pos + 1) < leaf.len => {
						return LeafResult::Retry;
					}
					_ => {}
				}

				if leaf.upper_fence.is_none() || self.parent.is_none() {
					return LeafResult::End;
				}
			}

			let _ = self.leaf.take();

			match self.optimistic_jump(Direction::Forward) {
				JumpResult::Ok => {
					return LeafResult::Ok;
				}
				JumpResult::End => {
					return LeafResult::End;
				}
				JumpResult::Err => {
					self.restore_from_anchor(&anchor);
					continue;
				}
			}
		}
	}

	/// Attempts to move to the previous (leftward) leaf.
	fn prev_leaf(&mut self) -> LeafResult {
		if self.leaf.is_none() {
			return LeafResult::End;
		}

		let anchor = self.current_anchor().expect("leaf exists");

		loop {
			{
				let (guard, cursor) = self.leaf.as_ref().unwrap();
				let leaf = guard.as_leaf();
				match *cursor {
					Cursor::Before(pos) if pos > 0 => {
						return LeafResult::Retry;
					}
					Cursor::After(_pos) => {
						return LeafResult::Retry;
					}
					_ => {}
				}

				if leaf.lower_fence.is_none() || self.parent.is_none() {
					return LeafResult::End;
				}
			}

			let _ = self.leaf.take();

			match self.optimistic_jump(Direction::Reverse) {
				JumpResult::Ok => {
					return LeafResult::Ok;
				}
				JumpResult::End => {
					return LeafResult::End;
				}
				JumpResult::Err => {
					self.restore_from_anchor(&anchor);
					continue;
				}
			}
		}
	}

	/// Performs an optimistic jump to a sibling leaf, acquiring exclusive access.
	///
	/// Same algorithm as `RawSharedIter::optimistic_jump`, but upgrades to
	/// exclusive lock instead of shared.
	fn optimistic_jump(&mut self, direction: Direction) -> JumpResult {
		enum Outcome<L, P> {
			Leaf(L, u16),
			LeafAndParent(L, P, u16),
			End,
		}

		let optimistic_perform = || {
			if let Some((parent_guard, p_cursor)) = self.parent.as_ref() {
				let bounded_pos = match direction {
					Direction::Forward if *p_cursor < parent_guard.as_internal().len => {
						Some(*p_cursor + 1)
					}
					Direction::Reverse if *p_cursor > 0 => Some(*p_cursor - 1),
					_ => None,
				};
				if let Some(pos) = bounded_pos {
					// Fast path: sibling in same parent
					let guard = {
						let swip = parent_guard.as_internal().edge_at(pos)?;
						GenericTree::lock_coupling(parent_guard, swip, &self.eg)?
					};

					assert!(guard.is_leaf());

					// Upgrade to EXCLUSIVE (not shared like RawSharedIter)
					error::Result::Ok(Outcome::Leaf(Self::leaf_lt(guard.to_exclusive()?), pos))
				} else {
					// Slow path: traverse up/down
					let opt =
						match self.tree.find_nearest_leaf(parent_guard, direction, &self.eg)? {
							Some((guard, (parent, pos))) => Outcome::LeafAndParent(
								Self::leaf_lt(guard.to_exclusive()?),
								Self::parent_lt(parent),
								pos,
							),
							None => Outcome::End,
						};
					error::Result::Ok(opt)
				}
			} else {
				error::Result::Ok(Outcome::End)
			}
		};

		match optimistic_perform() {
			Ok(Outcome::Leaf(leaf_guard, p_cursor)) => {
				let l_cursor = match direction {
					Direction::Forward => Cursor::Before(0),
					Direction::Reverse => Cursor::After(leaf_guard.as_leaf().len - 1),
				};

				self.leaf = Some((leaf_guard, l_cursor));
				if let Some((_, p_c)) = self.parent.as_mut() {
					*p_c = p_cursor;
				}

				JumpResult::Ok
			}
			Ok(Outcome::LeafAndParent(leaf_guard, parent_guard, p_cursor)) => {
				let l_cursor = match direction {
					Direction::Forward => Cursor::Before(0),
					Direction::Reverse => Cursor::After(leaf_guard.as_leaf().len - 1),
				};

				self.leaf = Some((leaf_guard, l_cursor));
				self.parent = Some((parent_guard, p_cursor));

				JumpResult::Ok
			}
			Ok(Outcome::End) => JumpResult::End,
			Err(_) => JumpResult::Err,
		}
	}

	// -----------------------------------------------------------------------
	// Seek Methods (same API as RawSharedIter, but acquires exclusive locks)
	// -----------------------------------------------------------------------

	/// Positions the cursor immediately before the given key (exclusive lock).
	///
	/// See `RawSharedIter::seek` for detailed documentation.
	pub fn seek<Q>(&mut self, key: &Q)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		// Try to reuse current leaf if key is within its bounds
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().within_bounds(key) => (guard, self.parent.take()),
			_ => {
				self.parent.take();
				// Acquire EXCLUSIVE lock on the target leaf
				let (guard, parent_opt) =
					self.tree.find_exclusive_leaf_and_optimistic_parent(key, &self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		let leaf = guard.as_leaf();
		let leaf_len = leaf.len;
		let (pos, _) = leaf.lower_bound(key);
		if pos >= leaf_len {
			self.leaf = Some((guard, Cursor::Before(leaf_len)));
			self.parent = parent_opt;
		} else {
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		}
	}

	/// Positions the cursor so that `prev()` returns the largest key <= the given key (exclusive lock).
	///
	/// See `RawSharedIter::seek_for_prev` for detailed documentation.
	pub fn seek_for_prev<Q>(&mut self, key: &Q)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().within_bounds(key) => (guard, self.parent.take()),
			_ => {
				self.parent.take();
				let (guard, parent_opt) =
					self.tree.find_exclusive_leaf_and_optimistic_parent(key, &self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		let leaf = guard.as_leaf();
		let leaf_len = leaf.len;
		let (pos, exact) = leaf.lower_bound(key);
		if exact {
			self.leaf = Some((guard, Cursor::After(pos)));
			self.parent = parent_opt;
		} else if pos == 0 {
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		} else if pos >= leaf_len {
			// Key would be after all entries - position after last entry so prev() returns it
			self.leaf = Some((guard, Cursor::After(leaf_len - 1)));
			self.parent = parent_opt;
		} else {
			// Key not found but in range - position after the previous entry so prev() returns largest key < seek_key
			self.leaf = Some((guard, Cursor::After(pos - 1)));
			self.parent = parent_opt;
		}
	}

	/// Positions the cursor before the key and returns whether the key exists.
	///
	/// This is the primary method used by `insert()` to find the insertion point.
	/// See `RawSharedIter::seek_exact` for detailed documentation.
	pub fn seek_exact<Q>(&mut self, key: &Q) -> bool
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().within_bounds(key) => (guard, self.parent.take()),
			_ => {
				self.parent.take();
				let (guard, parent_opt) =
					self.tree.find_exclusive_leaf_and_optimistic_parent(key, &self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		let leaf = guard.as_leaf();
		let leaf_len = leaf.len;
		let (pos, exact) = leaf.lower_bound(key);
		if pos >= leaf_len {
			self.leaf = Some((guard, Cursor::Before(leaf_len)));
			self.parent = parent_opt;
		} else {
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		}

		exact
	}

	/// Positions the cursor at the very beginning of the tree (exclusive lock).
	pub fn seek_to_first(&mut self) {
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().lower_fence.is_none() => {
				(guard, self.parent.take())
			}
			_ => {
				self.parent.take();
				let (guard, parent_opt) =
					self.tree.find_first_exclusive_leaf_and_optimistic_parent(&self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		self.leaf = Some((guard, Cursor::Before(0)));
		self.parent = parent_opt;
	}

	/// Positions the cursor at the very end of the tree (exclusive lock).
	pub fn seek_to_last(&mut self) {
		let (guard, parent_opt) = match self.leaf.take() {
			Some((guard, _)) if guard.as_leaf().upper_fence.is_none() => {
				(guard, self.parent.take())
			}
			_ => {
				self.parent.take();
				let (guard, parent_opt) =
					self.tree.find_last_exclusive_leaf_and_optimistic_parent(&self.eg);
				(
					Self::leaf_lt(guard),
					parent_opt.map(|(parent, pos)| (Self::parent_lt(parent), pos)),
				)
			}
		};

		let leaf_len = guard.as_leaf().len;
		self.leaf = Some((guard, Cursor::Before(leaf_len)));
		self.parent = parent_opt;
	}

	// -----------------------------------------------------------------------
	// Mutation Methods
	// -----------------------------------------------------------------------

	/// Inserts a key-value pair into the tree.
	///
	/// If the key already exists, updates the value and returns the old value.
	/// If the key is new, inserts it and returns `None`.
	///
	/// # Algorithm
	///
	/// 1. Seek to the key's position
	/// 2. If key exists: Update value in place
	/// 3. If key is new and leaf has space: Insert directly
	/// 4. If key is new and leaf is full:
	///    a. Release the leaf lock
	///    b. Trigger split operation
	///    c. Retry from step 1 (tree structure may have changed)
	///
	/// # Split Handling
	///
	/// The split operation is performed optimistically:
	/// - We release our lock before splitting (splits need parent access)
	/// - After split, we re-seek because the key might now be in a different leaf
	/// - This loop continues until we find a leaf with space
	pub fn insert(&mut self, key: K, value: V) -> Option<V> {
		'start: loop {
			// Step 1: Find the position for this key
			if self.seek_exact(&key) {
				// Step 2: Key exists - update the value
				let (_k, v) = self.next().unwrap();
				let old = std::mem::replace(v, value);
				break Some(old);
			} else {
				// Key doesn't exist - need to insert
				let (guard, cursor) = self.leaf.as_mut().expect("just seeked");

				if guard.as_leaf().has_space() {
					// Step 3: Leaf has space - insert directly
					let leaf = guard.as_leaf_mut();
					match *cursor {
						Cursor::Before(pos) => {
							leaf.insert_at(pos, key, value).expect("just checked for space");
						}
						Cursor::After(_) => {
							// seek_exact always positions Before
							unreachable!("seek_exact always sets cursor to before");
						}
					}
					break None;
				} else {
					// Step 4: Leaf is full - need to split first
					// Release our locks before splitting
					self.parent.take();
					let (guard, _cursor) = self.leaf.take().expect("just seeked");

					// Unlock exclusive guard to get an optimistic guard for split
					let mut guard = guard.unlock();

					// Attempt to split the leaf
					loop {
						let perform_split = || {
							// Double-check the leaf still needs splitting
							if !guard.as_leaf().has_space() {
								guard.recheck()?;
								// Perform the split operation
								self.tree.try_split(&guard, &self.eg)?;
							}
							error::Result::Ok(())
						};

						match perform_split() {
							Ok(_) => {
								// Split succeeded (or wasn't needed) - break inner loop
								break;
							}
							Err(error::Error::Reclaimed) => {
								// Tree structure changed significantly - restart from scratch
								continue 'start;
							}
							Err(_) => {
								// Optimistic validation failed - re-acquire and retry
								guard = guard.latch().optimistic_or_spin();
								continue;
							}
						}
					}

					// After split, restart the outer loop to re-seek
					// (the key might now be in a different leaf)
					continue;
				}
			}
		}
	}

	/// Removes the entry with the given key from the tree.
	///
	/// # Returns
	///
	/// - `Some((key, value))` if the key existed and was removed
	/// - `None` if the key didn't exist
	pub fn remove<Q>(&mut self, key: &Q) -> Option<(K, V)>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		if self.seek_exact(key) {
			// Key exists - remove it
			Some(self.remove_next().expect("just seeked for remove"))
		} else {
			// Key doesn't exist
			None
		}
	}

	/// Removes the entry at the current cursor position.
	///
	/// This is the internal implementation for `remove()`. After removal,
	/// if the leaf becomes underfull, it attempts to merge with a sibling.
	fn remove_next(&mut self) -> Option<(K, V)> {
		match self.leaf.as_mut() {
			Some((guard, cursor)) => {
				let leaf = guard.as_leaf_mut();

				// Remove the entry based on cursor position
				let removed = match cursor {
					Cursor::Before(pos) => {
						let curr_pos = *pos;
						if curr_pos < leaf.len {
							Some(leaf.remove_at(curr_pos))
						} else {
							None
						}
					}
					Cursor::After(pos) => {
						let pos = *pos;
						let curr_pos = pos + 1;
						if curr_pos < leaf.len {
							Some(leaf.remove_at(curr_pos))
						} else {
							None
						}
					}
				};

				// Check if merge is needed after removal
				if let Some((removed_key, _)) = removed.as_ref() {
					if guard.is_underfull() {
						// Release locks before merge operation
						self.parent.take();
						let (guard, _cursor) = self.leaf.take().expect("just seeked");

						// Unlock to get optimistic guard for merge
						let guard = guard.unlock();

						// Attempt to merge with a sibling (best-effort, ignore result)
						let _ = self.tree.try_merge(&guard, &self.eg);

						// Re-position iterator after structural change
						self.seek(removed_key);
					}
				}

				removed
			}
			None => None,
		}
	}

	// -----------------------------------------------------------------------
	// Iteration Methods (returns mutable value references)
	// -----------------------------------------------------------------------

	/// Returns the next entry with a mutable reference to the value.
	///
	/// This differs from `RawSharedIter::next()` in that:
	/// 1. It holds an exclusive lock (blocking all other access)
	/// 2. It returns `&mut V` instead of `&V`
	///
	/// This allows modifying values during iteration.
	///
	/// Note: This method intentionally doesn't implement `Iterator` because the
	/// iterator holds locks and has complex retry semantics incompatible with
	/// the standard trait.
	#[inline]
	#[allow(clippy::should_implement_trait)]
	pub fn next(&mut self) -> Option<(&K, &mut V)> {
		loop {
			let opt = match self.leaf.as_ref() {
				Some((guard, cursor)) => cursor.next_entry(guard.as_leaf().len),
				None => return None,
			};

			if let Some((curr_pos, new_cursor)) = opt {
				let (guard, cursor) = self.leaf.as_mut().unwrap();
				// Get mutable access to the leaf
				let leaf = guard.as_leaf_mut();
				*cursor = new_cursor;
				// SAFETY: cursor.next_entry() already validated that curr_pos < leaf.len
				return Some(unsafe { leaf.kv_at_mut_unchecked(curr_pos) });
			} else {
				match self.next_leaf() {
					LeafResult::Ok | LeafResult::Retry => {
						continue;
					}
					LeafResult::End => {
						return None;
					}
				}
			}
		}
	}

	/// Returns the previous entry with a mutable reference to the value.
	#[inline]
	pub fn prev(&mut self) -> Option<(&K, &mut V)> {
		loop {
			let opt = match self.leaf.as_ref() {
				Some((_guard, cursor)) => cursor.prev_entry(),
				None => return None,
			};

			if let Some((curr_pos, new_cursor)) = opt {
				let (guard, cursor) = self.leaf.as_mut().unwrap();
				let leaf = guard.as_leaf_mut();
				*cursor = new_cursor;
				// SAFETY: cursor.prev_entry() already validated that curr_pos is valid
				return Some(unsafe { leaf.kv_at_mut_unchecked(curr_pos) });
			} else {
				match self.prev_leaf() {
					LeafResult::Ok | LeafResult::Retry => {
						continue;
					}
					LeafResult::End => {
						return None;
					}
				}
			}
		}
	}

	/// Returns the next entry without advancing the cursor (mutable value reference).
	///
	/// Repeated calls return the same entry until `next()` or `prev()` is called.
	/// If at a leaf boundary, may advance to the next leaf (but not the cursor
	/// position within that leaf).
	#[inline]
	pub fn peek(&mut self) -> Option<(&K, &mut V)> {
		loop {
			// Get position to return (or None if at end of leaf)
			let pos_opt = match self.leaf.as_ref() {
				Some((guard, cursor)) => cursor.peek_next_pos(guard.as_leaf().len),
				None => return None,
			};

			if let Some(pos) = pos_opt {
				// Return entry at position (new mutable borrow)
				let (guard, _) = self.leaf.as_mut().unwrap();
				// SAFETY: peek_next_pos() already validated that pos < leaf.len
				return Some(unsafe { guard.as_leaf_mut().kv_at_mut_unchecked(pos) });
			} else {
				// At end of current leaf - try to move to next leaf
				match self.next_leaf() {
					LeafResult::Ok | LeafResult::Retry => continue,
					LeafResult::End => return None,
				}
			}
		}
	}

	/// Returns the previous entry without moving the cursor backward (mutable value reference).
	///
	/// Repeated calls return the same entry until `next()` or `prev()` is called.
	/// If at a leaf boundary, may move to the previous leaf (but not the cursor
	/// position within that leaf).
	#[inline]
	pub fn peek_prev(&mut self) -> Option<(&K, &mut V)> {
		loop {
			// Get position to return (or None if at start of leaf)
			let pos_opt = match self.leaf.as_ref() {
				Some((_, cursor)) => cursor.peek_prev_pos(),
				None => return None,
			};

			if let Some(pos) = pos_opt {
				// Return entry at position (new mutable borrow)
				let (guard, _) = self.leaf.as_mut().unwrap();
				// SAFETY: peek_prev_pos() already validated that pos is valid
				return Some(unsafe { guard.as_leaf_mut().kv_at_mut_unchecked(pos) });
			} else {
				// At start of current leaf - try to move to previous leaf
				match self.prev_leaf() {
					LeafResult::Ok | LeafResult::Retry => continue,
					LeafResult::End => return None,
				}
			}
		}
	}

	/// Process all remaining entries in the current leaf with a closure.
	///
	/// Similar to `RawSharedIter::for_each_in_leaf` but provides mutable
	/// access to values.
	///
	/// This method iterates through all entries from the current cursor position
	/// to the end of the current leaf, calling the provided closure for each entry.
	/// After exhausting the leaf, the cursor is positioned at the end.
	///
	/// Returns `true` if more leaves may exist (for continued iteration),
	/// `false` if this was the last leaf.
	///
	/// This is more efficient than repeated `next()` calls when processing
	/// many sequential entries, as it avoids per-entry cursor bookkeeping.
	#[inline]
	pub fn for_each_in_leaf<F>(&mut self, mut f: F) -> bool
	where
		F: FnMut(&K, &mut V),
	{
		let Some((guard, cursor)) = self.leaf.as_mut() else {
			return false;
		};

		let leaf = guard.as_leaf_mut();
		let leaf_len = leaf.len;

		// Determine starting position based on cursor state
		let start = match *cursor {
			Cursor::Before(pos) => pos,
			Cursor::After(pos) => pos.saturating_add(1),
		};

		// Process all remaining entries in this leaf
		// SAFETY: i is bounded by start..leaf_len where:
		// - start comes from cursor state which tracks valid positions
		// - leaf_len is the actual length of the leaf
		// - the loop only iterates while i < leaf_len
		for i in start..leaf_len {
			let (k, v) = unsafe { leaf.kv_at_mut_unchecked(i) };
			f(k, v);
		}

		// Mark cursor as exhausted (positioned at end of leaf)
		*cursor = Cursor::Before(leaf_len);

		// Return whether more leaves may exist
		leaf.upper_fence.is_some()
	}
}

// ===========================================================================
// High-Level Iterator Wrappers
// ===========================================================================

/// A range iterator over the entries of a B+ tree.
///
/// This iterator yields key-value pairs within the specified bounds.
/// It wraps `RawSharedIter` and provides bounds checking.
///
/// # Usage
///
/// ```
/// use ferntree::Tree;
/// use std::ops::Bound::{Included, Excluded, Unbounded};
///
/// let tree: Tree<i32, &str> = Tree::new();
/// tree.insert(1, "one");
/// tree.insert(2, "two");
/// tree.insert(3, "three");
///
/// let mut range = tree.range(Included(&2), Unbounded);
/// assert_eq!(range.next(), Some((&2, &"two")));
/// assert_eq!(range.next(), Some((&3, &"three")));
/// assert_eq!(range.next(), None);
/// ```
pub struct Range<'t, K, V, const IC: usize, const LC: usize> {
	/// The underlying iterator.
	iter: RawSharedIter<'t, K, V, IC, LC>,
	/// The upper bound for iteration (owned).
	upper_bound: Bound<K>,
	/// Whether we've finished iterating.
	finished: bool,
}

impl<'t, K: Clone + Ord, V, const IC: usize, const LC: usize> Range<'t, K, V, IC, LC> {
	/// Creates a new range iterator.
	///
	/// The iterator is positioned based on the lower bound and will stop
	/// when entries exceed the upper bound.
	pub(crate) fn new<Q>(
		tree: &'t GenericTree<K, V, IC, LC>,
		min: Bound<&Q>,
		max: Bound<&Q>,
	) -> Range<'t, K, V, IC, LC>
	where
		K: Borrow<Q>,
		Q: ?Sized + Ord,
	{
		let mut iter = tree.raw_iter();

		// Convert upper bound to owned by seeking and cloning the boundary key.
		// We do this first, then reposition for the lower bound.
		let upper_bound = match max {
			Bound::Unbounded => Bound::Unbounded,
			Bound::Included(k) => {
				iter.seek(k);
				match iter.next() {
					Some((key, _)) if key.borrow() == k => Bound::Included(key.clone()),
					Some((key, _)) => {
						// k doesn't exist, use key > k as exclusive bound
						Bound::Excluded(key.clone())
					}
					None => Bound::Unbounded, // No keys at or after k
				}
			}
			Bound::Excluded(k) => {
				iter.seek(k);
				match iter.next() {
					Some((key, _)) if key.borrow() == k => Bound::Excluded(key.clone()),
					Some((key, _)) => {
						// k doesn't exist, use key > k as exclusive bound
						Bound::Excluded(key.clone())
					}
					None => Bound::Unbounded, // No keys at or after k
				}
			}
		};

		// Position the iterator based on the lower bound
		match min {
			Bound::Unbounded => iter.seek_to_first(),
			Bound::Included(k) => iter.seek(k),
			Bound::Excluded(k) => {
				// Seek to the key, then skip it if it exists
				if iter.seek_exact(k) {
					let _ = iter.next(); // Skip the excluded key
				}
			}
		}

		Range {
			iter,
			upper_bound,
			finished: false,
		}
	}

	/// Returns the next key-value pair within the range.
	///
	/// Returns `None` when iteration is complete or when entries
	/// exceed the upper bound.
	#[allow(clippy::should_implement_trait)]
	pub fn next(&mut self) -> Option<(&K, &V)> {
		if self.finished {
			return None;
		}

		match self.iter.next() {
			Some((k, v)) => {
				// Check if we've exceeded the upper bound
				let within_bounds = match &self.upper_bound {
					Bound::Unbounded => true,
					Bound::Included(max) => k <= max,
					Bound::Excluded(max) => k < max,
				};

				if within_bounds {
					Some((k, v))
				} else {
					self.finished = true;
					None
				}
			}
			None => {
				self.finished = true;
				None
			}
		}
	}

	/// Returns the previous key-value pair within the range.
	///
	/// Note: This does not check the lower bound - it simply iterates backwards.
	pub fn prev(&mut self) -> Option<(&K, &V)> {
		self.iter.prev()
	}

	/// Returns the next key-value pair within the range without advancing.
	///
	/// Repeated calls return the same entry until `next()` or `prev()` is called.
	/// Returns `None` when the next entry would exceed the upper bound.
	///
	/// # Usage
	///
	/// ```
	/// use ferntree::Tree;
	/// use std::ops::Bound::{Included, Unbounded};
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// tree.insert(1, "one");
	/// tree.insert(2, "two");
	///
	/// let mut range = tree.range(Included(&1), Unbounded);
	///
	/// // peek() returns the same value repeatedly
	/// assert_eq!(range.peek(), Some((&1, &"one")));
	/// assert_eq!(range.peek(), Some((&1, &"one")));
	///
	/// // next() returns the same value and advances
	/// assert_eq!(range.next(), Some((&1, &"one")));
	/// assert_eq!(range.peek(), Some((&2, &"two")));
	/// ```
	pub fn peek(&mut self) -> Option<(&K, &V)> {
		if self.finished {
			return None;
		}

		match self.iter.peek() {
			Some((k, v)) => {
				// Check if we've exceeded the upper bound
				let within_bounds = match &self.upper_bound {
					Bound::Unbounded => true,
					Bound::Included(max) => k <= max,
					Bound::Excluded(max) => k < max,
				};

				if within_bounds {
					Some((k, v))
				} else {
					None
				}
			}
			None => None,
		}
	}

	/// Returns the previous key-value pair within the range without moving backward.
	///
	/// Repeated calls return the same entry until `next()` or `prev()` is called.
	/// Note: This does not check the lower bound.
	pub fn peek_prev(&mut self) -> Option<(&K, &V)> {
		self.iter.peek_prev()
	}
}

/// A reverse range iterator over the entries of a B+ tree.
///
/// This iterator yields key-value pairs within the specified bounds in
/// descending key order. It wraps `RawSharedIter` and provides bounds checking.
///
/// # Usage
///
/// ```
/// use ferntree::Tree;
/// use std::ops::Bound::{Included, Excluded, Unbounded};
///
/// let tree: Tree<i32, &str> = Tree::new();
/// tree.insert(1, "one");
/// tree.insert(2, "two");
/// tree.insert(3, "three");
///
/// let mut range = tree.range_rev(Included(&1), Unbounded);
/// assert_eq!(range.next(), Some((&3, &"three")));
/// assert_eq!(range.next(), Some((&2, &"two")));
/// assert_eq!(range.next(), Some((&1, &"one")));
/// assert_eq!(range.next(), None);
/// ```
pub struct RangeRev<'t, K, V, const IC: usize, const LC: usize> {
	/// The underlying iterator.
	iter: RawSharedIter<'t, K, V, IC, LC>,
	/// The lower bound for iteration (owned).
	lower_bound: Bound<K>,
	/// Whether we've finished iterating.
	finished: bool,
}

impl<'t, K: Clone + Ord, V, const IC: usize, const LC: usize> RangeRev<'t, K, V, IC, LC> {
	/// Creates a new reverse range iterator.
	///
	/// The iterator is positioned based on the upper bound and will stop
	/// when entries go below the lower bound.
	pub(crate) fn new<Q>(
		tree: &'t GenericTree<K, V, IC, LC>,
		min: Bound<&Q>,
		max: Bound<&Q>,
	) -> RangeRev<'t, K, V, IC, LC>
	where
		K: Borrow<Q>,
		Q: ?Sized + Ord,
	{
		let mut iter = tree.raw_iter();

		// Convert lower bound to owned by seeking and cloning the boundary key.
		// We do this first, then reposition for the upper bound.
		let lower_bound = match min {
			Bound::Unbounded => Bound::Unbounded,
			Bound::Included(k) => {
				iter.seek(k);
				match iter.next() {
					Some((key, _)) if key.borrow() == k => Bound::Included(key.clone()),
					Some((key, _)) => {
						// k doesn't exist, use key > k as exclusive bound
						// For lower bound, we want to include keys >= the next existing key
						Bound::Included(key.clone())
					}
					None => Bound::Unbounded, // No keys at or after k
				}
			}
			Bound::Excluded(k) => {
				iter.seek(k);
				match iter.next() {
					Some((key, _)) if key.borrow() == k => Bound::Excluded(key.clone()),
					Some((key, _)) => {
						// k doesn't exist, use key > k as the bound (include it)
						Bound::Included(key.clone())
					}
					None => Bound::Unbounded, // No keys at or after k
				}
			}
		};

		// Position the iterator based on the upper bound
		match max {
			Bound::Unbounded => iter.seek_to_last(),
			Bound::Included(k) => {
				// Seek for prev positions so that prev() returns the largest key <= k
				iter.seek_for_prev(k);
			}
			Bound::Excluded(k) => {
				// Seek for prev, then we need to skip if we landed exactly on k
				iter.seek_for_prev(k);
				// Check if we're positioned at the excluded key and skip it
				if let Some((key, _)) = iter.peek_prev() {
					if key.borrow() == k {
						let _ = iter.prev(); // Skip the excluded key
					}
				}
			}
		}

		RangeRev {
			iter,
			lower_bound,
			finished: false,
		}
	}

	/// Returns the next key-value pair within the range (in descending order).
	///
	/// Returns `None` when iteration is complete or when entries
	/// go below the lower bound.
	#[allow(clippy::should_implement_trait)]
	pub fn next(&mut self) -> Option<(&K, &V)> {
		if self.finished {
			return None;
		}

		match self.iter.prev() {
			Some((k, v)) => {
				// Check if we've gone below the lower bound
				let within_bounds = match &self.lower_bound {
					Bound::Unbounded => true,
					Bound::Included(min) => k >= min,
					Bound::Excluded(min) => k > min,
				};

				if within_bounds {
					Some((k, v))
				} else {
					self.finished = true;
					None
				}
			}
			None => {
				self.finished = true;
				None
			}
		}
	}

	/// Returns the previous key-value pair within the range (going forward).
	///
	/// Note: This does not check the upper bound.
	pub fn prev(&mut self) -> Option<(&K, &V)> {
		self.iter.next()
	}

	/// Returns the next key-value pair within the range without advancing.
	///
	/// Repeated calls return the same entry until `next()` or `prev()` is called.
	/// Returns `None` when the next entry would go below the lower bound.
	pub fn peek(&mut self) -> Option<(&K, &V)> {
		if self.finished {
			return None;
		}

		match self.iter.peek_prev() {
			Some((k, v)) => {
				// Check if we've gone below the lower bound
				let within_bounds = match &self.lower_bound {
					Bound::Unbounded => true,
					Bound::Included(min) => k >= min,
					Bound::Excluded(min) => k > min,
				};

				if within_bounds {
					Some((k, v))
				} else {
					None
				}
			}
			None => None,
		}
	}

	/// Returns the previous key-value pair within the range without moving backward.
	///
	/// Repeated calls return the same entry until `next()` or `prev()` is called.
	/// Note: This does not check the upper bound.
	pub fn peek_prev(&mut self) -> Option<(&K, &V)> {
		self.iter.peek()
	}
}

/// An iterator over the keys of a B+ tree.
///
/// This iterator yields references to keys in ascending order.
///
/// # Usage
///
/// ```
/// use ferntree::Tree;
///
/// let tree: Tree<i32, &str> = Tree::new();
/// tree.insert(3, "three");
/// tree.insert(1, "one");
/// tree.insert(2, "two");
///
/// let mut keys = tree.keys();
/// assert_eq!(keys.next(), Some(&1));
/// assert_eq!(keys.next(), Some(&2));
/// assert_eq!(keys.next(), Some(&3));
/// assert_eq!(keys.next(), None);
/// ```
pub struct Keys<'t, K, V, const IC: usize, const LC: usize> {
	/// The underlying iterator.
	iter: RawSharedIter<'t, K, V, IC, LC>,
}

impl<'t, K: Clone + Ord, V, const IC: usize, const LC: usize> Keys<'t, K, V, IC, LC> {
	/// Creates a new keys iterator positioned at the first key.
	pub(crate) fn new(tree: &'t GenericTree<K, V, IC, LC>) -> Keys<'t, K, V, IC, LC> {
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		Keys {
			iter,
		}
	}

	/// Returns the next key.
	#[allow(clippy::should_implement_trait)]
	pub fn next(&mut self) -> Option<&K> {
		self.iter.next().map(|(k, _)| k)
	}

	/// Returns the previous key.
	pub fn prev(&mut self) -> Option<&K> {
		self.iter.prev().map(|(k, _)| k)
	}
}

/// An iterator over the values of a B+ tree.
///
/// This iterator yields references to values in key-ascending order.
///
/// # Usage
///
/// ```
/// use ferntree::Tree;
///
/// let tree: Tree<i32, &str> = Tree::new();
/// tree.insert(3, "three");
/// tree.insert(1, "one");
/// tree.insert(2, "two");
///
/// let mut values = tree.values();
/// assert_eq!(values.next(), Some(&"one"));
/// assert_eq!(values.next(), Some(&"two"));
/// assert_eq!(values.next(), Some(&"three"));
/// assert_eq!(values.next(), None);
/// ```
pub struct Values<'t, K, V, const IC: usize, const LC: usize> {
	/// The underlying iterator.
	iter: RawSharedIter<'t, K, V, IC, LC>,
}

impl<'t, K: Clone + Ord, V, const IC: usize, const LC: usize> Values<'t, K, V, IC, LC> {
	/// Creates a new values iterator positioned at the first value.
	pub(crate) fn new(tree: &'t GenericTree<K, V, IC, LC>) -> Values<'t, K, V, IC, LC> {
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		Values {
			iter,
		}
	}

	/// Returns the next value.
	#[allow(clippy::should_implement_trait)]
	pub fn next(&mut self) -> Option<&V> {
		self.iter.next().map(|(_, v)| v)
	}

	/// Returns the previous value.
	pub fn prev(&mut self) -> Option<&V> {
		self.iter.prev().map(|(_, v)| v)
	}
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
	use crate::Tree;

	// -----------------------------------------------------------------------
	// Empty Tree Tests
	// -----------------------------------------------------------------------

	#[test]
	fn iter_empty_tree_next_returns_none() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		assert!(iter.next().is_none());
	}

	#[test]
	fn iter_empty_tree_prev_returns_none() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut iter = tree.raw_iter();
		iter.seek_to_last();
		assert!(iter.prev().is_none());
	}

	#[test]
	fn iter_mut_empty_tree_next_returns_none() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut iter = tree.raw_iter_mut();
		iter.seek_to_first();
		assert!(iter.next().is_none());
	}

	#[test]
	fn iter_mut_empty_tree_prev_returns_none() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut iter = tree.raw_iter_mut();
		iter.seek_to_last();
		assert!(iter.prev().is_none());
	}

	// -----------------------------------------------------------------------
	// Single Element Tests
	// -----------------------------------------------------------------------

	#[test]
	fn iter_single_element_forward() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(42, 100);

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let (k, v) = iter.next().unwrap();
		assert_eq!(*k, 42);
		assert_eq!(*v, 100);

		assert!(iter.next().is_none());
	}

	#[test]
	fn iter_single_element_backward() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(42, 100);

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		let (k, v) = iter.prev().unwrap();
		assert_eq!(*k, 42);
		assert_eq!(*v, 100);

		assert!(iter.prev().is_none());
	}

	// -----------------------------------------------------------------------
	// Seek Variants Tests
	// -----------------------------------------------------------------------

	#[test]
	fn seek_to_existing_key() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i * 10, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek(&50);

		let (k, v) = iter.next().unwrap();
		assert_eq!(*k, 50);
		assert_eq!(*v, 5);
	}

	#[test]
	fn seek_to_nonexisting_key() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i * 10, i);
		}

		let mut iter = tree.raw_iter();
		// Seek to 55, which doesn't exist. Should position before 60.
		iter.seek(&55);

		let (k, v) = iter.next().unwrap();
		assert_eq!(*k, 60);
		assert_eq!(*v, 6);
	}

	#[test]
	fn seek_exact_returns_true_for_existing() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i * 10, i);
		}

		let mut iter = tree.raw_iter();
		let found = iter.seek_exact(&50);
		assert!(found);

		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 50);
	}

	#[test]
	fn seek_exact_returns_false_for_nonexisting() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i * 10, i);
		}

		let mut iter = tree.raw_iter();
		let found = iter.seek_exact(&55);
		assert!(!found);

		// Should still be positioned at 60 (next key)
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 60);
	}

	#[test]
	fn seek_for_prev_existing_key() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i * 10, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek_for_prev(&50);

		// prev() should return the key we sought
		let (k, v) = iter.prev().unwrap();
		assert_eq!(*k, 50);
		assert_eq!(*v, 5);
	}

	#[test]
	fn seek_for_prev_nonexisting_key() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i * 10, i);
		}

		let mut iter = tree.raw_iter();
		// Seek for prev at 55 (doesn't exist)
		// Keys are: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
		// seek_for_prev positions so prev() returns the largest key <= 55
		iter.seek_for_prev(&55);

		// prev() should return 50 (largest key <= 55)
		let (k, _) = iter.prev().unwrap();
		assert_eq!(*k, 50);

		// And prev() again should give us 40
		let (k, _) = iter.prev().unwrap();
		assert_eq!(*k, 40);
	}

	#[test]
	fn seek_to_first_positions_at_beginning() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in (0..10).rev() {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let (k, v) = iter.next().unwrap();
		assert_eq!(*k, 0);
		assert_eq!(*v, 0);
	}

	#[test]
	fn seek_to_last_positions_at_end() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		// next() should return None (we're past the last element)
		assert!(iter.next().is_none());

		// prev() should return the last element
		let (k, v) = iter.prev().unwrap();
		assert_eq!(*k, 9);
		assert_eq!(*v, 90);
	}

	// -----------------------------------------------------------------------
	// Bidirectional Iteration Tests
	// -----------------------------------------------------------------------

	#[test]
	fn mixed_next_and_prev() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek(&5);

		// next() returns 5
		assert_eq!(*iter.next().unwrap().0, 5);
		// next() returns 6
		assert_eq!(*iter.next().unwrap().0, 6);
		// prev() returns 6 (we just visited it)
		assert_eq!(*iter.prev().unwrap().0, 6);
		// prev() returns 5
		assert_eq!(*iter.prev().unwrap().0, 5);
		// prev() returns 4
		assert_eq!(*iter.prev().unwrap().0, 4);
		// next() returns 4
		assert_eq!(*iter.next().unwrap().0, 4);
	}

	#[test]
	fn reverse_from_middle() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek(&5);

		// Go forward once
		iter.next();

		// Now go backwards to the beginning
		let mut collected: Vec<i32> = Vec::new();
		while let Some((k, _)) = iter.prev() {
			collected.push(*k);
		}

		assert_eq!(collected, vec![5, 4, 3, 2, 1, 0]);
	}

	#[test]
	fn forward_then_full_reverse() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..5 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		// Go to the end
		while iter.next().is_some() {}

		// Now reverse
		let mut collected: Vec<i32> = Vec::new();
		while let Some((k, _)) = iter.prev() {
			collected.push(*k);
		}

		assert_eq!(collected, vec![4, 3, 2, 1, 0]);
	}

	// -----------------------------------------------------------------------
	// Full Forward and Reverse Iteration Tests
	// -----------------------------------------------------------------------

	#[test]
	fn full_forward_iteration() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..100 {
			tree.insert(i, i * 2);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let mut count = 0;
		while let Some((k, v)) = iter.next() {
			assert_eq!(*k, count);
			assert_eq!(*v, count * 2);
			count += 1;
		}
		assert_eq!(count, 100);
	}

	#[test]
	fn full_reverse_iteration() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..100 {
			tree.insert(i, i * 2);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		let mut count = 99i32;
		while let Some((k, v)) = iter.prev() {
			assert_eq!(*k, count);
			assert_eq!(*v, count * 2);
			count -= 1;
		}
		assert_eq!(count, -1);
	}

	// -----------------------------------------------------------------------
	// Cross-Leaf Navigation Tests (need enough keys to cause splits)
	// -----------------------------------------------------------------------

	#[test]
	fn iteration_crosses_leaf_boundaries_forward() {
		let tree: Tree<i32, i32> = Tree::new();
		// Insert enough to cause splits (LEAF_CAPACITY is 64)
		for i in 0..200 {
			tree.insert(i, i);
		}
		assert!(tree.height() > 1, "Tree should have multiple levels");

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let mut count = 0;
		while let Some((k, _)) = iter.next() {
			assert_eq!(*k, count);
			count += 1;
		}
		assert_eq!(count, 200);
	}

	#[test]
	fn iteration_crosses_leaf_boundaries_reverse() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..200 {
			tree.insert(i, i);
		}
		assert!(tree.height() > 1);

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		let mut count = 199i32;
		while let Some((k, _)) = iter.prev() {
			assert_eq!(*k, count);
			count -= 1;
		}
		assert_eq!(count, -1);
	}

	// -----------------------------------------------------------------------
	// Batch Iteration Tests
	// -----------------------------------------------------------------------

	#[test]
	fn for_each_in_leaf_processes_all_entries() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let mut collected = Vec::new();
		let has_more = iter.for_each_in_leaf(|k, v| {
			collected.push((*k, *v));
		});

		// Should have collected all entries (single leaf for small tree)
		assert_eq!(collected.len(), 10);
		assert_eq!(collected[0], (0, 0));
		assert_eq!(collected[9], (9, 90));
		// Small tree has single leaf, so no more leaves
		assert!(!has_more);
	}

	#[test]
	fn for_each_in_leaf_respects_cursor_position() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek(&5);

		let mut collected = Vec::new();
		iter.for_each_in_leaf(|k, v| {
			collected.push((*k, *v));
		});

		// Should start from key 5
		assert_eq!(collected[0], (5, 50));
		assert_eq!(collected.len(), 5); // 5, 6, 7, 8, 9
	}

	#[test]
	fn for_each_in_leaf_after_next() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		// Consume first 3 entries
		iter.next();
		iter.next();
		iter.next();

		let mut collected = Vec::new();
		iter.for_each_in_leaf(|k, v| {
			collected.push((*k, *v));
		});

		// Should start from key 3
		assert_eq!(collected[0], (3, 30));
		assert_eq!(collected.len(), 7); // 3, 4, 5, 6, 7, 8, 9
	}

	#[test]
	fn for_each_in_leaf_empty_after_exhausted() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..5 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		// Exhaust the leaf
		iter.for_each_in_leaf(|_, _| {});

		// Second call should process nothing
		let mut count = 0;
		iter.for_each_in_leaf(|_, _| {
			count += 1;
		});
		assert_eq!(count, 0);
	}

	#[test]
	fn for_each_in_leaf_unpositioned_iterator() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(1, 10);

		let mut iter = tree.raw_iter();
		// Don't seek - iterator is unpositioned

		let has_more = iter.for_each_in_leaf(|_, _| {
			panic!("Should not be called");
		});
		assert!(!has_more);
	}

	#[test]
	fn for_each_in_leaf_mut_can_modify_values() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		{
			let mut iter = tree.raw_iter_mut();
			iter.seek_to_first();

			iter.for_each_in_leaf(|_, v| {
				*v *= 2;
			});
		}

		// Verify modifications
		for i in 0..10 {
			let val = tree.lookup(&i, |v| *v).unwrap();
			assert_eq!(val, i * 2);
		}
	}

	#[test]
	fn for_each_in_leaf_crosses_multiple_leaves() {
		let tree: Tree<i32, i32> = Tree::new();
		// Insert enough to cause splits (LEAF_CAPACITY is 64)
		for i in 0..200 {
			tree.insert(i, i);
		}
		assert!(tree.height() > 1, "Tree should have multiple levels");

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let mut total_count = 0;
		let mut leaf_count = 0;

		loop {
			let mut leaf_entries = 0;
			let has_more = iter.for_each_in_leaf(|_, _| {
				leaf_entries += 1;
			});
			total_count += leaf_entries;
			leaf_count += 1;

			if !has_more {
				break;
			}
			// Move to next leaf - next() also returns the first entry of the new leaf
			if iter.next().is_some() {
				total_count += 1; // Count the entry returned by next()
			} else {
				break;
			}
		}

		assert_eq!(total_count, 200);
		assert!(leaf_count > 1, "Should have visited multiple leaves");
	}

	// -----------------------------------------------------------------------
	// Exclusive Iterator Tests
	// -----------------------------------------------------------------------

	#[test]
	fn exclusive_iter_can_modify_values() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		{
			let mut iter = tree.raw_iter_mut();
			iter.seek_to_first();

			while let Some((_, v)) = iter.next() {
				*v *= 2;
			}
		}

		// Verify modifications
		for i in 0..10 {
			assert_eq!(tree.lookup(&i, |v| *v), Some(i * 2));
		}
	}

	#[test]
	fn exclusive_iter_insert() {
		let tree: Tree<i32, i32> = Tree::new();

		{
			let mut iter = tree.raw_iter_mut();
			iter.insert(5, 50);
			iter.insert(3, 30);
			iter.insert(7, 70);
		}

		assert_eq!(tree.lookup(&3, |v| *v), Some(30));
		assert_eq!(tree.lookup(&5, |v| *v), Some(50));
		assert_eq!(tree.lookup(&7, |v| *v), Some(70));
	}

	#[test]
	fn exclusive_iter_insert_updates_existing() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(5, 50);

		{
			let mut iter = tree.raw_iter_mut();
			let old = iter.insert(5, 500);
			assert_eq!(old, Some(50));
		}

		assert_eq!(tree.lookup(&5, |v| *v), Some(500));
	}

	#[test]
	fn exclusive_iter_remove() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		{
			let mut iter = tree.raw_iter_mut();
			let removed = iter.remove(&5);
			assert_eq!(removed, Some((5, 5)));
		}

		assert_eq!(tree.lookup(&5, |v| *v), None);
		assert_eq!(tree.len(), 9);
	}

	#[test]
	fn exclusive_iter_remove_nonexistent() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		{
			let mut iter = tree.raw_iter_mut();
			let removed = iter.remove(&100);
			assert!(removed.is_none());
		}

		assert_eq!(tree.len(), 10);
	}

	// -----------------------------------------------------------------------
	// Seek Boundary Conditions
	// -----------------------------------------------------------------------

	#[test]
	fn seek_before_first_key() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 10..20 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek(&5); // Before any key

		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 10); // Should get first key
	}

	#[test]
	fn seek_after_last_key() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 10..20 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek(&100); // After any key

		assert!(iter.next().is_none());
	}

	#[test]
	fn seek_for_prev_before_first_key() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 10..20 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek_for_prev(&5); // Before any key

		// prev should return None (nothing before)
		// but we should be positioned at the start
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 10);
	}

	// -----------------------------------------------------------------------
	// String Key Tests (to match fixture type)
	// -----------------------------------------------------------------------

	#[test]
	fn iter_with_string_keys() {
		let tree: Tree<String, i32> = Tree::new();
		tree.insert("apple".to_string(), 1);
		tree.insert("banana".to_string(), 2);
		tree.insert("cherry".to_string(), 3);

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let (k, _) = iter.next().unwrap();
		assert_eq!(k, "apple");

		let (k, _) = iter.next().unwrap();
		assert_eq!(k, "banana");

		let (k, _) = iter.next().unwrap();
		assert_eq!(k, "cherry");

		assert!(iter.next().is_none());
	}

	#[test]
	fn seek_with_string_borrow() {
		let tree: Tree<String, i32> = Tree::new();
		tree.insert("apple".to_string(), 1);
		tree.insert("banana".to_string(), 2);
		tree.insert("cherry".to_string(), 3);

		let mut iter = tree.raw_iter();
		// Seek using &str instead of String
		iter.seek("banana");

		let (k, v) = iter.next().unwrap();
		assert_eq!(k, "banana");
		assert_eq!(*v, 2);
	}

	// -----------------------------------------------------------------------
	// Reuse Leaf Optimization Tests
	// -----------------------------------------------------------------------

	#[test]
	fn seek_reuses_current_leaf_when_within_bounds() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek(&5);

		// Seek to a nearby key - should reuse current leaf
		iter.seek(&7);

		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 7);
	}

	// -----------------------------------------------------------------------
	// Cursor State Transition Unit Tests
	// -----------------------------------------------------------------------
	//
	// These tests verify the Cursor::prev_entry() and Cursor::next_entry()
	// helper methods in isolation, ensuring correct state transitions.

	mod cursor_tests {
		use super::super::Cursor;

		// prev_entry() tests

		#[test]
		fn cursor_prev_from_after_middle() {
			// After(2) should return entry 2 and transition to After(1)
			let cursor = Cursor::After(2);
			let (pos, new) = cursor.prev_entry().unwrap();
			assert_eq!(pos, 2);
			assert_eq!(new, Cursor::After(1));
		}

		#[test]
		fn cursor_prev_from_after_one() {
			// After(1) should return entry 1 and transition to After(0)
			let cursor = Cursor::After(1);
			let (pos, new) = cursor.prev_entry().unwrap();
			assert_eq!(pos, 1);
			assert_eq!(new, Cursor::After(0));
		}

		#[test]
		fn cursor_prev_from_after_zero() {
			// After(0) should return entry 0 and transition to Before(0)
			// (can't go further back in this leaf)
			let cursor = Cursor::After(0);
			let (pos, new) = cursor.prev_entry().unwrap();
			assert_eq!(pos, 0);
			assert_eq!(new, Cursor::Before(0));
		}

		#[test]
		fn cursor_prev_from_before_zero() {
			// Before(0) means at the start - nothing to return
			let cursor = Cursor::Before(0);
			assert!(cursor.prev_entry().is_none());
		}

		#[test]
		fn cursor_prev_from_before_one() {
			// Before(1) should return entry 0 and transition to Before(0)
			let cursor = Cursor::Before(1);
			let (pos, new) = cursor.prev_entry().unwrap();
			assert_eq!(pos, 0);
			assert_eq!(new, Cursor::Before(0));
		}

		#[test]
		fn cursor_prev_from_before_two() {
			// Before(2) should return entry 1 and transition to After(0)
			let cursor = Cursor::Before(2);
			let (pos, new) = cursor.prev_entry().unwrap();
			assert_eq!(pos, 1);
			assert_eq!(new, Cursor::After(0));
		}

		#[test]
		fn cursor_prev_from_before_middle() {
			// Before(5) should return entry 4 and transition to After(3)
			let cursor = Cursor::Before(5);
			let (pos, new) = cursor.prev_entry().unwrap();
			assert_eq!(pos, 4);
			assert_eq!(new, Cursor::After(3));
		}

		// next_entry() tests

		#[test]
		fn cursor_next_from_before_start() {
			// Before(0) with len=4 should return entry 0 and transition to Before(1)
			let cursor = Cursor::Before(0);
			let (pos, new) = cursor.next_entry(4).unwrap();
			assert_eq!(pos, 0);
			assert_eq!(new, Cursor::Before(1));
		}

		#[test]
		fn cursor_next_from_before_middle() {
			// Before(2) with len=4 should return entry 2 and transition to Before(3)
			let cursor = Cursor::Before(2);
			let (pos, new) = cursor.next_entry(4).unwrap();
			assert_eq!(pos, 2);
			assert_eq!(new, Cursor::Before(3));
		}

		#[test]
		fn cursor_next_from_before_last() {
			// Before(3) with len=4 should return entry 3 and transition to Before(4)
			let cursor = Cursor::Before(3);
			let (pos, new) = cursor.next_entry(4).unwrap();
			assert_eq!(pos, 3);
			assert_eq!(new, Cursor::Before(4));
		}

		#[test]
		fn cursor_next_from_before_past_end() {
			// Before(4) with len=4 means past end - nothing to return
			let cursor = Cursor::Before(4);
			assert!(cursor.next_entry(4).is_none());
		}

		#[test]
		fn cursor_next_from_after_start() {
			// After(0) with len=4 should return entry 1 and transition to Before(2)
			let cursor = Cursor::After(0);
			let (pos, new) = cursor.next_entry(4).unwrap();
			assert_eq!(pos, 1);
			assert_eq!(new, Cursor::Before(2));
		}

		#[test]
		fn cursor_next_from_after_middle() {
			// After(1) with len=4 should return entry 2 and transition to Before(3)
			let cursor = Cursor::After(1);
			let (pos, new) = cursor.next_entry(4).unwrap();
			assert_eq!(pos, 2);
			assert_eq!(new, Cursor::Before(3));
		}

		#[test]
		fn cursor_next_from_after_second_to_last() {
			// After(2) with len=4 should return entry 3 and transition to Before(4)
			let cursor = Cursor::After(2);
			let (pos, new) = cursor.next_entry(4).unwrap();
			assert_eq!(pos, 3);
			assert_eq!(new, Cursor::Before(4));
		}

		#[test]
		fn cursor_next_from_after_last() {
			// After(3) with len=4 means past end - nothing to return
			let cursor = Cursor::After(3);
			assert!(cursor.next_entry(4).is_none());
		}

		// Bidirectional consistency tests

		#[test]
		fn cursor_next_then_prev_returns_same() {
			// Starting at Before(0), next() returns 0 -> Before(1)
			// Then prev() should return 0 -> Before(0)
			let cursor = Cursor::Before(0);
			let (pos1, cursor) = cursor.next_entry(4).unwrap();
			assert_eq!(pos1, 0);
			assert_eq!(cursor, Cursor::Before(1));

			let (pos2, cursor) = cursor.prev_entry().unwrap();
			assert_eq!(pos2, 0);
			assert_eq!(cursor, Cursor::Before(0));
		}

		#[test]
		fn cursor_prev_then_next_returns_same() {
			// Starting at After(2), prev() returns 2 -> After(1)
			// Then next() returns 2 -> Before(3)
			let cursor = Cursor::After(2);
			let (pos1, cursor) = cursor.prev_entry().unwrap();
			assert_eq!(pos1, 2);
			assert_eq!(cursor, Cursor::After(1));

			let (pos2, cursor) = cursor.next_entry(4).unwrap();
			assert_eq!(pos2, 2);
			assert_eq!(cursor, Cursor::Before(3));
		}

		#[test]
		fn cursor_alternating_next_prev() {
			// Verify alternating between next/prev behaves correctly
			// Start at Before(2), next() -> 2, Before(3)
			let cursor = Cursor::Before(2);
			let (pos, cursor) = cursor.next_entry(5).unwrap();
			assert_eq!(pos, 2);
			assert_eq!(cursor, Cursor::Before(3));

			// prev() -> 2, After(1)
			let (pos, cursor) = cursor.prev_entry().unwrap();
			assert_eq!(pos, 2);
			assert_eq!(cursor, Cursor::After(1));

			// next() -> 2, Before(3)
			let (pos, cursor) = cursor.next_entry(5).unwrap();
			assert_eq!(pos, 2);
			assert_eq!(cursor, Cursor::Before(3));

			// next() -> 3, Before(4)
			let (pos, cursor) = cursor.next_entry(5).unwrap();
			assert_eq!(pos, 3);
			assert_eq!(cursor, Cursor::Before(4));

			// prev() -> 3, After(2)
			let (pos, cursor) = cursor.prev_entry().unwrap();
			assert_eq!(pos, 3);
			assert_eq!(cursor, Cursor::After(2));
		}
	}

	// -----------------------------------------------------------------------
	// Integration Tests for Bidirectional Iteration
	// -----------------------------------------------------------------------

	#[test]
	fn bidirectional_next_prev_consistency() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek(&5);

		// next() returns 5
		let (k1, _) = iter.next().unwrap();
		assert_eq!(*k1, 5);

		// prev() should return 5 (we just visited it)
		let (k2, _) = iter.prev().unwrap();
		assert_eq!(*k2, 5);

		// next() should return 5 again
		let (k3, _) = iter.next().unwrap();
		assert_eq!(*k3, 5);

		// next() should return 6
		let (k4, _) = iter.next().unwrap();
		assert_eq!(*k4, 6);
	}

	#[test]
	fn bidirectional_from_start() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..5 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		// Get first entry
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 0);

		// prev() returns 0 (just visited)
		let (k, _) = iter.prev().unwrap();
		assert_eq!(*k, 0);

		// prev() should return None (at start)
		assert!(iter.prev().is_none());

		// next() should return 0 again
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 0);
	}

	#[test]
	fn bidirectional_from_end() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..5 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		// prev() returns 4 (last entry)
		let (k, _) = iter.prev().unwrap();
		assert_eq!(*k, 4);

		// next() returns 4 (just visited)
		let (k, _) = iter.next().unwrap();
		assert_eq!(*k, 4);

		// next() should return None (at end)
		assert!(iter.next().is_none());

		// prev() should return 4 again
		let (k, _) = iter.prev().unwrap();
		assert_eq!(*k, 4);
	}

	// -----------------------------------------------------------------------
	// seek_for_prev Edge Case Tests
	// -----------------------------------------------------------------------

	#[test]
	fn seek_for_prev_empty_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut iter = tree.raw_iter();
		iter.seek_for_prev(&100);
		assert!(iter.prev().is_none());
	}

	#[test]
	fn seek_for_prev_before_all_keys() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 10..20 {
			tree.insert(i, i);
		}
		let mut iter = tree.raw_iter();
		iter.seek_for_prev(&5); // Before any key
		assert!(iter.prev().is_none());
	}

	#[test]
	fn seek_for_prev_after_all_keys() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 10..20 {
			tree.insert(i, i);
		}
		let mut iter = tree.raw_iter();
		iter.seek_for_prev(&100); // After all keys
		let (k, _) = iter.prev().unwrap();
		assert_eq!(*k, 19); // Largest key
	}

	#[test]
	fn seek_for_prev_middle_gap() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(10, 1);
		tree.insert(20, 2);
		tree.insert(40, 4);
		tree.insert(50, 5);
		// Gap at 30
		let mut iter = tree.raw_iter();
		iter.seek_for_prev(&35); // In gap between 20 and 40
		let (k, _) = iter.prev().unwrap();
		assert_eq!(*k, 20); // Largest key <= 35
	}

	#[test]
	fn seek_for_prev_mut_empty_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut iter = tree.raw_iter_mut();
		iter.seek_for_prev(&100);
		assert!(iter.prev().is_none());
	}

	#[test]
	fn seek_for_prev_mut_before_all_keys() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 10..20 {
			tree.insert(i, i);
		}
		let mut iter = tree.raw_iter_mut();
		iter.seek_for_prev(&5); // Before any key
		assert!(iter.prev().is_none());
	}

	#[test]
	fn seek_for_prev_mut_after_all_keys() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 10..20 {
			tree.insert(i, i);
		}
		let mut iter = tree.raw_iter_mut();
		iter.seek_for_prev(&100); // After all keys
		let (k, _) = iter.prev().unwrap();
		assert_eq!(*k, 19); // Largest key
	}

	#[test]
	fn seek_for_prev_mut_middle_gap() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(10, 1);
		tree.insert(20, 2);
		tree.insert(40, 4);
		tree.insert(50, 5);
		// Gap at 30
		let mut iter = tree.raw_iter_mut();
		iter.seek_for_prev(&35); // In gap between 20 and 40
		let (k, _) = iter.prev().unwrap();
		assert_eq!(*k, 20); // Largest key <= 35
	}

	// -----------------------------------------------------------------------
	// Peek Tests
	// -----------------------------------------------------------------------

	#[test]
	fn peek_empty_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		assert!(iter.peek().is_none());
	}

	#[test]
	fn peek_returns_same_as_next() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		// peek and next should return the same entry
		// Compare by dereferencing to owned values to avoid borrow issues
		let peek_key = iter.peek().map(|(k, v)| (*k, *v));
		let next_key = iter.next().map(|(k, v)| (*k, *v));

		assert_eq!(peek_key, next_key);
		assert_eq!(peek_key.unwrap().0, 0);
	}

	#[test]
	fn peek_does_not_advance() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		// Multiple peek calls should return the same entry
		assert_eq!(iter.peek(), Some((&0, &0)));
		assert_eq!(iter.peek(), Some((&0, &0)));
		assert_eq!(iter.peek(), Some((&0, &0)));

		// next() should also return the same entry
		assert_eq!(iter.next(), Some((&0, &0)));

		// Now peek should return the next entry
		assert_eq!(iter.peek(), Some((&1, &10)));
	}

	#[test]
	fn peek_then_next_returns_same() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		for expected in 0..10 {
			let peeked = iter.peek().map(|(k, v)| (*k, *v));
			let nexted = iter.next().map(|(k, v)| (*k, *v));
			assert_eq!(peeked, nexted);
			assert_eq!(peeked.unwrap().0, expected);
		}

		// Both should return None at end
		assert!(iter.peek().is_none());
		assert!(iter.next().is_none());
	}

	#[test]
	fn peek_at_leaf_boundary() {
		let tree: Tree<i32, i32> = Tree::new();
		// Insert enough to cause splits (LEAF_CAPACITY is 64)
		for i in 0..200 {
			tree.insert(i, i);
		}
		assert!(tree.height() > 1, "Tree should have multiple levels");

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		// Iterate through all entries, checking peek at each step
		let mut count = 0;
		loop {
			let peeked = iter.peek().map(|(k, v)| (*k, *v));
			if peeked.is_none() {
				break;
			}
			let nexted = iter.next().map(|(k, v)| (*k, *v)).unwrap();
			assert_eq!(peeked.unwrap(), nexted);
			assert_eq!(nexted.0, count);
			count += 1;
		}
		assert_eq!(count, 200);
	}

	#[test]
	fn peek_from_middle() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		let mut iter = tree.raw_iter();
		iter.seek(&5);

		assert_eq!(iter.peek(), Some((&5, &5)));
		assert_eq!(iter.next(), Some((&5, &5)));
		assert_eq!(iter.peek(), Some((&6, &6)));
	}

	// -----------------------------------------------------------------------
	// Peek Prev Tests
	// -----------------------------------------------------------------------

	#[test]
	fn peek_prev_empty_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut iter = tree.raw_iter();
		iter.seek_to_last();
		assert!(iter.peek_prev().is_none());
	}

	#[test]
	fn peek_prev_returns_same_as_prev() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		// peek_prev and prev should return the same entry
		let peek_result = iter.peek_prev().map(|(k, v)| (*k, *v));
		let prev_result = iter.prev().map(|(k, v)| (*k, *v));

		assert_eq!(peek_result, prev_result);
		assert_eq!(peek_result.unwrap().0, 9);
	}

	#[test]
	fn peek_prev_does_not_advance() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		// Multiple peek_prev calls should return the same entry
		assert_eq!(iter.peek_prev(), Some((&9, &90)));
		assert_eq!(iter.peek_prev(), Some((&9, &90)));
		assert_eq!(iter.peek_prev(), Some((&9, &90)));

		// prev() should also return the same entry
		assert_eq!(iter.prev(), Some((&9, &90)));

		// Now peek_prev should return the previous entry
		assert_eq!(iter.peek_prev(), Some((&8, &80)));
	}

	#[test]
	fn peek_prev_then_prev_returns_same() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		for expected in (0..10).rev() {
			let peeked = iter.peek_prev().map(|(k, v)| (*k, *v));
			let preved = iter.prev().map(|(k, v)| (*k, *v));
			assert_eq!(peeked, preved);
			assert_eq!(peeked.unwrap().0, expected);
		}

		// Both should return None at start
		assert!(iter.peek_prev().is_none());
		assert!(iter.prev().is_none());
	}

	// -----------------------------------------------------------------------
	// Peek Mut Tests (RawExclusiveIter)
	// -----------------------------------------------------------------------

	#[test]
	fn peek_mut_does_not_advance() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut iter = tree.raw_iter_mut();
		iter.seek_to_first();

		// Multiple peek calls should return the same entry
		assert_eq!(*iter.peek().unwrap().0, 0);
		assert_eq!(*iter.peek().unwrap().0, 0);

		// next() should also return the same entry
		assert_eq!(*iter.next().unwrap().0, 0);

		// Now peek should return the next entry
		assert_eq!(*iter.peek().unwrap().0, 1);
	}

	#[test]
	fn peek_mut_allows_modification() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		{
			let mut iter = tree.raw_iter_mut();
			iter.seek_to_first();

			// Modify via peek
			if let Some((_, v)) = iter.peek() {
				*v = 100;
			}
		}

		// Verify modification
		assert_eq!(tree.lookup(&0, |v| *v), Some(100));
	}

	// -----------------------------------------------------------------------
	// Range Peek Tests
	// -----------------------------------------------------------------------

	#[test]
	fn range_peek_within_bounds() {
		use std::ops::Bound::Included;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range(Included(&3), Included(&7));

		// peek should return first entry in range
		assert_eq!(range.peek(), Some((&3, &30)));
		assert_eq!(range.peek(), Some((&3, &30)));

		// next should return same and advance
		assert_eq!(range.next(), Some((&3, &30)));
		assert_eq!(range.peek(), Some((&4, &40)));
	}

	#[test]
	fn range_peek_respects_upper_bound() {
		use std::ops::Bound::{Excluded, Included};

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range(Included(&7), Excluded(&9));

		assert_eq!(range.peek(), Some((&7, &70)));
		assert_eq!(range.next(), Some((&7, &70)));
		assert_eq!(range.peek(), Some((&8, &80)));
		assert_eq!(range.next(), Some((&8, &80)));
		// 9 is excluded
		assert!(range.peek().is_none());
		assert!(range.next().is_none());
	}

	#[test]
	fn range_peek_empty_range() {
		use std::ops::Bound::{Excluded, Included};

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		// Empty range (5 to 5 excluded)
		let mut range = tree.range(Included(&5), Excluded(&5));
		assert!(range.peek().is_none());
	}

	// -----------------------------------------------------------------------
	// Reverse Range Peek Tests
	// -----------------------------------------------------------------------

	#[test]
	fn range_rev_peek_within_bounds() {
		use std::ops::Bound::Included;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range_rev(Included(&3), Included(&7));

		// peek should return last entry in range (descending order)
		assert_eq!(range.peek(), Some((&7, &70)));
		assert_eq!(range.peek(), Some((&7, &70)));

		// next should return same and advance backwards
		assert_eq!(range.next(), Some((&7, &70)));
		assert_eq!(range.peek(), Some((&6, &60)));
	}

	#[test]
	fn range_rev_peek_respects_lower_bound() {
		use std::ops::Bound::{Excluded, Included};

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range_rev(Excluded(&1), Included(&3));

		assert_eq!(range.peek(), Some((&3, &30)));
		assert_eq!(range.next(), Some((&3, &30)));
		assert_eq!(range.peek(), Some((&2, &20)));
		assert_eq!(range.next(), Some((&2, &20)));
		// 1 is excluded
		assert!(range.peek().is_none());
		assert!(range.next().is_none());
	}

	#[test]
	fn range_rev_peek_empty_range() {
		use std::ops::Bound::{Excluded, Included};

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i);
		}

		// Empty range (5 excluded to 5 included going backwards - impossible)
		let mut range = tree.range_rev(Excluded(&5), Included(&5));
		assert!(range.peek().is_none());
	}

	// -----------------------------------------------------------------------
	// Cursor peek_next_pos and peek_prev_pos Unit Tests
	// -----------------------------------------------------------------------

	mod cursor_peek_tests {
		use super::super::Cursor;

		#[test]
		fn peek_next_pos_before_start() {
			let cursor = Cursor::Before(0);
			assert_eq!(cursor.peek_next_pos(4), Some(0));
		}

		#[test]
		fn peek_next_pos_before_middle() {
			let cursor = Cursor::Before(2);
			assert_eq!(cursor.peek_next_pos(4), Some(2));
		}

		#[test]
		fn peek_next_pos_before_end() {
			let cursor = Cursor::Before(4);
			assert_eq!(cursor.peek_next_pos(4), None);
		}

		#[test]
		fn peek_next_pos_after_start() {
			let cursor = Cursor::After(0);
			assert_eq!(cursor.peek_next_pos(4), Some(1));
		}

		#[test]
		fn peek_next_pos_after_second_to_last() {
			let cursor = Cursor::After(2);
			assert_eq!(cursor.peek_next_pos(4), Some(3));
		}

		#[test]
		fn peek_next_pos_after_last() {
			let cursor = Cursor::After(3);
			assert_eq!(cursor.peek_next_pos(4), None);
		}

		#[test]
		fn peek_prev_pos_after_start() {
			let cursor = Cursor::After(0);
			assert_eq!(cursor.peek_prev_pos(), Some(0));
		}

		#[test]
		fn peek_prev_pos_after_middle() {
			let cursor = Cursor::After(2);
			assert_eq!(cursor.peek_prev_pos(), Some(2));
		}

		#[test]
		fn peek_prev_pos_before_start() {
			let cursor = Cursor::Before(0);
			assert_eq!(cursor.peek_prev_pos(), None);
		}

		#[test]
		fn peek_prev_pos_before_middle() {
			let cursor = Cursor::Before(2);
			assert_eq!(cursor.peek_prev_pos(), Some(1));
		}
	}
}
