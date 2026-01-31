//! # Ferntree: A Concurrent In-Memory B+ Tree
//!
//! This crate provides a fast, concurrent B+ tree implementation featuring **optimistic lock
//! coupling**, a technique that enables high-throughput concurrent access with minimal
//! blocking.
//!
//! ## Design Overview
//!
//! The implementation is based on research from:
//! - [LeanStore](https://dbis1.github.io/leanstore.html) - Optimistic lock coupling for B-trees
//! - [Umbra](https://umbra-db.com/#publications) - High-performance database engine techniques
//!
//! ### Key Concepts
//!
//! **Optimistic Lock Coupling**: Instead of holding locks while traversing the tree, readers
//! acquire "optimistic" access (no locks) and validate at the end that no concurrent
//! modifications occurred. If validation fails, the operation retries. This allows readers
//! to proceed without blocking writers, and vice versa.
//!
//! **Hybrid Latches**: Each node is protected by a [`latch::HybridLatch`] that supports three
//! access modes:
//! - **Optimistic**: Version-based read access with no blocking. Must be validated before
//!   trusting any read data.
//! - **Shared**: Traditional blocking read lock for guaranteed consistent reads.
//! - **Exclusive**: Blocking write lock for modifications.
//!
//! **Fence Keys**: Each node stores `lower_fence` and `upper_fence` keys that define the
//! key range the node is responsible for. These enable efficient range checks and help
//! detect when a node has been split or merged during optimistic traversal.
//!
//! **Sample Keys**: Each node stores a `sample_key` that can be used to relocate the node
//! in the tree after an optimistic validation failure. This avoids restarting from the root.
//!
//! ### Tree Structure
//!
//! ```text
//!                    ┌─────────────────┐
//!                    │   Root Latch    │  <- Protects the root pointer
//!                    │  (HybridLatch)  │
//!                    └────────┬────────┘
//!                             │
//!                             ▼
//!                    ┌─────────────────┐
//!                    │  Internal Node  │  <- Contains keys and child pointers
//!                    │   keys: [K]     │
//!                    │   edges: [ptr]  │
//!                    │   upper_edge    │
//!                    └────────┬────────┘
//!                             │
//!              ┌──────────────┼──────────────┐
//!              ▼              ▼              ▼
//!        ┌──────────┐  ┌──────────┐  ┌──────────┐
//!        │   Leaf   │  │   Leaf   │  │   Leaf   │  <- Store actual key-value pairs
//!        │ keys:[K] │  │ keys:[K] │  │ keys:[K] │
//!        │ vals:[V] │  │ vals:[V] │  │ vals:[V] │
//!        └──────────┘  └──────────┘  └──────────┘
//! ```
//!
//! ## Basic Usage
//!
//! ```
//! use ferntree::Tree;
//!
//! let tree = Tree::new();
//!
//! // Insert key-value pairs
//! tree.insert("key1", "value1");
//! tree.insert("key2", "value2");
//!
//! // Lookup values (requires a closure due to optimistic access)
//! let value = tree.lookup(&"key1", |v| v.to_string());
//! assert_eq!(value, Some("value1".to_string()));
//!
//! // Remove entries
//! tree.remove(&"key1");
//! ```
//!
//! ## Thread Safety
//!
//! The tree is fully thread-safe and can be shared across threads via `Arc<Tree<K, V>>`.
//! All operations use epoch-based memory reclamation via `crossbeam_epoch` to safely
//! deallocate nodes that may still be accessed by concurrent readers.

// Complex types are intentional in this crate for expressing tree traversal results
#![allow(clippy::type_complexity)]

use crossbeam_epoch::{self as epoch, Atomic, Owned};
use smallvec::{smallvec, SmallVec};

use std::borrow::Borrow;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod error;
pub mod iter;
pub mod latch;

use latch::{ExclusiveGuard, HybridGuard, HybridLatch, OptimisticGuard, SharedGuard};

// ---------------------------------------------------------------------------
// Configuration Constants
// ---------------------------------------------------------------------------

/// Default capacity for internal (index) nodes.
/// Each internal node can hold up to `INNER_CAPACITY` keys and `INNER_CAPACITY + 1` child pointers.
/// A larger value reduces tree height but increases node size and split/merge costs.
const INNER_CAPACITY: usize = 64;

/// Default capacity for leaf nodes.
/// Each leaf node can hold up to `LEAF_CAPACITY` key-value pairs.
/// A larger value reduces tree height but increases node size and split/merge costs.
const LEAF_CAPACITY: usize = 64;

// ---------------------------------------------------------------------------
// Public Type Aliases
// ---------------------------------------------------------------------------

/// A B+ tree with default node capacities (64 keys per internal node, 64 entries per leaf).
///
/// This is the recommended type for most use cases. If you need custom node sizes
/// (e.g., for cache optimization or specific workload characteristics), use
/// [`GenericTree`] directly with custom `IC` and `LC` parameters.
pub type Tree<K, V> = GenericTree<K, V, INNER_CAPACITY, LEAF_CAPACITY>;

// ---------------------------------------------------------------------------
// Core Tree Structure
// ---------------------------------------------------------------------------

/// A concurrent B+ tree with configurable node capacities.
///
/// This is the main data structure of the crate. It provides a thread-safe,
/// concurrent B+ tree implementation using optimistic lock coupling.
///
/// # Type Parameters
///
/// - `K`: The key type. Must implement `Clone + Ord`.
/// - `V`: The value type.
/// - `IC`: Internal node capacity (max keys per internal node). Affects tree height
///   and memory layout. Default is 64.
/// - `LC`: Leaf node capacity (max key-value pairs per leaf). Affects tree height
///   and iteration performance. Default is 64.
///
/// # Internal Structure
///
/// The tree consists of:
/// - A **root latch** (`HybridLatch<Atomic<...>>`) that protects the root pointer.
///   This double indirection allows the root node itself to be replaced during splits.
/// - A **height counter** tracking the current tree depth (1 = only root leaf).
///
/// Each node in the tree is wrapped in a `HybridLatch` for concurrency control,
/// and nodes are connected via `Atomic` pointers for safe concurrent access.
pub struct GenericTree<K, V, const IC: usize, const LC: usize> {
	/// The root of the tree, doubly latched for safe root replacement.
	///
	/// Structure: `HybridLatch<Atomic<HybridLatch<Node>>>>`
	/// - Outer latch: Protects the Atomic pointer to the root node
	/// - Atomic: Allows atomic swapping of the root node during splits
	/// - Inner latch: Protects the root node's contents
	root: HybridLatch<Atomic<HybridLatch<Node<K, V, IC, LC>>>>,

	/// The current height of the tree.
	/// - Height 1: Tree contains only a single leaf node (the root)
	/// - Height 2: One internal root node with leaf children
	/// - Height N: N-1 levels of internal nodes plus one level of leaves
	height: AtomicUsize,
}

impl<K: Clone + Ord, V, const IC: usize, const LC: usize> Default for GenericTree<K, V, IC, LC> {
	fn default() -> Self {
		Self::new()
	}
}

// ---------------------------------------------------------------------------
// Internal Helper Types
// ---------------------------------------------------------------------------

/// Result of finding a node's parent during tree traversal.
///
/// When performing operations like split or merge, we need to find a node's parent
/// to update child pointers. This enum distinguishes between:
/// - The node being the root (no parent, but we have the tree-level guard)
/// - The node having a parent internal node
///
/// # Lifetimes
/// - `'r`: Lifetime of the tree guard (when node is root)
/// - `'p`: Lifetime of the parent guard (when node has a parent)
pub(crate) enum ParentHandler<'r, 'p, K, V, const IC: usize, const LC: usize> {
	/// The target node is the root of the tree.
	Root {
		/// Guard on the tree's root pointer, needed to replace the root during splits.
		tree_guard: OptimisticGuard<'r, Atomic<HybridLatch<Node<K, V, IC, LC>>>>,
	},
	/// The target node has a parent internal node.
	Parent {
		/// Optimistic guard on the parent internal node.
		parent_guard: OptimisticGuard<'p, Node<K, V, IC, LC>>,
		/// Position of the target node within the parent's edges array.
		/// If `pos == parent.len`, the target is at `parent.upper_edge`.
		pos: u16,
	},
}

/// Direction for tree traversal operations.
///
/// Used when finding sibling nodes or traversing to the first/last leaf.
#[derive(Debug, PartialEq, Copy, Clone)]
pub(crate) enum Direction {
	/// Traverse toward higher keys (right in the tree).
	Forward,
	/// Traverse toward lower keys (left in the tree).
	Reverse,
}

// ---------------------------------------------------------------------------
// GenericTree Implementation
// ---------------------------------------------------------------------------

impl<K: Clone + Ord, V, const IC: usize, const LC: usize> GenericTree<K, V, IC, LC> {
	// -----------------------------------------------------------------------
	// Construction
	// -----------------------------------------------------------------------

	/// Creates a new, empty B+ tree.
	///
	/// The tree is initialized with a single empty leaf node as the root.
	/// This allocation happens immediately, so `new()` does allocate memory.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<String, i32> = Tree::new();
	/// assert!(tree.is_empty());
	/// assert_eq!(tree.height(), 1); // Single leaf node
	/// ```
	pub fn new() -> Self {
		// Initialize the tree with an empty leaf node as the root.
		// Structure: root_latch -> Atomic -> node_latch -> Node::Leaf
		GenericTree {
			root: HybridLatch::new(Atomic::new(HybridLatch::new(Node::Leaf(LeafNode {
				len: 0,
				keys: smallvec![],
				values: smallvec![],
				// No fences for the root leaf - it covers the entire key space
				lower_fence: None,
				upper_fence: None,
				// No sample key yet (will be set on first insert)
				sample_key: None,
			})))),
			// Height 1 means the tree has only a single leaf node
			height: AtomicUsize::new(1),
		}
	}

	// -----------------------------------------------------------------------
	// Tree Metadata
	// -----------------------------------------------------------------------

	/// Returns the current height of the tree.
	///
	/// - Height 1: Tree contains only a single leaf node (the root)
	/// - Height 2: One internal root with leaf children
	/// - Height N: N-1 levels of internal nodes plus leaves
	///
	/// Note: Height can change during concurrent modifications, so the returned
	/// value may be stale by the time you use it.
	pub fn height(&self) -> usize {
		self.height.load(Ordering::Relaxed)
	}

	// -----------------------------------------------------------------------
	// Parent Finding (for splits and merges)
	// -----------------------------------------------------------------------

	/// Finds the parent of a given node in the tree.
	///
	/// This is a core operation needed for splits and merges, where we must
	/// update the parent's child pointers. The function traverses from the root
	/// using the node's `sample_key` to locate it.
	///
	/// # Algorithm
	///
	/// 1. Acquire optimistic access to the root
	/// 2. Check if the needle IS the root (return `ParentHandler::Root`)
	/// 3. Otherwise, use the needle's `sample_key` to traverse down
	/// 4. At each level, compare the needle's latch pointer to find it
	/// 5. Return the parent and the position of the needle within the parent
	///
	/// # Parameters
	///
	/// - `needle`: The node whose parent we want to find (any guard type)
	/// - `eg`: Epoch guard for memory safety
	///
	/// # Returns
	///
	/// - `Ok(ParentHandler::Root)` if the needle is the root node
	/// - `Ok(ParentHandler::Parent)` with the parent guard and position
	/// - `Err(Error::Reclaimed)` if the node was retired from the tree
	/// - `Err(Error::Unwind)` if optimistic validation failed
	pub(crate) fn find_parent<'t>(
		&'t self,
		needle: &impl HybridGuard<Node<K, V, IC, LC>>,
		eg: &'t epoch::Guard,
	) -> error::Result<ParentHandler<'t, 't, K, V, IC, LC>>
	where
		K: Ord,
	{
		// Step 1: Acquire optimistic access to the tree's root pointer
		let tree_guard = self.root.optimistic_or_spin();

		// Step 2: Load the root node through the Atomic pointer
		// SAFETY: The epoch guard ensures the pointer remains valid
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_latch_ptr = root_latch as *const _;

		// Acquire optimistic access to the root node
		let root_guard = root_latch.optimistic_or_spin();

		// Step 3: Check if the needle IS the root
		// Compare latch pointers (identity check, not content)
		if std::ptr::eq(needle.latch(), root_latch_ptr) {
			// Validate our optimistic reads were consistent
			tree_guard.recheck()?;
			return Ok(ParentHandler::Root {
				tree_guard,
			});
		}

		// Step 4: Get the sample_key from the needle to navigate to it
		// The sample_key is a key known to exist in (or route to) this node
		let search_key = match needle.inner().sample_key().cloned() {
			Some(key) => key,
			None => {
				// Node has no sample_key - it may have been emptied and reclaimed
				needle.recheck()?;
				return Err(error::Error::Reclaimed);
			}
		};

		// Step 5: Traverse from root toward the needle using lock coupling
		// We keep track of:
		// - t_guard: Tree guard (released after leaving root level)
		// - p_guard: Previous (potential parent) guard
		// - target_guard: Current node being examined
		// - pos: Position within parent's edges
		let mut t_guard = Some(tree_guard);
		let mut p_guard: Option<OptimisticGuard<'_, Node<K, V, IC, LC>>> = None;
		let mut target_guard = root_guard;
		let mut pos = 0u16;

		// Descend the tree looking for the needle
		let parent_guard = loop {
			// Determine which child to descend into based on the search key
			let (c_swip, c_pos) = match *target_guard {
				Node::Internal(ref internal) => {
					// Binary search for the correct child position
					let (c_pos, _) = internal.lower_bound(&search_key);
					// Get the child pointer (swip = "software-implemented pointer")
					let swip = internal.edge_at(c_pos)?;
					(swip, c_pos)
				}
				Node::Leaf(ref _leaf) => {
					// Reached a leaf without finding the needle in internal nodes
					// The previous node (p_guard) must be the parent
					break p_guard.expect("must have parent");
				}
			};

			// Load the child node
			// SAFETY: Epoch guard ensures the pointer is valid
			let c_latch = unsafe { c_swip.load(Ordering::Acquire, eg).deref() };
			let c_latch_ptr = c_latch as *const _;

			// Check if this child IS the needle we're looking for
			if std::ptr::eq(needle.latch(), c_latch_ptr) {
				// Found it! The current target_guard is the parent
				target_guard.recheck()?;
				if let Some(tree_guard) = t_guard.take() {
					tree_guard.recheck()?;
				}
				pos = c_pos; // Update pos to the actual position of the needle
				break target_guard;
			}

			// Not found yet - descend to the child using lock coupling
			// Lock coupling: acquire child lock, then validate parent, then release parent
			let guard = Self::lock_coupling(&target_guard, c_swip, eg)?;

			// The current target becomes the previous (potential parent)
			p_guard = Some(target_guard);
			pos = c_pos;
			target_guard = guard;

			// Release tree guard after leaving root level (no longer needed)
			if let Some(tree_guard) = t_guard.take() {
				tree_guard.recheck()?;
			}
		};

		Ok(ParentHandler::Parent {
			parent_guard,
			pos,
		})
	}

	// -----------------------------------------------------------------------
	// Sibling Leaf Finding (for iteration)
	// -----------------------------------------------------------------------

	/// Finds the nearest sibling leaf node in the given direction.
	///
	/// This is used by iterators to move to the next/previous leaf when the
	/// current leaf is exhausted. The algorithm handles the case where the
	/// sibling may be in a different subtree (requiring traversal up and down).
	///
	/// # Algorithm
	///
	/// 1. Find the needle's parent
	/// 2. Check if there's a sibling in the parent (adjacent edge)
	/// 3. If yes, descend to the appropriate leaf in that sibling subtree
	/// 4. If no (at edge of parent), go up to grandparent and repeat
	///
	/// # Parameters
	///
	/// - `needle`: The current leaf node (optimistic guard)
	/// - `direction`: Which sibling to find (Forward = right/next, Reverse = left/prev)
	/// - `eg`: Epoch guard for memory safety
	///
	/// # Returns
	///
	/// - `Ok(Some((leaf_guard, (parent_guard, pos))))` - Found the sibling leaf
	/// - `Ok(None)` - No sibling exists (at tree boundary)
	/// - `Err(...)` - Optimistic validation failed
	pub(crate) fn find_nearest_leaf<'t, 'g>(
		&'t self,
		needle: &OptimisticGuard<'g, Node<K, V, IC, LC>>,
		direction: Direction,
		eg: &'t epoch::Guard,
	) -> error::Result<
		Option<(
			OptimisticGuard<'t, Node<K, V, IC, LC>>,
			(OptimisticGuard<'t, Node<K, V, IC, LC>>, u16),
		)>,
	>
	where
		K: Ord,
	{
		// Check if needle is the root (no siblings possible)
		let tree_guard = self.root.optimistic_or_spin();
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_latch_ptr = root_latch as *const _;
		let root_guard = root_latch.optimistic_or_spin();

		if std::ptr::eq(needle.latch(), root_latch_ptr) {
			// Needle is the root - no siblings exist
			root_guard.recheck()?;
			tree_guard.recheck()?;
			return error::Result::Ok(None);
		}

		// Find the needle's parent
		let (parent_guard, pos) = match self.find_parent(needle, eg)? {
			ParentHandler::Root {
				tree_guard: _,
			} => {
				// Needle is root - no siblings
				return error::Result::Ok(None);
			}
			ParentHandler::Parent {
				parent_guard,
				pos,
			} => (parent_guard, pos),
		};

		// Check if there's a sibling within the parent's children
		// For Forward: can we go to pos+1?
		// For Reverse: can we go to pos-1?
		let within_bounds = match direction {
			Direction::Forward => pos < parent_guard.as_internal().len,
			Direction::Reverse => pos > 0,
		};

		if within_bounds {
			// Sibling exists within the same parent - simple case
			let lookup_pos = match direction {
				Direction::Forward => pos + 1,
				Direction::Reverse => pos - 1,
			};

			// Get the sibling's edge
			let swip = parent_guard.as_internal().edge_at(lookup_pos)?;

			// Lock couple to the sibling
			let guard = GenericTree::lock_coupling(&parent_guard, swip, eg)?;

			if guard.is_leaf() {
				// Sibling is directly a leaf - we're done
				guard.recheck()?;
				error::Result::Ok(Some((guard, (parent_guard, lookup_pos))))
			} else {
				// Sibling is an internal node - descend to the appropriate leaf
				// (leftmost for Forward, rightmost for Reverse)
				let (leaf, parent_opt) =
					self.find_leaf_and_parent_from_node(guard, direction, eg)?;
				error::Result::Ok(Some((leaf, parent_opt.expect("must have parent here"))))
			}
		} else {
			// No sibling in parent - must go up to grandparent and try again
			// This handles the case where we're at the edge of a subtree
			let mut target_guard = parent_guard;

			loop {
				// Find the grandparent
				let (parent_guard, pos) = match self.find_parent(&target_guard, eg)? {
					ParentHandler::Root {
						tree_guard: _,
					} => {
						// Reached the root without finding a sibling - at tree boundary
						return error::Result::Ok(None);
					}
					ParentHandler::Parent {
						parent_guard,
						pos,
					} => (parent_guard, pos),
				};

				// Check if there's a sibling at this level
				let within_bounds = match direction {
					Direction::Forward => pos < parent_guard.as_internal().len,
					Direction::Reverse => pos > 0,
				};

				if within_bounds {
					// Found a sibling subtree - descend to find the target leaf
					let lookup_pos = match direction {
						Direction::Forward => pos + 1,
						Direction::Reverse => pos - 1,
					};
					let swip = parent_guard.as_internal().edge_at(lookup_pos)?;

					let guard = GenericTree::lock_coupling(&parent_guard, swip, eg)?;

					if guard.is_leaf() {
						guard.recheck()?;
						return error::Result::Ok(Some((guard, (parent_guard, lookup_pos))));
					} else {
						// Descend to the appropriate leaf in this subtree
						let (leaf, parent_opt) =
							self.find_leaf_and_parent_from_node(guard, direction, eg)?;
						return error::Result::Ok(Some((
							leaf,
							parent_opt.expect("must have parent here"),
						)));
					}
				} else {
					// Still at edge - continue going up
					target_guard = parent_guard;
					continue;
				}
			}
		}
	}

	// -----------------------------------------------------------------------
	// Lock Coupling Primitives
	// -----------------------------------------------------------------------
	//
	// These functions implement the core "lock coupling" pattern from LeanStore.
	// The pattern is: to descend from parent to child safely:
	//   1. Load the child pointer from the parent
	//   2. Acquire access to the child
	//   3. Validate the parent hasn't changed (recheck)
	//   4. Only then is it safe to release/continue without parent
	//
	// This ensures we don't follow a stale child pointer if the parent was
	// concurrently modified (split/merged).

	/// Acquires optimistic access to a child node using lock coupling.
	///
	/// This is the standard traversal pattern for read operations. The child
	/// is accessed optimistically (no blocking), and we validate the parent
	/// to ensure the child pointer we followed is still valid.
	///
	/// # Safety
	///
	/// The `swip` pointer is dereferenced under the epoch guard's protection.
	/// The guard ensures the memory isn't reclaimed while we're accessing it.
	pub(crate) fn lock_coupling<'e>(
		p_guard: &OptimisticGuard<'e, Node<K, V, IC, LC>>,
		swip: &Atomic<HybridLatch<Node<K, V, IC, LC>>>,
		eg: &'e epoch::Guard,
	) -> error::Result<OptimisticGuard<'e, Node<K, V, IC, LC>>> {
		// Step 1: Load the child pointer and dereference to get the latch
		// SAFETY: Epoch guard protects the pointer from being freed
		let c_latch = unsafe { swip.load(Ordering::Acquire, eg).deref() };

		// Step 2: Acquire optimistic access to the child
		let c_guard = c_latch.optimistic_or_spin();

		// Step 3: Validate the parent - ensures the child pointer was valid
		// If parent changed, the child pointer might be stale
		p_guard.recheck()?;

		Ok(c_guard)
	}

	/// Acquires shared (blocking read) access to a child node using lock coupling.
	///
	/// Used when we need guaranteed consistent reads, typically at the leaf
	/// level for read iterators. The shared lock blocks writers but allows
	/// concurrent readers.
	fn lock_coupling_shared<'e>(
		p_guard: &OptimisticGuard<'e, Node<K, V, IC, LC>>,
		swip: &Atomic<HybridLatch<Node<K, V, IC, LC>>>,
		eg: &'e epoch::Guard,
	) -> error::Result<SharedGuard<'e, Node<K, V, IC, LC>>> {
		// SAFETY: Epoch guard protects the pointer
		let c_latch = unsafe { swip.load(Ordering::Acquire, eg).deref() };

		// Acquire shared (blocking) access to the child
		let c_guard = c_latch.shared();

		// Validate parent after acquiring child lock
		p_guard.recheck()?;

		Ok(c_guard)
	}

	/// Acquires exclusive (blocking write) access to a child node using lock coupling.
	///
	/// Used when we need to modify the child node, typically at the leaf
	/// level for write iterators and insert/remove operations.
	fn lock_coupling_exclusive<'e>(
		p_guard: &OptimisticGuard<'e, Node<K, V, IC, LC>>,
		swip: &Atomic<HybridLatch<Node<K, V, IC, LC>>>,
		eg: &'e epoch::Guard,
	) -> error::Result<ExclusiveGuard<'e, Node<K, V, IC, LC>>> {
		// SAFETY: Epoch guard protects the pointer
		let c_latch = unsafe { swip.load(Ordering::Acquire, eg).deref() };

		// Acquire exclusive (blocking) access to the child
		let c_guard = c_latch.exclusive();

		// Validate parent after acquiring child lock
		p_guard.recheck()?;

		Ok(c_guard)
	}

	// -----------------------------------------------------------------------
	// Leaf Finding (Tree Traversal)
	// -----------------------------------------------------------------------

	/// Descends from a starting node to a leaf, following the given direction.
	///
	/// Used after finding a sibling subtree to locate the appropriate leaf:
	/// - `Forward`: Descend to the leftmost (first) leaf in the subtree
	/// - `Reverse`: Descend to the rightmost (last) leaf in the subtree
	///
	/// # Returns
	///
	/// A tuple of:
	/// - The leaf guard
	/// - The parent info (parent guard and position), or `None` if starting from root
	fn find_leaf_and_parent_from_node<'e>(
		&self,
		needle: OptimisticGuard<'e, Node<K, V, IC, LC>>,
		direction: Direction,
		eg: &'e epoch::Guard,
	) -> error::Result<(
		OptimisticGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)> {
		let mut p_guard = None;
		let mut target_guard = needle;

		// Descend until we reach a leaf
		let leaf_guard = loop {
			let (c_swip, pos) = match *target_guard {
				Node::Internal(ref internal) => {
					// Choose which edge to follow based on direction:
					// - Forward (seeking first): take leftmost child (position 0)
					// - Reverse (seeking last): take rightmost child (position len = upper_edge)
					let pos = match direction {
						Direction::Forward => 0,
						Direction::Reverse => internal.len,
					};
					let swip = internal.edge_at(pos)?;
					(swip, pos)
				}
				Node::Leaf(ref _leaf) => {
					// Reached the target leaf
					break target_guard;
				}
			};

			// Lock couple to the child
			let guard = GenericTree::lock_coupling(&target_guard, c_swip, eg)?;
			p_guard = Some((target_guard, pos));
			target_guard = guard;
		};

		// Final validation of the leaf guard
		leaf_guard.recheck()?;

		Ok((leaf_guard, p_guard))
	}

	/// Finds the first (leftmost) leaf in the tree with its parent info.
	///
	/// Used by iterators to initialize at the beginning of the tree.
	fn find_first_leaf_and_parent<'e>(
		&self,
		eg: &'e epoch::Guard,
	) -> error::Result<(
		OptimisticGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)> {
		// Start from the root
		let tree_guard = self.root.optimistic_or_spin();
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_guard = root_latch.optimistic_or_spin();
		tree_guard.recheck()?;

		// Descend to the leftmost leaf
		self.find_leaf_and_parent_from_node(root_guard, Direction::Forward, eg)
	}

	/// Finds the last (rightmost) leaf in the tree with its parent info.
	///
	/// Used by iterators to initialize at the end of the tree.
	fn find_last_leaf_and_parent<'e>(
		&self,
		eg: &'e epoch::Guard,
	) -> error::Result<(
		OptimisticGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)> {
		// Start from the root
		let tree_guard = self.root.optimistic_or_spin();
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_guard = root_latch.optimistic_or_spin();
		tree_guard.recheck()?;

		// Descend to the rightmost leaf
		self.find_leaf_and_parent_from_node(root_guard, Direction::Reverse, eg)
	}

	/// Finds the leaf containing (or that would contain) the given key.
	///
	/// This is the primary tree traversal function. It descends from the root,
	/// using binary search at each internal node to find the correct child,
	/// until reaching a leaf node.
	///
	/// # Returns
	///
	/// A tuple of:
	/// - The leaf guard (optimistic)
	/// - The parent info (parent guard and position in parent), or `None` if tree has only root
	fn find_leaf_and_parent<'e, Q>(
		&self,
		key: &Q,
		eg: &'e epoch::Guard,
	) -> error::Result<(
		OptimisticGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		// Acquire access to the root
		let tree_guard = self.root.optimistic_or_spin();
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_guard = root_latch.optimistic_or_spin();
		tree_guard.recheck()?;

		// Track the tree guard until we leave the root level
		let mut t_guard = Some(tree_guard);
		let mut p_guard = None;
		let mut target_guard = root_guard;

		// Descend the tree following the key
		let leaf_guard = loop {
			let (c_swip, pos) = match *target_guard {
				Node::Internal(ref internal) => {
					// Binary search to find which child contains the key
					let (pos, _) = internal.lower_bound(key);
					let swip = internal.edge_at(pos)?;
					(swip, pos)
				}
				Node::Leaf(ref _leaf) => {
					// Reached the target leaf
					break target_guard;
				}
			};

			// Lock couple to the child
			let guard = GenericTree::lock_coupling(&target_guard, c_swip, eg)?;
			p_guard = Some((target_guard, pos));
			target_guard = guard;

			// Release tree guard after leaving root level
			if let Some(tree_guard) = t_guard.take() {
				tree_guard.recheck()?;
			}
		};

		Ok((leaf_guard, p_guard))
	}

	/// Convenience function to find just the leaf (without parent info).
	#[allow(dead_code)]
	fn find_leaf<'e, Q>(
		&self,
		key: &Q,
		eg: &'e epoch::Guard,
	) -> error::Result<OptimisticGuard<'e, Node<K, V, IC, LC>>>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		self.find_leaf_and_parent(key, eg).map(|(leaf, _)| leaf)
	}

	// -----------------------------------------------------------------------
	// Leaf Finding with Specific Lock Types (for iterators)
	// -----------------------------------------------------------------------
	//
	// These functions find leaves and acquire specific lock types:
	// - Shared: For read iterators (RawSharedIter)
	// - Exclusive: For write iterators (RawExclusiveIter)
	//
	// They keep the parent optimistically locked so the iterator can
	// efficiently move to sibling leaves without re-traversing from root.

	/// Finds a leaf by key and acquires a shared lock on it.
	///
	/// Used by `RawSharedIter` for read-only iteration. The traversal uses
	/// optimistic locks until the leaf level, where a shared lock is acquired.
	/// This blocks writers but allows concurrent readers.
	///
	/// # Retry Loop
	///
	/// This function loops internally until successful. Validation failures
	/// at any point cause a retry from the beginning.
	pub(crate) fn find_shared_leaf_and_optimistic_parent<'e, Q>(
		&self,
		key: &Q,
		eg: &'e epoch::Guard,
	) -> (SharedGuard<'e, Node<K, V, IC, LC>>, Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		// Retry loop for optimistic validation failures
		loop {
			let perform = || {
				// Start traversal from root
				let tree_guard = self.root.optimistic_or_spin();
				let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
				let root_guard = root_latch.optimistic_or_spin();
				tree_guard.recheck()?;

				let mut t_guard = Some(tree_guard);
				let mut p_guard = None;
				let mut target_guard = root_guard;

				// Track current level to know when we're about to reach leaves
				let mut level = 1u16;

				let leaf_guard = loop {
					let (c_swip, pos) = match *target_guard {
						Node::Internal(ref internal) => {
							// Binary search for the correct child
							let (pos, _) = internal.lower_bound(key);
							let swip = internal.edge_at(pos)?;
							(swip, pos)
						}
						Node::Leaf(ref _leaf) => {
							// Edge case: root is a leaf (single-node tree)
							if let Some(tree_guard) = t_guard.take() {
								tree_guard.recheck()?;
							}

							if p_guard.is_none() {
								// Root is the only node - upgrade to shared lock
								break target_guard.to_shared()?;
							} else {
								// Should never happen - found leaf before expected level
								unreachable!(
									"tree structure corruption: encountered leaf at internal level during traversal"
								)
							}
						}
					};

					// Check if next level is the leaf level
					if (level + 1) as usize == self.height.load(Ordering::Acquire) {
						// About to access leaf - use shared lock coupling
						if let Some(tree_guard) = t_guard.take() {
							tree_guard.recheck()?;
						}

						// Acquire shared lock on the leaf
						let guard = Self::lock_coupling_shared(&target_guard, c_swip, eg)?;
						p_guard = Some((target_guard, pos));

						break guard;
					} else {
						// Still in internal nodes - use optimistic lock coupling
						let guard = GenericTree::lock_coupling(&target_guard, c_swip, eg)?;
						p_guard = Some((target_guard, pos));
						target_guard = guard;

						if let Some(tree_guard) = t_guard.take() {
							tree_guard.recheck()?;
						}

						level += 1;
					}
				};

				error::Result::Ok((leaf_guard, p_guard))
			};

			match perform() {
				Ok(tup) => {
					return tup;
				}
				Err(_) => {
					// Validation failed - retry from beginning
					continue;
				}
			}
		}
	}

	/// Finds the first leaf and acquires a shared lock.
	///
	/// Used by iterators when seeking to the beginning of the tree.
	pub(crate) fn find_first_shared_leaf_and_optimistic_parent<'e>(
		&self,
		eg: &'e epoch::Guard,
	) -> (SharedGuard<'e, Node<K, V, IC, LC>>, Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>)
	{
		loop {
			let perform = || {
				// Find first leaf with optimistic traversal
				let (leaf, parent_opt) = self.find_first_leaf_and_parent(eg)?;
				// Upgrade to shared lock
				let shared_leaf = leaf.to_shared()?;
				error::Result::Ok((shared_leaf, parent_opt))
			};

			match perform() {
				Ok(tup) => {
					return tup;
				}
				Err(_) => {
					continue;
				}
			}
		}
	}

	/// Finds the last leaf and acquires a shared lock.
	///
	/// Used by iterators when seeking to the end of the tree.
	pub(crate) fn find_last_shared_leaf_and_optimistic_parent<'e>(
		&self,
		eg: &'e epoch::Guard,
	) -> (SharedGuard<'e, Node<K, V, IC, LC>>, Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>)
	{
		loop {
			let perform = || {
				// Find last leaf with optimistic traversal
				let (leaf, parent_opt) = self.find_last_leaf_and_parent(eg)?;
				// Upgrade to shared lock
				let shared_leaf = leaf.to_shared()?;
				error::Result::Ok((shared_leaf, parent_opt))
			};

			match perform() {
				Ok(tup) => {
					return tup;
				}
				Err(_) => {
					continue;
				}
			}
		}
	}

	/// Finds a leaf by key and acquires an exclusive lock on it.
	///
	/// Used by `RawExclusiveIter` for read-write iteration (insert/remove).
	/// The traversal uses optimistic locks until the leaf level, where an
	/// exclusive lock is acquired. This blocks all other readers and writers.
	pub(crate) fn find_exclusive_leaf_and_optimistic_parent<'e, Q>(
		&self,
		key: &Q,
		eg: &'e epoch::Guard,
	) -> (
		ExclusiveGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		loop {
			let perform = || {
				// Start traversal from root
				let tree_guard = self.root.optimistic_or_spin();
				let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
				let root_guard = root_latch.optimistic_or_spin();
				tree_guard.recheck()?;

				let mut t_guard = Some(tree_guard);
				let mut p_guard = None;
				let mut target_guard = root_guard;

				let mut level = 1u16;

				let leaf_guard = loop {
					let (c_swip, pos) = match *target_guard {
						Node::Internal(ref internal) => {
							let (pos, _) = internal.lower_bound(key);
							let swip = internal.edge_at(pos)?;
							(swip, pos)
						}
						Node::Leaf(ref _leaf) => {
							// Root is a leaf - upgrade to exclusive
							if let Some(tree_guard) = t_guard.take() {
								tree_guard.recheck()?;
							}

							if p_guard.is_none() {
								break target_guard.to_exclusive()?;
							} else {
								unreachable!(
									"tree structure corruption: encountered leaf at internal level during traversal"
								)
							}
						}
					};

					if (level + 1) as usize == self.height.load(Ordering::Acquire) {
						// About to access leaf - use exclusive lock coupling
						if let Some(tree_guard) = t_guard.take() {
							tree_guard.recheck()?;
						}

						let guard = Self::lock_coupling_exclusive(&target_guard, c_swip, eg)?;
						p_guard = Some((target_guard, pos));

						break guard;
					} else {
						let guard = GenericTree::lock_coupling(&target_guard, c_swip, eg)?;
						p_guard = Some((target_guard, pos));
						target_guard = guard;

						if let Some(tree_guard) = t_guard.take() {
							tree_guard.recheck()?;
						}

						level += 1;
					}
				};

				error::Result::Ok((leaf_guard, p_guard))
			};

			match perform() {
				Ok(tup) => {
					return tup;
				}
				Err(_) => {
					continue;
				}
			}
		}
	}

	/// Finds a leaf containing an exact key match and acquires exclusive lock.
	///
	/// Returns `None` if the key doesn't exist. Used for remove operations
	/// where we need to find a specific entry.
	#[allow(dead_code)]
	pub(crate) fn find_exact_exclusive_leaf_and_optimistic_parent<'e, Q>(
		&self,
		key: &Q,
		eg: &'e epoch::Guard,
	) -> Option<(
		(ExclusiveGuard<'e, Node<K, V, IC, LC>>, u16),
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		loop {
			let perform = || {
				// Find the leaf that would contain the key
				let (leaf, parent_opt) = self.find_leaf_and_parent(key, eg)?;

				// Check if the key actually exists in this leaf
				let (pos, exact) = leaf.as_leaf().lower_bound(key);

				if exact {
					// Key found - upgrade to exclusive lock
					let exclusive_leaf = leaf.to_exclusive()?;
					error::Result::Ok(Some(((exclusive_leaf, pos), parent_opt)))
				} else {
					// Key not found
					leaf.recheck()?;
					error::Result::Ok(None)
				}
			};

			match perform() {
				Ok(opt) => {
					return opt;
				}
				Err(_) => {
					continue;
				}
			}
		}
	}

	/// Finds the first leaf and acquires an exclusive lock.
	pub(crate) fn find_first_exclusive_leaf_and_optimistic_parent<'e>(
		&self,
		eg: &'e epoch::Guard,
	) -> (
		ExclusiveGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	) {
		loop {
			let perform = || {
				let (leaf, parent_opt) = self.find_first_leaf_and_parent(eg)?;
				let exclusive_leaf = leaf.to_exclusive()?;
				error::Result::Ok((exclusive_leaf, parent_opt))
			};

			match perform() {
				Ok(tup) => {
					return tup;
				}
				Err(_) => {
					continue;
				}
			}
		}
	}

	/// Finds the last leaf and acquires an exclusive lock.
	pub(crate) fn find_last_exclusive_leaf_and_optimistic_parent<'e>(
		&self,
		eg: &'e epoch::Guard,
	) -> (
		ExclusiveGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	) {
		loop {
			let perform = || {
				let (leaf, parent_opt) = self.find_last_leaf_and_parent(eg)?;
				let exclusive_leaf = leaf.to_exclusive()?;
				error::Result::Ok((exclusive_leaf, parent_opt))
			};

			match perform() {
				Ok(tup) => {
					return tup;
				}
				Err(_) => {
					continue;
				}
			}
		}
	}

	// -----------------------------------------------------------------------
	// Public API: Read Operations
	// -----------------------------------------------------------------------

	/// Looks up a value in the tree using optimistic concurrency.
	///
	/// This method uses a closure-based API because the underlying access is
	/// optimistic and may be retried. The closure receives a reference to the
	/// value and should extract/clone whatever data is needed.
	///
	/// # Important
	///
	/// The closure `f` may be executed multiple times if concurrent modifications
	/// cause validation failures. **Do not perform side effects in the closure.**
	/// The value reference passed to `f` may contain inconsistent data during
	/// retries; only the final successful call's result is returned.
	///
	/// # Parameters
	///
	/// - `key`: The key to look up
	/// - `f`: A closure that receives `&V` and returns the desired result
	///
	/// # Returns
	///
	/// - `Some(R)`: The result of `f` if the key was found
	/// - `None`: If the key doesn't exist in the tree
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<String, Vec<i32>> = Tree::new();
	/// tree.insert("key".to_string(), vec![1, 2, 3]);
	///
	/// // Clone the entire value
	/// let value = tree.lookup(&"key".to_string(), |v| v.clone());
	/// assert_eq!(value, Some(vec![1, 2, 3]));
	///
	/// // Extract just what you need
	/// let len = tree.lookup(&"key".to_string(), |v| v.len());
	/// assert_eq!(len, Some(3));
	/// ```
	pub fn lookup<Q, R, F>(&self, key: &Q, f: F) -> Option<R>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
		F: Fn(&V) -> R,
	{
		// Pin the current epoch for memory safety
		let eg = &epoch::pin();

		// Retry loop for optimistic validation failures
		loop {
			let perform = || {
				// Find the leaf that would contain this key
				let guard = self.find_leaf(key, eg)?;

				if let Node::Leaf(ref leaf) = *guard {
					// Binary search for the key within the leaf
					let (pos, exact) = leaf.lower_bound(key);

					if exact {
						// Key found - call the user's closure
						// Note: This read might be invalid if the tree changed
						let result = f(leaf.value_at(pos)?);

						// Validate that our reads were consistent
						guard.recheck()?;

						error::Result::Ok(Some(result))
					} else {
						// Key not found in this leaf
						guard.recheck()?;
						error::Result::Ok(None)
					}
				} else {
					// find_leaf should always return a leaf node
					unreachable!(
						"find_leaf returned non-leaf node - tree traversal invariant violated"
					)
				}
			};

			match perform() {
				Ok(opt) => {
					return opt;
				}
				Err(_) => {
					// Validation failed - retry
					continue;
				}
			}
		}
	}

	/// Returns `true` if the tree contains the specified key.
	///
	/// This is a convenience method equivalent to `lookup(key, |_| ()).is_some()`.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// tree.insert(1, "one");
	///
	/// assert!(tree.contains_key(&1));
	/// assert!(!tree.contains_key(&2));
	/// ```
	pub fn contains_key<Q>(&self, key: &Q) -> bool
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		self.lookup(key, |_| ()).is_some()
	}

	/// Returns a clone of the value corresponding to the key.
	///
	/// This is a convenience method equivalent to `lookup(key, |v| v.clone())`.
	/// For more control over what is extracted, use [`lookup`](Self::lookup).
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, String> = Tree::new();
	/// tree.insert(1, "one".to_string());
	///
	/// assert_eq!(tree.get(&1), Some("one".to_string()));
	/// assert_eq!(tree.get(&2), None);
	/// ```
	pub fn get<Q>(&self, key: &Q) -> Option<V>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
		V: Clone,
	{
		self.lookup(key, |v| v.clone())
	}

	/// Returns the first (minimum) key-value pair in the tree.
	///
	/// The closure receives references to the key and value and should extract
	/// whatever data is needed. Returns `None` if the tree is empty.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// tree.insert(3, "three");
	/// tree.insert(1, "one");
	/// tree.insert(2, "two");
	///
	/// let first = tree.first_key_value(|k, v| (*k, *v));
	/// assert_eq!(first, Some((1, "one")));
	/// ```
	pub fn first_key_value<R, F>(&self, f: F) -> Option<R>
	where
		K: Ord,
		F: Fn(&K, &V) -> R,
	{
		let mut iter = self.raw_iter();
		iter.seek_to_first();
		iter.next().map(|(k, v)| f(k, v))
	}

	/// Returns the last (maximum) key-value pair in the tree.
	///
	/// The closure receives references to the key and value and should extract
	/// whatever data is needed. Returns `None` if the tree is empty.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// tree.insert(1, "one");
	/// tree.insert(3, "three");
	/// tree.insert(2, "two");
	///
	/// let last = tree.last_key_value(|k, v| (*k, *v));
	/// assert_eq!(last, Some((3, "three")));
	/// ```
	pub fn last_key_value<R, F>(&self, f: F) -> Option<R>
	where
		K: Ord,
		F: Fn(&K, &V) -> R,
	{
		let mut iter = self.raw_iter();
		iter.seek_to_last();
		iter.prev().map(|(k, v)| f(k, v))
	}

	// -----------------------------------------------------------------------
	// Public API: Write Operations
	// -----------------------------------------------------------------------

	/// Removes a key from the tree, returning the value if it existed.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// tree.insert(1, "one");
	///
	/// assert_eq!(tree.remove(&1), Some("one"));
	/// assert_eq!(tree.remove(&1), None); // Already removed
	/// ```
	pub fn remove<Q>(&self, key: &Q) -> Option<V>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		// Delegate to remove_entry and discard the key
		self.remove_entry(key).map(|(_, v)| v)
	}

	/// Removes a key from the tree, returning the stored key and value.
	///
	/// This is useful when you need to recover the owned key (e.g., for
	/// case-insensitive lookups where you want the original key).
	///
	/// # Algorithm
	///
	/// 1. Find the leaf containing the exact key
	/// 2. Acquire exclusive lock on the leaf
	/// 3. Remove the entry
	/// 4. If the leaf is now underfull, attempt to merge with a sibling
	pub fn remove_entry<Q>(&self, key: &Q) -> Option<(K, V)>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		// Pin the epoch for memory safety during the operation
		let eg = epoch::pin();

		let result = if let Some(((mut guard, pos), _parent_opt)) =
			self.find_exact_exclusive_leaf_and_optimistic_parent(key, &eg)
		{
			// Remove the key-value pair from the leaf
			let kv = guard.as_leaf_mut().remove_at(pos);

			// Check if the leaf is now underfull and needs merging
			if guard.is_underfull() {
				// Unlock the leaf before attempting merge (merge needs fresh traversal)
				let guard = guard.unlock();

				// Attempt to merge with a sibling (best-effort, ignore result)
				let _ = self.try_merge(&guard, &eg);
			}

			Some(kv)
		} else {
			// Key not found
			None
		};

		// Explicitly drop the epoch guard to allow memory reclamation
		drop(eg);
		result
	}

	/// Inserts a key-value pair into the tree.
	///
	/// If the key already exists, the value is updated and the old value is
	/// returned. If the key is new, `None` is returned.
	///
	/// # Algorithm
	///
	/// Insertion is delegated to `RawExclusiveIter::insert`, which:
	/// 1. Seeks to the position where the key should be
	/// 2. If key exists, updates the value in place
	/// 3. If key is new, inserts it (potentially triggering a split)
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	///
	/// assert_eq!(tree.insert(1, "one"), None);       // New key
	/// assert_eq!(tree.insert(1, "uno"), Some("one")); // Update existing
	/// ```
	pub fn insert(&self, key: K, value: V) -> Option<V>
	where
		K: Ord,
	{
		// Use the mutable iterator for insertion
		// This handles splits automatically
		let mut iter = self.raw_iter_mut();
		iter.insert(key, value)
	}

	/// Removes all entries from the tree.
	///
	/// After calling this method, the tree will be empty with height 1.
	/// Old nodes are scheduled for deferred destruction via epoch-based
	/// reclamation, ensuring concurrent readers can safely finish.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// tree.insert(1, "one");
	/// tree.insert(2, "two");
	/// assert_eq!(tree.len(), 2);
	///
	/// tree.clear();
	/// assert!(tree.is_empty());
	/// assert_eq!(tree.height(), 1);
	/// ```
	pub fn clear(&self)
	where
		K: Ord,
	{
		let eg = epoch::pin();

		// Acquire exclusive access to root
		let mut tree_guard = self.root.exclusive();

		// Schedule old root for deferred destruction
		let old_root = tree_guard.load(Ordering::Acquire, &eg);
		if !old_root.is_null() {
			// SAFETY: The epoch guard ensures safe reclamation
			unsafe { eg.defer_destroy(old_root) };
		}

		// Create fresh empty leaf as new root
		let new_root = Owned::new(HybridLatch::new(Node::Leaf(LeafNode::new())));
		*tree_guard = Atomic::from(new_root);

		// Reset height to 1
		self.height.store(1, Ordering::Release);
	}

	// -----------------------------------------------------------------------
	// Node Splitting
	// -----------------------------------------------------------------------

	/// Attempts to split an overfull node.
	///
	/// This is called when a node exceeds its capacity after an insertion.
	/// The split creates a new sibling node and redistributes entries.
	///
	/// # Algorithm Overview
	///
	/// ```text
	/// Before split (node is full):
	///   Parent: [..., K_prev, ptr] [K_next, ...]
	///                        │
	///                        ▼
	///   Node: [K1, K2, K3, K4, K5, K6, K7, K8]  <- FULL
	///
	/// After split:
	///   Parent: [..., K_prev, ptr] [K_split, new_ptr] [K_next, ...]
	///                        │              │
	///                        ▼              ▼
	///   Left:  [K1, K2, K3, K4]    Right: [K5, K6, K7, K8]
	/// ```
	///
	/// # Root Split Special Case
	///
	/// When the root is split, we must create a new root node. This is the
	/// only operation that increases tree height:
	///
	/// ```text
	/// Before (root is full):
	///   Root: [K1, K2, K3, K4, K5, K6, K7, K8]
	///
	/// After:
	///   New Root: [K_split]
	///              /     \
	///   Left: [K1..K4]   Right: [K5..K8]
	/// ```
	///
	/// # Parameters
	///
	/// - `needle`: The node to split (optimistic guard)
	/// - `eg`: Epoch guard for memory safety
	///
	/// # Returns
	///
	/// - `Ok(())`: Split succeeded
	/// - `Err(Error::Reclaimed)`: Tree structure changed, caller should retry
	/// - `Err(Error::Unwind)`: Optimistic validation failed
	pub(crate) fn try_split<'t, 'g, 'e>(
		&'t self,
		needle: &OptimisticGuard<'g, Node<K, V, IC, LC>>,
		eg: &'e epoch::Guard,
	) -> error::Result<()>
	where
		K: Ord,
	{
		// Step 1: Find the needle's parent
		let parent_handler = self.find_parent(needle, eg)?;

		match parent_handler {
			// ===================================================================
			// Case 1: Splitting the root node
			// ===================================================================
			ParentHandler::Root {
				tree_guard,
			} => {
				// Upgrade tree guard to exclusive - we're going to replace the root
				let mut tree_guard_x = tree_guard.to_exclusive()?;

				// Get exclusive access to the root node
				let root_latch = unsafe { tree_guard_x.load(Ordering::Acquire, eg).deref() };
				let mut root_guard_x = root_latch.exclusive();

				// Allocate the new root node (will be an internal node)
				let mut new_root_owned: Owned<HybridLatch<Node<K, V, IC, LC>>> =
					Owned::new(HybridLatch::new(Node::Internal(InternalNode::new())));

				match root_guard_x.as_mut() {
					Node::Internal(root_internal_node) => {
						// Root is an internal node that needs splitting

						// Don't split if too small (need at least 3 keys to split)
						if root_internal_node.len <= 2 {
							return Ok(());
						}

						// Choose the middle position for the split
						let split_pos = root_internal_node.len / 2;
						let split_key = root_internal_node
							.key_at(split_pos)
							.expect("split position must be within node bounds")
							.clone();

						// Allocate the new right sibling (also internal)
						let mut new_right_node_owned =
							Owned::new(HybridLatch::new(Node::Internal(InternalNode::new())));

						// Perform the actual split - moves entries after split_pos to right
						{
							let new_right_node =
								new_right_node_owned.as_mut().as_mut().as_internal_mut();
							root_internal_node.split(new_right_node, split_pos);
						}

						// Create atomic pointers for the new tree structure
						// Old root becomes the left child of new root
						let old_root_edge = Atomic::from(tree_guard_x.load(Ordering::Acquire, eg));
						let new_right_node_edge =
							Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(new_right_node_owned);

						// Set up the new root: split_key separates left and right
						{
							let new_root = new_root_owned.as_mut().as_mut().as_internal_mut();
							new_root.insert(split_key, old_root_edge);
							new_root.upper_edge = Some(new_right_node_edge);
						}
					}
					Node::Leaf(root_leaf_node) => {
						// Root is a leaf that needs splitting (tree is growing from height 1 to 2)

						// Don't split if too small
						if root_leaf_node.len <= 2 {
							return Ok(());
						}

						// Choose the middle position for the split
						let split_pos = root_leaf_node.len / 2;
						let split_key = root_leaf_node
							.key_at(split_pos)
							.expect("split position must be within node bounds")
							.clone();

						// Allocate the new right sibling (also a leaf)
						let mut new_right_node_owned =
							Owned::new(HybridLatch::new(Node::Leaf(LeafNode::new())));

						// Perform the split
						{
							let new_right_node =
								new_right_node_owned.as_mut().as_mut().as_leaf_mut();
							root_leaf_node.split(new_right_node, split_pos);
						}

						// Create atomic pointers
						let old_root_edge = Atomic::from(tree_guard_x.load(Ordering::Acquire, eg));
						let new_right_node_edge =
							Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(new_right_node_owned);

						// Set up the new root
						{
							let new_root = new_root_owned.as_mut().as_mut().as_internal_mut();
							new_root.insert(split_key, old_root_edge);
							new_root.upper_edge = Some(new_right_node_edge);
						}
					}
				}

				// Install the new root
				let new_root_node_edge =
					Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(new_root_owned);
				*tree_guard_x = new_root_node_edge;

				// Increment tree height
				self.height.fetch_add(1, Ordering::Relaxed);
			}

			// ===================================================================
			// Case 2: Splitting a non-root node
			// ===================================================================
			ParentHandler::Parent {
				parent_guard,
				pos,
			} => {
				// Check if parent has space for another child pointer
				if parent_guard.as_internal().has_space() {
					// Parent has space - we can split the child

					// Get the child (needle) through lock coupling
					let swip = parent_guard.as_internal().edge_at(pos)?;
					let target_guard = GenericTree::lock_coupling(&parent_guard, swip, eg)?;

					// Verify that the target we found is actually the needle
					// (Tree structure might have changed during our traversal)
					let target_latch = target_guard.latch() as *const _;
					let needle_latch = needle.latch() as *const _;
					if target_latch != needle_latch {
						// The tree structure has changed - return Reclaimed so we re-seek
						return Err(error::Error::Reclaimed);
					}

					// Upgrade both guards to exclusive for modification
					let mut parent_guard_x = parent_guard.to_exclusive()?;
					let mut target_guard_x = target_guard.to_exclusive()?;

					match target_guard_x.as_mut() {
						Node::Internal(left_internal) => {
							// Splitting an internal node

							// Don't split if too small
							if left_internal.len <= 2 {
								return Ok(());
							}

							// Choose split position
							let split_pos = left_internal.len / 2;
							let split_key = left_internal
								.key_at(split_pos)
								.expect("split position must be within node bounds")
								.clone();

							// Allocate the new right sibling
							let mut new_right_node_owned =
								Owned::new(HybridLatch::new(Node::Internal(InternalNode::new())));

							// Perform the split
							{
								let new_right_node =
									new_right_node_owned.as_mut().as_mut().as_internal_mut();
								left_internal.split(new_right_node, split_pos);
							}

							// Create atomic pointer for right node
							let new_right_node_edge =
								Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(
									new_right_node_owned,
								);

							let parent_internal = parent_guard_x.as_internal_mut();

							// Insert the new separator key and right child into parent
							// After split: left node has lower keys, right node has higher keys.
							// Left node stays at current position, right node is inserted after.
							if pos == parent_internal.len {
								// Node was at upper_edge - it becomes the left.
								// Insert split key and make right the new upper_edge.
								let left_edge = parent_internal
									.upper_edge
									.replace(new_right_node_edge)
									.expect("internal node upper_edge must be set before split");
								parent_internal.insert(split_key, left_edge);
							} else {
								// Node was at edges[pos] - keep it there (it's now the left).
								// Insert split key with right node after it.
								parent_internal.insert_after(pos, split_key, new_right_node_edge);
							}
						}
						Node::Leaf(left_leaf) => {
							// Splitting a leaf node

							// Don't split if too small
							if left_leaf.len <= 2 {
								return Ok(());
							}

							// Choose split position
							let split_pos = left_leaf.len / 2;
							let split_key = left_leaf
								.key_at(split_pos)
								.expect("split position must be within node bounds")
								.clone();

							// Allocate the new right sibling
							let mut new_right_node_owned =
								Owned::new(HybridLatch::new(Node::Leaf(LeafNode::new())));

							// Perform the split
							{
								let new_right_node =
									new_right_node_owned.as_mut().as_mut().as_leaf_mut();
								left_leaf.split(new_right_node, split_pos);
							}

							// Create atomic pointer for right node
							let new_right_node_edge =
								Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(
									new_right_node_owned,
								);

							let parent_internal = parent_guard_x.as_internal_mut();

							// Insert the new separator and right child into parent
							// After split: left node has lower keys, right node has higher keys.
							// Left node stays at current position, right node is inserted after.
							if pos == parent_internal.len {
								// Node was at upper_edge - it becomes the left.
								// Insert split key and make right the new upper_edge.
								let left_edge = parent_internal
									.upper_edge
									.replace(new_right_node_edge)
									.expect("internal node upper_edge must be set before split");
								parent_internal.insert(split_key, left_edge);
							} else {
								// Node was at edges[pos] - keep it there (it's now the left).
								// Insert split key with right node after it.
								parent_internal.insert_after(pos, split_key, new_right_node_edge);
							}
						}
					}
				} else {
					// Parent is full - must split it first (recursive)
					self.try_split(&parent_guard, eg)?;

					// After splitting parent, the tree structure changed.
					// Return Reclaimed so caller goes back to seek_exact and finds the right leaf.
					return Err(error::Error::Reclaimed);
				}
			}
		}

		Ok(())
	}

	// -----------------------------------------------------------------------
	// Node Merging
	// -----------------------------------------------------------------------

	/// Attempts to merge an underfull node with a sibling.
	///
	/// This is called when a node falls below the minimum occupancy threshold
	/// after a deletion. Merging combines two nodes to maintain B+ tree balance.
	///
	/// # Algorithm Overview
	///
	/// 1. Find the underfull node's parent
	/// 2. Try to merge with the **left sibling** first (if exists)
	/// 3. If that fails, try to merge with the **right sibling**
	/// 4. After merge, recursively check if parent needs merging
	///
	/// ```text
	/// Before merge (target is underfull):
	///   Parent: [K1, ptr1] [K2, ptr2] [K3, ptr3]
	///                 │          │
	///                 ▼          ▼
	///   Left: [a,b,c]     Target: [x]  <- UNDERFULL
	///
	/// After merge (target absorbed into left):
	///   Parent: [K1, ptr1] [K3, ptr3]
	///                 │
	///                 ▼
	///   Left: [a,b,c,x]   (Target's node is freed)
	/// ```
	///
	/// # Memory Reclamation
	///
	/// The merged (absorbed) node is scheduled for deferred destruction via
	/// `eg.defer_destroy()`. This ensures concurrent readers can still access
	/// the node until they leave their epoch.
	///
	/// # Parameters
	///
	/// - `needle`: The underfull node to merge
	/// - `eg`: Epoch guard for memory safety
	///
	/// # Returns
	///
	/// - `Ok(true)`: Merge succeeded
	/// - `Ok(false)`: Merge not needed or couldn't be done (node not underfull, or is root)
	/// - `Err(...)`: Validation failed
	pub(crate) fn try_merge<'t, 'g, 'e>(
		&'t self,
		needle: &OptimisticGuard<'g, Node<K, V, IC, LC>>,
		eg: &'e epoch::Guard,
	) -> error::Result<bool>
	where
		K: Ord,
	{
		// Find the needle's parent
		let parent_handler = self.find_parent(needle, eg)?;

		match parent_handler {
			ParentHandler::Root {
				tree_guard: _,
			} => {
				// Root node - can't merge (no sibling)
				// Note: We could potentially shrink the tree here if root is underfull
				// and has only one child, but that's not implemented
				Ok(false)
			}
			ParentHandler::Parent {
				mut parent_guard,
				pos,
			} => {
				let parent_len = parent_guard.as_internal().len;

				// Re-acquire the target through lock coupling
				let swip = parent_guard.as_internal().edge_at(pos)?;
				let mut target_guard = GenericTree::lock_coupling(&parent_guard, swip, eg)?;

				// Verify the node is actually underfull (might have changed)
				if !target_guard.is_underfull() {
					target_guard.recheck()?;
					return Ok(false);
				}

				// ===============================================================
				// Try 1: Merge with LEFT sibling
				// ===============================================================
				let merge_succeeded = if parent_len > 1 && pos > 0 {
					// Left sibling exists - try to merge

					// Get the left sibling
					let l_swip = parent_guard.as_internal().edge_at(pos - 1)?;
					let left_guard = GenericTree::lock_coupling(&parent_guard, l_swip, eg)?;

					// Check if merge is possible (combined size fits in one node)
					if !left_guard.can_merge_with(&target_guard) {
						// Can't merge - nodes too big combined
						left_guard.recheck()?;
						target_guard.recheck()?;
						false
					} else {
						// Upgrade all guards to exclusive for modification
						let mut parent_guard_x = parent_guard.to_exclusive()?;
						let mut target_guard_x = target_guard.to_exclusive()?;
						let mut left_guard_x = left_guard.to_exclusive()?;

						match target_guard_x.as_mut() {
							Node::Leaf(ref mut target_leaf) => {
								// Merging two leaf nodes
								assert!(left_guard_x.is_leaf());

								// Attempt the merge (left absorbs target)
								if !left_guard_x.as_leaf_mut().merge(target_leaf) {
									// Merge failed (shouldn't happen after can_merge_with check)
									parent_guard = parent_guard_x.unlock();
									target_guard = target_guard_x.unlock();
									false
								} else {
									// Merge succeeded - update parent
									let parent_internal = parent_guard_x.as_internal_mut();

									// Remove the separator key and update pointers
									if pos == parent_len {
										// Target was at upper_edge
										// Remove separator, left becomes the new upper_edge
										let (_, left_edge) = parent_internal.remove_at(pos - 1);
										let dropped_edge = parent_internal
											.upper_edge
											.replace(left_edge)
											.expect("parent upper_edge must exist during merge");

										// Schedule the old target for deferred destruction
										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											// SAFETY: The epoch guard ensures safe reclamation
											unsafe { eg.defer_destroy(shared) };
										}
									} else {
										// Target was at edges[pos]
										let (_, left_edge) = parent_internal.remove_at(pos - 1);
										let dropped_edge = std::mem::replace(
											&mut parent_internal.edges[(pos - 1) as usize],
											left_edge,
										);

										// Schedule deferred destruction
										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											unsafe { eg.defer_destroy(shared) };
										}
									}

									// Unlock and continue
									parent_guard = parent_guard_x.unlock();
									target_guard = target_guard_x.unlock();
									true
								}
							}
							Node::Internal(target_internal) => {
								// Merging two internal nodes
								assert!(!left_guard_x.is_leaf());

								if !left_guard_x.as_internal_mut().merge(target_internal) {
									parent_guard = parent_guard_x.unlock();
									target_guard = target_guard_x.unlock();
									false
								} else {
									let parent_internal = parent_guard_x.as_internal_mut();

									if pos == parent_len {
										let (_, left_edge) = parent_internal.remove_at(pos - 1);
										let dropped_edge = parent_internal
											.upper_edge
											.replace(left_edge)
											.expect("parent upper_edge must exist during merge");

										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											unsafe { eg.defer_destroy(shared) };
										}
									} else {
										let (_, left_edge) = parent_internal.remove_at(pos - 1);
										let dropped_edge = std::mem::replace(
											&mut parent_internal.edges[(pos - 1) as usize],
											left_edge,
										);

										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											unsafe { eg.defer_destroy(shared) };
										}
									}

									parent_guard = parent_guard_x.unlock();
									target_guard = target_guard_x.unlock();
									true
								}
							}
						}
					}
				} else {
					// No left sibling (pos == 0 or only one child in parent)
					false
				};

				// ===============================================================
				// Try 2: Merge with RIGHT sibling (if left merge failed)
				// ===============================================================
				let merge_succeeded = if !merge_succeeded
					&& parent_len > 0
					&& (pos + 1) <= parent_len
				{
					// Right sibling exists - try to merge

					let r_swip = parent_guard.as_internal().edge_at(pos + 1)?;
					let right_guard = GenericTree::lock_coupling(&parent_guard, r_swip, eg)?;

					if !right_guard.can_merge_with(&target_guard) {
						// Can't merge with right sibling either
						right_guard.recheck()?;
						target_guard.recheck()?;
						false
					} else {
						// Upgrade to exclusive
						let mut parent_guard_x = parent_guard.to_exclusive()?;
						let mut target_guard_x = target_guard.to_exclusive()?;
						let mut right_guard_x = right_guard.to_exclusive()?;

						match target_guard_x.as_mut() {
							Node::Leaf(ref mut target_leaf) => {
								// Merging leaf nodes (target absorbs right)
								assert!(right_guard_x.is_leaf());

								if !target_leaf.merge(right_guard_x.as_leaf_mut()) {
									parent_guard = parent_guard_x.unlock();
									let _ = target_guard_x.unlock();
									false
								} else {
									let parent_internal = parent_guard_x.as_internal_mut();

									// Remove separator and schedule right node for destruction
									if pos + 1 == parent_len {
										let (_, left_edge) = parent_internal.remove_at(pos);
										let dropped_edge = parent_internal
											.upper_edge
											.replace(left_edge)
											.expect("parent upper_edge must exist during merge");

										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											unsafe { eg.defer_destroy(shared) };
										}
									} else {
										let (_, left_edge) = parent_internal.remove_at(pos);
										let dropped_edge = std::mem::replace(
											&mut parent_internal.edges[pos as usize],
											left_edge,
										);

										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											unsafe { eg.defer_destroy(shared) };
										}
									}

									parent_guard = parent_guard_x.unlock();
									let _ = target_guard_x.unlock();
									true
								}
							}
							Node::Internal(target_internal) => {
								// Merging internal nodes
								assert!(!right_guard_x.is_leaf());

								if !target_internal.merge(right_guard_x.as_internal_mut()) {
									parent_guard = parent_guard_x.unlock();
									let _ = target_guard_x.unlock();
									false
								} else {
									let parent_internal = parent_guard_x.as_internal_mut();

									if pos + 1 == parent_len {
										let (_, left_edge) = parent_internal.remove_at(pos);
										let dropped_edge = parent_internal
											.upper_edge
											.replace(left_edge)
											.expect("parent upper_edge must exist during merge");

										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											unsafe { eg.defer_destroy(shared) };
										}
									} else {
										let (_, left_edge) = parent_internal.remove_at(pos);
										let dropped_edge = std::mem::replace(
											&mut parent_internal.edges[pos as usize],
											left_edge,
										);

										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											unsafe { eg.defer_destroy(shared) };
										}
									}

									parent_guard = parent_guard_x.unlock();
									let _ = target_guard_x.unlock();
									true
								}
							}
						}
					}
				} else {
					merge_succeeded
				};

				// ===============================================================
				// Recursive: Check if parent also needs merging
				// ===============================================================
				let parent_merge = || {
					if parent_guard.is_underfull() {
						parent_guard.recheck()?;
						let _ = self.try_merge(&parent_guard, eg)?;
					}
					error::Result::Ok(())
				};

				// Best-effort parent merge (ignore errors)
				let _ = parent_merge();

				Ok(merge_succeeded)
			}
		}
	}

	// -----------------------------------------------------------------------
	// Iterators
	// -----------------------------------------------------------------------

	/// Returns a shared (read-only) iterator over the tree entries.
	///
	/// The iterator acquires shared locks on leaf nodes, allowing concurrent
	/// readers but blocking writers on the current leaf.
	///
	/// # Example
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
	/// while let Some((k, v)) = iter.next() {
	///     println!("{}: {}", k, v);
	/// }
	/// ```
	pub fn raw_iter(&self) -> iter::RawSharedIter<'_, K, V, IC, LC>
	where
		K: Ord,
	{
		iter::RawSharedIter::new(self)
	}

	/// Returns an exclusive (read-write) iterator over the tree entries.
	///
	/// The iterator acquires exclusive locks on leaf nodes, allowing
	/// modifications (insert, update, remove) during iteration.
	///
	/// # Example
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
	/// // Modify values during iteration
	/// while let Some((k, v)) = iter.next() {
	///     *v *= 2; // Double each value
	/// }
	/// ```
	pub fn raw_iter_mut(&self) -> iter::RawExclusiveIter<'_, K, V, IC, LC>
	where
		K: Ord,
	{
		iter::RawExclusiveIter::new(self)
	}

	// -----------------------------------------------------------------------
	// Size Operations
	// -----------------------------------------------------------------------

	/// Returns the number of key-value pairs in the tree.
	///
	/// **Note**: This is an O(n) operation that iterates through all entries.
	/// For large trees, consider maintaining a separate count if you need
	/// frequent size checks.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// assert_eq!(tree.len(), 0);
	///
	/// tree.insert(1, "one");
	/// tree.insert(2, "two");
	/// assert_eq!(tree.len(), 2);
	/// ```
	pub fn len(&self) -> usize {
		let mut count = 0usize;
		let mut iter = self.raw_iter();
		iter.seek_to_first();

		// Count all entries by iterating through the tree
		while iter.next().is_some() {
			count += 1;
		}
		count
	}

	/// Returns `true` if the tree contains no entries.
	///
	/// This is an O(1) operation that checks if the first leaf has any entries.
	pub fn is_empty(&self) -> bool {
		let mut iter = self.raw_iter();
		iter.seek_to_first();
		iter.next().is_none()
	}
}

// ===========================================================================
// Node Types
// ===========================================================================

/// A node in the B+ tree, either internal (index) or leaf (data).
///
/// The tree is made up of two types of nodes:
/// - **Internal nodes**: Store keys and child pointers for navigation
/// - **Leaf nodes**: Store actual key-value pairs
///
/// All leaves are at the same depth, and internal nodes contain only routing
/// information (no values).
pub(crate) enum Node<K, V, const IC: usize, const LC: usize> {
	/// An internal (index) node containing keys and child pointers.
	Internal(InternalNode<K, V, IC, LC>),
	/// A leaf node containing key-value pairs.
	Leaf(LeafNode<K, V, LC>),
}

impl<K: fmt::Debug, V: fmt::Debug, const IC: usize, const LC: usize> fmt::Debug
	for Node<K, V, IC, LC>
{
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Node::Internal(ref internal) => f.debug_tuple("Internal").field(internal).finish(),
			Node::Leaf(ref leaf) => f.debug_tuple("Leaf").field(leaf).finish(),
		}
	}
}

impl<K, V, const IC: usize, const LC: usize> Node<K, V, IC, LC> {
	/// Returns `true` if this is a leaf node.
	#[inline]
	pub(crate) fn is_leaf(&self) -> bool {
		matches!(self, Node::Leaf(_))
	}

	/// Returns a reference to the inner leaf node, if this is a leaf.
	///
	/// Returns `None` if this is an internal node.
	#[inline]
	#[allow(dead_code)]
	pub(crate) fn try_as_leaf(&self) -> Option<&LeafNode<K, V, LC>> {
		match self {
			Node::Leaf(ref leaf) => Some(leaf),
			Node::Internal(_) => None,
		}
	}

	/// Returns a reference to the inner leaf node.
	///
	/// # Panics
	///
	/// Panics if called on an internal node. Use `try_as_leaf()` for
	/// a fallible alternative.
	#[inline]
	pub(crate) fn as_leaf(&self) -> &LeafNode<K, V, LC> {
		match self {
			Node::Leaf(ref leaf) => leaf,
			Node::Internal(_) => {
				unreachable!(
					"as_leaf() called on internal node - this indicates a tree traversal bug"
				)
			}
		}
	}

	/// Returns a mutable reference to the inner leaf node, if this is a leaf.
	///
	/// Returns `None` if this is an internal node.
	#[inline]
	#[allow(dead_code)]
	pub(crate) fn try_as_leaf_mut(&mut self) -> Option<&mut LeafNode<K, V, LC>> {
		match self {
			Node::Leaf(ref mut leaf) => Some(leaf),
			Node::Internal(_) => None,
		}
	}

	/// Returns a mutable reference to the inner leaf node.
	///
	/// # Panics
	///
	/// Panics if called on an internal node. Use `try_as_leaf_mut()` for
	/// a fallible alternative.
	#[inline]
	pub(crate) fn as_leaf_mut(&mut self) -> &mut LeafNode<K, V, LC> {
		match self {
			Node::Leaf(ref mut leaf) => leaf,
			Node::Internal(_) => {
				unreachable!(
					"as_leaf_mut() called on internal node - this indicates a tree traversal bug"
				)
			}
		}
	}

	/// Returns a reference to the inner internal node, if this is an internal node.
	///
	/// Returns `None` if this is a leaf node.
	#[inline]
	#[allow(dead_code)]
	pub(crate) fn try_as_internal(&self) -> Option<&InternalNode<K, V, IC, LC>> {
		match self {
			Node::Internal(ref internal) => Some(internal),
			Node::Leaf(_) => None,
		}
	}

	/// Returns a reference to the inner internal node.
	///
	/// # Panics
	///
	/// Panics if called on a leaf node. Use `try_as_internal()` for
	/// a fallible alternative.
	#[inline]
	pub(crate) fn as_internal(&self) -> &InternalNode<K, V, IC, LC> {
		match self {
			Node::Internal(ref internal) => internal,
			Node::Leaf(_) => {
				unreachable!(
					"as_internal() called on leaf node - this indicates a tree traversal bug"
				)
			}
		}
	}

	/// Returns a mutable reference to the inner internal node, if this is an internal node.
	///
	/// Returns `None` if this is a leaf node.
	#[inline]
	#[allow(dead_code)]
	pub(crate) fn try_as_internal_mut(&mut self) -> Option<&mut InternalNode<K, V, IC, LC>> {
		match self {
			Node::Internal(ref mut internal) => Some(internal),
			Node::Leaf(_) => None,
		}
	}

	/// Returns a mutable reference to the inner internal node.
	///
	/// # Panics
	///
	/// Panics if called on a leaf node. Use `try_as_internal_mut()` for
	/// a fallible alternative.
	#[inline]
	pub(crate) fn as_internal_mut(&mut self) -> &mut InternalNode<K, V, IC, LC> {
		match self {
			Node::Internal(ref mut internal) => internal,
			Node::Leaf(_) => {
				unreachable!(
					"as_internal_mut() called on leaf node - this indicates a tree traversal bug"
				)
			}
		}
	}

	/// Returns the keys stored in this node (for testing).
	#[cfg(test)]
	#[inline]
	pub(crate) fn keys(&self) -> &[K] {
		match self {
			Node::Internal(ref internal) => &internal.keys,
			Node::Leaf(ref leaf) => &leaf.keys,
		}
	}

	/// Returns a sample key that can be used to find this node in the tree.
	///
	/// The sample key is set during splits and is guaranteed to route to this
	/// node when searched from the root. Used by `find_parent()` to relocate
	/// a node after an operation.
	#[inline]
	pub(crate) fn sample_key(&self) -> Option<&K> {
		match self {
			Node::Internal(ref internal) => internal.sample_key.as_ref(),
			Node::Leaf(ref leaf) => leaf.sample_key.as_ref(),
		}
	}

	/// Returns `true` if the node is below the minimum occupancy threshold.
	///
	/// Underfull nodes should be merged with siblings to maintain B+ tree
	/// balance properties. The threshold is 40% of capacity.
	#[inline]
	pub(crate) fn is_underfull(&self) -> bool {
		match self {
			Node::Internal(ref internal) => internal.is_underfull(),
			Node::Leaf(ref leaf) => leaf.is_underfull(),
		}
	}

	/// Checks if this node can be merged with another node.
	///
	/// Two nodes can merge if their combined size fits within the capacity.
	/// For internal nodes, we add 1 for the separator key that will be added.
	///
	/// # Returns
	///
	/// - `true` if merge is possible
	/// - `false` if nodes are different types or combined size exceeds capacity
	#[inline]
	pub(crate) fn can_merge_with(&self, other: &Self) -> bool {
		match self {
			Node::Internal(ref internal) => match other {
				Node::Internal(ref other) => {
					// +1 for the separator key that gets added during merge
					((internal.len + 1 + other.len) as usize) < IC
				}
				_ => false, // Can't merge internal with leaf
			},
			Node::Leaf(ref leaf) => match other {
				Node::Leaf(ref other) => {
					// Leaf merge doesn't add a separator key
					((leaf.len + other.len) as usize) < LC
				}
				_ => false, // Can't merge leaf with internal
			},
		}
	}
}

// ===========================================================================
// Leaf Node
// ===========================================================================

/// A leaf node in the B+ tree, storing actual key-value pairs.
///
/// Leaf nodes are where the data lives. They store keys and values in sorted
/// order, allowing efficient range scans when traversing from leaf to leaf.
///
/// # Fence Keys
///
/// Each leaf maintains `lower_fence` and `upper_fence` keys that define the
/// key range this leaf is responsible for:
/// - `lower_fence < key <= upper_fence` for keys in this leaf
/// - `lower_fence = None` means this is the leftmost leaf
/// - `upper_fence = None` means this is the rightmost leaf
///
/// Fence keys are crucial for optimistic concurrency:
/// - They allow quick bounds checking during optimistic reads
/// - They help detect if a node was split/merged during traversal
/// - They enable efficient iterator recovery after validation failures
///
/// # Sample Key
///
/// The `sample_key` is a key known to be in (or route to) this leaf. It's
/// used by `find_parent()` to relocate this leaf in the tree after structural
/// changes. Set during splits.
pub(crate) struct LeafNode<K, V, const LC: usize> {
	/// Number of key-value pairs in this leaf.
	pub(crate) len: u16,
	/// Sorted array of keys.
	pub(crate) keys: SmallVec<[K; LC]>,
	/// Values corresponding to keys (same index).
	pub(crate) values: SmallVec<[V; LC]>,
	/// Exclusive lower bound - keys in this leaf are > lower_fence.
	/// None means this is the leftmost leaf (no lower bound).
	pub(crate) lower_fence: Option<K>,
	/// Inclusive upper bound - keys in this leaf are <= upper_fence.
	/// None means this is the rightmost leaf (no upper bound).
	pub(crate) upper_fence: Option<K>,
	/// A key that routes to this leaf, used for relocation after splits.
	pub(crate) sample_key: Option<K>,
}

impl<K: fmt::Debug, V: fmt::Debug, const LC: usize> fmt::Debug for LeafNode<K, V, LC> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("LeafNode")
			.field("len", &self.len)
			.field("keys", &self.keys)
			.field("values", &self.values)
			.field("lower_fence", &self.lower_fence)
			.field("upper_fence", &self.upper_fence)
			.field("sample_key", &self.sample_key)
			.finish()
	}
}

impl<K, V, const LC: usize> LeafNode<K, V, LC> {
	/// Creates a new, empty leaf node.
	pub fn new() -> LeafNode<K, V, LC> {
		LeafNode {
			len: 0,
			keys: smallvec![],
			values: smallvec![],
			lower_fence: None,
			upper_fence: None,
			sample_key: None,
		}
	}

	/// Binary search for a key, returning position and whether it's an exact match.
	///
	/// This is the core lookup operation for leaves. It first checks fence keys
	/// for a quick bounds check, then performs binary search.
	///
	/// # Returns
	///
	/// `(position, exact_match)` where:
	/// - `position`: Index where the key is or should be inserted
	/// - `exact_match`: `true` if `keys[position] == key`
	///
	/// # Concurrency Safety
	///
	/// Uses safe bounds checking to handle concurrent access. Under optimistic
	/// locking, `self.len` may be inconsistent with `self.keys.len()` during
	/// concurrent modifications. The caller's recheck will detect this, but
	/// we must not cause undefined behavior in the meantime.
	#[inline]
	pub(crate) fn lower_bound<Q>(&self, key: &Q) -> (u16, bool)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		// Quick check against fence keys to potentially avoid binary search
		if self.lower_fence().map(|fk| key < fk.borrow()).unwrap_or(false) {
			// Key is below our range - would be at position 0
			return (0, false);
		}

		if let Some(fk) = self.upper_fence() {
			if key > fk.borrow() {
				// Key is above our range - would be at position len
				return (self.len, false);
			}
		}

		// Use actual keys length for safe bounds - handles concurrent modifications
		let keys_len = self.keys.len() as u16;
		let mut lower = 0;
		let mut upper = self.len.min(keys_len);

		while lower < upper {
			let mid = ((upper - lower) / 2) + lower;

			// Safe bounds check - concurrent modifications may cause len > keys.len()
			let Some(mid_key) = self.keys.get(mid as usize) else {
				// Index out of bounds due to concurrent modification - return conservative result
				return (lower, false);
			};

			if key < mid_key.borrow() {
				upper = mid;
			} else if key > mid_key.borrow() {
				lower = mid + 1;
			} else {
				// Exact match found
				return (mid, true);
			}
		}

		// No exact match - lower is the insertion point
		(lower, false)
	}

	/// Returns the lower fence key, if any.
	#[inline]
	pub(crate) fn lower_fence(&self) -> Option<&K> {
		self.lower_fence.as_ref()
	}

	/// Returns the upper fence key, if any.
	#[inline]
	pub(crate) fn upper_fence(&self) -> Option<&K> {
		self.upper_fence.as_ref()
	}

	/// Returns a reference to the value at the given position.
	///
	/// # Concurrency Safety
	///
	/// Uses safe bounds checking - returns `Err(Unwind)` if position is invalid
	/// due to concurrent modification, triggering a retry.
	#[inline]
	pub(crate) fn value_at(&self, pos: u16) -> error::Result<&V> {
		self.values.get(pos as usize).ok_or(error::Error::Unwind)
	}

	/// Returns a reference to the key at the given position.
	///
	/// # Concurrency Safety
	///
	/// Uses safe bounds checking - returns `Err(Unwind)` if position is invalid.
	#[inline]
	pub(crate) fn key_at(&self, pos: u16) -> error::Result<&K> {
		self.keys.get(pos as usize).ok_or(error::Error::Unwind)
	}

	/// Returns references to the key and value at the given position.
	#[inline]
	pub(crate) fn kv_at(&self, pos: u16) -> error::Result<(&K, &V)> {
		Ok((self.key_at(pos)?, self.value_at(pos)?))
	}

	/// Returns references to the key (immutable) and value (mutable) at position.
	///
	/// # Concurrency Safety
	///
	/// Uses safe bounds checking - returns `Err(Unwind)` if position is invalid.
	#[inline]
	pub(crate) fn kv_at_mut(&mut self, pos: u16) -> error::Result<(&K, &mut V)> {
		let pos = pos as usize;
		if pos >= self.keys.len() || pos >= self.values.len() {
			return Err(error::Error::Unwind);
		}
		// SAFETY: We just checked bounds above
		Ok(unsafe { (self.keys.get_unchecked(pos), self.values.get_unchecked_mut(pos)) })
	}

	/// Returns `true` if there's room for another entry.
	#[inline]
	pub(crate) fn has_space(&self) -> bool {
		(self.len as usize) < LC
	}

	/// Returns `true` if the node is below minimum occupancy (40% of capacity).
	#[inline]
	pub(crate) fn is_underfull(&self) -> bool {
		(self.len as usize) < (LC as f32 * 0.4) as usize
	}

	/// Inserts a key-value pair at the specified position.
	///
	/// # Returns
	///
	/// - `Some(pos)` if insertion succeeded
	/// - `None` if the node is full
	pub(crate) fn insert_at(&mut self, pos: u16, key: K, value: V) -> Option<u16> {
		if !self.has_space() {
			return None;
		}

		// Insert into both arrays at the same position
		self.keys.insert(pos as usize, key);
		self.values.insert(pos as usize, value);
		self.len += 1;

		Some(pos)
	}

	/// Removes and returns the key-value pair at the specified position.
	pub(crate) fn remove_at(&mut self, pos: u16) -> (K, V) {
		self.len -= 1;
		let key = self.keys.remove(pos as usize);
		let value = self.values.remove(pos as usize);

		(key, value)
	}

	/// Checks if a key falls within this leaf's fence boundaries.
	///
	/// Used by iterators to check if they can reuse the current leaf
	/// or need to seek to a new one.
	///
	/// # Boundary Rules
	///
	/// - `lower_fence < key <= upper_fence` for middle leaves
	/// - `key <= upper_fence` for leftmost leaf (no lower_fence)
	/// - `key > lower_fence` for rightmost leaf (no upper_fence)
	/// - All keys valid for single-node tree (both fences None)
	#[inline]
	pub(crate) fn within_bounds<Q>(&self, key: &Q) -> bool
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		match (self.lower_fence().map(Borrow::borrow), self.upper_fence().map(Borrow::borrow)) {
			(Some(lf), Some(uf)) => key > lf && key <= uf,
			(Some(lf), None) => key > lf,
			(None, Some(uf)) => key <= uf,
			(None, None) => true,
		}
	}
}

impl<K: Clone, V, const LC: usize> LeafNode<K, V, LC> {
	/// Splits this leaf node, moving entries after `split_pos` to `right`.
	///
	/// After split:
	/// - `self` (left) contains keys `[0, split_pos]`
	/// - `right` contains keys `[split_pos + 1, len)`
	/// - `split_key = keys[split_pos]` becomes the separator
	///
	/// # Fence Key Updates
	///
	/// ```text
	/// Before: self.fences = (lower, upper)
	/// After:
	///   self.fences  = (lower, split_key)      // Left gets lower range
	///   right.fences = (split_key, upper)      // Right gets upper range
	/// ```
	///
	/// # Parameters
	///
	/// - `right`: An empty leaf node to receive the upper half
	/// - `split_pos`: Position of the key that becomes the separator
	pub(crate) fn split(&mut self, right: &mut LeafNode<K, V, LC>, split_pos: u16) {
		// The key at split_pos becomes the boundary between left and right
		let split_key =
			self.key_at(split_pos).expect("split position must be within node bounds").clone();

		// Update fence keys:
		// - Right's lower fence is the split key (exclusive)
		// - Right inherits our upper fence
		// - Our new upper fence is the split key (inclusive)
		right.lower_fence = Some(split_key.clone());
		right.upper_fence = self.upper_fence.clone();
		self.upper_fence = Some(split_key);

		// Move entries after split_pos to the right node
		assert!(right.keys.is_empty());
		assert!(right.values.is_empty());
		right.keys.extend(self.keys.drain((split_pos + 1) as usize..));
		right.values.extend(self.values.drain((split_pos + 1) as usize..));

		// Set sample keys for node relocation
		// Use first key of each node (guaranteed to route to that node)
		self.sample_key = Some(self.keys[0].clone());
		right.sample_key = Some(right.keys[0].clone());

		// Update lengths
		right.len = right.keys.len() as u16;
		self.len = self.keys.len() as u16;
	}

	/// Merges the `right` leaf into `self`.
	///
	/// All entries from `right` are appended to `self`, and `right` is emptied.
	///
	/// # Returns
	///
	/// - `true` if merge succeeded
	/// - `false` if combined size would exceed capacity
	pub(crate) fn merge(&mut self, right: &mut LeafNode<K, V, LC>) -> bool {
		// Check if combined entries fit
		if (self.len + right.len) as usize > LC {
			return false;
		}

		// Mark right as empty
		right.len = 0;

		// Inherit right's upper fence (we now cover its range too)
		self.upper_fence = right.upper_fence.take();

		// Move all entries from right to self
		self.keys.extend(right.keys.drain(..));
		self.values.extend(right.values.drain(..));

		// Update sample key to one from the absorbed node
		// (useful if our original sample key was at the boundary)
		self.sample_key = right.sample_key.take();

		// Update length
		self.len = self.keys.len() as u16;
		true
	}
}

// ===========================================================================
// Internal Node
// ===========================================================================

/// An internal (index) node in the B+ tree, storing keys and child pointers.
///
/// Internal nodes don't store values - they only contain separator keys and
/// pointers to child nodes for navigation.
///
/// # Structure
///
/// ```text
/// keys:       [K0,  K1,  K2,  ...  K(n-1)]
/// edges:      [E0,  E1,  E2,  ...  E(n-1)]  upper_edge
///               │    │    │         │            │
///               ▼    ▼    ▼         ▼            ▼
///            child0 child1 child2 child(n-1) child(n)
///
/// Navigation: For key K, follow edge[i] where keys[i-1] <= K < keys[i]
///             (edge[0] for K < keys[0], upper_edge for K >= keys[n-1])
/// ```
///
/// # Invariants
///
/// - `len` = number of keys = number of edges (excluding upper_edge)
/// - `edges[i]` leads to children with keys < `keys[i]`
/// - `upper_edge` leads to children with keys >= `keys[len-1]`
/// - Keys are sorted in ascending order
///
/// # Fence Keys
///
/// Similar to leaf nodes, internal nodes have fence keys defining their
/// key range. These are used for optimistic validation and node relocation.
pub(crate) struct InternalNode<K, V, const IC: usize, const LC: usize> {
	/// Number of keys (and regular edges) in this node.
	pub(crate) len: u16,
	/// Separator keys, sorted in ascending order.
	pub(crate) keys: SmallVec<[K; IC]>,
	/// Child pointers corresponding to keys.
	/// `edges[i]` points to subtree with keys < `keys[i]`.
	pub(crate) edges: SmallVec<[Atomic<HybridLatch<Node<K, V, IC, LC>>>; IC]>,
	/// Rightmost child pointer, for keys >= last key.
	/// This is separate because we have N+1 children for N keys.
	pub(crate) upper_edge: Option<Atomic<HybridLatch<Node<K, V, IC, LC>>>>,
	/// Exclusive lower bound for keys routed through this node.
	pub(crate) lower_fence: Option<K>,
	/// Inclusive upper bound for keys routed through this node.
	pub(crate) upper_fence: Option<K>,
	/// Sample key for node relocation.
	pub(crate) sample_key: Option<K>,
}

impl<K: fmt::Debug, V, const IC: usize, const LC: usize> fmt::Debug for InternalNode<K, V, IC, LC> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("InternalNode")
			.field("len", &self.len)
			.field("keys", &self.keys)
			.field("edges", &self.edges)
			.field("upper_edge", &self.upper_edge)
			.field("lower_fence", &self.lower_fence)
			.field("upper_fence", &self.upper_fence)
			.field("sample_key", &self.sample_key)
			.finish()
	}
}

impl<K, V, const IC: usize, const LC: usize> InternalNode<K, V, IC, LC> {
	/// Creates a new, empty internal node.
	pub(crate) fn new() -> InternalNode<K, V, IC, LC> {
		InternalNode {
			len: 0,
			keys: smallvec![],
			edges: smallvec![],
			upper_edge: None,
			lower_fence: None,
			upper_fence: None,
			sample_key: None,
		}
	}

	/// Binary search for the child edge to follow for a given key.
	///
	/// Returns `(position, exact_match)` where:
	/// - `position`: The edge index to follow (use `edge_at(position)`)
	/// - `exact_match`: True if key exactly matches `keys[position]`
	///
	/// # Navigation Rules
	///
	/// - If `key < keys[0]`, return position 0 (leftmost child)
	/// - If `key >= keys[len-1]`, return position len (upper_edge)
	/// - Otherwise, find i where `keys[i-1] <= key < keys[i]`
	///
	/// # Concurrency Safety
	///
	/// Uses safe bounds checking to handle concurrent access.
	#[inline]
	pub(crate) fn lower_bound<Q>(&self, key: &Q) -> (u16, bool)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		// Quick fence check
		if self.lower_fence().map(|fk| key < fk.borrow()).unwrap_or(false) {
			return (0, false);
		}

		if let Some(fk) = self.upper_fence() {
			if key > fk.borrow() {
				return (self.len, false);
			}
		}

		// Use actual keys length for safe bounds - handles concurrent modifications
		let keys_len = self.keys.len() as u16;
		let mut lower = 0;
		let mut upper = self.len.min(keys_len);

		while lower < upper {
			let mid = ((upper - lower) / 2) + lower;

			// Safe bounds check - concurrent modifications may cause len > keys.len()
			let Some(mid_key) = self.keys.get(mid as usize) else {
				// Index out of bounds due to concurrent modification - return conservative result
				return (lower, false);
			};

			if key < mid_key.borrow() {
				upper = mid;
			} else if key > mid_key.borrow() {
				lower = mid + 1;
			} else {
				// Exact match on separator key
				return (mid, true);
			}
		}

		(lower, false)
	}

	/// Returns the lower fence key, if any.
	#[inline]
	pub(crate) fn lower_fence(&self) -> Option<&K> {
		self.lower_fence.as_ref()
	}

	/// Returns the upper fence key, if any.
	#[inline]
	pub(crate) fn upper_fence(&self) -> Option<&K> {
		self.upper_fence.as_ref()
	}

	/// Returns the child pointer at the given position.
	///
	/// - Positions `0..len` return `edges[pos]`
	/// - Position `len` returns `upper_edge`
	///
	/// # Errors
	///
	/// Returns `Error::Unwind` if `pos == len` but `upper_edge` is None,
	/// or if position is out of bounds due to concurrent modification.
	///
	/// # Concurrency Safety
	///
	/// Uses safe bounds checking to handle concurrent access.
	#[inline]
	pub(crate) fn edge_at(
		&self,
		pos: u16,
	) -> error::Result<&Atomic<HybridLatch<Node<K, V, IC, LC>>>> {
		if pos == self.len {
			// Rightmost child - use upper_edge
			if let Some(upper_edge) = self.upper_edge.as_ref() {
				Ok(upper_edge)
			} else {
				Err(error::Error::Unwind)
			}
		} else {
			// Regular child - use edges array with safe bounds check
			self.edges.get(pos as usize).ok_or(error::Error::Unwind)
		}
	}

	/// Returns the separator key at the given position.
	///
	/// # Concurrency Safety
	///
	/// Uses safe bounds checking - returns `Err(Unwind)` if position is invalid.
	#[inline]
	pub(crate) fn key_at(&self, pos: u16) -> error::Result<&K> {
		self.keys.get(pos as usize).ok_or(error::Error::Unwind)
	}

	/// Returns `true` if there's room for another key/edge pair.
	#[inline]
	pub(crate) fn has_space(&self) -> bool {
		(self.len as usize) < IC
	}

	/// Returns `true` if the node is below minimum occupancy (40%).
	#[inline]
	pub(crate) fn is_underfull(&self) -> bool {
		(self.len as usize) < (IC as f32 * 0.4) as usize
	}

	/// Inserts a key and its left child pointer.
	///
	/// Used during splits where the new separator key is inserted along
	/// with the pointer to the new left child.
	///
	/// # Returns
	///
	/// - `Some(pos)` if insertion succeeded
	/// - `None` if the node is full
	pub(crate) fn insert(
		&mut self,
		key: K,
		value: Atomic<HybridLatch<Node<K, V, IC, LC>>>,
	) -> Option<u16>
	where
		K: Ord,
	{
		let (pos, exact) = self.lower_bound(&key);

		if exact {
			// Key already exists - this shouldn't happen in normal B+ tree operations
			unimplemented!("upserts");
		} else {
			if !self.has_space() {
				return None;
			}

			// Insert key and edge at the found position
			self.keys.insert(pos as usize, key);
			self.edges.insert(pos as usize, value);
			self.len += 1;
		}
		Some(pos)
	}

	/// Removes the key and edge at the given position.
	pub(crate) fn remove_at(&mut self, pos: u16) -> (K, Atomic<HybridLatch<Node<K, V, IC, LC>>>) {
		let key = self.keys.remove(pos as usize);
		let edge = self.edges.remove(pos as usize);
		self.len -= 1;

		(key, edge)
	}

	/// Inserts a separator key and new right child after a node split.
	///
	/// After splitting a child at position `pos`, we need to insert:
	/// - The split key as a new separator
	/// - The right sibling as a new child
	///
	/// The left child remains at `edges[pos]`, and we insert:
	/// - `key` at `keys[pos]` (shifting existing keys right)
	/// - `edge` (right child) at `edges[pos+1]` (shifting existing edges right)
	///
	/// # Example
	///
	/// ```text
	/// Before: keys=[A, B], edges=[e0, e1], upper=e2
	///         Splitting child at e1
	///
	/// After:  keys=[A, split_key, B], edges=[e0, e1, new_right], upper=e2
	///         (e1 is now left half, new_right is right half)
	/// ```
	pub(crate) fn insert_after(
		&mut self,
		pos: u16,
		key: K,
		edge: Atomic<HybridLatch<Node<K, V, IC, LC>>>,
	) {
		// Insert key at position pos (becomes separator between left and right children)
		self.keys.insert(pos as usize, key);
		// Insert edge at position pos+1 (right child, after the left child at pos)
		self.edges.insert((pos + 1) as usize, edge);
		self.len += 1;
	}
}

impl<K: Clone, V, const IC: usize, const LC: usize> InternalNode<K, V, IC, LC> {
	/// Splits this internal node, moving entries after `split_pos` to `right`.
	///
	/// Internal node splitting is more complex than leaf splitting because
	/// the separator key at `split_pos` is "pushed up" to the parent rather
	/// than remaining in either child.
	///
	/// # Algorithm
	///
	/// ```text
	/// Before split (self):
	///   keys:  [K0, K1, K2, K3, K4, K5]  (split_pos = 3)
	///   edges: [E0, E1, E2, E3, E4, E5]  upper_edge: E6
	///
	/// After split:
	///   self (left):
	///     keys:  [K0, K1, K2]
	///     edges: [E0, E1, E2]  upper_edge: E3
	///
	///   right:
	///     keys:  [K4, K5]
	///     edges: [E4, E5]  upper_edge: E6
	///
	///   separator (pushed to parent): K3
	/// ```
	///
	/// Note: K3 is removed from both children and used as the separator
	/// in the parent. The edge that was at K3's position (E3) becomes
	/// the left node's upper_edge.
	pub(crate) fn split(&mut self, right: &mut InternalNode<K, V, IC, LC>, split_pos: u16) {
		// Get the split key - this will be pushed up to the parent
		let split_key =
			self.key_at(split_pos).expect("split position must be within node bounds").clone();

		// Update fence keys
		right.lower_fence = Some(split_key.clone());
		right.upper_fence = self.upper_fence.clone();
		self.upper_fence = Some(split_key);

		// Move keys and edges after split_pos to right
		assert!(right.keys.is_empty());
		assert!(right.edges.is_empty());
		right.keys.extend(self.keys.drain((split_pos + 1) as usize..));
		right.edges.extend(self.edges.drain((split_pos + 1) as usize..));

		// Right gets our upper_edge (it's now the rightmost in its range)
		right.upper_edge = self.upper_edge.take();

		// The edge at split_pos becomes our new upper_edge
		// (it was pointing to children between K(split_pos-1) and K(split_pos))
		self.upper_edge =
			Some(self.edges.pop().expect("edges non-empty: split requires at least one edge"));
		// Remove the key at split_pos (it's being pushed to parent)
		self.keys.pop().expect("keys non-empty: split requires at least one key");

		// Set sample keys for node relocation
		self.sample_key = Some(self.keys[0].clone());
		right.sample_key = Some(right.keys[0].clone());

		// Update lengths
		right.len = right.keys.len() as u16;
		self.len = self.keys.len() as u16;
	}

	/// Merges the `right` internal node into `self`.
	///
	/// Internal node merging is the inverse of splitting. The separator key
	/// from the parent (stored as right's lower_fence) must be re-inserted
	/// between the two nodes' contents.
	///
	/// # Algorithm
	///
	/// ```text
	/// Before merge:
	///   self:  keys=[K0, K1], edges=[E0, E1], upper=E2
	///   right: keys=[K3, K4], edges=[E3, E4], upper=E5
	///   right.lower_fence = K2 (was the separator in parent)
	///
	/// After merge (into self):
	///   keys:  [K0, K1, K2, K3, K4]
	///   edges: [E0, E1, E2, E3, E4]  upper=E5
	/// ```
	///
	/// # Returns
	///
	/// - `true` if merge succeeded
	/// - `false` if combined size would exceed capacity
	pub(crate) fn merge(&mut self, right: &mut InternalNode<K, V, IC, LC>) -> bool {
		// Check if combined entries fit
		// +1 for the separator key that gets added back
		if (self.len + right.len + 1) as usize > IC {
			return false;
		}

		// Inherit right's upper_fence (we now cover its range too)
		let _left_upper_fence = std::mem::replace(&mut self.upper_fence, right.upper_fence.take());

		// Our upper_edge will be used as a regular edge
		let left_upper_edge = std::mem::replace(&mut self.upper_edge, right.upper_edge.take());

		// Re-insert the separator key (was in parent, stored as right's lower_fence)
		// This key goes between our old content and right's content
		self.keys.push(
			right
				.lower_fence
				.take()
				.expect("merge requires right node to have lower_fence (separator key)"),
		);

		// Our old upper_edge becomes a regular edge (points to children < separator)
		self.edges.push(left_upper_edge.expect("merge requires left node to have upper_edge"));

		// Append all of right's content
		self.keys.extend(right.keys.drain(..));
		self.edges.extend(right.edges.drain(..));

		// Update sample key
		self.sample_key = right.sample_key.take();

		// Update lengths
		self.len = self.keys.len() as u16;
		right.len = 0;

		true
	}
}

// ===========================================================================
// Test-Only Validation Module
// ===========================================================================

/// Invariant validation for testing. Validates tree structure to ensure
/// unreachable code paths are never reached.
#[cfg(any(test, feature = "test-utils"))]
impl<K: Clone + Ord + std::fmt::Debug, V, const IC: usize, const LC: usize>
	GenericTree<K, V, IC, LC>
{
	/// Validates all tree invariants. Panics with diagnostic info if any invariant is violated.
	///
	/// This function should be called after operations in tests to verify the tree
	/// maintains its structural integrity.
	///
	/// # Invariants Checked
	///
	/// 1. Height consistency: All leaves at same depth
	/// 2. Node type consistency: Internal nodes at internal levels, leaves at leaf level
	/// 3. Key ordering: Keys sorted within each node
	/// 4. Fence key consistency: Keys fall within fence bounds
	/// 5. Upper edge presence: All internal nodes have upper_edge set
	/// 6. Length consistency: len field matches actual key count
	pub fn assert_invariants(&self) {
		let eg = epoch::pin();
		let height = self.height.load(Ordering::Acquire);

		// Get root
		let tree_guard = self.root.optimistic_or_spin();
		let root_ptr = tree_guard.load(Ordering::Acquire, &eg);

		if root_ptr.is_null() {
			// Empty tree - only valid if height is 1
			assert_eq!(height, 1, "Empty tree should have height 1");
			return;
		}

		// SAFETY: We're in a test context and hold an epoch guard
		let root_latch = unsafe { root_ptr.deref() };
		let root_guard = root_latch.optimistic_or_spin();

		// Validate recursively
		self.validate_node_recursive(&root_guard, 0, height, None, None, &eg);
	}

	/// Recursively validates a node and its subtree.
	///
	/// # Arguments
	/// * `node` - The node to validate
	/// * `level` - Current level (0 = root)
	/// * `height` - Total tree height
	/// * `expected_lower` - Lower bound from parent (exclusive), None if leftmost
	/// * `expected_upper` - Upper bound from parent (inclusive), None if rightmost
	fn validate_node_recursive<'a>(
		&self,
		guard: &OptimisticGuard<'a, Node<K, V, IC, LC>>,
		level: usize,
		height: usize,
		expected_lower: Option<&K>,
		expected_upper: Option<&K>,
		eg: &epoch::Guard,
	) {
		let is_leaf_level = level == height - 1;

		match guard.inner() {
			Node::Leaf(leaf) => {
				// Invariant 2: Node type consistency
				assert!(
					is_leaf_level,
					"Found leaf at level {} but expected internal (height={})",
					level, height
				);

				// Invariant 6: Length consistency
				assert_eq!(
					leaf.len as usize,
					leaf.keys.len(),
					"Leaf len {} != keys.len() {}",
					leaf.len,
					leaf.keys.len()
				);
				assert_eq!(
					leaf.keys.len(),
					leaf.values.len(),
					"Leaf keys.len() {} != values.len() {}",
					leaf.keys.len(),
					leaf.values.len()
				);

				// Invariant 3: Key ordering
				for i in 1..leaf.keys.len() {
					assert!(
						leaf.keys[i - 1] < leaf.keys[i],
						"Keys not sorted at positions {} and {}: {:?} >= {:?}",
						i - 1,
						i,
						leaf.keys[i - 1],
						leaf.keys[i]
					);
				}

				// Invariant 4: Fence key consistency
				if let Some(lower) = &leaf.lower_fence {
					for key in &leaf.keys[..] {
						assert!(
							key > lower,
							"Key {:?} not greater than lower_fence {:?}",
							key,
							lower
						);
					}
				}
				if let Some(upper) = &leaf.upper_fence {
					for key in &leaf.keys[..] {
						assert!(key <= upper, "Key {:?} not <= upper_fence {:?}", key, upper);
					}
				}

				// Check against parent's expected bounds
				if let Some(lower) = expected_lower {
					for key in &leaf.keys[..] {
						assert!(
							key > lower,
							"Key {:?} not greater than parent lower bound {:?}",
							key,
							lower
						);
					}
				}
				if let Some(upper) = expected_upper {
					for key in &leaf.keys[..] {
						assert!(
							key <= upper,
							"Key {:?} not <= parent upper bound {:?}",
							key,
							upper
						);
					}
				}
			}
			Node::Internal(internal) => {
				// Invariant 2: Node type consistency
				assert!(
					!is_leaf_level,
					"Found internal node at leaf level {} (height={})",
					level, height
				);

				// Invariant 5: Upper edge presence
				assert!(
					internal.upper_edge.is_some(),
					"Internal node at level {} has no upper_edge",
					level
				);

				// Invariant 6: Length consistency
				assert_eq!(
					internal.len as usize,
					internal.keys.len(),
					"Internal len {} != keys.len() {}",
					internal.len,
					internal.keys.len()
				);
				assert_eq!(
					internal.keys.len(),
					internal.edges.len(),
					"Internal keys.len() {} != edges.len() {}",
					internal.keys.len(),
					internal.edges.len()
				);

				// Invariant 3: Key ordering
				for i in 1..internal.keys.len() {
					assert!(
						internal.keys[i - 1] < internal.keys[i],
						"Internal keys not sorted at {} and {}: {:?} >= {:?}",
						i - 1,
						i,
						internal.keys[i - 1],
						internal.keys[i]
					);
				}

				// Invariant 4: Fence key consistency
				if let Some(lower) = &internal.lower_fence {
					for key in &internal.keys[..] {
						assert!(
							key > lower,
							"Internal key {:?} not > lower_fence {:?}",
							key,
							lower
						);
					}
				}
				if let Some(upper) = &internal.upper_fence {
					for key in &internal.keys[..] {
						assert!(
							key <= upper,
							"Internal key {:?} not <= upper_fence {:?}",
							key,
							upper
						);
					}
				}

				// Recurse into children
				let mut prev_upper: Option<&K> = expected_lower;

				for (i, edge) in internal.edges.iter().enumerate() {
					let child_ptr = edge.load(Ordering::Acquire, eg);
					if !child_ptr.is_null() {
						// SAFETY: Test context with epoch guard
						let child_latch = unsafe { child_ptr.deref() };
						let child_guard = child_latch.optimistic_or_spin();

						let child_upper = Some(&internal.keys[i]);
						self.validate_node_recursive(
							&child_guard,
							level + 1,
							height,
							prev_upper,
							child_upper,
							eg,
						);
						prev_upper = child_upper;
					}
				}

				// Validate upper_edge child
				if let Some(upper_edge) = &internal.upper_edge {
					let child_ptr = upper_edge.load(Ordering::Acquire, eg);
					if !child_ptr.is_null() {
						let child_latch = unsafe { child_ptr.deref() };
						let child_guard = child_latch.optimistic_or_spin();

						self.validate_node_recursive(
							&child_guard,
							level + 1,
							height,
							prev_upper,
							expected_upper,
							eg,
						);
					}
				}
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	// -----------------------------------------------------------------------
	// Basic Tree Operation Tests
	// -----------------------------------------------------------------------

	#[test]
	fn basic_insert_and_lookup() {
		let tree: Tree<i32, &str> = Tree::new();

		assert_eq!(tree.insert(1, "one"), None);
		assert_eq!(tree.insert(2, "two"), None);
		assert_eq!(tree.insert(3, "three"), None);

		tree.assert_invariants();

		assert_eq!(tree.lookup(&1, |v| *v), Some("one"));
		assert_eq!(tree.lookup(&2, |v| *v), Some("two"));
		assert_eq!(tree.lookup(&3, |v| *v), Some("three"));
		assert_eq!(tree.lookup(&4, |v| *v), None);
	}

	#[test]
	fn insert_update() {
		let tree: Tree<i32, &str> = Tree::new();

		assert_eq!(tree.insert(1, "one"), None);
		assert_eq!(tree.insert(1, "uno"), Some("one"));
		assert_eq!(tree.lookup(&1, |v| *v), Some("uno"));

		tree.assert_invariants();
	}

	#[test]
	fn remove() {
		let tree: Tree<i32, &str> = Tree::new();

		tree.insert(1, "one");
		tree.insert(2, "two");

		tree.assert_invariants();

		assert_eq!(tree.remove(&1), Some("one"));
		assert_eq!(tree.lookup(&1, |v| *v), None);
		assert_eq!(tree.lookup(&2, |v| *v), Some("two"));

		tree.assert_invariants();
	}

	#[test]
	fn raw_iter() {
		let tree: Tree<i32, i32> = Tree::new();

		for i in 0..100 {
			tree.insert(i, i * 10);
		}

		tree.assert_invariants();

		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		for i in 0..100 {
			let result = iter.next();
			let (k, v) = result.unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i * 10);
		}

		assert!(iter.next().is_none());
	}

	#[test]
	fn raw_iter_reverse() {
		let tree: Tree<i32, i32> = Tree::new();

		for i in 0..100 {
			tree.insert(i, i * 10);
		}

		tree.assert_invariants();

		let mut iter = tree.raw_iter();
		iter.seek_to_last();

		for i in (0..100).rev() {
			let (k, v) = iter.prev().unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i * 10);
		}

		assert!(iter.prev().is_none());
	}

	#[test]
	fn len_and_is_empty() {
		let tree: Tree<i32, i32> = Tree::new();

		assert!(tree.is_empty());
		assert_eq!(tree.len(), 0);

		tree.insert(1, 10);
		assert!(!tree.is_empty());
		assert_eq!(tree.len(), 1);

		tree.insert(2, 20);
		assert_eq!(tree.len(), 2);

		tree.assert_invariants();

		tree.remove(&1);
		assert_eq!(tree.len(), 1);

		tree.assert_invariants();
	}

	// -----------------------------------------------------------------------
	// LeafNode Unit Tests
	// -----------------------------------------------------------------------

	#[test]
	fn leaf_lower_bound_empty() {
		let leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		let (pos, exact) = leaf.lower_bound(&5);
		assert_eq!(pos, 0);
		assert!(!exact);
	}

	#[test]
	fn leaf_lower_bound_exact_match() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.insert_at(0, 10, 100);
		leaf.insert_at(1, 20, 200);
		leaf.insert_at(2, 30, 300);

		let (pos, exact) = leaf.lower_bound(&20);
		assert_eq!(pos, 1);
		assert!(exact);
	}

	#[test]
	fn leaf_lower_bound_between_keys() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.insert_at(0, 10, 100);
		leaf.insert_at(1, 20, 200);
		leaf.insert_at(2, 30, 300);

		let (pos, exact) = leaf.lower_bound(&25);
		assert_eq!(pos, 2); // Would insert at position 2
		assert!(!exact);
	}

	#[test]
	fn leaf_lower_bound_before_all() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.insert_at(0, 10, 100);
		leaf.insert_at(1, 20, 200);

		let (pos, exact) = leaf.lower_bound(&5);
		assert_eq!(pos, 0);
		assert!(!exact);
	}

	#[test]
	fn leaf_lower_bound_after_all() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.insert_at(0, 10, 100);
		leaf.insert_at(1, 20, 200);

		let (pos, exact) = leaf.lower_bound(&25);
		assert_eq!(pos, 2);
		assert!(!exact);
	}

	#[test]
	fn leaf_lower_bound_respects_lower_fence() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.lower_fence = Some(50);
		leaf.insert_at(0, 60, 600);
		leaf.insert_at(1, 70, 700);

		// Key below lower fence
		let (pos, exact) = leaf.lower_bound(&40);
		assert_eq!(pos, 0);
		assert!(!exact);
	}

	#[test]
	fn leaf_lower_bound_respects_upper_fence() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.upper_fence = Some(50);
		leaf.insert_at(0, 30, 300);
		leaf.insert_at(1, 40, 400);

		// Key above upper fence
		let (pos, exact) = leaf.lower_bound(&60);
		assert_eq!(pos, 2);
		assert!(!exact);
	}

	#[test]
	fn leaf_within_bounds() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.lower_fence = Some(10);
		leaf.upper_fence = Some(50);

		assert!(!leaf.within_bounds(&5)); // Below lower
		assert!(!leaf.within_bounds(&10)); // At lower (exclusive)
		assert!(leaf.within_bounds(&30)); // In range
		assert!(leaf.within_bounds(&50)); // At upper (inclusive)
		assert!(!leaf.within_bounds(&55)); // Above upper
	}

	#[test]
	fn leaf_within_bounds_no_lower_fence() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.upper_fence = Some(50);

		assert!(leaf.within_bounds(&5)); // No lower bound
		assert!(leaf.within_bounds(&50));
		assert!(!leaf.within_bounds(&55));
	}

	#[test]
	fn leaf_within_bounds_no_upper_fence() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.lower_fence = Some(10);

		assert!(!leaf.within_bounds(&5));
		assert!(leaf.within_bounds(&50)); // No upper bound
		assert!(leaf.within_bounds(&1000));
	}

	#[test]
	fn leaf_within_bounds_no_fences() {
		let leaf: LeafNode<i32, i32, 64> = LeafNode::new();

		assert!(leaf.within_bounds(&0));
		assert!(leaf.within_bounds(&100));
		assert!(leaf.within_bounds(&i32::MAX));
	}

	#[test]
	fn leaf_insert_at_and_remove_at() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();

		assert!(leaf.insert_at(0, 10, 100).is_some());
		assert!(leaf.insert_at(1, 30, 300).is_some());
		assert!(leaf.insert_at(1, 20, 200).is_some()); // Insert in middle

		assert_eq!(leaf.len, 3);
		assert_eq!(*leaf.key_at(0).unwrap(), 10);
		assert_eq!(*leaf.key_at(1).unwrap(), 20);
		assert_eq!(*leaf.key_at(2).unwrap(), 30);

		let (k, v) = leaf.remove_at(1);
		assert_eq!(k, 20);
		assert_eq!(v, 200);
		assert_eq!(leaf.len, 2);
	}

	#[test]
	fn leaf_split_sets_fences_correctly() {
		let mut left: LeafNode<i32, i32, 64> = LeafNode::new();
		for i in 0..10 {
			left.insert_at(i as u16, i * 10, i * 100);
		}

		let mut right: LeafNode<i32, i32, 64> = LeafNode::new();
		left.split(&mut right, 5);

		// Left should have keys 0-50, right should have keys 60-90
		assert_eq!(left.len, 6); // 0, 10, 20, 30, 40, 50
		assert_eq!(right.len, 4); // 60, 70, 80, 90

		// Check fence keys
		assert!(left.lower_fence.is_none()); // Left keeps original lower fence
		assert_eq!(left.upper_fence, Some(50)); // Split key becomes left's upper fence

		assert_eq!(right.lower_fence, Some(50)); // Split key becomes right's lower fence
		assert!(right.upper_fence.is_none()); // Right inherits original upper fence

		// Check sample keys
		assert_eq!(left.sample_key, Some(0));
		assert_eq!(right.sample_key, Some(60));
	}

	#[test]
	fn leaf_merge_combines_entries() {
		let mut left: LeafNode<i32, i32, 64> = LeafNode::new();
		left.insert_at(0, 10, 100);
		left.insert_at(1, 20, 200);
		left.upper_fence = Some(25);

		let mut right: LeafNode<i32, i32, 64> = LeafNode::new();
		right.lower_fence = Some(25);
		right.upper_fence = Some(50);
		right.insert_at(0, 30, 300);
		right.insert_at(1, 40, 400);
		right.sample_key = Some(30);

		let result = left.merge(&mut right);
		assert!(result);

		assert_eq!(left.len, 4);
		assert_eq!(*left.key_at(0).unwrap(), 10);
		assert_eq!(*left.key_at(1).unwrap(), 20);
		assert_eq!(*left.key_at(2).unwrap(), 30);
		assert_eq!(*left.key_at(3).unwrap(), 40);

		// Left inherits right's upper fence
		assert_eq!(left.upper_fence, Some(50));
		// Left inherits right's sample key
		assert_eq!(left.sample_key, Some(30));

		// Right should be empty
		assert_eq!(right.len, 0);
	}

	#[test]
	fn leaf_merge_fails_when_too_full() {
		let mut left: LeafNode<i32, i32, 4> = LeafNode::new();
		left.insert_at(0, 10, 100);
		left.insert_at(1, 20, 200);
		left.insert_at(2, 30, 300);

		let mut right: LeafNode<i32, i32, 4> = LeafNode::new();
		right.insert_at(0, 40, 400);
		right.insert_at(1, 50, 500);

		// Combined size (5) > capacity (4)
		let result = left.merge(&mut right);
		assert!(!result);
		// Both should be unchanged
		assert_eq!(left.len, 3);
		assert_eq!(right.len, 2);
	}

	#[test]
	fn leaf_has_space() {
		let mut leaf: LeafNode<i32, i32, 3> = LeafNode::new();
		assert!(leaf.has_space());

		leaf.insert_at(0, 1, 1);
		assert!(leaf.has_space());

		leaf.insert_at(1, 2, 2);
		assert!(leaf.has_space());

		leaf.insert_at(2, 3, 3);
		assert!(!leaf.has_space());
	}

	#[test]
	fn leaf_is_underfull() {
		// With capacity 10, underfull threshold is 4 (40%)
		let mut leaf: LeafNode<i32, i32, 10> = LeafNode::new();

		// Empty is underfull
		assert!(leaf.is_underfull());

		for i in 0..3 {
			leaf.insert_at(i as u16, i, i);
		}
		// 3 entries with capacity 10 = 30%, still underfull
		assert!(leaf.is_underfull());

		leaf.insert_at(3, 3, 3);
		// 4 entries = 40%, at threshold, NOT underfull
		assert!(!leaf.is_underfull());
	}

	// -----------------------------------------------------------------------
	// InternalNode Unit Tests
	// -----------------------------------------------------------------------

	#[test]
	fn internal_lower_bound_empty() {
		let internal: InternalNode<i32, i32, 64, 64> = InternalNode::new();
		let (pos, exact) = internal.lower_bound(&5);
		assert_eq!(pos, 0);
		assert!(!exact);
	}

	#[test]
	fn internal_lower_bound_finds_correct_child() {
		let mut internal: InternalNode<i32, i32, 64, 64> = InternalNode::new();
		// Keys: 10, 20, 30
		// Children: [<10], [10-20), [20-30), [>=30]
		internal.keys.push(10);
		internal.keys.push(20);
		internal.keys.push(30);
		internal.len = 3;

		let (pos, exact) = internal.lower_bound(&5);
		assert_eq!(pos, 0); // < 10, go to child 0
		assert!(!exact);

		let (pos, exact) = internal.lower_bound(&10);
		assert_eq!(pos, 0); // == 10, exact match
		assert!(exact);

		let (pos, exact) = internal.lower_bound(&15);
		assert_eq!(pos, 1); // > 10, < 20
		assert!(!exact);

		let (pos, exact) = internal.lower_bound(&25);
		assert_eq!(pos, 2); // > 20, < 30
		assert!(!exact);

		let (pos, exact) = internal.lower_bound(&35);
		assert_eq!(pos, 3); // >= 30, go to upper_edge
		assert!(!exact);
	}

	#[test]
	fn internal_has_space() {
		let mut internal: InternalNode<i32, i32, 3, 64> = InternalNode::new();
		assert!(internal.has_space());

		internal.len = 2;
		assert!(internal.has_space());

		internal.len = 3;
		assert!(!internal.has_space());
	}

	#[test]
	fn internal_is_underfull() {
		// With capacity 10, underfull threshold is 4 (40%)
		let mut internal: InternalNode<i32, i32, 10, 64> = InternalNode::new();

		internal.len = 3;
		assert!(internal.is_underfull());

		internal.len = 4;
		assert!(!internal.is_underfull());
	}

	// -----------------------------------------------------------------------
	// Tree Structure Tests
	// -----------------------------------------------------------------------

	#[test]
	fn new_tree_has_height_one() {
		let tree: Tree<i32, i32> = Tree::new();
		assert_eq!(tree.height(), 1);
	}

	#[test]
	fn inserts_cause_splits_and_height_increase() {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to cause splits (LEAF_CAPACITY is 64)
		for i in 0..200 {
			tree.insert(i, i);
		}

		tree.assert_invariants();
		assert!(tree.height() > 1);

		// Verify all entries are still findable
		for i in 0..200 {
			assert_eq!(tree.lookup(&i, |v| *v), Some(i));
		}
	}

	#[test]
	fn many_inserts_cause_multiple_levels() {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to cause multiple levels of splits
		for i in 0..1000 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		assert!(tree.height() >= 2);
		assert_eq!(tree.len(), 1000);

		// Verify random access works
		assert_eq!(tree.lookup(&0, |v| *v), Some(0));
		assert_eq!(tree.lookup(&500, |v| *v), Some(500));
		assert_eq!(tree.lookup(&999, |v| *v), Some(999));
	}

	#[test]
	fn reverse_insertion_order() {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert in reverse order
		for i in (0..200).rev() {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		// Verify all entries and order
		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		for i in 0..200 {
			let (k, v) = iter.next().unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i);
		}
	}

	#[test]
	fn random_insertion_order() {
		use rand::prelude::*;

		let tree: Tree<i32, i32> = Tree::new();

		let mut keys: Vec<i32> = (0..200).collect();
		let mut rng = rand::rng();
		keys.shuffle(&mut rng);

		for k in keys {
			tree.insert(k, k * 10);
		}

		tree.assert_invariants();

		// Verify all entries are findable
		for i in 0..200 {
			assert_eq!(tree.lookup(&i, |v| *v), Some(i * 10));
		}

		// Verify iteration order is sorted
		let mut iter = tree.raw_iter();
		iter.seek_to_first();

		let mut prev = -1;
		while let Some((k, _)) = iter.next() {
			assert!(*k > prev);
			prev = *k;
		}
	}

	// -----------------------------------------------------------------------
	// Delete and Merge Tests
	// -----------------------------------------------------------------------

	#[test]
	fn delete_all_entries() {
		let tree: Tree<i32, i32> = Tree::new();

		for i in 0..100 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		for i in 0..100 {
			assert_eq!(tree.remove(&i), Some(i));
		}

		tree.assert_invariants();
		assert!(tree.is_empty());
	}

	#[test]
	fn delete_in_reverse_order() {
		let tree: Tree<i32, i32> = Tree::new();

		for i in 0..100 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		for i in (0..100).rev() {
			assert_eq!(tree.remove(&i), Some(i));
		}

		tree.assert_invariants();
		assert!(tree.is_empty());
	}

	#[test]
	fn delete_random_order() {
		use rand::prelude::*;

		let tree: Tree<i32, i32> = Tree::new();

		for i in 0..100 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		let mut keys: Vec<i32> = (0..100).collect();
		let mut rng = rand::rng();
		keys.shuffle(&mut rng);

		for k in keys {
			assert_eq!(tree.remove(&k), Some(k));
		}

		tree.assert_invariants();
		assert!(tree.is_empty());
	}

	#[test]
	fn delete_nonexistent_returns_none() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(1, 10);

		assert_eq!(tree.remove(&999), None);
		assert_eq!(tree.len(), 1);
		tree.assert_invariants();
	}

	#[test]
	fn remove_entry_returns_key_and_value() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(42, 420);

		let result = tree.remove_entry(&42);
		assert_eq!(result, Some((42, 420)));
		tree.assert_invariants();
	}

	#[test]
	fn interleaved_insert_and_delete() {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert some entries
		for i in 0..50 {
			tree.insert(i, i);
		}

		tree.assert_invariants();

		// Delete half
		for i in 0..25 {
			tree.remove(&i);
		}

		tree.assert_invariants();

		// Insert more
		for i in 50..100 {
			tree.insert(i, i);
		}

		// Delete some more
		for i in 50..75 {
			tree.remove(&i);
		}

		tree.assert_invariants();

		// Verify remaining entries
		assert_eq!(tree.len(), 50); // 25-49 and 75-99

		for i in 25..50 {
			assert_eq!(tree.lookup(&i, |v| *v), Some(i));
		}
		for i in 75..100 {
			assert_eq!(tree.lookup(&i, |v| *v), Some(i));
		}
	}

	// -----------------------------------------------------------------------
	// Node Type Tests
	// -----------------------------------------------------------------------

	#[test]
	fn node_is_leaf() {
		let leaf_node: Node<i32, i32, 64, 64> = Node::Leaf(LeafNode::new());
		let internal_node: Node<i32, i32, 64, 64> = Node::Internal(InternalNode::new());

		assert!(leaf_node.is_leaf());
		assert!(!internal_node.is_leaf());
	}

	#[test]
	fn node_can_merge_with_same_type() {
		let leaf1: Node<i32, i32, 64, 64> = Node::Leaf(LeafNode::new());
		let leaf2: Node<i32, i32, 64, 64> = Node::Leaf(LeafNode::new());

		assert!(leaf1.can_merge_with(&leaf2));
	}

	#[test]
	fn node_cannot_merge_with_different_type() {
		let leaf: Node<i32, i32, 64, 64> = Node::Leaf(LeafNode::new());
		let internal: Node<i32, i32, 64, 64> = Node::Internal(InternalNode::new());

		assert!(!leaf.can_merge_with(&internal));
		assert!(!internal.can_merge_with(&leaf));
	}

	// -----------------------------------------------------------------------
	// Edge Case Tests
	// -----------------------------------------------------------------------

	#[test]
	fn empty_tree_lookup() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.assert_invariants();
		assert_eq!(tree.lookup(&1, |v| *v), None);
	}

	#[test]
	fn empty_tree_remove() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.assert_invariants();
		assert_eq!(tree.remove(&1), None);
		tree.assert_invariants();
	}

	#[test]
	fn duplicate_inserts_update_value() {
		let tree: Tree<i32, i32> = Tree::new();

		tree.insert(1, 10);
		tree.insert(1, 20);
		tree.insert(1, 30);

		tree.assert_invariants();
		assert_eq!(tree.lookup(&1, |v| *v), Some(30));
		assert_eq!(tree.len(), 1);
	}

	#[test]
	fn string_keys() {
		let tree: Tree<String, i32> = Tree::new();

		tree.insert("apple".to_string(), 1);
		tree.insert("banana".to_string(), 2);
		tree.insert("cherry".to_string(), 3);

		tree.assert_invariants();
		assert_eq!(tree.lookup(&"banana".to_string(), |v| *v), Some(2));
	}

	#[test]
	fn lookup_with_borrowed_key() {
		let tree: Tree<String, i32> = Tree::new();
		tree.insert("hello".to_string(), 42);

		tree.assert_invariants();
		// Lookup using &str instead of String
		assert_eq!(tree.lookup("hello", |v| *v), Some(42));
	}

	#[test]
	fn empty_string_key() {
		let tree: Tree<String, i32> = Tree::new();
		tree.insert("".to_string(), 42);

		tree.assert_invariants();
		assert_eq!(tree.lookup(&"".to_string(), |v| *v), Some(42));
	}

	#[test]
	fn large_values() {
		let tree: Tree<i32, Vec<u8>> = Tree::new();

		let large_value = vec![0u8; 10000];
		tree.insert(1, large_value.clone());

		tree.assert_invariants();
		let result = tree.lookup(&1, |v| v.len());
		assert_eq!(result, Some(10000));
	}

	// -----------------------------------------------------------------------
	// Default Implementation Tests
	// -----------------------------------------------------------------------

	#[test]
	fn tree_default_creates_empty_tree() {
		let tree: Tree<i32, i32> = Tree::default();
		tree.assert_invariants();
		assert!(tree.is_empty());
		assert_eq!(tree.height(), 1);
	}

	// -----------------------------------------------------------------------
	// Keys Method Test
	// -----------------------------------------------------------------------

	#[test]
	fn node_keys_returns_keys() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.insert_at(0, 10, 100);
		leaf.insert_at(1, 20, 200);
		leaf.insert_at(2, 30, 300);

		let node: Node<i32, i32, 64, 64> = Node::Leaf(leaf);
		let keys = node.keys();

		assert_eq!(keys, &[10, 20, 30]);
	}

	// -----------------------------------------------------------------------
	// Sample Key Tests
	// -----------------------------------------------------------------------

	#[test]
	fn leaf_sample_key_initially_none() {
		let leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		let node: Node<i32, i32, 64, 64> = Node::Leaf(leaf);
		assert!(node.sample_key().is_none());
	}

	#[test]
	fn internal_sample_key_initially_none() {
		let internal: InternalNode<i32, i32, 64, 64> = InternalNode::new();
		let node: Node<i32, i32, 64, 64> = Node::Internal(internal);
		assert!(node.sample_key().is_none());
	}

	// -----------------------------------------------------------------------
	// Convenience Method Tests
	// -----------------------------------------------------------------------

	#[test]
	fn contains_key_returns_true_for_existing() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(1, "one");
		tree.insert(2, "two");
		tree.insert(3, "three");

		assert!(tree.contains_key(&1));
		assert!(tree.contains_key(&2));
		assert!(tree.contains_key(&3));
	}

	#[test]
	fn contains_key_returns_false_for_missing() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(1, "one");

		assert!(!tree.contains_key(&0));
		assert!(!tree.contains_key(&2));
		assert!(!tree.contains_key(&100));
	}

	#[test]
	fn contains_key_empty_tree() {
		let tree: Tree<i32, &str> = Tree::new();
		assert!(!tree.contains_key(&1));
	}

	#[test]
	fn get_returns_cloned_value() {
		let tree: Tree<i32, String> = Tree::new();
		tree.insert(1, "one".to_string());
		tree.insert(2, "two".to_string());

		assert_eq!(tree.get(&1), Some("one".to_string()));
		assert_eq!(tree.get(&2), Some("two".to_string()));
		assert_eq!(tree.get(&3), None);
	}

	#[test]
	fn get_empty_tree() {
		let tree: Tree<i32, String> = Tree::new();
		assert_eq!(tree.get(&1), None);
	}

	#[test]
	fn first_key_value_returns_minimum() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(3, "three");
		tree.insert(1, "one");
		tree.insert(2, "two");

		let first = tree.first_key_value(|k, v| (*k, *v));
		assert_eq!(first, Some((1, "one")));
	}

	#[test]
	fn first_key_value_empty_tree() {
		let tree: Tree<i32, &str> = Tree::new();
		let first = tree.first_key_value(|k, v| (*k, *v));
		assert_eq!(first, None);
	}

	#[test]
	fn first_key_value_single_entry() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(42, "answer");

		let first = tree.first_key_value(|k, v| (*k, *v));
		assert_eq!(first, Some((42, "answer")));
	}

	#[test]
	fn last_key_value_returns_maximum() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(1, "one");
		tree.insert(3, "three");
		tree.insert(2, "two");

		let last = tree.last_key_value(|k, v| (*k, *v));
		assert_eq!(last, Some((3, "three")));
	}

	#[test]
	fn last_key_value_empty_tree() {
		let tree: Tree<i32, &str> = Tree::new();
		let last = tree.last_key_value(|k, v| (*k, *v));
		assert_eq!(last, None);
	}

	#[test]
	fn last_key_value_single_entry() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(42, "answer");

		let last = tree.last_key_value(|k, v| (*k, *v));
		assert_eq!(last, Some((42, "answer")));
	}

	#[test]
	fn clear_empties_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..100 {
			tree.insert(i, i * 10);
		}

		assert_eq!(tree.len(), 100);
		assert!(tree.height() > 1);

		tree.clear();

		assert!(tree.is_empty());
		assert_eq!(tree.height(), 1);
		assert_eq!(tree.len(), 0);
	}

	#[test]
	fn clear_empty_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		tree.clear();

		assert!(tree.is_empty());
		assert_eq!(tree.height(), 1);
	}

	#[test]
	fn clear_then_insert() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(1, "one");
		tree.insert(2, "two");

		tree.clear();
		assert!(tree.is_empty());

		// Can insert again after clear
		tree.insert(3, "three");
		tree.insert(4, "four");

		assert_eq!(tree.len(), 2);
		assert_eq!(tree.get(&3), Some("three"));
		assert_eq!(tree.get(&4), Some("four"));
		assert_eq!(tree.get(&1), None); // Old entries gone
	}

	#[test]
	fn contains_key_with_borrowed_key() {
		let tree: Tree<String, i32> = Tree::new();
		tree.insert("hello".to_string(), 42);
		tree.insert("world".to_string(), 99);

		// Lookup using &str instead of String
		assert!(tree.contains_key("hello"));
		assert!(tree.contains_key("world"));
		assert!(!tree.contains_key("missing"));
	}

	#[test]
	fn get_with_borrowed_key() {
		let tree: Tree<String, i32> = Tree::new();
		tree.insert("hello".to_string(), 42);

		// Lookup using &str instead of String
		assert_eq!(tree.get("hello"), Some(42));
		assert_eq!(tree.get("missing"), None);
	}

	#[test]
	fn first_key_value_multilevel_tree() {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to cause splits and create multiple levels
		for i in (0..200).rev() {
			tree.insert(i, i * 10);
		}

		tree.assert_invariants();
		assert!(tree.height() > 1);

		let first = tree.first_key_value(|k, v| (*k, *v));
		assert_eq!(first, Some((0, 0)));
	}

	#[test]
	fn last_key_value_multilevel_tree() {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to cause splits and create multiple levels
		for i in 0..200 {
			tree.insert(i, i * 10);
		}

		tree.assert_invariants();
		assert!(tree.height() > 1);

		let last = tree.last_key_value(|k, v| (*k, *v));
		assert_eq!(last, Some((199, 1990)));
	}

	#[test]
	fn clear_maintains_invariants() {
		let tree: Tree<i32, i32> = Tree::new();

		for i in 0..200 {
			tree.insert(i, i);
		}

		tree.assert_invariants();
		tree.clear();
		tree.assert_invariants();

		// Insert again and verify invariants still hold
		for i in 0..50 {
			tree.insert(i, i * 2);
		}
		tree.assert_invariants();
	}
}
