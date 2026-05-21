//! # Ferntree: A Concurrent In-Memory B+ Tree
//!
//! This crate provides a fast, concurrent B+ tree implementation featuring **optimistic lock
//! coupling**, a technique that enables high-throughput concurrent access with minimal
//! blocking.
//!
//! ## Design overview
//!
//! The implementation is based on research from:
//! - [LeanStore](https://dbis1.github.io/leanstore.html) - Optimistic lock coupling for B-trees
//! - [Umbra](https://umbra-db.com/#publications) - High-performance database engine techniques
//!
//! ### Key concepts
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
//! ### Tree structure
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
//! ## Basic usage
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
//! ## Concurrent usage
//!
//! The tree is designed for high-concurrency workloads. Wrap it in an `Arc` to share
//! across threads:
//!
//! ### Multi-threaded inserts
//!
//! ```
//! use ferntree::Tree;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let tree = Arc::new(Tree::<i32, i32>::new());
//!
//! // Spawn multiple writer threads
//! let handles: Vec<_> = (0..4).map(|t| {
//!     let tree = Arc::clone(&tree);
//!     thread::spawn(move || {
//!         for i in 0..1000 {
//!             tree.insert(t * 1000 + i, i);
//!         }
//!     })
//! }).collect();
//!
//! for h in handles {
//!     h.join().unwrap();
//! }
//!
//! assert_eq!(tree.len(), 4000);
//! ```
//!
//! ### Concurrent readers and writers
//!
//! ```
//! use ferntree::Tree;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let tree = Arc::new(Tree::<i32, i32>::new());
//!
//! // Pre-populate some data
//! for i in 0..100 {
//!     tree.insert(i, i * 10);
//! }
//!
//! let tree_writer = Arc::clone(&tree);
//! let tree_reader = Arc::clone(&tree);
//!
//! // Writer thread adds new entries
//! let writer = thread::spawn(move || {
//!     for i in 100..200 {
//!         tree_writer.insert(i, i * 10);
//!     }
//! });
//!
//! // Reader thread performs lookups concurrently
//! let reader = thread::spawn(move || {
//!     let mut found = 0;
//!     for i in 0..100 {
//!         if tree_reader.lookup(&i, |v| *v).is_some() {
//!             found += 1;
//!         }
//!     }
//!     found
//! });
//!
//! writer.join().unwrap();
//! let found = reader.join().unwrap();
//! assert_eq!(found, 100); // Reader sees consistent data
//! ```
//!
//! ### Iterator usage
//!
//! ```
//! use ferntree::Tree;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let tree = Arc::new(Tree::<i32, i32>::new());
//!
//! for i in 0..100 {
//!     tree.insert(i, i);
//! }
//!
//! // Iterators acquire locks on leaf nodes, ensuring consistent reads
//! let tree_iter = Arc::clone(&tree);
//! let handle = thread::spawn(move || {
//!     let mut iter = tree_iter.raw_iter();
//!     iter.seek_to_first();
//!
//!     let mut count = 0;
//!     while iter.next().is_some() {
//!         count += 1;
//!     }
//!     count
//! });
//!
//! // Concurrent modifications are safe
//! tree.insert(100, 100);
//!
//! let count = handle.join().unwrap();
//! // Iterator sees a consistent snapshot
//! assert!(count >= 100);
//! ```
//!
//! ## Thread safety
//!
//! ### Send and Sync bounds
//!
//! `Tree<K, V>` implements `Send` and `Sync` when both `K` and `V` implement `Send + Sync`.
//! This means the tree can be safely shared across threads and accessed concurrently.
//!
//! ```
//! use ferntree::Tree;
//!
//! fn assert_send_sync<T: Send + Sync>() {}
//!
//! // Tree is Send + Sync for thread-safe key/value types
//! assert_send_sync::<Tree<String, i32>>();
//! assert_send_sync::<Tree<i32, Vec<u8>>>();
//! ```
//!
//! ### Operation atomicity
//!
//! Individual operations (`insert`, `remove`, `lookup`) are atomic at the key level:
//!
//! - **Atomic reads**: `lookup()` always returns a consistent value for a key
//! - **Atomic writes**: `insert()` and `remove()` complete atomically
//! - **No cross-key transactions**: Multiple operations are NOT transactional; if you need
//!   to update multiple keys atomically, you must use external synchronization
//!
//! ### Retry semantics
//!
//! The tree uses optimistic concurrency control internally:
//!
//! - Operations may retry automatically if concurrent modifications are detected
//! - **Users don't need to handle retries** - this is managed internally
//! - **Important**: Closures passed to `lookup()` may be called multiple times if retries
//!   occur. Avoid side effects in lookup closures:
//!
//! ```
//! use ferntree::Tree;
//!
//! let tree = Tree::<i32, i32>::new();
//! tree.insert(1, 42);
//!
//! // Good: Pure closure that just extracts data
//! let value = tree.lookup(&1, |v| *v);
//!
//! // Avoid: Side effects in lookup closure (may execute multiple times)
//! // let mut counter = 0;
//! // tree.lookup(&1, |v| { counter += 1; *v }); // counter may be > 1
//! ```
//!
//! ### Memory safety
//!
//! The tree uses epoch-based memory reclamation via [`crossbeam_epoch`] to ensure safe
//! concurrent access:
//!
//! - **Safe concurrent reads**: Readers never see freed memory, even during concurrent
//!   modifications
//! - **Deferred deallocation**: Nodes removed from the tree are not immediately freed;
//!   they remain valid until all concurrent readers have finished
//! - **No use-after-free**: The epoch system guarantees that memory is only reclaimed
//!   when it's safe to do so
//!
//! This means you can safely perform concurrent reads and writes without worrying about
//! memory safety - the tree handles all synchronization internally.

// Complex types are intentional in this crate for expressing tree traversal results
#![allow(clippy::type_complexity)]
// Force every operation in an `unsafe fn` to be wrapped in its own `unsafe`
// block, so each pointer dereference / reclamation is individually justified.
#![deny(unsafe_op_in_unsafe_fn)]
// Require a `// SAFETY:` comment for every `unsafe` block. Combined with
// `unsafe_op_in_unsafe_fn` above, this guarantees every unsafe operation in
// the crate has a documented justification.
#![warn(clippy::undocumented_unsafe_blocks)]

use core::ptr;
use std::borrow::Borrow;
use std::fmt;
use std::ops::Bound;

pub mod alloc;
pub mod atomic_slot;
pub mod error;
pub(crate) mod inline_vec;
pub mod iter;
pub mod latch;
pub mod optimistic;
pub(crate) mod sync;

use sync::epoch::{self as epoch, Atomic, Owned};
use sync::{AtomicUsize, Ordering};

use atomic_slot::{AtomicLen, SlotArray};
use inline_vec::InlineVec;
use latch::{ExclusiveGuard, HybridGuard, HybridLatch, OptimisticGuard, SharedGuard};
pub use optimistic::OptimisticRead;

// ---------------------------------------------------------------------------
// Prefetch hint
// ---------------------------------------------------------------------------

/// CPU read-prefetch hint. Issued on the optimistic descent to overlap
/// the L3-miss latency of the child latch with the parent's binary-search
/// and recheck work. `_mm_prefetch` is a non-faulting hint instruction —
/// it is sound even if `ptr` is dangling, misaligned, or null.
///
/// x86_64 uses [`core::arch::x86_64::_mm_prefetch`]; other targets get a
/// no-op fallback. aarch64 prefetch intrinsics remain nightly-only at
/// time of writing, so darwin/aarch64 dev machines do not get the hint
/// yet — the optimisation is opportunistic.
mod prefetch {
	#[inline(always)]
	#[allow(unused_variables)]
	pub fn read_data<T>(ptr: *const T) {
		#[cfg(target_arch = "x86_64")]
		// SAFETY: `_mm_prefetch` is a CPU hint; no memory access is performed
		// at the architectural level, so dangling / null pointers are
		// tolerated.
		unsafe {
			core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(
				ptr as *const i8,
			);
		}
	}
}

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
pub struct GenericTree<K: OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize> {
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

impl<K: Clone + Ord + OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize> Default
	for GenericTree<K, V, IC, LC>
{
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
pub(crate) enum ParentHandler<
	'r,
	'p,
	K: OptimisticRead,
	V: OptimisticRead,
	const IC: usize,
	const LC: usize,
> {
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

impl<K: Clone + Ord + OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize>
	GenericTree<K, V, IC, LC>
{
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
				len: AtomicLen::new(0),
				keys: SlotArray::new(),
				values: SlotArray::new(),
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
		// SAFETY: `eg` is pinned for the lifetime of this borrow, so crossbeam-epoch
		// cannot reclaim the loaded `HybridLatch` while we hold the reference. The
		// root pointer is always non-null after `Tree::new` initialises it.
		// SAFETY: see the function-level safety contract.
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

		// Descend the tree looking for the needle. Raw-pointer projection
		// throughout — no `*target_guard` or `&InternalNode` reborrow —
		// so a concurrent `shift_remove_raw` on an internal node cannot
		// cause us to dereference a transiently-null boxed key slot
		// during the binary search.
		let parent_guard = loop {
			let target_ptr = target_guard.as_ptr();
			// SAFETY: `target_ptr` is owned by a HybridLatch we hold an
			// optimistic guard on.
			let (c_swip_ptr, c_pos) = match unsafe { Node::variant_raw(target_ptr) } {
				NodeKindRaw::Internal(internal_ptr) => {
					// SAFETY: `internal_ptr` is valid for the optimistic
					// guard's lifetime; `K: OptimisticRead` (impl bound)
					// certifies the snapshot discipline.
					// `lower_bound_raw` short-circuits on a null peek.
					let (c_pos, _) =
						unsafe { InternalNode::lower_bound_raw(internal_ptr, &search_key) };
					// SAFETY: same conditions as `lower_bound_raw`.
					let swip_ptr = unsafe { InternalNode::edge_at_raw(internal_ptr, c_pos)? };
					(swip_ptr, c_pos)
				}
				NodeKindRaw::Leaf(_) => {
					// Reached a leaf without finding the needle in internal nodes
					// The previous node (p_guard) must be the parent
					break p_guard.expect("must have parent");
				}
			};

			// `&Atomic` reborrow is sound (interior-mutable).
			//
			// SAFETY: `c_swip_ptr` is a valid `*const Atomic` for the
			// parent guard's lifetime.
			let c_swip = unsafe { &*c_swip_ptr };

			// Load the child pointer for the needle-equality check. We use
			// `as_raw` rather than `deref` here on purpose: this is a pure
			// pointer comparison and dereferencing the loaded `Shared` would
			// touch a potentially-freed `HybridLatch` *before* we get a
			// chance to validate the parent's snapshot (issue #14). On a
			// pointer match we break out without ever derefing this slot;
			// on a mismatch the descent calls `lock_coupling` below, which
			// validates the parent before touching the latch.
			let c_latch_ptr = c_swip.load(Ordering::Acquire, eg).as_raw();

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
		// SAFETY: `eg` is pinned, so the loaded `HybridLatch` cannot be reclaimed
		// for the lifetime of `root_latch`.
		// SAFETY: see the function-level safety contract.
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
			Direction::Forward => pos < parent_guard.as_internal().len.load(),
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
					Direction::Forward => pos < parent_guard.as_internal().len.load(),
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
		// Step 1: Load the child pointer. `InternalNode::upper_edge` uses a
		// null-pointer sentinel for "no upper edge" and the slot can be
		// transiently null during a concurrent split / merge; treat a null
		// load as a snapshot-validation failure (the parent's `recheck`
		// below would catch the staleness anyway, but if we deref first
		// we crash before getting there).
		let shared = swip.load(Ordering::Acquire, eg);
		if shared.is_null() {
			std::hint::cold_path();
			return Err(error::Error::Unwind);
		}

		// Step 2: Validate the parent BEFORE dereferencing the child.
		//
		// This pre-deref recheck closes issue #14: if a concurrent (or
		// earlier same-thread) writer has merged this child away under the
		// parent's exclusive lock and `defer_destroy`d its `HybridLatch`,
		// the parent's version has been bumped and `recheck()` will fail.
		// Without this gate, the deref below would touch a latch that
		// `crossbeam-epoch::collect` may have already freed — including the
		// same-thread case where pinning a fresh epoch guard advanced the
		// local epoch and ran `collect` on a prior op's deferred queue.
		p_guard.recheck()?;

		// SAFETY: `shared` is non-null per the check above; `recheck()` just
		// confirmed the parent has not been modified since `p_guard` was
		// taken, so the swip we loaded is still the current child pointer
		// and that latch has not been unlinked + `defer_destroy`d. With `eg`
		// pinned, the latch is alive for the lifetime of this reference.
		let c_latch = unsafe { shared.deref() };

		// Step 3: Acquire optimistic access to the validated child
		let c_guard = c_latch.optimistic_or_spin();

		// Step 4: Re-validate the parent AFTER capturing the child snapshot.
		//
		// The pre-deref recheck only guarantees the deref was safe at that
		// moment; between then and the `optimistic_or_spin` capture above a
		// concurrent writer could have acquired the parent exclusively,
		// detached this child, and bumped the parent's version. If that
		// happened, our child snapshot is from a now-stale subtree — fail
		// out and let the caller retry. (This second recheck is what the
		// pre-fix code relied on; we keep it, just with the deref gated.)
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
		// See `lock_coupling` for the null-load rationale.
		let shared = swip.load(Ordering::Acquire, eg);
		if shared.is_null() {
			std::hint::cold_path();
			return Err(error::Error::Unwind);
		}

		// Validate parent BEFORE dereferencing the child — see the matching
		// note in `lock_coupling`. Skipping this gate exposes the descent to
		// a use-after-free on a latch the same-thread writer-churn epoch
		// race just freed (issue #14).
		p_guard.recheck()?;

		// SAFETY: `shared` is non-null and `recheck()` confirmed the swip is
		// still the current child pointer; with `eg` pinned the latch is
		// alive for the lifetime of `c_latch`.
		let c_latch = unsafe { shared.deref() };

		// Acquire shared (blocking) access to the validated child
		let c_guard = c_latch.shared();

		// Re-validate the parent after acquiring the child lock — see the
		// matching note in `lock_coupling`. A writer can take parent
		// exclusively, detach this child, and `defer_destroy` it during the
		// `shared()` block; without this recheck we would return a lock on
		// a detached subtree and the caller would mutate the tree off-path.
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
		// See `lock_coupling` for the null-load rationale.
		let shared = swip.load(Ordering::Acquire, eg);
		if shared.is_null() {
			std::hint::cold_path();
			return Err(error::Error::Unwind);
		}

		// Validate parent BEFORE dereferencing the child — see the matching
		// note in `lock_coupling`. Without this gate, an insert that follows
		// a same-thread remove can deref (and `CAS`-spin on) a latch whose
		// `Box` was freed by `crossbeam-epoch::collect` running during the
		// insert's own `epoch::pin` (issue #14, t5 in `race_stress.rs`).
		p_guard.recheck()?;

		// SAFETY: `shared` is non-null and `recheck()` confirmed the swip is
		// still the current child pointer; with `eg` pinned the latch is
		// alive for the lifetime of `c_latch`.
		let c_latch = unsafe { shared.deref() };

		// Acquire exclusive (blocking) access to the validated child
		let c_guard = c_latch.exclusive();

		// Re-validate the parent after acquiring the child lock — see the
		// matching note in `lock_coupling`. Without this second recheck a
		// writer that took parent-exclusive during our `exclusive()` block
		// can detach this child, leaving us with an exclusive lock on a
		// node that is no longer reachable from the root; the caller's
		// insert/remove would then mutate a detached subtree and the tree's
		// sort invariant would be violated.
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
						Direction::Reverse => internal.len.load(),
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
		// SAFETY: `eg` is pinned, so the loaded `HybridLatch` cannot be reclaimed
		// for the lifetime of `root_latch`.
		// SAFETY: see the function-level safety contract.
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
		// SAFETY: `eg` is pinned, so the loaded `HybridLatch` cannot be reclaimed
		// for the lifetime of `root_latch`.
		// SAFETY: see the function-level safety contract.
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
		K: Borrow<Q> + Ord + OptimisticRead,
		Q: ?Sized + Ord,
	{
		// Acquire access to the root
		let tree_guard = self.root.optimistic_or_spin();
		// SAFETY: `eg` is pinned, so the loaded `HybridLatch` cannot be reclaimed
		// for the lifetime of `root_latch`.
		// SAFETY: see the function-level safety contract.
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_guard = root_latch.optimistic_or_spin();
		tree_guard.recheck()?;

		// Track the tree guard until we leave the root level
		let mut t_guard = Some(tree_guard);
		let mut p_guard = None;
		let mut target_guard = root_guard;

		// Descend the tree following the key. We use raw-pointer projection
		// throughout — never `*target_guard` — so we never create an
		// `&Node` / `&InternalNode` reborrow that would dereference a
		// transiently-null boxed key slot during a concurrent
		// `shift_remove_raw` on an internal node. The version `recheck()`
		// at every level validates the snapshot.
		let leaf_guard = loop {
			let target_ptr = target_guard.as_ptr();
			// SAFETY: `target_ptr` is owned by a HybridLatch we hold an
			// optimistic guard on; the discriminant read is validated by
			// `lock_coupling`'s parent recheck on the next iteration or
			// by the caller's recheck after this function returns.
			let (c_swip_ptr, pos) = match unsafe { Node::variant_raw(target_ptr) } {
				NodeKindRaw::Internal(internal_ptr) => {
					// SAFETY: pointer to `InternalNode` obtained via raw
					// projection; `K: OptimisticRead` (added above)
					// certifies the binary-search snapshot discipline.
					// `lower_bound_raw` itself short-circuits to
					// `(lower, false)` on a null peek, which we then
					// resolve via the parent's `recheck()` inside
					// `lock_coupling`.
					let (pos, _) = unsafe { InternalNode::lower_bound_raw(internal_ptr, key) };
					// SAFETY: same conditions as `lower_bound_raw`.
					let swip_ptr = unsafe { InternalNode::edge_at_raw(internal_ptr, pos)? };
					(swip_ptr, pos)
				}
				NodeKindRaw::Leaf(_) => {
					// Reached the target leaf. Validate the snapshot
					// before handing the guard back so callers can rely
					// on the variant being stable.
					target_guard.recheck()?;
					break target_guard;
				}
			};

			// Lock couple to the child via the raw swip pointer.
			//
			// SAFETY: `c_swip_ptr` points at an `Atomic<...>` inside the
			// parent node we hold an optimistic guard on; the `&Atomic`
			// reborrow is interior-mutable (its load is an atomic op
			// rather than a non-atomic read), so it does not retag and
			// is sound to alias concurrently with a writer's `&mut` on
			// the surrounding node. `lock_coupling` performs the
			// atomic load and a parent recheck.
			let c_swip = unsafe { &*c_swip_ptr };
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
				// SAFETY: `eg` is pinned, so the loaded `HybridLatch` cannot be
				// reclaimed for the lifetime of `root_latch`.
				// SAFETY: see the function-level safety contract.
				let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
				let root_guard = root_latch.optimistic_or_spin();
				tree_guard.recheck()?;

				let mut t_guard = Some(tree_guard);
				let mut p_guard = None;
				let mut target_guard = root_guard;

				// Track current level to know when we're about to reach leaves
				let mut level = 1u16;

				let leaf_guard = loop {
					// Descend via raw-pointer projection — never
					// `*target_guard` — so we never form an `&Node` or
					// `&InternalNode` reborrow that would dereference a
					// transiently-null boxed key slot during a concurrent
					// `shift_remove_raw` on an internal node.
					let target_ptr = target_guard.as_ptr();
					// SAFETY: `target_ptr` is owned by a HybridLatch we
					// hold an optimistic guard on.
					let (c_swip_ptr, pos) = match unsafe { Node::variant_raw(target_ptr) } {
						NodeKindRaw::Internal(internal_ptr) => {
							// SAFETY: pointer to `InternalNode` obtained
							// via raw projection; `K: OptimisticRead`
							// (impl bound) certifies the snapshot
							// discipline. `lower_bound_raw` itself
							// short-circuits on a null peek.
							let (pos, _) =
								unsafe { InternalNode::lower_bound_raw(internal_ptr, key) };
							// SAFETY: same conditions as `lower_bound_raw`.
							let swip_ptr = unsafe { InternalNode::edge_at_raw(internal_ptr, pos)? };
							(swip_ptr, pos)
						}
						NodeKindRaw::Leaf(_) => {
							// Edge case: root is a leaf (single-node tree)
							if let Some(tree_guard) = t_guard.take() {
								tree_guard.recheck()?;
							}

							if p_guard.is_none() {
								// Root is the only node - upgrade to shared lock
								break target_guard.to_shared()?;
							} else {
								// Concurrent height shrink: the tree
								// collapsed a level between our
								// initial `height` load and this
								// descent, so the optimistic walk
								// ended up at a leaf with a parent
								// still set. Treat as a snapshot
								// failure and retry rather than
								// panicking on a benign race.
								std::hint::cold_path();
								return Err(error::Error::Unwind);
							}
						}
					};

					// `&Atomic` reborrow is sound: `Atomic` is interior-
					// mutable (its load is an atomic op), so it does not
					// retag against a concurrent writer's `&mut` on the
					// surrounding node.
					//
					// SAFETY: `c_swip_ptr` is a valid `*const Atomic` for
					// the parent guard's lifetime.
					let c_swip = unsafe { &*c_swip_ptr };

					// Check if next level is the leaf level.
					//
					// `Relaxed` is sufficient here: correctness of the
					// optimistic-to-shared transition is ultimately gated by
					// the parent's version `recheck()` performed inside
					// `lock_coupling_shared` / `lock_coupling`. A stale
					// height load either causes us to take a shared lock on
					// an internal node (which then fails the leaf-pattern
					// match and triggers retry) or an optimistic lock on a
					// leaf (which is sound for OptimisticRead values but
					// would fall through to the post-loop assertion for the
					// shared path — so we ALSO validate the height after
					// reading it by relying on the parent recheck). Either
					// way the structural change is detected and the
					// operation retries.
					if (level + 1) as usize == self.height.load(Ordering::Relaxed) {
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

	/// Descends to the leaf that should contain `key` using **only optimistic
	/// access** all the way down. The returned leaf guard is optimistic — the
	/// caller must `recheck()` it before trusting any data read from the leaf.
	///
	/// This is the read-only fast path for point lookups when the value type
	/// is [`crate::OptimisticRead`]. Unlike
	/// [`find_shared_leaf_and_optimistic_parent`](Self::find_shared_leaf_and_optimistic_parent),
	/// it never acquires the leaf's blocking shared lock, so writers are
	/// never blocked by readers (and vice versa) on the leaf level either.
	///
	/// # Safety
	///
	/// The returned leaf is held under an optimistic guard; the caller MUST
	/// validate any data read from it with `recheck()` before acting on it.
	#[inline]
	pub(crate) fn find_optimistic_leaf<'e, Q>(
		&self,
		key: &Q,
		eg: &'e epoch::Guard,
	) -> error::Result<OptimisticGuard<'e, Node<K, V, IC, LC>>>
	where
		K: Borrow<Q> + Ord + OptimisticRead,
		Q: ?Sized + Ord,
	{
		// Start traversal from root
		let tree_guard = self.root.optimistic_or_spin();
		// SAFETY: `eg` is pinned, so the loaded `HybridLatch` cannot be
		// reclaimed for the lifetime of `root_latch`.
		// SAFETY: see the function-level safety contract.
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_guard = root_latch.optimistic_or_spin();
		tree_guard.recheck()?;

		let mut t_guard = Some(tree_guard);
		let mut target_guard = root_guard;

		// Descend until we land on a leaf. We use raw-pointer projection
		// throughout — never `*target_guard` — so we never create an
		// `&Node` reborrow that would race (under Tree Borrows) with a
		// concurrent writer mutating the node under exclusive lock. The
		// version `recheck()` at every level validates the snapshot.
		loop {
			let target_ptr = target_guard.as_ptr();
			// SAFETY: target_ptr is owned by a HybridLatch we hold an
			// optimistic guard on; the read of the discriminant is
			// validated by `recheck()` on the next iteration's
			// `lock_coupling`.
			// SAFETY: see the function-level safety contract.
			let kind = unsafe { Node::variant_raw(target_ptr) };
			let c_swip_ptr = match kind {
				NodeKindRaw::Internal(internal_ptr) => {
					// SAFETY: pointer to InternalNode obtained via
					// raw projection; OptimisticRead bound on K
					// certifies the binary-search snapshot
					// discipline.
					// SAFETY: see the function-level safety contract.
					let (pos, _) = unsafe { InternalNode::lower_bound_raw(internal_ptr, key) };
					// SAFETY: same conditions as lower_bound_raw above.
					unsafe { InternalNode::edge_at_raw(internal_ptr, pos)? }
				}
				NodeKindRaw::Leaf(_) => {
					// Root is a leaf (single-node tree) or we've reached a
					// leaf via lock coupling. Either way, we're done —
					// after one final recheck of the tree guard if it's
					// still live.
					if let Some(tree_guard) = t_guard.take() {
						tree_guard.recheck()?;
					}
					// Final recheck of the leaf guard itself so callers
					// observe a consistent state before reading.
					target_guard.recheck()?;
					return Ok(target_guard);
				}
			};

			// Optimistic lock coupling: acquire child optimistically and
			// validate parent. `lock_coupling` reborrows the swip via
			// `&Atomic<...>`; that's sound because `Atomic` is its own
			// interior-mutable type whose load is an atomic operation
			// rather than a non-atomic read.
			//
			// SAFETY: c_swip_ptr is valid for the lifetime of the parent
			// guard; the `&` reborrow only lives for the lock_coupling
			// call which performs an atomic load and parent recheck.
			// SAFETY: see the function-level safety contract.
			let c_swip = unsafe { &*c_swip_ptr };
			// Issue a read-prefetch hint for the child latch while we are
			// still in this iteration. By the time `lock_coupling` does
			// its Acquire load + optimistic_or_spin on the child, the
			// pointed-to HybridLatch should be in L1/L2. A Relaxed load
			// is fine here — it's a hint, not a fence; if it tears we
			// just prefetch the wrong address, which the CPU silently
			// drops.
			let prefetch_target = c_swip.load(Ordering::Relaxed, eg).as_raw();
			prefetch::read_data(prefetch_target);
			let guard = GenericTree::lock_coupling(&target_guard, c_swip, eg)?;
			target_guard = guard;

			if let Some(tree_guard) = t_guard.take() {
				tree_guard.recheck()?;
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
				// SAFETY: `eg` is pinned, so the loaded `HybridLatch` cannot be
				// reclaimed for the lifetime of `root_latch`.
				// SAFETY: see the function-level safety contract.
				let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
				let root_guard = root_latch.optimistic_or_spin();
				tree_guard.recheck()?;

				let mut t_guard = Some(tree_guard);
				let mut p_guard = None;
				let mut target_guard = root_guard;

				let mut level = 1u16;

				let leaf_guard = loop {
					// Descend via raw-pointer projection — see the matching
					// note in `find_shared_leaf_and_optimistic_parent` —
					// so a concurrent `shift_remove_raw` on an internal
					// node cannot cause us to dereference a transiently
					// null boxed key slot during binary search.
					let target_ptr = target_guard.as_ptr();
					// SAFETY: `target_ptr` is owned by a HybridLatch we
					// hold an optimistic guard on.
					let (c_swip_ptr, pos) = match unsafe { Node::variant_raw(target_ptr) } {
						NodeKindRaw::Internal(internal_ptr) => {
							// SAFETY: `internal_ptr` is a valid
							// `*const InternalNode` for the optimistic
							// guard's lifetime; `K: OptimisticRead`
							// (impl bound) certifies the snapshot
							// discipline. `lower_bound_raw` itself
							// short-circuits on a null peek.
							let (pos, _) =
								unsafe { InternalNode::lower_bound_raw(internal_ptr, key) };
							// SAFETY: same conditions as `lower_bound_raw`.
							let swip_ptr = unsafe { InternalNode::edge_at_raw(internal_ptr, pos)? };
							(swip_ptr, pos)
						}
						NodeKindRaw::Leaf(_) => {
							// Root is a leaf - upgrade to exclusive
							if let Some(tree_guard) = t_guard.take() {
								tree_guard.recheck()?;
							}

							if p_guard.is_none() {
								break target_guard.to_exclusive()?;
							} else {
								// We descended through internal nodes
								// and unexpectedly landed on a leaf —
								// concurrent height shrink between our
								// initial `height.load(Relaxed)` and the
								// next iteration can leave the
								// optimistic descent one level "too
								// deep". Treat as a snapshot-validation
								// failure and retry; the parent's
								// version check would have caught it
								// anyway, but doing it explicitly here
								// keeps us from panicking on a benign
								// race.
								std::hint::cold_path();
								return Err(error::Error::Unwind);
							}
						}
					};

					// `&Atomic` reborrow is sound: `Atomic` is interior-
					// mutable, so it does not retag against a concurrent
					// writer's `&mut` on the surrounding node.
					//
					// SAFETY: `c_swip_ptr` is a valid `*const Atomic` for
					// the parent guard's lifetime.
					let c_swip = unsafe { &*c_swip_ptr };

					// `Relaxed` is sufficient — see the matching note in
					// `find_shared_leaf_and_optimistic_parent`. Correctness
					// is gated by the parent's `recheck()`.
					if (level + 1) as usize == self.height.load(Ordering::Relaxed) {
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

				// Check if the key actually exists in this leaf — via
				// raw-pointer projection rather than an `&LeafNode`
				// reborrow. Going through `as_leaf()` here would call
				// the safe-`&self` `lower_bound`, whose `load_into`
				// dereferences the boxed slot's raw pointer without
				// null-checking — and a concurrent `shift_remove_raw`
				// on the same leaf transiently null-stores intermediate
				// slots before incrementing the writer's epoch. The
				// raw-pointer path uses `try_load_into_raw` and bails
				// out via `Err(Unwind)` on a torn read.
				let node_ptr = leaf.as_ptr();
				// SAFETY: `leaf` is an `OptimisticGuard` whose pointer
				// is valid for the guard's lifetime; the variant
				// discriminant is validated by the `recheck` below
				// (or by `to_exclusive`).
				let leaf_ptr = match unsafe { Node::variant_raw(node_ptr) } {
					NodeKindRaw::Leaf(l) => l,
					NodeKindRaw::Internal(_) => {
						// Torn discriminant under a concurrent
						// structural change — retry.
						std::hint::cold_path();
						return Err(error::Error::Unwind);
					}
				};
				// SAFETY: `leaf_ptr` is a valid `*const LeafNode` for
				// the optimistic guard's lifetime; `K: OptimisticRead`
				// (impl bound) certifies the bitwise-snapshot
				// comparison discipline. `lower_bound_raw` itself
				// short-circuits on a null peek.
				let (_, exact) = unsafe { LeafNode::lower_bound_raw(leaf_ptr, key) };

				if exact {
					// Key tentatively found. Upgrade to exclusive
					// lock to stabilise the leaf, then re-locate the
					// key under the stable view — the optimistic
					// position can be stale if a concurrent writer
					// moved keys between the snapshot and the upgrade.
					let exclusive_leaf = leaf.to_exclusive()?;
					let (pos, exact) = exclusive_leaf.as_leaf().lower_bound(key);
					if !exact {
						// Concurrent remove of the same key; report
						// "not found" rather than retry — that is
						// the visible outcome of a serialised remove
						// race anyway.
						return error::Result::Ok(None);
					}
					error::Result::Ok(Some(((exclusive_leaf, pos), parent_opt)))
				} else {
					// Validate the negative result. If the optimistic
					// read saw a torn snapshot, `recheck` will fail and
					// we'll retry; otherwise the key really is absent.
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

	/// Looks up a value in the tree and applies a closure to it.
	///
	/// This method uses a closure-based API so the borrowed reference does not
	/// escape the locked region. The closure receives a reference to the value
	/// and should extract/clone whatever data is needed.
	///
	/// # Concurrency
	///
	/// A shared lock is held on the containing leaf for the duration of the
	/// closure. This blocks concurrent writers to that leaf but allows other
	/// readers, so `f` always observes a consistent `&V`. The closure runs
	/// exactly once.
	///
	/// Because writers are blocked while the closure runs, `f` should be
	/// short-running — avoid expensive work or operations that could call back
	/// into the tree.
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

		// Acquire a shared lock on the target leaf. The user's closure runs
		// while the lock is held, which blocks concurrent writers from
		// mutating `V` during the read. Optimistic reads are unsound for
		// values with interior pointers (e.g. SmallVec/Vec/String): a torn
		// read of length/tag/pointer bytes can trigger UB inside the value
		// type's own methods before recheck() ever runs.
		let (guard, _parent) = self.find_shared_leaf_and_optimistic_parent(key, eg);

		if let Node::Leaf(ref leaf) = *guard {
			// Binary search for the key within the leaf
			let (pos, exact) = leaf.lower_bound(key);

			if exact {
				// Safe to call user code: the shared guard blocks writers,
				// so `V` cannot be mutated for the duration of `f`. The
				// value is atomically loaded into a stack-local and
				// passed by reference to the closure.
				leaf.value_at(pos).ok().map(|v| f(&v))
			} else {
				None
			}
		} else {
			unreachable!(
				"find_shared_leaf_and_optimistic_parent returned non-leaf node - tree traversal invariant violated"
			)
		}
	}

	/// Returns `true` if the tree contains the specified key.
	///
	/// Uses the optimistic read fast path: the leaf is never shared-locked,
	/// so concurrent writers are never blocked by this call (and vice versa).
	/// The presence check only inspects key bytes that are already read
	/// optimistically during tree descent, so no extra safety bound on `V`
	/// is needed.
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
		// Uses the safe shared-lock path so it works for any `K`,
		// including non-`OptimisticRead` types like `String` /
		// `Vec<u8>`. Users who want the optimistic fast path with
		// `K: OptimisticRead` (typically `K: Copy` or `K: Bytes`) can
		// call [`contains_key_optimistic`](Self::contains_key_optimistic).
		self.lookup(key, |_| ()).is_some()
	}

	/// Same as [`contains_key`](Self::contains_key) but uses the optimistic
	/// read fast path — skips the leaf's shared lock entirely.
	///
	/// Requires [`K: OptimisticRead`](crate::OptimisticRead). All [`Copy`]
	/// key types satisfy this automatically. For refcounted key types
	/// like `bytes::Bytes`, opt in by implementing `OptimisticRead` with
	/// `EPOCH_DEFERRED_DROP = true` and using
	/// [`insert_defer`](Self::insert_defer) / [`remove_defer`](Self::remove_defer)
	/// for all writes.
	pub fn contains_key_optimistic<Q>(&self, key: &Q) -> bool
	where
		K: Borrow<Q> + Ord + OptimisticRead,
		Q: ?Sized + Ord,
	{
		let eg = &epoch::pin();

		// Retry loop for optimistic validation failures
		loop {
			let perform = || -> error::Result<bool> {
				let leaf_guard = self.find_optimistic_leaf(key, eg)?;

				// Raw-pointer projection: no `&Node` / `&LeafNode`
				// reborrow on the optimistic descent path. See
				// `Node::variant_raw` + `LeafNode::lower_bound_raw`.
				let node_ptr = leaf_guard.as_ptr();
				// SAFETY: `leaf_guard` is an OptimisticGuard on the
				// HybridLatch holding this Node; the raw discriminant
				// read is validated by `recheck()` below.
				// SAFETY: see the function-level safety contract.
				let leaf_ptr = match unsafe { Node::variant_raw(node_ptr) } {
					NodeKindRaw::Leaf(l) => l,
					NodeKindRaw::Internal(_) => {
						// Possible under a torn discriminant; recheck
						// will fail and we'll retry.
						std::hint::cold_path();
						return Err(error::Error::Unwind);
					}
				};

				// SAFETY: leaf_ptr is a valid pointer for the lifetime
				// of the optimistic guard. K: OptimisticRead certifies
				// the comparison snapshot discipline.
				// SAFETY: see the function-level safety contract.
				let (_, exact) = unsafe { LeafNode::lower_bound_raw(leaf_ptr, key) };

				// Validate the descent and the position we observed.
				leaf_guard.recheck()?;
				Ok(exact)
			};

			match perform() {
				Ok(result) => return result,
				Err(_) => {
					// Retry is the cold path — in the uncontended case
					// the first attempt succeeds.
					std::hint::cold_path();
					continue;
				}
			}
		}
	}

	/// Looks up a value using the optimistic read fast path.
	///
	/// Like [`lookup`](Self::lookup), but skips the leaf's shared lock. The
	/// value is snapshotted bitwise into a stack-local copy, the version is
	/// validated, and then the closure is invoked with a borrow of the
	/// validated snapshot. The snapshot's `Drop` is suppressed, so the
	/// original value in the leaf is the only one that gets dropped (when
	/// the writer eventually replaces or removes it).
	///
	/// This avoids the atomic acquire/release on the leaf's `RwLock` and the
	/// writer-blocking section that `lookup` holds across the closure.
	///
	/// # Trait bound
	///
	/// Requires [`V: OptimisticRead`](crate::OptimisticRead). Every [`Copy`]
	/// type satisfies this automatically. For values containing heap-owned
	/// interior pointers (e.g. `Vec<T>`, `String`, `SmallVec<…>`), use
	/// [`lookup`](Self::lookup) instead — its safety contract relies on the
	/// shared lock and is not relaxed by a marker trait.
	///
	/// # Closure execution
	///
	/// As with [`lookup`](Self::lookup), the closure may be invoked more
	/// than once if optimistic validation fails and the operation retries.
	/// Avoid side effects in the closure.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, u64> = Tree::new();
	/// tree.insert(1, 42);
	///
	/// let doubled = tree.lookup_optimistic(&1, |v| *v * 2);
	/// assert_eq!(doubled, Some(84));
	/// ```
	pub fn lookup_optimistic<Q, R, F>(&self, key: &Q, f: F) -> Option<R>
	where
		K: Borrow<Q> + Ord + OptimisticRead,
		Q: ?Sized + Ord,
		V: OptimisticRead,
		F: Fn(&V) -> R,
	{
		let eg = &epoch::pin();

		// Retry loop for optimistic validation failures
		loop {
			let perform = || -> error::Result<Option<R>> {
				let leaf_guard = self.find_optimistic_leaf(key, eg)?;

				// Raw-pointer projection — never `*leaf_guard` to `&Node`
				// or `&LeafNode`, so no retag races with a concurrent
				// writer.
				let node_ptr = leaf_guard.as_ptr();
				// SAFETY: leaf_guard is an OptimisticGuard on the
				// HybridLatch holding this Node.
				// SAFETY: see the function-level safety contract.
				let leaf_ptr = match unsafe { Node::variant_raw(node_ptr) } {
					NodeKindRaw::Leaf(l) => l,
					NodeKindRaw::Internal(_) => {
						std::hint::cold_path();
						return Err(error::Error::Unwind);
					}
				};

				// SAFETY: see `LeafNode::lower_bound_raw`. K: OptimisticRead
				// certifies the binary-search snapshot discipline.
				// SAFETY: see the function-level safety contract.
				let (pos, exact) = unsafe { LeafNode::lower_bound_raw(leaf_ptr, key) };

				if !exact {
					// Validate that the negative result is real.
					leaf_guard.recheck()?;
					return Ok(None);
				}

				// Bound by the compile-time capacity LC since `len` may be
				// inconsistent under concurrent mutation. We have
				// `pos <= LC` because `lower_bound_raw`'s `upper` is
				// bounded by `len.min(LC)`.
				if (pos as usize) >= LC {
					return Err(error::Error::Unwind);
				}

				// Atomic load from the leaf's atomic mirror via raw-
				// pointer projection (no `&LeafNode` reborrow). For
				// boxed-storage V, `try_load_raw` performs
				// `AtomicPtr::load(Acquire)` and clones through the
				// pointer; for inline-storage V, it does an atomic-sized
				// load of V's bits. Either way, the load synchronises
				// with the writer's `Release` store in `swap_init` /
				// `shift_*`, so Miri's data-race detector is satisfied.
				// `try_load` returns `None` if the slot was concurrently
				// emptied (boxed null pointer), which we treat as a
				// recheck-must-retry condition.
				//
				// SAFETY: `leaf_ptr` is valid for the lifetime of the
				// optimistic guard; `values` is a `SlotArray<…,
				// LC>` at a known field offset; `pos < LC` checked.
				let values_ptr: *const SlotArray<V::Slot, LC> =
					// SAFETY: see the function-level safety contract.
					unsafe { ptr::addr_of!((*leaf_ptr).values) };
				// SAFETY: see the function-level safety contract.
				let snapshot: V = match unsafe { SlotArray::try_load_raw(values_ptr, pos as usize) }
				{
					Some(v) => v,
					None => {
						// Slot was concurrently emptied; retry.
						std::hint::cold_path();
						return Err(error::Error::Unwind);
					}
				};

				// Validate that the snapshot is internally consistent
				// (i.e. no concurrent writer touched the leaf between
				// our binary search and the atomic load above).
				if let Err(err) = leaf_guard.recheck() {
					core::mem::forget(snapshot);
					return Err(err);
				}

				// The snapshot is validated. Hand a borrow to the user
				// closure. For boxed storage `snapshot` is a Clone of
				// the boxed V (its drop releases the cloned heap
				// allocation); for inline storage `snapshot` is a copy
				// of the V's bits (drop is a no-op).
				let result = f(&snapshot);
				drop(snapshot);
				Ok(Some(result))
			};

			match perform() {
				Ok(result) => return result,
				Err(_) => {
					std::hint::cold_path();
					continue;
				}
			}
		}
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

	/// Returns a clone of the value using the optimistic read fast path.
	///
	/// Convenience wrapper around [`lookup_optimistic`](Self::lookup_optimistic)
	/// for the common "get a copy of the value" use case. Faster than
	/// [`get`](Self::get) for [`Copy`] / [`OptimisticRead`] values because
	/// it never blocks on the leaf's shared lock.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, u64> = Tree::new();
	/// tree.insert(1, 42);
	///
	/// assert_eq!(tree.get_optimistic(&1), Some(42));
	/// assert_eq!(tree.get_optimistic(&2), None);
	/// ```
	pub fn get_optimistic<Q>(&self, key: &Q) -> Option<V>
	where
		K: Borrow<Q> + Ord + OptimisticRead,
		Q: ?Sized + Ord,
		V: OptimisticRead + Clone,
	{
		self.lookup_optimistic(key, |v| v.clone())
	}

	/// Returns the first (minimum) key-value pair in the tree.
	///
	/// The closure receives references to the key and value and should extract
	/// whatever data is needed. Returns `None` if the tree is empty.
	///
	/// # Important
	///
	/// The closure `f` may be executed multiple times if concurrent
	/// modifications cause optimistic-validation failures. **Do not perform
	/// side effects in `f`.** Only the result of the final successful call is
	/// returned.
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
	/// let first = tree.first(|k, v| (*k, *v));
	/// assert_eq!(first, Some((1, "one")));
	/// ```
	pub fn first<R, F>(&self, f: F) -> Option<R>
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
	/// # Important
	///
	/// The closure `f` may be executed multiple times if concurrent
	/// modifications cause optimistic-validation failures. **Do not perform
	/// side effects in `f`.** Only the result of the final successful call is
	/// returned.
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
	/// let last = tree.last(|k, v| (*k, *v));
	/// assert_eq!(last, Some((3, "three")));
	/// ```
	pub fn last<R, F>(&self, f: F) -> Option<R>
	where
		K: Ord,
		F: Fn(&K, &V) -> R,
	{
		let mut iter = self.raw_iter();
		iter.seek_to_last();
		iter.prev().map(|(k, v)| f(k, v))
	}

	/// Removes and returns the first (minimum) key-value pair from the tree.
	///
	/// Returns `None` if the tree is empty.
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
	/// assert_eq!(tree.pop_first(), Some((1, "one")));
	/// assert_eq!(tree.pop_first(), Some((2, "two")));
	/// assert_eq!(tree.pop_first(), Some((3, "three")));
	/// assert_eq!(tree.pop_first(), None);
	/// ```
	pub fn pop_first(&self) -> Option<(K, V)>
	where
		K: Clone + Ord,
	{
		self.first(|k, _| k.clone()).and_then(|k| self.remove_entry(&k))
	}

	/// Removes and returns the last (maximum) key-value pair from the tree.
	///
	/// Returns `None` if the tree is empty.
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
	/// assert_eq!(tree.pop_last(), Some((3, "three")));
	/// assert_eq!(tree.pop_last(), Some((2, "two")));
	/// assert_eq!(tree.pop_last(), Some((1, "one")));
	/// assert_eq!(tree.pop_last(), None);
	/// ```
	pub fn pop_last(&self) -> Option<(K, V)>
	where
		K: Clone + Ord,
	{
		self.last(|k, _| k.clone()).and_then(|k| self.remove_entry(&k))
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

	/// Removes a key from the tree, deferring the value's `Drop` via the
	/// epoch GC.
	///
	/// This is the epoch-aware companion to [`remove`](Self::remove), used
	/// when the tree stores values that are read via the optimistic fast
	/// path ([`lookup_optimistic`](Self::lookup_optimistic),
	/// [`get_optimistic`](Self::get_optimistic)) AND whose `Drop` would
	/// free a shared heap buffer (e.g. `bytes::Bytes`, `Arc<T>`). Without
	/// deferring, a concurrent reader could hold a validated snapshot
	/// whose interior pointer is invalidated by the synchronous drop.
	///
	/// The behavioural contract:
	///
	/// - returns `true` if the key was present and the entry was removed;
	/// - the removed value is NOT returned — it is moved into an epoch
	///   deferral and dropped at the next epoch reclamation tick;
	/// - for `V` with [`EPOCH_DEFERRED_DROP`](OptimisticRead::EPOCH_DEFERRED_DROP)
	///   `= false`, the value is dropped immediately (the monomorphised
	///   defer branch is dead code and elided).
	///
	/// If you need both the old value AND epoch-safe reclamation, do an
	/// optimistic read first to obtain a clone, then call this method.
	pub fn remove_defer<Q>(&self, key: &Q) -> bool
	where
		K: Borrow<Q> + Ord + OptimisticRead + Send + 'static,
		Q: ?Sized + Ord,
		V: OptimisticRead + Send + 'static,
	{
		// We need to share the same epoch pin between the removal and the
		// defer so the deferred closure is registered against the same
		// epoch the reader could currently be in.
		let eg = epoch::pin();
		let removed = self.remove_entry(key);
		if let Some((k, v)) = removed {
			// Symmetric K and V defer: leaf K is dropped on remove, so
			// for K: EPOCH_DEFERRED_DROP = true (e.g. `bytes::Bytes`)
			// the K's interior pointer must outlive any in-flight
			// optimistic reader's snapshot. `drop_or_defer` routes
			// through epoch when the const says so; otherwise it's an
			// immediate drop (the dead branch is elided at
			// monomorphisation).
			optimistic::drop_or_defer(k, &eg);
			optimistic::drop_or_defer(v, &eg);
			drop(eg);
			true
		} else {
			drop(eg);
			false
		}
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

		let result = if let Some(((guard, pos), _parent_opt)) =
			self.find_exact_exclusive_leaf_and_optimistic_parent(key, &eg)
		{
			// Remove the key-value pair from the leaf
			// SAFETY: see the function-level safety contract.
			let kv = unsafe {
				let node_ptr = guard.as_mut_ptr();
				let leaf_ptr = Node::as_leaf_ptr_mut(node_ptr);
				LeafNode::remove_at_raw(leaf_ptr, pos, &eg)
			};

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
		V: Clone,
	{
		// Use the mutable iterator for insertion
		// This handles splits automatically
		let mut iter = self.raw_iter_mut();
		iter.insert(key, value)
	}

	/// Inserts a key-value pair, deferring the `Drop` of any displaced old
	/// value via the epoch GC.
	///
	/// This is the epoch-aware companion to [`insert`](Self::insert), used
	/// when the tree stores values that are read via the optimistic fast
	/// path ([`lookup_optimistic`](Self::lookup_optimistic),
	/// [`get_optimistic`](Self::get_optimistic)) AND whose `Drop` would
	/// free a shared heap buffer (e.g. `bytes::Bytes`, `Arc<T>`). Without
	/// deferring, a concurrent reader could hold a validated snapshot
	/// whose interior pointer is invalidated by the synchronous drop.
	///
	/// The behavioural contract:
	///
	/// - returns `true` if the key was previously present (the displaced
	///   old value was deferred for epoch-safe drop);
	/// - returns `false` if the key was new;
	/// - the old value is NOT returned — it is moved into an epoch
	///   deferral and dropped at the next epoch reclamation tick;
	/// - for `V` with [`EPOCH_DEFERRED_DROP`](OptimisticRead::EPOCH_DEFERRED_DROP)
	///   `= false`, the displaced value is dropped immediately (the
	///   monomorphised defer branch is dead code and elided).
	pub fn insert_defer(&self, key: K, value: V) -> bool
	where
		K: Ord + OptimisticRead,
		V: OptimisticRead + Clone + Send + 'static,
	{
		// Note: K is not displaced on overwrite (only V is) and is not
		// dropped from the leaf on a fresh insert, so we don't need a
		// K: Send + 'static defer path here. The `K: OptimisticRead`
		// bound is there for API symmetry — callers using the
		// optimistic-read fast path must satisfy it on K anyway, and
		// requiring it here gives an earlier compile error if the
		// caller's K type doesn't opt in.
		let eg = epoch::pin();
		let displaced = {
			let mut iter = self.raw_iter_mut();
			iter.insert(key, value)
		};
		if let Some(old_v) = displaced {
			optimistic::drop_or_defer(old_v, &eg);
			drop(eg);
			true
		} else {
			drop(eg);
			false
		}
	}

	/// Returns a clone of the value for the key, inserting `default` if the key
	/// was not present.
	///
	/// This is a convenience method that calls `get_or_insert_with` with a
	/// closure that returns the default value.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, String> = Tree::new();
	///
	/// // Key doesn't exist - inserts and returns default
	/// let value = tree.get_or_insert(1, "default".to_string());
	/// assert_eq!(value, "default");
	///
	/// // Key exists - returns existing value without inserting
	/// let value = tree.get_or_insert(1, "other".to_string());
	/// assert_eq!(value, "default");
	/// ```
	pub fn get_or_insert(&self, key: K, default: V) -> V
	where
		K: Ord,
		V: Clone,
	{
		self.get_or_insert_with(key, || default)
	}

	/// Returns a clone of the value for the key, inserting the result of `f`
	/// if the key was not present.
	///
	/// The closure `f` is only called if the key does not exist in the tree.
	/// This allows for lazy initialization of values.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	///
	/// let tree: Tree<i32, String> = Tree::new();
	///
	/// // Key doesn't exist - closure is called
	/// let value = tree.get_or_insert_with(1, || "computed".to_string());
	/// assert_eq!(value, "computed");
	///
	/// // Key exists - closure is NOT called
	/// let value = tree.get_or_insert_with(1, || panic!("should not be called"));
	/// assert_eq!(value, "computed");
	/// ```
	pub fn get_or_insert_with<F>(&self, key: K, f: F) -> V
	where
		K: Ord,
		V: Clone,
		F: FnOnce() -> V,
	{
		let mut iter = self.raw_iter_mut();

		if iter.seek_exact(&key) {
			// Key exists - return clone of existing value
			let (_k, v) = iter.next().expect("seek_exact returned true");
			v.clone()
		} else {
			// Key doesn't exist - compute value, insert, and return clone
			let value = f();
			let result = value.clone();
			iter.insert(key, value);
			result
		}
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
			// SAFETY: We hold exclusive access to the root pointer (via `tree_guard`),
			// so no thread can begin a new traversal that finds `old_root`. Optimistic
			// readers still holding it will fail validation on their next `recheck()`.
			// Crossbeam-epoch defers `Drop` until every guard pinned at this moment
			// has been released.
			// SAFETY: see the function-level safety contract.
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
				// SAFETY: `eg` is pinned, so the loaded `HybridLatch` cannot be
				// reclaimed for the lifetime of `root_latch`. `tree_guard_x` holds
				// the root pointer exclusively, preventing concurrent replacement.
				// SAFETY: see the function-level safety contract.
				let root_latch = unsafe { tree_guard_x.load(Ordering::Acquire, eg).deref() };
				let mut root_guard_x = root_latch.exclusive();

				// Allocate the new root node (will be an internal node)
				let mut new_root_owned: Owned<HybridLatch<Node<K, V, IC, LC>>> =
					Owned::new(HybridLatch::new(Node::Internal(InternalNode::new())));

				match root_guard_x.as_mut() {
					Node::Internal(root_internal_node) => {
						// Root is an internal node that needs splitting

						// Don't split if too small (need at least 3 keys to split)
						if root_internal_node.len.load() <= 2 {
							return Ok(());
						}

						// Choose the middle position for the split
						let split_pos = root_internal_node.len.load() / 2;
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
							root_internal_node.split(new_right_node, split_pos, eg);
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
							new_root.upper_edge = new_right_node_edge;
						}
					}
					Node::Leaf(root_leaf_node) => {
						// Root is a leaf that needs splitting (tree is growing from height 1 to 2)

						// Don't split if too small
						if root_leaf_node.len.load() <= 2 {
							return Ok(());
						}

						// Choose the middle position for the split
						let split_pos = root_leaf_node.len.load() / 2;
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
							new_root.upper_edge = new_right_node_edge;
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
							if left_internal.len.load() <= 2 {
								return Ok(());
							}

							// Choose split position
							let split_pos = left_internal.len.load() / 2;
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
								left_internal.split(new_right_node, split_pos, eg);
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
							if pos == parent_internal.len.load() {
								// Node was at upper_edge - it becomes the left.
								// Insert split key and make right the new upper_edge.
								let left_edge = std::mem::replace(
									&mut parent_internal.upper_edge,
									new_right_node_edge,
								);
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
							if left_leaf.len.load() <= 2 {
								return Ok(());
							}

							// Choose split position
							let split_pos = left_leaf.len.load() / 2;
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
							if pos == parent_internal.len.load() {
								// Node was at upper_edge - it becomes the left.
								// Insert split key and make right the new upper_edge.
								let left_edge = std::mem::replace(
									&mut parent_internal.upper_edge,
									new_right_node_edge,
								);
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
				let parent_len = parent_guard.as_internal().len.load();

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
										let left_edge = parent_internal.remove_at(pos - 1, eg);
										let dropped_edge = std::mem::replace(
											&mut parent_internal.upper_edge,
											left_edge,
										);

										// Schedule the old target for deferred destruction
										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											// SAFETY: `dropped_edge` was unlinked from the
											// parent under an exclusive latch above, so no new
											// traversal can reach it. Optimistic readers still
											// holding the stale pointer will fail validation on
											// `recheck()` because the parent's version is
											// bumped on unlock. Crossbeam-epoch defers `Drop`
											// until every currently pinned epoch guard has
											// been released.
											// SAFETY: see the function-level safety contract.
											unsafe { eg.defer_destroy(shared) };
										}
									} else {
										// Target was at edges[pos]
										let left_edge = parent_internal.remove_at(pos - 1, eg);
										let dropped_edge = std::mem::replace(
											&mut parent_internal.edges[(pos - 1) as usize],
											left_edge,
										);

										// Schedule deferred destruction
										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											// SAFETY: `dropped_edge` was unlinked from the
											// parent under an exclusive latch above, so no new
											// traversal can reach it. Optimistic readers still
											// holding the stale pointer will fail validation on
											// `recheck()` because the parent's version is
											// bumped on unlock. Crossbeam-epoch defers `Drop`
											// until every currently pinned epoch guard has
											// been released.
											// SAFETY: see the function-level safety contract.
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

								if !left_guard_x.as_internal_mut().merge(target_internal, eg) {
									parent_guard = parent_guard_x.unlock();
									target_guard = target_guard_x.unlock();
									false
								} else {
									let parent_internal = parent_guard_x.as_internal_mut();

									if pos == parent_len {
										let left_edge = parent_internal.remove_at(pos - 1, eg);
										let dropped_edge = std::mem::replace(
											&mut parent_internal.upper_edge,
											left_edge,
										);

										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											// SAFETY: `dropped_edge` was unlinked from the
											// parent under an exclusive latch above, so no new
											// traversal can reach it. Optimistic readers still
											// holding the stale pointer will fail validation on
											// `recheck()` because the parent's version is
											// bumped on unlock. Crossbeam-epoch defers `Drop`
											// until every currently pinned epoch guard has
											// been released.
											// SAFETY: see the function-level safety contract.
											unsafe { eg.defer_destroy(shared) };
										}
									} else {
										let left_edge = parent_internal.remove_at(pos - 1, eg);
										let dropped_edge = std::mem::replace(
											&mut parent_internal.edges[(pos - 1) as usize],
											left_edge,
										);

										let shared = dropped_edge.load(Ordering::Relaxed, eg);
										if !shared.is_null() {
											// SAFETY: `dropped_edge` was unlinked from the
											// parent under an exclusive latch above, so no new
											// traversal can reach it. Optimistic readers still
											// holding the stale pointer will fail validation on
											// `recheck()` because the parent's version is
											// bumped on unlock. Crossbeam-epoch defers `Drop`
											// until every currently pinned epoch guard has
											// been released.
											// SAFETY: see the function-level safety contract.
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
				let merge_succeeded =
					if !merge_succeeded && parent_len > 0 && (pos + 1) <= parent_len {
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
											let left_edge = parent_internal.remove_at(pos, eg);
											let dropped_edge = std::mem::replace(
												&mut parent_internal.upper_edge,
												left_edge,
											);

											let shared = dropped_edge.load(Ordering::Relaxed, eg);
											if !shared.is_null() {
												// SAFETY: `dropped_edge` was unlinked from the
												// parent under an exclusive latch above, so no new
												// traversal can reach it. Optimistic readers still
												// holding the stale pointer will fail validation on
												// `recheck()` because the parent's version is
												// bumped on unlock. Crossbeam-epoch defers `Drop`
												// until every currently pinned epoch guard has
												// been released.
												// SAFETY: see the function-level safety contract.
												unsafe { eg.defer_destroy(shared) };
											}
										} else {
											let left_edge = parent_internal.remove_at(pos, eg);
											let dropped_edge = std::mem::replace(
												&mut parent_internal.edges[pos as usize],
												left_edge,
											);

											let shared = dropped_edge.load(Ordering::Relaxed, eg);
											if !shared.is_null() {
												// SAFETY: `dropped_edge` was unlinked from the
												// parent under an exclusive latch above, so no new
												// traversal can reach it. Optimistic readers still
												// holding the stale pointer will fail validation on
												// `recheck()` because the parent's version is
												// bumped on unlock. Crossbeam-epoch defers `Drop`
												// until every currently pinned epoch guard has
												// been released.
												// SAFETY: see the function-level safety contract.
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

									if !target_internal.merge(right_guard_x.as_internal_mut(), eg) {
										parent_guard = parent_guard_x.unlock();
										let _ = target_guard_x.unlock();
										false
									} else {
										let parent_internal = parent_guard_x.as_internal_mut();

										if pos + 1 == parent_len {
											let left_edge = parent_internal.remove_at(pos, eg);
											let dropped_edge = std::mem::replace(
												&mut parent_internal.upper_edge,
												left_edge,
											);

											let shared = dropped_edge.load(Ordering::Relaxed, eg);
											if !shared.is_null() {
												// SAFETY: `dropped_edge` was unlinked from the
												// parent under an exclusive latch above, so no new
												// traversal can reach it. Optimistic readers still
												// holding the stale pointer will fail validation on
												// `recheck()` because the parent's version is
												// bumped on unlock. Crossbeam-epoch defers `Drop`
												// until every currently pinned epoch guard has
												// been released.
												// SAFETY: see the function-level safety contract.
												unsafe { eg.defer_destroy(shared) };
											}
										} else {
											let left_edge = parent_internal.remove_at(pos, eg);
											let dropped_edge = std::mem::replace(
												&mut parent_internal.edges[pos as usize],
												left_edge,
											);

											let shared = dropped_edge.load(Ordering::Relaxed, eg);
											if !shared.is_null() {
												// SAFETY: `dropped_edge` was unlinked from the
												// parent under an exclusive latch above, so no new
												// traversal can reach it. Optimistic readers still
												// holding the stale pointer will fail validation on
												// `recheck()` because the parent's version is
												// bumped on unlock. Crossbeam-epoch defers `Drop`
												// until every currently pinned epoch guard has
												// been released.
												// SAFETY: see the function-level safety contract.
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
		V: Clone,
	{
		iter::RawExclusiveIter::new(self)
	}

	/// Returns an iterator over the entries of the tree within the specified bounds.
	///
	/// The iterator yields key-value pairs in ascending key order, starting from
	/// entries that satisfy the lower bound and stopping when entries exceed
	/// the upper bound.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	/// use std::ops::Bound::{Included, Excluded, Unbounded};
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// tree.insert(1, "one");
	/// tree.insert(2, "two");
	/// tree.insert(3, "three");
	/// tree.insert(4, "four");
	/// tree.insert(5, "five");
	///
	/// // Range from 2 (inclusive) to 4 (exclusive)
	/// let mut range = tree.range(Included(&2), Excluded(&4));
	/// assert_eq!(range.next(), Some((&2, &"two")));
	/// assert_eq!(range.next(), Some((&3, &"three")));
	/// assert_eq!(range.next(), None);
	///
	/// // Range from 3 to end
	/// let mut range = tree.range(Included(&3), Unbounded);
	/// assert_eq!(range.next(), Some((&3, &"three")));
	/// assert_eq!(range.next(), Some((&4, &"four")));
	/// assert_eq!(range.next(), Some((&5, &"five")));
	/// assert_eq!(range.next(), None);
	/// ```
	pub fn range<Q>(&self, min: Bound<&Q>, max: Bound<&Q>) -> iter::Range<'_, K, V, IC, LC>
	where
		K: Borrow<Q> + Clone + Ord,
		Q: ?Sized + Ord,
	{
		iter::Range::new(self, min, max)
	}

	/// Returns a reverse iterator over the entries of the tree within the specified bounds.
	///
	/// The iterator yields key-value pairs in descending key order, starting from
	/// entries that satisfy the upper bound and stopping when entries go below
	/// the lower bound.
	///
	/// # Example
	///
	/// ```
	/// use ferntree::Tree;
	/// use std::ops::Bound::{Included, Excluded, Unbounded};
	///
	/// let tree: Tree<i32, &str> = Tree::new();
	/// tree.insert(1, "one");
	/// tree.insert(2, "two");
	/// tree.insert(3, "three");
	/// tree.insert(4, "four");
	/// tree.insert(5, "five");
	///
	/// // Reverse range from 2 (inclusive) to 4 (exclusive)
	/// let mut range = tree.range_rev(Included(&2), Excluded(&4));
	/// assert_eq!(range.next(), Some((&3, &"three")));
	/// assert_eq!(range.next(), Some((&2, &"two")));
	/// assert_eq!(range.next(), None);
	///
	/// // Reverse range from start to 3 (inclusive)
	/// let mut range = tree.range_rev(Unbounded, Included(&3));
	/// assert_eq!(range.next(), Some((&3, &"three")));
	/// assert_eq!(range.next(), Some((&2, &"two")));
	/// assert_eq!(range.next(), Some((&1, &"one")));
	/// assert_eq!(range.next(), None);
	/// ```
	pub fn range_rev<Q>(&self, min: Bound<&Q>, max: Bound<&Q>) -> iter::RangeRev<'_, K, V, IC, LC>
	where
		K: Borrow<Q> + Clone + Ord,
		Q: ?Sized + Ord,
	{
		iter::RangeRev::new(self, min, max)
	}

	/// Returns an iterator over the keys of the tree in ascending order.
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
	/// let mut keys = tree.keys();
	/// assert_eq!(keys.next(), Some(&1));
	/// assert_eq!(keys.next(), Some(&2));
	/// assert_eq!(keys.next(), Some(&3));
	/// assert_eq!(keys.next(), None);
	/// ```
	pub fn keys(&self) -> iter::Keys<'_, K, V, IC, LC>
	where
		K: Clone + Ord,
	{
		iter::Keys::new(self)
	}

	/// Returns an iterator over the values of the tree in key-ascending order.
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
	/// let mut values = tree.values();
	/// assert_eq!(values.next(), Some(&"one"));
	/// assert_eq!(values.next(), Some(&"two"));
	/// assert_eq!(values.next(), Some(&"three"));
	/// assert_eq!(values.next(), None);
	/// ```
	pub fn values(&self) -> iter::Values<'_, K, V, IC, LC>
	where
		K: Clone + Ord,
	{
		iter::Values::new(self)
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
	/// This is an O(height) operation that traverses to the first leaf
	/// and checks if it has any entries. This is more efficient than
	/// the full iterator approach as it avoids iterator allocation and
	/// only needs optimistic locks.
	pub fn is_empty(&self) -> bool {
		let eg = epoch::pin();

		loop {
			let perform = || {
				// Find the first (leftmost) leaf in the tree
				let (leaf_guard, _parent_opt) = self.find_first_leaf_and_parent(&eg)?;

				// Check if the first leaf has any entries
				let is_empty = leaf_guard.as_leaf().len.load() == 0;

				// Validate our optimistic read
				leaf_guard.recheck()?;

				error::Result::Ok(is_empty)
			};

			match perform() {
				Ok(result) => return result,
				Err(_) => continue, // Retry on validation failure
			}
		}
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
///
/// # Layout
///
/// `#[repr(C, u8)]` gives this enum a stable layout: a `u8` discriminant
/// at offset 0, followed by the variant data (with padding for alignment).
/// This lets the optimistic read fast path read the discriminant via a
/// raw pointer projection without creating an `&Node` reborrow that would
/// race (under Tree Borrows) with a concurrent writer's mutation. See
/// [`Node::variant_raw`].
#[repr(C, u8)]
pub(crate) enum Node<K: OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize> {
	/// An internal (index) node containing keys and child pointers.
	Internal(InternalNode<K, V, IC, LC>) = 0,
	/// A leaf node containing key-value pairs.
	Leaf(LeafNode<K, V, LC>) = 1,
}

/// Raw-pointer view of a [`Node`] variant, used by the optimistic read
/// fast path. Each variant carries a `*const` to the variant's inner type
/// — never an `&` reborrow — so the caller can project further to leaf /
/// internal fields without retags.
pub(crate) enum NodeKindRaw<K: OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize>
{
	Internal(*const InternalNode<K, V, IC, LC>),
	Leaf(*const LeafNode<K, V, LC>),
}

impl<
		K: fmt::Debug + OptimisticRead,
		V: fmt::Debug + OptimisticRead,
		const IC: usize,
		const LC: usize,
	> fmt::Debug for Node<K, V, IC, LC>
{
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Node::Internal(ref internal) => f.debug_tuple("Internal").field(internal).finish(),
			Node::Leaf(ref leaf) => f.debug_tuple("Leaf").field(leaf).finish(),
		}
	}
}

impl<K: OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize> Node<K, V, IC, LC> {
	/// Returns `true` if this is a leaf node.
	#[inline]
	pub(crate) fn is_leaf(&self) -> bool {
		matches!(self, Node::Leaf(_))
	}

	/// Reads the variant discriminant via raw pointer projection and
	/// returns a [`NodeKindRaw`] carrying a `*const` to the variant's
	/// inner type — without creating an `&Node` reborrow.
	///
	/// Used by the optimistic read fast path; see [`OptimisticGuard::as_ptr`]
	/// and the safety contract on [`crate::optimistic::OptimisticRead`].
	///
	/// # Safety
	///
	/// `this` must be a valid pointer to a `Node` owned by a
	/// [`HybridLatch`] the caller holds an [`OptimisticGuard`] on. The
	/// returned variant data pointers are only valid for the lifetime of
	/// that guard.
	///
	/// The discriminant read is bitwise; if a concurrent writer is
	/// mid-replacement the value may be torn, but the caller's
	/// `recheck()` validates this. The cast from the variant pointer to
	/// the inner type's pointer relies on `#[repr(C, u8)]` placing the
	/// payload at a known offset.
	#[inline]
	pub(crate) unsafe fn variant_raw(this: *const Self) -> NodeKindRaw<K, V, IC, LC> {
		// `#[repr(C, u8)]` lays the discriminant first, padded to the
		// alignment of the variant data. The payload starts at offset
		// equal to the alignment of the largest variant.
		//
		// SAFETY: `this` is a valid pointer per the caller; the u8
		// discriminant is at the very start.
		// SAFETY: see the function-level safety contract.
		let tag = unsafe { ptr::read(this as *const u8) };
		// Compute the payload offset: it sits at `align_of::<Self>()`
		// from the start (the discriminant is in the same bucket but
		// padded out).
		let payload_offset = core::mem::align_of::<Self>();
		// SAFETY: see `tag` above; `payload_offset` is the documented
		// `#[repr(C, u8)]` layout offset.
		// SAFETY: see the function-level safety contract.
		let payload = unsafe { (this as *const u8).add(payload_offset) };
		match tag {
			0 => NodeKindRaw::Internal(payload.cast::<InternalNode<K, V, IC, LC>>()),
			1 => NodeKindRaw::Leaf(payload.cast::<LeafNode<K, V, LC>>()),
			// SAFETY: only two variants exist; under a torn read the
			// caller's recheck will fail. We treat any unknown tag as
			// the leaf variant so the caller observes a consistent
			// (but invalid) state until recheck triggers a retry.
			_ => {
				std::hint::cold_path();
				NodeKindRaw::Leaf(payload.cast::<LeafNode<K, V, LC>>())
			}
		}
	}

	/// Mutable raw-pointer projection from a `*mut Node` to its inner
	/// `*mut LeafNode`, without creating any `&mut Node` reborrow.
	///
	/// Used by writer code paths that hold the leaf's exclusive lock
	/// (via [`crate::latch::ExclusiveGuard`]) but want to mutate the
	/// leaf's atomic mirror without establishing a Tree-Borrows
	/// "Reserved" tag on the surrounding node — that tag would conflict
	/// with concurrent optimistic readers' atomic loads from sibling
	/// `*const Node` raw pointers.
	///
	/// # Safety
	///
	/// - `this` must be a valid pointer to a `Node` whose current
	///   variant is `Node::Leaf` (caller's responsibility — usually
	///   guaranteed by the tree's structural invariants and the
	///   exclusive lock).
	/// - Caller must hold the exclusive lock so no concurrent variant
	///   change happens.
	#[inline]
	pub(crate) unsafe fn as_leaf_ptr_mut(this: *mut Self) -> *mut LeafNode<K, V, LC> {
		debug_assert!(matches!(
			// SAFETY: see the function-level safety contract.
			unsafe { Self::variant_raw(this as *const Self) },
			NodeKindRaw::Leaf(_)
		));
		let payload_offset = core::mem::align_of::<Self>();
		// SAFETY: `#[repr(C, u8)]` places the variant payload at
		// `align_of::<Self>()`; `this` is a valid mutable pointer per
		// caller's invariant.
		// SAFETY: see the function-level safety contract.
		let payload = unsafe { (this as *mut u8).add(payload_offset) };
		payload.cast::<LeafNode<K, V, LC>>()
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
	pub(crate) fn keys(&self) -> Vec<K>
	where
		K: Clone,
	{
		match self {
			Node::Internal(ref internal) => internal.keys.iter().cloned().collect(),
			Node::Leaf(ref leaf) => {
				let len = leaf.len.load() as usize;
				(0..len)
					// SAFETY: see the function-level safety contract.
					.map(|i| unsafe { SlotArray::load_raw(ptr::addr_of!(leaf.keys), i) })
					.collect()
			}
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
					((internal.len.load() + 1 + other.len.load()) as usize) < IC
				}
				_ => false, // Can't merge internal with leaf
			},
			Node::Leaf(ref leaf) => match other {
				Node::Leaf(ref other) => {
					// Leaf merge doesn't add a separator key
					((leaf.len.load() + other.len.load()) as usize) < LC
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
#[repr(C, align(64))]
pub(crate) struct LeafNode<K: OptimisticRead, V: OptimisticRead, const LC: usize> {
	/// Number of key-value pairs in this leaf.
	///
	/// Atomic so that the optimistic-read fast path can load it
	/// concurrently with writer updates without violating the C/Rust
	/// memory model's data-race rules. Writers (under exclusive lock)
	/// use `Release` stores; readers use `Acquire` loads.
	pub(crate) len: AtomicLen,
	/// Atomic key storage. Writers under exclusive lock use
	/// `SlotArray::shift_insert_raw` / `shift_remove_raw` to mutate;
	/// readers (optimistic or shared-lock) use atomic `Acquire` loads
	/// via `SlotArray::try_load_raw` / `load_raw`. The atomic load /
	/// store discipline satisfies both Tree Borrows and the C/Rust
	/// memory model's data-race detector, so the optimistic fast path
	/// can read concurrently with a writer's mutation under exclusive
	/// lock without UB.
	pub(crate) keys: SlotArray<K::Slot, LC>,
	/// Atomic value storage. Same discipline as `keys`.
	pub(crate) values: SlotArray<V::Slot, LC>,
	/// Exclusive lower bound - keys in this leaf are > lower_fence.
	/// None means this is the leftmost leaf (no lower bound).
	pub(crate) lower_fence: Option<K>,
	/// Inclusive upper bound - keys in this leaf are <= upper_fence.
	/// None means this is the rightmost leaf (no upper bound).
	pub(crate) upper_fence: Option<K>,
	/// A key that routes to this leaf, used for relocation after splits.
	pub(crate) sample_key: Option<K>,
}

impl<K: fmt::Debug + OptimisticRead, V: fmt::Debug + OptimisticRead, const LC: usize> fmt::Debug
	for LeafNode<K, V, LC>
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("LeafNode")
			.field("len", &self.len.load_relaxed())
			.field("lower_fence", &self.lower_fence)
			.field("upper_fence", &self.upper_fence)
			.field("sample_key", &self.sample_key)
			.finish()
	}
}

impl<K: OptimisticRead, V: OptimisticRead, const LC: usize> LeafNode<K, V, LC> {
	/// Creates a new, empty leaf node.
	pub fn new() -> LeafNode<K, V, LC> {
		LeafNode {
			len: AtomicLen::new(0),
			keys: SlotArray::new(),
			values: SlotArray::new(),
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
	/// - `exact_match`: `true` if `entries[position].0 == key`
	///
	/// # Concurrency Safety
	///
	/// Uses safe bounds checking to handle concurrent access. Under optimistic
	/// locking, `self.len.load_relaxed()` may be inconsistent with `self.entries.len()` during
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
				return (self.len.load_relaxed(), false);
			}
		}

		let mut lower = 0;
		let mut upper = self.len.load_relaxed().min(LC as u16);

		while lower < upper {
			let mid = ((upper - lower) / 2) + lower;

			// Atomic load of the key at position `mid` as a borrow. For
			// boxed K the borrow points into the slot's `Box<K>` (zero
			// clone); for inline K the bits are loaded into `buf` and
			// the borrow points there. The shared / exclusive lock on
			// the leaf keeps the slot init for the duration of the
			// comparison.
			let mut buf: core::mem::MaybeUninit<K> = core::mem::MaybeUninit::uninit();
			// SAFETY: `mid < upper <= LC`; slot is init under our guard.
			let mid_key: &K = unsafe { self.keys.load_into(mid as usize, &mut buf) };

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

	// =====================================================================
	// Raw-pointer projection (optimistic-read fast path)
	// =====================================================================

	/// Reads `len` via raw pointer projection (no `&self` reborrow).
	///
	/// # Safety
	///
	/// `this` must be a valid pointer to a `LeafNode` held under an
	/// [`OptimisticGuard`]. Caller must `recheck()` before acting on the
	/// result.
	#[inline]
	pub(crate) unsafe fn len_raw(this: *const Self) -> u16 {
		// SAFETY: `len` is a u16 at a known field offset; the read is an
		// unsynchronised aligned load with no retag.
		// SAFETY: see the function-level safety contract.
		unsafe { AtomicLen::load_raw(ptr::addr_of!((*this).len)) }
	}

	/// Binary search for `key` via raw pointer projection, without
	/// creating an `&LeafNode` or `&InlineVec` reborrow.
	///
	/// Caller MUST `recheck()` after this call to validate the result.
	///
	/// # Safety
	///
	/// `this` must point to a valid `LeafNode` held under an
	/// [`OptimisticGuard`].
	///
	/// `K: OptimisticRead` certifies that a bitwise snapshot of K is
	/// sound to use across the brief stack-local comparison window: if
	/// `K::EPOCH_DEFERRED_DROP` is true (e.g. `bytes::Bytes`) the
	/// caller must additionally have ensured all leaf-K drops are
	/// routed through the epoch GC via `insert_defer` / `remove_defer`.
	#[inline]
	pub(crate) unsafe fn lower_bound_raw<Q>(this: *const Self, key: &Q) -> (u16, bool)
	where
		K: Borrow<Q> + Ord + OptimisticRead,
		Q: ?Sized + Ord,
	{
		// Fence checks via raw projection.
		// SAFETY: `lower_fence` is `Option<K>` at a known field offset;
		// `addr_of!` does not reborrow.
		// SAFETY: see the function-level safety contract.
		let lower_fence: *const Option<K> = unsafe { ptr::addr_of!((*this).lower_fence) };
		// SAFETY: see `lower_fence` above.
		let upper_fence: *const Option<K> = unsafe { ptr::addr_of!((*this).upper_fence) };

		// We bitwise-read the Option<K> into a ManuallyDrop on the stack so
		// the inner K's `Drop` is suppressed if recheck fails / on early
		// return. `&` borrows from here on are to stack memory, not the
		// shared node — no retag race.
		// SAFETY: `lower_fence` is a valid pointer to `Option<K>`. The
		// bitwise read is sound for `K: OptimisticRead` (permits torn
		// snapshot, validated by recheck-or-discard via ManuallyDrop).
		let lf_snapshot: core::mem::ManuallyDrop<Option<K>> =
			// SAFETY: see the function-level safety contract.
			unsafe { core::mem::ManuallyDrop::new(ptr::read(lower_fence)) };
		if let Some(fk) = lf_snapshot.as_ref() {
			if key < fk.borrow() {
				return (0, false);
			}
		}

		// SAFETY: see `Self::len_raw`.
		let len = unsafe { Self::len_raw(this) };
		// SAFETY: see `lf_snapshot` above.
		let uf_snapshot: core::mem::ManuallyDrop<Option<K>> =
			// SAFETY: see the function-level safety contract.
			unsafe { core::mem::ManuallyDrop::new(ptr::read(upper_fence)) };
		if let Some(fk) = uf_snapshot.as_ref() {
			if key > fk.borrow() {
				return (len, false);
			}
		}

		// Binary search over the entries array via raw pointer arithmetic.
		// Bound `upper` by the InlineVec's compile-time capacity LC since
		// `len` may be inconsistent under concurrent mutation (recheck
		// will catch).
		// SAFETY: `keys` is a `SlotArray<K::Slot, LC>` at a known
		// field offset; raw projection without `&LeafNode` reborrow.
		// SAFETY: see the function-level safety contract.
		let keys_ptr: *const SlotArray<K::Slot, LC> = unsafe { ptr::addr_of!((*this).keys) };
		let mut lower: u16 = 0;
		let mut upper: u16 = len.min(LC as u16);

		while lower < upper {
			let mid = ((upper - lower) / 2) + lower;

			// Atomic-load the K at position mid as a borrow via raw-
			// pointer projection. For boxed K the borrow points into
			// the slot's `Box<K>` (kept alive by the epoch guard the
			// caller holds); for inline K the bits are loaded into the
			// stack buffer and the borrow points into it. Either way,
			// no clone — and no Drop concern, because we hold a borrow
			// rather than an owned snapshot.
			//
			// `try_load_into_raw` returns `None` if the slot was
			// concurrently emptied (boxed null pointer race); treat
			// that as a recheck signal and bail with conservative
			// bounds. For inline storage this branch is unreachable.
			//
			// SAFETY: `mid < upper <= LC`; outer recheck validates the
			// snapshot.
			let mut buf: core::mem::MaybeUninit<K> = core::mem::MaybeUninit::uninit();
			// SAFETY: see the function-level safety contract.
			let mid_key_opt =
				unsafe { SlotArray::try_load_into_raw(keys_ptr, mid as usize, &mut buf) };
			let mid_key = match mid_key_opt {
				Some(k) => k,
				None => {
					std::hint::cold_path();
					return (lower, false);
				}
			};

			if key < mid_key.borrow() {
				upper = mid;
			} else if key > mid_key.borrow() {
				lower = mid + 1;
			} else {
				return (mid, true);
			}
		}

		(lower, false)
	}

	/// Atomic-loads the value at the given position.
	///
	/// Returns an owned `V` (cloned via the slot's atomic load semantics —
	/// for inline storage this is a Copy of the bits; for boxed storage
	/// it clones through the Acquire-loaded `Box<V>` pointer).
	///
	/// # Concurrency Safety
	///
	/// Returns `Err(Unwind)` if position is out of bounds or if a
	/// concurrent writer has temporarily emptied the slot (boxed-storage
	/// path). The caller's recheck triggers a retry.
	#[inline]
	pub(crate) fn value_at(&self, pos: u16) -> error::Result<V> {
		if pos as usize >= self.len.load_relaxed() as usize {
			return Err(error::Error::Unwind);
		}
		// SAFETY: pos < len; slot is init under our shared / exclusive lock.
		unsafe { SlotArray::try_load_raw(ptr::addr_of!(self.values), pos as usize) }
			.ok_or(error::Error::Unwind)
	}

	/// Atomic-loads the key at the given position. Same semantics as
	/// [`value_at`](Self::value_at) but for keys.
	#[inline]
	pub(crate) fn key_at(&self, pos: u16) -> error::Result<K> {
		if pos as usize >= self.len.load_relaxed() as usize {
			return Err(error::Error::Unwind);
		}
		// SAFETY: see the function-level safety contract.
		unsafe { SlotArray::try_load_raw(ptr::addr_of!(self.keys), pos as usize) }
			.ok_or(error::Error::Unwind)
	}

	/// Atomic-loads the key and value at the given position.
	#[inline]
	#[allow(dead_code)]
	pub(crate) fn kv_at(&self, pos: u16) -> error::Result<(K, V)> {
		let k = self.key_at(pos)?;
		let v = self.value_at(pos)?;
		Ok((k, v))
	}

	/// Atomic-loads (K, V) at the given position without bounds checking.
	///
	/// # Safety
	///
	/// Caller must ensure `pos < self.len.load_relaxed()` and that the
	/// slot at `pos` is currently init (i.e. the leaf is held under a
	/// shared or exclusive lock).
	#[inline]
	pub(crate) unsafe fn kv_at_unchecked(&self, pos: u16) -> (K, V) {
		// SAFETY: caller guarantees pos < len and init.
		let k = unsafe { SlotArray::load_raw(ptr::addr_of!(self.keys), pos as usize) };
		// SAFETY: see the function-level safety contract.
		let v = unsafe { SlotArray::load_raw(ptr::addr_of!(self.values), pos as usize) };
		(k, v)
	}

	/// Returns `true` if there's room for another entry.
	#[inline]
	pub(crate) fn has_space(&self) -> bool {
		(self.len.load_relaxed() as usize) < LC
	}

	/// Returns `true` if the node is below minimum occupancy (40% of capacity).
	#[inline]
	pub(crate) fn is_underfull(&self) -> bool {
		(self.len.load_relaxed() as usize) * 10 < LC * 4
	}

	/// Removes and returns the key-value pair at the specified position.
	///
	/// Maintains the atomic mirror in lock-step so concurrent optimistic
	/// readers observe a consistent view. The displaced mirror entries
	/// are routed through the epoch GC so an in-flight reader's
	/// `Acquire`-loaded pointer stays valid until no reader could still
	/// be using it.
	///
	/// Takes `*mut Self` (not `&mut self`) so that no Tree-Borrows
	/// "Reserved" tag is established on the leaf — concurrent
	/// optimistic readers' atomic reads stay foreign-safe while the
	/// writer is mid-mutation.
	///
	/// # Safety
	///
	/// - `this` must be a valid `*mut LeafNode` owned by an
	///   [`crate::latch::ExclusiveGuard`] (i.e. the caller holds the
	///   exclusive lock).
	/// - `pos < len`.
	pub(crate) unsafe fn remove_at_raw(this: *mut Self, pos: u16, eg: &epoch::Guard) -> (K, V) {
		// SAFETY: caller holds the exclusive lock on this leaf — the lock
		// release supplies the Release fence; a Relaxed load is sufficient
		// while the writer is the sole observer.
		let len = unsafe { AtomicLen::load_raw_relaxed(ptr::addr_of!((*this).len)) } as usize;
		// SAFETY: caller holds exclusive lock; pos < len <= LC.
		let keys_ptr = unsafe { ptr::addr_of!((*this).keys) };
		// SAFETY: see the function-level safety contract.
		let values_ptr = unsafe { ptr::addr_of!((*this).values) };

		// Atomic-load a copy of (K, V) for the caller. For inline
		// storage this is a Copy; for boxed storage this clones through
		// the Acquire-loaded pointer (refcount bump for refcounted V).
		// SAFETY: see the function-level safety contract.
		let removed_k: K = unsafe { SlotArray::load_raw(keys_ptr, pos as usize) };
		// SAFETY: see the function-level safety contract.
		let removed_v: V = unsafe { SlotArray::load_raw(values_ptr, pos as usize) };

		// Shift the storage to fill the gap. The displaced owners
		// (Box<K> / Box<V> for boxed storage, K / V for inline) are
		// routed through the epoch GC so concurrent optimistic readers'
		// pointers stay valid.
		// SAFETY: see the function-level safety contract.
		let displaced_k = unsafe { SlotArray::shift_remove_raw(keys_ptr, len, pos as usize) };
		// SAFETY: see the function-level safety contract.
		let displaced_v = unsafe { SlotArray::shift_remove_raw(values_ptr, len, pos as usize) };
		eg.defer(move || drop(displaced_k));
		eg.defer(move || drop(displaced_v));

		// Update len atomically.
		// SAFETY: see the function-level safety contract.
		let len_ptr: *const AtomicLen = unsafe { ptr::addr_of!((*this).len) };
		// SAFETY: caller holds the exclusive lock; Relaxed is sufficient
		// because the lock release supplies the Release fence.
		unsafe { (*len_ptr).fetch_sub_relaxed(1) };

		(removed_k, removed_v)
	}

	/// Replace the value at `pos` and return the previous value.
	/// Maintains the atomic mirror via raw-pointer projection.
	///
	/// Takes `*mut Self` (not `&mut self`) so that no Tree-Borrows
	/// "Reserved" tag is established on the leaf for the duration of
	/// the call — concurrent optimistic readers' atomic reads stay
	/// foreign-safe while the writer is mid-mutation.
	///
	/// # Safety
	///
	/// - `this` must be a valid `*mut LeafNode` owned by an
	///   [`crate::latch::ExclusiveGuard`] (i.e. the caller holds the
	///   exclusive lock).
	/// - `pos < len`.
	pub(crate) unsafe fn swap_value_at_raw(
		this: *mut Self,
		pos: u16,
		value: V,
		eg: &epoch::Guard,
	) -> V {
		// SAFETY: caller holds the exclusive lock — Relaxed load suffices.
		let len = unsafe { AtomicLen::load_raw_relaxed(ptr::addr_of!((*this).len)) } as usize;
		debug_assert!((pos as usize) < len);
		// Atomic load + swap on the values slot. For inline storage the
		// swap is one AcqRel atomic op; for boxed storage it allocates
		// a fresh `Box<V>` from `value` and AcqRel-swaps the AtomicPtr.
		// SAFETY: see the function-level safety contract.
		let values_ptr = unsafe { ptr::addr_of!((*this).values) };
		// Capture a copy of the old V to return to the caller.
		// SAFETY: see the function-level safety contract.
		let old_v: V = unsafe { SlotArray::load_raw(values_ptr, pos as usize) };
		// Now swap in the new V. The displaced owner is routed through
		// the epoch GC so concurrent optimistic readers still hold
		// valid pointers until the next epoch tick.
		// SAFETY: see the function-level safety contract.
		let displaced = unsafe { SlotArray::swap_init_raw(values_ptr, pos as usize, value) };
		eg.defer(move || drop(displaced));
		old_v
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

impl<K: Clone + OptimisticRead, V: Clone + OptimisticRead, const LC: usize> LeafNode<K, V, LC> {
	/// Inserts a key-value pair at the specified position.
	///
	/// Maintains the atomic mirror so the optimistic-read fast path
	/// observes the new entry under `Acquire`-load semantics.
	///
	/// # Returns
	///
	/// - `Some(pos)` if insertion succeeded
	/// - `None` if the node is full
	///
	/// Takes `*mut Self` (not `&mut self`) so that no Tree-Borrows
	/// "Reserved" tag is established on the leaf — concurrent
	/// optimistic readers' atomic reads stay foreign-safe while the
	/// writer is mid-mutation.
	///
	/// # Safety
	///
	/// - `this` must be a valid `*mut LeafNode` owned by an
	///   [`crate::latch::ExclusiveGuard`].
	/// - `pos <= len`.
	pub(crate) unsafe fn insert_at_raw(this: *mut Self, pos: u16, key: K, value: V) -> Option<u16> {
		// SAFETY: see the function-level safety contract.
		let len_ptr: *const AtomicLen = unsafe { ptr::addr_of!((*this).len) };
		// SAFETY: caller holds the exclusive lock — Relaxed load suffices.
		let len = unsafe { AtomicLen::load_raw_relaxed(len_ptr) } as usize;
		if len >= LC {
			return None;
		}

		// sample_key check via raw projection.
		// SAFETY: see the function-level safety contract.
		let sample_key_ptr: *mut Option<K> = unsafe { ptr::addr_of_mut!((*this).sample_key) };
		// SAFETY: under exclusive lock; no concurrent writer.
		if unsafe { (*sample_key_ptr).is_none() } {
			// SAFETY: see the function-level safety contract.
			unsafe { ptr::write(sample_key_ptr, Some(key.clone())) };
		}

		// Storage updates via raw-pointer projection; pos <= len < LC.
		// SAFETY: see the function-level safety contract.
		let keys_ptr = unsafe { ptr::addr_of!((*this).keys) };
		// SAFETY: see the function-level safety contract.
		let values_ptr = unsafe { ptr::addr_of!((*this).values) };
		// SAFETY: see the function-level safety contract.
		unsafe { SlotArray::shift_insert_raw(keys_ptr, len, pos as usize, key) };
		// SAFETY: see the function-level safety contract.
		unsafe { SlotArray::shift_insert_raw(values_ptr, len, pos as usize, value) };

		// Update len atomically.
		// SAFETY: caller holds the exclusive lock; Relaxed increment is
		// sufficient — the lock release supplies the Release fence.
		unsafe { (*len_ptr).fetch_add_relaxed(1) };

		Some(pos)
	}
}

impl<K: Clone + OptimisticRead, V: OptimisticRead, const LC: usize> LeafNode<K, V, LC> {
	/// Splits this leaf node, moving entries after `split_pos` to `right`.
	///
	/// After split:
	/// - `self` (left) contains entries `[0, split_pos]`
	/// - `right` contains entries `[split_pos + 1, len)`
	/// - `split_key = entries[split_pos].0` becomes the separator
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
	/// - `split_pos`: Position of the entry whose key becomes the separator
	pub(crate) fn split(&mut self, right: &mut LeafNode<K, V, LC>, split_pos: u16) {
		// Total entries in self before the split.
		let total = self.len.load_relaxed() as usize;
		let right_start = (split_pos + 1) as usize;
		let right_count = total - right_start;

		// Atomic-load the split key. For boxed K this clones through
		// the Acquire-loaded pointer; for inline K it's an atomic copy.
		let self_keys_ptr: *const _ = ptr::addr_of!(self.keys);
		// SAFETY: see the function-level safety contract.
		let split_key: K = unsafe { SlotArray::load_raw(self_keys_ptr, split_pos as usize) };

		// Update fence keys.
		right.lower_fence = Some(split_key.clone());
		right.upper_fence = self.upper_fence.clone();
		self.upper_fence = Some(split_key);

		// Move storage [right_start..total) from self to right[0..right_count).
		// SAFETY: both leaves held under exclusive lock; src slots init,
		// dst slots empty (right is newly allocated).
		let right_ptr: *const Self = right;
		let self_keys = self_keys_ptr;
		let self_values = ptr::addr_of!(self.values);
		// SAFETY: see the function-level safety contract.
		let right_keys = unsafe { ptr::addr_of!((*right_ptr).keys) };
		// SAFETY: see the function-level safety contract.
		let right_values = unsafe { ptr::addr_of!((*right_ptr).values) };
		for (dst_idx, src_pos) in (right_start..total).enumerate() {
			// SAFETY: see the function-level safety contract.
			unsafe {
				SlotArray::move_raw(self_keys, src_pos, right_keys, dst_idx);
				SlotArray::move_raw(self_values, src_pos, right_values, dst_idx);
			}
		}

		// Set sample keys for node relocation (load first key of each leaf).
		// SAFETY: both leaves have at least one entry post-split.
		let self_first: K = unsafe { SlotArray::load_raw(self_keys, 0) };
		// SAFETY: see the function-level safety contract.
		let right_first: K = unsafe { SlotArray::load_raw(right_keys, 0) };
		self.sample_key = Some(self_first);
		right.sample_key = Some(right_first);

		// Update lengths.
		// SAFETY: both leaves are held under their respective exclusive
		// locks (taken by the caller via `&mut self` / `&mut right`); the
		// lock releases supply the Release fences.
		unsafe {
			self.len.store_relaxed(right_start as u16); // self retains [0..right_start)
			right.len.store_relaxed(right_count as u16);
		}
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
		let self_len = self.len.load_relaxed() as usize;
		let right_len = right.len.load_relaxed() as usize;
		// Check if combined entries fit
		if self_len + right_len > LC {
			return false;
		}

		// Inherit right's upper fence (we now cover its range too)
		self.upper_fence = right.upper_fence.take();

		// Move right's storage [0..right_len) to self[self_len..self_len + right_len).
		// SAFETY: both leaves under exclusive lock; src slots init in
		// `right`, dst slots empty in `self` (beyond self_len).
		let self_keys = ptr::addr_of!(self.keys);
		let self_values = ptr::addr_of!(self.values);
		let right_ptr: *const Self = right;
		// SAFETY: see the function-level safety contract.
		let right_keys = unsafe { ptr::addr_of!((*right_ptr).keys) };
		// SAFETY: see the function-level safety contract.
		let right_values = unsafe { ptr::addr_of!((*right_ptr).values) };
		for src_idx in 0..right_len {
			let dst_idx = self_len + src_idx;
			// SAFETY: see the function-level safety contract.
			unsafe {
				SlotArray::move_raw(right_keys, src_idx, self_keys, dst_idx);
				SlotArray::move_raw(right_values, src_idx, self_values, dst_idx);
			}
		}

		// Mark right as empty.
		// SAFETY: caller holds exclusive locks on both leaves; Relaxed
		// stores are sufficient — the lock releases supply Release fences.
		unsafe { right.len.store_relaxed(0) };

		// Update sample_key: prefer right's sample_key if available,
		// otherwise ensure we have one if we have entries (prevents find_parent failures)
		if let Some(sample) = right.sample_key.take() {
			self.sample_key = Some(sample);
		} else if self.sample_key.is_none() && (self_len + right_len) > 0 {
			// Load first key for sample_key.
			// SAFETY: at least one entry exists in merged storage.
			let first_key: K = unsafe { SlotArray::load_raw(self_keys, 0) };
			self.sample_key = Some(first_key);
		}

		// Update length to combined size.
		// SAFETY: see the store_relaxed comment above.
		unsafe { self.len.store_relaxed((self_len + right_len) as u16) };
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
#[repr(C, align(64))]
pub(crate) struct InternalNode<
	K: OptimisticRead,
	V: OptimisticRead,
	const IC: usize,
	const LC: usize,
> {
	/// Number of keys (and regular edges) in this node.
	///
	/// Atomic so that readers descending through this node (including
	/// shared-lock readers which hold only an optimistic guard on the
	/// internal node) can load it without violating the data-race rules.
	pub(crate) len: AtomicLen,
	/// Separator keys, sorted in ascending order.
	///
	/// Backed by [`InlineVec`] — see the comment on `LeafNode::entries`.
	pub(crate) keys: InlineVec<K, IC>,
	/// Child pointers corresponding to keys.
	/// `edges[i]` points to subtree with keys < `keys[i]`.
	pub(crate) edges: InlineVec<Atomic<HybridLatch<Node<K, V, IC, LC>>>, IC>,
	/// Rightmost child pointer, for keys >= last key.
	/// This is separate because we have N+1 children for N keys.
	///
	/// Uses a null-pointer sentinel rather than `Option<Atomic<…>>` so the
	/// optimistic-read fast path can project to it via raw pointer
	/// arithmetic without the indeterminate-offset issue of the Option's
	/// tagged discriminant. A null load means "no upper edge".
	pub(crate) upper_edge: Atomic<HybridLatch<Node<K, V, IC, LC>>>,
	/// Exclusive lower bound for keys routed through this node.
	pub(crate) lower_fence: Option<K>,
	/// Inclusive upper bound for keys routed through this node.
	pub(crate) upper_fence: Option<K>,
	/// Sample key for node relocation.
	pub(crate) sample_key: Option<K>,
}

impl<K: fmt::Debug + OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize> fmt::Debug
	for InternalNode<K, V, IC, LC>
{
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("InternalNode")
			.field("len", &self.len.load_relaxed())
			.field("keys", &self.keys)
			.field("edges", &self.edges)
			.field("upper_edge", &self.upper_edge)
			.field("lower_fence", &self.lower_fence)
			.field("upper_fence", &self.upper_fence)
			.field("sample_key", &self.sample_key)
			.finish()
	}
}

impl<K: OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize>
	InternalNode<K, V, IC, LC>
{
	/// Creates a new, empty internal node.
	pub(crate) fn new() -> InternalNode<K, V, IC, LC> {
		InternalNode {
			len: AtomicLen::new(0),
			keys: InlineVec::new(),
			edges: InlineVec::new(),
			upper_edge: Atomic::null(),
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
				return (self.len.load_relaxed(), false);
			}
		}

		// Use actual keys length for safe bounds - handles concurrent modifications
		let keys_len = self.keys.len() as u16;
		let mut lower = 0;
		let mut upper = self.len.load_relaxed().min(keys_len);

		while lower < upper {
			let mid = ((upper - lower) / 2) + lower;

			// Safe bounds check - concurrent modifications may cause len > keys.len()
			let Some(mid_key) = self.keys.get(mid as usize) else {
				// Index out of bounds due to concurrent modification - return
				// conservative result. Cold-path hint keeps the hot binary
				// search body straight-line.
				std::hint::cold_path();
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

	// =====================================================================
	// Raw-pointer projection (optimistic-read fast path)
	// =====================================================================

	/// Reads `len` via raw pointer projection (no `&self` reborrow).
	///
	/// # Safety
	///
	/// `this` must be a valid pointer to an `InternalNode` held under an
	/// [`OptimisticGuard`]. Caller must `recheck()` before acting on it.
	#[inline]
	pub(crate) unsafe fn len_raw(this: *const Self) -> u16 {
		// SAFETY: `len` is a u16 at a known field offset.
		unsafe { AtomicLen::load_raw(ptr::addr_of!((*this).len)) }
	}

	/// Returns a raw pointer to the keys array via projection.
	#[inline]
	#[allow(dead_code)]
	pub(crate) unsafe fn keys_ptr_raw(this: *const Self) -> *const K {
		// SAFETY: see Self::len_raw.
		unsafe { InlineVec::raw_data_ptr(ptr::addr_of!((*this).keys)) }
	}

	/// Returns a raw pointer to the edges array via projection.
	#[inline]
	pub(crate) unsafe fn edges_ptr_raw(
		this: *const Self,
	) -> *const Atomic<HybridLatch<Node<K, V, IC, LC>>> {
		// SAFETY: see Self::len_raw.
		unsafe { InlineVec::raw_data_ptr(ptr::addr_of!((*this).edges)) }
	}

	/// Returns a raw pointer to the `Atomic` swip at position `pos`,
	/// or `Err(Unwind)` if `pos == len` and `upper_edge` is None / out
	/// of bounds.
	///
	/// # Safety
	///
	/// `this` must be a valid pointer to an `InternalNode` held under an
	/// [`OptimisticGuard`]. Caller must `recheck()` before dereferencing
	/// the returned pointer through epoch::Atomic load.
	#[inline]
	pub(crate) unsafe fn edge_at_raw(
		this: *const Self,
		pos: u16,
	) -> error::Result<*const Atomic<HybridLatch<Node<K, V, IC, LC>>>> {
		// SAFETY: same source pointer as len_raw / edges_ptr_raw.
		let len = unsafe { Self::len_raw(this) };
		if pos == len {
			// Rightmost child → upper_edge. `upper_edge` is a plain
			// `Atomic<...>` (with null sentinel for "no upper edge"),
			// so we can return a pointer to it directly via addr_of!.
			// The caller's subsequent `Atomic::load` checks for null.
			//
			// SAFETY: `upper_edge` is at a known field offset; addr_of!
			// does not reborrow.
			let upper_edge_ptr: *const Atomic<HybridLatch<Node<K, V, IC, LC>>> =
				// SAFETY: see the function-level safety contract.
				unsafe { ptr::addr_of!((*this).upper_edge) };
			Ok(upper_edge_ptr)
		} else if pos < IC as u16 {
			// SAFETY: same source pointer as len_raw.
			let edges_ptr = unsafe { Self::edges_ptr_raw(this) };
			// SAFETY: pos < IC, edges_ptr valid for at least IC.
			Ok(unsafe { edges_ptr.add(pos as usize) })
		} else {
			Err(error::Error::Unwind)
		}
	}

	/// Binary search for the child edge to follow for `key`, via raw
	/// pointer projection. Caller MUST `recheck()` after this call.
	///
	/// # Safety
	///
	/// See [`Self::edge_at_raw`]. `K: OptimisticRead` certifies the
	/// bitwise-snapshot-and-compare discipline (see `LeafNode::lower_bound_raw`).
	#[inline]
	pub(crate) unsafe fn lower_bound_raw<Q>(this: *const Self, key: &Q) -> (u16, bool)
	where
		K: Borrow<Q> + Ord + OptimisticRead,
		Q: ?Sized + Ord,
	{
		// SAFETY: fence reads via projection — addr_of! does not reborrow.
		let lower_fence: *const Option<K> = unsafe { ptr::addr_of!((*this).lower_fence) };
		// SAFETY: see `lower_fence` above.
		let upper_fence: *const Option<K> = unsafe { ptr::addr_of!((*this).upper_fence) };

		// SAFETY: bitwise read of Option<K>; `K: OptimisticRead`
		// permits torn snapshot, ManuallyDrop suppresses Drop until
		// caller validates via recheck.
		let lf_snapshot: core::mem::ManuallyDrop<Option<K>> =
			// SAFETY: see the function-level safety contract.
			unsafe { core::mem::ManuallyDrop::new(ptr::read(lower_fence)) };
		if let Some(fk) = lf_snapshot.as_ref() {
			if key < fk.borrow() {
				return (0, false);
			}
		}

		// SAFETY: see `Self::len_raw`.
		let len = unsafe { Self::len_raw(this) };
		// SAFETY: see `lf_snapshot` above.
		let uf_snapshot: core::mem::ManuallyDrop<Option<K>> =
			// SAFETY: see the function-level safety contract.
			unsafe { core::mem::ManuallyDrop::new(ptr::read(upper_fence)) };
		if let Some(fk) = uf_snapshot.as_ref() {
			if key > fk.borrow() {
				return (len, false);
			}
		}

		// SAFETY: see `Self::keys_ptr_raw`.
		let keys_ptr = unsafe { Self::keys_ptr_raw(this) };
		let mut lower: u16 = 0;
		let mut upper: u16 = len.min(IC as u16);

		while lower < upper {
			let mid = ((upper - lower) / 2) + lower;
			// SAFETY: see LeafNode::lower_bound_raw.
			let mid_key_snapshot: core::mem::ManuallyDrop<K> =
				// SAFETY: see the function-level safety contract.
				unsafe { core::mem::ManuallyDrop::new(ptr::read(keys_ptr.add(mid as usize))) };
			let mid_key: &K = &mid_key_snapshot;
			if key < mid_key.borrow() {
				upper = mid;
			} else if key > mid_key.borrow() {
				lower = mid + 1;
			} else {
				return (mid, true);
			}
		}

		(lower, false)
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
		if pos == self.len.load_relaxed() {
			// Rightmost child - use upper_edge. We detect "no upper
			// edge" via a null-pointer sentinel (rather than Option's
			// None) because the optimistic fast path projects through
			// this field via raw pointers.
			//
			// SAFETY: `self.upper_edge` is the live Atomic in the node;
			// loading it under the caller's epoch guard is sound. A
			// shared / unpinned epoch guard is acceptable here because
			// we only inspect the pointer value, not the pointee.
			let eg = epoch::pin();
			let shared = self.upper_edge.load(Ordering::Relaxed, &eg);
			if shared.is_null() {
				Err(error::Error::Unwind)
			} else {
				Ok(&self.upper_edge)
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
		(self.len.load_relaxed() as usize) < IC
	}

	/// Returns `true` if the node is below minimum occupancy (40%).
	#[inline]
	pub(crate) fn is_underfull(&self) -> bool {
		(self.len.load_relaxed() as usize) * 10 < IC * 4
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
			// SAFETY: writer holds `&mut self` ⇒ exclusive lock; Relaxed
			// increment suffices, the lock release supplies the Release fence.
			unsafe { self.len.fetch_add_relaxed(1) };
		}
		Some(pos)
	}

	/// Removes the key and edge at the given position.
	///
	/// The displaced K is routed through the epoch GC so the heap buffer
	/// behind a boxed K (e.g. `Vec<u8>`, `String`) survives until every
	/// concurrent optimistic reader has dropped its snapshot. Without
	/// this, a reader inside `InternalNode::lower_bound_raw`'s binary
	/// search holds a `ptr::read` snapshot of `(ptr, len, cap)` whose
	/// `ptr` would dangle after the synchronous drop, and the
	/// subsequent `memcmp` would read freed memory. See issue #15.
	pub(crate) fn remove_at(
		&mut self,
		pos: u16,
		eg: &epoch::Guard,
	) -> Atomic<HybridLatch<Node<K, V, IC, LC>>> {
		let key = self.keys.remove(pos as usize);
		let edge = self.edges.remove(pos as usize);
		// SAFETY: see the fetch_add_relaxed in `insert`.
		unsafe { self.len.fetch_sub_relaxed(1) };

		// `drop_or_defer` is an immediate drop for inline K (no interior
		// pointer) and routes through `eg.defer` for boxed K. The branch
		// is a `const` on `K::EPOCH_DEFERRED_DROP`, elided at monomorph.
		optimistic::drop_or_defer(key, eg);

		edge
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
		// SAFETY: writer holds `&mut self` ⇒ exclusive lock.
		unsafe { self.len.fetch_add_relaxed(1) };
	}
}

impl<K: Clone + OptimisticRead, V: OptimisticRead, const IC: usize, const LC: usize>
	InternalNode<K, V, IC, LC>
{
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
	///
	/// The `eg` parameter is used to route synchronously-displaced K
	/// values through the epoch GC so concurrent optimistic readers
	/// holding a `ptr::read` snapshot (in `InternalNode::lower_bound_raw`)
	/// don't observe a freed interior buffer. See issue #15.
	pub(crate) fn split(
		&mut self,
		right: &mut InternalNode<K, V, IC, LC>,
		split_pos: u16,
		eg: &epoch::Guard,
	) {
		// Get the split key - this will be pushed up to the parent
		let split_key =
			self.key_at(split_pos).expect("split position must be within node bounds").clone();

		// Update fence keys. `right` is freshly allocated so its fences
		// are None; the only fence whose displaced value matters is
		// `self.upper_fence`, which a concurrent reader may have a
		// `ptr::read` snapshot of.
		right.lower_fence = Some(split_key.clone());
		right.upper_fence = self.upper_fence.clone();
		if let Some(k) = self.upper_fence.replace(split_key) {
			optimistic::drop_or_defer(k, eg);
		}

		// Move keys and edges after split_pos to right
		assert!(right.keys.is_empty());
		assert!(right.edges.is_empty());
		right.keys.extend(self.keys.drain((split_pos + 1) as usize..));
		right.edges.extend(self.edges.drain((split_pos + 1) as usize..));

		// Right gets our upper_edge (it's now the rightmost in its range).
		// We use mem::replace with Atomic::null() since upper_edge no
		// longer uses Option's None.
		right.upper_edge = std::mem::replace(&mut self.upper_edge, Atomic::null());

		// The edge at split_pos becomes our new upper_edge
		// (it was pointing to children between K(split_pos-1) and K(split_pos))
		self.upper_edge =
			self.edges.pop().expect("edges non-empty: split requires at least one edge");
		// Remove the key at split_pos (it's being pushed to parent). The
		// popped K may have an interior pointer being read by a concurrent
		// optimistic descent; defer its drop through the epoch GC.
		let popped_key = self.keys.pop().expect("keys non-empty: split requires at least one key");
		optimistic::drop_or_defer(popped_key, eg);

		// Set sample keys for node relocation. The pre-existing
		// `self.sample_key` may be observed by an in-flight reader (sample
		// keys are read from internal-node descent paths under
		// `find_parent`); defer-drop the displaced value.
		if let Some(k) = self.sample_key.replace(self.keys[0].clone()) {
			optimistic::drop_or_defer(k, eg);
		}
		right.sample_key = Some(right.keys[0].clone());

		// Update lengths
		// SAFETY: both nodes are held under their exclusive locks; Relaxed
		// stores suffice — the lock releases supply Release fences.
		unsafe {
			right.len.store_relaxed(right.keys.len() as u16);
			self.len.store_relaxed(self.keys.len() as u16);
		}
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
	///
	/// The `eg` parameter is used to route synchronously-displaced K
	/// values (`self.upper_fence` and `self.sample_key`) through the
	/// epoch GC so a concurrent optimistic descent that snapshotted them
	/// via `ptr::read` doesn't observe a freed interior buffer. See
	/// issue #15.
	pub(crate) fn merge(
		&mut self,
		right: &mut InternalNode<K, V, IC, LC>,
		eg: &epoch::Guard,
	) -> bool {
		// Check if combined entries fit
		// +1 for the separator key that gets added back
		if (self.len.load_relaxed() + right.len.load_relaxed() + 1) as usize > IC {
			return false;
		}

		// Inherit right's upper_fence (we now cover its range too). The
		// displaced `self.upper_fence` may have a `ptr::read` snapshot
		// outstanding in a concurrent `lower_bound_raw`; defer-drop it.
		let displaced_upper_fence =
			std::mem::replace(&mut self.upper_fence, right.upper_fence.take());
		if let Some(k) = displaced_upper_fence {
			optimistic::drop_or_defer(k, eg);
		}

		// Our upper_edge will be used as a regular edge. Swap right's
		// upper_edge into ours; right is consumed so it doesn't matter
		// what we leave there.
		let left_upper_edge = std::mem::replace(
			&mut self.upper_edge,
			std::mem::replace(&mut right.upper_edge, Atomic::null()),
		);

		// Re-insert the separator key (was in parent, stored as right's lower_fence)
		// This key goes between our old content and right's content
		self.keys.push(
			right
				.lower_fence
				.take()
				.expect("merge requires right node to have lower_fence (separator key)"),
		);

		// Our old upper_edge becomes a regular edge (points to children < separator).
		// The node's upper_edge field is now a plain Atomic; it must not
		// be null at this point (merge contract).
		self.edges.push(left_upper_edge);

		// Append all of right's content
		self.keys.extend(right.keys.drain(..));
		self.edges.extend(right.edges.drain(..));

		// Update sample_key: prefer right's sample_key if available,
		// otherwise ensure we have one if we have keys (prevents find_parent failures).
		// The displaced `self.sample_key` may be snapshotted by a concurrent
		// reader; defer-drop it just like the fence above.
		if let Some(sample) = right.sample_key.take() {
			if let Some(k) = self.sample_key.replace(sample) {
				optimistic::drop_or_defer(k, eg);
			}
		} else if self.sample_key.is_none() && !self.keys.is_empty() {
			self.sample_key = Some(self.keys[0].clone());
		}

		// Update lengths
		// SAFETY: both nodes are held under their exclusive locks; Relaxed
		// stores suffice — the lock releases supply Release fences.
		unsafe {
			self.len.store_relaxed(self.keys.len() as u16);
			right.len.store_relaxed(0);
		}

		true
	}
}

// ===========================================================================
// Test-Only Validation Module
// ===========================================================================

/// Invariant validation for testing. Validates tree structure to ensure
/// unreachable code paths are never reached.
#[cfg(any(test, feature = "test-utils"))]
impl<
		K: Clone + Ord + std::fmt::Debug + OptimisticRead,
		V: OptimisticRead,
		const IC: usize,
		const LC: usize,
	> GenericTree<K, V, IC, LC>
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

		// SAFETY: `eg` is pinned in the caller and live for the duration of this
		// validation. No concurrent writer can run because the test calls this
		// while holding a `&mut` reference to the tree. The loaded `HybridLatch`
		// is non-null (checked above) and cannot be reclaimed until `eg` retires.
		// SAFETY: see the function-level safety contract.
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

				// Invariant 3: Key ordering. Atomic-load each key into
				// owned form for comparison.
				let leaf_len = leaf.len.load() as usize;
				let keys_owned: Vec<K> = (0..leaf_len)
					// SAFETY: see the function-level safety contract.
					.map(|i| unsafe { SlotArray::load_raw(ptr::addr_of!(leaf.keys), i) })
					.collect();
				for i in 1..leaf_len {
					assert!(
						keys_owned[i - 1] < keys_owned[i],
						"Keys not sorted at positions {} and {}: {:?} >= {:?}",
						i - 1,
						i,
						keys_owned[i - 1],
						keys_owned[i]
					);
				}

				// Invariant 4: Fence key consistency
				if let Some(lower) = &leaf.lower_fence {
					for key in &keys_owned {
						assert!(
							key > lower,
							"Key {:?} not greater than lower_fence {:?}",
							key,
							lower
						);
					}
				}
				if let Some(upper) = &leaf.upper_fence {
					for key in &keys_owned {
						assert!(key <= upper, "Key {:?} not <= upper_fence {:?}", key, upper);
					}
				}

				// Check against parent's expected bounds
				if let Some(lower) = expected_lower {
					for key in &keys_owned {
						assert!(
							key > lower,
							"Key {:?} not greater than parent lower bound {:?}",
							key,
							lower
						);
					}
				}
				if let Some(upper) = expected_upper {
					for key in &keys_owned {
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

				// Invariant 5: Upper edge presence (null = "no upper edge")
				let eg_local = epoch::pin();
				assert!(
					!internal.upper_edge.load(Ordering::Relaxed, &eg_local).is_null(),
					"Internal node at level {} has no upper_edge",
					level
				);

				// Invariant 6: Length consistency
				assert_eq!(
					internal.len.load() as usize,
					internal.keys.len(),
					"Internal len {} != keys.len() {}",
					internal.len.load(),
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
						// SAFETY: `eg` is pinned and is held for the duration of this
						// validation pass; the non-null `child_ptr` is therefore safe
						// to dereference.
						// SAFETY: see the function-level safety contract.
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

				// Validate upper_edge child (null = "no upper edge")
				let child_ptr = internal.upper_edge.load(Ordering::Acquire, eg);
				if !child_ptr.is_null() {
					// SAFETY: `eg` is pinned and is held for the duration of this
					// validation pass; the non-null `child_ptr` is therefore safe
					// to dereference.
					// SAFETY: see the function-level safety contract.
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

#[cfg(test)]
mod node_layout {
	use super::*;

	#[test]
	fn leaf_node_is_cache_line_aligned() {
		assert_eq!(core::mem::align_of::<LeafNode<i64, i64, 16>>(), 64);
		assert_eq!(core::mem::align_of::<LeafNode<String, String, 16>>(), 64);
	}

	#[test]
	fn internal_node_is_cache_line_aligned() {
		assert_eq!(core::mem::align_of::<InternalNode<i64, i64, 16, 16>>(), 64);
		assert_eq!(core::mem::align_of::<InternalNode<String, i64, 16, 16>>(), 64);
	}
}

#[cfg(test)]
mod raw_shared_iter_load_into {
	use super::*;

	#[test]
	fn raw_shared_iter_string_kv_no_clone() {
		// Builds a tree with 1000 String K/V (BoxedSlot path) and walks
		// the full range via raw_iter().next(). The iterator's
		// MaybeUninit buffers borrow into the slot's Box<String> — no
		// clone per step. We assert the keys come back in sorted order
		// and that the borrowed (&k, &v) compare equal to the inserted
		// data.
		let tree: Tree<String, String> = Tree::new();
		for i in 0..1000 {
			let k = format!("k{:06}", i);
			tree.insert(k.clone(), k);
		}

		let mut iter = tree.raw_iter();
		iter.seek_to_first();
		let mut seen = 0;
		while let Some((k, v)) = iter.next() {
			assert_eq!(k, v);
			assert_eq!(*k, format!("k{:06}", seen));
			seen += 1;
		}
		assert_eq!(seen, 1000);

		// Reverse iteration via prev() reads the same buffers.
		let mut iter = tree.raw_iter();
		iter.seek_to_last();
		let mut seen = 1000usize;
		while let Some((k, _v)) = iter.prev() {
			seen -= 1;
			assert_eq!(*k, format!("k{:06}", seen));
		}
		assert_eq!(seen, 0);
	}
}

#[cfg(test)]
mod leaf_binary_search_load_into {
	use super::*;

	#[test]
	fn leaf_lower_bound_load_into_smoke() {
		// Build a tree with String keys (BoxedSlot path) and exercise
		// the binary-search probes that now run via `load_into`.
		let tree: Tree<String, i64> = Tree::new();
		for i in 0..10 {
			tree.insert(format!("k{:04}", i), i);
		}

		// Each lookup walks a tree of String slots; the binary-search
		// probes borrow into the Box<String> rather than cloning.
		for i in 0..10 {
			let k = format!("k{:04}", i);
			assert_eq!(tree.lookup(&k, |v| *v), Some(i));
			assert_eq!(tree.lookup_optimistic(&k, |v| *v), Some(i));
		}

		// Missing keys should also drive the comparison path correctly.
		assert_eq!(tree.lookup(&"k9999".to_string(), |v| *v), None);
		assert_eq!(tree.lookup_optimistic(&"k9999".to_string(), |v| *v), None);
		assert_eq!(tree.lookup(&"a".to_string(), |v| *v), None);
		assert_eq!(tree.lookup_optimistic(&"a".to_string(), |v| *v), None);
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
	fn optimistic_lookup_basic() {
		let tree: Tree<i32, u64> = Tree::new();
		tree.insert(1, 10);
		tree.insert(2, 20);
		tree.insert(3, 30);

		// lookup_optimistic on hits and misses
		assert_eq!(tree.lookup_optimistic(&1, |v| *v), Some(10));
		assert_eq!(tree.lookup_optimistic(&2, |v| *v), Some(20));
		assert_eq!(tree.lookup_optimistic(&3, |v| *v), Some(30));
		assert_eq!(tree.lookup_optimistic(&4, |v| *v), None);

		// get_optimistic
		assert_eq!(tree.get_optimistic(&1), Some(10));
		assert_eq!(tree.get_optimistic(&99), None);

		// contains_key uses the optimistic path now
		assert!(tree.contains_key(&1));
		assert!(!tree.contains_key(&99));
	}

	#[test]
	fn optimistic_lookup_across_splits() {
		// Force multiple splits / multi-level tree to exercise the
		// optimistic descent through internal nodes.
		let tree: Tree<i64, u64> = Tree::new();
		let n: i64 = 5_000;
		for i in 0..n {
			tree.insert(i, i as u64);
		}
		tree.assert_invariants();

		for i in (0..n).step_by(37) {
			assert_eq!(tree.get_optimistic(&i), Some(i as u64));
			assert!(tree.contains_key(&i));
		}
		assert_eq!(tree.get_optimistic(&(n + 1)), None);
		assert!(!tree.contains_key(&(n + 1)));
	}

	// Mock refcounted type that opts into EPOCH_DEFERRED_DROP. The interior
	// `Arc<Vec<u8>>` makes clones cheap and exercises the deferred-drop
	// machinery used by `bytes::Bytes` etc.
	#[derive(Clone)]
	struct RefcountedBlob(std::sync::Arc<Vec<u8>>);

	// SAFETY: `RefcountedBlob` wraps `Arc<Vec<u8>>` which is `Send + Sync`.
	// A bitwise snapshot followed by a successful version recheck yields a
	// valid `Arc`. The implementor opts into `EPOCH_DEFERRED_DROP = true`,
	// which (combined with using `insert_defer`/`remove_defer` for all
	// writes) keeps the `Vec`'s buffer alive until the epoch tick.
	unsafe impl OptimisticRead for RefcountedBlob {
		const EPOCH_DEFERRED_DROP: bool = true;
		type Slot = crate::atomic_slot::BoxedSlot<Self>;
	}

	#[test]
	fn epoch_deferred_drop_basic() {
		let tree: Tree<i32, RefcountedBlob> = Tree::new();
		let blob = RefcountedBlob(std::sync::Arc::new(b"hello world".to_vec()));

		// insert_defer reports key was new.
		assert!(!tree.insert_defer(1, blob.clone()));
		// Strong count: the leaf's atomic storage holds one clone of
		// the Arc; our local `blob` is the second.
		assert_eq!(std::sync::Arc::strong_count(&blob.0), 2);

		// Optimistic lookup clones through the atomic-loaded pointer,
		// bumping the refcount once more.
		let got =
			tree.lookup_optimistic(&1, |v| v.clone()).expect("inserted value should be present");
		assert_eq!(std::sync::Arc::strong_count(&blob.0), 3);
		assert_eq!(&got.0[..], b"hello world");
		drop(got);
		assert_eq!(std::sync::Arc::strong_count(&blob.0), 2);

		// Overwriting: insert_defer returns true (was present). The
		// displaced slot value is deferred for epoch-safe drop and the
		// load-clone returned through `iter::insert` is also deferred
		// (via `optimistic::drop_or_defer` inside `insert_defer`), so
		// two refcounts on the old V are parked in the deferral queue.
		let blob2 = RefcountedBlob(std::sync::Arc::new(b"replaced".to_vec()));
		assert!(tree.insert_defer(1, blob2.clone()));
		// blob.0 strong count: local(1) + displaced Box's inner Arc(1) +
		// load-clone in defer queue(1) = 3.
		assert_eq!(std::sync::Arc::strong_count(&blob.0), 3);
		// blob2.0: local(1) + new slot's Box's inner Arc(1) = 2.
		assert_eq!(std::sync::Arc::strong_count(&blob2.0), 2);

		// remove_defer reports presence and defers the displaced drop.
		assert!(tree.remove_defer(&1));
		assert!(!tree.remove_defer(&1));
		// blob2's leaf refcount (the slot's Box's inner Arc) is now
		// in epoch deferral, plus the load-clone from remove_defer's
		// `optimistic::drop_or_defer` is also deferred.
		assert_eq!(std::sync::Arc::strong_count(&blob2.0), 3);
	}

	// Concurrent stress tests for the optimistic-read fast path. These
	// used to live in `tests/concurrency.rs` (kept out of `cargo miri
	// test --lib`) because the optimistic reader's non-atomic
	// `ptr::read(V)` would race with the writer's non-atomic
	// `mem::replace(V)` per Miri's data-race detector. The atomic
	// mirror migration (see [`crate::atomic_slot`]) replaces those
	// non-atomic reads/writes with atomic loads/stores at matching
	// `Acquire` / `Release` orderings, so the tests now run cleanly
	// under Miri's `--lib` job.

	#[test]
	fn epoch_deferred_drop_optimistic_reader_vs_defer_writer() {
		use std::sync::atomic::AtomicBool;
		use std::sync::Arc;
		use std::thread;

		// Drastically scaled down for Miri's slower interpreter; the
		// non-Miri build still gets a meaningful concurrent workload.
		let keys: i32 = if cfg!(miri) {
			8
		} else {
			200
		};
		let rounds: i32 = if cfg!(miri) {
			4
		} else {
			50
		};
		let reader_threads = if cfg!(miri) {
			2
		} else {
			4
		};

		let tree: Arc<Tree<i32, RefcountedBlob>> = Arc::new(Tree::new());
		let stop = Arc::new(AtomicBool::new(false));

		for i in 0..keys {
			tree.insert_defer(i, RefcountedBlob(Arc::new(vec![i as u8; 32])));
		}

		let mut handles = Vec::new();
		for _ in 0..reader_threads {
			let tree = Arc::clone(&tree);
			let stop = Arc::clone(&stop);
			handles.push(thread::spawn(move || {
				while !stop.load(Ordering::Relaxed) {
					for k in 0..keys {
						if let Some(blob) = tree.lookup_optimistic(&k, |v| v.clone()) {
							// Touch the buffer so the optimiser cannot
							// dead-code the clone away.
							let s: usize = blob.0.iter().map(|&b| b as usize).sum();
							std::hint::black_box(s);
						}
					}
				}
			}));
		}

		let writer = {
			let tree = Arc::clone(&tree);
			let stop = Arc::clone(&stop);
			thread::spawn(move || {
				for round in 0..rounds {
					for k in 0..keys {
						let v = RefcountedBlob(Arc::new(vec![(k + round) as u8; 32]));
						tree.insert_defer(k, v);
					}
				}
				stop.store(true, Ordering::Relaxed);
			})
		};

		writer.join().unwrap();
		for h in handles {
			h.join().unwrap();
		}
	}

	/// Stress for the K-deferred-drop extension: K is a refcounted type
	/// with `EPOCH_DEFERRED_DROP = true`, and writers alternate
	/// `insert_defer` / `remove_defer` so leaf K is actually dropped
	/// (`remove_defer` defers both K and V drops via the epoch GC).
	#[test]
	fn k_deferred_drop_optimistic_reader_vs_defer_writer() {
		use std::sync::atomic::AtomicBool;
		use std::sync::Arc;
		use std::thread;

		// Wraps an Arc<Vec<u8>> so cloning is cheap and the K's `Drop`
		// frees a shared heap buffer.
		#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
		struct RcKey(Arc<Vec<u8>>);

		// SAFETY: refcounted K with epoch-deferred drop semantics —
		// concurrent optimistic readers' interior pointers stay valid
		// until the next epoch tick.
		unsafe impl OptimisticRead for RcKey {
			const EPOCH_DEFERRED_DROP: bool = true;
			type Slot = crate::atomic_slot::BoxedSlot<Self>;
		}

		let keys: u8 = if cfg!(miri) {
			8
		} else {
			64
		};
		let rounds: u64 = if cfg!(miri) {
			4
		} else {
			50
		};
		let reader_threads = if cfg!(miri) {
			2
		} else {
			4
		};

		let mk_key = |i: u8| RcKey(Arc::new(vec![i; 8]));

		let tree: Arc<Tree<RcKey, u64>> = Arc::new(Tree::new());
		let stop = Arc::new(AtomicBool::new(false));

		for i in 0..keys {
			tree.insert_defer(mk_key(i), i as u64);
		}

		let mut handles = Vec::new();
		for _ in 0..reader_threads {
			let tree = Arc::clone(&tree);
			let stop = Arc::clone(&stop);
			handles.push(thread::spawn(move || {
				while !stop.load(Ordering::Relaxed) {
					for i in 0..keys {
						let k = mk_key(i);
						let _ = tree.lookup_optimistic(&k, |v| *v);
					}
				}
			}));
		}

		let writer = {
			let tree = Arc::clone(&tree);
			let stop = Arc::clone(&stop);
			thread::spawn(move || {
				for round in 0..rounds {
					for i in 0..keys {
						let k = mk_key(i);
						if round % 2 == 0 {
							tree.remove_defer(&k);
						} else {
							tree.insert_defer(k, round * 100 + i as u64);
						}
					}
				}
				stop.store(true, Ordering::Relaxed);
			})
		};

		writer.join().unwrap();
		for h in handles {
			h.join().unwrap();
		}
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
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 0, 10, 100) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 20, 200) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 2, 30, 300) };

		let (pos, exact) = leaf.lower_bound(&20);
		assert_eq!(pos, 1);
		assert!(exact);
	}

	#[test]
	fn leaf_lower_bound_between_keys() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 0, 10, 100) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 20, 200) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 2, 30, 300) };

		let (pos, exact) = leaf.lower_bound(&25);
		assert_eq!(pos, 2); // Would insert at position 2
		assert!(!exact);
	}

	#[test]
	fn leaf_lower_bound_before_all() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 0, 10, 100) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 20, 200) };

		let (pos, exact) = leaf.lower_bound(&5);
		assert_eq!(pos, 0);
		assert!(!exact);
	}

	#[test]
	fn leaf_lower_bound_after_all() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 0, 10, 100) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 20, 200) };

		let (pos, exact) = leaf.lower_bound(&25);
		assert_eq!(pos, 2);
		assert!(!exact);
	}

	#[test]
	fn leaf_lower_bound_respects_lower_fence() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.lower_fence = Some(50);
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 0, 60, 600) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 70, 700) };

		// Key below lower fence
		let (pos, exact) = leaf.lower_bound(&40);
		assert_eq!(pos, 0);
		assert!(!exact);
	}

	#[test]
	fn leaf_lower_bound_respects_upper_fence() {
		let mut leaf: LeafNode<i32, i32, 64> = LeafNode::new();
		leaf.upper_fence = Some(50);
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 0, 30, 300) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 40, 400) };

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

		// SAFETY: see the function-level safety contract.
		assert!(unsafe { LeafNode::insert_at_raw(&mut leaf, 0, 10, 100) }.is_some());
		// SAFETY: see the function-level safety contract.
		assert!(unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 30, 300) }.is_some());
		// SAFETY: see the function-level safety contract.
		assert!(unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 20, 200) }.is_some()); // Insert in middle

		assert_eq!(leaf.len.load(), 3);
		assert_eq!(leaf.key_at(0).unwrap(), 10);
		assert_eq!(leaf.key_at(1).unwrap(), 20);
		assert_eq!(leaf.key_at(2).unwrap(), 30);

		let eg = epoch::pin();
		// SAFETY: see the function-level safety contract.
		let (k, v) = unsafe { LeafNode::remove_at_raw(&mut leaf, 1, &eg) };
		assert_eq!(k, 20);
		assert_eq!(v, 200);
		assert_eq!(leaf.len.load(), 2);
	}

	#[test]
	fn leaf_split_sets_fences_correctly() {
		let mut left: LeafNode<i32, i32, 64> = LeafNode::new();
		for i in 0..10 {
			// SAFETY: see the function-level safety contract.
			unsafe { LeafNode::insert_at_raw(&mut left, i as u16, i * 10, i * 100) };
		}

		let mut right: LeafNode<i32, i32, 64> = LeafNode::new();
		left.split(&mut right, 5);

		// Left should have keys 0-50, right should have keys 60-90
		assert_eq!(left.len.load(), 6); // 0, 10, 20, 30, 40, 50
		assert_eq!(right.len.load(), 4); // 60, 70, 80, 90

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
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut left, 0, 10, 100) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut left, 1, 20, 200) };
		left.upper_fence = Some(25);

		let mut right: LeafNode<i32, i32, 64> = LeafNode::new();
		right.lower_fence = Some(25);
		right.upper_fence = Some(50);
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut right, 0, 30, 300) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut right, 1, 40, 400) };
		right.sample_key = Some(30);

		let result = left.merge(&mut right);
		assert!(result);

		assert_eq!(left.len.load(), 4);
		assert_eq!(left.key_at(0).unwrap(), 10);
		assert_eq!(left.key_at(1).unwrap(), 20);
		assert_eq!(left.key_at(2).unwrap(), 30);
		assert_eq!(left.key_at(3).unwrap(), 40);

		// Left inherits right's upper fence
		assert_eq!(left.upper_fence, Some(50));
		// Left inherits right's sample key
		assert_eq!(left.sample_key, Some(30));

		// Right should be empty
		assert_eq!(right.len.load(), 0);
	}

	#[test]
	fn leaf_merge_fails_when_too_full() {
		let mut left: LeafNode<i32, i32, 4> = LeafNode::new();
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut left, 0, 10, 100) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut left, 1, 20, 200) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut left, 2, 30, 300) };

		let mut right: LeafNode<i32, i32, 4> = LeafNode::new();
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut right, 0, 40, 400) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut right, 1, 50, 500) };

		// Combined size (5) > capacity (4)
		let result = left.merge(&mut right);
		assert!(!result);
		// Both should be unchanged
		assert_eq!(left.len.load(), 3);
		assert_eq!(right.len.load(), 2);
	}

	#[test]
	fn leaf_has_space() {
		let mut leaf: LeafNode<i32, i32, 3> = LeafNode::new();
		assert!(leaf.has_space());

		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 0, 1, 1) };
		assert!(leaf.has_space());

		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 2, 2) };
		assert!(leaf.has_space());

		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 2, 3, 3) };
		assert!(!leaf.has_space());
	}

	#[test]
	fn leaf_is_underfull() {
		// With capacity 10, underfull threshold is 4 (40%)
		let mut leaf: LeafNode<i32, i32, 10> = LeafNode::new();

		// Empty is underfull
		assert!(leaf.is_underfull());

		for i in 0..3 {
			// SAFETY: see the function-level safety contract.
			unsafe { LeafNode::insert_at_raw(&mut leaf, i as u16, i, i) };
		}
		// 3 entries with capacity 10 = 30%, still underfull
		assert!(leaf.is_underfull());

		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 3, 3, 3) };
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
		internal.len.store(3);

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
		let internal: InternalNode<i32, i32, 3, 64> = InternalNode::new();
		assert!(internal.has_space());

		internal.len.store(2);
		assert!(internal.has_space());

		internal.len.store(3);
		assert!(!internal.has_space());
	}

	#[test]
	fn internal_is_underfull() {
		// With capacity 10, underfull threshold is 4 (40%)
		let internal: InternalNode<i32, i32, 10, 64> = InternalNode::new();

		internal.len.store(3);
		assert!(internal.is_underfull());

		internal.len.store(4);
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
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 0, 10, 100) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 1, 20, 200) };
		// SAFETY: see the function-level safety contract.
		unsafe { LeafNode::insert_at_raw(&mut leaf, 2, 30, 300) };

		let node: Node<i32, i32, 64, 64> = Node::Leaf(leaf);
		let keys = node.keys();

		assert_eq!(keys.as_slice(), &[10, 20, 30]);
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
	fn first_returns_minimum() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(3, "three");
		tree.insert(1, "one");
		tree.insert(2, "two");

		let first = tree.first(|k, v| (*k, *v));
		assert_eq!(first, Some((1, "one")));
	}

	#[test]
	fn first_empty_tree() {
		let tree: Tree<i32, &str> = Tree::new();
		let first = tree.first(|k, v| (*k, *v));
		assert_eq!(first, None);
	}

	#[test]
	fn first_single_entry() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(42, "answer");

		let first = tree.first(|k, v| (*k, *v));
		assert_eq!(first, Some((42, "answer")));
	}

	#[test]
	fn last_returns_maximum() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(1, "one");
		tree.insert(3, "three");
		tree.insert(2, "two");

		let last = tree.last(|k, v| (*k, *v));
		assert_eq!(last, Some((3, "three")));
	}

	#[test]
	fn last_empty_tree() {
		let tree: Tree<i32, &str> = Tree::new();
		let last = tree.last(|k, v| (*k, *v));
		assert_eq!(last, None);
	}

	#[test]
	fn last_single_entry() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(42, "answer");

		let last = tree.last(|k, v| (*k, *v));
		assert_eq!(last, Some((42, "answer")));
	}

	#[test]
	fn pop_first_returns_minimum() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(3, "three");
		tree.insert(1, "one");
		tree.insert(2, "two");

		assert_eq!(tree.pop_first(), Some((1, "one")));
		assert_eq!(tree.len(), 2);
		assert_eq!(tree.pop_first(), Some((2, "two")));
		assert_eq!(tree.len(), 1);
		assert_eq!(tree.pop_first(), Some((3, "three")));
		assert!(tree.is_empty());
	}

	#[test]
	fn pop_first_empty_tree() {
		let tree: Tree<i32, &str> = Tree::new();
		assert_eq!(tree.pop_first(), None);
	}

	#[test]
	fn pop_last_returns_maximum() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(1, "one");
		tree.insert(3, "three");
		tree.insert(2, "two");

		assert_eq!(tree.pop_last(), Some((3, "three")));
		assert_eq!(tree.len(), 2);
		assert_eq!(tree.pop_last(), Some((2, "two")));
		assert_eq!(tree.len(), 1);
		assert_eq!(tree.pop_last(), Some((1, "one")));
		assert!(tree.is_empty());
	}

	#[test]
	fn pop_last_empty_tree() {
		let tree: Tree<i32, &str> = Tree::new();
		assert_eq!(tree.pop_last(), None);
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
	fn first_multilevel_tree() {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to cause splits and create multiple levels
		for i in (0..200).rev() {
			tree.insert(i, i * 10);
		}

		tree.assert_invariants();
		assert!(tree.height() > 1);

		let first = tree.first(|k, v| (*k, *v));
		assert_eq!(first, Some((0, 0)));
	}

	#[test]
	fn last_multilevel_tree() {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough to cause splits and create multiple levels
		for i in 0..200 {
			tree.insert(i, i * 10);
		}

		tree.assert_invariants();
		assert!(tree.height() > 1);

		let last = tree.last(|k, v| (*k, *v));
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

	// -----------------------------------------------------------------------
	// Range Iterator Tests
	// -----------------------------------------------------------------------

	#[test]
	fn range_full() {
		use std::ops::Bound::Unbounded;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range(Unbounded, Unbounded);
		for i in 0..10 {
			let (k, v) = range.next().unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i * 10);
		}
		assert!(range.next().is_none());
	}

	#[test]
	fn range_included_bounds() {
		use std::ops::Bound::Included;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range(Included(&3), Included(&6));
		assert_eq!(range.next(), Some((&3, &30)));
		assert_eq!(range.next(), Some((&4, &40)));
		assert_eq!(range.next(), Some((&5, &50)));
		assert_eq!(range.next(), Some((&6, &60)));
		assert_eq!(range.next(), None);
	}

	#[test]
	fn range_excluded_bounds() {
		use std::ops::Bound::Excluded;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range(Excluded(&3), Excluded(&6));
		assert_eq!(range.next(), Some((&4, &40)));
		assert_eq!(range.next(), Some((&5, &50)));
		assert_eq!(range.next(), None);
	}

	#[test]
	fn range_mixed_bounds() {
		use std::ops::Bound::{Excluded, Included, Unbounded};

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		// From start to 5 (exclusive)
		let mut range = tree.range(Unbounded, Excluded(&5));
		for i in 0..5 {
			let (k, v) = range.next().unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i * 10);
		}
		assert!(range.next().is_none());

		// From 5 (included) to end
		let mut range = tree.range(Included(&5), Unbounded);
		for i in 5..10 {
			let (k, v) = range.next().unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i * 10);
		}
		assert!(range.next().is_none());
	}

	#[test]
	fn range_empty_tree() {
		use std::ops::Bound::Unbounded;

		let tree: Tree<i32, i32> = Tree::new();
		let mut range = tree.range(Unbounded, Unbounded);
		assert!(range.next().is_none());
	}

	#[test]
	fn range_nonexistent_bounds() {
		use std::ops::Bound::Included;

		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(0, 0);
		tree.insert(2, 20);
		tree.insert(4, 40);
		tree.insert(6, 60);

		// Range from 1 to 5 (neither exist)
		let mut range = tree.range(Included(&1), Included(&5));
		assert_eq!(range.next(), Some((&2, &20)));
		assert_eq!(range.next(), Some((&4, &40)));
		assert_eq!(range.next(), None);
	}

	// -----------------------------------------------------------------------
	// Reverse Range Iterator Tests
	// -----------------------------------------------------------------------

	#[test]
	fn range_rev_full() {
		use std::ops::Bound::Unbounded;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range_rev(Unbounded, Unbounded);
		for i in (0..10).rev() {
			let (k, v) = range.next().unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i * 10);
		}
		assert!(range.next().is_none());
	}

	#[test]
	fn range_rev_included_bounds() {
		use std::ops::Bound::Included;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range_rev(Included(&3), Included(&6));
		assert_eq!(range.next(), Some((&6, &60)));
		assert_eq!(range.next(), Some((&5, &50)));
		assert_eq!(range.next(), Some((&4, &40)));
		assert_eq!(range.next(), Some((&3, &30)));
		assert_eq!(range.next(), None);
	}

	#[test]
	fn range_rev_excluded_bounds() {
		use std::ops::Bound::Excluded;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range_rev(Excluded(&3), Excluded(&6));
		assert_eq!(range.next(), Some((&5, &50)));
		assert_eq!(range.next(), Some((&4, &40)));
		assert_eq!(range.next(), None);
	}

	#[test]
	fn range_rev_mixed_bounds() {
		use std::ops::Bound::{Excluded, Included, Unbounded};

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		// From 5 (exclusive) to end (descending)
		let mut range = tree.range_rev(Excluded(&5), Unbounded);
		for i in (6..10).rev() {
			let (k, v) = range.next().unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i * 10);
		}
		assert!(range.next().is_none());

		// From start to 5 (included) (descending)
		let mut range = tree.range_rev(Unbounded, Included(&5));
		for i in (0..=5).rev() {
			let (k, v) = range.next().unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i * 10);
		}
		assert!(range.next().is_none());
	}

	#[test]
	fn range_rev_empty_tree() {
		use std::ops::Bound::Unbounded;

		let tree: Tree<i32, i32> = Tree::new();
		let mut range = tree.range_rev(Unbounded, Unbounded);
		assert!(range.next().is_none());
	}

	#[test]
	fn range_rev_nonexistent_bounds() {
		use std::ops::Bound::Included;

		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(0, 0);
		tree.insert(2, 20);
		tree.insert(4, 40);
		tree.insert(6, 60);

		// Reverse range from 1 to 5 (neither exist)
		let mut range = tree.range_rev(Included(&1), Included(&5));
		assert_eq!(range.next(), Some((&4, &40)));
		assert_eq!(range.next(), Some((&2, &20)));
		assert_eq!(range.next(), None);
	}

	#[test]
	fn range_rev_single_element() {
		use std::ops::Bound::Included;

		let tree: Tree<i32, i32> = Tree::new();
		tree.insert(5, 50);

		let mut range = tree.range_rev(Included(&5), Included(&5));
		assert_eq!(range.next(), Some((&5, &50)));
		assert_eq!(range.next(), None);
	}

	#[test]
	fn range_rev_peek() {
		use std::ops::Bound::{Included, Unbounded};

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range_rev(Included(&3), Unbounded);

		// peek should return 9 (the largest key)
		assert_eq!(range.peek(), Some((&9, &90)));
		assert_eq!(range.peek(), Some((&9, &90))); // peek again returns same

		// next should return 9 and advance
		assert_eq!(range.next(), Some((&9, &90)));

		// now peek should return 8
		assert_eq!(range.peek(), Some((&8, &80)));
	}

	#[test]
	fn range_rev_peek_respects_lower_bound() {
		use std::ops::Bound::Included;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..10 {
			tree.insert(i, i * 10);
		}

		let mut range = tree.range_rev(Included(&7), Included(&9));

		assert_eq!(range.peek(), Some((&9, &90)));
		assert_eq!(range.next(), Some((&9, &90)));
		assert_eq!(range.peek(), Some((&8, &80)));
		assert_eq!(range.next(), Some((&8, &80)));
		assert_eq!(range.peek(), Some((&7, &70)));
		assert_eq!(range.next(), Some((&7, &70)));
		// Should stop at lower bound
		assert!(range.peek().is_none());
		assert!(range.next().is_none());
	}

	#[test]
	fn range_rev_large_tree() {
		use std::ops::Bound::Unbounded;

		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..200 {
			tree.insert(i, i);
		}
		assert!(tree.height() > 1, "Tree should have multiple levels");

		let mut range = tree.range_rev(Unbounded, Unbounded);
		for i in (0..200).rev() {
			let (k, v) = range.next().unwrap();
			assert_eq!(*k, i);
			assert_eq!(*v, i);
		}
		assert!(range.next().is_none());
	}

	// -----------------------------------------------------------------------
	// Keys Iterator Tests
	// -----------------------------------------------------------------------

	#[test]
	fn keys_basic() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(3, "three");
		tree.insert(1, "one");
		tree.insert(2, "two");

		let mut keys = tree.keys();
		assert_eq!(keys.next(), Some(&1));
		assert_eq!(keys.next(), Some(&2));
		assert_eq!(keys.next(), Some(&3));
		assert_eq!(keys.next(), None);
	}

	#[test]
	fn keys_empty_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut keys = tree.keys();
		assert!(keys.next().is_none());
	}

	#[test]
	fn keys_large_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..200 {
			tree.insert(i, i);
		}

		let mut keys = tree.keys();
		for i in 0..200 {
			assert_eq!(keys.next(), Some(&i));
		}
		assert!(keys.next().is_none());
	}

	// -----------------------------------------------------------------------
	// Values Iterator Tests
	// -----------------------------------------------------------------------

	#[test]
	fn values_basic() {
		let tree: Tree<i32, &str> = Tree::new();
		tree.insert(3, "three");
		tree.insert(1, "one");
		tree.insert(2, "two");

		let mut values = tree.values();
		assert_eq!(values.next(), Some(&"one"));
		assert_eq!(values.next(), Some(&"two"));
		assert_eq!(values.next(), Some(&"three"));
		assert_eq!(values.next(), None);
	}

	#[test]
	fn values_empty_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		let mut values = tree.values();
		assert!(values.next().is_none());
	}

	#[test]
	fn values_large_tree() {
		let tree: Tree<i32, i32> = Tree::new();
		for i in 0..200 {
			tree.insert(i, i * 10);
		}

		let mut values = tree.values();
		for i in 0..200 {
			assert_eq!(values.next(), Some(&(i * 10)));
		}
		assert!(values.next().is_none());
	}

	// -----------------------------------------------------------------------
	// get_or_insert Tests
	// -----------------------------------------------------------------------

	#[test]
	fn get_or_insert_new_key() {
		let tree: Tree<i32, String> = Tree::new();

		// Key doesn't exist - should insert and return the default
		let value = tree.get_or_insert(1, "default".to_string());
		assert_eq!(value, "default");

		// Verify it was inserted
		assert_eq!(tree.lookup(&1, |v| v.clone()), Some("default".to_string()));
		assert_eq!(tree.len(), 1);

		tree.assert_invariants();
	}

	#[test]
	fn get_or_insert_existing_key() {
		let tree: Tree<i32, String> = Tree::new();

		// Pre-insert a value
		tree.insert(1, "existing".to_string());

		// Key exists - should return existing value without inserting
		let value = tree.get_or_insert(1, "new_default".to_string());
		assert_eq!(value, "existing");

		// Verify the value wasn't changed
		assert_eq!(tree.lookup(&1, |v| v.clone()), Some("existing".to_string()));
		assert_eq!(tree.len(), 1);

		tree.assert_invariants();
	}

	#[test]
	fn get_or_insert_with_lazy_evaluation() {
		let tree: Tree<i32, String> = Tree::new();

		// Pre-insert a value
		tree.insert(1, "existing".to_string());

		// Closure should NOT be called when key exists
		let value = tree.get_or_insert_with(1, || {
			panic!("closure should not be called for existing key");
		});
		assert_eq!(value, "existing");

		// Closure SHOULD be called when key doesn't exist
		let value = tree.get_or_insert_with(2, || "computed".to_string());
		assert_eq!(value, "computed");

		assert_eq!(tree.len(), 2);
		tree.assert_invariants();
	}

	#[test]
	fn get_or_insert_triggers_split() {
		let tree: Tree<i32, i32> = Tree::new();

		// Insert enough entries to fill a leaf (capacity 64)
		for i in 0..64 {
			tree.get_or_insert(i, i * 10);
		}

		tree.assert_invariants();
		assert_eq!(tree.len(), 64);

		// This insertion should trigger a split
		let value = tree.get_or_insert(64, 640);
		assert_eq!(value, 640);

		tree.assert_invariants();
		assert_eq!(tree.len(), 65);
		assert!(tree.height() >= 2, "Tree should have split");

		// Verify all values are correct
		for i in 0..=64 {
			assert_eq!(tree.lookup(&i, |v| *v), Some(i * 10));
		}
	}

	#[test]
	fn get_or_insert_multiple_operations() {
		let tree: Tree<i32, i32> = Tree::new();

		// Mix of new insertions and lookups of existing keys
		for i in 0..100 {
			let value = tree.get_or_insert(i % 50, i);
			if i < 50 {
				// First 50 operations insert new keys
				assert_eq!(value, i);
			} else {
				// Next 50 operations find existing keys
				assert_eq!(value, i - 50);
			}
		}

		tree.assert_invariants();
		assert_eq!(tree.len(), 50);
	}
}
