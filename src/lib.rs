//! Implementation of a fast in-memory concurrent B+ Tree featuring optimistic lock coupling.
//!
//! The implementation is based on [LeanStore](https://dbis1.github.io/leanstore.html) with some
//! adaptations from [Umbra](https://umbra-db.com/#publications).
//!
//! ```
//! use ferntree::Tree;
//!
//! let tree = Tree::new();
//!
//! tree.insert("some", "data");
//! ```

use crossbeam_epoch::{self as epoch, Atomic, Owned};
use smallvec::{smallvec, SmallVec};

use std::borrow::Borrow;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

pub mod error;
pub mod iter;
pub mod latch;
#[cfg(test)]
pub mod util;

use latch::{ExclusiveGuard, HybridGuard, HybridLatch, OptimisticGuard, SharedGuard};

const INNER_CAPACITY: usize = 64;
const LEAF_CAPACITY: usize = 64;

/// Type alias for the `GenericTree` with preset node sizes
pub type Tree<K, V> = GenericTree<K, V, INNER_CAPACITY, LEAF_CAPACITY>;

/// Concurrent, optimistically locked B+ Tree
///
/// `InnerNode` and `LeafNode` capacities can be configured through the const generic parameters `IC`
/// and `LC` respectively.
pub struct GenericTree<K, V, const IC: usize, const LC: usize> {
	root: HybridLatch<Atomic<HybridLatch<Node<K, V, IC, LC>>>>,
	height: AtomicUsize,
}

impl<K: Clone + Ord, V, const IC: usize, const LC: usize> Default for GenericTree<K, V, IC, LC> {
	fn default() -> Self {
		Self::new()
	}
}

pub(crate) enum ParentHandler<'r, 'p, K, V, const IC: usize, const LC: usize> {
	Root {
		tree_guard: OptimisticGuard<'r, Atomic<HybridLatch<Node<K, V, IC, LC>>>>,
	},
	Parent {
		parent_guard: OptimisticGuard<'p, Node<K, V, IC, LC>>,
		pos: u16,
	},
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub(crate) enum Direction {
	Forward,
	Reverse,
}

impl<K: Clone + Ord, V, const IC: usize, const LC: usize> GenericTree<K, V, IC, LC> {
	/// Makes a new, empty `GenericTree`
	///
	/// Allocates the root node on creation
	pub fn new() -> Self {
		GenericTree {
			root: HybridLatch::new(Atomic::new(HybridLatch::new(Node::Leaf(LeafNode {
				len: 0,
				keys: smallvec![],
				values: smallvec![],
				lower_fence: None,
				upper_fence: None,
				sample_key: None,
			})))),
			height: AtomicUsize::new(1),
		}
	}

	/// Returns the height of the tree
	pub fn height(&self) -> usize {
		self.height.load(Ordering::Relaxed)
	}

	pub(crate) fn find_parent<'t, 'g>(
		&'t self,
		needle: &impl HybridGuard<Node<K, V, IC, LC>>,
		eg: &'t epoch::Guard,
	) -> error::Result<ParentHandler<'t, 't, K, V, IC, LC>>
	where
		K: Ord,
	{
		let tree_guard = self.root.optimistic_or_spin();
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_latch_ptr = root_latch as *const _;
		let root_guard = root_latch.optimistic_or_spin();

		if needle.latch() as *const _ == root_latch_ptr {
			tree_guard.recheck()?;
			return Ok(ParentHandler::Root {
				tree_guard,
			});
		}

		let search_key = match needle.inner().sample_key().cloned() {
			Some(key) => key,
			None => {
				needle.recheck()?;
				return Err(error::Error::Reclaimed);
			}
		};

		let mut t_guard = Some(tree_guard);
		let mut p_guard: Option<OptimisticGuard<'_, Node<K, V, IC, LC>>> = None;
		let mut target_guard = root_guard;
		let mut pos = 0u16;

		let parent_guard = loop {
			let (c_swip, c_pos) = match *target_guard {
				Node::Internal(ref internal) => {
					let (c_pos, _) = internal.lower_bound(&search_key);
					let swip = internal.edge_at(c_pos)?;
					(swip, c_pos)
				}
				Node::Leaf(ref _leaf) => {
					break p_guard.expect("must have parent");
				}
			};

			let c_latch = unsafe { c_swip.load(Ordering::Acquire, eg).deref() };
			let c_latch_ptr = c_latch as *const _;

			if needle.latch() as *const _ == c_latch_ptr {
				target_guard.recheck()?;
				if let Some(tree_guard) = t_guard.take() {
					tree_guard.recheck()?;
				}
				pos = c_pos; // Update pos to the actual position of the needle
				break target_guard;
			}

			let guard = Self::lock_coupling(&target_guard, c_swip, eg)?;
			p_guard = Some(target_guard);
			pos = c_pos;
			target_guard = guard;

			if let Some(tree_guard) = t_guard.take() {
				tree_guard.recheck()?;
			}
		};

		Ok(ParentHandler::Parent {
			parent_guard,
			pos,
		})
	}

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
		let tree_guard = self.root.optimistic_or_spin();
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_latch_ptr = root_latch as *const _;
		let root_guard = root_latch.optimistic_or_spin();

		if needle.latch() as *const _ == root_latch_ptr {
			root_guard.recheck()?;
			tree_guard.recheck()?;
			return error::Result::Ok(None);
		}

		let (parent_guard, pos) = match self.find_parent(needle, eg)? {
			ParentHandler::Root {
				tree_guard: _,
			} => {
				return error::Result::Ok(None);
			}
			ParentHandler::Parent {
				parent_guard,
				pos,
			} => (parent_guard, pos),
		};

		let within_bounds = match direction {
			Direction::Forward => pos + 1 <= parent_guard.as_internal().len,
			Direction::Reverse => pos > 0,
		};

		if within_bounds {
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
				let (leaf, parent_opt) =
					self.find_leaf_and_parent_from_node(guard, direction, eg)?;
				return error::Result::Ok(Some((leaf, parent_opt.expect("must have parent here"))));
			}
		} else {
			let mut target_guard = parent_guard;

			loop {
				let (parent_guard, pos) = match self.find_parent(&target_guard, eg)? {
					ParentHandler::Root {
						tree_guard: _,
					} => {
						return error::Result::Ok(None);
					}
					ParentHandler::Parent {
						parent_guard,
						pos,
					} => (parent_guard, pos),
				};

				let within_bounds = match direction {
					Direction::Forward => pos + 1 <= parent_guard.as_internal().len,
					Direction::Reverse => pos > 0,
				};

				if within_bounds {
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
						let (leaf, parent_opt) =
							self.find_leaf_and_parent_from_node(guard, direction, eg)?;
						return error::Result::Ok(Some((
							leaf,
							parent_opt.expect("must have parent here"),
						)));
					}
				} else {
					target_guard = parent_guard;
					continue;
				}
			}
		}
	}

	pub(crate) fn lock_coupling<'e>(
		p_guard: &OptimisticGuard<'e, Node<K, V, IC, LC>>,
		swip: &Atomic<HybridLatch<Node<K, V, IC, LC>>>,
		eg: &'e epoch::Guard,
	) -> error::Result<OptimisticGuard<'e, Node<K, V, IC, LC>>> {
		let c_latch = unsafe { swip.load(Ordering::Acquire, eg).deref() };
		let c_guard = c_latch.optimistic_or_spin();
		p_guard.recheck()?;
		Ok(c_guard)
	}

	fn lock_coupling_shared<'e>(
		p_guard: &OptimisticGuard<'e, Node<K, V, IC, LC>>,
		swip: &Atomic<HybridLatch<Node<K, V, IC, LC>>>,
		eg: &'e epoch::Guard,
	) -> error::Result<SharedGuard<'e, Node<K, V, IC, LC>>> {
		let c_latch = unsafe { swip.load(Ordering::Acquire, eg).deref() };
		let c_guard = c_latch.shared();
		p_guard.recheck()?;
		Ok(c_guard)
	}

	fn lock_coupling_exclusive<'e>(
		p_guard: &OptimisticGuard<'e, Node<K, V, IC, LC>>,
		swip: &Atomic<HybridLatch<Node<K, V, IC, LC>>>,
		eg: &'e epoch::Guard,
	) -> error::Result<ExclusiveGuard<'e, Node<K, V, IC, LC>>> {
		let c_latch = unsafe { swip.load(Ordering::Acquire, eg).deref() };
		let c_guard = c_latch.exclusive();
		p_guard.recheck()?;
		Ok(c_guard)
	}

	fn find_leaf_and_parent_from_node<'t, 'e>(
		&'t self,
		needle: OptimisticGuard<'e, Node<K, V, IC, LC>>,
		direction: Direction,
		eg: &'e epoch::Guard,
	) -> error::Result<(
		OptimisticGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)> {
		let mut p_guard = None;
		let mut target_guard = needle;

		let leaf_guard = loop {
			let (c_swip, pos) = match *target_guard {
				Node::Internal(ref internal) => {
					let pos = match direction {
						Direction::Forward => 0,
						Direction::Reverse => internal.len,
					};
					let swip = internal.edge_at(pos)?;
					(swip, pos)
				}
				Node::Leaf(ref _leaf) => {
					break target_guard;
				}
			};

			let guard = GenericTree::lock_coupling(&target_guard, c_swip, eg)?;
			p_guard = Some((target_guard, pos));
			target_guard = guard;
		};

		leaf_guard.recheck()?;

		Ok((leaf_guard, p_guard))
	}

	fn find_first_leaf_and_parent<'t, 'e>(
		&'t self,
		eg: &'e epoch::Guard,
	) -> error::Result<(
		OptimisticGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)> {
		let tree_guard = self.root.optimistic_or_spin();
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_guard = root_latch.optimistic_or_spin();
		tree_guard.recheck()?;

		self.find_leaf_and_parent_from_node(root_guard, Direction::Forward, eg)
	}

	fn find_last_leaf_and_parent<'t, 'e>(
		&'t self,
		eg: &'e epoch::Guard,
	) -> error::Result<(
		OptimisticGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)> {
		let tree_guard = self.root.optimistic_or_spin();
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_guard = root_latch.optimistic_or_spin();
		tree_guard.recheck()?;

		self.find_leaf_and_parent_from_node(root_guard, Direction::Reverse, eg)
	}

	fn find_leaf_and_parent<'t, 'k, 'e, Q>(
		&'t self,
		key: &'k Q,
		eg: &'e epoch::Guard,
	) -> error::Result<(
		OptimisticGuard<'e, Node<K, V, IC, LC>>,
		Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>,
	)>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		let tree_guard = self.root.optimistic_or_spin();
		let root_latch = unsafe { tree_guard.load(Ordering::Acquire, eg).deref() };
		let root_guard = root_latch.optimistic_or_spin();
		tree_guard.recheck()?;

		let mut t_guard = Some(tree_guard);
		let mut p_guard = None;
		let mut target_guard = root_guard;

		let leaf_guard = loop {
			let (c_swip, pos) = match *target_guard {
				Node::Internal(ref internal) => {
					let (pos, _) = internal.lower_bound(key);
					let swip = internal.edge_at(pos)?;
					(swip, pos)
				}
				Node::Leaf(ref _leaf) => {
					break target_guard;
				}
			};

			let guard = GenericTree::lock_coupling(&target_guard, c_swip, eg)?;
			p_guard = Some((target_guard, pos));
			target_guard = guard;

			if let Some(tree_guard) = t_guard.take() {
				tree_guard.recheck()?;
			}
		};

		Ok((leaf_guard, p_guard))
	}

	#[allow(dead_code)]
	fn find_leaf<'t, 'k, 'e, Q>(
		&'t self,
		key: &'k Q,
		eg: &'e epoch::Guard,
	) -> error::Result<OptimisticGuard<'e, Node<K, V, IC, LC>>>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		self.find_leaf_and_parent(key, eg).map(|(leaf, _)| leaf)
	}

	pub(crate) fn find_shared_leaf_and_optimistic_parent<'t, 'k, 'e, Q>(
		&'t self,
		key: &'k Q,
		eg: &'e epoch::Guard,
	) -> (SharedGuard<'e, Node<K, V, IC, LC>>, Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		loop {
			let perform = || {
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
							if let Some(tree_guard) = t_guard.take() {
								tree_guard.recheck()?;
							}

							if p_guard.is_none() {
								break target_guard.to_shared()?;
							} else {
								panic!("got a leaf on the wrong level");
							}
						}
					};

					if (level + 1) as usize == self.height.load(Ordering::Acquire) {
						if let Some(tree_guard) = t_guard.take() {
							tree_guard.recheck()?;
						}

						let guard = Self::lock_coupling_shared(&target_guard, c_swip, eg)?;
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

	pub(crate) fn find_first_shared_leaf_and_optimistic_parent<'t, 'e>(
		&'t self,
		eg: &'e epoch::Guard,
	) -> (SharedGuard<'e, Node<K, V, IC, LC>>, Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>)
	{
		loop {
			let perform = || {
				let (leaf, parent_opt) = self.find_first_leaf_and_parent(eg)?;
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

	pub(crate) fn find_last_shared_leaf_and_optimistic_parent<'t, 'e>(
		&'t self,
		eg: &'e epoch::Guard,
	) -> (SharedGuard<'e, Node<K, V, IC, LC>>, Option<(OptimisticGuard<'e, Node<K, V, IC, LC>>, u16)>)
	{
		loop {
			let perform = || {
				let (leaf, parent_opt) = self.find_last_leaf_and_parent(eg)?;
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

	pub(crate) fn find_exclusive_leaf_and_optimistic_parent<'t, 'k, 'e, Q>(
		&'t self,
		key: &'k Q,
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
							if let Some(tree_guard) = t_guard.take() {
								tree_guard.recheck()?;
							}

							if p_guard.is_none() {
								break target_guard.to_exclusive()?;
							} else {
								panic!("got a leaf on the wrong level");
							}
						}
					};

					if (level + 1) as usize == self.height.load(Ordering::Acquire) {
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

	#[allow(dead_code)]
	pub(crate) fn find_exact_exclusive_leaf_and_optimistic_parent<'t, 'k, 'e, Q>(
		&'t self,
		key: &'k Q,
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
				let (leaf, parent_opt) = self.find_leaf_and_parent(key, eg)?;
				let (pos, exact) = leaf.as_leaf().lower_bound(key);
				if exact {
					let exclusive_leaf = leaf.to_exclusive()?;
					error::Result::Ok(Some(((exclusive_leaf, pos), parent_opt)))
				} else {
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

	pub(crate) fn find_first_exclusive_leaf_and_optimistic_parent<'t, 'e>(
		&'t self,
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

	pub(crate) fn find_last_exclusive_leaf_and_optimistic_parent<'t, 'e>(
		&'t self,
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

	/// Zero copy optimistic getter for a value in the tree
	///
	/// Accepts a function that may be executed multiple times until a valid access is performed,
	/// calling the function should not have any side effects because it may be executed with
	/// invalid data.
	pub fn lookup<Q, R, F>(&self, key: &Q, f: F) -> Option<R>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
		F: Fn(&V) -> R,
	{
		let eg = &epoch::pin();
		loop {
			let perform = || {
				let guard = self.find_leaf(key, eg)?;
				if let Node::Leaf(ref leaf) = *guard {
					let (pos, exact) = leaf.lower_bound(key);
					if exact {
						let result = f(leaf.value_at(pos)?);
						guard.recheck()?;
						error::Result::Ok(Some(result))
					} else {
						guard.recheck()?;
						error::Result::Ok(None)
					}
				} else {
					unreachable!("must be a leaf node");
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

	/// Removes a key from the tree, returning the value at the key if the key
	/// was previously in the tree.
	pub fn remove<Q>(&self, key: &Q) -> Option<V>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		self.remove_entry(key).map(|(_, v)| v)
	}

	/// Removes a key from the tree, returning the stored key and value if the key
	/// was previously in the tree.
	pub fn remove_entry<Q>(&self, key: &Q) -> Option<(K, V)>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		let eg = epoch::pin();

		let result = if let Some(((mut guard, pos), _parent_opt)) =
			self.find_exact_exclusive_leaf_and_optimistic_parent(key, &eg)
		{
			let kv = guard.as_leaf_mut().remove_at(pos);

			if guard.is_underfull() {
				let guard = guard.unlock();
				loop {
					let perform_merge = || {
						let _ = self.try_merge(&guard, &eg)?;
						error::Result::Ok(())
					};

					match perform_merge() {
						Ok(_) | Err(error::Error::Reclaimed) => {
							break;
						}
						Err(_) => {
							break;
						}
					}
				}
			}

			Some(kv)
		} else {
			None
		};

		drop(eg);
		result
	}

	/// Inserts a key-value pair into the tree.
	///
	/// If the tree did not have this key present, `None` is returned.
	///
	/// If the tree did have this key present, the value is updated, and the old
	/// value is returned.
	pub fn insert(&self, key: K, value: V) -> Option<V>
	where
		K: Ord,
	{
		let mut iter = self.raw_iter_mut();
		iter.insert(key, value)
	}

	pub(crate) fn try_split<'t, 'g, 'e>(
		&'t self,
		needle: &OptimisticGuard<'g, Node<K, V, IC, LC>>,
		eg: &'e epoch::Guard,
	) -> error::Result<()>
	where
		K: Ord,
	{
		let parent_handler = self.find_parent(needle, eg)?;

		match parent_handler {
			ParentHandler::Root {
				tree_guard,
			} => {
				let mut tree_guard_x = tree_guard.to_exclusive()?;

				let root_latch = unsafe { tree_guard_x.load(Ordering::Acquire, eg).deref() };
				let mut root_guard_x = root_latch.exclusive();

				let mut new_root_owned: Owned<HybridLatch<Node<K, V, IC, LC>>> =
					Owned::new(HybridLatch::new(Node::Internal(InternalNode::new())));

				match root_guard_x.as_mut() {
					Node::Internal(root_internal_node) => {
						if root_internal_node.len <= 2 {
							return Ok(());
						}

						let split_pos = root_internal_node.len / 2;
						let split_key =
							root_internal_node.key_at(split_pos).expect("should exist").clone();

						let mut new_right_node_owned =
							Owned::new(HybridLatch::new(Node::Internal(InternalNode::new())));

						{
							let new_right_node =
								new_right_node_owned.as_mut().as_mut().as_internal_mut();
							root_internal_node.split(new_right_node, split_pos);
						}

						let old_root_edge = Atomic::from(tree_guard_x.load(Ordering::Acquire, eg));
						let new_right_node_edge =
							Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(new_right_node_owned);

						{
							let new_root = new_root_owned.as_mut().as_mut().as_internal_mut();
							new_root.insert(split_key, old_root_edge);
							new_root.upper_edge = Some(new_right_node_edge);
						}
					}
					Node::Leaf(root_leaf_node) => {
						if root_leaf_node.len <= 2 {
							return Ok(());
						}

						let split_pos = root_leaf_node.len / 2;
						let split_key =
							root_leaf_node.key_at(split_pos).expect("should exist").clone();

						let mut new_right_node_owned =
							Owned::new(HybridLatch::new(Node::Leaf(LeafNode::new())));

						{
							let new_right_node =
								new_right_node_owned.as_mut().as_mut().as_leaf_mut();
							root_leaf_node.split(new_right_node, split_pos);
						}

						let old_root_edge = Atomic::from(tree_guard_x.load(Ordering::Acquire, eg));
						let new_right_node_edge =
							Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(new_right_node_owned);

						{
							let new_root = new_root_owned.as_mut().as_mut().as_internal_mut();
							new_root.insert(split_key, old_root_edge);
							new_root.upper_edge = Some(new_right_node_edge);
						}
					}
				}

				let new_root_node_edge =
					Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(new_root_owned);
				*tree_guard_x = new_root_node_edge;
				self.height.fetch_add(1, Ordering::Relaxed);
			}
			ParentHandler::Parent {
				parent_guard,
				pos,
			} => {
				if parent_guard.as_internal().has_space() {
					let swip = parent_guard.as_internal().edge_at(pos)?;
					let target_guard = GenericTree::lock_coupling(&parent_guard, swip, eg)?;

					// Verify that the target we found is actually the needle
					let target_latch = target_guard.latch() as *const _;
					let needle_latch = needle.latch() as *const _;
					if target_latch != needle_latch {
						// The tree structure has changed - return Reclaimed so we re-seek
						return Err(error::Error::Reclaimed);
					}
					let mut parent_guard_x = parent_guard.to_exclusive()?;
					let mut target_guard_x = target_guard.to_exclusive()?;

					match target_guard_x.as_mut() {
						Node::Internal(left_internal) => {
							if left_internal.len <= 2 {
								return Ok(());
							}

							let split_pos = left_internal.len / 2;
							let split_key =
								left_internal.key_at(split_pos).expect("should exist").clone();

							let mut new_right_node_owned =
								Owned::new(HybridLatch::new(Node::Internal(InternalNode::new())));

							{
								let new_right_node =
									new_right_node_owned.as_mut().as_mut().as_internal_mut();
								left_internal.split(new_right_node, split_pos);
							}

							let new_right_node_edge =
								Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(
									new_right_node_owned,
								);

							let parent_internal = parent_guard_x.as_internal_mut();

							// After split: left node has lower keys, right node has higher keys.
							// Left node stays at current position, right node is inserted after.
							if pos == parent_internal.len {
								// Node was at upper_edge - it becomes the left.
								// Insert split key and make right the new upper_edge.
								let left_edge = parent_internal
									.upper_edge
									.replace(new_right_node_edge)
									.expect("upper_edge must be populated");
								parent_internal.insert(split_key, left_edge);
							} else {
								// Node was at edges[pos] - keep it there (it's now the left).
								// Insert split key with right node after it.
								parent_internal.insert_after(pos, split_key, new_right_node_edge);
							}
						}
						Node::Leaf(left_leaf) => {
							if left_leaf.len <= 2 {
								return Ok(());
							}

							let split_pos = left_leaf.len / 2;
							let split_key =
								left_leaf.key_at(split_pos).expect("should exist").clone();

							let mut new_right_node_owned =
								Owned::new(HybridLatch::new(Node::Leaf(LeafNode::new())));

							{
								let new_right_node =
									new_right_node_owned.as_mut().as_mut().as_leaf_mut();
								left_leaf.split(new_right_node, split_pos);
							}

							let new_right_node_edge =
								Atomic::<HybridLatch<Node<K, V, IC, LC>>>::from(
									new_right_node_owned,
								);

							let parent_internal = parent_guard_x.as_internal_mut();

							// After split: left node has lower keys, right node has higher keys.
							// Left node stays at current position, right node is inserted after.
							if pos == parent_internal.len {
								// Node was at upper_edge - it becomes the left.
								// Insert split key and make right the new upper_edge.
								let left_edge = parent_internal
									.upper_edge
									.replace(new_right_node_edge)
									.expect("upper_edge must be populated");
								parent_internal.insert(split_key, left_edge);
							} else {
								// Node was at edges[pos] - keep it there (it's now the left).
								// Insert split key with right node after it.
								parent_internal.insert_after(pos, split_key, new_right_node_edge);
							}
						}
					}
				} else {
					// Parent is full, split it first
					self.try_split(&parent_guard, eg)?;
					// After splitting parent, the tree structure changed.
					// Return Reclaimed so caller goes back to seek_exact and finds the right leaf.
					return Err(error::Error::Reclaimed);
				}
			}
		}

		Ok(())
	}

	pub(crate) fn try_merge<'t, 'g, 'e>(
		&'t self,
		needle: &OptimisticGuard<'g, Node<K, V, IC, LC>>,
		eg: &'e epoch::Guard,
	) -> error::Result<bool>
	where
		K: Ord,
	{
		let parent_handler = self.find_parent(needle, eg)?;

		match parent_handler {
			ParentHandler::Root {
				tree_guard: _,
			} => Ok(false),
			ParentHandler::Parent {
				mut parent_guard,
				pos,
			} => {
				let parent_len = parent_guard.as_internal().len;

				let swip = parent_guard.as_internal().edge_at(pos)?;
				let mut target_guard = GenericTree::lock_coupling(&parent_guard, swip, eg)?;

				if !target_guard.is_underfull() {
					target_guard.recheck()?;
					return Ok(false);
				}

				let merge_succeeded = if parent_len > 1 && pos > 0 {
					let l_swip = parent_guard.as_internal().edge_at(pos - 1)?;
					let left_guard = GenericTree::lock_coupling(&parent_guard, l_swip, eg)?;

					if !left_guard.can_merge_with(&target_guard) {
						left_guard.recheck()?;
						target_guard.recheck()?;
						false
					} else {
						let mut parent_guard_x = parent_guard.to_exclusive()?;
						let mut target_guard_x = target_guard.to_exclusive()?;
						let mut left_guard_x = left_guard.to_exclusive()?;

						match target_guard_x.as_mut() {
							Node::Leaf(ref mut target_leaf) => {
								assert!(left_guard_x.is_leaf());

								if !left_guard_x.as_leaf_mut().merge(target_leaf) {
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
											.expect("must exist");

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
							Node::Internal(target_internal) => {
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
											.expect("must exist");

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
					false
				};

				let merge_succeeded =
					if !merge_succeeded && parent_len > 0 && (pos + 1) <= parent_len {
						let r_swip = parent_guard.as_internal().edge_at(pos + 1)?;
						let right_guard = GenericTree::lock_coupling(&parent_guard, r_swip, eg)?;

						if !right_guard.can_merge_with(&target_guard) {
							right_guard.recheck()?;
							target_guard.recheck()?;
							false
						} else {
							let mut parent_guard_x = parent_guard.to_exclusive()?;
							let mut target_guard_x = target_guard.to_exclusive()?;
							let mut right_guard_x = right_guard.to_exclusive()?;

							match target_guard_x.as_mut() {
								Node::Leaf(ref mut target_leaf) => {
									assert!(right_guard_x.is_leaf());

									if !target_leaf.merge(right_guard_x.as_leaf_mut()) {
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
												.expect("must exist");

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
												.expect("must exist");

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

				let parent_merge = || {
					if parent_guard.is_underfull() {
						parent_guard.recheck()?;
						let _ = self.try_merge(&parent_guard, eg)?;
					}
					error::Result::Ok(())
				};

				let _ = parent_merge();

				Ok(merge_succeeded)
			}
		}
	}

	/// Returns a raw iterator over the entries of the tree.
	pub fn raw_iter(&self) -> iter::RawSharedIter<'_, K, V, IC, LC>
	where
		K: Ord,
	{
		iter::RawSharedIter::new(self)
	}

	/// Returns a raw mutable iterator over the entries of the tree.
	pub fn raw_iter_mut(&self) -> iter::RawExclusiveIter<'_, K, V, IC, LC>
	where
		K: Ord,
	{
		iter::RawExclusiveIter::new(self)
	}

	/// Returns the number of elements in the tree.
	pub fn len(&self) -> usize {
		let mut count = 0usize;
		let mut iter = self.raw_iter();
		iter.seek_to_first();
		while iter.next().is_some() {
			count += 1;
		}
		count
	}

	/// Returns true if the tree is empty.
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}
}

pub(crate) enum Node<K, V, const IC: usize, const LC: usize> {
	Internal(InternalNode<K, V, IC, LC>),
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
	#[inline]
	pub(crate) fn is_leaf(&self) -> bool {
		matches!(self, Node::Leaf(_))
	}

	#[inline]
	pub(crate) fn as_leaf(&self) -> &LeafNode<K, V, LC> {
		match self {
			Node::Leaf(ref leaf) => leaf,
			Node::Internal(_) => {
				panic!("expected leaf node");
			}
		}
	}

	#[inline]
	pub(crate) fn as_leaf_mut(&mut self) -> &mut LeafNode<K, V, LC> {
		match self {
			Node::Leaf(ref mut leaf) => leaf,
			Node::Internal(_) => {
				panic!("expected leaf node");
			}
		}
	}

	#[inline]
	pub(crate) fn as_internal(&self) -> &InternalNode<K, V, IC, LC> {
		match self {
			Node::Internal(ref internal) => internal,
			Node::Leaf(_) => {
				panic!("expected internal node");
			}
		}
	}

	#[inline]
	pub(crate) fn as_internal_mut(&mut self) -> &mut InternalNode<K, V, IC, LC> {
		match self {
			Node::Internal(ref mut internal) => internal,
			Node::Leaf(_) => {
				panic!("expected internal node");
			}
		}
	}

	#[cfg(test)]
	#[inline]
	pub(crate) fn keys(&self) -> &[K] {
		match self {
			Node::Internal(ref internal) => &internal.keys,
			Node::Leaf(ref leaf) => &leaf.keys,
		}
	}

	#[inline]
	pub(crate) fn sample_key(&self) -> Option<&K> {
		match self {
			Node::Internal(ref internal) => internal.sample_key.as_ref(),
			Node::Leaf(ref leaf) => leaf.sample_key.as_ref(),
		}
	}

	#[inline]
	pub(crate) fn is_underfull(&self) -> bool {
		match self {
			Node::Internal(ref internal) => internal.is_underfull(),
			Node::Leaf(ref leaf) => leaf.is_underfull(),
		}
	}

	#[inline]
	pub(crate) fn can_merge_with(&self, other: &Self) -> bool {
		match self {
			Node::Internal(ref internal) => match other {
				Node::Internal(ref other) => ((internal.len + 1 + other.len) as usize) < IC,
				_ => false,
			},
			Node::Leaf(ref leaf) => match other {
				Node::Leaf(ref other) => ((leaf.len + other.len) as usize) < LC,
				_ => false,
			},
		}
	}
}

pub(crate) struct LeafNode<K, V, const LC: usize> {
	pub(crate) len: u16,
	pub(crate) keys: SmallVec<[K; LC]>,
	pub(crate) values: SmallVec<[V; LC]>,
	pub(crate) lower_fence: Option<K>,
	pub(crate) upper_fence: Option<K>,
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

	#[inline]
	pub(crate) fn lower_bound<Q>(&self, key: &Q) -> (u16, bool)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		if self.lower_fence().map(|fk| key < fk.borrow()).unwrap_or(false) {
			return (0, false);
		}

		if let Some(fk) = self.upper_fence() {
			if key > fk.borrow() {
				return (self.len, false);
			}
		}

		let mut lower = 0;
		let mut upper = self.len;

		while lower < upper {
			let mid = ((upper - lower) / 2) + lower;

			if key < unsafe { self.keys.get_unchecked(mid as usize) }.borrow() {
				upper = mid;
			} else if key > unsafe { self.keys.get_unchecked(mid as usize) }.borrow() {
				lower = mid + 1;
			} else {
				return (mid, true);
			}
		}

		(lower, false)
	}

	#[inline]
	pub(crate) fn lower_fence(&self) -> Option<&K> {
		self.lower_fence.as_ref()
	}

	#[inline]
	pub(crate) fn upper_fence(&self) -> Option<&K> {
		self.upper_fence.as_ref()
	}

	#[inline]
	pub(crate) fn value_at(&self, pos: u16) -> error::Result<&V> {
		Ok(unsafe { self.values.get_unchecked(pos as usize) })
	}

	#[inline]
	pub(crate) fn key_at(&self, pos: u16) -> error::Result<&K> {
		Ok(unsafe { self.keys.get_unchecked(pos as usize) })
	}

	#[inline]
	pub(crate) fn kv_at(&self, pos: u16) -> error::Result<(&K, &V)> {
		Ok((self.key_at(pos)?, self.value_at(pos)?))
	}

	#[inline]
	pub(crate) fn kv_at_mut(&mut self, pos: u16) -> error::Result<(&K, &mut V)> {
		Ok(unsafe {
			(self.keys.get_unchecked(pos as usize), self.values.get_unchecked_mut(pos as usize))
		})
	}

	#[inline]
	pub(crate) fn has_space(&self) -> bool {
		(self.len as usize) < LC
	}

	#[inline]
	pub(crate) fn is_underfull(&self) -> bool {
		(self.len as usize) < (LC as f32 * 0.4) as usize
	}

	pub(crate) fn insert_at(&mut self, pos: u16, key: K, value: V) -> Option<u16> {
		if !self.has_space() {
			return None;
		}

		self.keys.insert(pos as usize, key);
		self.values.insert(pos as usize, value);
		self.len += 1;

		Some(pos)
	}

	pub(crate) fn remove_at(&mut self, pos: u16) -> (K, V) {
		self.len -= 1;
		let key = self.keys.remove(pos as usize);
		let value = self.values.remove(pos as usize);

		(key, value)
	}

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
	pub(crate) fn split(&mut self, right: &mut LeafNode<K, V, LC>, split_pos: u16) {
		let split_key = self.key_at(split_pos).expect("should exist").clone();
		right.lower_fence = Some(split_key.clone());
		right.upper_fence = self.upper_fence.clone();

		self.upper_fence = Some(split_key);

		assert!(right.keys.is_empty());
		assert!(right.values.is_empty());
		right.keys.extend(self.keys.drain((split_pos + 1) as usize..));
		right.values.extend(self.values.drain((split_pos + 1) as usize..));

		self.sample_key = Some(self.keys[0].clone());
		right.sample_key = Some(right.keys[0].clone());

		right.len = right.keys.len() as u16;
		self.len = self.keys.len() as u16;
	}

	pub(crate) fn merge(&mut self, right: &mut LeafNode<K, V, LC>) -> bool {
		if (self.len + right.len) as usize > LC {
			return false;
		}

		right.len = 0;
		self.upper_fence = right.upper_fence.take();
		self.keys.extend(right.keys.drain(..));
		self.values.extend(right.values.drain(..));

		self.sample_key = right.sample_key.take();

		self.len = self.keys.len() as u16;
		true
	}
}

pub(crate) struct InternalNode<K, V, const IC: usize, const LC: usize> {
	pub(crate) len: u16,
	pub(crate) keys: SmallVec<[K; IC]>,
	pub(crate) edges: SmallVec<[Atomic<HybridLatch<Node<K, V, IC, LC>>>; IC]>,
	pub(crate) upper_edge: Option<Atomic<HybridLatch<Node<K, V, IC, LC>>>>,
	pub(crate) lower_fence: Option<K>,
	pub(crate) upper_fence: Option<K>,
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

	#[inline]
	pub(crate) fn lower_bound<Q>(&self, key: &Q) -> (u16, bool)
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		if self.lower_fence().map(|fk| key < fk.borrow()).unwrap_or(false) {
			return (0, false);
		}

		if let Some(fk) = self.upper_fence() {
			if key > fk.borrow() {
				return (self.len, false);
			}
		}

		let mut lower = 0;
		let mut upper = self.len;

		while lower < upper {
			let mid = ((upper - lower) / 2) + lower;

			if key < unsafe { self.keys.get_unchecked(mid as usize) }.borrow() {
				upper = mid;
			} else if key > unsafe { self.keys.get_unchecked(mid as usize) }.borrow() {
				lower = mid + 1;
			} else {
				return (mid, true);
			}
		}

		(lower, false)
	}

	#[inline]
	pub(crate) fn lower_fence(&self) -> Option<&K> {
		self.lower_fence.as_ref()
	}

	#[inline]
	pub(crate) fn upper_fence(&self) -> Option<&K> {
		self.upper_fence.as_ref()
	}

	#[inline]
	pub(crate) fn edge_at(
		&self,
		pos: u16,
	) -> error::Result<&Atomic<HybridLatch<Node<K, V, IC, LC>>>> {
		if pos == self.len {
			if let Some(upper_edge) = self.upper_edge.as_ref() {
				Ok(upper_edge)
			} else {
				Err(error::Error::Unwind)
			}
		} else {
			Ok(unsafe { self.edges.get_unchecked(pos as usize) })
		}
	}

	#[inline]
	pub(crate) fn key_at(&self, pos: u16) -> error::Result<&K> {
		Ok(unsafe { self.keys.get_unchecked(pos as usize) })
	}

	#[inline]
	pub(crate) fn has_space(&self) -> bool {
		(self.len as usize) < IC
	}

	#[inline]
	pub(crate) fn is_underfull(&self) -> bool {
		(self.len as usize) < (IC as f32 * 0.4) as usize
	}

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
			unimplemented!("upserts");
		} else {
			if !self.has_space() {
				return None;
			}

			self.keys.insert(pos as usize, key);
			self.edges.insert(pos as usize, value);
			self.len += 1;
		}
		Some(pos)
	}

	pub(crate) fn remove_at(&mut self, pos: u16) -> (K, Atomic<HybridLatch<Node<K, V, IC, LC>>>) {
		let key = self.keys.remove(pos as usize);
		let edge = self.edges.remove(pos as usize);
		self.len -= 1;

		(key, edge)
	}

	/// Insert a separator key and new right child after a split.
	/// The left child remains at edges[pos], and we insert:
	/// - the split_key at keys[pos] (shifting keys[pos:] right)
	/// - the right child at edges[pos+1] (shifting edges[pos+1:] right)
	pub(crate) fn insert_after(
		&mut self,
		pos: u16,
		key: K,
		edge: Atomic<HybridLatch<Node<K, V, IC, LC>>>,
	) {
		// Insert key at position pos (this becomes the separator between left and right)
		self.keys.insert(pos as usize, key);
		// Insert edge at position pos+1 (right child, after the left child at pos)
		self.edges.insert((pos + 1) as usize, edge);
		self.len += 1;
	}
}

impl<K: Clone, V, const IC: usize, const LC: usize> InternalNode<K, V, IC, LC> {
	pub(crate) fn split(&mut self, right: &mut InternalNode<K, V, IC, LC>, split_pos: u16) {
		let split_key = self.key_at(split_pos).expect("should exist").clone();
		right.lower_fence = Some(split_key.clone());
		right.upper_fence = self.upper_fence.clone();

		self.upper_fence = Some(split_key);

		assert!(right.keys.is_empty());
		assert!(right.edges.is_empty());
		right.keys.extend(self.keys.drain((split_pos + 1) as usize..));
		right.edges.extend(self.edges.drain((split_pos + 1) as usize..));
		right.upper_edge = self.upper_edge.take();

		self.upper_edge = Some(self.edges.pop().unwrap());
		self.keys.pop().unwrap();

		self.sample_key = Some(self.keys[0].clone());
		right.sample_key = Some(right.keys[0].clone());

		right.len = right.keys.len() as u16;
		self.len = self.keys.len() as u16;
	}

	pub(crate) fn merge(&mut self, right: &mut InternalNode<K, V, IC, LC>) -> bool {
		if (self.len + right.len + 1) as usize > IC {
			return false;
		}

		let _left_upper_fence = std::mem::replace(&mut self.upper_fence, right.upper_fence.take());
		let left_upper_edge = std::mem::replace(&mut self.upper_edge, right.upper_edge.take());

		self.keys.push(right.lower_fence.take().expect("right node must have lower fence"));
		self.edges.push(left_upper_edge.expect("left node must have upper edge"));

		self.keys.extend(right.keys.drain(..));
		self.edges.extend(right.edges.drain(..));

		self.sample_key = right.sample_key.take();

		self.len = self.keys.len() as u16;
		right.len = 0;

		true
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn basic_insert_and_lookup() {
		let tree: Tree<i32, &str> = Tree::new();

		assert_eq!(tree.insert(1, "one"), None);
		assert_eq!(tree.insert(2, "two"), None);
		assert_eq!(tree.insert(3, "three"), None);

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
	}

	#[test]
	fn remove() {
		let tree: Tree<i32, &str> = Tree::new();

		tree.insert(1, "one");
		tree.insert(2, "two");

		assert_eq!(tree.remove(&1), Some("one"));
		assert_eq!(tree.lookup(&1, |v| *v), None);
		assert_eq!(tree.lookup(&2, |v| *v), Some("two"));
	}

	#[test]
	fn raw_iter() {
		let tree: Tree<i32, i32> = Tree::new();

		for i in 0..100 {
			tree.insert(i, i * 10);
		}

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

		tree.remove(&1);
		assert_eq!(tree.len(), 1);
	}
}
