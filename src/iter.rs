//! Iterators for the `GenericTree` data structure
use crate::error;
use crate::latch::{ExclusiveGuard, OptimisticGuard, SharedGuard};
use crate::{Direction, GenericTree, Node};
use crossbeam_epoch::{self as epoch};
use std::borrow::Borrow;

#[derive(Debug, PartialEq, Copy, Clone)]
enum Anchor<T> {
	Start,
	After(T),
	Before(T),
	End,
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum Cursor {
	After(u16),
	Before(u16),
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum LeafResult {
	Ok,
	End,
	Retry,
}

#[derive(Debug)]
enum JumpResult {
	Ok,
	End,
	Err(error::Error),
}

/// Raw shared iterator over the entries of the tree.
pub struct RawSharedIter<'t, K, V, const IC: usize, const LC: usize> {
	tree: &'t GenericTree<K, V, IC, LC>,
	eg: epoch::Guard,
	parent: Option<(OptimisticGuard<'t, Node<K, V, IC, LC>>, u16)>,
	leaf: Option<(SharedGuard<'t, Node<K, V, IC, LC>>, Cursor)>,
}

impl<'t, K: Clone + Ord, V, const IC: usize, const LC: usize> RawSharedIter<'t, K, V, IC, LC> {
	pub(crate) fn new(tree: &'t GenericTree<K, V, IC, LC>) -> RawSharedIter<'t, K, V, IC, LC> {
		RawSharedIter {
			tree,
			eg: epoch::pin(),
			parent: None,
			leaf: None,
		}
	}

	#[inline]
	fn leaf_lt<'g>(
		guard: SharedGuard<'g, Node<K, V, IC, LC>>,
	) -> SharedGuard<'t, Node<K, V, IC, LC>> {
		// Safety: We hold the epoch guard at all times so 'g should equal 't
		unsafe { std::mem::transmute(guard) }
	}

	#[inline]
	fn parent_lt<'g>(
		guard: OptimisticGuard<'g, Node<K, V, IC, LC>>,
	) -> OptimisticGuard<'t, Node<K, V, IC, LC>> {
		// Safety: We hold the epoch guard at all times so 'g should equal 't
		unsafe { std::mem::transmute(guard) }
	}

	fn current_anchor(&self) -> Option<Anchor<K>> {
		if let Some((guard, cursor)) = self.leaf.as_ref() {
			let leaf = guard.as_leaf();
			let anchor = match *cursor {
				Cursor::Before(pos) => {
					if pos >= leaf.len {
						if leaf.len > 0 {
							Anchor::After(leaf.key_at(leaf.len - 1).expect("should exist").clone())
						} else if let Some(k) = &leaf.lower_fence {
							Anchor::After(k.clone())
						} else {
							Anchor::Start
						}
					} else {
						Anchor::Before(leaf.key_at(pos).expect("should exist").clone())
					}
				}
				Cursor::After(pos) => {
					if pos >= leaf.len {
						if leaf.len > 0 {
							Anchor::After(leaf.key_at(leaf.len - 1).expect("should exist").clone())
						} else if let Some(k) = &leaf.upper_fence {
							Anchor::After(k.clone())
						} else {
							Anchor::End
						}
					} else {
						Anchor::After(leaf.key_at(pos).expect("should exist").clone())
					}
				}
			};

			Some(anchor)
		} else {
			None
		}
	}

	fn restore_from_anchor(&mut self, anchor: &Anchor<K>) {
		self.parent.take();
		self.leaf.take(); // Make sure there are no locks held

		match anchor {
			Anchor::Start => self.seek_to_first(),
			Anchor::End => self.seek_to_last(),
			Anchor::Before(key) => self.seek(key),
			Anchor::After(key) => self.seek_for_prev(key),
		}
	}

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
				JumpResult::Err(_) => {
					self.restore_from_anchor(&anchor);
					continue;
				}
			}
		}
	}

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
				JumpResult::Err(_) => {
					self.restore_from_anchor(&anchor);
					continue;
				}
			}
		}
	}

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
					let guard = {
						let swip = parent_guard.as_internal().edge_at(pos)?;
						GenericTree::lock_coupling(parent_guard, swip, &self.eg)?
					};

					assert!(guard.is_leaf());

					error::Result::Ok(Outcome::Leaf(Self::leaf_lt(guard.to_shared()?), pos))
				} else {
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
				self.parent.as_mut().map(|(_, p_c)| *p_c = p_cursor);

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
			Err(e) => JumpResult::Err(e),
		}
	}

	/// Sets the iterator cursor immediately before the position for this key.
	pub fn seek<Q>(&mut self, key: &Q)
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
		let (pos, _) = leaf.lower_bound(key);
		if pos >= leaf_len {
			self.leaf = Some((guard, Cursor::Before(leaf_len)));
			self.parent = parent_opt;
		} else {
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		}
	}

	/// Sets the iterator cursor immediately after the position for this key.
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
		if exact {
			self.leaf = Some((guard, Cursor::After(pos)));
			self.parent = parent_opt;
		} else if pos == 0 {
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		} else if pos >= leaf_len {
			self.leaf = Some((guard, Cursor::Before(leaf_len)));
			self.parent = parent_opt;
		} else {
			self.leaf = Some((guard, Cursor::After(pos)));
			self.parent = parent_opt;
		}
	}

	/// Sets the iterator cursor immediately before the position for this key, returning `true` if
	/// the next entry matches the provided key.
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
		if pos >= leaf_len {
			self.leaf = Some((guard, Cursor::Before(leaf_len)));
			self.parent = parent_opt;
		} else {
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		}

		exact
	}

	/// Sets the iterator cursor immediately before the position for the first key in the tree.
	pub fn seek_to_first(&mut self) {
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

		self.leaf = Some((guard, Cursor::Before(0)));
		self.parent = parent_opt;
	}

	/// Sets the iterator cursor immediately after the position for the last key in the tree.
	pub fn seek_to_last(&mut self) {
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

		let leaf_len = guard.as_leaf().len;
		self.leaf = Some((guard, Cursor::Before(leaf_len)));
		self.parent = parent_opt;
	}

	/// Returns the next entry from the current cursor position.
	#[inline]
	pub fn next(&mut self) -> Option<(&K, &V)> {
		loop {
			let opt = match self.leaf.as_ref() {
				Some((guard, cursor)) => {
					let leaf = guard.as_leaf();
					match *cursor {
						Cursor::Before(pos) => {
							if pos < leaf.len {
								Some((pos, Cursor::Before(pos + 1)))
							} else {
								None
							}
						}
						Cursor::After(pos) => {
							let curr_pos = pos + 1;
							if curr_pos < leaf.len {
								Some((curr_pos, Cursor::Before(curr_pos + 1)))
							} else {
								None
							}
						}
					}
				}
				None => {
					return None;
				}
			};

			if let Some((curr_pos, new_cursor)) = opt {
				let (guard, cursor) = self.leaf.as_mut().unwrap();
				let leaf = guard.as_leaf();
				*cursor = new_cursor;
				return Some(leaf.kv_at(curr_pos).expect("should exist"));
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

	/// Returns the previous entry from the current cursor position.
	#[inline]
	pub fn prev(&mut self) -> Option<(&K, &V)> {
		loop {
			let opt = match self.leaf.as_ref() {
				Some((_guard, cursor)) => match *cursor {
					Cursor::After(pos) => {
						if pos > 0 {
							Some((pos, Cursor::After(pos - 1)))
						} else if pos == 0 {
							Some((pos, Cursor::Before(pos)))
						} else {
							None
						}
					}
					Cursor::Before(pos) => {
						if pos > 0 {
							let curr_pos = pos - 1;
							if curr_pos == 0 {
								Some((curr_pos, Cursor::Before(curr_pos)))
							} else {
								Some((curr_pos, Cursor::After(curr_pos - 1)))
							}
						} else {
							None
						}
					}
				},
				None => {
					return None;
				}
			};

			if let Some((curr_pos, new_cursor)) = opt {
				let (guard, cursor) = self.leaf.as_mut().unwrap();
				let leaf = guard.as_leaf();
				*cursor = new_cursor;
				return Some(leaf.kv_at(curr_pos).expect("should exist"));
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
}

/// Raw exclusive iterator over the entries of the tree.
pub struct RawExclusiveIter<'t, K, V, const IC: usize, const LC: usize> {
	tree: &'t GenericTree<K, V, IC, LC>,
	eg: epoch::Guard,
	parent: Option<(OptimisticGuard<'t, Node<K, V, IC, LC>>, u16)>,
	leaf: Option<(ExclusiveGuard<'t, Node<K, V, IC, LC>>, Cursor)>,
}

impl<'t, K: Clone + Ord, V, const IC: usize, const LC: usize> RawExclusiveIter<'t, K, V, IC, LC> {
	pub(crate) fn new(tree: &'t GenericTree<K, V, IC, LC>) -> RawExclusiveIter<'t, K, V, IC, LC> {
		RawExclusiveIter {
			tree,
			eg: epoch::pin(),
			parent: None,
			leaf: None,
		}
	}

	#[inline]
	fn leaf_lt<'g>(
		guard: ExclusiveGuard<'g, Node<K, V, IC, LC>>,
	) -> ExclusiveGuard<'t, Node<K, V, IC, LC>> {
		unsafe { std::mem::transmute(guard) }
	}

	#[inline]
	fn parent_lt<'g>(
		guard: OptimisticGuard<'g, Node<K, V, IC, LC>>,
	) -> OptimisticGuard<'t, Node<K, V, IC, LC>> {
		unsafe { std::mem::transmute(guard) }
	}

	fn current_anchor(&self) -> Option<Anchor<K>> {
		if let Some((guard, cursor)) = self.leaf.as_ref() {
			let leaf = guard.as_leaf();
			let anchor = match *cursor {
				Cursor::Before(pos) => {
					if pos >= leaf.len {
						if leaf.len > 0 {
							Anchor::After(leaf.key_at(leaf.len - 1).expect("should exist").clone())
						} else if let Some(k) = &leaf.lower_fence {
							Anchor::After(k.clone())
						} else {
							Anchor::Start
						}
					} else {
						Anchor::Before(leaf.key_at(pos).expect("should exist").clone())
					}
				}
				Cursor::After(pos) => {
					if pos >= leaf.len {
						if leaf.len > 0 {
							Anchor::After(leaf.key_at(leaf.len - 1).expect("should exist").clone())
						} else if let Some(k) = &leaf.upper_fence {
							Anchor::After(k.clone())
						} else {
							Anchor::End
						}
					} else {
						Anchor::After(leaf.key_at(pos).expect("should exist").clone())
					}
				}
			};

			Some(anchor)
		} else {
			None
		}
	}

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
				JumpResult::Err(_) => {
					self.restore_from_anchor(&anchor);
					continue;
				}
			}
		}
	}

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
				JumpResult::Err(_) => {
					self.restore_from_anchor(&anchor);
					continue;
				}
			}
		}
	}

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
					let guard = {
						let swip = parent_guard.as_internal().edge_at(pos)?;
						GenericTree::lock_coupling(parent_guard, swip, &self.eg)?
					};

					assert!(guard.is_leaf());

					error::Result::Ok(Outcome::Leaf(Self::leaf_lt(guard.to_exclusive()?), pos))
				} else {
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
				self.parent.as_mut().map(|(_, p_c)| *p_c = p_cursor);

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
			Err(e) => JumpResult::Err(e),
		}
	}

	/// Sets the iterator cursor immediately before the position for this key.
	pub fn seek<Q>(&mut self, key: &Q)
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
		let (pos, _) = leaf.lower_bound(key);
		if pos >= leaf_len {
			self.leaf = Some((guard, Cursor::Before(leaf_len)));
			self.parent = parent_opt;
		} else {
			self.leaf = Some((guard, Cursor::Before(pos)));
			self.parent = parent_opt;
		}
	}

	/// Sets the iterator cursor immediately after the position for this key.
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
			self.leaf = Some((guard, Cursor::Before(leaf_len)));
			self.parent = parent_opt;
		} else {
			self.leaf = Some((guard, Cursor::After(pos)));
			self.parent = parent_opt;
		}
	}

	/// Sets the iterator cursor immediately before the position for this key, returning `true` if
	/// the next entry matches the provided key.
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

	/// Sets the iterator cursor immediately before the position for the first key in the tree.
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

	/// Sets the iterator cursor immediately after the position for the last key in the tree.
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

	/// Inserts the key value pair in the tree.
	pub fn insert(&mut self, key: K, value: V) -> Option<V> {
		'start: loop {
			if self.seek_exact(&key) {
				let (_k, v) = self.next().unwrap();
				let old = std::mem::replace(v, value);
				break Some(old);
			} else {
				let (guard, cursor) = self.leaf.as_mut().expect("just seeked");
				if guard.as_leaf().has_space() {
					let leaf = guard.as_leaf_mut();
					match *cursor {
						Cursor::Before(pos) => {
							leaf.insert_at(pos, key, value).expect("just checked for space");
						}
						Cursor::After(_) => {
							unreachable!("seek_exact always sets cursor to before");
						}
					}
					break None;
				} else {
					self.parent.take();
					let (guard, _cursor) = self.leaf.take().expect("just seeked");
					let mut guard = guard.unlock();

					loop {
						let perform_split = || {
							if !guard.as_leaf().has_space() {
								guard.recheck()?;
								self.tree.try_split(&guard, &self.eg)?;
							}
							error::Result::Ok(())
						};

						match perform_split() {
							Ok(_) => break,
							Err(error::Error::Reclaimed) => {
								continue 'start;
							}
							Err(_) => {
								guard = guard.latch().optimistic_or_spin();
								continue;
							}
						}
					}

					continue;
				}
			}
		}
	}

	/// Removes the entry associated with this key from the tree.
	pub fn remove<Q>(&mut self, key: &Q) -> Option<(K, V)>
	where
		K: Borrow<Q> + Ord,
		Q: ?Sized + Ord,
	{
		if self.seek_exact(key) {
			Some(self.remove_next().expect("just seeked for remove"))
		} else {
			None
		}
	}

	fn remove_next(&mut self) -> Option<(K, V)> {
		match self.leaf.as_mut() {
			Some((guard, cursor)) => {
				let leaf = guard.as_leaf_mut();

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

				if let Some((removed_key, _)) = removed.as_ref() {
					if guard.is_underfull() {
						self.parent.take();
						let (guard, _cursor) = self.leaf.take().expect("just seeked");

						let guard = guard.unlock();
						loop {
							let perform_merge = || {
								let _ = self.tree.try_merge(&guard, &self.eg)?;
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

						self.seek(removed_key);
					}
				}

				removed
			}
			None => None,
		}
	}

	/// Returns the next entry from the current cursor position.
	#[inline]
	pub fn next(&mut self) -> Option<(&K, &mut V)> {
		loop {
			let opt = match self.leaf.as_ref() {
				Some((guard, cursor)) => {
					let leaf = guard.as_leaf();
					match cursor {
						Cursor::Before(pos) => {
							let pos = *pos;
							if pos < leaf.len {
								Some((pos, Cursor::Before(pos + 1)))
							} else {
								None
							}
						}
						Cursor::After(pos) => {
							let pos = *pos;
							let curr_pos = pos + 1;
							if curr_pos < leaf.len {
								Some((curr_pos, Cursor::Before(curr_pos + 1)))
							} else {
								None
							}
						}
					}
				}
				None => {
					return None;
				}
			};

			if let Some((curr_pos, new_cursor)) = opt {
				let (guard, cursor) = self.leaf.as_mut().unwrap();
				let leaf = guard.as_leaf_mut();
				*cursor = new_cursor;
				return Some(leaf.kv_at_mut(curr_pos).expect("should exist"));
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

	/// Returns the previous entry from the current cursor position.
	#[inline]
	pub fn prev(&mut self) -> Option<(&K, &mut V)> {
		loop {
			let opt = match self.leaf.as_ref() {
				Some((_guard, cursor)) => match *cursor {
					Cursor::After(pos) => {
						if pos > 0 {
							Some((pos, Cursor::After(pos - 1)))
						} else if pos == 0 {
							Some((pos, Cursor::Before(pos)))
						} else {
							None
						}
					}
					Cursor::Before(pos) => {
						if pos > 0 {
							let curr_pos = pos - 1;
							if curr_pos == 0 {
								Some((curr_pos, Cursor::Before(curr_pos)))
							} else {
								Some((curr_pos, Cursor::After(curr_pos - 1)))
							}
						} else {
							None
						}
					}
				},
				None => {
					return None;
				}
			};

			if let Some((curr_pos, new_cursor)) = opt {
				let (guard, cursor) = self.leaf.as_mut().unwrap();
				let leaf = guard.as_leaf_mut();
				*cursor = new_cursor;
				return Some(leaf.kv_at_mut(curr_pos).expect("should exist"));
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
}
