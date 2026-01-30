//! Test utilities for loading sample trees from JSON fixtures
use crate::latch::HybridLatch;
use crate::{InternalNode, LeafNode, Node, Tree};
use crossbeam_epoch::Atomic;
use serde::Deserialize;
use smallvec::smallvec;
use std::sync::atomic::AtomicUsize;

type DefaultNode = Node<String, u64, 64, 64>;

#[derive(Deserialize, Debug)]
struct Edge {
	key: String,
	child: TreeNode,
}

#[derive(Deserialize, Debug)]
struct Value {
	key: String,
	value: u64,
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum TreeNode {
	Internal {
		edges: Vec<Edge>,
		upper_edge: Box<TreeNode>,
		lower_fence: Option<String>,
		upper_fence: Option<String>,
		sample_key: Option<String>,
	},
	Leaf {
		values: Vec<Value>,
		lower_fence: Option<String>,
		upper_fence: Option<String>,
		sample_key: Option<String>,
	},
}

#[derive(Deserialize, Debug)]
struct SampleTree {
	root: TreeNode,
	height: usize,
}

fn translate_node(tree_node: TreeNode) -> Atomic<HybridLatch<DefaultNode>> {
	match tree_node {
		TreeNode::Internal {
			edges,
			upper_edge,
			lower_fence,
			upper_fence,
			sample_key,
		} => {
			let mut out_keys = smallvec![];
			let mut out_edges = smallvec![];
			for edge in edges {
				out_keys.push(edge.key);
				out_edges.push(translate_node(edge.child));
			}

			let out_upper_edge = Some(translate_node(*upper_edge));

			Atomic::new(HybridLatch::new(Node::Internal(InternalNode {
				len: out_keys.len() as u16,
				keys: out_keys,
				edges: out_edges,
				upper_edge: out_upper_edge,
				lower_fence,
				upper_fence,
				sample_key,
			})))
		}
		TreeNode::Leaf {
			values,
			lower_fence,
			upper_fence,
			sample_key,
		} => {
			let mut out_keys = smallvec![];
			let mut out_values = smallvec![];
			for value in values {
				out_keys.push(value.key);
				out_values.push(value.value);
			}

			Atomic::new(HybridLatch::new(Node::Leaf(LeafNode {
				len: out_keys.len() as u16,
				keys: out_keys,
				values: out_values,
				lower_fence,
				upper_fence,
				sample_key,
			})))
		}
	}
}

pub fn sample_tree<P: AsRef<std::path::Path>>(path: P) -> Tree<String, u64> {
	let file = std::fs::File::open(path).expect("failed to find file");
	let json_tree: SampleTree = serde_json::from_reader(file).unwrap();
	let translated = translate_node(json_tree.root);
	Tree {
		root: HybridLatch::new(translated),
		height: AtomicUsize::new(json_tree.height),
	}
}
