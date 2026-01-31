//! # Test Utilities for the B+ Tree
//!
//! This module provides utilities for loading sample trees from JSON fixtures.
//! It's only compiled in test builds (`#[cfg(test)]`).
//!
//! ## Purpose
//!
//! Creating B+ trees programmatically for testing is tedious - you'd need to
//! perform many insertions and the tree structure depends on the insertion
//! order and split points. Instead, this module lets you define the exact
//! tree structure in JSON format.
//!
//! ## JSON Format
//!
//! The JSON fixture format mirrors the internal tree structure:
//!
//! ```json
//! {
//!   "root": <TreeNode>,
//!   "height": <number>
//! }
//! ```
//!
//! Where `<TreeNode>` is either an internal node or a leaf node:
//!
//! ### Internal Node
//!
//! ```json
//! {
//!   "edges": [
//!     { "key": "separator_key", "child": <TreeNode> },
//!     ...
//!   ],
//!   "upper_edge": <TreeNode>,
//!   "lower_fence": "optional_lower_bound",
//!   "upper_fence": "optional_upper_bound",
//!   "sample_key": "optional_sample_key"
//! }
//! ```
//!
//! ### Leaf Node
//!
//! ```json
//! {
//!   "values": [
//!     { "key": "key1", "value": 123 },
//!     { "key": "key2", "value": 456 },
//!     ...
//!   ],
//!   "lower_fence": "optional_lower_bound",
//!   "upper_fence": "optional_upper_bound",
//!   "sample_key": "optional_sample_key"
//! }
//! ```
//!
//! ## Example Fixture
//!
//! A simple two-level tree with one internal node and two leaves:
//!
//! ```json
//! {
//!   "root": {
//!     "edges": [
//!       {
//!         "key": "m",
//!         "child": {
//!           "values": [
//!             { "key": "apple", "value": 1 },
//!             { "key": "banana", "value": 2 }
//!           ],
//!           "lower_fence": null,
//!           "upper_fence": "m",
//!           "sample_key": "apple"
//!         }
//!       }
//!     ],
//!     "upper_edge": {
//!       "values": [
//!         { "key": "orange", "value": 3 },
//!         { "key": "pear", "value": 4 }
//!       ],
//!       "lower_fence": "m",
//!       "upper_fence": null,
//!       "sample_key": "orange"
//!     },
//!     "lower_fence": null,
//!     "upper_fence": null,
//!     "sample_key": "apple"
//!   },
//!   "height": 2
//! }
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! #[test]
//! fn test_with_fixture() {
//!     let tree = sample_tree("fixtures/sample.json");
//!     // Now test operations on the tree
//!     assert_eq!(tree.lookup(&"apple", |v| *v), Some(1));
//! }
//! ```
//!
//! ## Limitations
//!
//! - Keys are always `String`
//! - Values are always `u64`
//! - Uses default node capacities (64)

use crate::latch::HybridLatch;
use crate::{InternalNode, LeafNode, Node, Tree};
use crossbeam_epoch::Atomic;
use serde::Deserialize;
use smallvec::smallvec;
use std::sync::atomic::AtomicUsize;

// ---------------------------------------------------------------------------
// Type Aliases
// ---------------------------------------------------------------------------

/// The node type used in fixtures: String keys, u64 values, default capacities.
type DefaultNode = Node<String, u64, 64, 64>;

// ---------------------------------------------------------------------------
// JSON Deserialization Structures
// ---------------------------------------------------------------------------

/// An edge in an internal node - a separator key and child pointer.
#[derive(Deserialize, Debug)]
struct Edge {
	/// The separator key. Keys < this go to this edge, keys >= go to the next edge.
	key: String,
	/// The child node (can be internal or leaf).
	child: TreeNode,
}

/// A key-value entry in a leaf node.
#[derive(Deserialize, Debug)]
struct Value {
	/// The entry's key.
	key: String,
	/// The entry's value.
	value: u64,
}

/// A node in the JSON tree structure.
///
/// Uses `#[serde(untagged)]` to distinguish between internal and leaf nodes
/// based on their structure (internal has `edges`, leaf has `values`).
#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum TreeNode {
	/// An internal (index) node with separator keys and child pointers.
	Internal {
		/// List of (separator_key, child) pairs.
		/// The child at edges[i] contains keys < edges[i].key.
		edges: Vec<Edge>,
		/// The rightmost child (contains keys >= last separator).
		upper_edge: Box<TreeNode>,
		/// Optional lower bound for keys in this subtree.
		lower_fence: Option<String>,
		/// Optional upper bound for keys in this subtree.
		upper_fence: Option<String>,
		/// A key that routes to this node (for find_parent).
		sample_key: Option<String>,
	},
	/// A leaf node containing actual key-value pairs.
	Leaf {
		/// The key-value entries, sorted by key.
		values: Vec<Value>,
		/// Optional lower bound (keys in this leaf are > lower_fence).
		lower_fence: Option<String>,
		/// Optional upper bound (keys in this leaf are <= upper_fence).
		upper_fence: Option<String>,
		/// A key that routes to this node.
		sample_key: Option<String>,
	},
}

/// Top-level structure of a JSON fixture file.
#[derive(Deserialize, Debug)]
struct SampleTree {
	/// The root node of the tree.
	root: TreeNode,
	/// The height of the tree (1 = single leaf, 2 = one internal + leaves, etc.)
	height: usize,
}

// ---------------------------------------------------------------------------
// Translation Functions
// ---------------------------------------------------------------------------

/// Recursively translates a JSON tree node into the actual tree node structure.
///
/// This function converts the deserialized JSON representation into the
/// internal tree representation with proper latches and atomic pointers.
///
/// # Arguments
///
/// * `tree_node` - A deserialized JSON tree node
///
/// # Returns
///
/// An atomic pointer to a latched node, suitable for use in the tree.
fn translate_node(tree_node: TreeNode) -> Atomic<HybridLatch<DefaultNode>> {
	match tree_node {
		TreeNode::Internal {
			edges,
			upper_edge,
			lower_fence,
			upper_fence,
			sample_key,
		} => {
			// Build the keys and edges arrays
			let mut out_keys = smallvec![];
			let mut out_edges = smallvec![];

			for edge in edges {
				out_keys.push(edge.key);
				// Recursively translate child nodes
				out_edges.push(translate_node(edge.child));
			}

			// Translate the upper_edge (rightmost child)
			let out_upper_edge = Some(translate_node(*upper_edge));

			// Wrap in HybridLatch and Atomic for thread-safe access
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
			// Build the keys and values arrays
			let mut out_keys = smallvec![];
			let mut out_values = smallvec![];

			for value in values {
				out_keys.push(value.key);
				out_values.push(value.value);
			}

			// Wrap in HybridLatch and Atomic
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Loads a sample tree from a JSON fixture file.
///
/// This is the main entry point for using fixtures in tests. It reads the
/// JSON file, parses it, and constructs a fully-functional B+ tree.
///
/// # Arguments
///
/// * `path` - Path to the JSON fixture file
///
/// # Returns
///
/// A `Tree<String, u64>` with the structure defined in the fixture.
///
/// # Panics
///
/// Panics if the file cannot be found or parsed.
///
/// # Example
///
/// ```ignore
/// let tree = sample_tree("fixtures/sample.json");
/// assert_eq!(tree.lookup(&"key1".to_string(), |v| *v), Some(100));
/// ```
pub fn sample_tree<P: AsRef<std::path::Path>>(path: P) -> Tree<String, u64> {
	// Read and parse the JSON file
	let file = std::fs::File::open(path).expect("failed to find file");
	let json_tree: SampleTree = serde_json::from_reader(file).unwrap();

	// Translate the JSON structure to actual tree nodes
	let translated = translate_node(json_tree.root);

	// Construct the tree with the translated root
	Tree {
		root: HybridLatch::new(translated),
		height: AtomicUsize::new(json_tree.height),
	}
}
