<br>

<p align="center">
    <a href="https://surrealdb.com#gh-dark-mode-only" target="_blank">
        <img width="200" src="/img/white/logo.svg" alt="Ferntree Logo">
    </a>
    <a href="https://surrealdb.com#gh-light-mode-only" target="_blank">
        <img width="200" src="/img/black/logo.svg" alt="Ferntree Logo">
    </a>
</p>

<h3 align="center">Ferntree</h3>

<p align="center">A concurrent in-memory B+ tree featuring optimistic lock coupling.</p>

<br>

<p align="center">
	<a href="https://github.com/surrealdb/ferntree"><img src="https://img.shields.io/badge/status-beta-ff00bb.svg?style=flat-square"></a>
	&nbsp;
	<a href="https://docs.rs/ferntree/"><img src="https://img.shields.io/docsrs/ferntree?style=flat-square"></a>
	&nbsp;
	<a href="https://crates.io/crates/ferntree"><img src="https://img.shields.io/crates/v/ferntree?style=flat-square"></a>
	&nbsp;
	<a href="https://github.com/surrealdb/ferntree"><img src="https://img.shields.io/badge/license-Apache_License_2.0-00bfff.svg?style=flat-square"></a>
</p>

#### Features

- High-throughput concurrent access using optimistic lock coupling
- Three access modes: Optimistic (non-blocking), Shared (read), Exclusive (write)
- Thread-safe (`Send + Sync`) when key and value types are thread-safe
- Epoch-based memory reclamation for safe concurrent access
- Bidirectional iteration with `next()` and `prev()`
- Range queries with configurable bounds
- Automatic node splitting and merging

#### Design

Ferntree is based on research from:
- [LeanStore](https://dbis1.github.io/leanstore.html) - Optimistic lock coupling for B-trees
- [Umbra](https://umbra-db.com/#publications) - High-performance database engine techniques

The key innovation is **optimistic lock coupling**: readers acquire "optimistic" access (no locks) and validate at the end that no concurrent modifications occurred. If validation fails, the operation retries automatically. This allows readers to proceed without blocking writers, and vice versa.

#### Quick start

```rust
use ferntree::Tree;

fn main() {
    let tree: Tree<String, i32> = Tree::new();

    // Insert key-value pairs
    tree.insert("apple".to_string(), 1);
    tree.insert("banana".to_string(), 2);
    tree.insert("cherry".to_string(), 3);

    // Lookup values using a closure
    let value = tree.lookup(&"banana".to_string(), |v| *v);
    assert_eq!(value, Some(2));

    // Convenience method for cloneable values
    let value = tree.get("apple");
    assert_eq!(value, Some(1));

    // Check existence
    assert!(tree.contains_key("cherry"));
    assert!(!tree.contains_key("durian"));

    // Remove entries
    let removed = tree.remove(&"banana".to_string());
    assert_eq!(removed, Some(2));

    // Tree statistics
    assert_eq!(tree.len(), 2);
    assert!(!tree.is_empty());
}
```

#### Concurrent usage

The tree is designed for high-concurrency workloads. Wrap it in an `Arc` to share across threads:

```rust
use ferntree::Tree;
use std::sync::Arc;
use std::thread;

fn main() {
    let tree = Arc::new(Tree::<i32, i32>::new());

    // Spawn multiple writer threads
    let handles: Vec<_> = (0..4).map(|t| {
        let tree = Arc::clone(&tree);
        thread::spawn(move || {
            for i in 0..1000 {
                tree.insert(t * 1000 + i, i);
            }
        })
    }).collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(tree.len(), 4000);
}
```

Concurrent readers and writers can operate simultaneously:

```rust
use ferntree::Tree;
use std::sync::Arc;
use std::thread;

fn main() {
    let tree = Arc::new(Tree::<i32, i32>::new());

    // Pre-populate some data
    for i in 0..100 {
        tree.insert(i, i * 10);
    }

    let tree_writer = Arc::clone(&tree);
    let tree_reader = Arc::clone(&tree);

    // Writer thread adds new entries
    let writer = thread::spawn(move || {
        for i in 100..200 {
            tree_writer.insert(i, i * 10);
        }
    });

    // Reader thread performs lookups concurrently
    let reader = thread::spawn(move || {
        let mut found = 0;
        for i in 0..100 {
            if tree_reader.lookup(&i, |v| *v).is_some() {
                found += 1;
            }
        }
        found
    });

    writer.join().unwrap();
    let found = reader.join().unwrap();

    assert_eq!(found, 100); // Reader sees consistent data
}
```

#### Iterator usage

Ferntree provides several ways to iterate over entries:

```rust
use ferntree::Tree;
use std::ops::Bound::{Included, Excluded, Unbounded};

fn main() {
    let tree: Tree<i32, &str> = Tree::new();
    tree.insert(1, "one");
    tree.insert(2, "two");
    tree.insert(3, "three");
    tree.insert(4, "four");
    tree.insert(5, "five");

    // Raw iterator for manual control
    let mut iter = tree.raw_iter();
    iter.seek_to_first();
    while let Some((k, v)) = iter.next() {
        println!("{}: {}", k, v);
    }

    // Reverse iteration
    iter.seek_to_last();
    while let Some((k, v)) = iter.prev() {
        println!("{}: {}", k, v);
    }

    // Keys iterator
    let mut keys = tree.keys();
    assert_eq!(keys.next(), Some(&1));
    assert_eq!(keys.next(), Some(&2));

    // Values iterator
    let mut values = tree.values();
    assert_eq!(values.next(), Some(&"five")); // Sorted by key
    
    // Range queries
    let mut range = tree.range(Included(&2), Excluded(&5));
    assert_eq!(range.next(), Some((&2, &"two")));
    assert_eq!(range.next(), Some((&3, &"three")));
    assert_eq!(range.next(), Some((&4, &"four")));
    assert_eq!(range.next(), None);
}
```

#### Additional operations

```rust
use ferntree::Tree;

fn main() {
    let tree: Tree<i32, i32> = Tree::new();
    
    for i in 0..10 {
        tree.insert(i, i * 10);
    }

    // First and last entries
    let first = tree.first(|k, v| (*k, *v));
    assert_eq!(first, Some((0, 0)));

    let last = tree.last(|k, v| (*k, *v));
    assert_eq!(last, Some((9, 90)));

    // Pop operations (remove and return)
    let popped = tree.pop_first();
    assert_eq!(popped, Some((0, 0)));

    let popped = tree.pop_last();
    assert_eq!(popped, Some((9, 90)));

    // Get or insert (returns existing or inserts default)
    let value = tree.get_or_insert(100, 1000);
    assert_eq!(value, 1000);

    let value = tree.get_or_insert(1, 9999); // Key exists
    assert_eq!(value, 10); // Returns existing value

    // Clear all entries
    tree.clear();
    assert!(tree.is_empty());
}
```

#### API overview

**Read operations:**
- `lookup(key, closure)` - Look up a value, applying a closure to extract data
- `get(key)` - Get a cloned value (requires `V: Clone`)
- `contains_key(key)` - Check if a key exists
- `first(closure)` - Get the first (minimum) entry
- `last(closure)` - Get the last (maximum) entry
- `len()` - Number of entries
- `is_empty()` - Check if tree is empty
- `height()` - Current tree height

**Write operations:**
- `insert(key, value)` - Insert or update, returns old value if key existed
- `remove(key)` - Remove an entry, returns the value if it existed
- `pop_first()` - Remove and return the first entry
- `pop_last()` - Remove and return the last entry
- `get_or_insert(key, default)` - Get existing or insert default
- `get_or_insert_with(key, closure)` - Get existing or insert computed value
- `clear()` - Remove all entries

**Iteration:**
- `raw_iter()` - Low-level shared iterator with `seek*`, `next()`, `prev()`
- `raw_iter_mut()` - Low-level exclusive iterator for modifications
- `keys()` - Iterator over keys
- `values()` - Iterator over values  
- `range(min, max)` - Iterator over a key range

#### Important notes

**Closure execution:** The closures passed to `lookup()`, `first()`, `last()`, etc. may be executed multiple times if concurrent modifications cause validation failures. Avoid side effects in these closures.

```rust
// Good: Pure closure that just extracts data
let value = tree.lookup(&key, |v| v.clone());

// Avoid: Side effects may execute multiple times
let mut counter = 0;
tree.lookup(&key, |v| { counter += 1; v.clone() }); // counter may be > 1
```

**Thread safety:** `Tree<K, V>` implements `Send` and `Sync` when both `K` and `V` implement `Send + Sync`. This means the tree can be safely shared across threads using `Arc`.

**Memory reclamation:** The tree uses epoch-based memory reclamation via `crossbeam-epoch`. Nodes removed from the tree are not immediately freed; they remain valid until all concurrent readers have finished. This ensures safe concurrent access without use-after-free bugs.

#### Original

This code is forked originally from [bplustree](https://crates.io/crates/bplustree), dual-licensed under the [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/) and [MIT](https://choosealicense.com/licenses/mit/) licenses.
