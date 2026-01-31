//! # Error Types for the Concurrent B+ Tree
//!
//! This module defines error types used internally by the B+ tree for
//! handling optimistic concurrency failures.
//!
//! ## Error Handling Strategy
//!
//! The B+ tree uses optimistic concurrency control, where most operations
//! proceed without blocking and validate their reads at the end. When
//! validation fails, operations don't panic - they return errors that
//! signal the caller to retry.
//!
//! ## Error Flow
//!
//! ```text
//! Operation starts
//!      │
//!      ▼
//! Acquire optimistic access
//!      │
//!      ▼
//! Read data (may be inconsistent)
//!      │
//!      ▼
//! Validate reads ──────────► Err(Unwind) ───► Retry operation
//!      │
//!      ▼ (Ok)
//! Perform side effects
//!      │
//!      ▼
//! Return success
//! ```
//!
//! ## Common Patterns
//!
//! Most tree operations follow this pattern:
//!
//! ```ignore
//! loop {
//!     let perform = || {
//!         let guard = self.find_leaf(key, eg)?;  // May return Unwind
//!         let result = guard.some_operation()?;   // May return Unwind
//!         guard.recheck()?;                       // May return Unwind
//!         Ok(result)
//!     };
//!
//!     match perform() {
//!         Ok(result) => return result,
//!         Err(Error::Unwind) => continue,        // Retry
//!         Err(Error::Reclaimed) => continue,     // Retry
//!     }
//! }
//! ```

use thiserror::Error;

/// Errors that can occur during B+ tree operations.
///
/// These errors are used for internal flow control in the optimistic
/// concurrency system. They typically cause operations to retry rather
/// than fail permanently.
#[derive(Error, Debug)]
pub enum Error {
	/// Optimistic validation failed - the data we read may be invalid.
	///
	/// This error occurs when:
	/// - A write occurred between our read and validation
	/// - We couldn't acquire a lock upgrade (try_read/try_write failed)
	/// - The version number changed during our operation
	///
	/// # Response
	///
	/// When receiving this error:
	/// 1. Discard any data read during the failed operation
	/// 2. Retry the entire operation from the beginning
	/// 3. The retry will capture a fresh version number
	///
	/// # Name Origin
	///
	/// "Unwind" refers to unwinding the call stack - discarding all
	/// work done and returning to a safe state to retry.
	#[error("optimistic validation failed")]
	Unwind,

	/// The node we're operating on has been removed from the tree.
	///
	/// This error occurs during `find_parent()` when:
	/// - The target node has no `sample_key` (was emptied and reclaimed)
	/// - The tree structure changed significantly during traversal
	///
	/// This is a stronger form of `Unwind` - not only is our read invalid,
	/// but the node itself may no longer be part of the tree structure.
	///
	/// # Response
	///
	/// When receiving this error:
	/// 1. The node reference is no longer useful
	/// 2. Re-seek from a known-good position (e.g., from an anchor)
	/// 3. Or restart the operation entirely
	///
	/// # Difference from Unwind
	///
	/// - `Unwind`: "Your reads might be wrong, try again"
	/// - `Reclaimed`: "This node doesn't exist anymore, start over"
	#[error("called find_parent on reclaimed node")]
	Reclaimed,
}

/// A Result type alias using our custom Error type.
///
/// Used throughout the B+ tree codebase for operations that can fail
/// due to optimistic concurrency issues.
pub type Result<T> = std::result::Result<T, Error>;
