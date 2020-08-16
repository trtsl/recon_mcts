#![cfg_attr(feature = "nightly", feature(generic_associated_types))]

use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::collections::{BinaryHeap, HashSet};
use std::hash::Hash;
use std::ops::Deref;

struct OrdWrap<T>(T);

// TODO: use GAT `Order<'a>` such that it can reference `self` instead of having to return an owned
// value.  `UID` must still be owned if restricted to safe code because it is self-referential in
// `UniqueHeap`.  Not really needed for the mcts since `Order` is only implemented as a `usize`,
// but possibly desirable for more generic circumstances.
#[cfg(feature = "nightly")]
pub trait HeapElem {
    /// `Order` and `UID` may not be modified while stored in the `UniqueHeap` (required by
    /// `BinaryHeap` and `HashSet`, respectively)
    type Order<'a>: PartialOrd + Ord + PartialEq + Eq;
    type UID: Hash + Eq;
    fn order<'b>(&'b self) -> Self::Order<'b>;
    fn unique_id(&self) -> Self::UID;
}

#[cfg(feature = "stable")]
pub trait HeapElem {
    /// `Order` and `UID` may not be modified while stored in the `UniqueHeap` (required by
    /// `BinaryHeap` and `HashSet`, respectively)
    type Order: PartialOrd + Ord + PartialEq + Eq;
    type UID: Hash + Eq;
    fn order(&self) -> Self::Order;
    fn unique_id(&self) -> Self::UID;
}

impl<T> Deref for OrdWrap<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: HeapElem> Ord for OrdWrap<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.order().cmp(&other.order())
    }
}

impl<T: HeapElem> PartialOrd for OrdWrap<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.order().partial_cmp(&other.order())
    }
}

impl<T: HeapElem> PartialEq for OrdWrap<T> {
    fn eq(&self, other: &Self) -> bool {
        self.order().eq(&other.order())
    }
}

impl<T: HeapElem> Eq for OrdWrap<T> {}

pub struct UniqueHeap<N, H> {
    heap: BinaryHeap<OrdWrap<N>>,
    registry: HashSet<H>,
}

impl<N, H> UniqueHeap<N, H>
where
    N: HeapElem<UID = H>,
    H: Hash + Eq,
{
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            registry: HashSet::new(),
        }
    }

    pub fn push(&mut self, n: N) -> bool {
        if self.registry.insert(n.unique_id()) {
            let wrap = OrdWrap(n);
            self.heap.push(wrap);
            true
        } else {
            false
        }
    }

    pub fn pop(&mut self) -> Option<N> {
        let n = self.heap.pop()?.0;
        let r = self.registry.remove(&n.unique_id());
        debug_assert!(r);
        Some(n)
    }
}

#[cfg(all(test, feature = "stable"))]
mod test {
    use super::*;

    impl HeapElem for u8 {
        type Order = Self;
        type UID = Self;
        fn order(&self) -> Self::Order {
            *self
        }
        fn unique_id(&self) -> Self::UID {
            *self
        }
    }

    #[test]
    fn test_heap_u8() {
        let mut h = UniqueHeap::new();
        assert_eq!(h.registry.len(), 0);
        assert_eq!(h.push(5), true);
        assert_eq!(h.registry.len(), 1);
        assert_eq!(h.push(8), true);
        assert_eq!(h.registry.len(), 2);
        assert_eq!(h.push(1), true);
        assert_eq!(h.registry.len(), 3);
        assert_eq!(h.push(8), false);
        assert_eq!(h.registry.len(), 3);
        assert_eq!(h.pop(), Some(8));
        assert_eq!(h.pop(), Some(5));
        assert_eq!(h.pop(), Some(1));
        assert_eq!(h.pop(), None);
        assert_eq!(h.registry.len(), 0);
    }
}

#[cfg(all(test, feature = "nightly"))]
mod test {
    use super::*;

    impl HeapElem for Vec<u8> {
        type Order<'a> = &'a u8;
        type UID = u8;
        fn order<'b>(&'b self) -> Self::Order<'b> {
            &self[0]
        }
        fn unique_id(&self) -> Self::UID {
            *&self[0]
        }
    }

    #[test]
    fn test_heap_u8() {
        let mut h = UniqueHeap::new();
        assert_eq!(h.registry.len(), 0);
        assert_eq!(h.push(vec![5, 0, 0]), true);
        assert_eq!(h.registry.len(), 1);
        assert_eq!(h.push(vec![8, 0, 0]), true);
        assert_eq!(h.registry.len(), 2);
        assert_eq!(h.push(vec![1, 0, 0]), true);
        assert_eq!(h.registry.len(), 3);
        assert_eq!(h.push(vec![8, 0, 0]), false);
        assert_eq!(h.registry.len(), 3);
        assert_eq!(h.pop(), Some(vec![8, 0, 0]));
        assert_eq!(h.pop(), Some(vec![5, 0, 0]));
        assert_eq!(h.pop(), Some(vec![1, 0, 0]));
        assert_eq!(h.pop(), None);
        assert_eq!(h.registry.len(), 0);
    }
}
