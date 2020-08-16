// The purpose of this module is to provide a way to turn an iterator of generic items into an
// iterator of trait object safe (i.e. `Sized`) items without heap allocation (a streaming iterator
// adapter).  Creating the trait object without heap allocation requires creating a reference, but
// the reference cannot be created inside e.g. `Iterator::map` because `Iterator::map` receives an
// owned value and therefore cannot return a reference to it. The approach used here has the user
// provide a stack space where the variable can be stored by the iterator.  The implementation uses
// a `RefCell` to store the variable because the iterator needs to mutate the variable, and if a
// mutable reference is used, only the reference held by the iterator can ever exist.  But we also
// need to return a value referencing the variable from the iterator, so using a mutable reference
// for the stack placeholder is not possible.  An additional benefit of using a shared reference is
// that `RefIter` is now `Clone`, though iterations of cloned iterators cannot be interleaved.  If
// the returned `Ref` of a cloned iterator is stored on the stack, calling `next` on the other
// iterator will panic.

use std::cell::{Ref, RefCell};

#[derive(Clone, Debug)]
pub struct RefIter<'a, I, T> {
    iter: I,
    item: &'a RefCell<Option<T>>,
}

impl<'a, I, T> RefIter<'a, I, T> {
    fn new(iter: I, item: &'a RefCell<Option<T>>) -> Self {
        Self { iter, item }
    }
}

// Iterator extension trait
pub trait RefIterator: Sized + Iterator {
    fn ref_iter<'a>(self, item: &'a RefCell<Option<Self::Item>>) -> RefIter<'a, Self, Self::Item> {
        RefIter::new(self, item)
    }
}

impl<T: Sized + Iterator> RefIterator for T {}

impl<'a, I, T> Iterator for RefIter<'a, I, T>
where
    I: Iterator<Item = T>,
{
    type Item = Ref<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.item
            .try_borrow_mut()
            .expect("A prior `Ref` item was borrowed after `next()` was called on the iterator")
            .replace(self.iter.next()?);
        // The `try_borrow_mut` in the line above requires that the `Ref` returned by `borrow`
        // below must go out of scope before `next` is called; this implies that an item returned
        // by the iterator cannot be dereferenced after the next item is obtained from the iterator
        Some(Ref::map(self.item.borrow(), |x: &Option<T>| {
            x.as_ref().unwrap()
        }))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::ops::Deref;

    #[test]
    fn test_iter() {
        let item = RefCell::new(None);
        let items = (0..5).ref_iter(&item);
        let mapped = items.map(|x| *x).collect::<Vec<_>>();
        assert_eq!(mapped, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        let item = RefCell::new(None);
        let items = RefIter::<_, u8>::new(0..5, &item);
        let _ = items.collect::<Vec<_>>();
    }

    #[test]
    fn test_interleave_cloned() {
        let item = RefCell::new(None);
        let items = RefIter::<_, u8>::new(0..5, &item);

        let mut a = items.clone();
        let mut b = items.clone();

        assert_eq!(a.next().map(|x| *x), Some(0));
        assert_eq!(a.next().map(|x| *x), Some(1));
        assert_eq!(b.next().map(|x| *x), Some(0));
        assert_eq!(a.next().map(|x| *x), Some(2));
        assert_eq!(b.next().map(|x| *x), Some(1));
        assert_eq!(a.next().map(|x| *x), Some(3));
        assert_eq!(b.next().map(|x| *x), Some(2));
        assert_eq!(a.next().map(|x| *x), Some(4));
        assert_eq!(a.next().map(|x| *x), None);
        assert_eq!(a.next().map(|x| *x), None);
        assert_eq!(b.next().map(|x| *x), Some(3));
        assert_eq!(b.next().map(|x| *x), Some(4));
        assert_eq!(b.next().map(|x| *x), None);
        assert_eq!(b.next().map(|x| *x), None);
        assert_eq!(a.next().map(|x| *x), None);
    }

    #[test]
    #[allow(clippy::map_clone)]
    #[allow(clippy::clone_on_copy)]
    fn test_interleave_map() {
        let item = RefCell::new(None);
        let items = (0..5).ref_iter(&item);

        // `max_by` causes items to be interleaved, so using it requires either dereferencing and
        // cloning prior to calling max_by or collecting everything into a `Vec`
        items
            .map(|x| x.clone())
            .max_by(|a, b| a.partial_cmp(&*b).unwrap());
    }

    #[test]
    #[should_panic(
        expected = "A prior `Ref` item was borrowed after `next()` was called on the iterator"
    )]
    fn test_interleave_cloned_and_held() {
        let item = RefCell::new(None);
        let items = RefIter::<_, u8>::new(0..5, &item);

        let mut a = items.clone();
        let mut b = items.clone();

        let _r1 = a.next();
        let _r2 = b.next();
    }

    #[test]
    fn test_dyn() {
        #[allow(dead_code, unused_variables)]
        fn transform_iter_generic_to_dyn<I, S, T>(iter_in: I)
        where
            I: Iterator<Item = S>,
            S: Deref<Target = T>,
        {
            let item = RefCell::new(None);
            let mut iter = iter_in.ref_iter(&item).map(|x| Ref::map(x, |x| x.deref()));
            let iter: &mut (dyn Iterator<Item = Ref<T>>) = &mut iter;
        }
    }
}
