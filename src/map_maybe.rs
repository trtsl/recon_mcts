use std::marker::PhantomData;
use std::ops::Deref;

/// This module provides functionality similar to `lockref` but allows mapping the contents of an
/// `&Option<I>` using `Fn(&I) -> &O` returning `Option<&O>`; the `lockref` module cannot be used
/// for this because `Deref::deref` would need to return an `&Option<&O>` but the type
/// `Option::<&O>` does not already exist anywhere and needs to be created inside `Deref::deref`,
/// which then cannot return it as a reference; thus the trait `MapMaybe<'_>` returns the
/// `Option::<&O>` by value

pub trait MapMaybe<'a> {
    type Target;
    fn map<'b>(&'b self) -> Option<&'b Self::Target>
    where
        'a: 'b;
}

impl<'a, T, O> MapMaybe<'a> for T
where
    T: Deref<Target = Option<O>>,
{
    type Target = O;
    fn map<'b>(&'b self) -> Option<&'b Self::Target>
    where
        'a: 'b,
    {
        self.deref().as_ref()
    }
}

pub(crate) struct Ref<T, F, I, O> {
    r: T,
    f: F,
    _m: PhantomData<fn(&I) -> &O>,
}

impl<T, F, I, O> Ref<T, F, I, O>
where
    F: Fn(&I) -> &O,
{
    pub(crate) fn new(r: T, f: F) -> Self {
        Self {
            r,
            f,
            _m: PhantomData,
        }
    }
}

impl<'a, T, F, I, O> MapMaybe<'a> for Ref<T, F, I, O>
where
    T: MapMaybe<'a, Target = I>,
    F: Fn(&I) -> &O,
    I: 'a,
{
    type Target = O;
    fn map<'b>(&'b self) -> Option<&'b Self::Target>
    where
        'a: 'b,
    {
        Some((self.f)(self.r.map()?))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::RwLock;

    fn make_iter<'a>(
        v: &'a Vec<(u8, RwLock<Option<(u64, String)>>)>,
    ) -> impl IntoIterator<Item = (&'a u8, impl 'a + Deref<Target = Option<(u64, String)>>)> {
        v.iter().map(|(a, s)| (a, s.read().unwrap()))
    }

    // note that some of these explicit lifetimes can be elided as in `as_map_maybe` below
    fn as_deref<'a: 'b, 'b, II, A, S>(
        v: II,
    ) -> impl IntoIterator<Item = (A, impl 'b + MapMaybe<'a, Target = String>)>
    where
        II: IntoIterator<Item = (A, S)>,
        A: 'a + Deref<Target = u8>,
        S: 'a + Deref<Target = Option<(u64, String)>>,
    {
        v.into_iter().map(|x| {
            (
                x.0,
                Ref::new(x.1, (|x| &x.1) as fn(&(u64, String)) -> &String),
            )
        })
    }

    fn as_map_maybe<'a, II, A, S>(
        v: II,
    ) -> impl IntoIterator<Item = (A, impl MapMaybe<'a, Target = String>)>
    where
        II: IntoIterator<Item = (A, S)>,
        A: Deref<Target = u8>,
        S: MapMaybe<'a, Target = (u64, String)>,
    {
        v.into_iter().map(|x| {
            (
                x.0,
                Ref::new(x.1, (|x| &x.1) as fn(&(u64, String)) -> &String),
            )
        })
    }

    fn checks<'a, II, A, M>(v: II)
    where
        II: IntoIterator<Item = (A, M)>,
        A: Deref<Target = u8>,
        M: MapMaybe<'a, Target = String>,
    {
        let mut it = v.into_iter();
        it.next()
            .map(|(x, y)| assert_eq!((x.deref(), y.map()), (&0, Some(&"10".to_string()))));
        it.next()
            .map(|(x, y)| assert_eq!((x.deref(), y.map()), (&1, Some(&"11".to_string()))));
        it.next()
            .map(|(x, y)| assert_eq!((x.deref(), y.map()), (&2, None)));
        assert!(it.next().is_none());
    }

    #[test]
    fn test() {
        let mut v = Vec::new();
        v.push((0, RwLock::new(Some((10, "10".to_string())))));
        v.push((1, RwLock::new(Some((11, "11".to_string())))));
        v.push((2, RwLock::new(None)));

        let v1 = make_iter(&v);
        let v1 = as_deref(v1);
        checks(v1);

        let v2 = make_iter(&v);
        let v2 = as_map_maybe(v2);
        checks(v2);
    }
}
