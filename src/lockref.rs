// This module is used to emulate the behavior of `std::cell::Ref::map` for smart pointer types
// that do not provide a similar construct (e.g. `Mutex` and `RwLock`); it is slightly less
// efficient because the map is recalculated on each `Deref::deref` whereas `Ref::map` is able to
// store the results of the map using `Ref`'s internals; for the reason `map` is not implemented
// for `Mutex` and `RwLock` see: https://stackoverflow.com/q/40095383/#comment90293227_40095383

use std::ops::Deref;

pub struct Ref<In, F> {
    r: In,
    f: F,
}

impl<In, F, Out> Ref<In, F>
where
    F: Fn(&In) -> &Out,
    Out: ?Sized,
{
    pub fn new(r: In, f: F) -> Self {
        Ref { r, f }
    }
}

impl<In, F, Out> Deref for Ref<In, F>
where
    F: Fn(&In) -> &Out,
    Out: ?Sized,
{
    type Target = Out;
    fn deref(&self) -> &Self::Target {
        (self.f)(&self.r)
    }
}
