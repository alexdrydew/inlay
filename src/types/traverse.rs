use std::convert::Infallible;

use super::{
    CallableType, LazyRefType, OpaqueParamSpec, OpaqueTypeVar, ParamSpecType, PlainType,
    ProtocolType, PyType, Qualified, SentinelType, TypeVarSupport, TypeVarType, TypedDictType,
    UnionType, ViewRef, Wrapper,
};

pub(crate) trait TypeChildren<D> {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a;
}

impl<T: TypeChildren<D>, D> TypeChildren<D> for Qualified<T> {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a,
    {
        self.inner.children()
    }
}

impl<T: TypeChildren<D>, D> TypeChildren<D> for &T {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a,
    {
        (**self).children()
    }
}

impl<T: TypeChildren<D>, D> TypeChildren<D> for ViewRef<'_, T> {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a,
    {
        (**self).children()
    }
}

impl<D> TypeChildren<D> for Infallible {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a,
    {
        std::iter::empty()
    }
}

impl<D> TypeChildren<D> for SentinelType {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a,
    {
        std::iter::empty()
    }
}

impl<D> TypeChildren<D> for TypeVarType {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a,
    {
        std::iter::empty()
    }
}

impl<D> TypeChildren<D> for ParamSpecType {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a,
    {
        std::iter::empty()
    }
}

impl<D> TypeChildren<D> for OpaqueTypeVar {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a,
    {
        std::iter::empty()
    }
}

impl<D> TypeChildren<D> for OpaqueParamSpec {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a D>
    where
        D: 'a,
    {
        std::iter::empty()
    }
}

impl<I: Wrapper, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for PlainType<I, G> {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a PyType<I, I, G>>
    where
        PyType<I, I, G>: 'a,
    {
        self.args.iter()
    }
}

impl<I: Wrapper, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for ProtocolType<I, G> {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a PyType<I, I, G>>
    where
        PyType<I, I, G>: 'a,
    {
        self.methods
            .values()
            .chain(self.attributes.values())
            .chain(self.properties.values())
            .chain(self.type_params.iter())
    }
}

impl<I: Wrapper, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for TypedDictType<I, G> {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a PyType<I, I, G>>
    where
        PyType<I, I, G>: 'a,
    {
        self.attributes.values().chain(self.type_params.iter())
    }
}

impl<I: Wrapper, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for UnionType<I, G> {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a PyType<I, I, G>>
    where
        PyType<I, I, G>: 'a,
    {
        self.variants.iter()
    }
}

impl<I: Wrapper, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for CallableType<I, G> {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a PyType<I, I, G>>
    where
        PyType<I, I, G>: 'a,
    {
        self.params
            .values()
            .chain(std::iter::once(&self.return_type))
            .chain(self.type_params.iter())
    }
}

impl<I: Wrapper, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for LazyRefType<I, G> {
    fn children<'a>(&'a self) -> impl Iterator<Item = &'a PyType<I, I, G>>
    where
        PyType<I, I, G>: 'a,
    {
        std::iter::once(&self.target)
    }
}
