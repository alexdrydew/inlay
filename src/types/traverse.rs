use std::convert::Infallible;

use super::{
    ArenaFamily, CallableType, Concrete, Keyed, LazyRefType, OpaqueParamSpec, OpaqueTypeVar,
    ParamSpecType, Parametric, PlainType, ProtocolType, PyType, PyTypeConcreteKey, PyTypeKey,
    PyTypeParametricKey, Qual, Qualified, SentinelType, TypeVarSupport, TypeVarType, TypedDictType,
    UnionType, Wrapper,
};

pub(crate) trait TypeChildren<D: 'static> {
    fn children(&self) -> impl Iterator<Item = &D>;
}

impl<T: TypeChildren<D>, D: 'static> TypeChildren<D> for Qualified<T> {
    fn children(&self) -> impl Iterator<Item = &D> {
        self.inner.children()
    }
}

impl<T: TypeChildren<D>, D: 'static> TypeChildren<D> for &T {
    fn children(&self) -> impl Iterator<Item = &D> {
        (**self).children()
    }
}

impl<D: 'static> TypeChildren<D> for Infallible {
    fn children(&self) -> impl Iterator<Item = &D> {
        std::iter::empty()
    }
}

impl<D: 'static> TypeChildren<D> for SentinelType {
    fn children(&self) -> impl Iterator<Item = &D> {
        std::iter::empty()
    }
}

impl<D: 'static> TypeChildren<D> for TypeVarType {
    fn children(&self) -> impl Iterator<Item = &D> {
        std::iter::empty()
    }
}

impl<D: 'static> TypeChildren<D> for ParamSpecType {
    fn children(&self) -> impl Iterator<Item = &D> {
        std::iter::empty()
    }
}

impl<D: 'static> TypeChildren<D> for OpaqueTypeVar {
    fn children(&self) -> impl Iterator<Item = &D> {
        std::iter::empty()
    }
}

impl<D: 'static> TypeChildren<D> for OpaqueParamSpec {
    fn children(&self) -> impl Iterator<Item = &D> {
        std::iter::empty()
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for PlainType<I, G> {
    fn children(&self) -> impl Iterator<Item = &PyType<I, I, G>> {
        self.args.iter()
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for ProtocolType<I, G> {
    fn children(&self) -> impl Iterator<Item = &PyType<I, I, G>> {
        self.methods
            .values()
            .chain(self.attributes.values())
            .chain(self.properties.values())
            .chain(self.type_params.iter())
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> TypeChildren<PyType<I, I, G>>
    for TypedDictType<I, G>
{
    fn children(&self) -> impl Iterator<Item = &PyType<I, I, G>> {
        self.attributes.values().chain(self.type_params.iter())
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for UnionType<I, G> {
    fn children(&self) -> impl Iterator<Item = &PyType<I, I, G>> {
        self.variants.iter()
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for CallableType<I, G> {
    fn children(&self) -> impl Iterator<Item = &PyType<I, I, G>> {
        self.params
            .values()
            .chain(std::iter::once(&self.return_type))
            .chain(self.type_params.iter())
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> TypeChildren<PyType<I, I, G>> for LazyRefType<I, G> {
    fn children(&self) -> impl Iterator<Item = &PyType<I, I, G>> {
        std::iter::once(&self.target)
    }
}

// --- MapChildren ---

pub(crate) trait MapChildren<S: ArenaFamily, From: TypeVarSupport, To: TypeVarSupport> {
    type Output;
    fn map_children(
        self,
        apply: &mut impl FnMut(PyTypeKey<S, From>) -> PyTypeKey<S, To>,
    ) -> Self::Output;
}

impl<T: MapChildren<S, From, To>, S: ArenaFamily, From: TypeVarSupport, To: TypeVarSupport>
    MapChildren<S, From, To> for Qualified<T>
{
    type Output = Qualified<T::Output>;
    fn map_children(
        self,
        apply: &mut impl FnMut(PyTypeKey<S, From>) -> PyTypeKey<S, To>,
    ) -> Self::Output {
        Qualified {
            inner: self.inner.map_children(apply),
            qualifier: self.qualifier,
        }
    }
}

impl<S: ArenaFamily> MapChildren<S, Parametric, Concrete>
    for PlainType<Qual<Keyed<S>>, Parametric>
{
    type Output = PlainType<Qual<Keyed<S>>, Concrete>;
    fn map_children(
        self,
        apply: &mut impl FnMut(PyTypeParametricKey<S>) -> PyTypeConcreteKey<S>,
    ) -> Self::Output {
        PlainType {
            descriptor: self.descriptor,
            args: self.args.into_iter().map(|k| apply(k)).collect(),
        }
    }
}

impl<S: ArenaFamily> MapChildren<S, Parametric, Concrete>
    for ProtocolType<Qual<Keyed<S>>, Parametric>
{
    type Output = ProtocolType<Qual<Keyed<S>>, Concrete>;
    fn map_children(
        self,
        apply: &mut impl FnMut(PyTypeParametricKey<S>) -> PyTypeConcreteKey<S>,
    ) -> Self::Output {
        ProtocolType {
            descriptor: self.descriptor,
            methods: self
                .methods
                .into_iter()
                .map(|(n, t)| (n, apply(t)))
                .collect(),
            attributes: self
                .attributes
                .into_iter()
                .map(|(n, t)| (n, apply(t)))
                .collect(),
            properties: self
                .properties
                .into_iter()
                .map(|(n, t)| (n, apply(t)))
                .collect(),
            type_params: self.type_params.into_iter().map(|k| apply(k)).collect(),
        }
    }
}

impl<S: ArenaFamily> MapChildren<S, Parametric, Concrete>
    for TypedDictType<Qual<Keyed<S>>, Parametric>
{
    type Output = TypedDictType<Qual<Keyed<S>>, Concrete>;
    fn map_children(
        self,
        apply: &mut impl FnMut(PyTypeParametricKey<S>) -> PyTypeConcreteKey<S>,
    ) -> Self::Output {
        TypedDictType {
            descriptor: self.descriptor,
            attributes: self
                .attributes
                .into_iter()
                .map(|(n, t)| (n, apply(t)))
                .collect(),
            type_params: self.type_params.into_iter().map(|k| apply(k)).collect(),
        }
    }
}

impl<S: ArenaFamily> MapChildren<S, Parametric, Concrete>
    for UnionType<Qual<Keyed<S>>, Parametric>
{
    type Output = UnionType<Qual<Keyed<S>>, Concrete>;
    fn map_children(
        self,
        apply: &mut impl FnMut(PyTypeParametricKey<S>) -> PyTypeConcreteKey<S>,
    ) -> Self::Output {
        UnionType {
            variants: self.variants.into_iter().map(|k| apply(k)).collect(),
        }
    }
}

impl<S: ArenaFamily> MapChildren<S, Parametric, Concrete>
    for CallableType<Qual<Keyed<S>>, Parametric>
{
    type Output = CallableType<Qual<Keyed<S>>, Concrete>;
    fn map_children(
        self,
        apply: &mut impl FnMut(PyTypeParametricKey<S>) -> PyTypeConcreteKey<S>,
    ) -> Self::Output {
        CallableType {
            params: self
                .params
                .into_iter()
                .map(|(n, t)| (n, apply(t)))
                .collect(),
            param_kinds: self.param_kinds,
            param_has_default: self.param_has_default,
            accepts_varargs: self.accepts_varargs,
            accepts_varkw: self.accepts_varkw,
            return_type: apply(self.return_type),
            return_wrapper: self.return_wrapper,
            type_params: self.type_params.into_iter().map(|k| apply(k)).collect(),
            function_name: self.function_name,
        }
    }
}

impl<S: ArenaFamily> MapChildren<S, Parametric, Concrete>
    for LazyRefType<Qual<Keyed<S>>, Parametric>
{
    type Output = LazyRefType<Qual<Keyed<S>>, Concrete>;
    fn map_children(
        self,
        apply: &mut impl FnMut(PyTypeParametricKey<S>) -> PyTypeConcreteKey<S>,
    ) -> Self::Output {
        LazyRefType {
            target: apply(self.target),
        }
    }
}
