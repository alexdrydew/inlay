use std::convert::Infallible;

use super::{
    ArenaSelector, CallableType, ClassInit, ClassType, Concrete, LazyRefType, OpaqueParamSpec,
    OpaqueTypeVar, ParamSpecType, Parametric, PlainType, ProtocolType, PyType, PyTypeConcreteKey,
    PyTypeKey, PyTypeParametricKey, Qualified, QualifiedMode, SentinelType, TypeArenas,
    TypeVarSupport, TypeVarType, TypedDictType, UnionType, UnqualifiedMode, ViewRef, Wrapper,
};

// --- Trait ---

pub(crate) trait ShallowEq<Rhs = Self> {
    fn shallow_eq(&self, other: &Rhs) -> bool;
}

impl<T: ShallowEq<U> + ?Sized, U> ShallowEq<&U> for &T {
    fn shallow_eq(&self, other: &&U) -> bool {
        (**self).shallow_eq(*other)
    }
}

impl<T: ShallowEq<U>, U> ShallowEq<ViewRef<'_, U>> for ViewRef<'_, T> {
    fn shallow_eq(&self, other: &ViewRef<'_, U>) -> bool {
        (**self).shallow_eq(&**other)
    }
}

impl<T: ShallowEq<U>, U> ShallowEq<Qualified<U>> for Qualified<T> {
    fn shallow_eq(&self, other: &Qualified<U>) -> bool {
        self.qualifier == other.qualifier && self.inner.shallow_eq(&other.inner)
    }
}

// --- Per-type impls ---

impl ShallowEq for Infallible {
    fn shallow_eq(&self, _other: &Self) -> bool {
        match *self {}
    }
}

impl ShallowEq for SentinelType {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl ShallowEq for TypeVarType {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
    }
}

impl ShallowEq for ParamSpecType {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
    }
}

impl ShallowEq for OpaqueTypeVar {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
    }
}

impl ShallowEq for OpaqueParamSpec {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
    }
}

fn class_init_shape_eq<I: Wrapper, G: TypeVarSupport, H: TypeVarSupport>(
    left: Option<&ClassInit<I, G>>,
    right: Option<&ClassInit<I, H>>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => {
            left.params.keys().eq(right.params.keys())
                && left.param_kinds == right.param_kinds
                && left.param_has_default == right.param_has_default
        }
        (None, None) => true,
        _ => false,
    }
}

impl<I: Wrapper, G: TypeVarSupport> ShallowEq for PlainType<I, G> {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
    }
}

impl<I: Wrapper, G: TypeVarSupport> ShallowEq for ClassType<I, G> {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
            && class_init_shape_eq(self.init.as_ref(), other.init.as_ref())
    }
}

impl<I: Wrapper, G: TypeVarSupport> ShallowEq for ProtocolType<I, G> {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
            && self.methods.keys().eq(other.methods.keys())
            && self.attributes.keys().eq(other.attributes.keys())
            && self.properties.keys().eq(other.properties.keys())
    }
}

impl<I: Wrapper, G: TypeVarSupport> ShallowEq for TypedDictType<I, G> {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor && self.attributes.keys().eq(other.attributes.keys())
    }
}

impl<I: Wrapper, G: TypeVarSupport> ShallowEq for UnionType<I, G> {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.variants.len() == other.variants.len()
    }
}

impl<I: Wrapper, G: TypeVarSupport> ShallowEq for CallableType<I, G> {
    fn shallow_eq(&self, other: &Self) -> bool {
        self.params.keys().eq(other.params.keys())
            && self.param_kinds == other.param_kinds
            && self.param_has_default == other.param_has_default
            && self.accepts_varargs == other.accepts_varargs
            && self.accepts_varkw == other.accepts_varkw
            && self.return_wrapper == other.return_wrapper
            && self.type_params.len() == other.type_params.len()
    }
}

impl<I: Wrapper, G: TypeVarSupport> ShallowEq for LazyRefType<I, G> {
    fn shallow_eq(&self, _other: &Self) -> bool {
        true
    }
}

// --- Cross-arena ShallowEq (Concrete ↔ Parametric) ---
// Shallow identity fields (descriptor, member names, etc.) are the same concrete
// types regardless of arena, so cross-arena comparison is well-defined for all
// variants except TypeVar/ParamSpec (which have fundamentally different
// representations per arena and are routed to the wildcard bucket).

impl<I: Wrapper> ShallowEq<PlainType<I, Parametric>> for PlainType<I, Concrete> {
    fn shallow_eq(&self, other: &PlainType<I, Parametric>) -> bool {
        self.descriptor == other.descriptor
    }
}

impl<I: Wrapper> ShallowEq<ClassType<I, Parametric>> for ClassType<I, Concrete> {
    fn shallow_eq(&self, other: &ClassType<I, Parametric>) -> bool {
        self.descriptor == other.descriptor
            && class_init_shape_eq(self.init.as_ref(), other.init.as_ref())
    }
}

impl<I: Wrapper> ShallowEq<ProtocolType<I, Parametric>> for ProtocolType<I, Concrete> {
    fn shallow_eq(&self, other: &ProtocolType<I, Parametric>) -> bool {
        self.descriptor == other.descriptor
            && self.methods.keys().eq(other.methods.keys())
            && self.attributes.keys().eq(other.attributes.keys())
            && self.properties.keys().eq(other.properties.keys())
    }
}

impl<I: Wrapper> ShallowEq<TypedDictType<I, Parametric>> for TypedDictType<I, Concrete> {
    fn shallow_eq(&self, other: &TypedDictType<I, Parametric>) -> bool {
        self.descriptor == other.descriptor && self.attributes.keys().eq(other.attributes.keys())
    }
}

impl<I: Wrapper> ShallowEq<UnionType<I, Parametric>> for UnionType<I, Concrete> {
    fn shallow_eq(&self, other: &UnionType<I, Parametric>) -> bool {
        self.variants.len() == other.variants.len()
    }
}

impl<I: Wrapper> ShallowEq<CallableType<I, Parametric>> for CallableType<I, Concrete> {
    fn shallow_eq(&self, other: &CallableType<I, Parametric>) -> bool {
        self.params.keys().eq(other.params.keys())
            && self.param_kinds == other.param_kinds
            && self.param_has_default == other.param_has_default
            && self.accepts_varargs == other.accepts_varargs
            && self.accepts_varkw == other.accepts_varkw
            && self.return_wrapper == other.return_wrapper
            && self.type_params.len() == other.type_params.len()
    }
}

impl<I: Wrapper> ShallowEq<LazyRefType<I, Parametric>> for LazyRefType<I, Concrete> {
    fn shallow_eq(&self, _other: &LazyRefType<I, Parametric>) -> bool {
        true
    }
}

// --- ShallowEq for PyType ---

impl<O: Wrapper, I: Wrapper, G: TypeVarSupport> ShallowEq for PyType<O, I, G>
where
    O::Wrap<SentinelType>: ShallowEq,
    O::Wrap<G::TypeVar>: ShallowEq,
    O::Wrap<G::ParamSpec>: ShallowEq,
    O::Wrap<PlainType<I, G>>: ShallowEq,
    O::Wrap<ClassType<I, G>>: ShallowEq,
    O::Wrap<ProtocolType<I, G>>: ShallowEq,
    O::Wrap<TypedDictType<I, G>>: ShallowEq,
    O::Wrap<UnionType<I, G>>: ShallowEq,
    O::Wrap<CallableType<I, G>>: ShallowEq,
    O::Wrap<LazyRefType<I, G>>: ShallowEq,
{
    fn shallow_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PyType::Sentinel(a), PyType::Sentinel(b)) => a.shallow_eq(b),
            (PyType::ParamSpec(a), PyType::ParamSpec(b)) => a.shallow_eq(b),
            (PyType::Plain(a), PyType::Plain(b)) => a.shallow_eq(b),
            (PyType::Class(a), PyType::Class(b)) => a.shallow_eq(b),
            (PyType::Protocol(a), PyType::Protocol(b)) => a.shallow_eq(b),
            (PyType::TypedDict(a), PyType::TypedDict(b)) => a.shallow_eq(b),
            (PyType::Union(a), PyType::Union(b)) => a.shallow_eq(b),
            (PyType::Callable(a), PyType::Callable(b)) => a.shallow_eq(b),
            (PyType::LazyRef(a), PyType::LazyRef(b)) => a.shallow_eq(b),
            (PyType::TypeVar(a), PyType::TypeVar(b)) => a.shallow_eq(b),
            _ => false,
        }
    }
}

// --- Cross-arena ShallowEq for PyType (Concrete vs Parametric) ---
// TypeVar/ParamSpec variants have incompatible representations across arenas,
// so no ShallowEq bound is required for them — they fall through to `_ => false`.

impl<O: Wrapper, I: Wrapper> ShallowEq<PyType<O, I, Parametric>> for PyType<O, I, Concrete>
where
    O::Wrap<SentinelType>: ShallowEq,
    O::Wrap<PlainType<I, Concrete>>: ShallowEq<O::Wrap<PlainType<I, Parametric>>>,
    O::Wrap<ClassType<I, Concrete>>: ShallowEq<O::Wrap<ClassType<I, Parametric>>>,
    O::Wrap<ProtocolType<I, Concrete>>: ShallowEq<O::Wrap<ProtocolType<I, Parametric>>>,
    O::Wrap<TypedDictType<I, Concrete>>: ShallowEq<O::Wrap<TypedDictType<I, Parametric>>>,
    O::Wrap<UnionType<I, Concrete>>: ShallowEq<O::Wrap<UnionType<I, Parametric>>>,
    O::Wrap<CallableType<I, Concrete>>: ShallowEq<O::Wrap<CallableType<I, Parametric>>>,
    O::Wrap<LazyRefType<I, Concrete>>: ShallowEq<O::Wrap<LazyRefType<I, Parametric>>>,
{
    fn shallow_eq(&self, other: &PyType<O, I, Parametric>) -> bool {
        match (self, other) {
            (PyType::Sentinel(a), PyType::Sentinel(b)) => a.shallow_eq(b),
            (PyType::Plain(a), PyType::Plain(b)) => a.shallow_eq(b),
            (PyType::Class(a), PyType::Class(b)) => a.shallow_eq(b),
            (PyType::Protocol(a), PyType::Protocol(b)) => a.shallow_eq(b),
            (PyType::TypedDict(a), PyType::TypedDict(b)) => a.shallow_eq(b),
            (PyType::Union(a), PyType::Union(b)) => a.shallow_eq(b),
            (PyType::Callable(a), PyType::Callable(b)) => a.shallow_eq(b),
            (PyType::LazyRef(a), PyType::LazyRef(b)) => a.shallow_eq(b),
            _ => false,
        }
    }
}

// --- ShallowEqMode ---

pub(crate) trait ShallowEqMode {
    fn eq<'ty, G: ArenaSelector<'ty>>(
        arenas: &TypeArenas<'ty>,
        a: PyTypeKey<'ty, G>,
        b: PyTypeKey<'ty, G>,
    ) -> bool
    where
        G::TypeVar: ShallowEq,
        G::ParamSpec: ShallowEq;

    fn cross_eq<'ty>(
        arenas: &TypeArenas<'ty>,
        concrete: PyTypeConcreteKey<'ty>,
        parametric: PyTypeParametricKey<'ty>,
    ) -> bool;
}

impl ShallowEqMode for UnqualifiedMode {
    fn eq<'ty, G: ArenaSelector<'ty>>(
        arenas: &TypeArenas<'ty>,
        a: PyTypeKey<'ty, G>,
        b: PyTypeKey<'ty, G>,
    ) -> bool
    where
        G::TypeVar: ShallowEq,
        G::ParamSpec: ShallowEq,
    {
        let va = arenas.get_as::<Self, G>(a);
        let vb = arenas.get_as::<Self, G>(b);
        va.shallow_eq(&vb)
    }

    fn cross_eq<'ty>(
        arenas: &TypeArenas<'ty>,
        concrete: PyTypeConcreteKey<'ty>,
        parametric: PyTypeParametricKey<'ty>,
    ) -> bool {
        let vc = arenas.get_as::<Self, Concrete>(concrete);
        let vp = arenas.get_as::<Self, Parametric>(parametric);
        vc.shallow_eq(&vp)
    }
}

impl ShallowEqMode for QualifiedMode {
    fn eq<'ty, G: ArenaSelector<'ty>>(
        arenas: &TypeArenas<'ty>,
        a: PyTypeKey<'ty, G>,
        b: PyTypeKey<'ty, G>,
    ) -> bool
    where
        G::TypeVar: ShallowEq,
        G::ParamSpec: ShallowEq,
    {
        let va = arenas.get_as::<Self, G>(a);
        let vb = arenas.get_as::<Self, G>(b);
        va.shallow_eq(&vb)
    }

    fn cross_eq<'ty>(
        arenas: &TypeArenas<'ty>,
        concrete: PyTypeConcreteKey<'ty>,
        parametric: PyTypeParametricKey<'ty>,
    ) -> bool {
        let vc = arenas.get_as::<Self, Concrete>(concrete);
        let vp = arenas.get_as::<Self, Parametric>(parametric);
        vc.shallow_eq(&vp)
    }
}
