use rustc_hash::FxHashSet as HashSet;

use super::{
    ArenaSelector, CallableType, ClassType, Concrete, Keyed, LazyRefType, PlainType, ProtocolType,
    PyType, PyTypeConcreteKey, PyTypeKey, Qual, QualifiedMode, SentinelType, ShallowEq, TypeArenas,
    TypeChildren, TypedDictType, UnionType, UnqualifiedMode, Wrapper,
};

// --- DeepEqMode trait ---

pub(crate) trait DeepEqMode<'ty, G: ArenaSelector<'ty>> {
    fn resolve_and_eq(
        a: PyTypeKey<'ty, G>,
        b: PyTypeKey<'ty, G>,
        arenas: &TypeArenas<'ty>,
        visited: &mut HashSet<(PyTypeKey<'ty, G>, PyTypeKey<'ty, G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>;
}

// --- Recursion skeleton ---

fn deep_eq_impl<'ty, M: DeepEqMode<'ty, G>, G: ArenaSelector<'ty>>(
    a: PyTypeKey<'ty, G>,
    b: PyTypeKey<'ty, G>,
    arenas: &TypeArenas<'ty>,
    visited: &mut HashSet<(PyTypeKey<'ty, G>, PyTypeKey<'ty, G>)>,
) -> bool
where
    G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
    G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
{
    if a == b {
        return true;
    }
    if !visited.insert((a, b)) {
        return true;
    }
    let result = M::resolve_and_eq(a, b, arenas, visited);
    visited.remove(&(a, b));
    result
}

// --- One-level helper ---

fn eq_and_recurse<'ty, V, M: DeepEqMode<'ty, G>, G: ArenaSelector<'ty>>(
    a: V,
    b: V,
    arenas: &TypeArenas<'ty>,
    visited: &mut HashSet<(PyTypeKey<'ty, G>, PyTypeKey<'ty, G>)>,
) -> bool
where
    V: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
    G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
    G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
{
    a.shallow_eq(&b)
        && a.children().count() == b.children().count()
        && a.children()
            .zip(b.children())
            .all(|(&ca, &cb)| deep_eq_impl::<M, G>(ca, cb, arenas, visited))
}

// --- Generic dispatch (single 9+1-arm match, generic over O and G) ---

impl<'ty, O: Wrapper, G: ArenaSelector<'ty>> PyType<O, Qual<Keyed<'ty>>, G> {
    fn dispatch_deep_eq<M: DeepEqMode<'ty, G>>(
        self,
        other: Self,
        arenas: &TypeArenas<'ty>,
        visited: &mut HashSet<(PyTypeKey<'ty, G>, PyTypeKey<'ty, G>)>,
    ) -> bool
    where
        O::Wrap<SentinelType>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<G::TypeVar>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<G::ParamSpec>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<PlainType<Qual<Keyed<'ty>>, G>>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<ClassType<Qual<Keyed<'ty>>, G>>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<ProtocolType<Qual<Keyed<'ty>>, G>>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<TypedDictType<Qual<Keyed<'ty>>, G>>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<UnionType<Qual<Keyed<'ty>>, G>>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<CallableType<Qual<Keyed<'ty>>, G>>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<LazyRefType<Qual<Keyed<'ty>>, G>>: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
    {
        match (self, other) {
            (PyType::Sentinel(a), PyType::Sentinel(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            (PyType::ParamSpec(a), PyType::ParamSpec(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            (PyType::Plain(a), PyType::Plain(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            (PyType::Class(a), PyType::Class(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            (PyType::Protocol(a), PyType::Protocol(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            (PyType::TypedDict(a), PyType::TypedDict(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            (PyType::Union(a), PyType::Union(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            (PyType::Callable(a), PyType::Callable(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            (PyType::LazyRef(a), PyType::LazyRef(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            (PyType::TypeVar(a), PyType::TypeVar(b)) => {
                eq_and_recurse::<_, M, G>(a, b, arenas, visited)
            }
            _ => false,
        }
    }
}

// --- Mode impls ---

impl<'ty, G: ArenaSelector<'ty>> DeepEqMode<'ty, G> for UnqualifiedMode {
    fn resolve_and_eq(
        a: PyTypeKey<'ty, G>,
        b: PyTypeKey<'ty, G>,
        arenas: &TypeArenas<'ty>,
        visited: &mut HashSet<(PyTypeKey<'ty, G>, PyTypeKey<'ty, G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
    {
        let (va, vb) = (arenas.get_as::<Self, G>(a), arenas.get_as::<Self, G>(b));
        va.dispatch_deep_eq::<Self>(vb, arenas, visited)
    }
}

impl<'ty, G: ArenaSelector<'ty>> DeepEqMode<'ty, G> for QualifiedMode {
    fn resolve_and_eq(
        a: PyTypeKey<'ty, G>,
        b: PyTypeKey<'ty, G>,
        arenas: &TypeArenas<'ty>,
        visited: &mut HashSet<(PyTypeKey<'ty, G>, PyTypeKey<'ty, G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
    {
        let (va, vb) = (arenas.get_as::<Self, G>(a), arenas.get_as::<Self, G>(b));
        va.dispatch_deep_eq::<Self>(vb, arenas, visited)
    }
}

// --- TypeArenas methods ---

impl<'ty> TypeArenas<'ty> {
    pub(crate) fn deep_eq_of<M: DeepEqMode<'ty, G>, G: ArenaSelector<'ty>>(
        &self,
        a: PyTypeKey<'ty, G>,
        b: PyTypeKey<'ty, G>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
    {
        if a == b {
            return true;
        }
        let mut visited = HashSet::default();
        deep_eq_impl::<M, G>(a, b, self, &mut visited)
    }

    pub(crate) fn deep_eq_value<M: DeepEqMode<'ty, G>, G: ArenaSelector<'ty>, V>(
        &self,
        a: V,
        b: V,
    ) -> bool
    where
        V: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'ty, G>>,
    {
        let mut visited = HashSet::default();
        eq_and_recurse::<_, M, G>(a, b, self, &mut visited)
    }

    pub(crate) fn deep_eq_concrete<M: DeepEqMode<'ty, Concrete>>(
        &self,
        a: PyTypeConcreteKey<'ty>,
        b: PyTypeConcreteKey<'ty>,
    ) -> bool {
        self.deep_eq_of::<M, Concrete>(a, b)
    }
}
