use rustc_hash::FxHashSet as HashSet;

use super::{
    ArenaSelector, CallableType, Concrete, Keyed, LazyRefType, PlainType, ProtocolType, PyType,
    PyTypeConcreteKey, PyTypeKey, Qual, QualifiedMode, SentinelType, ShallowEq, TypeArenas,
    TypeChildren, TypedDictType, UnionType, UnqualifiedMode, Wrapper,
};

// --- DeepEqMode trait ---

pub(crate) trait DeepEqMode<'types, G: ArenaSelector<'types>> {
    fn resolve_and_eq(
        a: PyTypeKey<'types, G>,
        b: PyTypeKey<'types, G>,
        arenas: &TypeArenas<'types>,
        visited: &mut HashSet<(PyTypeKey<'types, G>, PyTypeKey<'types, G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'types, G>>;
}

// --- Recursion skeleton ---

fn deep_eq_impl<'types, M: DeepEqMode<'types, G>, G: ArenaSelector<'types>>(
    a: PyTypeKey<'types, G>,
    b: PyTypeKey<'types, G>,
    arenas: &TypeArenas<'types>,
    visited: &mut HashSet<(PyTypeKey<'types, G>, PyTypeKey<'types, G>)>,
) -> bool
where
    G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
    G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
{
    if !visited.insert((a, b)) {
        return true;
    }
    let result = M::resolve_and_eq(a, b, arenas, visited);
    visited.remove(&(a, b));
    result
}

// --- One-level helper ---

fn eq_and_recurse<'types, V, M: DeepEqMode<'types, G>, G: ArenaSelector<'types>>(
    a: V,
    b: V,
    arenas: &TypeArenas<'types>,
    visited: &mut HashSet<(PyTypeKey<'types, G>, PyTypeKey<'types, G>)>,
) -> bool
where
    V: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
    G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
    G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
{
    a.shallow_eq(&b)
        && a.children().count() == b.children().count()
        && a.children()
            .zip(b.children())
            .all(|(&ca, &cb)| deep_eq_impl::<M, G>(ca, cb, arenas, visited))
}

// --- Generic dispatch (single 9+1-arm match, generic over O and G) ---

impl<'types, O: Wrapper, G: ArenaSelector<'types>> PyType<O, Qual<Keyed<'types>>, G> {
    fn dispatch_deep_eq<M: DeepEqMode<'types, G>>(
        self,
        other: Self,
        arenas: &TypeArenas<'types>,
        visited: &mut HashSet<(PyTypeKey<'types, G>, PyTypeKey<'types, G>)>,
    ) -> bool
    where
        O::Wrap<SentinelType>: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<G::TypeVar>: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<G::ParamSpec>: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<PlainType<Qual<Keyed<'types>>, G>>: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<ProtocolType<Qual<Keyed<'types>>, G>>:
            ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<TypedDictType<Qual<Keyed<'types>>, G>>:
            ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<UnionType<Qual<Keyed<'types>>, G>>: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<CallableType<Qual<Keyed<'types>>, G>>:
            ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        O::Wrap<LazyRefType<Qual<Keyed<'types>>, G>>:
            ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
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

impl<'types, G: ArenaSelector<'types>> DeepEqMode<'types, G> for UnqualifiedMode {
    fn resolve_and_eq(
        a: PyTypeKey<'types, G>,
        b: PyTypeKey<'types, G>,
        arenas: &TypeArenas<'types>,
        visited: &mut HashSet<(PyTypeKey<'types, G>, PyTypeKey<'types, G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
    {
        let (va, vb) = (arenas.get_as::<Self, G>(a), arenas.get_as::<Self, G>(b));
        va.dispatch_deep_eq::<Self>(vb, arenas, visited)
    }
}

impl<'types, G: ArenaSelector<'types>> DeepEqMode<'types, G> for QualifiedMode {
    fn resolve_and_eq(
        a: PyTypeKey<'types, G>,
        b: PyTypeKey<'types, G>,
        arenas: &TypeArenas<'types>,
        visited: &mut HashSet<(PyTypeKey<'types, G>, PyTypeKey<'types, G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
    {
        let (va, vb) = (arenas.get_as::<Self, G>(a), arenas.get_as::<Self, G>(b));
        va.dispatch_deep_eq::<Self>(vb, arenas, visited)
    }
}

// --- TypeArenas methods ---

impl<'types> TypeArenas<'types> {
    pub(crate) fn deep_eq_of<M: DeepEqMode<'types, G>, G: ArenaSelector<'types>>(
        &self,
        a: PyTypeKey<'types, G>,
        b: PyTypeKey<'types, G>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<'types, G>>,
    {
        let mut visited = HashSet::default();
        deep_eq_impl::<M, G>(a, b, self, &mut visited)
    }

    pub(crate) fn deep_eq_concrete<M: DeepEqMode<'types, Concrete>>(
        &self,
        a: PyTypeConcreteKey<'types>,
        b: PyTypeConcreteKey<'types>,
    ) -> bool {
        self.deep_eq_of::<M, Concrete>(a, b)
    }
}
