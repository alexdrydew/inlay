use rustc_hash::FxHashSet as HashSet;

use super::{
    ArenaSelector, CallableType, Concrete, Keyed, LazyRefType, PlainType, ProtocolType, PyType,
    PyTypeConcreteKey, PyTypeKey, Qual, QualifiedMode, SentinelType, ShallowEq, TypeArenas,
    TypeChildren, TypedDictType, UnionType, UnqualifiedMode, Wrapper,
};

// --- DeepEqMode trait ---

pub(crate) trait DeepEqMode<G: ArenaSelector> {
    fn resolve_and_eq(
        a: PyTypeKey<G>,
        b: PyTypeKey<G>,
        arenas: &TypeArenas,
        visited: &mut HashSet<(PyTypeKey<G>, PyTypeKey<G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<G>>;
}

// --- Recursion skeleton ---

fn deep_eq_impl<M: DeepEqMode<G>, G: ArenaSelector>(
    a: PyTypeKey<G>,
    b: PyTypeKey<G>,
    arenas: &TypeArenas,
    visited: &mut HashSet<(PyTypeKey<G>, PyTypeKey<G>)>,
) -> bool
where
    G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<G>>,
    G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<G>>,
{
    if !visited.insert((a, b)) {
        return true;
    }
    let result = M::resolve_and_eq(a, b, arenas, visited);
    visited.remove(&(a, b));
    result
}

// --- One-level helper ---

fn eq_and_recurse<V, M: DeepEqMode<G>, G: ArenaSelector>(
    a: V,
    b: V,
    arenas: &TypeArenas,
    visited: &mut HashSet<(PyTypeKey<G>, PyTypeKey<G>)>,
) -> bool
where
    V: ShallowEq + TypeChildren<PyTypeKey<G>>,
    G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<G>>,
    G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<G>>,
{
    a.shallow_eq(&b)
        && a.children().count() == b.children().count()
        && a.children()
            .zip(b.children())
            .all(|(&ca, &cb)| deep_eq_impl::<M, G>(ca, cb, arenas, visited))
}

// --- Generic dispatch (single 9+1-arm match, generic over O and G) ---

impl<O: Wrapper, G: ArenaSelector> PyType<O, Qual<Keyed>, G> {
    fn dispatch_deep_eq<M: DeepEqMode<G>>(
        self,
        other: Self,
        arenas: &TypeArenas,
        visited: &mut HashSet<(PyTypeKey<G>, PyTypeKey<G>)>,
    ) -> bool
    where
        O::Wrap<SentinelType>: ShallowEq + TypeChildren<PyTypeKey<G>>,
        O::Wrap<G::TypeVar>: ShallowEq + TypeChildren<PyTypeKey<G>>,
        O::Wrap<G::ParamSpec>: ShallowEq + TypeChildren<PyTypeKey<G>>,
        O::Wrap<PlainType<Qual<Keyed>, G>>: ShallowEq + TypeChildren<PyTypeKey<G>>,
        O::Wrap<ProtocolType<Qual<Keyed>, G>>: ShallowEq + TypeChildren<PyTypeKey<G>>,
        O::Wrap<TypedDictType<Qual<Keyed>, G>>: ShallowEq + TypeChildren<PyTypeKey<G>>,
        O::Wrap<UnionType<Qual<Keyed>, G>>: ShallowEq + TypeChildren<PyTypeKey<G>>,
        O::Wrap<CallableType<Qual<Keyed>, G>>: ShallowEq + TypeChildren<PyTypeKey<G>>,
        O::Wrap<LazyRefType<Qual<Keyed>, G>>: ShallowEq + TypeChildren<PyTypeKey<G>>,
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<G>>,
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

impl<G: ArenaSelector> DeepEqMode<G> for UnqualifiedMode {
    fn resolve_and_eq(
        a: PyTypeKey<G>,
        b: PyTypeKey<G>,
        arenas: &TypeArenas,
        visited: &mut HashSet<(PyTypeKey<G>, PyTypeKey<G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<G>>,
    {
        let (va, vb) = (
            arenas.get_as::<Self, G>(a).expect("dangling key"),
            arenas.get_as::<Self, G>(b).expect("dangling key"),
        );
        va.dispatch_deep_eq::<Self>(vb, arenas, visited)
    }
}

impl<G: ArenaSelector> DeepEqMode<G> for QualifiedMode {
    fn resolve_and_eq(
        a: PyTypeKey<G>,
        b: PyTypeKey<G>,
        arenas: &TypeArenas,
        visited: &mut HashSet<(PyTypeKey<G>, PyTypeKey<G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<G>>,
    {
        let (va, vb) = (
            arenas.get_as::<Self, G>(a).expect("dangling key"),
            arenas.get_as::<Self, G>(b).expect("dangling key"),
        );
        va.dispatch_deep_eq::<Self>(vb, arenas, visited)
    }
}

// --- TypeArenas methods ---

impl TypeArenas {
    pub(crate) fn deep_eq_of<M: DeepEqMode<G>, G: ArenaSelector>(
        &self,
        a: PyTypeKey<G>,
        b: PyTypeKey<G>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<G>>,
    {
        let mut visited = HashSet::default();
        deep_eq_impl::<M, G>(a, b, self, &mut visited)
    }

    pub(crate) fn deep_eq_concrete<M: DeepEqMode<Concrete>>(
        &self,
        a: PyTypeConcreteKey,
        b: PyTypeConcreteKey,
    ) -> bool {
        self.deep_eq_of::<M, Concrete>(a, b)
    }
}
