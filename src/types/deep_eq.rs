use rustc_hash::FxHashSet as HashSet;

use super::{
    ArenaFamily, ArenaSelector, CallableType, Concrete, Keyed, LazyRefType, Parametric, PlainType,
    ProtocolType, PyType, PyTypeConcreteKey, PyTypeKey, PyTypeParametricKey, Qual, QualifiedMode,
    SentinelType, ShallowEq, TypeArenas, TypeChildren, TypedDictType, UnionType, UnqualifiedMode,
    Wrapper,
};

// --- DeepEqMode trait ---

pub(crate) trait DeepEqMode<S: ArenaFamily, G: ArenaSelector> {
    fn resolve_and_eq(
        a: PyTypeKey<S, G>,
        b: PyTypeKey<S, G>,
        arenas: &TypeArenas<S>,
        visited: &mut HashSet<(PyTypeKey<S, G>, PyTypeKey<S, G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<S, G>>;
}

// --- Recursion skeleton ---

fn deep_eq_impl<S: ArenaFamily, M: DeepEqMode<S, G>, G: ArenaSelector>(
    a: PyTypeKey<S, G>,
    b: PyTypeKey<S, G>,
    arenas: &TypeArenas<S>,
    visited: &mut HashSet<(PyTypeKey<S, G>, PyTypeKey<S, G>)>,
) -> bool
where
    G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
    G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
{
    if !visited.insert((a, b)) {
        return true;
    }
    let result = M::resolve_and_eq(a, b, arenas, visited);
    visited.remove(&(a, b));
    result
}

// --- One-level helper ---

fn eq_and_recurse<V, S: ArenaFamily, M: DeepEqMode<S, G>, G: ArenaSelector>(
    a: V,
    b: V,
    arenas: &TypeArenas<S>,
    visited: &mut HashSet<(PyTypeKey<S, G>, PyTypeKey<S, G>)>,
) -> bool
where
    V: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
    G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
    G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
{
    a.shallow_eq(&b)
        && a.children().count() == b.children().count()
        && a.children()
            .zip(b.children())
            .all(|(&ca, &cb)| deep_eq_impl::<S, M, G>(ca, cb, arenas, visited))
}

// --- Generic dispatch (single 9+1-arm match, generic over O and G) ---

impl<O: Wrapper, S: ArenaFamily, G: ArenaSelector> PyType<O, Qual<Keyed<S>>, G> {
    fn dispatch_deep_eq<M: DeepEqMode<S, G>>(
        self,
        other: Self,
        arenas: &TypeArenas<S>,
        visited: &mut HashSet<(PyTypeKey<S, G>, PyTypeKey<S, G>)>,
    ) -> bool
    where
        O::Wrap<SentinelType>: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<G::TypeVar>: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<G::ParamSpec>: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<PlainType<Qual<Keyed<S>>, G>>: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<ProtocolType<Qual<Keyed<S>>, G>>: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<TypedDictType<Qual<Keyed<S>>, G>>: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<UnionType<Qual<Keyed<S>>, G>>: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<CallableType<Qual<Keyed<S>>, G>>: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        O::Wrap<LazyRefType<Qual<Keyed<S>>, G>>: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
    {
        match (self, other) {
            (PyType::Sentinel(a), PyType::Sentinel(b)) => {
                eq_and_recurse::<_, S, M, G>(a, b, arenas, visited)
            }
            (PyType::ParamSpec(a), PyType::ParamSpec(b)) => {
                eq_and_recurse::<_, S, M, G>(a, b, arenas, visited)
            }
            (PyType::Plain(a), PyType::Plain(b)) => {
                eq_and_recurse::<_, S, M, G>(a, b, arenas, visited)
            }
            (PyType::Protocol(a), PyType::Protocol(b)) => {
                eq_and_recurse::<_, S, M, G>(a, b, arenas, visited)
            }
            (PyType::TypedDict(a), PyType::TypedDict(b)) => {
                eq_and_recurse::<_, S, M, G>(a, b, arenas, visited)
            }
            (PyType::Union(a), PyType::Union(b)) => {
                eq_and_recurse::<_, S, M, G>(a, b, arenas, visited)
            }
            (PyType::Callable(a), PyType::Callable(b)) => {
                eq_and_recurse::<_, S, M, G>(a, b, arenas, visited)
            }
            (PyType::LazyRef(a), PyType::LazyRef(b)) => {
                eq_and_recurse::<_, S, M, G>(a, b, arenas, visited)
            }
            (PyType::TypeVar(a), PyType::TypeVar(b)) => {
                eq_and_recurse::<_, S, M, G>(a, b, arenas, visited)
            }
            _ => false,
        }
    }
}

// --- Mode impls ---

impl<S: ArenaFamily, G: ArenaSelector> DeepEqMode<S, G> for UnqualifiedMode {
    fn resolve_and_eq(
        a: PyTypeKey<S, G>,
        b: PyTypeKey<S, G>,
        arenas: &TypeArenas<S>,
        visited: &mut HashSet<(PyTypeKey<S, G>, PyTypeKey<S, G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
    {
        let (va, vb) = (
            arenas.get_as::<Self, G>(a).expect("dangling key"),
            arenas.get_as::<Self, G>(b).expect("dangling key"),
        );
        va.dispatch_deep_eq::<Self>(vb, arenas, visited)
    }
}

impl<S: ArenaFamily, G: ArenaSelector> DeepEqMode<S, G> for QualifiedMode {
    fn resolve_and_eq(
        a: PyTypeKey<S, G>,
        b: PyTypeKey<S, G>,
        arenas: &TypeArenas<S>,
        visited: &mut HashSet<(PyTypeKey<S, G>, PyTypeKey<S, G>)>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
    {
        let (va, vb) = (
            arenas.get_as::<Self, G>(a).expect("dangling key"),
            arenas.get_as::<Self, G>(b).expect("dangling key"),
        );
        va.dispatch_deep_eq::<Self>(vb, arenas, visited)
    }
}

// --- TypeArenas methods ---

impl<S: ArenaFamily> TypeArenas<S> {
    pub(crate) fn deep_eq_of<M: DeepEqMode<S, G>, G: ArenaSelector>(
        &self,
        a: PyTypeKey<S, G>,
        b: PyTypeKey<S, G>,
    ) -> bool
    where
        G::TypeVar: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
        G::ParamSpec: ShallowEq + TypeChildren<PyTypeKey<S, G>>,
    {
        let mut visited = HashSet::default();
        deep_eq_impl::<S, M, G>(a, b, self, &mut visited)
    }

    pub(crate) fn deep_eq_concrete<M: DeepEqMode<S, Concrete>>(
        &self,
        a: PyTypeConcreteKey<S>,
        b: PyTypeConcreteKey<S>,
    ) -> bool {
        self.deep_eq_of::<M, Concrete>(a, b)
    }

    pub(crate) fn deep_eq_parametric<M: DeepEqMode<S, Parametric>>(
        &self,
        a: PyTypeParametricKey<S>,
        b: PyTypeParametricKey<S>,
    ) -> bool {
        self.deep_eq_of::<M, Parametric>(a, b)
    }
}
