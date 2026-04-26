use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use std::convert::Infallible;

use derive_where::derive_where;
use rustc_hash::FxHasher;

use super::{
    ArenaFamily, ArenaSelector, CallableType, Concrete, LazyRefType, OpaqueParamSpec,
    OpaqueTypeVar, ParamSpecType, Parametric, PlainType, ProtocolType, PyType, PyTypeConcreteKey,
    PyTypeKey, PyTypeParametricKey, Qualified, QualifiedMode, SentinelType, TypeArenas,
    TypeVarSupport, TypeVarType, TypedDictType, UnionType, UnqualifiedMode, Wrapper,
};

// --- ShallowHashMode ---

pub(crate) trait ShallowHashMode<S: ArenaFamily> {
    fn compute<G: ArenaSelector>(
        arenas: &TypeArenas<S>,
        key: PyTypeKey<S, G>,
        state: &mut impl Hasher,
    ) where
        G::TypeVar: ShallowHash,
        G::ParamSpec: ShallowHash;
}

impl<S: ArenaFamily> ShallowHashMode<S> for UnqualifiedMode {
    fn compute<G: ArenaSelector>(
        arenas: &TypeArenas<S>,
        key: PyTypeKey<S, G>,
        state: &mut impl Hasher,
    ) where
        G::TypeVar: ShallowHash,
        G::ParamSpec: ShallowHash,
    {
        arenas
            .get_as::<Self, G>(key)
            .expect("dangling key")
            .shallow_hash(state);
    }
}

impl<S: ArenaFamily> ShallowHashMode<S> for QualifiedMode {
    fn compute<G: ArenaSelector>(
        arenas: &TypeArenas<S>,
        key: PyTypeKey<S, G>,
        state: &mut impl Hasher,
    ) where
        G::TypeVar: ShallowHash,
        G::ParamSpec: ShallowHash,
    {
        arenas
            .get_as::<Self, G>(key)
            .expect("dangling key")
            .shallow_hash(state);
    }
}

// --- Trait ---

pub(crate) trait ShallowHash {
    fn shallow_hash(&self, state: &mut impl Hasher);
}

impl<T: ShallowHash + ?Sized> ShallowHash for &T {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        (**self).shallow_hash(state);
    }
}

impl<T: ShallowHash> ShallowHash for Qualified<T> {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.qualifier.hash(state);
        self.inner.shallow_hash(state);
    }
}

// --- Hash value newtype ---

#[derive(Debug)]
#[derive_where(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ShallowHashValue<M>(u64, PhantomData<M>);

impl<M> ShallowHashValue<M> {
    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

// --- Per-type impls ---

impl ShallowHash for Infallible {
    fn shallow_hash(&self, _state: &mut impl Hasher) {
        match *self {}
    }
}

impl ShallowHash for SentinelType {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.value.hash(state);
    }
}

impl ShallowHash for TypeVarType {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.descriptor.hash(state);
    }
}

impl ShallowHash for ParamSpecType {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.descriptor.hash(state);
    }
}

impl ShallowHash for OpaqueTypeVar {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.descriptor.hash(state);
    }
}

impl ShallowHash for OpaqueParamSpec {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.descriptor.hash(state);
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> ShallowHash for PlainType<I, G> {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.descriptor.hash(state);
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> ShallowHash for ProtocolType<I, G> {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.descriptor.hash(state);
        for key in self.methods.keys() {
            key.hash(state);
        }
        for key in self.attributes.keys() {
            key.hash(state);
        }
        for key in self.properties.keys() {
            key.hash(state);
        }
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> ShallowHash for TypedDictType<I, G> {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.descriptor.hash(state);
        for key in self.attributes.keys() {
            key.hash(state);
        }
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> ShallowHash for UnionType<I, G> {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        self.variants.len().hash(state);
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> ShallowHash for CallableType<I, G> {
    fn shallow_hash(&self, state: &mut impl Hasher) {
        for key in self.params.keys() {
            key.hash(state);
        }
        self.return_wrapper.hash(state);
        self.type_params.len().hash(state);
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> ShallowHash for LazyRefType<I, G> {
    fn shallow_hash(&self, _state: &mut impl Hasher) {}
}

// --- ShallowHash for PyType view (generic over outer wrapper O) ---

impl<O: Wrapper, I: Wrapper + 'static, G: TypeVarSupport> ShallowHash for PyType<O, I, G>
where
    O::Wrap<SentinelType>: ShallowHash,
    O::Wrap<G::TypeVar>: ShallowHash,
    O::Wrap<G::ParamSpec>: ShallowHash,
    O::Wrap<PlainType<I, G>>: ShallowHash,
    O::Wrap<ProtocolType<I, G>>: ShallowHash,
    O::Wrap<TypedDictType<I, G>>: ShallowHash,
    O::Wrap<UnionType<I, G>>: ShallowHash,
    O::Wrap<CallableType<I, G>>: ShallowHash,
    O::Wrap<LazyRefType<I, G>>: ShallowHash,
{
    fn shallow_hash(&self, state: &mut impl Hasher) {
        std::mem::discriminant(self).hash(state);
        match self {
            PyType::Sentinel(v) => v.shallow_hash(state),
            PyType::ParamSpec(v) => v.shallow_hash(state),
            PyType::Plain(v) => v.shallow_hash(state),
            PyType::Protocol(v) => v.shallow_hash(state),
            PyType::TypedDict(v) => v.shallow_hash(state),
            PyType::Union(v) => v.shallow_hash(state),
            PyType::Callable(v) => v.shallow_hash(state),
            PyType::LazyRef(v) => v.shallow_hash(state),
            PyType::TypeVar(v) => v.shallow_hash(state),
        }
    }
}

// --- TypeArenas method ---

impl<S: ArenaFamily> TypeArenas<S> {
    pub(crate) fn shallow_hash_of<M: ShallowHashMode<S>, G: ArenaSelector>(
        &self,
        key: PyTypeKey<S, G>,
    ) -> ShallowHashValue<M>
    where
        G::TypeVar: ShallowHash,
        G::ParamSpec: ShallowHash,
    {
        let mut hasher = FxHasher::default();
        M::compute(self, key, &mut hasher);
        ShallowHashValue(hasher.finish(), PhantomData)
    }

    pub(crate) fn shallow_hash_concrete<M: ShallowHashMode<S>>(
        &self,
        key: PyTypeConcreteKey<S>,
    ) -> ShallowHashValue<M> {
        self.shallow_hash_of::<M, Concrete>(key)
    }

    pub(crate) fn shallow_hash_parametric<M: ShallowHashMode<S>>(
        &self,
        key: PyTypeParametricKey<S>,
    ) -> ShallowHashValue<M> {
        self.shallow_hash_of::<M, Parametric>(key)
    }
}
