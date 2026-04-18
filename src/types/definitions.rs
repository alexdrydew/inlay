use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
    marker::PhantomData,
    sync::Arc,
};

use derive_where::derive_where;
use indexmap::IndexMap;

use super::{ArenaFamily, KeyOf, Qualified, Wrapper};

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct PyTypeId(Arc<str>);

impl PyTypeId {
    pub fn new(s: String) -> Self {
        Self(Arc::from(s))
    }
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PyTypeDescriptor {
    pub(crate) id: PyTypeId,
    pub(crate) display_name: Arc<str>,
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TypeVarDescriptor {
    pub(crate) id: PyTypeId,
    pub(crate) display_name: Arc<str>,
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum SentinelTypeKind {
    None,
    Ellipsis,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub(crate) enum WrapperKind {
    None,
    Awaitable,
    Cm,
    Acm,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub(crate) enum ParamKind {
    PositionalOnly,
    PositionalOrKeyword,
    KeywordOnly,
}

// --- Leaf types (shared across concrete/parametric) ---

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct SentinelType {
    pub(crate) value: SentinelTypeKind,
}

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TypeVarType {
    pub(crate) descriptor: TypeVarDescriptor,
}

#[derive(Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct ParamSpecType {
    pub(crate) descriptor: TypeVarDescriptor,
}

/// Placeholder for a TypeVar that has no binding during monomorphization.
///
/// Method-scoped TypeVars (e.g. `T` in `def load[T](self, item: T) -> T`)
/// are never bound at the class level. Instead of panicking, they become
/// opaque placeholders in the concrete store. No resolution rule can resolve
/// them, so any attempt to compile a type that requires resolving an
/// `OpaqueTypeVar` will produce a graceful error.
#[derive(Hash, PartialEq, Eq, Clone, PartialOrd, Ord)]
pub struct OpaqueTypeVar {
    pub(crate) descriptor: TypeVarDescriptor,
}

/// Placeholder for a ParamSpec that has no binding during monomorphization.
///
/// Same rationale as `OpaqueTypeVar`: method-scoped ParamSpecs (e.g. `**P`
/// in `def execute[**P, R](self, *args: P.args, **kwargs: P.kwargs) -> R`)
/// cannot be resolved at compile time and become opaque placeholders.
#[derive(Hash, PartialEq, Eq, Clone, PartialOrd, Ord)]
pub struct OpaqueParamSpec {
    pub(crate) descriptor: TypeVarDescriptor,
}

// --- TypeVarSupport variants ---
// Concrete types use OpaqueTypeVar for unresolvable method-scoped TypeVars.
// Parametric types MAY contain TypeVar/ParamSpec nodes but don't have to.

pub(crate) trait TypeVarSupport: 'static {
    type TypeVar: 'static;
    type ParamSpec: 'static;
}

pub(crate) struct Concrete;
impl TypeVarSupport for Concrete {
    type TypeVar = OpaqueTypeVar;
    type ParamSpec = OpaqueParamSpec;
}

pub(crate) struct Parametric;
impl TypeVarSupport for Parametric {
    type TypeVar = TypeVarType;
    type ParamSpec = ParamSpecType;
}

// --- Parameterized types ---

#[derive_where(Clone, Hash, PartialEq, Eq, PartialOrd, Ord; PyType<I, I, G>)]
pub struct PlainType<I: Wrapper + 'static, G: TypeVarSupport> {
    pub(crate) descriptor: PyTypeDescriptor,
    pub(crate) args: Vec<PyType<I, I, G>>,
}

#[derive_where(Clone, Hash, PartialEq, Eq, PartialOrd, Ord; PyType<I, I, G>)]
pub struct ProtocolType<I: Wrapper + 'static, G: TypeVarSupport> {
    pub(crate) descriptor: PyTypeDescriptor,
    pub(crate) methods: BTreeMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) attributes: BTreeMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) properties: BTreeMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) type_params: Vec<PyType<I, I, G>>,
}

#[derive_where(Clone, Hash, PartialEq, Eq, PartialOrd, Ord; PyType<I, I, G>)]
pub struct TypedDictType<I: Wrapper + 'static, G: TypeVarSupport> {
    pub(crate) descriptor: PyTypeDescriptor,
    pub(crate) attributes: BTreeMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) type_params: Vec<PyType<I, I, G>>,
}

#[derive_where(Clone, Hash, PartialEq, Eq, PartialOrd, Ord; PyType<I, I, G>)]
pub struct UnionType<I: Wrapper + 'static, G: TypeVarSupport> {
    pub(crate) variants: Vec<PyType<I, I, G>>,
}

#[derive_where(Clone; PyType<I, I, G>)]
pub struct CallableType<I: Wrapper + 'static, G: TypeVarSupport> {
    pub(crate) params: IndexMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) param_kinds: Vec<ParamKind>,
    pub(crate) param_has_default: Vec<bool>,
    pub(crate) return_type: PyType<I, I, G>,
    pub(crate) return_wrapper: WrapperKind,
    pub(crate) type_params: Vec<PyType<I, I, G>>,
    pub(crate) function_name: Option<Arc<str>>,
}

#[derive_where(Clone, Hash, PartialEq, Eq, PartialOrd, Ord; PyType<I, I, G>)]
pub struct LazyRefType<I: Wrapper + 'static, G: TypeVarSupport> {
    pub(crate) target: PyType<I, I, G>,
}

// --- Hash + PartialEq + Eq for compound types ---

impl<I: Wrapper + 'static, G: TypeVarSupport> Hash for CallableType<I, G>
where
    PyType<I, I, G>: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.function_name.hash(state);
        for (k, v) in &self.params {
            k.hash(state);
            v.hash(state);
        }
        self.return_type.hash(state);
        self.return_wrapper.hash(state);
        self.type_params.hash(state);
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> PartialEq for CallableType<I, G>
where
    PyType<I, I, G>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.function_name == other.function_name
            && self.params == other.params
            && self.return_type == other.return_type
            && self.return_wrapper == other.return_wrapper
            && self.type_params == other.type_params
    }
}

impl<I: Wrapper + 'static, G: TypeVarSupport> Eq for CallableType<I, G> where PyType<I, I, G>: Eq {}

// --- PyType enum ---

#[derive_where(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash;
    <O as Wrapper>::Wrap<SentinelType>,
    <O as Wrapper>::Wrap<G::ParamSpec>,
    <O as Wrapper>::Wrap<PlainType<I, G>>,
    <O as Wrapper>::Wrap<ProtocolType<I, G>>,
    <O as Wrapper>::Wrap<TypedDictType<I, G>>,
    <O as Wrapper>::Wrap<UnionType<I, G>>,
    <O as Wrapper>::Wrap<CallableType<I, G>>,
    <O as Wrapper>::Wrap<LazyRefType<I, G>>,
    <O as Wrapper>::Wrap<G::TypeVar>
)]
pub enum PyType<O: Wrapper, I: Wrapper + 'static, G: TypeVarSupport> {
    Sentinel(O::Wrap<SentinelType>),
    ParamSpec(O::Wrap<G::ParamSpec>),
    Plain(O::Wrap<PlainType<I, G>>),
    Protocol(O::Wrap<ProtocolType<I, G>>),
    TypedDict(O::Wrap<TypedDictType<I, G>>),
    Union(O::Wrap<UnionType<I, G>>),
    Callable(O::Wrap<CallableType<I, G>>),
    LazyRef(O::Wrap<LazyRefType<I, G>>),
    TypeVar(O::Wrap<G::TypeVar>),
}

// --- Wrapper impls ---

pub struct Owned;
impl Wrapper for Owned {
    type Wrap<T: 'static> = T;
}

pub(crate) struct Viewed<'a>(PhantomData<&'a ()>);
impl<'a> Wrapper for Viewed<'a> {
    type Wrap<T: 'static> = &'a T;
}

pub struct Keyed<S: ArenaFamily>(PhantomData<S>);
impl<S: ArenaFamily> Wrapper for Keyed<S> {
    type Wrap<T: 'static> = KeyOf<S, T>;
}

pub struct Qual<W>(PhantomData<W>);
impl<W: Wrapper> Wrapper for Qual<W> {
    type Wrap<T: 'static> = W::Wrap<Qualified<T>>;
}

// --- Type aliases ---
pub(crate) type PyTypeKey<S, G> = PyType<Qual<Keyed<S>>, Qual<Keyed<S>>, G>;
pub(crate) type PyTypeConcreteKey<S> = PyTypeKey<S, Concrete>;
pub(crate) type PyTypeParametricKey<S> = PyTypeKey<S, Parametric>;
pub(crate) type PyTypeRef<'a, S, G> = PyType<Qual<Viewed<'a>>, Qual<Keyed<S>>, G>;

// --- Key type aliases ---
pub(crate) type PlainKey<S, G> = KeyOf<S, Qualified<PlainType<Qual<Keyed<S>>, G>>>;
pub(crate) type ProtocolKey<S, G> = KeyOf<S, Qualified<ProtocolType<Qual<Keyed<S>>, G>>>;
pub(crate) type TypedDictKey<S, G> = KeyOf<S, Qualified<TypedDictType<Qual<Keyed<S>>, G>>>;
pub(crate) type UnionKey<S, G> = KeyOf<S, Qualified<UnionType<Qual<Keyed<S>>, G>>>;
pub(crate) type CallableKey<S, G> = KeyOf<S, Qualified<CallableType<Qual<Keyed<S>>, G>>>;
pub(crate) type LazyRefKey<S, G> = KeyOf<S, Qualified<LazyRefType<Qual<Keyed<S>>, G>>>;
pub(crate) type TypeVarKey<S> = KeyOf<S, Qualified<TypeVarType>>;
pub(crate) type ParamSpecKey<S> = KeyOf<S, Qualified<ParamSpecType>>;
