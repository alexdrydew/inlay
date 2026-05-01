use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::Deref,
    sync::Arc,
};

use derive_where::derive_where;
use indexmap::IndexMap;

use super::{KeyOf, Qualified, Wrapper};

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
    ContextManager,
    AsyncContextManager,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub(crate) enum MemberAccessKind {
    Attribute,
    DictItem,
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
pub struct PlainType<I: Wrapper, G: TypeVarSupport> {
    pub(crate) descriptor: PyTypeDescriptor,
    pub(crate) args: Vec<PyType<I, I, G>>,
}

#[derive_where(Clone, Hash, PartialEq, Eq, PartialOrd, Ord; PyType<I, I, G>)]
pub struct ProtocolType<I: Wrapper, G: TypeVarSupport> {
    pub(crate) descriptor: PyTypeDescriptor,
    pub(crate) methods: BTreeMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) attributes: BTreeMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) properties: BTreeMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) type_params: Vec<PyType<I, I, G>>,
}

#[derive_where(Clone, Hash, PartialEq, Eq, PartialOrd, Ord; PyType<I, I, G>)]
pub struct TypedDictType<I: Wrapper, G: TypeVarSupport> {
    pub(crate) descriptor: PyTypeDescriptor,
    pub(crate) attributes: BTreeMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) type_params: Vec<PyType<I, I, G>>,
}

#[derive_where(Clone, Hash, PartialEq, Eq, PartialOrd, Ord; PyType<I, I, G>)]
pub struct UnionType<I: Wrapper, G: TypeVarSupport> {
    pub(crate) variants: Vec<PyType<I, I, G>>,
}

#[derive_where(Clone; PyType<I, I, G>)]
pub struct CallableType<I: Wrapper, G: TypeVarSupport> {
    pub(crate) params: IndexMap<Arc<str>, PyType<I, I, G>>,
    pub(crate) param_kinds: Vec<ParamKind>,
    pub(crate) param_has_default: Vec<bool>,
    pub(crate) param_context_inject: Vec<bool>,
    pub(crate) accepts_varargs: bool,
    pub(crate) accepts_varkw: bool,
    pub(crate) return_type: PyType<I, I, G>,
    pub(crate) return_wrapper: WrapperKind,
    pub(crate) type_params: Vec<PyType<I, I, G>>,
    pub(crate) function_name: Option<Arc<str>>,
}

#[derive_where(Clone, Hash, PartialEq, Eq, PartialOrd, Ord; PyType<I, I, G>)]
pub struct LazyRefType<I: Wrapper, G: TypeVarSupport> {
    pub(crate) target: PyType<I, I, G>,
}

// --- Hash + PartialEq + Eq for compound types ---

impl<I: Wrapper, G: TypeVarSupport> Hash for CallableType<I, G>
where
    PyType<I, I, G>: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.function_name.hash(state);
        for (k, v) in &self.params {
            k.hash(state);
            v.hash(state);
        }
        self.param_kinds.hash(state);
        self.param_has_default.hash(state);
        self.param_context_inject.hash(state);
        self.accepts_varargs.hash(state);
        self.accepts_varkw.hash(state);
        self.return_type.hash(state);
        self.return_wrapper.hash(state);
        self.type_params.hash(state);
    }
}

impl<I: Wrapper, G: TypeVarSupport> PartialEq for CallableType<I, G>
where
    PyType<I, I, G>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.function_name == other.function_name
            && self.params == other.params
            && self.param_kinds == other.param_kinds
            && self.param_has_default == other.param_has_default
            && self.param_context_inject == other.param_context_inject
            && self.accepts_varargs == other.accepts_varargs
            && self.accepts_varkw == other.accepts_varkw
            && self.return_type == other.return_type
            && self.return_wrapper == other.return_wrapper
            && self.type_params == other.type_params
    }
}

impl<I: Wrapper, G: TypeVarSupport> Eq for CallableType<I, G> where PyType<I, I, G>: Eq {}

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
pub enum PyType<O: Wrapper, I: Wrapper, G: TypeVarSupport> {
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
    type Wrap<T> = T;
}

pub(crate) struct Viewed<'a>(PhantomData<&'a ()>);
impl<'a> Wrapper for Viewed<'a> {
    type Wrap<T> = ViewRef<'a, T>;
}

pub struct Keyed<'arena>(PhantomData<&'arena ()>);
impl<'arena> Wrapper for Keyed<'arena> {
    type Wrap<T> = KeyOf<'arena, T>;
}

#[derive(Clone, Copy)]
pub(crate) struct ViewRef<'a, T> {
    value: *const T,
    _marker: PhantomData<&'a ()>,
}

impl<'a, T> ViewRef<'a, T> {
    pub(crate) fn new(value: &'a T) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }
}

impl<T> Deref for ViewRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // The arena borrow that created ViewRef guarantees this pointer remains valid.
        unsafe { &*self.value }
    }
}

pub struct Qual<W>(PhantomData<W>);
impl<W: Wrapper> Wrapper for Qual<W> {
    type Wrap<T> = W::Wrap<Qualified<T>>;
}

// --- Type aliases ---
pub(crate) type PyTypeKey<'arena, G> = PyType<Qual<Keyed<'arena>>, Qual<Keyed<'arena>>, G>;
pub(crate) type PyTypeConcreteKey<'arena> = PyTypeKey<'arena, Concrete>;
pub(crate) type PyTypeParametricKey<'arena> = PyTypeKey<'arena, Parametric>;

// --- Key type aliases ---
pub(crate) type ProtocolKey<'arena, G> =
    KeyOf<'arena, Qualified<ProtocolType<Qual<Keyed<'arena>>, G>>>;
pub(crate) type TypedDictKey<'arena, G> =
    KeyOf<'arena, Qualified<TypedDictType<Qual<Keyed<'arena>>, G>>>;
pub(crate) type CallableKey<'arena, G> =
    KeyOf<'arena, Qualified<CallableType<Qual<Keyed<'arena>>, G>>>;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indexmap::IndexMap;

    use super::*;
    use crate::qualifier::Qualifier;
    use crate::types::TypeArenas;

    fn callable_with<'arena>(
        value_type: PyTypeParametricKey<'arena>,
        param_kind: ParamKind,
        has_default: bool,
    ) -> Qualified<CallableType<Qual<Keyed<'arena>>, Parametric>> {
        let mut params = IndexMap::new();
        params.insert(Arc::from("value"), value_type);
        Qualified {
            inner: CallableType {
                params,
                param_kinds: vec![param_kind],
                param_has_default: vec![has_default],
                param_context_inject: vec![false],
                accepts_varargs: false,
                accepts_varkw: false,
                return_type: value_type,
                return_wrapper: WrapperKind::None,
                type_params: Vec::new(),
                function_name: Some(Arc::from("call")),
            },
            qualifier: Qualifier::any(),
        }
    }

    fn sentinel_type<'arena>(arenas: &mut TypeArenas<'arena>) -> PyTypeParametricKey<'arena> {
        PyType::Sentinel(arenas.sentinels.insert(Qualified {
            inner: SentinelType {
                value: SentinelTypeKind::None,
            },
            qualifier: Qualifier::any(),
        }))
    }

    #[test]
    fn callable_arena_keeps_param_kinds_distinct() {
        let mut arenas = TypeArenas::default();
        let value_type = sentinel_type(&mut arenas);

        let positional = arenas.parametric.callables.insert(callable_with(
            value_type,
            ParamKind::PositionalOnly,
            false,
        ));
        let keyword = arenas.parametric.callables.insert(callable_with(
            value_type,
            ParamKind::KeywordOnly,
            false,
        ));

        assert_ne!(positional, keyword);
    }

    #[test]
    fn callable_arena_keeps_param_defaults_distinct() {
        let mut arenas = TypeArenas::default();
        let value_type = sentinel_type(&mut arenas);

        let required = arenas.parametric.callables.insert(callable_with(
            value_type,
            ParamKind::PositionalOrKeyword,
            false,
        ));
        let defaulted = arenas.parametric.callables.insert(callable_with(
            value_type,
            ParamKind::PositionalOrKeyword,
            true,
        ));

        assert_ne!(required, defaulted);
    }
}
