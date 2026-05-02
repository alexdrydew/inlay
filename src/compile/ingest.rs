use std::collections::BTreeMap;
use std::sync::Arc;

use indexmap::IndexMap;
use inlay_instrument::instrumented;
use pyo3::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use pyo3::types::PyType;

use crate::normalized::{self, NormalizedTypeRef};
use crate::python_identity::PythonIdentity;
use crate::types::{
    Arena, ArenaKey, CallableType, Keyed, LazyRefType, ParamKind, ParamSpecType, Parametric,
    PlainType, ProtocolType, PyType as PyTypeEnum, PyTypeDescriptor, PyTypeId, PyTypeParametricKey,
    Qual, Qualified, SentinelType, TypeArenas, TypeVarDescriptor, TypeVarType, TypedDictType,
    UnionType, WrapperKind,
};

fn make_type_descriptor(origin: &Bound<'_, PyAny>) -> PyResult<PyTypeDescriptor> {
    let id = PyTypeId::new(PythonIdentity::from_bound(origin).to_string());
    let display_name: Arc<str> = match origin.cast::<PyType>() {
        Ok(t) => Arc::from(t.qualname()?.to_string()),
        Err(_) => Arc::from(origin.repr()?.to_string()),
    };
    Ok(PyTypeDescriptor { id, display_name })
}

fn make_typevar_descriptor(tv: &Bound<'_, PyAny>) -> PyResult<TypeVarDescriptor> {
    let id = PyTypeId::new(PythonIdentity::from_bound(tv).to_string());
    let display_name: Arc<str> = match tv.getattr("__name__") {
        Ok(n) => Arc::from(n.extract::<String>()?),
        Err(_) => Arc::from(tv.repr()?.to_string()),
    };
    Ok(TypeVarDescriptor { id, display_name })
}

fn ntype_identity(ntype: &NormalizedTypeRef) -> PythonIdentity {
    match ntype {
        NormalizedTypeRef::Plain(p) => PythonIdentity::from_ptr(p.as_ptr()),
        NormalizedTypeRef::Protocol(p) => PythonIdentity::from_ptr(p.as_ptr()),
        NormalizedTypeRef::TypedDict(t) => PythonIdentity::from_ptr(t.as_ptr()),
        NormalizedTypeRef::Union(u) => PythonIdentity::from_ptr(u.as_ptr()),
        NormalizedTypeRef::Callable(c) => PythonIdentity::from_ptr(c.as_ptr()),
        NormalizedTypeRef::LazyRef(l) => PythonIdentity::from_ptr(l.as_ptr()),
        NormalizedTypeRef::Sentinel(s) => PythonIdentity::from_ptr(s.as_ptr()),
        NormalizedTypeRef::TypeVar(t) => PythonIdentity::from_ptr(t.as_ptr()),
        NormalizedTypeRef::ParamSpec(p) => PythonIdentity::from_ptr(p.as_ptr()),
        NormalizedTypeRef::CyclePlaceholder(c) => PythonIdentity::from_ptr(c.as_ptr()),
    }
}

type ParametricPlain<'arena> = Qualified<PlainType<Qual<Keyed<'arena>>, Parametric>>;
type ParametricProtocol<'arena> = Qualified<ProtocolType<Qual<Keyed<'arena>>, Parametric>>;
type ParametricTypedDict<'arena> = Qualified<TypedDictType<Qual<Keyed<'arena>>, Parametric>>;
type ParametricUnion<'arena> = Qualified<UnionType<Qual<Keyed<'arena>>, Parametric>>;
type ParametricCallable<'arena> = Qualified<CallableType<Qual<Keyed<'arena>>, Parametric>>;
type ParametricLazyRef<'arena> = Qualified<LazyRefType<Qual<Keyed<'arena>>, Parametric>>;

type SeenMap<'temp> = HashMap<PythonIdentity, PyTypeParametricKey<'temp>>;

#[derive(Default)]
struct TempParametricArenas<'temp> {
    sentinels: Arena<'temp, Qualified<SentinelType>, Option<Qualified<SentinelType>>>,
    type_vars: Arena<'temp, Qualified<TypeVarType>, Option<Qualified<TypeVarType>>>,
    param_specs: Arena<'temp, Qualified<ParamSpecType>, Option<Qualified<ParamSpecType>>>,
    plains: Arena<'temp, ParametricPlain<'temp>, Option<ParametricPlain<'temp>>>,
    protocols: Arena<'temp, ParametricProtocol<'temp>, Option<ParametricProtocol<'temp>>>,
    typed_dicts: Arena<'temp, ParametricTypedDict<'temp>, Option<ParametricTypedDict<'temp>>>,
    unions: Arena<'temp, ParametricUnion<'temp>, Option<ParametricUnion<'temp>>>,
    callables: Arena<'temp, ParametricCallable<'temp>, Option<ParametricCallable<'temp>>>,
    lazy_refs: Arena<'temp, ParametricLazyRef<'temp>, Option<ParametricLazyRef<'temp>>>,
}

struct TempArenaKeysMappings<'types> {
    sentinels: Vec<ArenaKey<'types, Qualified<SentinelType>>>,
    type_vars: Vec<ArenaKey<'types, Qualified<TypeVarType>>>,
    param_specs: Vec<ArenaKey<'types, Qualified<ParamSpecType>>>,
    plains: Vec<ArenaKey<'types, ParametricPlain<'types>>>,
    protocols: Vec<ArenaKey<'types, ParametricProtocol<'types>>>,
    typed_dicts: Vec<ArenaKey<'types, ParametricTypedDict<'types>>>,
    unions: Vec<ArenaKey<'types, ParametricUnion<'types>>>,
    callables: Vec<ArenaKey<'types, ParametricCallable<'types>>>,
    lazy_refs: Vec<ArenaKey<'types, ParametricLazyRef<'types>>>,
}

fn allocate_keys<'types, T>(store: &Arena<'types, T>, count: usize) -> Vec<ArenaKey<'types, T>> {
    (0..count).map(|offset| store.future_key(offset)).collect()
}

fn remap_temp_key<'types, T, U>(
    key: ArenaKey<'_, T>,
    keys: &[ArenaKey<'types, U>],
) -> ArenaKey<'types, U> {
    keys[key.index()]
}

fn remap_parametric_key<'types>(
    key: PyTypeParametricKey<'_>,
    keys: &TempArenaKeysMappings<'types>,
) -> PyTypeParametricKey<'types> {
    match key {
        PyTypeEnum::Sentinel(key) => PyTypeEnum::Sentinel(remap_temp_key(key, &keys.sentinels)),
        PyTypeEnum::TypeVar(key) => PyTypeEnum::TypeVar(remap_temp_key(key, &keys.type_vars)),
        PyTypeEnum::ParamSpec(key) => PyTypeEnum::ParamSpec(remap_temp_key(key, &keys.param_specs)),
        PyTypeEnum::Plain(key) => PyTypeEnum::Plain(remap_temp_key(key, &keys.plains)),
        PyTypeEnum::Protocol(key) => PyTypeEnum::Protocol(remap_temp_key(key, &keys.protocols)),
        PyTypeEnum::TypedDict(key) => PyTypeEnum::TypedDict(remap_temp_key(key, &keys.typed_dicts)),
        PyTypeEnum::Union(key) => PyTypeEnum::Union(remap_temp_key(key, &keys.unions)),
        PyTypeEnum::Callable(key) => PyTypeEnum::Callable(remap_temp_key(key, &keys.callables)),
        PyTypeEnum::LazyRef(key) => PyTypeEnum::LazyRef(remap_temp_key(key, &keys.lazy_refs)),
    }
}

fn commit_parametric_temp<'types, 'temp>(
    arenas: &mut TypeArenas<'types>,
    temp: TempParametricArenas<'temp>,
    root: PyTypeParametricKey<'temp>,
) -> PyTypeParametricKey<'types> {
    let keys = TempArenaKeysMappings {
        sentinels: allocate_keys(&arenas.sentinels, temp.sentinels.values().len()),
        type_vars: allocate_keys(&arenas.parametric.type_vars, temp.type_vars.values().len()),
        param_specs: allocate_keys(
            &arenas.parametric.param_specs,
            temp.param_specs.values().len(),
        ),
        plains: allocate_keys(&arenas.parametric.plains, temp.plains.values().len()),
        protocols: allocate_keys(&arenas.parametric.protocols, temp.protocols.values().len()),
        typed_dicts: allocate_keys(
            &arenas.parametric.typed_dicts,
            temp.typed_dicts.values().len(),
        ),
        unions: allocate_keys(&arenas.parametric.unions, temp.unions.values().len()),
        callables: allocate_keys(&arenas.parametric.callables, temp.callables.values().len()),
        lazy_refs: allocate_keys(&arenas.parametric.lazy_refs, temp.lazy_refs.values().len()),
    };
    let root = remap_parametric_key(root, &keys);

    let TempParametricArenas {
        sentinels,
        type_vars,
        param_specs,
        plains,
        protocols,
        typed_dicts,
        unions,
        callables,
        lazy_refs,
    } = temp;

    for value in sentinels.into_values() {
        arenas
            .sentinels
            .push_committed(value.expect("parametric temp sentinel should be filled"));
    }
    for value in type_vars.into_values() {
        arenas
            .parametric
            .type_vars
            .push_committed(value.expect("parametric temp type var should be filled"));
    }
    for value in param_specs.into_values() {
        arenas
            .parametric
            .param_specs
            .push_committed(value.expect("parametric temp param spec should be filled"));
    }
    for value in plains.into_values() {
        let value = value.expect("parametric temp plain should be filled");
        arenas.parametric.plains.push_committed(Qualified {
            inner: PlainType {
                descriptor: value.inner.descriptor,
                args: value
                    .inner
                    .args
                    .into_iter()
                    .map(|child| remap_parametric_key(child, &keys))
                    .collect(),
            },
            qualifier: value.qualifier,
        });
    }
    for value in protocols.into_values() {
        let value = value.expect("parametric temp protocol should be filled");
        arenas.parametric.protocols.push_committed(Qualified {
            inner: ProtocolType {
                descriptor: value.inner.descriptor,
                methods: value
                    .inner
                    .methods
                    .into_iter()
                    .map(|(name, child)| (name, remap_parametric_key(child, &keys)))
                    .collect(),
                attributes: value
                    .inner
                    .attributes
                    .into_iter()
                    .map(|(name, child)| (name, remap_parametric_key(child, &keys)))
                    .collect(),
                properties: value
                    .inner
                    .properties
                    .into_iter()
                    .map(|(name, child)| (name, remap_parametric_key(child, &keys)))
                    .collect(),
                type_params: value
                    .inner
                    .type_params
                    .into_iter()
                    .map(|child| remap_parametric_key(child, &keys))
                    .collect(),
            },
            qualifier: value.qualifier,
        });
    }
    for value in typed_dicts.into_values() {
        let value = value.expect("parametric temp typed dict should be filled");
        arenas.parametric.typed_dicts.push_committed(Qualified {
            inner: TypedDictType {
                descriptor: value.inner.descriptor,
                attributes: value
                    .inner
                    .attributes
                    .into_iter()
                    .map(|(name, child)| (name, remap_parametric_key(child, &keys)))
                    .collect(),
                type_params: value
                    .inner
                    .type_params
                    .into_iter()
                    .map(|child| remap_parametric_key(child, &keys))
                    .collect(),
            },
            qualifier: value.qualifier,
        });
    }
    for value in unions.into_values() {
        let value = value.expect("parametric temp union should be filled");
        arenas.parametric.unions.push_committed(Qualified {
            inner: UnionType {
                variants: value
                    .inner
                    .variants
                    .into_iter()
                    .map(|child| remap_parametric_key(child, &keys))
                    .collect(),
            },
            qualifier: value.qualifier,
        });
    }
    for value in callables.into_values() {
        let value = value.expect("parametric temp callable should be filled");
        arenas.parametric.callables.push_committed(Qualified {
            inner: CallableType {
                params: value
                    .inner
                    .params
                    .into_iter()
                    .map(|(name, child)| (name, remap_parametric_key(child, &keys)))
                    .collect(),
                param_kinds: value.inner.param_kinds,
                param_has_default: value.inner.param_has_default,
                param_context_inject: value.inner.param_context_inject,
                accepts_varargs: value.inner.accepts_varargs,
                accepts_varkw: value.inner.accepts_varkw,
                return_type: remap_parametric_key(value.inner.return_type, &keys),
                return_wrapper: value.inner.return_wrapper,
                type_params: value
                    .inner
                    .type_params
                    .into_iter()
                    .map(|child| remap_parametric_key(child, &keys))
                    .collect(),
                function_name: value.inner.function_name,
            },
            qualifier: value.qualifier,
        });
    }
    for value in lazy_refs.into_values() {
        let value = value.expect("parametric temp lazy ref should be filled");
        arenas.parametric.lazy_refs.push_committed(Qualified {
            inner: LazyRefType {
                target: remap_parametric_key(value.inner.target, &keys),
            },
            qualifier: value.qualifier,
        });
    }

    root
}

#[instrumented(
    name = "inlay.ingest_parametric",
    target = "inlay",
    level = "trace",
    skip_all
)]
pub(crate) fn ingest_parametric<'types>(
    arenas: &mut TypeArenas<'types>,
    py: Python<'_>,
    ntype: &NormalizedTypeRef,
) -> PyResult<PyTypeParametricKey<'types>> {
    let mut temp = TempParametricArenas::default();
    let mut seen = SeenMap::default();
    let root = ingest_inner(&mut temp, py, ntype, &mut seen)?;
    Ok(commit_parametric_temp(arenas, temp, root))
}

fn ingest_inner<'temp>(
    arenas: &mut TempParametricArenas<'temp>,
    py: Python<'_>,
    ntype: &NormalizedTypeRef,
    seen: &mut SeenMap<'temp>,
) -> PyResult<PyTypeParametricKey<'temp>> {
    let identity = ntype_identity(ntype);
    if let Some(&key) = seen.get(&identity) {
        return Ok(key);
    }

    match ntype {
        NormalizedTypeRef::Sentinel(s) => {
            let s = s.bind(py).borrow();
            let val = Qualified {
                inner: SentinelType {
                    value: s.kind.clone(),
                },
                qualifier: s.qualifiers.clone(),
            };
            Ok(PyTypeEnum::Sentinel(arenas.sentinels.insert(Some(val))))
        }
        NormalizedTypeRef::TypeVar(t) => {
            let t = t.bind(py).borrow();
            let descriptor = make_typevar_descriptor(t.typevar.bind(py))?;
            let val = Qualified {
                inner: TypeVarType { descriptor },
                qualifier: t.qualifiers.clone(),
            };
            Ok(PyTypeEnum::TypeVar(arenas.type_vars.insert(Some(val))))
        }
        NormalizedTypeRef::ParamSpec(p) => {
            let p = p.bind(py).borrow();
            let descriptor = make_typevar_descriptor(p.paramspec.bind(py))?;
            let val = Qualified {
                inner: ParamSpecType { descriptor },
                qualifier: p.qualifiers.clone(),
            };
            Ok(PyTypeEnum::ParamSpec(arenas.param_specs.insert(Some(val))))
        }
        NormalizedTypeRef::Plain(p) => {
            let placeholder_key = arenas.plains.insert(None);
            let result_key = PyTypeEnum::Plain(placeholder_key);
            seen.insert(identity, result_key);

            let p = p.bind(py).borrow();
            let descriptor = make_type_descriptor(p.origin.bind(py))?;
            let args = p
                .args
                .iter()
                .map(|a| ingest_inner(arenas, py, a, seen))
                .collect::<PyResult<Vec<_>>>()?;
            let val = Qualified {
                inner: PlainType { descriptor, args },
                qualifier: p.qualifiers.clone(),
            };
            assert!(
                arenas
                    .plains
                    .get_mut(placeholder_key)
                    .replace(val)
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::Protocol(p) => {
            let placeholder_key = arenas.protocols.insert(None);
            let result_key = PyTypeEnum::Protocol(placeholder_key);
            seen.insert(identity, result_key);

            let p = p.bind(py).borrow();
            let descriptor = make_type_descriptor(p.origin.bind(py))?;
            let type_params = p
                .type_params
                .iter()
                .map(|tp| ingest_inner(arenas, py, tp, seen))
                .collect::<PyResult<Vec<_>>>()?;
            let methods = ingest_btree_map_tracked(arenas, py, &p.methods, seen)?;
            let attributes = ingest_btree_map_tracked(arenas, py, &p.attributes, seen)?;
            let properties = ingest_btree_map_tracked(arenas, py, &p.properties, seen)?;
            let val = Qualified {
                inner: ProtocolType {
                    descriptor,
                    type_params,
                    methods,
                    attributes,
                    properties,
                },
                qualifier: p.qualifiers.clone(),
            };
            assert!(
                arenas
                    .protocols
                    .get_mut(placeholder_key)
                    .replace(val)
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::TypedDict(t) => {
            let placeholder_key = arenas.typed_dicts.insert(None);
            let result_key = PyTypeEnum::TypedDict(placeholder_key);
            seen.insert(identity, result_key);

            let t = t.bind(py).borrow();
            let descriptor = make_type_descriptor(t.origin.bind(py))?;
            let type_params = t
                .type_params
                .iter()
                .map(|tp| ingest_inner(arenas, py, tp, seen))
                .collect::<PyResult<Vec<_>>>()?;
            let attributes = ingest_btree_map_tracked(arenas, py, &t.attributes, seen)?;
            let val = Qualified {
                inner: TypedDictType {
                    descriptor,
                    type_params,
                    attributes,
                },
                qualifier: t.qualifiers.clone(),
            };
            assert!(
                arenas
                    .typed_dicts
                    .get_mut(placeholder_key)
                    .replace(val)
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::Union(u) => {
            let placeholder_key = arenas.unions.insert(None);
            let result_key = PyTypeEnum::Union(placeholder_key);
            seen.insert(identity, result_key);

            let u = u.bind(py).borrow();
            let variants = u
                .variants
                .iter()
                .map(|v| ingest_inner(arenas, py, v, seen))
                .collect::<PyResult<Vec<_>>>()?;
            let val = Qualified {
                inner: UnionType { variants },
                qualifier: u.qualifiers.clone(),
            };
            assert!(
                arenas
                    .unions
                    .get_mut(placeholder_key)
                    .replace(val)
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::Callable(c) => {
            let placeholder_key = arenas.callables.insert(None);
            let result_key = PyTypeEnum::Callable(placeholder_key);
            seen.insert(identity, result_key);

            let c = c.bind(py).borrow();
            let val = ingest_callable_value(arenas, py, &c, seen)?;
            assert!(
                arenas
                    .callables
                    .get_mut(placeholder_key)
                    .replace(val)
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::LazyRef(l) => {
            let placeholder_key = arenas.lazy_refs.insert(None);
            let result_key = PyTypeEnum::LazyRef(placeholder_key);
            seen.insert(identity, result_key);

            let l = l.bind(py).borrow();
            let target = ingest_inner(arenas, py, &l.target, seen)?;
            let val = Qualified {
                inner: LazyRefType { target },
                qualifier: l.qualifiers.clone(),
            };
            assert!(
                arenas
                    .lazy_refs
                    .get_mut(placeholder_key)
                    .replace(val)
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::CyclePlaceholder(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CyclePlaceholder must be replaced before ingestion",
        )),
    }
}

fn parse_wrapper_kind(s: &str) -> PyResult<WrapperKind> {
    match s {
        "none" => Ok(WrapperKind::None),
        "awaitable" => Ok(WrapperKind::Awaitable),
        "context_manager" => Ok(WrapperKind::ContextManager),
        "async_context_manager" => Ok(WrapperKind::AsyncContextManager),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown return_wrapper: '{other}'"
        ))),
    }
}

pub(crate) fn parse_param_kind(s: &str) -> PyResult<ParamKind> {
    match s {
        "positional_only" => Ok(ParamKind::PositionalOnly),
        "positional_or_keyword" => Ok(ParamKind::PositionalOrKeyword),
        "keyword_only" => Ok(ParamKind::KeywordOnly),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown param_kind: '{other}'"
        ))),
    }
}

fn ingest_callable_value<'temp>(
    arenas: &mut TempParametricArenas<'temp>,
    py: Python<'_>,
    c: &normalized::CallableType,
    seen: &mut SeenMap<'temp>,
) -> PyResult<ParametricCallable<'temp>> {
    let params: IndexMap<Arc<str>, PyTypeParametricKey<'temp>> = c
        .param_names
        .iter()
        .zip(c.params.iter())
        .map(|(name, param_type)| {
            ingest_inner(arenas, py, param_type, seen).map(|r| (Arc::from(name.as_str()), r))
        })
        .collect::<PyResult<_>>()?;
    let param_kinds: Vec<ParamKind> = c
        .param_kinds
        .iter()
        .map(|s| parse_param_kind(s))
        .collect::<PyResult<_>>()?;
    let return_type = ingest_inner(arenas, py, &c.return_type, seen)?;
    let return_wrapper = parse_wrapper_kind(&c.return_wrapper)?;
    let type_params = c
        .type_params
        .iter()
        .map(|tp| ingest_inner(arenas, py, tp, seen))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Qualified {
        inner: CallableType {
            params,
            param_kinds,
            param_has_default: c.param_has_default.clone(),
            param_context_inject: c.param_context_inject.clone(),
            accepts_varargs: c.accepts_varargs,
            accepts_varkw: c.accepts_varkw,
            return_type,
            return_wrapper,
            type_params,
            function_name: c.function_name.as_deref().map(Arc::from),
        },
        qualifier: c.qualifiers.clone(),
    })
}

fn ingest_btree_map_tracked<'temp>(
    arenas: &mut TempParametricArenas<'temp>,
    py: Python<'_>,
    map: &BTreeMap<String, NormalizedTypeRef>,
    seen: &mut SeenMap<'temp>,
) -> PyResult<BTreeMap<Arc<str>, PyTypeParametricKey<'temp>>> {
    map.iter()
        .map(|(k, v)| ingest_inner(arenas, py, v, seen).map(|r| (Arc::from(k.as_str()), r)))
        .collect()
}
