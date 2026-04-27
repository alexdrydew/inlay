use std::collections::BTreeMap;
use std::sync::Arc;

use indexmap::IndexMap;
use inlay_instrument_macros::instrumented;
use pyo3::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use pyo3::types::PyType;

use crate::normalized::{self, NormalizedTypeRef};
use crate::types::{
    Arena, CallableType, LazyRefType, ParamKind, ParamSpecType, PlainType, ProtocolType,
    PyType as PyTypeEnum, PyTypeDescriptor, PyTypeId, PyTypeParametricKey, Qualified, SentinelType,
    TypeArenas, TypeVarDescriptor, TypeVarType, TypedDictType, UnionType, WrapperKind,
};

fn make_type_descriptor(origin: &Bound<'_, PyAny>) -> PyResult<PyTypeDescriptor> {
    let id = PyTypeId::new(format!("{}", origin.as_ptr() as usize));
    let display_name: Arc<str> = match origin.cast::<PyType>() {
        Ok(t) => Arc::from(t.qualname()?.to_string()),
        Err(_) => Arc::from(origin.repr()?.to_string()),
    };
    Ok(PyTypeDescriptor { id, display_name })
}

fn make_typevar_descriptor(tv: &Bound<'_, PyAny>) -> PyResult<TypeVarDescriptor> {
    let id = PyTypeId::new(format!("{}", tv.as_ptr() as usize));
    let display_name: Arc<str> = match tv.getattr("__name__") {
        Ok(n) => Arc::from(n.extract::<String>()?),
        Err(_) => Arc::from(tv.repr()?.to_string()),
    };
    Ok(TypeVarDescriptor { id, display_name })
}

fn ntype_ptr(ntype: &NormalizedTypeRef) -> usize {
    match ntype {
        NormalizedTypeRef::Plain(p) => p.as_ptr() as usize,
        NormalizedTypeRef::Protocol(p) => p.as_ptr() as usize,
        NormalizedTypeRef::TypedDict(t) => t.as_ptr() as usize,
        NormalizedTypeRef::Union(u) => u.as_ptr() as usize,
        NormalizedTypeRef::Callable(c) => c.as_ptr() as usize,
        NormalizedTypeRef::LazyRef(l) => l.as_ptr() as usize,
        NormalizedTypeRef::Sentinel(s) => s.as_ptr() as usize,
        NormalizedTypeRef::TypeVar(t) => t.as_ptr() as usize,
        NormalizedTypeRef::ParamSpec(p) => p.as_ptr() as usize,
        NormalizedTypeRef::CyclePlaceholder(c) => c.as_ptr() as usize,
    }
}

type Seen = HashMap<usize, PyTypeParametricKey>;

#[instrumented(
    name = "inlay.ingest_parametric",
    target = "inlay",
    level = "trace",
    skip_all
)]
pub(crate) fn ingest_parametric(
    arenas: &mut TypeArenas,
    py: Python<'_>,
    ntype: &NormalizedTypeRef,
) -> PyResult<PyTypeParametricKey> {
    let mut seen = Seen::default();
    ingest_inner(arenas, py, ntype, &mut seen)
}

fn ingest_inner(
    arenas: &mut TypeArenas,
    py: Python<'_>,
    ntype: &NormalizedTypeRef,
    seen: &mut Seen,
) -> PyResult<PyTypeParametricKey> {
    let ptr = ntype_ptr(ntype);
    if let Some(&key) = seen.get(&ptr) {
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
            Ok(PyTypeEnum::Sentinel(arenas.sentinels.insert(val)))
        }
        NormalizedTypeRef::TypeVar(t) => {
            let t = t.bind(py).borrow();
            let descriptor = make_typevar_descriptor(t.typevar.bind(py))?;
            let val = Qualified {
                inner: TypeVarType { descriptor },
                qualifier: t.qualifiers.clone(),
            };
            Ok(PyTypeEnum::TypeVar(arenas.parametric.type_vars.insert(val)))
        }
        NormalizedTypeRef::ParamSpec(p) => {
            let p = p.bind(py).borrow();
            let descriptor = make_typevar_descriptor(p.paramspec.bind(py))?;
            let val = Qualified {
                inner: ParamSpecType { descriptor },
                qualifier: p.qualifiers.clone(),
            };
            Ok(PyTypeEnum::ParamSpec(
                arenas.parametric.param_specs.insert(val),
            ))
        }
        NormalizedTypeRef::Plain(p) => {
            let placeholder_key = arenas.parametric.plains.insert_placeholder();
            let result_key = PyTypeEnum::Plain(placeholder_key);
            seen.insert(ptr, result_key);

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
                    .parametric
                    .plains
                    .replace(placeholder_key, val)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::Protocol(p) => {
            let placeholder_key = arenas.parametric.protocols.insert_placeholder();
            let result_key = PyTypeEnum::Protocol(placeholder_key);
            seen.insert(ptr, result_key);

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
                    .parametric
                    .protocols
                    .replace(placeholder_key, val)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::TypedDict(t) => {
            let placeholder_key = arenas.parametric.typed_dicts.insert_placeholder();
            let result_key = PyTypeEnum::TypedDict(placeholder_key);
            seen.insert(ptr, result_key);

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
                    .parametric
                    .typed_dicts
                    .replace(placeholder_key, val)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::Union(u) => {
            let placeholder_key = arenas.parametric.unions.insert_placeholder();
            let result_key = PyTypeEnum::Union(placeholder_key);
            seen.insert(ptr, result_key);

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
                    .parametric
                    .unions
                    .replace(placeholder_key, val)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::Callable(c) => {
            let placeholder_key = arenas.parametric.callables.insert_placeholder();
            let result_key = PyTypeEnum::Callable(placeholder_key);
            seen.insert(ptr, result_key);

            let c = c.bind(py).borrow();
            let val = ingest_callable_value(arenas, py, &c, seen)?;
            assert!(
                arenas
                    .parametric
                    .callables
                    .replace(placeholder_key, val)
                    .expect("placeholder key should exist")
                    .is_none(),
                "placeholder key already filled"
            );
            Ok(result_key)
        }
        NormalizedTypeRef::LazyRef(l) => {
            let placeholder_key = arenas.parametric.lazy_refs.insert_placeholder();
            let result_key = PyTypeEnum::LazyRef(placeholder_key);
            seen.insert(ptr, result_key);

            let l = l.bind(py).borrow();
            let target = ingest_inner(arenas, py, &l.target, seen)?;
            let val = Qualified {
                inner: LazyRefType { target },
                qualifier: l.qualifiers.clone(),
            };
            assert!(
                arenas
                    .parametric
                    .lazy_refs
                    .replace(placeholder_key, val)
                    .expect("placeholder key should exist")
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
        "cm" => Ok(WrapperKind::Cm),
        "acm" => Ok(WrapperKind::Acm),
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

fn ingest_callable_value(
    arenas: &mut TypeArenas,
    py: Python<'_>,
    c: &normalized::CallableType,
    seen: &mut Seen,
) -> PyResult<
    Qualified<CallableType<crate::types::Qual<crate::types::Keyed>, crate::types::Parametric>>,
> {
    let params: IndexMap<Arc<str>, PyTypeParametricKey> = c
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

fn ingest_btree_map_tracked(
    arenas: &mut TypeArenas,
    py: Python<'_>,
    map: &BTreeMap<String, NormalizedTypeRef>,
    seen: &mut Seen,
) -> PyResult<BTreeMap<Arc<str>, PyTypeParametricKey>> {
    map.iter()
        .map(|(k, v)| ingest_inner(arenas, py, v, seen).map(|r| (Arc::from(k.as_str()), r)))
        .collect()
}
