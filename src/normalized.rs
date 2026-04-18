use std::collections::BTreeMap;

use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::qualifier::Qualifier;
use crate::types::SentinelTypeKind;

// ---- CyclePlaceholder ----

#[pyclass(frozen, module = "inlay")]
pub struct CyclePlaceholder {}

#[pymethods]
impl CyclePlaceholder {
    #[new]
    fn new() -> Self {
        Self {}
    }
}

// ---- NormalizedTypeRef ----

#[derive(FromPyObject)]
pub(crate) enum NormalizedTypeRef {
    Plain(Py<PlainType>),
    Protocol(Py<ProtocolType>),
    TypedDict(Py<TypedDictType>),
    Union(Py<UnionType>),
    Callable(Py<CallableType>),
    LazyRef(Py<LazyRefType>),
    Sentinel(Py<SentinelType>),
    TypeVar(Py<TypeVarType>),
    ParamSpec(Py<ParamSpecType>),
    CyclePlaceholder(Py<CyclePlaceholder>),
}

impl NormalizedTypeRef {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            Self::Plain(p) => Self::Plain(p.clone_ref(py)),
            Self::Protocol(p) => Self::Protocol(p.clone_ref(py)),
            Self::TypedDict(t) => Self::TypedDict(t.clone_ref(py)),
            Self::Union(u) => Self::Union(u.clone_ref(py)),
            Self::Callable(c) => Self::Callable(c.clone_ref(py)),
            Self::LazyRef(l) => Self::LazyRef(l.clone_ref(py)),
            Self::Sentinel(s) => Self::Sentinel(s.clone_ref(py)),
            Self::TypeVar(t) => Self::TypeVar(t.clone_ref(py)),
            Self::ParamSpec(p) => Self::ParamSpec(p.clone_ref(py)),
            Self::CyclePlaceholder(c) => Self::CyclePlaceholder(c.clone_ref(py)),
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Plain(p) => visit.call(p),
            Self::Protocol(p) => visit.call(p),
            Self::TypedDict(t) => visit.call(t),
            Self::Union(u) => visit.call(u),
            Self::Callable(c) => visit.call(c),
            Self::LazyRef(l) => visit.call(l),
            Self::Sentinel(s) => visit.call(s),
            Self::TypeVar(t) => visit.call(t),
            Self::ParamSpec(p) => visit.call(p),
            Self::CyclePlaceholder(c) => visit.call(c),
        }
    }

    fn to_pyobject(&self, py: Python<'_>) -> Py<PyAny> {
        match self {
            Self::Plain(p) => p.clone_ref(py).into_any(),
            Self::Protocol(p) => p.clone_ref(py).into_any(),
            Self::TypedDict(t) => t.clone_ref(py).into_any(),
            Self::Union(u) => u.clone_ref(py).into_any(),
            Self::Callable(c) => c.clone_ref(py).into_any(),
            Self::LazyRef(l) => l.clone_ref(py).into_any(),
            Self::Sentinel(s) => s.clone_ref(py).into_any(),
            Self::TypeVar(t) => t.clone_ref(py).into_any(),
            Self::ParamSpec(p) => p.clone_ref(py).into_any(),
            Self::CyclePlaceholder(c) => c.clone_ref(py).into_any(),
        }
    }
}

// ---- GC traversal helpers ----

fn traverse_refs(refs: &[NormalizedTypeRef], visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
    for r in refs {
        r.traverse(visit)?;
    }
    Ok(())
}

fn traverse_ref_map(
    map: &BTreeMap<String, NormalizedTypeRef>,
    visit: &PyVisit<'_>,
) -> Result<(), PyTraverseError> {
    for val in map.values() {
        val.traverse(visit)?;
    }
    Ok(())
}

// ---- Equality helpers ----

fn refs_eq(a: &[NormalizedTypeRef], b: &[NormalizedTypeRef], py: Python<'_>) -> PyResult<bool> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for (x, y) in a.iter().zip(b.iter()) {
        if !x.to_pyobject(py).bind(py).eq(y.to_pyobject(py).bind(py))? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn map_eq(
    a: &BTreeMap<String, NormalizedTypeRef>,
    b: &BTreeMap<String, NormalizedTypeRef>,
    py: Python<'_>,
) -> PyResult<bool> {
    if a.len() != b.len() {
        return Ok(false);
    }
    for ((k1, v1), (k2, v2)) in a.iter().zip(b.iter()) {
        if k1 != k2 {
            return Ok(false);
        }
        if !v1
            .to_pyobject(py)
            .bind(py)
            .eq(v2.to_pyobject(py).bind(py))?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn replace_in_vec(
    vec: &mut [NormalizedTypeRef],
    old_ptr: *mut pyo3::ffi::PyObject,
    new: &NormalizedTypeRef,
    py: Python<'_>,
) {
    for child in vec.iter_mut() {
        if child.to_pyobject(py).as_ptr() == old_ptr {
            *child = new.clone_ref(py);
        }
    }
}

fn replace_in_map(
    map: &mut BTreeMap<String, NormalizedTypeRef>,
    old_ptr: *mut pyo3::ffi::PyObject,
    new: &NormalizedTypeRef,
    py: Python<'_>,
) {
    for val in map.values_mut() {
        if val.to_pyobject(py).as_ptr() == old_ptr {
            *val = new.clone_ref(py);
        }
    }
}

fn make_tuple<'py>(items: &[NormalizedTypeRef], py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
    let objects: Vec<Py<PyAny>> = items.iter().map(|r| r.to_pyobject(py)).collect();
    PyTuple::new(py, objects)
}

fn make_dict<'py>(
    map: &BTreeMap<String, NormalizedTypeRef>,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, v.to_pyobject(py))?;
    }
    Ok(dict)
}

// ---- SentinelType ----

#[pyclass(frozen, module = "inlay")]
pub struct SentinelType {
    pub(crate) value: Py<PyAny>,
    pub(crate) kind: SentinelTypeKind,
    pub(crate) qualifiers: Qualifier,
}

#[pymethods]
impl SentinelType {
    #[new]
    #[pyo3(signature = (value, qualifiers))]
    fn new(value: Bound<'_, PyAny>, qualifiers: Qualifier) -> Self {
        let kind = if value.is_none() {
            SentinelTypeKind::None
        } else {
            SentinelTypeKind::Ellipsis
        };
        Self {
            value: value.unbind(),
            kind,
            qualifiers,
        }
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> Py<PyAny> {
        self.value.clone_ref(py)
    }

    #[getter]
    fn qualifiers(&self) -> Qualifier {
        self.qualifiers.clone()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let other = other.borrow();
        Ok(self.kind == other.kind && self.qualifiers == other.qualifiers)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.value)
    }

    fn __repr__(&self) -> String {
        let val_str = match self.kind {
            SentinelTypeKind::None => "None",
            SentinelTypeKind::Ellipsis => "...",
        };
        format!(
            "SentinelType(value={}, qualifiers={})",
            val_str,
            self.qualifiers.__repr__()
        )
    }
}

// ---- TypeVarType ----

#[pyclass(frozen, module = "inlay")]
pub struct TypeVarType {
    pub(crate) typevar: Py<PyAny>,
    pub(crate) qualifiers: Qualifier,
}

#[pymethods]
impl TypeVarType {
    #[new]
    #[pyo3(signature = (typevar, qualifiers))]
    fn new(typevar: Bound<'_, PyAny>, qualifiers: Qualifier) -> Self {
        Self {
            typevar: typevar.unbind(),
            qualifiers,
        }
    }

    #[getter]
    fn typevar(&self, py: Python<'_>) -> Py<PyAny> {
        self.typevar.clone_ref(py)
    }

    #[getter]
    fn qualifiers(&self) -> Qualifier {
        self.qualifiers.clone()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let other = other.borrow();
        Ok(
            self.typevar.bind(py).eq(other.typevar.bind(py))?
                && self.qualifiers == other.qualifiers,
        )
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.typevar)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name: String = self.typevar.bind(py).repr()?.extract()?;
        Ok(format!(
            "TypeVarType(typevar={}, qualifiers={})",
            name,
            self.qualifiers.__repr__()
        ))
    }
}

// ---- ParamSpecType ----

#[pyclass(frozen, module = "inlay")]
pub struct ParamSpecType {
    pub(crate) paramspec: Py<PyAny>,
    pub(crate) qualifiers: Qualifier,
}

#[pymethods]
impl ParamSpecType {
    #[new]
    #[pyo3(signature = (paramspec, qualifiers))]
    fn new(paramspec: Bound<'_, PyAny>, qualifiers: Qualifier) -> Self {
        Self {
            paramspec: paramspec.unbind(),
            qualifiers,
        }
    }

    #[getter]
    fn paramspec(&self, py: Python<'_>) -> Py<PyAny> {
        self.paramspec.clone_ref(py)
    }

    #[getter]
    fn qualifiers(&self) -> Qualifier {
        self.qualifiers.clone()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let other = other.borrow();
        Ok(self.paramspec.bind(py).eq(other.paramspec.bind(py))?
            && self.qualifiers == other.qualifiers)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.paramspec)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let name: String = self.paramspec.bind(py).repr()?.extract()?;
        Ok(format!(
            "ParamSpecType(paramspec={}, qualifiers={})",
            name,
            self.qualifiers.__repr__()
        ))
    }
}

// ---- PlainType ----

#[pyclass(module = "inlay")]
pub struct PlainType {
    pub(crate) origin: Py<PyAny>,
    pub(crate) args: Vec<NormalizedTypeRef>,
    pub(crate) qualifiers: Qualifier,
}

#[pymethods]
impl PlainType {
    #[new]
    #[pyo3(signature = (origin, args, qualifiers))]
    fn new(origin: Bound<'_, PyAny>, args: Vec<NormalizedTypeRef>, qualifiers: Qualifier) -> Self {
        Self {
            origin: origin.unbind(),
            args,
            qualifiers,
        }
    }

    #[getter]
    fn origin(&self, py: Python<'_>) -> Py<PyAny> {
        self.origin.clone_ref(py)
    }

    #[getter]
    fn args<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        make_tuple(&self.args, py)
    }

    #[getter]
    fn qualifiers(&self) -> Qualifier {
        self.qualifiers.clone()
    }

    fn _replace_child(&mut self, old: &Bound<'_, PyAny>, new: &Bound<'_, PyAny>) -> PyResult<()> {
        let new_ref: NormalizedTypeRef = new.extract()?;
        let py = old.py();
        replace_in_vec(&mut self.args, old.as_ptr(), &new_ref, py);
        Ok(())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let other = other.borrow();
        if self.qualifiers != other.qualifiers {
            return Ok(false);
        }
        if !self.origin.bind(py).eq(other.origin.bind(py))? {
            return Ok(false);
        }
        refs_eq(&self.args, &other.args, py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.origin)?;
        traverse_refs(&self.args, &visit)
    }

    fn __clear__(&mut self) {
        self.args.clear();
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let origin_repr: String = self.origin.bind(py).repr()?.extract()?;
        Ok(format!(
            "PlainType(origin={}, args=..., qualifiers={})",
            origin_repr,
            self.qualifiers.__repr__()
        ))
    }
}

// ---- ProtocolType ----

#[pyclass(module = "inlay")]
pub struct ProtocolType {
    pub(crate) origin: Py<PyAny>,
    pub(crate) type_params: Vec<NormalizedTypeRef>,
    pub(crate) methods: BTreeMap<String, NormalizedTypeRef>,
    pub(crate) attributes: BTreeMap<String, NormalizedTypeRef>,
    pub(crate) properties: BTreeMap<String, NormalizedTypeRef>,
    pub(crate) qualifiers: Qualifier,
}

#[pymethods]
impl ProtocolType {
    #[new]
    #[pyo3(signature = (origin, type_params, methods, attributes, properties, qualifiers))]
    fn new(
        origin: Bound<'_, PyAny>,
        type_params: Vec<NormalizedTypeRef>,
        methods: BTreeMap<String, NormalizedTypeRef>,
        attributes: BTreeMap<String, NormalizedTypeRef>,
        properties: BTreeMap<String, NormalizedTypeRef>,
        qualifiers: Qualifier,
    ) -> Self {
        Self {
            origin: origin.unbind(),
            type_params,
            methods,
            attributes,
            properties,
            qualifiers,
        }
    }

    #[getter]
    fn origin(&self, py: Python<'_>) -> Py<PyAny> {
        self.origin.clone_ref(py)
    }

    #[getter]
    fn type_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        make_tuple(&self.type_params, py)
    }

    #[getter]
    fn methods<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        make_dict(&self.methods, py)
    }

    #[getter]
    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        make_dict(&self.attributes, py)
    }

    #[getter]
    fn properties<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        make_dict(&self.properties, py)
    }

    #[getter]
    fn qualifiers(&self) -> Qualifier {
        self.qualifiers.clone()
    }

    fn _replace_child(&mut self, old: &Bound<'_, PyAny>, new: &Bound<'_, PyAny>) -> PyResult<()> {
        let new_ref: NormalizedTypeRef = new.extract()?;
        let old_ptr = old.as_ptr();
        let py = old.py();
        replace_in_vec(&mut self.type_params, old_ptr, &new_ref, py);
        replace_in_map(&mut self.methods, old_ptr, &new_ref, py);
        replace_in_map(&mut self.attributes, old_ptr, &new_ref, py);
        replace_in_map(&mut self.properties, old_ptr, &new_ref, py);
        Ok(())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let other = other.borrow();
        if self.qualifiers != other.qualifiers {
            return Ok(false);
        }
        if !self.origin.bind(py).eq(other.origin.bind(py))? {
            return Ok(false);
        }
        if !refs_eq(&self.type_params, &other.type_params, py)? {
            return Ok(false);
        }
        if !map_eq(&self.methods, &other.methods, py)? {
            return Ok(false);
        }
        if !map_eq(&self.attributes, &other.attributes, py)? {
            return Ok(false);
        }
        map_eq(&self.properties, &other.properties, py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.origin)?;
        traverse_refs(&self.type_params, &visit)?;
        traverse_ref_map(&self.methods, &visit)?;
        traverse_ref_map(&self.attributes, &visit)?;
        traverse_ref_map(&self.properties, &visit)
    }

    fn __clear__(&mut self) {
        self.type_params.clear();
        self.methods.clear();
        self.attributes.clear();
        self.properties.clear();
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let origin_repr: String = self.origin.bind(py).repr()?.extract()?;
        Ok(format!(
            "ProtocolType(origin={}, qualifiers={})",
            origin_repr,
            self.qualifiers.__repr__()
        ))
    }
}

// ---- TypedDictType ----

#[pyclass(module = "inlay")]
pub struct TypedDictType {
    pub(crate) origin: Py<PyAny>,
    pub(crate) type_params: Vec<NormalizedTypeRef>,
    pub(crate) attributes: BTreeMap<String, NormalizedTypeRef>,
    pub(crate) qualifiers: Qualifier,
}

#[pymethods]
impl TypedDictType {
    #[new]
    #[pyo3(signature = (origin, type_params, attributes, qualifiers))]
    fn new(
        origin: Bound<'_, PyAny>,
        type_params: Vec<NormalizedTypeRef>,
        attributes: BTreeMap<String, NormalizedTypeRef>,
        qualifiers: Qualifier,
    ) -> Self {
        Self {
            origin: origin.unbind(),
            type_params,
            attributes,
            qualifiers,
        }
    }

    #[getter]
    fn origin(&self, py: Python<'_>) -> Py<PyAny> {
        self.origin.clone_ref(py)
    }

    #[getter]
    fn type_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        make_tuple(&self.type_params, py)
    }

    #[getter]
    fn attributes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        make_dict(&self.attributes, py)
    }

    #[getter]
    fn qualifiers(&self) -> Qualifier {
        self.qualifiers.clone()
    }

    fn _replace_child(&mut self, old: &Bound<'_, PyAny>, new: &Bound<'_, PyAny>) -> PyResult<()> {
        let new_ref: NormalizedTypeRef = new.extract()?;
        let old_ptr = old.as_ptr();
        let py = old.py();
        replace_in_vec(&mut self.type_params, old_ptr, &new_ref, py);
        replace_in_map(&mut self.attributes, old_ptr, &new_ref, py);
        Ok(())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let other = other.borrow();
        if self.qualifiers != other.qualifiers {
            return Ok(false);
        }
        if !self.origin.bind(py).eq(other.origin.bind(py))? {
            return Ok(false);
        }
        if !refs_eq(&self.type_params, &other.type_params, py)? {
            return Ok(false);
        }
        map_eq(&self.attributes, &other.attributes, py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.origin)?;
        traverse_refs(&self.type_params, &visit)?;
        traverse_ref_map(&self.attributes, &visit)
    }

    fn __clear__(&mut self) {
        self.type_params.clear();
        self.attributes.clear();
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let origin_repr: String = self.origin.bind(py).repr()?.extract()?;
        Ok(format!(
            "TypedDictType(origin={}, qualifiers={})",
            origin_repr,
            self.qualifiers.__repr__()
        ))
    }
}

// ---- UnionType ----

#[pyclass(module = "inlay")]
pub struct UnionType {
    pub(crate) variants: Vec<NormalizedTypeRef>,
    pub(crate) qualifiers: Qualifier,
}

#[pymethods]
impl UnionType {
    #[new]
    #[pyo3(signature = (variants, qualifiers))]
    fn new(variants: Vec<NormalizedTypeRef>, qualifiers: Qualifier) -> Self {
        Self {
            variants,
            qualifiers,
        }
    }

    #[getter]
    fn variants<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        make_tuple(&self.variants, py)
    }

    #[getter]
    fn qualifiers(&self) -> Qualifier {
        self.qualifiers.clone()
    }

    fn _replace_child(&mut self, old: &Bound<'_, PyAny>, new: &Bound<'_, PyAny>) -> PyResult<()> {
        let new_ref: NormalizedTypeRef = new.extract()?;
        replace_in_vec(&mut self.variants, old.as_ptr(), &new_ref, old.py());
        Ok(())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let other = other.borrow();
        if self.qualifiers != other.qualifiers {
            return Ok(false);
        }
        refs_eq(&self.variants, &other.variants, py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        traverse_refs(&self.variants, &visit)
    }

    fn __clear__(&mut self) {
        self.variants.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "UnionType(variants=..., qualifiers={})",
            self.qualifiers.__repr__()
        )
    }
}

// ---- CallableType ----

#[pyclass(module = "inlay")]
pub struct CallableType {
    pub(crate) params: Vec<NormalizedTypeRef>,
    pub(crate) param_names: Vec<String>,
    pub(crate) param_kinds: Vec<String>,
    pub(crate) param_has_default: Vec<bool>,
    pub(crate) return_type: NormalizedTypeRef,
    pub(crate) return_wrapper: String,
    pub(crate) type_params: Vec<NormalizedTypeRef>,
    pub(crate) qualifiers: Qualifier,
    pub(crate) function_name: Option<String>,
}

#[pymethods]
impl CallableType {
    #[new]
    #[pyo3(signature = (params, param_names, param_kinds, return_type, return_wrapper, type_params, qualifiers, function_name=None, param_has_default=None))]
    fn new(
        params: Vec<NormalizedTypeRef>,
        param_names: Vec<String>,
        param_kinds: Vec<String>,
        return_type: NormalizedTypeRef,
        return_wrapper: String,
        type_params: Vec<NormalizedTypeRef>,
        qualifiers: Qualifier,
        function_name: Option<String>,
        param_has_default: Option<Vec<bool>>,
    ) -> Self {
        let param_has_default = param_has_default.unwrap_or_else(|| vec![false; params.len()]);
        Self {
            params,
            param_names,
            param_kinds,
            param_has_default,
            return_type,
            return_wrapper,
            type_params,
            qualifiers,
            function_name,
        }
    }

    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        make_tuple(&self.params, py)
    }

    #[getter]
    fn param_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.param_names.iter().map(String::as_str))
    }

    #[getter]
    fn param_kinds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.param_kinds.iter().map(String::as_str))
    }

    #[getter]
    fn return_type(&self, py: Python<'_>) -> Py<PyAny> {
        self.return_type.to_pyobject(py)
    }

    #[getter]
    fn return_wrapper(&self) -> &str {
        &self.return_wrapper
    }

    #[getter]
    fn type_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        make_tuple(&self.type_params, py)
    }

    #[getter]
    fn qualifiers(&self) -> Qualifier {
        self.qualifiers.clone()
    }

    #[getter]
    fn function_name(&self) -> Option<&str> {
        self.function_name.as_deref()
    }

    fn _replace_child(&mut self, old: &Bound<'_, PyAny>, new: &Bound<'_, PyAny>) -> PyResult<()> {
        let new_ref: NormalizedTypeRef = new.extract()?;
        let old_ptr = old.as_ptr();
        let py = old.py();
        replace_in_vec(&mut self.params, old_ptr, &new_ref, py);
        if self.return_type.to_pyobject(py).as_ptr() == old_ptr {
            self.return_type = new_ref;
        }
        Ok(())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let other = other.borrow();
        if self.qualifiers != other.qualifiers {
            return Ok(false);
        }
        if self.function_name != other.function_name {
            return Ok(false);
        }
        if self.param_names != other.param_names {
            return Ok(false);
        }
        if self.param_kinds != other.param_kinds {
            return Ok(false);
        }
        if self.return_wrapper != other.return_wrapper {
            return Ok(false);
        }
        if !refs_eq(&self.params, &other.params, py)? {
            return Ok(false);
        }
        if !self
            .return_type
            .to_pyobject(py)
            .bind(py)
            .eq(other.return_type.to_pyobject(py).bind(py))?
        {
            return Ok(false);
        }
        if !refs_eq(&self.type_params, &other.type_params, py)? {
            return Ok(false);
        }
        Ok(true)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        traverse_refs(&self.params, &visit)?;
        self.return_type.traverse(&visit)?;
        traverse_refs(&self.type_params, &visit)
    }

    fn __clear__(&mut self) {
        self.params.clear();
        self.type_params.clear();
    }

    fn __repr__(&self) -> String {
        format!(
            "CallableType(params=..., qualifiers={})",
            self.qualifiers.__repr__()
        )
    }
}

// ---- LazyRefType ----

#[pyclass(module = "inlay")]
pub struct LazyRefType {
    pub(crate) target: NormalizedTypeRef,
    pub(crate) qualifiers: Qualifier,
}

#[pymethods]
impl LazyRefType {
    #[new]
    #[pyo3(signature = (target, qualifiers))]
    fn new(target: NormalizedTypeRef, qualifiers: Qualifier) -> Self {
        Self { target, qualifiers }
    }

    #[getter]
    fn target(&self, py: Python<'_>) -> Py<PyAny> {
        self.target.to_pyobject(py)
    }

    #[getter]
    fn qualifiers(&self) -> Qualifier {
        self.qualifiers.clone()
    }

    fn _replace_child(&mut self, old: &Bound<'_, PyAny>, new: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.target.to_pyobject(old.py()).as_ptr() == old.as_ptr() {
            self.target = new.extract()?;
        }
        Ok(())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        let other = other.borrow();
        if self.qualifiers != other.qualifiers {
            return Ok(false);
        }
        self.target
            .to_pyobject(py)
            .bind(py)
            .eq(other.target.to_pyobject(py).bind(py))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.target.traverse(&visit)
    }

    fn __repr__(&self) -> String {
        format!(
            "LazyRefType(target=..., qualifiers={})",
            self.qualifiers.__repr__()
        )
    }
}
