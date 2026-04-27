use std::sync::Arc;

use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::compile::{self, ingest::ingest_parametric};
use crate::normalized::NormalizedTypeRef;
use crate::registry::entries::{Constructor, Hook, MethodImplementation};
use crate::rules::builder::RuleGraph;
use crate::types::{CallableKey, Parametric, PyType, PyTypeParametricKey, TypeArenas};

#[pyclass(module = "inlay")]
pub struct Registry {
    pub(crate) arenas: TypeArenas,
    pub(crate) constructors: Vec<Constructor>,
    pub(crate) methods: Vec<MethodImplementation>,
    pub(crate) hooks: Vec<Hook>,
}

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Registry")
            .field("constructors", &self.constructors.len())
            .field("methods", &self.methods.len())
            .field("hooks", &self.hooks.len())
            .finish()
    }
}

#[pymethods]
impl Registry {
    #[new]
    fn new(registry: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = registry.py();
        let mut arenas = TypeArenas::default();
        let mut constructors = Vec::new();
        let mut methods = Vec::new();
        let mut hooks = Vec::new();

        // Walk registry.constructors: tuple[BuiltConstructorEntry, ...]
        let py_constructors: Bound<'_, PyAny> = registry.getattr("constructors")?;
        for entry in py_constructors.try_iter()? {
            let entry = entry?;
            constructors.push(convert_constructor(&mut arenas, py, &entry)?);
        }

        // Walk registry.methods: dict[str, tuple[BuiltMethodEntry, ...]]
        let py_methods: Bound<'_, PyAny> = registry.getattr("methods")?;
        for item in py_methods.call_method0("items")?.try_iter()? {
            let item = item?;
            let method_name: String = item.get_item(0)?.extract()?;
            let entries = item.get_item(1)?;
            for entry in entries.try_iter()? {
                let entry = entry?;
                methods.push(convert_method(&mut arenas, py, &entry, &method_name)?);
            }
        }

        // Walk registry.hooks: dict[str, tuple[BuiltHookEntry, ...]]
        let py_hooks: Bound<'_, PyAny> = registry.getattr("hooks")?;
        for item in py_hooks.call_method0("items")?.try_iter()? {
            let item = item?;
            let hook_name: String = item.get_item(0)?.extract()?;
            let entries = item.get_item(1)?;
            for entry in entries.try_iter()? {
                let entry = entry?;
                hooks.push(convert_hook(&mut arenas, py, &entry, &hook_name)?);
            }
        }

        Ok(Self {
            arenas,
            constructors,
            methods,
            hooks,
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for c in &self.constructors {
            visit.call(&*c.implementation)?;
        }
        for m in &self.methods {
            visit.call(&*m.implementation)?;
        }
        for h in &self.hooks {
            visit.call(&*h.implementation)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.constructors.clear();
        self.methods.clear();
        self.hooks.clear();
    }

    #[pyo3(signature = (rules, target))]
    fn compile(
        &mut self,
        py: Python<'_>,
        rules: &RuleGraph,
        target: NormalizedTypeRef,
    ) -> PyResult<Py<PyAny>> {
        compile::compile(
            py,
            &mut self.arenas,
            &self.constructors,
            &self.methods,
            &self.hooks,
            rules,
            target,
        )
    }
}

fn ingest_callable_type(
    arenas: &mut TypeArenas,
    py: Python<'_>,
    entry: &Bound<'_, PyAny>,
) -> PyResult<CallableKey<Parametric>> {
    let callable_type_obj: Bound<'_, PyAny> = entry.getattr("callable_type")?;
    let ntype: NormalizedTypeRef = callable_type_obj.extract()?;
    let parametric_ref = ingest_parametric(arenas, py, &ntype)?;
    match parametric_ref {
        PyType::Callable(key) => Ok(key),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "callable_type must be a CallableType",
        )),
    }
}

fn convert_constructor(
    arenas: &mut TypeArenas,
    py: Python<'_>,
    entry: &Bound<'_, PyAny>,
) -> PyResult<Constructor> {
    let fn_type = ingest_callable_type(arenas, py, entry)?;
    let implementation: Py<PyAny> = entry.getattr("constructor")?.unbind();
    Ok(Constructor {
        fn_type,
        implementation: Arc::new(implementation),
    })
}

fn convert_method(
    arenas: &mut TypeArenas,
    py: Python<'_>,
    entry: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<MethodImplementation> {
    let fn_type = ingest_callable_type(arenas, py, entry)?;
    let implementation: Py<PyAny> = entry.getattr("implementation")?.unbind();

    let bound_to_obj: Bound<'_, PyAny> = entry.getattr("bound_to")?;
    let bound_to: Option<PyTypeParametricKey> = if bound_to_obj.is_none() {
        None
    } else {
        let ntype: NormalizedTypeRef = bound_to_obj.extract()?;
        Some(ingest_parametric(arenas, py, &ntype)?)
    };

    Ok(MethodImplementation {
        name: Arc::from(name),
        fn_type,
        implementation: Arc::new(implementation),
        bound_to,
    })
}

fn convert_hook(
    arenas: &mut TypeArenas,
    py: Python<'_>,
    entry: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Hook> {
    let fn_type = ingest_callable_type(arenas, py, entry)?;
    let implementation: Py<PyAny> = entry.getattr("implementation")?.unbind();
    Ok(Hook {
        name: Arc::from(name),
        fn_type,
        implementation: Arc::new(implementation),
    })
}
