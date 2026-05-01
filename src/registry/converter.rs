use std::sync::Arc;

use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use crate::compile::{
    self, CompileRegistry, SOLVER_FIXPOINT_ITERATION_LIMIT, SOLVER_STACK_DEPTH_LIMIT, SolverLimits,
    ingest::ingest_parametric,
};
use crate::normalized::NormalizedTypeRef;
use crate::registry::entries::{Constructor, Hook, MethodImplementation};
use crate::rules::builder::RuleGraph;
use crate::types::{CallableKey, Parametric, PyType, TypeArenas};

struct RawConstructor {
    callable_type: NormalizedTypeRef,
    implementation: Arc<Py<PyAny>>,
}

struct RawMethodImplementation {
    name: Arc<str>,
    callable_type: NormalizedTypeRef,
    implementation: Arc<Py<PyAny>>,
    bound_to: Option<NormalizedTypeRef>,
}

struct RawHook {
    name: Arc<str>,
    callable_type: NormalizedTypeRef,
    implementation: Arc<Py<PyAny>>,
}

#[pyclass(module = "inlay")]
pub struct Registry {
    constructors: Vec<RawConstructor>,
    methods: Vec<RawMethodImplementation>,
    hooks: Vec<RawHook>,
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
        let mut constructors = Vec::new();
        let mut methods = Vec::new();
        let mut hooks = Vec::new();

        // Walk registry.constructors: tuple[BuiltConstructorEntry, ...]
        let py_constructors: Bound<'_, PyAny> = registry.getattr("constructors")?;
        for entry in py_constructors.try_iter()? {
            let entry = entry?;
            constructors.push(convert_constructor(&entry)?);
        }

        // Walk registry.methods: dict[str, tuple[BuiltMethodEntry, ...]]
        let py_methods: Bound<'_, PyAny> = registry.getattr("methods")?;
        for item in py_methods.call_method0("items")?.try_iter()? {
            let item = item?;
            let method_name: String = item.get_item(0)?.extract()?;
            let entries = item.get_item(1)?;
            for entry in entries.try_iter()? {
                let entry = entry?;
                methods.push(convert_method(&entry, &method_name)?);
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
                hooks.push(convert_hook(&entry, &hook_name)?);
            }
        }

        Ok(Self {
            constructors,
            methods,
            hooks,
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for c in &self.constructors {
            visit.call(&*c.implementation)?;
            c.callable_type.traverse(&visit)?;
        }
        for m in &self.methods {
            visit.call(&*m.implementation)?;
            m.callable_type.traverse(&visit)?;
            if let Some(bound_to) = &m.bound_to {
                bound_to.traverse(&visit)?;
            }
        }
        for h in &self.hooks {
            visit.call(&*h.implementation)?;
            h.callable_type.traverse(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&mut self) {
        self.constructors.clear();
        self.methods.clear();
        self.hooks.clear();
    }

    #[pyo3(signature = (
        rules,
        target,
        *,
        solver_fixpoint_iteration_limit = SOLVER_FIXPOINT_ITERATION_LIMIT,
        solver_stack_depth_limit = SOLVER_STACK_DEPTH_LIMIT,
    ))]
    fn compile(
        &mut self,
        py: Python<'_>,
        rules: &RuleGraph,
        target: NormalizedTypeRef,
        solver_fixpoint_iteration_limit: usize,
        solver_stack_depth_limit: usize,
    ) -> PyResult<Py<PyAny>> {
        let mut arenas = TypeArenas::default();
        let constructors = self
            .constructors
            .iter()
            .map(|entry| materialize_constructor(&mut arenas, py, entry))
            .collect::<PyResult<Vec<_>>>()?;
        let methods = self
            .methods
            .iter()
            .map(|entry| materialize_method(&mut arenas, py, entry))
            .collect::<PyResult<Vec<_>>>()?;
        let hooks = self
            .hooks
            .iter()
            .map(|entry| materialize_hook(&mut arenas, py, entry))
            .collect::<PyResult<Vec<_>>>()?;

        compile::compile(
            py,
            &mut arenas,
            CompileRegistry {
                constructors: &constructors,
                methods: &methods,
                hooks: &hooks,
            },
            rules,
            target,
            SolverLimits {
                fixpoint_iteration: solver_fixpoint_iteration_limit,
                stack_depth: solver_stack_depth_limit,
            },
        )
    }
}

fn ingest_callable_type<'types>(
    arenas: &mut TypeArenas<'types>,
    py: Python<'_>,
    callable_type: &NormalizedTypeRef,
) -> PyResult<CallableKey<'types, Parametric>> {
    let parametric_ref = ingest_parametric(arenas, py, callable_type)?;
    match parametric_ref {
        PyType::Callable(key) => Ok(key),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "callable_type must be a CallableType",
        )),
    }
}

fn convert_constructor(entry: &Bound<'_, PyAny>) -> PyResult<RawConstructor> {
    Ok(RawConstructor {
        callable_type: entry.getattr("callable_type")?.extract()?,
        implementation: Arc::new(entry.getattr("constructor")?.unbind()),
    })
}

fn materialize_constructor<'types>(
    arenas: &mut TypeArenas<'types>,
    py: Python<'_>,
    entry: &RawConstructor,
) -> PyResult<Constructor<'types>> {
    Ok(Constructor {
        fn_type: ingest_callable_type(arenas, py, &entry.callable_type)?,
        implementation: Arc::clone(&entry.implementation),
    })
}

fn convert_method(entry: &Bound<'_, PyAny>, name: &str) -> PyResult<RawMethodImplementation> {
    let bound_to_obj: Bound<'_, PyAny> = entry.getattr("bound_to")?;
    let bound_to: Option<NormalizedTypeRef> = if bound_to_obj.is_none() {
        None
    } else {
        Some(bound_to_obj.extract()?)
    };

    Ok(RawMethodImplementation {
        name: Arc::from(name),
        callable_type: entry.getattr("callable_type")?.extract()?,
        implementation: Arc::new(entry.getattr("implementation")?.unbind()),
        bound_to,
    })
}

fn materialize_method<'types>(
    arenas: &mut TypeArenas<'types>,
    py: Python<'_>,
    entry: &RawMethodImplementation,
) -> PyResult<MethodImplementation<'types>> {
    Ok(MethodImplementation {
        name: Arc::clone(&entry.name),
        fn_type: ingest_callable_type(arenas, py, &entry.callable_type)?,
        implementation: Arc::clone(&entry.implementation),
        bound_to: entry
            .bound_to
            .as_ref()
            .map(|bound_to| ingest_parametric(arenas, py, bound_to))
            .transpose()?,
    })
}

fn convert_hook(entry: &Bound<'_, PyAny>, name: &str) -> PyResult<RawHook> {
    Ok(RawHook {
        name: Arc::from(name),
        callable_type: entry.getattr("callable_type")?.extract()?,
        implementation: Arc::new(entry.getattr("implementation")?.unbind()),
    })
}

fn materialize_hook<'types>(
    arenas: &mut TypeArenas<'types>,
    py: Python<'_>,
    entry: &RawHook,
) -> PyResult<Hook<'types>> {
    Ok(Hook {
        name: Arc::clone(&entry.name),
        fn_type: ingest_callable_type(arenas, py, &entry.callable_type)?,
        implementation: Arc::clone(&entry.implementation),
    })
}
