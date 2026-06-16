use std::sync::Arc;

use inlay_instrument::instrumented;
use pyo3::PyTraverseError;
use pyo3::gc::PyVisit;
use pyo3::prelude::*;

use super::{
    RegistrySolver, SOLVER_FIXPOINT_ITERATION_LIMIT, SOLVER_STACK_DEPTH_LIMIT,
    execution_graph::create_execution_graph, ingest::ingest_parametric,
    solver_error_to_resolution_error,
};
use crate::normalized::NormalizedTypeRef;
use crate::registry::{Constructor, MethodImplementation};
use crate::rules::{
    BoundImplementation, PyStdoutWriter, RegistryEnv, RegistryResolutionRule, RegistrySharedState,
    ResolutionGraphJsonError, ResolutionQuery, builder::RuleGraph, write_resolution_graph_json,
};
use crate::runtime::executor::{ContextData, execute};
use crate::runtime::resources::RuntimeResources;
use crate::types::{Bindings, CallableKey, Parametric, ProtocolKey, PyType, TypeArenas};

struct RawConstructor {
    callable_type: NormalizedTypeRef,
    implementation: Arc<Py<PyAny>>,
}

struct RawMethodImplementation {
    name: Arc<str>,
    registration_protocol: NormalizedTypeRef,
    public_callable_type: NormalizedTypeRef,
    implementation_callable_type: NormalizedTypeRef,
    implementation: Arc<Py<PyAny>>,
    bound_to: Option<NormalizedTypeRef>,
    order: usize,
}

#[pyclass(module = "inlay")]
pub struct Compiler {
    constructors: Vec<RawConstructor>,
    methods: Vec<RawMethodImplementation>,
    solver: RegistrySolver,
    root_rule: crate::rules::RuleId,
}

impl std::fmt::Debug for Compiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compiler")
            .field("constructors", &self.constructors.len())
            .field("methods", &self.methods.len())
            .field("solver", &self.solver)
            .finish()
    }
}

#[pymethods]
impl Compiler {
    #[new]
    #[pyo3(signature = (
        registry,
        rules,
        solver_fixpoint_iteration_limit = SOLVER_FIXPOINT_ITERATION_LIMIT,
        solver_stack_depth_limit = SOLVER_STACK_DEPTH_LIMIT,
    ))]
    fn new(
        registry: &Bound<'_, PyAny>,
        rules: &RuleGraph,
        solver_fixpoint_iteration_limit: usize,
        solver_stack_depth_limit: usize,
    ) -> PyResult<Self> {
        let mut constructors = Vec::new();
        let mut methods = Vec::new();

        let py_constructors: Bound<'_, PyAny> = registry.getattr("constructors")?;
        for entry in py_constructors.try_iter()? {
            let entry = entry?;
            constructors.push(convert_constructor(&entry)?);
        }

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

        let mut arenas: TypeArenas<'static> = TypeArenas::default();
        let materialized_constructors = constructors
            .iter()
            .map(|entry| materialize_constructor(&mut arenas, registry.py(), entry))
            .collect::<PyResult<Vec<_>>>()?;
        let materialized_methods = methods
            .iter()
            .map(|entry| materialize_method(&mut arenas, registry.py(), entry))
            .collect::<PyResult<Vec<_>>>()?;
        let solver = RegistrySolver::new(
            RegistryResolutionRule::new(Arc::new(rules.arena.clone())),
            RegistrySharedState::new(&materialized_constructors, &materialized_methods, arenas),
            solver_fixpoint_iteration_limit,
            solver_stack_depth_limit,
        );

        Ok(Self {
            constructors,
            methods,
            solver,
            root_rule: rules.root,
        })
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for c in &self.constructors {
            visit.call(&*c.implementation)?;
            c.callable_type.traverse(&visit)?;
        }
        for m in &self.methods {
            visit.call(&*m.implementation)?;
            m.registration_protocol.traverse(&visit)?;
            m.public_callable_type.traverse(&visit)?;
            m.implementation_callable_type.traverse(&visit)?;
            if let Some(bound_to) = &m.bound_to {
                bound_to.traverse(&visit)?;
            }
        }
        self.solver.shared_state().types.traverse_py_refs(&visit)?;
        Ok(())
    }

    fn __clear__(&mut self) {
        self.constructors.clear();
        self.methods.clear();
        self.solver.clear_results_and_cache();
        self.solver.shared_state_mut().clear_for_gc();
    }

    #[instrumented(
        name = "inlay.compile",
        target = "inlay",
        level = "info",
        skip(py, self, debug)
    )]
    #[pyo3(signature = (target, debug=false))]
    fn compile(
        &mut self,
        py: Python<'_>,
        target: NormalizedTypeRef,
        debug: bool,
    ) -> PyResult<Py<PyAny>> {
        let parametric = ingest_parametric(self.solver.shared_state_mut().types(), py, &target)?;
        let concrete = self
            .solver
            .shared_state_mut()
            .types()
            .apply_bindings(parametric, &Bindings::default());
        self.compile_concrete(py, concrete, RegistryEnv::default(), debug)
    }

    #[pyo3(signature = (public_root_type, bound_public_type, implementation_type, debug=false))]
    fn compile_with_bound(
        &mut self,
        py: Python<'_>,
        public_root_type: NormalizedTypeRef,
        bound_public_type: NormalizedTypeRef,
        implementation_type: NormalizedTypeRef,
        debug: bool,
    ) -> PyResult<Py<PyAny>> {
        let public_root_parametric = ingest_parametric(
            self.solver.shared_state_mut().types(),
            py,
            &public_root_type,
        )?;
        let bound_public_parametric = ingest_parametric(
            self.solver.shared_state_mut().types(),
            py,
            &bound_public_type,
        )?;
        let implementation_parametric = ingest_parametric(
            self.solver.shared_state_mut().types(),
            py,
            &implementation_type,
        )?;

        let (public_root_concrete, binding) = {
            let types = self.solver.shared_state_mut().types();
            let public_root_concrete =
                types.apply_bindings(public_root_parametric, &Bindings::default());
            let bound_public_concrete =
                types.apply_bindings(bound_public_parametric, &Bindings::default());
            let implementation_concrete =
                types.apply_bindings(implementation_parametric, &Bindings::default());
            let binding =
                static_bound_implementation(types, bound_public_concrete, implementation_concrete)?;
            (public_root_concrete, binding)
        };
        let env = {
            let types = &self.solver.shared_state().types;
            RegistryEnv::default().with_bound_implementation(binding, types)
        };
        self.compile_concrete(py, public_root_concrete, env, debug)
    }
}

fn static_bound_implementation<'ty>(
    types: &TypeArenas<'ty>,
    public_type: crate::types::PyTypeConcreteKey<'ty>,
    implementation_type: crate::types::PyTypeConcreteKey<'ty>,
) -> PyResult<BoundImplementation<'ty>> {
    let PyType::CallableImplementation(implementation_key) = implementation_type else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "implementation_type must be a CallableType",
        ));
    };
    let implementation = types
        .concrete
        .callable_implementations
        .get(implementation_key);
    let PyType::Callable(implementation_type) = implementation.inner.signature else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "CallableType.signature must be a CallableSignatureType",
        ));
    };
    Ok(BoundImplementation {
        public_type,
        implementation_type: PyType::Callable(implementation_type),
        source: crate::registry::Source::provider_result(Arc::clone(
            &implementation.inner.implementation,
        )),
    })
}

impl Compiler {
    fn compile_concrete(
        &mut self,
        py: Python<'_>,
        concrete: crate::types::PyTypeConcreteKey<'static>,
        env: RegistryEnv<'static>,
        debug: bool,
    ) -> PyResult<Py<PyAny>> {
        let root_rule = self.root_rule;
        let (data, root) = py.detach(|| {
            let root = match self.solver.solve_with_env(
                ResolutionQuery::unnamed(concrete),
                root_rule,
                Arc::new(env),
            ) {
                Ok(root) => root,
                Err(error) => {
                    return Err(solver_error_to_resolution_error(error, concrete)
                        .into_py_err(&self.solver.shared_state().types));
                }
            };

            let (exec_graph, exec_root) = create_execution_graph(
                self.solver.results(),
                root,
                &self.solver.shared_state().types,
            )
            .map_err(|e| e.into_py_err(&self.solver.shared_state().types))?;

            Ok::<_, PyErr>((
                ContextData {
                    graph: Arc::new(exec_graph),
                    root_node: exec_root,
                },
                root,
            ))
        })?;
        if debug {
            let mut writer = PyStdoutWriter::new(py)?;
            write_resolution_graph_json(
                &mut writer,
                self.solver.results(),
                root,
                &self.solver.shared_state().types,
            )
            .map_err(|error| {
                resolution_graph_json_error_to_py_err(error, &self.solver.shared_state().types)
            })?;
            std::io::Write::write_all(&mut writer, b"\n")
                .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?;
            std::io::Write::flush(&mut writer)
                .map_err(|error| pyo3::exceptions::PyOSError::new_err(error.to_string()))?;
        }
        execute(py, &data, RuntimeResources::empty(), false)
    }
}

fn resolution_graph_json_error_to_py_err<'ty>(
    error: ResolutionGraphJsonError<'ty>,
    types: &TypeArenas<'ty>,
) -> PyErr {
    match error {
        ResolutionGraphJsonError::Resolution(error) => error.into_py_err(types),
        ResolutionGraphJsonError::Json(error) => {
            pyo3::exceptions::PyValueError::new_err(error.to_string())
        }
        ResolutionGraphJsonError::Io(error) => {
            pyo3::exceptions::PyOSError::new_err(error.to_string())
        }
    }
}

fn ingest_callable_type<'ty>(
    arenas: &mut TypeArenas<'ty>,
    py: Python<'_>,
    callable_type: &NormalizedTypeRef,
) -> PyResult<CallableKey<'ty, Parametric>> {
    let parametric_ref = ingest_parametric(arenas, py, callable_type)?;
    match parametric_ref {
        PyType::Callable(key) => Ok(key),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "callable_type must be a CallableSignatureType",
        )),
    }
}

fn ingest_callable_implementation_type<'ty>(
    arenas: &mut TypeArenas<'ty>,
    py: Python<'_>,
    callable_type: &NormalizedTypeRef,
) -> PyResult<CallableKey<'ty, Parametric>> {
    let parametric_ref = ingest_parametric(arenas, py, callable_type)?;
    let PyType::CallableImplementation(key) = parametric_ref else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "implementation_callable_type must be a CallableType",
        ));
    };
    match arenas
        .parametric
        .callable_implementations
        .get(key)
        .inner
        .signature
    {
        PyType::Callable(signature_key) => Ok(signature_key),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "CallableType.signature must be a CallableSignatureType",
        )),
    }
}

fn ingest_protocol_type<'ty>(
    arenas: &mut TypeArenas<'ty>,
    py: Python<'_>,
    protocol_type: &NormalizedTypeRef,
) -> PyResult<ProtocolKey<'ty, Parametric>> {
    let parametric_ref = ingest_parametric(arenas, py, protocol_type)?;
    match parametric_ref {
        PyType::Protocol(key) => Ok(key),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            "registration_protocol must be a ProtocolType",
        )),
    }
}

fn convert_constructor(entry: &Bound<'_, PyAny>) -> PyResult<RawConstructor> {
    Ok(RawConstructor {
        callable_type: entry.getattr("callable_type")?.extract()?,
        implementation: Arc::new(entry.getattr("constructor")?.unbind()),
    })
}

fn materialize_constructor<'ty>(
    arenas: &mut TypeArenas<'ty>,
    py: Python<'_>,
    entry: &RawConstructor,
) -> PyResult<Constructor<'ty>> {
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
        registration_protocol: entry.getattr("registration_protocol")?.extract()?,
        public_callable_type: entry.getattr("public_callable_type")?.extract()?,
        implementation_callable_type: entry.getattr("implementation_callable_type")?.extract()?,
        implementation: Arc::new(entry.getattr("implementation")?.unbind()),
        bound_to,
        order: entry.getattr("order")?.extract()?,
    })
}

fn materialize_method<'ty>(
    arenas: &mut TypeArenas<'ty>,
    py: Python<'_>,
    entry: &RawMethodImplementation,
) -> PyResult<MethodImplementation<'ty>> {
    Ok(MethodImplementation {
        name: Arc::clone(&entry.name),
        registration_protocol: ingest_protocol_type(arenas, py, &entry.registration_protocol)?,
        public_fn_type: ingest_callable_type(arenas, py, &entry.public_callable_type)?,
        implementation_fn_type: ingest_callable_implementation_type(
            arenas,
            py,
            &entry.implementation_callable_type,
        )?,
        implementation: Arc::clone(&entry.implementation),
        bound_to: entry
            .bound_to
            .as_ref()
            .map(|bound_to| ingest_parametric(arenas, py, bound_to))
            .transpose()?,
        order: entry.order,
    })
}
