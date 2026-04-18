#![allow(dead_code)]

mod compile;
mod ingest;
mod normalized;
mod qualifier;
mod registry;
mod rules;
mod runtime;
mod types;

pub use context_solver as solver;

use pyo3::prelude::*;

pyo3::create_exception!(inlay, ResolutionError, pyo3::exceptions::PyException);

#[pymodule(name = "_native")]
fn dicexdice_context(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use tracing_subscriber::EnvFilter;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .try_init();

    m.add("ResolutionError", m.py().get_type::<ResolutionError>())?;
    m.add_class::<qualifier::Qualifier>()?;
    m.add_class::<normalized::SentinelType>()?;
    m.add_class::<normalized::TypeVarType>()?;
    m.add_class::<normalized::ParamSpecType>()?;
    m.add_class::<normalized::PlainType>()?;
    m.add_class::<normalized::ProtocolType>()?;
    m.add_class::<normalized::TypedDictType>()?;
    m.add_class::<normalized::UnionType>()?;
    m.add_class::<normalized::CallableType>()?;
    m.add_class::<normalized::LazyRefType>()?;
    m.add_class::<normalized::CyclePlaceholder>()?;
    m.add_class::<rules::builder::RuleGraph>()?;
    m.add_class::<registry::converter::Registry>()?;
    m.add_class::<runtime::proxy::ContextProxy>()?;
    m.add_class::<runtime::proxy::DelegatedDict>()?;
    m.add_class::<runtime::proxy::DelegatedAttr>()?;
    m.add_class::<runtime::lazy_ref::LazyRefImpl>()?;
    m.add_class::<runtime::transition::Transition>()?;
    m.add_class::<runtime::transition::CmWrapper>()?;
    m.add_class::<runtime::transition::AwaitableWrapper>()?;
    m.add_class::<runtime::transition::AcmWrapper>()?;
    Ok(())
}
