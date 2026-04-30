mod compile;
mod normalized;
mod qualifier;
mod registry;
mod rules;
mod runtime;
mod types;

pub use context_solver as solver;

use pyo3::prelude::*;

pyo3::create_exception!(inlay, ResolutionError, pyo3::exceptions::PyException);

#[cfg(feature = "tracing")]
fn init_tracing() {
    use tracing_subscriber::{EnvFilter, Layer, prelude::*};

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_filter(EnvFilter::from_default_env());

    #[cfg(feature = "perfetto-tracing")]
    {
        use tracing_perfetto::PerfettoLayer;

        let subscriber = tracing_subscriber::registry().with(fmt_layer);
        if let Some(path) = std::env::var_os("INLAY_PERFETTO_TRACE_PATH") {
            match std::fs::File::create(path) {
                Ok(file) => {
                    let perfetto_layer = PerfettoLayer::new(file)
                        .with_debug_annotations(true)
                        .with_filter_by_marker(|field_name| field_name == "perfetto");
                    let _ = subscriber.with(perfetto_layer).try_init();
                }
                Err(_error) => {
                    let _ = subscriber.try_init();
                }
            }
        } else {
            let _ = subscriber.try_init();
        }
    }

    #[cfg(not(feature = "perfetto-tracing"))]
    {
        let _ = tracing_subscriber::registry().with(fmt_layer).try_init();
    }
}

#[cfg(not(feature = "tracing"))]
fn init_tracing() {}

#[pymodule(name = "_native")]
fn dicexdice_context(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_tracing();

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
