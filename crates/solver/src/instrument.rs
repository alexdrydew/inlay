pub(crate) const TARGET: &str = "context_solver";

macro_rules! solver_trace_enabled {
    () => {{
        #[cfg(feature = "tracing")]
        {
            ::tracing::enabled!(target: $crate::instrument::TARGET, ::tracing::Level::TRACE)
        }
        #[cfg(not(feature = "tracing"))]
        {
            false
        }
    }};
}

macro_rules! solver_event {
    (name: $name:expr $(, $($fields:tt)*)?) => {{
        #[cfg(feature = "tracing")]
        {
            ::tracing::event!(
                name: $name,
                target: $crate::instrument::TARGET,
                ::tracing::Level::TRACE,
                perfetto = true
                $(, $($fields)*)?
            );
        }
    }};
}

macro_rules! solver_in_span {
    ($name:expr, { $($fields:tt)* }, $body:block) => {{
        #[cfg(feature = "tracing")]
        {
            let _solver_span = ::tracing::trace_span!(
                target: $crate::instrument::TARGET,
                $name,
                perfetto = true
                , $($fields)*
            )
            .entered();
            $body
        }
        #[cfg(not(feature = "tracing"))]
        {
            $body
        }
    }};
}

pub(crate) use solver_event;
pub(crate) use solver_in_span;
pub(crate) use solver_trace_enabled;
