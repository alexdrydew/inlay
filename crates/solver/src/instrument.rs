#[cfg(feature = "tracing")]
pub(crate) const TARGET: &str = "context_solver";

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

macro_rules! solver_span_record {
    ($($field:ident $(= $value:expr)?),+ $(,)?) => {{
        #[cfg(feature = "tracing")]
        {
            let span = ::tracing::Span::current();
            $(
                $crate::instrument::solver_span_record!(@record span, $field $(= $value)?);
            )+
        }
    }};

    (@record $span:ident, $field:ident) => {
        $span.record(stringify!($field), $field);
    };

    (@record $span:ident, $field:ident = $value:expr) => {
        $span.record(stringify!($field), $value);
    };
}

pub(crate) use solver_event;
pub(crate) use solver_in_span;
pub(crate) use solver_span_record;
