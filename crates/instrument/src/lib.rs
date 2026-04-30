pub use inlay_instrument_macros::instrumented;

#[macro_export]
macro_rules! span_record {
    ($($field:ident $(= $value:expr)?),+ $(,)?) => {{
        #[cfg(feature = "tracing")]
        {
            let span = ::tracing::Span::current();
            $(
                $crate::span_record!(@record span, $field $(= $value)?);
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

#[macro_export]
macro_rules! solver_event {
    (name: $name:expr $(, $($fields:tt)*)?) => {{
        #[cfg(feature = "tracing")]
        {
            ::tracing::event!(
                name: $name,
                target: "context_solver",
                ::tracing::Level::TRACE,
                perfetto = true
                $(, $($fields)*)?
            );
        }
    }};
}

#[macro_export]
macro_rules! solver_in_span {
    ($name:expr, { $($fields:tt)* }, $body:block) => {{
        #[cfg(feature = "tracing")]
        {
            let _solver_span = ::tracing::trace_span!(
                target: "context_solver",
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
