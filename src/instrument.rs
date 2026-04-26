#[cfg(feature = "tracing")]
pub(crate) const TARGET: &str = "inlay";

macro_rules! inlay_in_span {
    ($name:expr, { $($fields:tt)* }, $body:block) => {{
        #[cfg(feature = "tracing")]
        {
            let _inlay_span = ::tracing::trace_span!(
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

macro_rules! inlay_span_record {
    ($($field:ident $(= $value:expr)?),+ $(,)?) => {{
        #[cfg(feature = "tracing")]
        {
            let span = ::tracing::Span::current();
            $(
                $crate::instrument::inlay_span_record!(@record span, $field $(= $value)?);
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

pub(crate) use inlay_in_span;
pub(crate) use inlay_span_record;
