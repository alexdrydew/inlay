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

pub(crate) use inlay_span_record;
