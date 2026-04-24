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

pub(crate) use inlay_in_span;
