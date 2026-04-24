use proc_macro::TokenStream;

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream};
use syn::{Block, Ident, ItemFn, LitStr, Result, Token, parenthesized, parse_macro_input};

struct InstrumentedArgs {
    name: LitStr,
    level: LitStr,
    setup: Option<Block>,
    trace_setup: Option<Block>,
    fields: Option<TokenStream2>,
}

impl Parse for InstrumentedArgs {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut name = None;
        let mut level = None;
        let mut setup = None;
        let mut trace_setup = None;
        let mut fields = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            match key.to_string().as_str() {
                "name" => {
                    input.parse::<Token![=]>()?;
                    name = Some(input.parse()?);
                }
                "level" => {
                    input.parse::<Token![=]>()?;
                    level = Some(input.parse()?);
                }
                "setup" => {
                    input.parse::<Token![=]>()?;
                    setup = Some(input.parse()?);
                }
                "trace_setup" => {
                    input.parse::<Token![=]>()?;
                    trace_setup = Some(input.parse()?);
                }
                "fields" => {
                    let content;
                    parenthesized!(content in input);
                    fields = Some(content.parse()?);
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        key,
                        "expected one of: name, level, setup, trace_setup, fields",
                    ));
                }
            }

            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }

        Ok(Self {
            name: name.ok_or_else(|| input.error("missing required argument: name"))?,
            level: level.unwrap_or_else(|| LitStr::new("trace", proc_macro2::Span::call_site())),
            setup,
            trace_setup,
            fields,
        })
    }
}

#[proc_macro_attribute]
pub fn instrumented(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as InstrumentedArgs);
    let mut function = parse_macro_input!(input as ItemFn);

    let span_macro = match args.level.value().as_str() {
        "trace" => format_ident!("trace_span"),
        "debug" => format_ident!("debug_span"),
        "info" => format_ident!("info_span"),
        "warn" => format_ident!("warn_span"),
        "error" => format_ident!("error_span"),
        other => {
            return syn::Error::new_spanned(args.level, format!("unsupported level '{other}'"))
                .to_compile_error()
                .into();
        }
    };

    let name = args.name;
    let setup = args.setup.map(|block| block.stmts).unwrap_or_default();
    let trace_setup = args
        .trace_setup
        .map(|block| block.stmts)
        .unwrap_or_default();
    let span_fields = match args.fields {
        Some(fields) => quote!(, #fields),
        None => TokenStream2::new(),
    };
    let body = function.block;

    function.block = Box::new(syn::parse_quote!({
        #(#setup)*
        #(
            #[cfg(feature = "tracing")]
            #trace_setup
        )*
        #[cfg(feature = "tracing")]
        let _instrumented_span = ::tracing::#span_macro!(
            target: crate::instrument::TARGET,
            #name,
            perfetto = true
            #span_fields
        )
        .entered();
        #body
    }));

    quote!(#function).into()
}
