use proc_macro::TokenStream;

use proc_macro2::{Delimiter, Group, TokenStream as TokenStream2, TokenTree};
use quote::quote;

fn args_with_perfetto_field(args: TokenStream2) -> TokenStream2 {
    let mut output = TokenStream2::new();
    let mut tokens = args.into_iter().peekable();
    let mut has_fields = false;

    while let Some(token) = tokens.next() {
        if let TokenTree::Ident(ident) = &token {
            if ident == "fields" {
                if let Some(TokenTree::Group(group)) = tokens.peek() {
                    if group.delimiter() == Delimiter::Parenthesis {
                        let span = ident.span();
                        output.extend([token]);
                        let TokenTree::Group(group) = tokens.next().expect("peeked group") else {
                            unreachable!();
                        };
                        let fields = group.stream();
                        let fields = if fields.is_empty() {
                            quote!(perfetto = true)
                        } else {
                            quote!(perfetto = true, #fields)
                        };
                        let mut group = Group::new(Delimiter::Parenthesis, fields);
                        group.set_span(span);
                        output.extend([TokenTree::Group(group)]);
                        has_fields = true;
                        continue;
                    }
                }
            }
        }

        output.extend([token]);
    }

    if !has_fields {
        if !output.is_empty() {
            output.extend(quote!(,));
        }
        output.extend(quote!(fields(perfetto = true)));
    }

    output
}

#[proc_macro_attribute]
pub fn instrumented(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = args_with_perfetto_field(TokenStream2::from(args));
    let input = TokenStream2::from(input);

    quote! {
        #[cfg_attr(feature = "tracing", ::tracing::instrument(#args))]
        #input
    }
    .into()
}
