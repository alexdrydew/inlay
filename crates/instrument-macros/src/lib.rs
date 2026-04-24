use proc_macro::TokenStream;

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;

#[proc_macro_attribute]
pub fn instrumented(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = TokenStream2::from(args);
    let input = TokenStream2::from(input);

    quote! {
        #[cfg_attr(feature = "tracing", ::tracing::instrument(#args))]
        #input
    }
    .into()
}
