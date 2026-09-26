//! Procedural macros for rig2: `#[tool]` and `#[derive(Embed)]`.
//!
//! Use them through `rig2-agent` (the tool attribute) and `rig2-core` (the
//! derive), or the `rig2` facade; the generated code finds whichever of those
//! crates the caller depends on.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::spanned::Spanned;
use syn::{
    Data, DeriveInput, Error, Expr, FnArg, ItemFn, Lit, Meta, Pat, ReturnType, Type,
    parse_macro_input,
};

/// The path to a rig2 crate as the caller names it: the facade module when
/// the caller depends on `rig2`, or the crate itself.
fn crate_path(krate: &str, facade_module: &str) -> TokenStream2 {
    use proc_macro_crate::{FoundCrate, crate_name};
    let facade_module = format_ident!("{facade_module}");
    match crate_name(krate) {
        // The crate itself, its tests and its doctests all see it by name:
        // each rig2 crate declares `extern crate self as <name>`.
        Ok(FoundCrate::Itself) => {
            let ident = format_ident!("{}", krate.replace('-', "_"));
            quote!(::#ident)
        }
        Ok(FoundCrate::Name(name)) => {
            let ident = format_ident!("{name}");
            quote!(::#ident)
        }
        Err(_) => match crate_name("rig2") {
            Ok(FoundCrate::Name(name)) => {
                let ident = format_ident!("{name}");
                quote!(::#ident::#facade_module)
            }
            _ => {
                let ident = format_ident!("{}", krate.replace('-', "_"));
                quote!(::#ident)
            }
        },
    }
}

/// The doc comment of an item, lines joined with spaces.
fn doc_string(attrs: &[syn::Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(expr) => match &expr.lit {
                    Lit::Str(s) => Some(s.value().trim().to_owned()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        })
        .collect();
    lines.join(" ").trim().to_owned()
}

fn to_pascal(name: &str) -> String {
    name.split('_')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let mut chars = p.chars();
            chars
                .next()
                .map(|c| c.to_ascii_uppercase().to_string() + chars.as_str())
                .unwrap_or_default()
        })
        .collect()
}

/// Whether a return type is spelled `Result<..>`.
fn returns_result(output: &ReturnType) -> bool {
    match output {
        ReturnType::Type(_, ty) => match ty.as_ref() {
            Type::Path(path) => path
                .path
                .segments
                .last()
                .is_some_and(|s| s.ident == "Result"),
            _ => false,
        },
        ReturnType::Default => false,
    }
}

/// Turn a function, async or not, into a tool.
///
/// The function keeps working as a function. Next to it, the attribute
/// generates a unit struct named after the function in `PascalCase`, which
/// implements `rig2_agent::tool::Tool`:
///
/// - the tool name is the function name;
/// - the description is the function's doc comment;
/// - the input schema is generated from the arguments with `schemars`;
///   describe an argument with `#[describe("...")]`;
/// - the output is the return value serialized as JSON, or text when it
///   serializes to a string. A function returning `Result<T, E>` reports
///   `Err` to the model as a failed tool call, using `E`'s `Display`.
///
/// Every argument must implement `Deserialize` and `JsonSchema`, and the
/// return value `Serialize`.
///
/// ```ignore
/// /// Add two numbers.
/// #[rig2_agent::tool]
/// async fn add(#[describe("the first number")] a: f64, b: f64) -> f64 {
///     a + b
/// }
/// // `Add` now implements `Tool`.
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return Error::new(Span::call_site(), "`#[tool]` takes no arguments")
            .to_compile_error()
            .into();
    }
    let function = parse_macro_input!(item as ItemFn);
    match expand_tool(function) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand_tool(mut function: ItemFn) -> syn::Result<TokenStream2> {
    let awaited = if function.sig.asyncness.is_some() {
        quote!(.await)
    } else {
        quote!()
    };
    let agent = crate_path("rig2-agent", "agent");
    let name = function.sig.ident.clone();
    let name_str = name.to_string();
    let struct_name = format_ident!("{}", to_pascal(&name_str));
    let args_name = format_ident!("__{}Arguments", to_pascal(&name_str));
    let description = doc_string(&function.attrs);
    let visibility = function.vis.clone();

    let mut fields = Vec::new();
    let mut arg_names = Vec::new();
    for input in &mut function.sig.inputs {
        let FnArg::Typed(arg) = input else {
            return Err(Error::new(input.span(), "a tool cannot take `self`"));
        };
        let Pat::Ident(ident) = arg.pat.as_ref() else {
            return Err(Error::new(
                arg.pat.span(),
                "tool arguments must be plain identifiers",
            ));
        };
        let ident = ident.ident.clone();
        let ty = arg.ty.clone();
        let mut describe = None;
        let mut kept = Vec::new();
        for attr in std::mem::take(&mut arg.attrs) {
            if attr.path().is_ident("describe") {
                let text: syn::LitStr = attr.parse_args()?;
                describe = Some(text.value());
            } else {
                kept.push(attr);
            }
        }
        arg.attrs = kept;
        let doc = describe.map(|d| quote!(#[doc = #d]));
        fields.push(quote! { #doc #ident: #ty });
        arg_names.push(ident);
    }

    let call = if returns_result(&function.sig.output) {
        quote! { #name(#(arguments.#arg_names),*)#awaited.map_err(#agent::__private::tool_failure)? }
    } else {
        quote! { #name(#(arguments.#arg_names),*)#awaited }
    };
    let struct_doc = format!("The `{name_str}` tool, generated by `#[tool]`.");
    let serde_path = format!("{agent}::__private::serde");
    let schemars_path = format!("{agent}::__private::schemars");

    Ok(quote! {
        // A tool receives its arguments by value, deserialized for the call.
        #[allow(clippy::needless_pass_by_value)]
        #function

        #[doc = #struct_doc]
        #[derive(Debug, Clone, Copy, Default)]
        #visibility struct #struct_name;

        #[derive(#agent::__private::serde::Deserialize, #agent::__private::schemars::JsonSchema)]
        #[serde(crate = #serde_path)]
        #[schemars(crate = #schemars_path)]
        #[allow(non_camel_case_types)]
        struct #args_name { #(#fields),* }

        impl #agent::tool::Tool for #struct_name {
            fn definition(&self) -> #agent::__private::ToolDefinition {
                #agent::__private::ToolDefinition {
                    name: #name_str.to_owned(),
                    description: #description.to_owned(),
                    parameters: #agent::__private::schema_of::<#args_name>(),
                }
            }

            fn call(
                &self,
                arguments: #agent::__private::serde_json::Value,
                _context: #agent::tool::ToolContext,
            ) -> #agent::__private::BoxFuture<'static, #agent::__private::Result<::std::vec::Vec<#agent::__private::ToolOutput>>> {
                ::std::boxed::Box::pin(async move {
                    let arguments: #args_name = #agent::__private::parse_arguments(#name_str, arguments)?;
                    let output = #call;
                    #agent::__private::to_output(&output)
                })
            }
        }
    })
}

/// Derive `rig2_core::store::Embed`: the text to embed is the `#[embed]`
/// fields, in declaration order, joined with newlines.
///
/// Each `#[embed]` field must implement `Display`. A struct with no
/// `#[embed]` field is an error.
///
/// ```ignore
/// #[derive(rig2_core::Embed, serde::Serialize)]
/// struct Article { #[embed] title: String, #[embed] body: String, year: u32 }
/// ```
#[proc_macro_derive(Embed, attributes(embed))]
pub fn derive_embed(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_embed(&input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand_embed(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let core = crate_path("rig2-core", "core");
    let Data::Struct(data) = &input.data else {
        return Err(Error::new(
            input.ident.span(),
            "`Embed` can only be derived for structs",
        ));
    };
    let embedded: Vec<TokenStream2> = data
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| f.attrs.iter().any(|a| a.path().is_ident("embed")))
        .map(|(i, f)| match &f.ident {
            Some(ident) => quote!(self.#ident.to_string()),
            None => {
                let index = syn::Index::from(i);
                quote!(self.#index.to_string())
            }
        })
        .collect();
    if embedded.is_empty() {
        return Err(Error::new(
            input.ident.span(),
            "mark at least one field with `#[embed]`",
        ));
    }
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics #core::store::Embed for #name #ty_generics #where_clause {
            fn embed_text(&self) -> ::std::string::String {
                [#(#embedded),*].join("\n")
            }
        }
    })
}
