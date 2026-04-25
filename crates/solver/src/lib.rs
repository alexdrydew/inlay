mod cache;
mod context;
#[cfg(any(test, feature = "example"))]
pub mod example;
mod instrument;
mod lookup_support;
pub mod rule;
mod search_graph;
pub mod solve;
mod stack;
pub mod traits;

pub use rule::{LazyDepthMode, RuleContext, RunError};
pub use traits::{Arena, ReplaceError, ResolutionEnv, Rule, RuleLookupSupport};
