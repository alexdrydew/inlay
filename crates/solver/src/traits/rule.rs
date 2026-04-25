use std::fmt::Debug;
use std::hash::Hash;

use crate::rule::{RuleContext, RunError};
use crate::traits::{Arena, ResolutionEnv};

pub trait Rule: Sized + Debug {
    type Query: Hash + Eq + Clone + Debug;
    type Output: 'static + Hash + Eq + Clone;
    type Err: 'static + Hash + Eq + Clone + std::error::Error;
    type Env: ResolutionEnv + Debug;
    type ResultsArena: Arena<Result<Self::Output, Self::Err>> + Default;
    type RuleStateId: Hash + Eq + Copy + Debug;

    fn run(
        &self,
        query: Self::Query,
        ctx: &mut RuleContext<Self>,
    ) -> Result<Self::Output, RunError<Self>>;

    fn debug_query_label(
        &self,
        _query: &Self::Query,
        _state_id: Self::RuleStateId,
    ) -> Option<String> {
        None
    }

    fn debug_env_label(&self, _env: &Self::Env) -> Option<String> {
        None
    }

    fn debug_lookup_query_label(
        &self,
        _query: &<Self::Env as ResolutionEnv>::Query,
    ) -> Option<String> {
        None
    }

    fn debug_lookup_result_label(
        &self,
        _result: &<Self::Env as ResolutionEnv>::QueryResult,
    ) -> Option<String> {
        None
    }
}
