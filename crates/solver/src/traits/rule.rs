use std::fmt::Debug;
use std::hash::Hash;

use crate::rule::{RuleContext, RunError};
use crate::traits::{Arena, ResolutionEnv};

pub trait Rule: Sized + Debug {
    type Query: Hash + Eq + Clone + Debug;
    type Output: Hash + Eq + Clone;
    type Err: Hash + Eq + Clone + std::error::Error;
    type Env: ResolutionEnv + Debug;
    type ResultsArena: Arena<Result<Self::Output, Self::Err>> + Default;
    type RuleStateId: Hash + Eq + Copy + Debug;

    fn run(
        &self,
        query: Self::Query,
        ctx: &mut RuleContext<Self>,
    ) -> Result<Self::Output, RunError<Self>>;

}
