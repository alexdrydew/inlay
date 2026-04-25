use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use crate::traits::RuleLookupSupport;

pub trait ResolutionEnv: Hash + Eq {
    type SharedState: Debug;
    type Query: Hash + Eq + Clone + Debug;
    type QueryResult: Hash + Eq + Clone + Debug;
    type DependencyEnvDelta: Hash + Eq + Clone + Debug;
    type LookupSupport: RuleLookupSupport;

    fn lookup(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
    ) -> Self::QueryResult;

    fn lookup_support(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        query: &Self::Query,
        result: &Self::QueryResult,
    ) -> Self::LookupSupport;

    fn lookup_support_matches(
        self: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        support: &Self::LookupSupport,
    ) -> bool;

    fn pullback_lookup_support(
        support: &Self::LookupSupport,
        delta: &Self::DependencyEnvDelta,
    ) -> Option<Self::LookupSupport>;

    fn dependency_env_delta(parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta;

    fn compose_dependency_env_delta(
        first: &Self::DependencyEnvDelta,
        second: &Self::DependencyEnvDelta,
    ) -> Self::DependencyEnvDelta;

    fn apply_dependency_env_delta(
        parent: &Arc<Self>,
        delta: &Self::DependencyEnvDelta,
    ) -> Arc<Self>;

    fn env_item_count(env: &Self) -> usize;

    fn dependency_env_delta_item_count(delta: &Self::DependencyEnvDelta) -> usize;
}
