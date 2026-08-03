use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use crate::traits::RuleLookupSupport;

pub trait ResolutionEnv: Default + Hash + Eq {
    type SharedState: Debug;
    type Query: Hash + Eq + Clone + Debug;
    type QueryResult: Hash + Eq + Clone + Debug;
    type DependencyEnvDeltaRequest;
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

    fn identity_dependency_env_delta() -> Self::DependencyEnvDeltaRequest;

    fn apply_dependency_env_delta(
        parent: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        requested: Self::DependencyEnvDeltaRequest,
    ) -> (Arc<Self>, Self::DependencyEnvDelta);

    fn pullback_lookup_support(
        support: &Self::LookupSupport,
        delta: &Self::DependencyEnvDelta,
    ) -> Self::LookupSupport;

    fn compose_dependency_env_delta(
        first: &Self::DependencyEnvDelta,
        second: &Self::DependencyEnvDelta,
    ) -> Self::DependencyEnvDelta;
}
