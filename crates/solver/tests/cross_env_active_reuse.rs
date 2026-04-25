#![cfg(feature = "cross-env-active-reuse")]

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use context_solver::{Arena, LazyDepthMode, ReplaceError, ResolutionEnv, Rule, RuleContext, RunError};
use context_solver::solve::{SolveResult, solve};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Query {
    Root,
    Probe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Lookup {
    Variant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Output {
    Pending,
    Done,
    Probe,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TestError;

impl fmt::Display for TestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("test error")
    }
}

impl Error for TestError {}

#[derive(Debug, Default)]
struct SharedState {
    root_runs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Env {
    variant: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EnvDelta {
    variant: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LookupSupport {
    variant: u8,
}

impl ResolutionEnv for Env {
    type SharedState = SharedState;
    type Query = Lookup;
    type QueryResult = u8;
    type DependencyEnvDelta = EnvDelta;
    type LookupSupport = LookupSupport;

    fn lookup(
        self: &Arc<Self>,
        _shared_state: &mut Self::SharedState,
        query: &Self::Query,
    ) -> Self::QueryResult {
        match query {
            Lookup::Variant => self.variant,
        }
    }

    fn lookup_support(
        self: &Arc<Self>,
        _shared_state: &mut Self::SharedState,
        query: &Self::Query,
        result: &Self::QueryResult,
    ) -> Self::LookupSupport {
        match query {
            Lookup::Variant => LookupSupport { variant: *result },
        }
    }

    fn lookup_support_matches(
        self: &Arc<Self>,
        candidate: &Arc<Self>,
        _shared_state: &mut Self::SharedState,
        support: &Self::LookupSupport,
    ) -> bool {
        let _ = self;
        candidate.variant == support.variant
    }

    fn dependency_env_delta(_parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta {
        EnvDelta {
            variant: child.variant,
        }
    }

    fn compose_dependency_env_delta(
        _first: &Self::DependencyEnvDelta,
        second: &Self::DependencyEnvDelta,
    ) -> Self::DependencyEnvDelta {
        second.clone()
    }

    fn apply_dependency_env_delta(
        _parent: &Arc<Self>,
        delta: &Self::DependencyEnvDelta,
    ) -> Arc<Self> {
        Arc::new(Env {
            variant: delta.variant,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ResultRef(usize);

type TestResult = Result<Output, TestError>;

#[derive(Debug, Default)]
struct ResultsArena {
    results: Vec<Option<TestResult>>,
}

impl ResultsArena {
    fn result(&self, key: ResultRef) -> &TestResult {
        self.results[key.0]
            .as_ref()
            .expect("test result ref must point to a stored result")
    }
}

impl Arena<TestResult> for ResultsArena {
    type Key = ResultRef;

    fn insert(&mut self, val: TestResult) -> Self::Key {
        let key = ResultRef(self.results.len());
        self.results.push(Some(val));
        key
    }

    fn insert_placeholder(&mut self) -> Self::Key {
        let key = ResultRef(self.results.len());
        self.results.push(None);
        key
    }

    fn replace(
        &mut self,
        key: Self::Key,
        val: TestResult,
    ) -> Result<Option<TestResult>, ReplaceError> {
        self.results
            .get_mut(key.0)
            .ok_or(ReplaceError::InvalidKey)
            .map(|slot| slot.replace(val))
    }

    fn get(&self, key: &Self::Key) -> Option<&TestResult> {
        self.results.get(key.0)?.as_ref()
    }

    fn len(&self) -> usize {
        self.results.len()
    }
}

#[derive(Debug)]
struct TestRule;

impl Rule for TestRule {
    type Query = Query;
    type Output = Output;
    type Err = TestError;
    type Env = Env;
    type ResultsArena = ResultsArena;
    type RuleStateId = ();

    fn run(
        &self,
        query: Query,
        ctx: &mut RuleContext<Self>,
    ) -> Result<Self::Output, RunError<Self>> {
        match query {
            Query::Root => {
                ctx.lookup(&Lookup::Variant);
                let root_runs = {
                    let shared = ctx.shared();
                    shared.root_runs += 1;
                    shared.root_runs
                };

                if root_runs <= 2 {
                    let env = Arc::new(ctx.env().clone());
                    let _ = ctx.solve(Query::Probe, (), LazyDepthMode::Keep, env)?;
                    Ok(Output::Pending)
                } else {
                    Ok(Output::Done)
                }
            }
            Query::Probe => {
                let variant = match ctx.shared().root_runs {
                    1 => 1,
                    2 => 2,
                    _ => 2,
                };
                let env = Arc::new(Env { variant });
                match ctx.solve(Query::Root, (), LazyDepthMode::Increment, env)? {
                    SolveResult::Resolved { .. } => unreachable!(
                        "cross-env active root should resolve lazily while root is on the stack"
                    ),
                    SolveResult::Lazy { .. } => Ok(Output::Probe),
                }
            }
        }
    }
}

#[test]
fn cross_env_block_created_by_unflagged_child_forces_ancestor_rerun() {
    let outcome = solve(
        &TestRule,
        Query::Root,
        (),
        Arc::new(Env { variant: 0 }),
        SharedState::default(),
        8,
        64,
    );
    let root_runs = outcome.shared_state.root_runs;
    let (root, results) = outcome.result.expect("solver should stabilize");

    assert_eq!(root_runs, 3);
    assert_eq!(results.result(root), &Ok(Output::Done));
}
