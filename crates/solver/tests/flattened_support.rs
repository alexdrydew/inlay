use std::error::Error;
use std::fmt;
use std::sync::Arc;

use context_solver::{Arena, LazyDepthMode, ReplaceError, ResolutionEnv, Rule, RuleContext, RunError};
use context_solver::solve::{SolveResult, solve};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Query {
    Root,
    Container,
    Leaf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Lookup {
    Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Output {
    Pair(ResultRef, ResultRef),
    Container(ResultRef),
    Leaf(u8),
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
struct SharedState;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Env {
    value: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EnvDelta {
    value: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LookupSupport {
    value: u8,
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
            Lookup::Value => self.value,
        }
    }

    fn lookup_support(
        self: &Arc<Self>,
        _shared_state: &mut Self::SharedState,
        query: &Self::Query,
        result: &Self::QueryResult,
    ) -> Self::LookupSupport {
        match query {
            Lookup::Value => LookupSupport { value: *result },
        }
    }

    fn lookup_support_matches(
        self: &Arc<Self>,
        candidate: &Arc<Self>,
        _shared_state: &mut Self::SharedState,
        support: &Self::LookupSupport,
    ) -> bool {
        let _ = self;
        candidate.value == support.value
    }

    fn dependency_env_delta(_parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta {
        EnvDelta {
            value: (child.value != _parent.value).then_some(child.value),
        }
    }

    fn compose_dependency_env_delta(
        first: &Self::DependencyEnvDelta,
        second: &Self::DependencyEnvDelta,
    ) -> Self::DependencyEnvDelta {
        EnvDelta {
            value: second.value.or(first.value),
        }
    }

    fn apply_dependency_env_delta(
        _parent: &Arc<Self>,
        delta: &Self::DependencyEnvDelta,
    ) -> Arc<Self> {
        delta
            .value
            .map(|value| Arc::new(Env { value }))
            .unwrap_or_else(|| Arc::clone(_parent))
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

fn result_ref(result: SolveResult<'_, TestRule>) -> ResultRef {
    match result {
        SolveResult::Resolved { result_ref, .. } | SolveResult::Lazy { result_ref } => result_ref,
    }
}

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
            Query::Root => Ok(Output::Pair(
                result_ref(ctx.solve(
                    Query::Container,
                    (),
                    LazyDepthMode::Keep,
                    Arc::new(Env { value: 1 }),
                )?),
                result_ref(ctx.solve(
                    Query::Container,
                    (),
                    LazyDepthMode::Keep,
                    Arc::new(Env { value: 2 }),
                )?),
            )),
            Query::Container => Ok(Output::Container(result_ref(ctx.solve(
                Query::Leaf,
                (),
                LazyDepthMode::Keep,
                Arc::new(ctx.env().clone()),
            )?))),
            Query::Leaf => Ok(Output::Leaf(ctx.lookup(&Lookup::Value))),
        }
    }
}

#[test]
fn flattened_support_preserves_transitive_lookup_requirements() {
    // given
    let rule = TestRule;

    // when
    let (root, results) = solve(
        &rule,
        Query::Root,
        (),
        Arc::new(Env { value: 0 }),
        SharedState,
        8,
        64,
    )
    .result
    .expect("solver should stabilize");

    // then
    let Ok(Output::Pair(first_container, second_container)) = results.result(root) else {
        panic!("unexpected root result: {:?}", results.result(root));
    };
    assert_ne!(first_container, second_container);

    let Ok(Output::Container(first_leaf)) = results.result(*first_container) else {
        panic!(
            "unexpected first container result: {:?}",
            results.result(*first_container)
        );
    };
    let Ok(Output::Container(second_leaf)) = results.result(*second_container) else {
        panic!(
            "unexpected second container result: {:?}",
            results.result(*second_container)
        );
    };
    assert_eq!(results.result(*first_leaf), &Ok(Output::Leaf(1)));
    assert_eq!(results.result(*second_leaf), &Ok(Output::Leaf(2)));
}
