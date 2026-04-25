use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use context_solver::{Arena, LazyDepthMode, ReplaceError, ResolutionEnv, Rule, RuleContext, RunError};
use context_solver::solve::{SolveResult, solve};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Query {
    Root,
    Leaf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Lookup {
    A,
    B,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Output {
    Pair(ResultRef, ResultRef),
    Leaf(u8, u8),
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
    support_validations: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Env {
    id: u8,
    values: BTreeMap<Lookup, u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct EnvDelta {
    values: Option<BTreeMap<Lookup, u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LookupSupport {
    values: BTreeMap<Lookup, u8>,
}

impl LookupSupport {
    fn new(query: Lookup, result: u8) -> Self {
        Self {
            values: [(query, result)].into(),
        }
    }
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
        *self.values.get(query).expect("test lookup must exist")
    }

    fn lookup_support(
        self: &Arc<Self>,
        _shared_state: &mut Self::SharedState,
        query: &Self::Query,
        result: &Self::QueryResult,
    ) -> Self::LookupSupport {
        LookupSupport::new(*query, *result)
    }

    fn lookup_support_matches(
        self: &Arc<Self>,
        candidate: &Arc<Self>,
        shared_state: &mut Self::SharedState,
        support: &Self::LookupSupport,
    ) -> bool {
        let _ = self;
        shared_state.support_validations += 1;
        support
            .values
            .iter()
            .all(|(query, expected)| candidate.values.get(query) == Some(expected))
    }

    fn merge_lookup_support(
        left: &Self::LookupSupport,
        right: &Self::LookupSupport,
    ) -> Option<Self::LookupSupport> {
        let mut values = left.values.clone();
        for (query, result) in &right.values {
            match values.insert(*query, *result) {
                Some(existing) if existing != *result => return None,
                _ => {}
            }
        }
        Some(LookupSupport { values })
    }

    fn dependency_env_delta(parent: &Arc<Self>, child: &Arc<Self>) -> Self::DependencyEnvDelta {
        EnvDelta {
            values: (child.values != parent.values).then(|| child.values.clone()),
        }
    }

    fn compose_dependency_env_delta(
        first: &Self::DependencyEnvDelta,
        second: &Self::DependencyEnvDelta,
    ) -> Self::DependencyEnvDelta {
        EnvDelta {
            values: second.values.clone().or_else(|| first.values.clone()),
        }
    }

    fn apply_dependency_env_delta(
        parent: &Arc<Self>,
        delta: &Self::DependencyEnvDelta,
    ) -> Arc<Self> {
        delta
            .values
            .clone()
            .map(|values| Arc::new(Env { id: parent.id, values }))
            .unwrap_or_else(|| Arc::clone(parent))
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

fn values() -> BTreeMap<Lookup, u8> {
    [(Lookup::A, 1), (Lookup::B, 2)].into()
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
                    Query::Leaf,
                    (),
                    LazyDepthMode::Keep,
                    Arc::new(Env {
                        id: 1,
                        values: values(),
                    }),
                )?),
                result_ref(ctx.solve(
                    Query::Leaf,
                    (),
                    LazyDepthMode::Keep,
                    Arc::new(Env {
                        id: 2,
                        values: values(),
                    }),
                )?),
            )),
            Query::Leaf => Ok(Output::Leaf(
                ctx.lookup(&Lookup::A),
                ctx.lookup(&Lookup::B),
            )),
        }
    }
}

#[test]
fn merged_lookup_support_validates_once_for_reused_answer() {
    // given
    let rule = TestRule;

    // when
    let outcome = solve(
        &rule,
        Query::Root,
        (),
        Arc::new(Env {
            id: 0,
            values: BTreeMap::new(),
        }),
        SharedState::default(),
        8,
        64,
    );
    let support_validations = outcome.shared_state.support_validations;
    let (root, results) = outcome.result.expect("solver should stabilize");

    // then
    let Ok(Output::Pair(first_leaf, second_leaf)) = results.result(root) else {
        panic!("unexpected root result: {:?}", results.result(root));
    };
    assert_eq!(first_leaf, second_leaf);
    assert_eq!(results.result(*first_leaf), &Ok(Output::Leaf(1, 2)));
    assert_eq!(support_validations, 1);
}
