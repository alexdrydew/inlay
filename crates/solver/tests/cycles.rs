#![cfg(feature = "example")]

use context_solver::example::{
    definition, eager, lazy, node, ExampleEdgeKind, ExampleOutput, ExampleRuleError, ExampleSystem,
};

#[test]
fn eager_self_cycle_returns_inductive_cycle_error() {
    // given
    let system = ExampleSystem::new([definition("root", node([eager("root")]))]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    assert_eq!(results.result(root), &Err(ExampleRuleError::InductiveCycle));
}

#[test]
fn lazy_self_cycle_reuses_root_result_ref() {
    // given
    let system = ExampleSystem::new([definition("root", node([lazy("root")]))]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    match results.result(root) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 1);
            assert_eq!(edges[0].kind, ExampleEdgeKind::Lazy);
            assert_eq!(edges[0].target, root);
        }
        other => panic!("unexpected root result: {other:?}"),
    }
}

#[test]
fn eager_edges_preserve_lazy_depth_after_lazy_transition() {
    // given
    let system = ExampleSystem::new([
        definition("root", node([lazy("middle")])),
        definition("middle", node([eager("root")])),
    ]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    let middle = match results.result(root) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 1);
            edges[0].target
        }
        other => panic!("unexpected root result: {other:?}"),
    };

    match results.result(middle) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 1);
            assert_eq!(edges[0].kind, ExampleEdgeKind::Eager);
            assert_eq!(edges[0].target, root);
        }
        other => panic!("unexpected middle result: {other:?}"),
    }
}

#[test]
fn cycle_head_iteration_finds_root_cycle() {
    // given
    let system = ExampleSystem::new([
        definition("root", node([eager("middle")])),
        definition("middle", node([lazy("middle"), eager("root")])),
    ]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    assert_eq!(results.result(root), &Err(ExampleRuleError::InductiveCycle));
}

#[test]
fn cycle_head_iteration_finds_higher_ancestor_cycle() {
    // given
    let system = ExampleSystem::new([
        definition("root", node([eager("ancestor")])),
        definition("ancestor", node([eager("middle")])),
        definition("middle", node([lazy("middle"), eager("ancestor")])),
    ]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    assert_eq!(results.result(root), &Err(ExampleRuleError::InductiveCycle));
}

#[test]
fn fixpoint_iteration_limit_is_enforced() {
    // given
    let system = ExampleSystem::new([
        definition("root", node([eager("middle")])),
        definition("middle", node([lazy("middle"), eager("root")])),
    ])
    .with_fixpoint_iteration_limit(0);

    // when
    let result = system.solve("root");

    // then
    assert!(matches!(
        result,
        Err(context_solver::solve::SolveError::FixpointIterationLimitReached)
    ));
}

#[test]
fn stack_overflow_limit_is_enforced() {
    // given
    let system = ExampleSystem::new([
        definition("root", node([eager("a")])),
        definition("a", node([eager("b")])),
        definition("b", node([eager("c")])),
        definition("c", node([eager("d")])),
        definition("d", node([eager("leaf")])),
        definition("leaf", node([])),
    ])
    .with_stack_overflow_depth(3);

    // when
    let result = system.solve("root");

    // then
    assert!(matches!(
        result,
        Err(context_solver::solve::SolveError::StackOverflowDepthReached)
    ));
}
