#![cfg(feature = "example")]

use context_solver::example::{
    ExampleOutput, ExampleRule, ExampleSharedState, ExampleState, definition, eager, leaf, node,
    scoped_eager,
};
use context_solver::solve::Solver;

#[test]
fn merged_lookup_support_validates_once_for_reused_answer() {
    // given
    let shared_state = ExampleSharedState::new([
        definition(
            "root",
            node([
                scoped_eager("alpha", "container"),
                scoped_eager("beta", "container"),
            ]),
        ),
        definition("container", node([eager("left"), eager("right")])),
        definition("left", leaf("left")),
        definition("right", leaf("right")),
    ]);

    // when
    let mut solver = Solver::new(ExampleRule, shared_state, 8, 64);
    let root = solver
        .solve("root".to_string(), ExampleState::Resolve)
        .expect("example solve should stabilize");
    let support_validations = solver.shared_state().support_validations;
    let (_, _, results) = solver.into_parts();

    // then
    let (first_container, second_container) = match results.result(root) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 2);
            (edges[0].target, edges[1].target)
        }
        other => panic!("unexpected root result: {other:?}"),
    };
    assert_eq!(first_container, second_container);

    let (left, right) = match results.result(first_container) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 2);
            (edges[0].target, edges[1].target)
        }
        other => panic!("unexpected container result: {other:?}"),
    };
    assert_eq!(
        results.result(left),
        &Ok(ExampleOutput::Leaf("left".to_string()))
    );
    assert_eq!(
        results.result(right),
        &Ok(ExampleOutput::Leaf("right".to_string()))
    );
    assert_eq!(support_validations, 1);
}
