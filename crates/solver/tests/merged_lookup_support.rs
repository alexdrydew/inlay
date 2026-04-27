#![cfg(feature = "example")]

use context_solver::example::{
    ExampleOutput, ExampleRule, ExampleSharedState, ExampleState, definition, eager, leaf, node,
    scoped_eager,
};
use context_solver::solve::solve;

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
    let outcome = solve(
        &ExampleRule,
        "root".to_string(),
        ExampleState::Resolve,
        shared_state,
        8,
        64,
    );
    let support_validations = outcome.shared_state.support_validations;
    let (root, results) = outcome.result.expect("example solve should stabilize");

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
