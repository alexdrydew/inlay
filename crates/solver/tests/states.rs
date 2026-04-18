#![cfg(feature = "example")]

use context_solver::example::{
    ExampleDescription, ExampleOutput, ExampleState, ExampleSystem, definition, eager, lazy, leaf,
    match_first, node,
};

#[test]
fn describe_leaf_returns_leaf_description() {
    // given
    let system = ExampleSystem::new([definition("leaf", leaf("value"))]);

    // when
    let (results, root) = system
        .solve_in_state("leaf", ExampleState::Describe)
        .expect("example solve should stabilize");

    // then
    assert_eq!(
        results.result(root),
        &Ok(ExampleOutput::Description(ExampleDescription::Leaf {
            value: "value".to_string(),
        }))
    );
}

#[test]
fn describe_node_returns_shallow_metadata() {
    // given
    let system = ExampleSystem::new([
        definition("root", node([eager("left"), lazy("right")])),
        definition("left", leaf("left")),
        definition("right", leaf("right")),
    ]);

    // when
    let (results, root) = system
        .solve_in_state("root", ExampleState::Describe)
        .expect("example solve should stabilize");

    // then
    assert_eq!(
        results.result(root),
        &Ok(ExampleOutput::Description(ExampleDescription::Node {
            edge_count: 2,
            lazy_edge_count: 1,
        }))
    );
}

#[test]
fn describe_match_first_returns_shallow_metadata() {
    // given
    let system = ExampleSystem::new([
        definition("root", match_first(["left", "right"])),
        definition("left", leaf("left")),
        definition("right", leaf("right")),
    ]);

    // when
    let (results, root) = system
        .solve_in_state("root", ExampleState::Describe)
        .expect("example solve should stabilize");

    // then
    assert_eq!(
        results.result(root),
        &Ok(ExampleOutput::Description(ExampleDescription::MatchFirst {
            branch_count: 2,
        }))
    );
}
