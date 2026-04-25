#![cfg(feature = "example")]

use context_solver::example::{
    ExampleOutput, ExampleSystem, deferred_switch, definition, eager, lazy, leaf, node,
};

#[test]
fn flattened_support_preserves_transitive_lookup_requirements() {
    // given
    let system = ExampleSystem::new([
        definition("root", node([lazy("container"), eager("container")])),
        definition("container", node([eager("leaf")])),
        definition("leaf", deferred_switch(leaf("immediate"), leaf("deferred"))),
    ]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    let (deferred_container, immediate_container) = match results.result(root) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 2);
            (edges[0].target, edges[1].target)
        }
        other => panic!("unexpected root result: {other:?}"),
    };
    assert_ne!(deferred_container, immediate_container);

    let deferred_leaf = match results.result(deferred_container) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 1);
            edges[0].target
        }
        other => panic!("unexpected deferred container result: {other:?}"),
    };
    let immediate_leaf = match results.result(immediate_container) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 1);
            edges[0].target
        }
        other => panic!("unexpected immediate container result: {other:?}"),
    };

    assert_eq!(
        results.result(deferred_leaf),
        &Ok(ExampleOutput::Leaf("deferred".to_string()))
    );
    assert_eq!(
        results.result(immediate_leaf),
        &Ok(ExampleOutput::Leaf("immediate".to_string()))
    );
}
