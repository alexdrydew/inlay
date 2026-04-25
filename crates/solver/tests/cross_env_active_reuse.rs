#![cfg(feature = "example")]

use context_solver::example::{
    ExampleEdgeKind, ExampleOutput, ExampleSystem, deferred_switch, definition, lazy, leaf, node,
};

#[test]
fn cross_env_block_created_by_unflagged_child_forces_ancestor_rerun() {
    // given
    let system = ExampleSystem::new([definition(
        "root",
        deferred_switch(node([lazy("root")]), leaf("deferred")),
    )]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    let deferred_root = match results.result(root) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 1);
            assert_eq!(edges[0].kind, ExampleEdgeKind::Lazy);
            assert_ne!(edges[0].target, root);
            edges[0].target
        }
        other => panic!("unexpected root result: {other:?}"),
    };

    assert_eq!(
        results.result(deferred_root),
        &Ok(ExampleOutput::Leaf("deferred".to_string()))
    );
}
