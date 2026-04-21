#![cfg(feature = "example")]

use context_solver::example::{
    deferred_switch, definition, eager, lazy, leaf, match_first, node, scoped_eager, ExampleOutput,
    ExampleSystem,
};

#[test]
fn repeated_child_queries_reuse_result_refs() {
    // given
    let system = ExampleSystem::new([
        definition("root", node([eager("leaf"), eager("leaf")])),
        definition("leaf", leaf("value")),
    ]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    let (left, right) = match results.result(root) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 2);
            (edges[0].target, edges[1].target)
        }
        other => panic!("unexpected root result: {other:?}"),
    };

    assert_eq!(left, right);
    assert_eq!(
        results.result(left),
        &Ok(ExampleOutput::Leaf("value".to_string()))
    );
}

#[test]
fn frozen_cache_reuses_portable_result_across_env_and_lazy_depth() {
    // given
    let system = ExampleSystem::new([
        definition(
            "root",
            node([eager("leaf"), context_solver::example::lazy("leaf")]),
        ),
        definition("leaf", leaf("value")),
    ]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    let (first, second) = match results.result(root) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 2);
            (edges[0].target, edges[1].target)
        }
        other => panic!("unexpected root result: {other:?}"),
    };

    assert_eq!(first, second);
    assert_eq!(
        results.result(first),
        &Ok(ExampleOutput::Leaf("value".to_string()))
    );
}

#[test]
fn frozen_cache_does_not_reuse_parent_with_env_sensitive_child() {
    // given
    let system = ExampleSystem::new([
        definition(
            "root",
            node([
                context_solver::example::lazy("container"),
                eager("container"),
            ]),
        ),
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

#[test]
fn frozen_cache_reuses_transitioned_child_across_rebased_envs() {
    // given
    let system = ExampleSystem::new([
        definition(
            "root",
            node([
                scoped_eager("alpha", "container"),
                scoped_eager("beta", "container"),
            ]),
        ),
        definition("container", node([lazy("leaf")])),
        definition("leaf", deferred_switch(leaf("immediate"), leaf("deferred"))),
    ]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    let (alpha_container, beta_container) = match results.result(root) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 2);
            (edges[0].target, edges[1].target)
        }
        other => panic!("unexpected root result: {other:?}"),
    };

    assert_eq!(alpha_container, beta_container);

    let deferred_leaf = match results.result(alpha_container) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 1);
            edges[0].target
        }
        other => panic!("unexpected container result: {other:?}"),
    };

    assert_eq!(
        results.result(deferred_leaf),
        &Ok(ExampleOutput::Leaf("deferred".to_string()))
    );
}

#[test]
fn match_first_skips_failed_prefix_branch() {
    // given
    let system = ExampleSystem::new([
        definition("root", match_first(["left", "right"])),
        definition("left", node([eager("left")])),
        definition("right", leaf("value")),
    ]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    let right = match results.result(root) {
        Ok(ExampleOutput::Delegate(target)) => *target,
        other => panic!("unexpected root result: {other:?}"),
    };

    assert_eq!(
        results.result(right),
        &Ok(ExampleOutput::Leaf("value".to_string()))
    );
}

#[test]
fn cached_descendant_reuses_stale_active_backref_result() {
    // given
    let system = ExampleSystem::new([
        definition(
            "root",
            deferred_switch(
                match_first(["enter_root", "probe_direct"]),
                node([eager("probe"), eager("dead")]),
            ),
        ),
        definition("enter_root", node([lazy("root")])),
        definition("probe_direct", node([lazy("probe")])),
        definition("probe", match_first(["loop", "fallback"])),
        definition("loop", node([lazy("root")])),
        definition("fallback", leaf("fallback")),
        definition("dead", node([eager("dead")])),
    ]);

    // when
    let (results, root) = system
        .solve("root")
        .expect("example solve should stabilize");

    // then
    let probe_direct = match results.result(root) {
        Ok(ExampleOutput::Delegate(target)) => *target,
        other => panic!("unexpected root result: {other:?}"),
    };
    let probe = match results.result(probe_direct) {
        Ok(ExampleOutput::Node(edges)) => {
            assert_eq!(edges.len(), 1);
            edges[0].target
        }
        other => panic!("unexpected probe_direct result: {other:?}"),
    };
    let fallback = match results.result(probe) {
        Ok(ExampleOutput::Delegate(target)) => *target,
        other => panic!("unexpected probe result: {other:?}"),
    };

    assert_eq!(
        results.result(fallback),
        &Ok(ExampleOutput::Leaf("fallback".to_string()))
    );
}
