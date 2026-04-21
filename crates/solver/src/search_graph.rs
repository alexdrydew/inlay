use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    ops::{Add, Index, IndexMut},
    sync::Arc,
};

use crate::{
    rule::{Lookups, ResolutionEnv, Rule, RuleEnv, RuleQuery, RuleResultRef},
    stack::StackDepth,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct LazyDepth(pub usize);

pub(crate) type ActiveBackrefKey<R> = (RuleQuery<R>, <R as Rule>::RuleStateId, Arc<RuleEnv<R>>);
pub(crate) type CrossEnvBackrefKey<R> = (RuleQuery<R>, <R as Rule>::RuleStateId);
pub(crate) type CacheKey<R> = (RuleQuery<R>, <R as Rule>::RuleStateId);

pub(crate) struct CacheEntry<R: Rule> {
    pub(crate) env: Arc<RuleEnv<R>>,
    pub(crate) result_ref: RuleResultRef<R>,
    pub(crate) fingerprint: u64,
}

#[derive(Clone)]
pub(crate) struct CacheBucket<R: Rule> {
    entries: Vec<CacheEntry<R>>,
    by_env: HashMap<Arc<RuleEnv<R>>, Vec<usize>>,
    by_env_fingerprint: HashMap<(Arc<RuleEnv<R>>, u64), Vec<usize>>,
}

pub(crate) struct Dependency<R: Rule> {
    pub(crate) result_ref: RuleResultRef<R>,
    pub(crate) env_delta: <RuleEnv<R> as ResolutionEnv>::DependencyEnvDelta,
}

pub(crate) struct GoalKey<R: Rule> {
    pub(crate) query: RuleQuery<R>,
    pub(crate) state_id: R::RuleStateId,
    pub(crate) env: Arc<RuleEnv<R>>,
    pub(crate) lazy_depth: LazyDepth,
}

pub(crate) struct Answer<R: Rule> {
    pub(crate) result_ref: RuleResultRef<R>,
    pub(crate) lookups: Lookups<R>,
    pub(crate) dependencies: Vec<Dependency<R>>,
}

pub(crate) struct Node<R: Rule> {
    pub(crate) goal: GoalKey<R>,
    pub(crate) answer: Answer<R>,
    pub(crate) cross_env_reuses: Vec<(RuleResultRef<R>, Arc<RuleEnv<R>>)>,
    pub(crate) stack_depth: Option<StackDepth>,
    pub(crate) links: Minimums,
}

pub(crate) struct SearchGraph<R: Rule> {
    indices: HashMap<GoalKey<R>, DepthFirstNumber>,
    closest_goals: HashMap<ActiveBackrefKey<R>, Vec<DepthFirstNumber>>,
    closest_goals_any_env: HashMap<CrossEnvBackrefKey<R>, Vec<DepthFirstNumber>>,
    nodes: Vec<Node<R>>,
}

#[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub(crate) struct DepthFirstNumber {
    index: usize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Minimums {
    ancestor: DepthFirstNumber,
}

impl Minimums {
    pub(crate) fn new() -> Self {
        Self {
            ancestor: DepthFirstNumber::MAX,
        }
    }

    pub(crate) fn from_self(dfn: DepthFirstNumber) -> Self {
        Self { ancestor: dfn }
    }

    pub(crate) fn update_from_dfn(&mut self, dfn: DepthFirstNumber) {
        self.ancestor = self.ancestor.min(dfn);
    }

    pub(crate) fn update_from(&mut self, other: Minimums) {
        self.update_from_dfn(other.ancestor);
    }

    pub(crate) fn ancestor(self) -> DepthFirstNumber {
        self.ancestor
    }
}

impl<R: Rule> SearchGraph<R> {
    pub(crate) fn new() -> Self {
        Self {
            indices: HashMap::new(),
            closest_goals: HashMap::new(),
            closest_goals_any_env: HashMap::new(),
            nodes: vec![],
        }
    }

    pub(crate) fn lookup(&self, goal: &GoalKey<R>) -> Option<DepthFirstNumber> {
        self.indices.get(goal).copied()
    }

    // For live lower-stratum backreference reuse it is enough to consider only the nearest
    // matching active goal for an exact `(query, state_id, env)` triple.
    //
    // Once the env is part of the active backreference key, farther matching ancestors cannot
    // expose any additional same-env reuse opportunities: the nearest one is the only candidate
    // whose current approximation can be observed by the caller.
    pub(crate) fn closest_goal(
        &self,
        query: &RuleQuery<R>,
        state_id: R::RuleStateId,
        env: &Arc<RuleEnv<R>>,
    ) -> Option<DepthFirstNumber> {
        self.closest_goals
            .get(&(query.clone(), state_id, Arc::clone(env)))
            .and_then(|stack| stack.last().copied())
    }

    pub(crate) fn closest_goal_any_env(
        &self,
        query: &RuleQuery<R>,
        state_id: R::RuleStateId,
    ) -> Option<DepthFirstNumber> {
        self.closest_goals_any_env
            .get(&(query.clone(), state_id))
            .and_then(|stack| stack.last().copied())
    }

    pub(crate) fn insert(
        &mut self,
        goal: &GoalKey<R>,
        stack_depth: StackDepth,
        result_ref: RuleResultRef<R>,
    ) -> DepthFirstNumber {
        let dfn = DepthFirstNumber {
            index: self.nodes.len(),
        };
        self.nodes.push(Node {
            goal: goal.clone(),
            answer: Answer {
                result_ref,
                lookups: vec![],
                dependencies: vec![],
            },
            cross_env_reuses: vec![],
            stack_depth: Some(stack_depth),
            links: Minimums::from_self(dfn),
        });
        let previous = self.indices.insert(goal.clone(), dfn);
        assert!(previous.is_none(), "active goals must be unique");
        self.closest_goals
            .entry((goal.query.clone(), goal.state_id, Arc::clone(&goal.env)))
            .or_default()
            .push(dfn);
        self.closest_goals_any_env
            .entry((goal.query.clone(), goal.state_id))
            .or_default()
            .push(dfn);
        dfn
    }

    pub(crate) fn pop_stack_goal(&mut self, dfn: DepthFirstNumber) {
        let node = &mut self[dfn];
        let key = (
            node.goal.query.clone(),
            node.goal.state_id,
            Arc::clone(&node.goal.env),
        );
        let any_env_key = (node.goal.query.clone(), node.goal.state_id);
        node.stack_depth = None;

        let stack = self
            .closest_goals
            .get_mut(&key)
            .expect("stack goal must exist in closest_goals");
        assert_eq!(stack.pop(), Some(dfn));
        if stack.is_empty() {
            self.closest_goals.remove(&key);
        }

        let any_env_stack = self
            .closest_goals_any_env
            .get_mut(&any_env_key)
            .expect("stack goal must exist in closest_goals_any_env");
        assert_eq!(any_env_stack.pop(), Some(dfn));
        if any_env_stack.is_empty() {
            self.closest_goals_any_env.remove(&any_env_key);
        }
    }

    pub(crate) fn rollback_to(&mut self, dfn: DepthFirstNumber) {
        self.indices.retain(|_, value| *value < dfn);
        for node in &self.nodes[dfn.index..] {
            assert!(
                node.stack_depth.is_none(),
                "only popped nodes may be rolled back"
            );
        }
        self.nodes.truncate(dfn.index);
    }

    pub(crate) fn suffix_result_refs(&self, dfn: DepthFirstNumber) -> Vec<RuleResultRef<R>> {
        self.nodes[dfn.index..]
            .iter()
            .map(|node| node.answer.result_ref)
            .collect()
    }

    pub(crate) fn suffix_cross_env_reuses(
        &self,
        dfn: DepthFirstNumber,
    ) -> Vec<(RuleResultRef<R>, Arc<RuleEnv<R>>)> {
        self.nodes[dfn.index..]
            .iter()
            .flat_map(|node| node.cross_env_reuses.iter().cloned())
            .collect()
    }

    pub(crate) fn take_cacheable_entries(
        &mut self,
        dfn: DepthFirstNumber,
    ) -> Vec<(CacheKey<R>, Arc<RuleEnv<R>>, RuleResultRef<R>)> {
        self.indices.retain(|_, value| *value < dfn);
        let mut cacheable = vec![];
        for (offset, node) in self.nodes.drain(dfn.index..).enumerate() {
            assert!(node.stack_depth.is_none(), "cached nodes must be popped");
            let node_dfn = DepthFirstNumber {
                index: dfn.index + offset,
            };
            if node.links.ancestor() >= node_dfn {
                cacheable.push((
                    (node.goal.query, node.goal.state_id),
                    node.goal.env,
                    node.answer.result_ref,
                ));
            }
        }
        cacheable
    }
}

impl<R: Rule> Default for CacheBucket<R> {
    fn default() -> Self {
        Self {
            entries: vec![],
            by_env: HashMap::new(),
            by_env_fingerprint: HashMap::new(),
        }
    }
}

impl<R: Rule> CacheBucket<R> {
    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

impl<R: Rule> Clone for CacheEntry<R> {
    fn clone(&self) -> Self {
        Self {
            env: Arc::clone(&self.env),
            result_ref: self.result_ref,
            fingerprint: self.fingerprint,
        }
    }
}

impl<R: Rule> CacheBucket<R> {
    pub(crate) fn insert(
        &mut self,
        env: Arc<RuleEnv<R>>,
        result_ref: RuleResultRef<R>,
        fingerprint: u64,
    ) {
        let index = self.entries.len();
        self.entries.push(CacheEntry {
            env: Arc::clone(&env),
            result_ref,
            fingerprint,
        });
        self.by_env.entry(env).or_default().push(index);
        self.by_env_fingerprint
            .entry((Arc::clone(&self.entries[index].env), fingerprint))
            .or_default()
            .push(index);
    }

    pub(crate) fn cloned_entries(&self) -> Vec<CacheEntry<R>> {
        self.entries.clone()
    }

    pub(crate) fn cloned_result_refs_for_env(
        &self,
        env: &Arc<RuleEnv<R>>,
    ) -> Option<Vec<RuleResultRef<R>>> {
        self.by_env.get(env).map(|indices| {
            indices
                .iter()
                .map(|index| self.entries[*index].result_ref)
                .collect()
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn cloned_indices_for_env_fingerprint(
        &self,
        env: &Arc<RuleEnv<R>>,
        fingerprint: u64,
    ) -> Option<Vec<usize>> {
        self.by_env_fingerprint
            .get(&(Arc::clone(env), fingerprint))
            .cloned()
    }
}

impl<R: Rule> Clone for GoalKey<R> {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            state_id: self.state_id,
            env: self.env.clone(),
            lazy_depth: self.lazy_depth,
        }
    }
}

impl<R: Rule> Clone for Dependency<R> {
    fn clone(&self) -> Self {
        Self {
            result_ref: self.result_ref,
            env_delta: self.env_delta.clone(),
        }
    }
}

impl<R: Rule> PartialEq for Dependency<R> {
    fn eq(&self, other: &Self) -> bool {
        self.result_ref == other.result_ref && self.env_delta == other.env_delta
    }
}

impl<R: Rule> Eq for Dependency<R> {}

impl<R: Rule> Hash for Dependency<R> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.result_ref.hash(state);
        self.env_delta.hash(state);
    }
}

impl<R: Rule> PartialEq for GoalKey<R> {
    fn eq(&self, other: &Self) -> bool {
        self.query == other.query
            && self.state_id == other.state_id
            && self.env == other.env
            && self.lazy_depth == other.lazy_depth
    }
}

impl<R: Rule> Eq for GoalKey<R> {}

impl<R: Rule> Hash for GoalKey<R> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.query.hash(state);
        self.state_id.hash(state);
        self.env.hash(state);
        self.lazy_depth.hash(state);
    }
}

impl<R: Rule> Clone for Answer<R> {
    fn clone(&self) -> Self {
        Self {
            result_ref: self.result_ref,
            lookups: self.lookups.clone(),
            dependencies: self.dependencies.clone(),
        }
    }
}

impl<R: Rule> Index<DepthFirstNumber> for SearchGraph<R> {
    type Output = Node<R>;

    fn index(&self, index: DepthFirstNumber) -> &Self::Output {
        &self.nodes[index.index]
    }
}

impl<R: Rule> IndexMut<DepthFirstNumber> for SearchGraph<R> {
    fn index_mut(&mut self, index: DepthFirstNumber) -> &mut Self::Output {
        &mut self.nodes[index.index]
    }
}

impl DepthFirstNumber {
    pub(crate) const MAX: Self = Self { index: usize::MAX };

    pub(crate) fn index(self) -> usize {
        self.index
    }
}

impl Add<usize> for DepthFirstNumber {
    type Output = DepthFirstNumber;

    fn add(self, rhs: usize) -> Self::Output {
        Self {
            index: self.index + rhs,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        arena::Arena,
        example::{definition, leaf, ExampleEnv, ExampleResultsArena, ExampleRule, ExampleState},
        stack::Stack,
    };

    use super::*;

    fn goal(query: &str, lazy_depth: usize) -> GoalKey<ExampleRule> {
        GoalKey {
            query: query.to_string(),
            state_id: ExampleState::Resolve,
            env: Arc::new(ExampleEnv::new([definition(query, leaf(query))])),
            lazy_depth: LazyDepth(lazy_depth),
        }
    }

    #[test]
    fn closest_goals_tracks_nearest_matching_goal() {
        // given
        let mut graph = SearchGraph::<ExampleRule>::new();
        let mut stack = Stack::new(8);
        let mut arena = ExampleResultsArena::default();
        let root_goal = goal("root", 0);
        let deferred_goal = goal("root", 1);
        let root_depth = stack.push().expect("stack push should succeed");
        let root_dfn = graph.insert(&root_goal, root_depth, arena.insert_placeholder());
        let deferred_depth = stack.push().expect("stack push should succeed");
        let deferred_dfn = graph.insert(&deferred_goal, deferred_depth, arena.insert_placeholder());

        // when
        let closest_before_pop =
            graph.closest_goal(&"root".to_string(), ExampleState::Resolve, &root_goal.env);
        graph.pop_stack_goal(deferred_dfn);
        stack.pop(deferred_depth);
        let closest_after_pop =
            graph.closest_goal(&"root".to_string(), ExampleState::Resolve, &root_goal.env);

        // then
        assert_eq!(closest_before_pop, Some(deferred_dfn));
        assert_eq!(closest_after_pop, Some(root_dfn));
    }

    #[test]
    fn closest_goals_any_env_tracks_nearest_matching_goal() {
        // given
        let mut graph = SearchGraph::<ExampleRule>::new();
        let mut stack = Stack::new(8);
        let mut arena = ExampleResultsArena::default();
        let root_goal = goal("root", 0);
        let deferred_goal = GoalKey {
            query: root_goal.query.clone(),
            state_id: root_goal.state_id,
            env: Arc::new(ExampleEnv::new([
                definition("root", leaf("root")),
                definition("extra", leaf("extra")),
            ])),
            lazy_depth: LazyDepth(1),
        };
        let root_depth = stack.push().expect("stack push should succeed");
        let root_dfn = graph.insert(&root_goal, root_depth, arena.insert_placeholder());
        let deferred_depth = stack.push().expect("stack push should succeed");
        let deferred_dfn = graph.insert(&deferred_goal, deferred_depth, arena.insert_placeholder());

        // when
        let closest_before_pop =
            graph.closest_goal_any_env(&"root".to_string(), ExampleState::Resolve);
        graph.pop_stack_goal(deferred_dfn);
        stack.pop(deferred_depth);
        let closest_after_pop =
            graph.closest_goal_any_env(&"root".to_string(), ExampleState::Resolve);

        // then
        assert_eq!(closest_before_pop, Some(deferred_dfn));
        assert_eq!(closest_after_pop, Some(root_dfn));
    }

    #[test]
    fn rollback_truncates_popped_suffix() {
        // given
        let mut graph = SearchGraph::<ExampleRule>::new();
        let mut stack = Stack::new(8);
        let mut arena = ExampleResultsArena::default();
        let root_goal = goal("root", 0);
        let child_goal = goal("child", 0);
        let root_depth = stack.push().expect("stack push should succeed");
        let root_dfn = graph.insert(&root_goal, root_depth, arena.insert_placeholder());
        let child_depth = stack.push().expect("stack push should succeed");
        let child_dfn = graph.insert(&child_goal, child_depth, arena.insert_placeholder());
        graph.pop_stack_goal(child_dfn);
        stack.pop(child_depth);

        // when
        graph.rollback_to(child_dfn);

        // then
        assert_eq!(graph.lookup(&root_goal), Some(root_dfn));
        assert_eq!(graph.lookup(&child_goal), None);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn take_cacheable_entries_moves_suffix_from_graph() {
        // given
        let mut graph = SearchGraph::<ExampleRule>::new();
        let mut stack = Stack::new(8);
        let mut arena = ExampleResultsArena::default();
        let root_goal = goal("root", 0);
        let root_result_ref = arena.insert_placeholder();
        let root_depth = stack.push().expect("stack push should succeed");
        let root_dfn = graph.insert(&root_goal, root_depth, root_result_ref);
        graph.pop_stack_goal(root_dfn);
        stack.pop(root_depth);

        // when
        let cacheable = graph.take_cacheable_entries(root_dfn);

        // then
        assert_eq!(graph.lookup(&root_goal), None);
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(cacheable.len(), 1);
        assert_eq!(
            cacheable[0].0,
            (root_goal.query.clone(), root_goal.state_id)
        );
        assert_eq!(cacheable[0].1, root_goal.env);
        assert_eq!(cacheable[0].2, root_result_ref);
    }
}
