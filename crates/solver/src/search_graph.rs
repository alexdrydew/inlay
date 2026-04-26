use std::{
    fmt,
    hash::{Hash, Hasher},
    ops::{Add, Index, IndexMut},
    sync::Arc,
};

use derive_where::derive_where;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::{
    cache::CacheEntry,
    lookup_support::{AnswerSupport, LookupSupports},
    rule::{RuleDependencyEnvDelta, RuleEnv, RuleQuery, RuleResultRef, RuleResultsArena},
    stack::StackDepth,
    traits::{Arena, Rule},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct LazyDepth(pub usize);

pub(crate) type ActiveBackrefKey<R> = (RuleQuery<R>, <R as Rule>::RuleStateId, Arc<RuleEnv<R>>);
pub(crate) type CrossEnvBackrefKey<R> = (RuleQuery<R>, <R as Rule>::RuleStateId);

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) struct Dependency<R: Rule, ResultRef: Clone + Eq + Hash = RuleResultRef<R>> {
    pub(crate) result_ref: ResultRef,
    pub(crate) env_delta: RuleDependencyEnvDelta<R>,
}

#[derive_where(Clone, PartialEq, Eq, Hash)]
pub(crate) struct GoalKey<R: Rule> {
    pub(crate) query: RuleQuery<R>,
    pub(crate) state_id: R::RuleStateId,
    pub(crate) env: Arc<RuleEnv<R>>,
    pub(crate) lazy_depth: LazyDepth,
}

fn debug_hash<T: Hash>(value: &T) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

impl<R: Rule> fmt::Debug for GoalKey<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GoalKey")
            .field("query_hash", &debug_hash(&self.query))
            .field("state_hash", &debug_hash(&self.state_id))
            .field("env", &self.env)
            .field("lazy_depth", &self.lazy_depth)
            .finish()
    }
}

#[derive_where(Clone, PartialEq, Eq)]
pub(crate) struct Answer<R: Rule, ResultRef: Clone + Eq + Hash = RuleResultRef<R>> {
    pub(crate) result_ref: ResultRef,
    pub(crate) direct_supports: LookupSupports<R>,
    pub(crate) dependencies: Vec<Dependency<R, ResultRef>>,
}

impl<R: Rule, ResultRef: Clone + Eq + Hash + fmt::Debug> fmt::Debug for Answer<R, ResultRef> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Answer")
            .field("result_ref", &self.result_ref)
            .field("direct_supports", &self.direct_supports.len())
            .field("dependencies", &self.dependencies.len())
            .finish()
    }
}

pub(crate) struct Node<R: Rule> {
    pub(crate) goal: GoalKey<R>,
    pub(crate) answer: Answer<R>,
    answer_support: Option<AnswerSupport<R>>,
    pub(crate) cross_env_reuses: Vec<(RuleResultRef<R>, Arc<RuleEnv<R>>)>,
    pub(crate) stack_depth: Option<StackDepth>,
    pub(crate) links: Minimums,
}

impl<R: Rule> Node<R> {
    fn stored_answer_support(&self) -> Option<&AnswerSupport<R>> {
        self.answer_support.as_ref()
    }

    fn store_answer_support(&mut self, answer_support: AnswerSupport<R>) {
        self.answer_support = Some(answer_support);
    }

    fn invalidate_answer_support(&mut self) -> bool {
        self.answer_support.take().is_some()
    }

    fn take_answer_support(&mut self) -> Option<AnswerSupport<R>> {
        self.answer_support.take()
    }
}

#[derive_where(Default)]
pub(crate) struct SearchGraph<R: Rule> {
    result_refs: HashMap<GoalKey<R>, RuleResultRef<R>>,
    indices: HashMap<GoalKey<R>, DepthFirstNumber>,
    nodes_by_result_ref: HashMap<RuleResultRef<R>, DepthFirstNumber>,
    answer_dependents: HashMap<RuleResultRef<R>, HashSet<RuleResultRef<R>>>,
    closest_goals: HashMap<ActiveBackrefKey<R>, Vec<DepthFirstNumber>>,
    closest_goals_any_env: HashMap<CrossEnvBackrefKey<R>, Vec<DepthFirstNumber>>,
    nodes: Vec<Node<R>>,
}

pub(crate) struct AnswerReplacement<R: Rule> {
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    pub(crate) changed: bool,
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    pub(crate) dependency_count: u64,
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    pub(crate) support_entries_cleared: u64,
    pub(crate) affected_result_refs: HashSet<RuleResultRef<R>>,
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

    pub(crate) fn update_from(&mut self, other: Minimums) {
        self.ancestor = self.ancestor.min(other.ancestor);
    }

    pub(crate) fn ancestor(self) -> DepthFirstNumber {
        self.ancestor
    }
}

impl<R: Rule> SearchGraph<R> {
    fn result_ref_for(
        &mut self,
        goal: &GoalKey<R>,
        results_arena: &mut RuleResultsArena<R>,
    ) -> RuleResultRef<R> {
        if let Some(result_ref) = self.result_refs.get(goal).copied() {
            return result_ref;
        }

        let result_ref = results_arena.insert_placeholder();
        self.result_refs.insert(goal.clone(), result_ref);
        result_ref
    }

    #[cfg(feature = "tracing")]
    pub(crate) fn goal_for_result_ref(&self, result_ref: RuleResultRef<R>) -> Option<&GoalKey<R>> {
        self.nodes_by_result_ref
            .get(&result_ref)
            .and_then(|dfn| self.nodes.get(dfn.index))
            .map(|node| &node.goal)
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
        results_arena: &mut RuleResultsArena<R>,
    ) -> (DepthFirstNumber, RuleResultRef<R>) {
        debug_assert!(
            !self.indices.contains_key(goal),
            "active goals must be unique"
        );
        let result_ref = self.result_ref_for(goal, results_arena);
        let dfn = DepthFirstNumber {
            index: self.nodes.len(),
        };
        self.nodes.push(Node {
            goal: goal.clone(),
            answer: Answer {
                result_ref,
                direct_supports: vec![],
                dependencies: vec![],
            },
            answer_support: None,
            cross_env_reuses: vec![],
            stack_depth: Some(stack_depth),
            links: Minimums::from_self(dfn),
        });
        self.indices.insert(goal.clone(), dfn);
        self.nodes_by_result_ref.insert(result_ref, dfn);
        self.closest_goals
            .entry((goal.query.clone(), goal.state_id, Arc::clone(&goal.env)))
            .or_default()
            .push(dfn);
        self.closest_goals_any_env
            .entry((goal.query.clone(), goal.state_id))
            .or_default()
            .push(dfn);
        (dfn, result_ref)
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
        debug_assert_eq!(stack.pop(), Some(dfn));
        if stack.is_empty() {
            self.closest_goals.remove(&key);
        }

        let any_env_stack = self
            .closest_goals_any_env
            .get_mut(&any_env_key)
            .expect("stack goal must exist in closest_goals_any_env");
        debug_assert_eq!(any_env_stack.pop(), Some(dfn));
        if any_env_stack.is_empty() {
            self.closest_goals_any_env.remove(&any_env_key);
        }
    }

    pub(crate) fn rollback_to(&mut self, dfn: DepthFirstNumber) {
        self.indices.retain(|_, value| *value < dfn);
        self.truncate_active_goal_indexes(dfn);
        let removed_dependencies: Vec<_> = self.nodes[dfn.index..]
            .iter()
            .map(|node| (node.answer.result_ref, node.answer.dependencies.clone()))
            .collect();
        for node in &self.nodes[dfn.index..] {
            debug_assert!(
                node.stack_depth.is_none(),
                "only popped nodes may be rolled back"
            );
            self.nodes_by_result_ref.remove(&node.answer.result_ref);
        }
        for (result_ref, dependencies) in removed_dependencies {
            self.remove_answer_dependency_edges(result_ref, &dependencies);
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

    pub(crate) fn take_cacheable_entries(&mut self, dfn: DepthFirstNumber) -> Vec<CacheEntry<R>> {
        self.indices.retain(|_, value| *value < dfn);
        self.truncate_active_goal_indexes(dfn);
        let mut cacheable = vec![];
        let drained = self.nodes.drain(dfn.index..).collect::<Vec<_>>();
        for mut node in drained {
            debug_assert!(node.stack_depth.is_none(), "cached nodes must be popped");
            self.nodes_by_result_ref.remove(&node.answer.result_ref);
            self.result_refs.remove(&node.goal);
            let result_ref = node.answer.result_ref;
            let dependencies = node.answer.dependencies.clone();
            let answer_support = node.take_answer_support();
            cacheable.push(CacheEntry::new(node.goal, node.answer, answer_support));
            self.remove_answer_dependency_edges(result_ref, &dependencies);
        }
        cacheable
    }

    fn truncate_active_goal_indexes(&mut self, dfn: DepthFirstNumber) {
        self.closest_goals.retain(|_, stack| {
            stack.retain(|value| *value < dfn);
            !stack.is_empty()
        });
        self.closest_goals_any_env.retain(|_, stack| {
            stack.retain(|value| *value < dfn);
            !stack.is_empty()
        });
    }

    pub(crate) fn answer_for(&self, result_ref: RuleResultRef<R>) -> Option<&Answer<R>> {
        self.nodes_by_result_ref
            .get(&result_ref)
            .and_then(|dfn| self.nodes.get(dfn.index))
            .map(|node| &node.answer)
    }

    fn remove_answer_dependency_edges(
        &mut self,
        result_ref: RuleResultRef<R>,
        dependencies: &[Dependency<R>],
    ) {
        for dependency in dependencies {
            if let Some(dependents) = self.answer_dependents.get_mut(&dependency.result_ref) {
                dependents.remove(&result_ref);
                if dependents.is_empty() {
                    self.answer_dependents.remove(&dependency.result_ref);
                }
            }
        }
    }

    fn dependent_closure(&self, result_ref: RuleResultRef<R>) -> HashSet<RuleResultRef<R>> {
        let mut stack = vec![result_ref];
        let mut visited = HashSet::default();

        while let Some(current) = stack.pop() {
            if !visited.insert(current) {
                continue;
            }
            if let Some(dependents) = self.answer_dependents.get(&current) {
                stack.extend(dependents.iter().copied());
            }
        }

        visited
    }

    fn invalidate_answer_supports(
        &mut self,
        result_refs: impl IntoIterator<Item = RuleResultRef<R>>,
    ) -> u64 {
        let mut removed = 0_u64;
        for result_ref in result_refs {
            let Some(dfn) = self.nodes_by_result_ref.get(&result_ref).copied() else {
                continue;
            };
            if self[dfn].invalidate_answer_support() {
                removed += 1;
            }
        }
        removed
    }

    pub(crate) fn replace_answer(
        &mut self,
        dfn: DepthFirstNumber,
        answer: Answer<R>,
    ) -> AnswerReplacement<R> {
        let result_ref = answer.result_ref;
        debug_assert_eq!(
            self[dfn].answer.result_ref, result_ref,
            "graph answer replacement must target the node result ref"
        );

        let changed = self[dfn].answer != answer;
        let dependency_count = answer.dependencies.len() as u64;

        if !changed {
            self[dfn].answer = answer;
            return AnswerReplacement {
                changed,
                dependency_count,
                support_entries_cleared: 0,
                affected_result_refs: HashSet::default(),
            };
        }

        let old_dependencies = self[dfn].answer.dependencies.clone();
        self.remove_answer_dependency_edges(result_ref, &old_dependencies);

        for dependency in &answer.dependencies {
            self.answer_dependents
                .entry(dependency.result_ref)
                .or_default()
                .insert(result_ref);
        }

        self[dfn].answer = answer;
        let affected_result_refs = self.dependent_closure(result_ref);
        let support_entries_cleared =
            self.invalidate_answer_supports(affected_result_refs.iter().copied());

        AnswerReplacement {
            changed,
            dependency_count,
            support_entries_cleared,
            affected_result_refs,
        }
    }

    pub(crate) fn stored_answer_support(
        &self,
        result_ref: RuleResultRef<R>,
    ) -> Option<&AnswerSupport<R>> {
        self.nodes_by_result_ref
            .get(&result_ref)
            .and_then(|dfn| self.nodes.get(dfn.index))
            .and_then(Node::stored_answer_support)
    }

    pub(crate) fn store_answer_support(
        &mut self,
        result_ref: RuleResultRef<R>,
        answer_support: AnswerSupport<R>,
    ) -> bool {
        let Some(dfn) = self.nodes_by_result_ref.get(&result_ref).copied() else {
            return false;
        };
        self[dfn].store_answer_support(answer_support);
        true
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

    #[cfg(feature = "tracing")]
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
        example::{ExampleEnv, ExampleResultsArena, ExampleRule, ExampleState, definition, leaf},
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
        let mut graph = SearchGraph::<ExampleRule>::default();
        let mut stack = Stack::new(8);
        let mut arena = ExampleResultsArena::default();
        let root_goal = goal("root", 0);
        let deferred_goal = goal("root", 1);
        let root_depth = stack.push().expect("stack push should succeed");
        let (root_dfn, _) = graph.insert(&root_goal, root_depth, &mut arena);
        let deferred_depth = stack.push().expect("stack push should succeed");
        let (deferred_dfn, _) = graph.insert(&deferred_goal, deferred_depth, &mut arena);

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
        let mut graph = SearchGraph::<ExampleRule>::default();
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
        let (root_dfn, _) = graph.insert(&root_goal, root_depth, &mut arena);
        let deferred_depth = stack.push().expect("stack push should succeed");
        let (deferred_dfn, _) = graph.insert(&deferred_goal, deferred_depth, &mut arena);

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
        let mut graph = SearchGraph::<ExampleRule>::default();
        let mut stack = Stack::new(8);
        let mut arena = ExampleResultsArena::default();
        let root_goal = goal("root", 0);
        let child_goal = goal("child", 0);
        let root_depth = stack.push().expect("stack push should succeed");
        let root_dfn = graph.insert(&root_goal, root_depth, &mut arena).0;
        let child_depth = stack.push().expect("stack push should succeed");
        let child_dfn = graph.insert(&child_goal, child_depth, &mut arena).0;
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
        let mut graph = SearchGraph::<ExampleRule>::default();
        let mut stack = Stack::new(8);
        let mut arena = ExampleResultsArena::default();
        let root_goal = goal("root", 0);
        let root_depth = stack.push().expect("stack push should succeed");
        let (root_dfn, root_result_ref) = graph.insert(&root_goal, root_depth, &mut arena);
        graph.pop_stack_goal(root_dfn);
        stack.pop(root_depth);

        // when
        let cacheable = graph.take_cacheable_entries(root_dfn);

        // then
        assert_eq!(graph.lookup(&root_goal), None);
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(cacheable.len(), 1);
        assert_eq!(
            (cacheable[0].goal.query.clone(), cacheable[0].goal.state_id),
            (root_goal.query.clone(), root_goal.state_id)
        );
        assert_eq!(cacheable[0].goal.env, root_goal.env);
        assert_eq!(cacheable[0].answer.result_ref.result_ref(), root_result_ref);
        assert!(cacheable[0].answer_support.is_none());
    }

    #[test]
    fn take_cacheable_entries_removes_stale_active_goal_indexes() {
        // given
        let mut graph = SearchGraph::<ExampleRule>::default();
        let mut stack = Stack::new(8);
        let mut arena = ExampleResultsArena::default();
        let root_goal = goal("root", 0);
        let child_goal = goal("child", 0);
        let root_depth = stack.push().expect("stack push should succeed");
        let root_dfn = graph.insert(&root_goal, root_depth, &mut arena).0;
        let child_depth = stack.push().expect("stack push should succeed");
        let child_dfn = graph.insert(&child_goal, child_depth, &mut arena).0;
        graph.pop_stack_goal(child_dfn);
        stack.pop(child_depth);
        graph.pop_stack_goal(root_dfn);
        stack.pop(root_depth);
        graph
            .closest_goals
            .entry((
                child_goal.query.clone(),
                child_goal.state_id,
                Arc::clone(&child_goal.env),
            ))
            .or_default()
            .push(child_dfn);
        graph
            .closest_goals_any_env
            .entry((child_goal.query.clone(), child_goal.state_id))
            .or_default()
            .push(child_dfn);

        // when
        let _ = graph.take_cacheable_entries(root_dfn);

        // then
        assert_eq!(
            graph.closest_goal(&child_goal.query, child_goal.state_id, &child_goal.env),
            None
        );
        assert_eq!(
            graph.closest_goal_any_env(&child_goal.query, child_goal.state_id),
            None
        );
    }
}
