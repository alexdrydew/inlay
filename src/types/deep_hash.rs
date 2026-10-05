use std::{
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use derive_where::derive_where;
use rustc_hash::{FxHashMap as HashMap, FxHasher};

use super::{
    ArenaSelector, CallableImplementationType, CallableType, CellType, ClassType, Concrete, Keyed,
    PlainType, ProtocolType, PyType, PyTypeConcreteKey, PyTypeKey, Qual, QualifiedMode,
    ReadCellType, SentinelType, ShallowHash, ShallowHashMode, TypeArenas, TypeChildren,
    TypedDictType, UnionType, UnqualifiedMode, Wrapper,
};

// None marks an active node or a subtree containing a cycle.
type HashMemo<'ty, G> = HashMap<PyTypeKey<'ty, G>, Option<u64>>;

#[derive(Debug)]
#[derive_where(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DeepHashValue<M>(u64, PhantomData<M>);

impl<M> DeepHashValue<M> {
    pub(crate) fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Default)]
pub(crate) struct DeepHashCaches<'ty> {
    concrete_unqualified: HashMap<PyTypeConcreteKey<'ty>, u64>,
    concrete_qualified: HashMap<PyTypeConcreteKey<'ty>, u64>,
}

impl<'ty> DeepHashCaches<'ty> {
    pub(crate) fn retain_concrete(
        &mut self,
        mut retain: impl FnMut(PyTypeConcreteKey<'ty>) -> bool,
    ) {
        self.concrete_unqualified.retain(|key, _| retain(*key));
        self.concrete_qualified.retain(|key, _| retain(*key));
    }
}

pub(crate) trait DeepHashMode<'ty, G: ArenaSelector<'ty>>: ShallowHashMode {
    fn resolve_and_hash(
        key: PyTypeKey<'ty, G>,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        memo: &mut HashMemo<'ty, G>,
    ) -> Option<()>
    where
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>;

    fn cache<'a>(caches: &'a DeepHashCaches<'ty>) -> &'a HashMap<PyTypeKey<'ty, G>, u64>;
    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'ty>,
    ) -> &'a mut HashMap<PyTypeKey<'ty, G>, u64>;
}

fn deep_hash_impl<'ty, M: DeepHashMode<'ty, G>, G: ArenaSelector<'ty>>(
    key: PyTypeKey<'ty, G>,
    arenas: &TypeArenas<'ty>,
    memo: &mut HashMemo<'ty, G>,
) -> Option<u64>
where
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
{
    if let Some(&hash) = memo.get(&key) {
        return hash;
    }
    memo.insert(key, None);
    let mut state = FxHasher::default();
    std::mem::discriminant(&key).hash(&mut state);
    M::resolve_and_hash(key, arenas, &mut state, memo)?;
    let hash = state.finish();
    memo.insert(key, Some(hash));
    Some(hash)
}

fn hash_and_recurse<'ty, V, M: DeepHashMode<'ty, G>, G: ArenaSelector<'ty>>(
    v: V,
    arenas: &TypeArenas<'ty>,
    state: &mut impl Hasher,
    memo: &mut HashMemo<'ty, G>,
) -> Option<()>
where
    V: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
    G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
    G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
{
    v.shallow_hash(state);
    for &dep in v.children() {
        deep_hash_impl::<M, G>(dep, arenas, memo)?.hash(state);
    }
    Some(())
}

impl<'ty, O: Wrapper, G: ArenaSelector<'ty>> PyType<O, Qual<Keyed<'ty>>, G> {
    fn dispatch_deep_hash<M: DeepHashMode<'ty, G>>(
        self,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        memo: &mut HashMemo<'ty, G>,
    ) -> Option<()>
    where
        O::Wrap<SentinelType>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<G::TypeVar>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<G::ParamSpec>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<PlainType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<ClassType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<ProtocolType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<TypedDictType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<UnionType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<CallableType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<CallableImplementationType<Qual<Keyed<'ty>>, G>>:
            ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<ReadCellType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        O::Wrap<CellType<Qual<Keyed<'ty>>, G>>: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        G::TypeVar: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
        G::ParamSpec: ShallowHash + TypeChildren<PyTypeKey<'ty, G>>,
    {
        match self {
            PyType::Sentinel(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::ParamSpec(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::Plain(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::Class(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::Protocol(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::TypedDict(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::Union(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::Callable(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::CallableImplementation(v) => {
                hash_and_recurse::<_, M, G>(v, arenas, state, memo)
            }
            PyType::ReadCell(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::Cell(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
            PyType::TypeVar(v) => hash_and_recurse::<_, M, G>(v, arenas, state, memo),
        }
    }
}

impl<'ty> DeepHashMode<'ty, Concrete> for UnqualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'ty, Concrete>,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        memo: &mut HashMemo<'ty, Concrete>,
    ) -> Option<()> {
        arenas
            .get_as::<Self, Concrete>(key)
            .dispatch_deep_hash::<Self>(arenas, state, memo)
    }

    fn cache<'a>(caches: &'a DeepHashCaches<'ty>) -> &'a HashMap<PyTypeConcreteKey<'ty>, u64> {
        &caches.concrete_unqualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'ty>,
    ) -> &'a mut HashMap<PyTypeConcreteKey<'ty>, u64> {
        &mut caches.concrete_unqualified
    }
}

impl<'ty> DeepHashMode<'ty, Concrete> for QualifiedMode {
    fn resolve_and_hash(
        key: PyTypeKey<'ty, Concrete>,
        arenas: &TypeArenas<'ty>,
        state: &mut impl Hasher,
        memo: &mut HashMemo<'ty, Concrete>,
    ) -> Option<()> {
        arenas
            .get_as::<Self, Concrete>(key)
            .dispatch_deep_hash::<Self>(arenas, state, memo)
    }

    fn cache<'a>(caches: &'a DeepHashCaches<'ty>) -> &'a HashMap<PyTypeConcreteKey<'ty>, u64> {
        &caches.concrete_qualified
    }

    fn cache_mut<'a>(
        caches: &'a mut DeepHashCaches<'ty>,
    ) -> &'a mut HashMap<PyTypeConcreteKey<'ty>, u64> {
        &mut caches.concrete_qualified
    }
}

impl<'ty> TypeArenas<'ty> {
    pub(crate) fn deep_hash_concrete<M: DeepHashMode<'ty, Concrete>>(
        &mut self,
        key: PyTypeConcreteKey<'ty>,
    ) -> DeepHashValue<M> {
        if let Some(&h) = M::cache(&self.deep_hash_caches).get(&key) {
            return DeepHashValue(h, PhantomData);
        }
        // Hash acyclic graphs completely, reusing completed child hashes. For
        // recursive graphs use the root's shallow hash: cycle markers would
        // give different hashes to equal graphs with different cycle lengths.
        // Full equality still checks collisions in either case.
        let hash = deep_hash_impl::<M, Concrete>(key, self, &mut HashMap::default())
            .unwrap_or_else(|| self.shallow_hash_of::<M, Concrete>(key).raw());
        M::cache_mut(&mut self.deep_hash_caches).insert(key, hash);
        DeepHashValue(hash, PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::qualifier::Qualifier;
    use crate::types::{PyTypeDescriptor, PyTypeId, Qualified, TypeKeyMap};

    fn plain<'ty>(
        arenas: &mut TypeArenas<'ty>,
        name: &str,
        args: Vec<PyTypeConcreteKey<'ty>>,
    ) -> PyTypeConcreteKey<'ty> {
        PyType::Plain(arenas.concrete.plains.insert(Qualified {
            inner: PlainType {
                descriptor: PyTypeDescriptor {
                    id: PyTypeId::new(name.to_owned()),
                    display_name: Arc::from(name),
                    origin: None,
                },
                args,
            },
            qualifier: Qualifier::any(),
        }))
    }

    fn set_args<'ty>(
        arenas: &mut TypeArenas<'ty>,
        key: PyTypeConcreteKey<'ty>,
        args: Vec<PyTypeConcreteKey<'ty>>,
    ) {
        let PyType::Plain(key) = key else {
            panic!("expected plain type");
        };
        arenas.concrete.plains.get_mut(key).inner.args = args;
    }

    fn assert_equal_hashes<'ty>(
        arenas: &mut TypeArenas<'ty>,
        a: PyTypeConcreteKey<'ty>,
        b: PyTypeConcreteKey<'ty>,
    ) {
        assert!(arenas.deep_eq_concrete::<QualifiedMode>(a, b));
        assert_eq!(
            arenas.deep_hash_concrete::<QualifiedMode>(a).raw(),
            arenas.deep_hash_concrete::<QualifiedMode>(b).raw()
        );
        assert_eq!(
            arenas.deep_hash_concrete::<UnqualifiedMode>(a).raw(),
            arenas.deep_hash_concrete::<UnqualifiedMode>(b).raw()
        );
    }

    #[test]
    fn hashes_do_not_depend_on_dag_sharing() {
        let mut arenas = TypeArenas::default();
        let a = plain(&mut arenas, "Leaf", vec![]);
        let b = plain(&mut arenas, "Leaf", vec![]);
        let shared = plain(&mut arenas, "Node", vec![a, a]);
        let duplicated = plain(&mut arenas, "Node", vec![a, b]);

        assert_equal_hashes(&mut arenas, shared, duplicated);
    }

    #[test]
    fn bisimilar_cycles_have_equal_hashes() {
        let mut arenas = TypeArenas::default();
        let a = plain(&mut arenas, "Node", vec![]);
        let b = plain(&mut arenas, "Node", vec![]);
        let c = plain(&mut arenas, "Node", vec![]);
        set_args(&mut arenas, a, vec![a]);
        set_args(&mut arenas, b, vec![c]);
        set_args(&mut arenas, c, vec![b]);

        assert_equal_hashes(&mut arenas, a, b);
        assert_equal_hashes(&mut arenas, b, c);
    }

    #[test]
    fn cyclic_hash_collisions_still_use_full_equality() {
        let mut arenas = TypeArenas::default();
        let a = plain(&mut arenas, "Node", vec![]);
        let b = plain(&mut arenas, "Node", vec![]);
        let child_a = plain(&mut arenas, "A", vec![]);
        let child_b = plain(&mut arenas, "B", vec![]);
        set_args(&mut arenas, a, vec![a, child_a]);
        set_args(&mut arenas, b, vec![b, child_b]);
        assert_eq!(
            arenas.deep_hash_concrete::<QualifiedMode>(a).raw(),
            arenas.deep_hash_concrete::<QualifiedMode>(b).raw()
        );
        assert!(!arenas.deep_eq_concrete::<QualifiedMode>(a, b));
        let mut map = TypeKeyMap::<QualifiedMode, usize>::default();

        map.insert(a, 1, &mut arenas);
        map.insert(b, 2, &mut arenas);

        assert_eq!(map.get(a, &mut arenas), Some(&1));
        assert_eq!(map.get(b, &mut arenas), Some(&2));
    }

    #[test]
    fn nested_qualifiers_remain_distinct() {
        let mut arenas = TypeArenas::default();
        let a = plain(&mut arenas, "Leaf", vec![]);
        let b = plain(&mut arenas, "Leaf", vec![]);
        let PyType::Plain(b_key) = b else {
            unreachable!()
        };
        arenas.concrete.plains.get_mut(b_key).qualifier = Qualifier::unqualified();
        let a = plain(&mut arenas, "Node", vec![a]);
        let b = plain(&mut arenas, "Node", vec![b]);

        assert!(arenas.deep_eq_concrete::<UnqualifiedMode>(a, b));
        assert_eq!(
            arenas.deep_hash_concrete::<UnqualifiedMode>(a).raw(),
            arenas.deep_hash_concrete::<UnqualifiedMode>(b).raw()
        );
        assert!(!arenas.deep_eq_concrete::<QualifiedMode>(a, b));
        let mut map = TypeKeyMap::<QualifiedMode, usize>::default();
        map.insert(a, 1, &mut arenas);
        map.insert(b, 2, &mut arenas);
        assert_eq!(map.get(a, &mut arenas), Some(&1));
        assert_eq!(map.get(b, &mut arenas), Some(&2));
    }

    #[test]
    fn shared_dag_hashing_memoizes_each_node_once() {
        let mut arenas = TypeArenas::default();
        let mut root = plain(&mut arenas, "Leaf", vec![]);
        let depth = 20;
        for _ in 0..depth {
            root = plain(&mut arenas, "Node", vec![root, root]);
        }
        let mut memo = HashMap::default();

        assert!(deep_hash_impl::<QualifiedMode, Concrete>(root, &arenas, &mut memo).is_some());

        assert_eq!(memo.len(), depth + 1);
        assert!(memo.values().all(Option::is_some));
    }

    #[test]
    fn acyclic_hashes_include_deep_children() {
        let mut arenas = TypeArenas::default();
        let mut a = plain(&mut arenas, "A", vec![]);
        let mut b = plain(&mut arenas, "B", vec![]);
        for _ in 0..20 {
            a = plain(&mut arenas, "Node", vec![a]);
            b = plain(&mut arenas, "Node", vec![b]);
        }

        assert_ne!(
            arenas.deep_hash_concrete::<QualifiedMode>(a).raw(),
            arenas.deep_hash_concrete::<QualifiedMode>(b).raw()
        );
    }

    #[test]
    fn unfolding_a_cycle_does_not_change_its_hash() {
        let mut arenas = TypeArenas::default();
        let cycle = plain(&mut arenas, "Node", vec![]);
        set_args(&mut arenas, cycle, vec![cycle, cycle]);
        let mut unfolded = cycle;
        for _ in 0..20 {
            unfolded = plain(&mut arenas, "Node", vec![unfolded, unfolded]);
        }

        assert_equal_hashes(&mut arenas, cycle, unfolded);
        let mut map = TypeKeyMap::<QualifiedMode, usize>::default();
        map.insert(cycle, 1, &mut arenas);
        assert_eq!(map.get(unfolded, &mut arenas), Some(&1));
    }

    #[test]
    fn rollback_invalidates_hashes_before_arena_indices_are_reused() {
        let mut arenas = TypeArenas::default();
        let kept = plain(&mut arenas, "Kept", vec![]);
        let kept_hash = arenas.deep_hash_concrete::<QualifiedMode>(kept).raw();
        let snapshot = arenas.concrete_snapshot();
        let old = plain(&mut arenas, "Old", vec![kept]);
        let old_qualified = arenas.deep_hash_concrete::<QualifiedMode>(old).raw();
        let old_unqualified = arenas.deep_hash_concrete::<UnqualifiedMode>(old).raw();

        arenas.truncate_concrete_to(snapshot);
        let new = plain(&mut arenas, "New", vec![]);

        assert!(old == new);
        assert_ne!(
            arenas.deep_hash_concrete::<QualifiedMode>(new).raw(),
            old_qualified
        );
        assert_ne!(
            arenas.deep_hash_concrete::<UnqualifiedMode>(new).raw(),
            old_unqualified
        );
        assert_eq!(
            arenas.deep_hash_concrete::<QualifiedMode>(kept).raw(),
            kept_hash
        );
    }
}
