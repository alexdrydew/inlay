use pyo3::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use super::{RuleArena, RuleId, RuleMode, TypeFamilyRules};
use crate::python_identity::PythonIdentity;

#[derive(Clone, PartialEq, Eq)]
struct TypeFamilySignature {
    sentinel: Vec<usize>,
    param_spec: Vec<usize>,
    plain: Vec<usize>,
    class_: Vec<usize>,
    protocol: Vec<usize>,
    typed_dict: Vec<usize>,
    union: Vec<usize>,
    callable: Vec<usize>,
    lazy_ref: Vec<usize>,
    type_var: Vec<usize>,
    fallback: Vec<usize>,
}

#[derive(Clone, PartialEq, Eq)]
enum RuleSignature {
    Constant,
    Property {
        inner: usize,
    },
    LazyRef {
        inner: usize,
    },
    Union {
        variant_rules: usize,
    },
    Protocol {
        property_rule: usize,
        attribute_rule: usize,
        method_rule: usize,
    },
    TypedDict {
        attribute_rule: usize,
    },
    SentinelNone,
    MethodImpl {
        target_rules: usize,
    },
    AttributeSource {
        inner: usize,
    },
    Constructor {
        param_rules: usize,
    },
    Init {
        param_rules: usize,
    },
    MatchFirst {
        rules: Vec<usize>,
    },
    MatchByType {
        rules: Box<TypeFamilySignature>,
    },
}

#[derive(Default)]
struct BuildingRuleArena(Vec<Option<RuleMode>>);

impl BuildingRuleArena {
    fn insert_pending(&mut self) -> RuleId {
        let rule_id = RuleId::new(self.0.len());
        self.0.push(None);
        rule_id
    }

    fn fill(&mut self, rule_id: RuleId, rule: RuleMode) {
        let slot = self
            .0
            .get_mut(rule_id.index())
            .expect("pending rule slot must exist");
        assert!(slot.is_none(), "pending rule slot must be filled once");
        *slot = Some(rule);
    }

    fn finish(self) -> PyResult<RuleArena> {
        let mut rules = Vec::with_capacity(self.0.len());

        for (index, rule) in self.0.into_iter().enumerate() {
            rules.push(rule.ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "internal error: unfilled rule slot {index}"
                ))
            })?);
        }

        Ok(rules.into())
    }
}

struct Converter {
    arena: BuildingRuleArena,
    identity_map: HashMap<PythonIdentity, RuleId>,
}

impl Converter {
    fn new() -> Self {
        Self {
            arena: BuildingRuleArena::default(),
            identity_map: HashMap::default(),
        }
    }

    fn convert(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<RuleId> {
        let type_name: String = obj.get_type().qualname()?.extract()?;
        if type_name == "Placeholder" {
            let inner: Bound<'_, PyAny> = obj.getattr("rule")?;
            if inner.is_none() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Placeholder.rule is None — unfilled lazy rule",
                ));
            }
            return self.convert(&inner);
        }

        let py_id = PythonIdentity::from_bound(obj);

        if let Some(&rule_id) = self.identity_map.get(&py_id) {
            return Ok(rule_id);
        }

        let slot = self.arena.insert_pending();
        self.identity_map.insert(py_id, slot);

        let rule = self.convert_rule(obj)?;
        self.arena.fill(slot, rule);
        Ok(slot)
    }

    fn convert_rule_list(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<Vec<RuleId>> {
        let mut rules = Vec::new();
        for item in obj.try_iter()? {
            rules.push(self.convert(&item?)?);
        }
        Ok(rules)
    }

    fn convert_rule(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<RuleMode> {
        let type_name: String = obj.get_type().qualname()?.extract()?;

        match type_name.as_str() {
            "Placeholder" => {
                unreachable!("Placeholder aliases are resolved before slot allocation")
            }
            "SentinelNoneRule" => Ok(RuleMode::SentinelNone),
            "ConstantRule" => Ok(RuleMode::Constant),
            "LazyRefRule" => {
                let inner = self.convert(&obj.getattr("resolve")?)?;
                Ok(RuleMode::LazyRef { inner })
            }
            "PropertyRule" => {
                let inner = self.convert(&obj.getattr("inner")?)?;
                Ok(RuleMode::Property { inner })
            }
            "AttributeSourceRule" => {
                let inner = self.convert(&obj.getattr("inner")?)?;
                Ok(RuleMode::AttributeSource { inner })
            }
            "ConstructorRule" => {
                let param_rules = self.convert(&obj.getattr("param_rules")?)?;
                Ok(RuleMode::Constructor { param_rules })
            }
            "InitRule" => {
                let param_rules = self.convert(&obj.getattr("param_rules")?)?;
                Ok(RuleMode::Init { param_rules })
            }
            "UnionRule" => {
                let variant_rules = self.convert(&obj.getattr("variant_rules")?)?;
                Ok(RuleMode::Union { variant_rules })
            }
            "ProtocolRule" => {
                let property_rule = self.convert(&obj.getattr("property_rule")?)?;
                let attribute_rule = self.convert(&obj.getattr("attribute_rule")?)?;
                let method_rule = self.convert(&obj.getattr("method_rule")?)?;
                Ok(RuleMode::Protocol {
                    property_rule,
                    attribute_rule,
                    method_rule,
                })
            }
            "TypedDictRule" => {
                let attribute_rule = self.convert(&obj.getattr("attribute_rule")?)?;
                Ok(RuleMode::TypedDict { attribute_rule })
            }
            "MethodImplRule" => {
                let target_rules = self.convert(&obj.getattr("target_rules")?)?;
                Ok(RuleMode::MethodImpl { target_rules })
            }
            "MatchFirstRule" => {
                let py_rules: Bound<'_, PyAny> = obj.getattr("rules")?;
                let rules = self.convert_rule_list(&py_rules)?;
                Ok(RuleMode::MatchFirst { rules })
            }
            "TypeMatchFirstRule" => Ok(RuleMode::MatchByType {
                rules: Box::new(TypeFamilyRules {
                    sentinel: self.convert_rule_list(&obj.getattr("sentinel")?)?,
                    param_spec: self.convert_rule_list(&obj.getattr("param_spec")?)?,
                    plain: self.convert_rule_list(&obj.getattr("plain")?)?,
                    class_: self.convert_rule_list(&obj.getattr("class_")?)?,
                    protocol: self.convert_rule_list(&obj.getattr("protocol")?)?,
                    typed_dict: self.convert_rule_list(&obj.getattr("typed_dict")?)?,
                    union: self.convert_rule_list(&obj.getattr("union")?)?,
                    callable: self.convert_rule_list(&obj.getattr("callable")?)?,
                    lazy_ref: self.convert_rule_list(&obj.getattr("lazy_ref")?)?,
                    type_var: self.convert_rule_list(&obj.getattr("type_var")?)?,
                    fallback: self.convert_rule_list(&obj.getattr("fallback")?)?,
                }),
            }),
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "unknown rule type: {other}"
            ))),
        }
    }
}

fn rule_class(rule_id: RuleId, classes: &[usize]) -> usize {
    classes[rule_id.index()]
}

fn rule_classes(rule_ids: &[RuleId], classes: &[usize]) -> Vec<usize> {
    rule_ids
        .iter()
        .map(|&rule_id| rule_class(rule_id, classes))
        .collect()
}

fn type_family_classes(rules: &TypeFamilyRules, classes: &[usize]) -> TypeFamilySignature {
    TypeFamilySignature {
        sentinel: rule_classes(&rules.sentinel, classes),
        param_spec: rule_classes(&rules.param_spec, classes),
        plain: rule_classes(&rules.plain, classes),
        class_: rule_classes(&rules.class_, classes),
        protocol: rule_classes(&rules.protocol, classes),
        typed_dict: rule_classes(&rules.typed_dict, classes),
        union: rule_classes(&rules.union, classes),
        callable: rule_classes(&rules.callable, classes),
        lazy_ref: rule_classes(&rules.lazy_ref, classes),
        type_var: rule_classes(&rules.type_var, classes),
        fallback: rule_classes(&rules.fallback, classes),
    }
}

fn rule_signature(rule: &RuleMode, classes: &[usize]) -> RuleSignature {
    match rule {
        RuleMode::Constant => RuleSignature::Constant,
        RuleMode::Property { inner } => RuleSignature::Property {
            inner: rule_class(*inner, classes),
        },
        RuleMode::LazyRef { inner } => RuleSignature::LazyRef {
            inner: rule_class(*inner, classes),
        },
        RuleMode::Union { variant_rules } => RuleSignature::Union {
            variant_rules: rule_class(*variant_rules, classes),
        },
        RuleMode::Protocol {
            property_rule,
            attribute_rule,
            method_rule,
        } => RuleSignature::Protocol {
            property_rule: rule_class(*property_rule, classes),
            attribute_rule: rule_class(*attribute_rule, classes),
            method_rule: rule_class(*method_rule, classes),
        },
        RuleMode::TypedDict { attribute_rule } => RuleSignature::TypedDict {
            attribute_rule: rule_class(*attribute_rule, classes),
        },
        RuleMode::SentinelNone => RuleSignature::SentinelNone,
        RuleMode::MethodImpl { target_rules } => RuleSignature::MethodImpl {
            target_rules: rule_class(*target_rules, classes),
        },
        RuleMode::AttributeSource { inner } => RuleSignature::AttributeSource {
            inner: rule_class(*inner, classes),
        },
        RuleMode::Constructor { param_rules } => RuleSignature::Constructor {
            param_rules: rule_class(*param_rules, classes),
        },
        RuleMode::Init { param_rules } => RuleSignature::Init {
            param_rules: rule_class(*param_rules, classes),
        },
        RuleMode::MatchFirst { rules } => RuleSignature::MatchFirst {
            rules: rule_classes(rules, classes),
        },
        RuleMode::MatchByType { rules } => RuleSignature::MatchByType {
            rules: Box::new(type_family_classes(rules, classes)),
        },
    }
}

fn compute_rule_classes(arena: &RuleArena) -> Vec<usize> {
    let mut classes = vec![0; arena.rules().len()];

    loop {
        let mut signatures = Vec::new();
        let mut next_classes = Vec::with_capacity(arena.rules().len());

        for rule in arena.rules() {
            let signature = rule_signature(rule, &classes);
            let class_id = signatures
                .iter()
                .position(|existing| existing == &signature)
                .unwrap_or_else(|| {
                    signatures.push(signature);
                    signatures.len() - 1
                });
            next_classes.push(class_id);
        }

        if next_classes == classes {
            return classes;
        }

        classes = next_classes;
    }
}

fn class_representatives(classes: &[usize]) -> Vec<RuleId> {
    let mut representatives = Vec::new();

    for (index, &class_id) in classes.iter().enumerate() {
        if class_id == representatives.len() {
            representatives.push(RuleId::new(index));
        }
    }

    representatives
}

fn canonical_id(
    rule_id: RuleId,
    classes: &[usize],
    canonical_rule_ids_by_class: &[RuleId],
) -> RuleId {
    canonical_rule_ids_by_class[classes[rule_id.index()]]
}

fn remap_rule_list(
    rules: &[RuleId],
    classes: &[usize],
    canonical_rule_ids_by_class: &[RuleId],
) -> Vec<RuleId> {
    rules
        .iter()
        .map(|&rule_id| canonical_id(rule_id, classes, canonical_rule_ids_by_class))
        .collect()
}

fn remap_type_family_rules(
    rules: &TypeFamilyRules,
    classes: &[usize],
    canonical_rule_ids_by_class: &[RuleId],
) -> TypeFamilyRules {
    TypeFamilyRules {
        sentinel: remap_rule_list(&rules.sentinel, classes, canonical_rule_ids_by_class),
        param_spec: remap_rule_list(&rules.param_spec, classes, canonical_rule_ids_by_class),
        plain: remap_rule_list(&rules.plain, classes, canonical_rule_ids_by_class),
        class_: remap_rule_list(&rules.class_, classes, canonical_rule_ids_by_class),
        protocol: remap_rule_list(&rules.protocol, classes, canonical_rule_ids_by_class),
        typed_dict: remap_rule_list(&rules.typed_dict, classes, canonical_rule_ids_by_class),
        union: remap_rule_list(&rules.union, classes, canonical_rule_ids_by_class),
        callable: remap_rule_list(&rules.callable, classes, canonical_rule_ids_by_class),
        lazy_ref: remap_rule_list(&rules.lazy_ref, classes, canonical_rule_ids_by_class),
        type_var: remap_rule_list(&rules.type_var, classes, canonical_rule_ids_by_class),
        fallback: remap_rule_list(&rules.fallback, classes, canonical_rule_ids_by_class),
    }
}

fn remap_rule_refs_to_canonical_ids(
    rule: &RuleMode,
    classes: &[usize],
    canonical_rule_ids_by_class: &[RuleId],
) -> RuleMode {
    match rule {
        RuleMode::Constant => RuleMode::Constant,
        RuleMode::Property { inner } => RuleMode::Property {
            inner: canonical_id(*inner, classes, canonical_rule_ids_by_class),
        },
        RuleMode::LazyRef { inner } => RuleMode::LazyRef {
            inner: canonical_id(*inner, classes, canonical_rule_ids_by_class),
        },
        RuleMode::Union { variant_rules } => RuleMode::Union {
            variant_rules: canonical_id(*variant_rules, classes, canonical_rule_ids_by_class),
        },
        RuleMode::Protocol {
            property_rule,
            attribute_rule,
            method_rule,
        } => RuleMode::Protocol {
            property_rule: canonical_id(*property_rule, classes, canonical_rule_ids_by_class),
            attribute_rule: canonical_id(*attribute_rule, classes, canonical_rule_ids_by_class),
            method_rule: canonical_id(*method_rule, classes, canonical_rule_ids_by_class),
        },
        RuleMode::TypedDict { attribute_rule } => RuleMode::TypedDict {
            attribute_rule: canonical_id(*attribute_rule, classes, canonical_rule_ids_by_class),
        },
        RuleMode::SentinelNone => RuleMode::SentinelNone,
        RuleMode::MethodImpl { target_rules } => RuleMode::MethodImpl {
            target_rules: canonical_id(*target_rules, classes, canonical_rule_ids_by_class),
        },
        RuleMode::AttributeSource { inner } => RuleMode::AttributeSource {
            inner: canonical_id(*inner, classes, canonical_rule_ids_by_class),
        },
        RuleMode::Constructor { param_rules } => RuleMode::Constructor {
            param_rules: canonical_id(*param_rules, classes, canonical_rule_ids_by_class),
        },
        RuleMode::Init { param_rules } => RuleMode::Init {
            param_rules: canonical_id(*param_rules, classes, canonical_rule_ids_by_class),
        },
        RuleMode::MatchFirst { rules } => RuleMode::MatchFirst {
            rules: remap_rule_list(rules, classes, canonical_rule_ids_by_class),
        },
        RuleMode::MatchByType { rules } => RuleMode::MatchByType {
            rules: Box::new(remap_type_family_rules(
                rules,
                classes,
                canonical_rule_ids_by_class,
            )),
        },
    }
}

fn canonicalize_rule_graph(arena: RuleArena, root: RuleId) -> (RuleArena, RuleId) {
    let classes = compute_rule_classes(&arena);
    let representatives = class_representatives(&classes);
    let canonical_rule_ids_by_class: Vec<RuleId> =
        (0..representatives.len()).map(RuleId::new).collect();

    let rules: Vec<RuleMode> = representatives
        .iter()
        .map(|&representative| {
            remap_rule_refs_to_canonical_ids(
                &arena.rules()[representative.index()],
                &classes,
                &canonical_rule_ids_by_class,
            )
        })
        .collect();

    let root = canonical_id(root, &classes, &canonical_rule_ids_by_class);

    (RuleArena::from(rules), root)
}

#[pyclass(frozen, module = "inlay")]
#[derive(Debug)]
pub struct RuleGraph {
    pub(crate) arena: RuleArena,
    pub(crate) root: RuleId,
}

#[pymethods]
impl RuleGraph {
    #[new]
    fn new(root: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut converter = Converter::new();
        let root_id = converter.convert(root)?;
        let arena = converter.arena.finish()?;
        let (arena, root_id) = canonicalize_rule_graph(arena, root_id);
        Ok(Self {
            arena,
            root: root_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule_id(index: usize) -> RuleId {
        RuleId::new(index)
    }

    #[test]
    fn duplicate_leaf_rules_collapse() {
        let arena = RuleArena::from(vec![RuleMode::Constant, RuleMode::Constant]);

        let (arena, root) = canonicalize_rule_graph(arena, rule_id(1));

        assert_eq!(arena.rules().len(), 1);
        assert_eq!(root, rule_id(0));
        assert!(matches!(arena.rules()[0], RuleMode::Constant));
    }

    #[test]
    fn duplicate_child_references_are_remapped() {
        let arena = RuleArena::from(vec![
            RuleMode::Constant,
            RuleMode::Constant,
            RuleMode::MatchFirst {
                rules: vec![rule_id(0), rule_id(1)],
            },
        ]);

        let (arena, root) = canonicalize_rule_graph(arena, rule_id(2));

        assert_eq!(arena.rules().len(), 2);
        assert_eq!(root, rule_id(1));
        match &arena.rules()[root.index()] {
            RuleMode::MatchFirst { rules } => {
                assert_eq!(rules.as_slice(), &[rule_id(0), rule_id(0)]);
            }
            _ => panic!("expected match_first root"),
        }
    }

    #[test]
    fn match_by_type_preserves_family_buckets() {
        let arena = RuleArena::from(vec![
            RuleMode::Constant,
            RuleMode::MatchByType {
                rules: Box::new(TypeFamilyRules {
                    sentinel: vec![rule_id(0)],
                    ..TypeFamilyRules::default()
                }),
            },
            RuleMode::MatchByType {
                rules: Box::new(TypeFamilyRules {
                    plain: vec![rule_id(0)],
                    ..TypeFamilyRules::default()
                }),
            },
        ]);

        let (arena, root) = canonicalize_rule_graph(arena, rule_id(2));

        assert_eq!(arena.rules().len(), 3);
        assert_eq!(root, rule_id(2));
        match &arena.rules()[1] {
            RuleMode::MatchByType { rules } => {
                assert_eq!(rules.sentinel, vec![rule_id(0)]);
                assert!(rules.plain.is_empty());
            }
            _ => panic!("expected match_by_type"),
        }
        match &arena.rules()[2] {
            RuleMode::MatchByType { rules } => {
                assert!(rules.sentinel.is_empty());
                assert_eq!(rules.plain, vec![rule_id(0)]);
            }
            _ => panic!("expected match_by_type"),
        }
    }

    #[test]
    fn equivalent_cycles_collapse() {
        let arena = RuleArena::from(vec![
            RuleMode::Constructor {
                param_rules: rule_id(1),
            },
            RuleMode::LazyRef { inner: rule_id(0) },
            RuleMode::Constructor {
                param_rules: rule_id(3),
            },
            RuleMode::LazyRef { inner: rule_id(2) },
        ]);

        let (arena, root) = canonicalize_rule_graph(arena, rule_id(2));

        assert_eq!(arena.rules().len(), 2);
        assert_eq!(root, rule_id(0));
        match &arena.rules()[0] {
            RuleMode::Constructor { param_rules } => assert_eq!(*param_rules, rule_id(1)),
            _ => panic!("expected constructor"),
        }
        match &arena.rules()[1] {
            RuleMode::LazyRef { inner } => assert_eq!(*inner, rule_id(0)),
            _ => panic!("expected lazy_ref"),
        }
    }
}
