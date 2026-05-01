use pyo3::prelude::*;
use rustc_hash::FxHashMap as HashMap;

use super::{RuleArena, RuleId, RuleMode};

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
    identity_map: HashMap<usize, RuleId>,
}

impl Converter {
    fn new() -> Self {
        Self {
            arena: BuildingRuleArena::default(),
            identity_map: HashMap::default(),
        }
    }

    fn convert(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<RuleId> {
        let py_id = obj.as_ptr() as usize;

        if let Some(&rule_id) = self.identity_map.get(&py_id) {
            return Ok(rule_id);
        }

        let slot = self.arena.insert_pending();
        self.identity_map.insert(py_id, slot);

        let rule = self.convert_rule(obj)?;
        self.arena.fill(slot, rule);
        Ok(slot)
    }

    fn convert_optional(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<Option<RuleId>> {
        if obj.is_none() {
            Ok(None)
        } else {
            self.convert(obj).map(Some)
        }
    }

    fn convert_rule(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<RuleMode> {
        let type_name: String = obj.get_type().qualname()?.extract()?;

        match type_name.as_str() {
            "Placeholder" => {
                let inner: Bound<'_, PyAny> = obj.getattr("rule")?;
                if inner.is_none() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "Placeholder.rule is None — unfilled lazy rule",
                    ));
                }
                let target_id = self.convert(&inner)?;
                Ok(RuleMode::MatchFirst {
                    rules: vec![target_id],
                })
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
            "UnionRule" => {
                let variant_rules = self.convert(&obj.getattr("variant_rules")?)?;
                let allow_none_fallback: bool = obj.getattr("allow_none_fallback")?.extract()?;
                Ok(RuleMode::Union {
                    variant_rules,
                    allow_none_fallback,
                })
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
                let hook_param_rule = self.convert_optional(&obj.getattr("hook_param_rule")?)?;
                Ok(RuleMode::MethodImpl {
                    target_rules,
                    hook_param_rule,
                })
            }
            "AutoMethodRule" => {
                let target_rules = self.convert(&obj.getattr("target_rules")?)?;
                let hook_param_rule = self.convert_optional(&obj.getattr("hook_param_rule")?)?;
                Ok(RuleMode::AutoMethod {
                    target_rules,
                    hook_param_rule,
                })
            }
            "MatchFirstRule" => {
                let py_rules: Bound<'_, PyAny> = obj.getattr("rules")?;
                let mut rules = Vec::new();
                for item in py_rules.try_iter()? {
                    rules.push(self.convert(&item?)?);
                }
                Ok(RuleMode::MatchFirst { rules })
            }
            other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "unknown rule type: {other}"
            ))),
        }
    }
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
        Ok(Self {
            arena: converter.arena.finish()?,
            root: root_id,
        })
    }
}
