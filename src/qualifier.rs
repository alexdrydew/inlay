use std::collections::BTreeSet;
use std::hash::Hash;

use pyo3::prelude::*;
use pyo3::types::PyFrozenSet;

type Alternative = BTreeSet<String>;
type Alternatives = BTreeSet<Alternative>;

#[pyclass(frozen, eq, hash, from_py_object, module = "inlay")]
#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Qualifier {
    alternatives: Alternatives,
    is_any: bool,
}

impl Qualifier {
    fn unqualified_alternatives() -> Alternatives {
        let mut a = Alternatives::new();
        a.insert(Alternative::new());
        a
    }

    pub fn is_unqualified(&self) -> bool {
        !self.is_any && self.alternatives == Self::unqualified_alternatives()
    }

    pub fn is_any(&self) -> bool {
        self.is_any
    }

    pub(crate) fn any() -> Self {
        Self {
            alternatives: Alternatives::new(),
            is_any: true,
        }
    }

    pub(crate) fn intersect(&self, other: &Qualifier) -> Qualifier {
        self.__and__(other)
    }

    pub(crate) fn display_compact(&self) -> String {
        if self.is_any {
            return "ANY".to_string();
        }
        let mut alternatives: Vec<Vec<&str>> = self
            .alternatives
            .iter()
            .map(|alt| {
                let mut tags: Vec<&str> = alt.iter().map(String::as_str).collect();
                tags.sort();
                tags
            })
            .collect();
        alternatives.sort();
        let parts: Vec<String> = alternatives
            .iter()
            .map(|alt| {
                if alt.is_empty() {
                    "EMPTY".to_string()
                } else {
                    alt.join(" & ")
                }
            })
            .collect();
        parts.join(" | ")
    }
}

/// Check if a registration covers all alternatives in a target qualifier.
///
/// Every target clause must appear **exactly** in the registration's clause set.
/// Clauses are compared by set equality (not subset): `qual('a') & qual('b')`
/// does NOT match a target of `qual('a')`.
///
/// `qual('chat')` matches `qual('chat') | qual('game')` (clause `{chat}` is present).
/// `qual('a')` does NOT match `qual('a') & qual('b')` (clause `{a}` ≠ `{a,b}`).
/// `qual()` does NOT match `qual('game')` (clause `{}` not in `{{game}}`).
///
/// `Qualifier::ANY` is a top registration: it matches any target when it is
/// registered, but an `ANY` target only matches an `ANY` registration.
pub fn qualifier_matches(target: &Qualifier, registration: &Qualifier) -> bool {
    if registration.is_any {
        return true;
    }
    if target.is_any {
        return false;
    }
    target.alternatives.is_subset(&registration.alternatives)
}

impl From<Alternatives> for Qualifier {
    fn from(mut value: Alternatives) -> Self {
        if value.is_empty() {
            value.insert(Alternative::new());
        }

        Self {
            alternatives: value,
            is_any: false,
        }
    }
}

#[pymethods]
impl Qualifier {
    #[new]
    #[pyo3(signature = (*tags, _union=None))]
    fn new(tags: Vec<String>, _union: Option<Alternatives>) -> Self {
        let mut alts = _union.unwrap_or_default();
        if !tags.is_empty() {
            alts.insert(tags.into_iter().collect());
        }
        alts.into()
    }

    #[classattr]
    #[pyo3(name = "ANY")]
    fn any_py() -> Self {
        Self::any()
    }

    #[getter]
    fn alternatives<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        let inner: Vec<_> = self
            .alternatives
            .iter()
            .map(|alt| {
                let tags: Vec<_> = alt.iter().collect();
                PyFrozenSet::new(py, &tags)
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyFrozenSet::new(py, &inner)
    }

    #[getter]
    fn is_qualified(&self) -> bool {
        !self.is_unqualified()
    }

    fn __and__(&self, other: &Qualifier) -> Qualifier {
        if self.is_any {
            return other.clone();
        }
        if other.is_any {
            return self.clone();
        }
        self.alternatives
            .iter()
            .flat_map(|alts| other.alternatives.iter().map(move |ralts| alts | ralts))
            .collect::<BTreeSet<_>>()
            .into()
    }

    fn __or__(&self, other: &Qualifier) -> Qualifier {
        if self.is_any || other.is_any {
            return Self::any();
        }
        (&self.alternatives | &other.alternatives).into()
    }

    pub(crate) fn __repr__(&self) -> String {
        if self.is_any {
            return "Qualifier.ANY".to_string();
        }
        if self.is_unqualified() {
            return "qual()".to_string();
        }
        let mut alternatives: Vec<Vec<&str>> = self
            .alternatives
            .iter()
            .map(|alt| {
                let mut tags: Vec<&str> = alt.iter().map(String::as_str).collect();
                tags.sort();
                tags
            })
            .collect();
        alternatives.sort();
        let parts: Vec<String> = alternatives
            .iter()
            .map(|alt| {
                if alt.is_empty() {
                    "qual()".to_string()
                } else {
                    alt.iter()
                        .map(|tag| format!("qual('{tag}')"))
                        .collect::<Vec<_>>()
                        .join(" & ")
                }
            })
            .collect();
        parts.join(" | ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qual(tags: &[&str]) -> Qualifier {
        let alt: Alternative = tags.iter().map(|s| s.to_string()).collect();
        let mut alts = Alternatives::new();
        if alt.is_empty() {
            alts.insert(Alternative::new());
        } else {
            alts.insert(alt);
        }
        alts.into()
    }

    fn unqualified() -> Qualifier {
        qual(&[])
    }

    #[test]
    fn unqualified_is_not_qualified() {
        assert!(!unqualified().is_qualified());
        assert!(unqualified().is_unqualified());
    }

    #[test]
    fn single_tag_is_qualified() {
        let q = qual(&["read"]);
        assert!(q.is_qualified());
        assert!(!q.is_unqualified());
    }

    #[test]
    fn and_combines_tags() {
        let a = qual(&["read"]);
        let b = qual(&["scoped"]);
        let result = a.__and__(&b);

        let expected_alt: Alternative = ["read", "scoped"].iter().map(|s| s.to_string()).collect();
        assert_eq!(result.alternatives.len(), 1);
        assert!(result.alternatives.contains(&expected_alt));
    }

    #[test]
    fn and_with_unqualified_preserves() {
        let a = qual(&["read"]);
        let u = unqualified();
        let result = a.__and__(&u);
        assert_eq!(result, a);
    }

    #[test]
    fn or_creates_union() {
        let a = qual(&["read"]);
        let b = qual(&["write"]);
        let result = a.__or__(&b);

        assert_eq!(result.alternatives.len(), 2);
        let read_alt: Alternative = ["read"].iter().map(|s| s.to_string()).collect();
        let write_alt: Alternative = ["write"].iter().map(|s| s.to_string()).collect();
        assert!(result.alternatives.contains(&read_alt));
        assert!(result.alternatives.contains(&write_alt));
    }

    #[test]
    fn and_distributes_over_or() {
        // (a | b) & c = (a & c) | (b & c)
        let a = qual(&["a"]);
        let b = qual(&["b"]);
        let c = qual(&["x"]);
        let a_or_b = a.__or__(&b);
        let result = a_or_b.__and__(&c);

        assert_eq!(result.alternatives.len(), 2);
        let ax: Alternative = ["a", "x"].iter().map(|s| s.to_string()).collect();
        let bx: Alternative = ["b", "x"].iter().map(|s| s.to_string()).collect();
        assert!(result.alternatives.contains(&ax));
        assert!(result.alternatives.contains(&bx));
    }

    #[test]
    fn and_cartesian_product() {
        // (a | b) & (x | y) = (a,x) | (a,y) | (b,x) | (b,y)
        let ab = qual(&["a"]).__or__(&qual(&["b"]));
        let xy = qual(&["x"]).__or__(&qual(&["y"]));
        let result = ab.__and__(&xy);

        assert_eq!(result.alternatives.len(), 4);
    }

    #[test]
    fn or_is_idempotent() {
        let a = qual(&["read"]);
        let result = a.__or__(&a);
        assert_eq!(result, a);
    }

    #[test]
    fn equality() {
        assert_eq!(qual(&["read"]), qual(&["read"]));
        assert_ne!(qual(&["read"]), qual(&["write"]));
        assert_eq!(unqualified(), unqualified());
        assert_ne!(unqualified(), qual(&["read"]));
    }

    #[test]
    fn hash_consistency() {
        use std::hash::{DefaultHasher, Hasher};

        let a = qual(&["read"]);
        let b = qual(&["read"]);

        let mut ha = DefaultHasher::new();
        a.hash(&mut ha);
        let mut hb = DefaultHasher::new();
        b.hash(&mut hb);

        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn repr_unqualified() {
        assert_eq!(unqualified().__repr__(), "qual()");
    }

    #[test]
    fn repr_single_tag() {
        assert_eq!(qual(&["read"]).__repr__(), "qual('read')");
    }

    #[test]
    fn repr_intersection() {
        let q = qual(&["read"]).__and__(&qual(&["scoped"]));
        assert_eq!(q.__repr__(), "qual('read') & qual('scoped')");
    }

    #[test]
    fn repr_union() {
        let q = qual(&["a"]).__or__(&qual(&["b"]));
        assert_eq!(q.__repr__(), "qual('a') | qual('b')");
    }

    #[test]
    fn constructor_with_tags() {
        let q = Qualifier::new(vec!["read".to_string()], None);
        assert_eq!(q, qual(&["read"]));
    }

    #[test]
    fn constructor_no_tags() {
        let q = Qualifier::new(vec![], None);
        assert_eq!(q, unqualified());
    }

    #[test]
    fn constructor_with_union() {
        let mut alts = Alternatives::new();
        alts.insert(["a"].iter().map(|s| s.to_string()).collect());
        alts.insert(["b"].iter().map(|s| s.to_string()).collect());
        let q = Qualifier::new(vec![], Some(alts));

        assert_eq!(q, qual(&["a"]).__or__(&qual(&["b"])));
    }

    #[test]
    fn constructor_tags_plus_union() {
        let mut alts = Alternatives::new();
        alts.insert(["a"].iter().map(|s| s.to_string()).collect());
        let q = Qualifier::new(vec!["b".to_string()], Some(alts));

        assert_eq!(q, qual(&["a"]).__or__(&qual(&["b"])));
    }

    #[test]
    fn display_compact_union_with_unqualified() {
        let q = unqualified().__or__(&qual(&["a"])).__or__(&qual(&["b"]));
        assert_eq!(q.display_compact(), "EMPTY | a | b");
    }

    #[test]
    fn repr_union_with_unqualified() {
        let q = unqualified().__or__(&qual(&["a"])).__or__(&qual(&["b"]));
        assert_eq!(q.__repr__(), "qual() | qual('a') | qual('b')");
    }

    // -- qualifier_matches: clause-exact, alternatives-subset --

    #[test]
    fn match_exact_single() {
        assert!(qualifier_matches(&qual(&["game"]), &qual(&["game"])));
    }

    #[test]
    fn match_exact_empty() {
        assert!(qualifier_matches(&unqualified(), &unqualified()));
    }

    #[test]
    fn match_request_subset_of_provider_alternatives() {
        // Provider covers chat|game, request only asks for chat
        let provider = qual(&["chat"]).__or__(&qual(&["game"]));
        assert!(qualifier_matches(&qual(&["chat"]), &provider));
        assert!(qualifier_matches(&qual(&["game"]), &provider));
    }

    #[test]
    fn match_any_module_qualifier_pattern() {
        // The real-world ANY_MODULE_QUALIFIER: qual('chat') | qual('game') | qual()
        let any = qual(&["chat"])
            .__or__(&qual(&["game"]))
            .__or__(&unqualified());
        assert!(qualifier_matches(&qual(&["chat"]), &any));
        assert!(qualifier_matches(&qual(&["game"]), &any));
        assert!(qualifier_matches(&unqualified(), &any));
    }

    #[test]
    fn no_match_qualified_provider_unqualified_request() {
        // The leaking bug: provider {game} must NOT match request {}
        assert!(!qualifier_matches(&unqualified(), &qual(&["game"])));
    }

    #[test]
    fn no_match_compound_vs_simple() {
        // No A&B vs A subtyping: {a,b} does NOT match request {a}
        let compound = qual(&["a"]).__and__(&qual(&["b"])); // {{a,b}}
        assert!(!qualifier_matches(&qual(&["a"]), &compound));
        assert!(!qualifier_matches(&qual(&["b"]), &compound));
    }

    #[test]
    fn no_match_simple_vs_compound() {
        // {a} does NOT match request {a,b} — no tag-level subsetting
        let compound = qual(&["a"]).__and__(&qual(&["b"]));
        assert!(!qualifier_matches(&compound, &qual(&["a"])));
    }

    #[test]
    fn no_match_different_tags() {
        assert!(!qualifier_matches(&qual(&["chat"]), &qual(&["game"])));
    }

    #[test]
    fn no_match_request_exceeds_provider() {
        // Request asks for chat|game|ai, provider only covers chat|game
        let provider = qual(&["chat"]).__or__(&qual(&["game"]));
        let request = qual(&["chat"])
            .__or__(&qual(&["game"]))
            .__or__(&qual(&["ai"]));
        assert!(!qualifier_matches(&request, &provider));
    }

    #[test]
    fn match_compound_exact() {
        let q = qual(&["a"]).__and__(&qual(&["b"]));
        assert!(qualifier_matches(&q, &q));
    }

    #[test]
    fn no_match_compound_request_vs_simple_registration() {
        // {ai, write} does NOT match {ai} — clause-exact
        let ai_write = qual(&["ai"]).__and__(&qual(&["write"]));
        assert!(!qualifier_matches(&ai_write, &qual(&["ai"])));
    }

    #[test]
    fn match_compound_request_with_explicit_alias() {
        // {ai, write} matches registration that explicitly includes {ai, write}
        let ai_write = qual(&["ai"]).__and__(&qual(&["write"]));
        let registration = qual(&["ai"]).__or__(&ai_write.clone());
        assert!(qualifier_matches(&ai_write, &registration));
    }

    #[test]
    fn no_match_unqualified_leaking() {
        // Unqualified registration does NOT match qualified request
        assert!(!qualifier_matches(&qual(&["game"]), &unqualified()));
        assert!(!qualifier_matches(&qual(&["ai"]), &unqualified()));
    }

    #[test]
    fn no_match_qualified_leaking() {
        // Module-specific registration does NOT match unqualified request
        assert!(!qualifier_matches(&unqualified(), &qual(&["game"])));
    }

    // -- Qualifier::ANY tests --

    #[test]
    fn any_is_qualified() {
        let any = Qualifier::any();
        assert!(any.is_qualified());
        assert!(!any.is_unqualified());
        assert!(any.is_any());
    }

    #[test]
    fn any_not_equal_to_unqualified() {
        assert_ne!(Qualifier::any(), unqualified());
    }

    #[test]
    fn any_equal_to_any() {
        assert_eq!(Qualifier::any(), Qualifier::any());
    }

    #[test]
    fn any_matches_as_registration() {
        let any = Qualifier::any();
        assert!(qualifier_matches(&qual(&["ai"]), &any));
        assert!(qualifier_matches(&unqualified(), &any));
        let compound = qual(&["ai"]).__and__(&qual(&["write"]));
        assert!(qualifier_matches(&compound, &any));
    }

    #[test]
    fn any_matches_as_target() {
        let any = Qualifier::any();
        assert!(!qualifier_matches(&any, &qual(&["ai"])));
        assert!(!qualifier_matches(&any, &unqualified()));
        assert!(qualifier_matches(&any, &any));
    }

    #[test]
    fn any_and_anything_is_any() {
        let any = Qualifier::any();
        assert_eq!(any.__and__(&qual(&["ai"])), qual(&["ai"]));
        assert_eq!(qual(&["ai"]).__and__(&any), qual(&["ai"]));
        assert_eq!(any.__and__(&unqualified()), unqualified());
        assert_eq!(qual(&["ai"]).intersect(&any), qual(&["ai"]));
        assert_eq!(any.__and__(&any), Qualifier::any());
    }

    #[test]
    fn any_or_anything_is_any() {
        let any = Qualifier::any();
        assert_eq!(any.__or__(&qual(&["ai"])), Qualifier::any());
        assert_eq!(qual(&["ai"]).__or__(&any), Qualifier::any());
    }

    #[test]
    fn any_repr() {
        assert_eq!(Qualifier::any().__repr__(), "Qualifier.ANY");
    }

    #[test]
    fn any_display_compact() {
        assert_eq!(Qualifier::any().display_compact(), "ANY");
    }
}
