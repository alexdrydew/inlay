use std::collections::{HashMap, HashSet};

use derive_where::derive_where;

use super::{
    Arena, ArenaFamily, CallableKey, Concrete, Parametric, PyType, PyTypeConcreteKey, PyTypeId,
    PyTypeParametricKey, TypeArenas, UnqualifiedMode,
};

#[derive_where(Default)]
pub(crate) struct Bindings<S: ArenaFamily> {
    pub(crate) type_vars: HashMap<PyTypeId, PyTypeConcreteKey<S>>,
    pub(crate) param_specs: HashMap<PyTypeId, PyTypeConcreteKey<S>>,
}

#[derive(Debug)]
pub(crate) enum UnifyError {
    VariantMismatch,
    LocalMismatch,
    DepCountMismatch,
    ConflictingBinding(PyTypeId),
}

fn cross_unify_pairs<S: ArenaFamily>(
    requests: &[PyTypeConcreteKey<S>],
    registrations: &[PyTypeParametricKey<S>],
    arenas: &TypeArenas<S>,
    bindings: Bindings<S>,
    visited: &mut HashSet<(PyTypeConcreteKey<S>, PyTypeParametricKey<S>)>,
) -> Result<Bindings<S>, UnifyError> {
    if requests.len() != registrations.len() {
        return Err(UnifyError::DepCountMismatch);
    }
    requests
        .iter()
        .copied()
        .zip(registrations.iter().copied())
        .try_fold(bindings, |b, (req, reg)| {
            cross_unify(req, reg, arenas, b, visited)
        })
}

fn cross_unify<S: ArenaFamily>(
    request: PyTypeConcreteKey<S>,
    registration: PyTypeParametricKey<S>,
    arenas: &TypeArenas<S>,
    bindings: Bindings<S>,
    visited: &mut HashSet<(PyTypeConcreteKey<S>, PyTypeParametricKey<S>)>,
) -> Result<Bindings<S>, UnifyError> {
    // TypeVar binding — keyed by Python TypeVar identity (PyTypeId),
    // not arena slot key. The same logical TypeVar may have different
    // slot keys due to different qualifier contexts (return vs params).
    if let PyType::TypeVar(tv_key) = registration {
        let tv_id = arenas
            .parametric
            .type_vars
            .get(&tv_key)
            .expect("dangling key")
            .inner
            .descriptor
            .id
            .clone();
        return match bindings.type_vars.get(&tv_id) {
            Some(&existing) => {
                if arenas.deep_eq_concrete::<UnqualifiedMode>(existing, request) {
                    Ok(bindings)
                } else {
                    Err(UnifyError::ConflictingBinding(tv_id))
                }
            }
            None => {
                let mut bindings = bindings;
                bindings.type_vars.insert(tv_id, request);
                Ok(bindings)
            }
        };
    }

    // ParamSpec binding — keyed by Python identity (PyTypeId), same as TypeVar.
    if let PyType::ParamSpec(ps_key) = registration {
        let ps_id = arenas
            .parametric
            .param_specs
            .get(&ps_key)
            .expect("dangling key")
            .inner
            .descriptor
            .id
            .clone();
        return match bindings.param_specs.get(&ps_id) {
            Some(&existing) => {
                if arenas.deep_eq_concrete::<UnqualifiedMode>(existing, request) {
                    Ok(bindings)
                } else {
                    Err(UnifyError::VariantMismatch)
                }
            }
            None => {
                let mut bindings = bindings;
                bindings.param_specs.insert(ps_id, request);
                Ok(bindings)
            }
        };
    }

    // Cycle detection
    if !visited.insert((request, registration)) {
        return Ok(bindings);
    }

    let result = cross_unify_known(request, registration, arenas, bindings, visited);

    visited.remove(&(request, registration));
    result
}

fn cross_unify_known<S: ArenaFamily>(
    request: PyTypeConcreteKey<S>,
    registration: PyTypeParametricKey<S>,
    arenas: &TypeArenas<S>,
    bindings: Bindings<S>,
    visited: &mut HashSet<(PyTypeConcreteKey<S>, PyTypeParametricKey<S>)>,
) -> Result<Bindings<S>, UnifyError> {
    match (request, registration) {
        (PyType::Sentinel(a), PyType::Sentinel(b)) => {
            let req = arenas.sentinels.get(&a).expect("dangling key");
            let reg = arenas.sentinels.get(&b).expect("dangling key");
            if req.inner.value != reg.inner.value {
                return Err(UnifyError::LocalMismatch);
            }
            Ok(bindings)
        }

        (PyType::Plain(a), PyType::Plain(b)) => {
            let req = arenas.concrete.plains.get(&a).expect("dangling key");
            let reg = arenas.parametric.plains.get(&b).expect("dangling key");
            if req.inner.descriptor != reg.inner.descriptor {
                return Err(UnifyError::LocalMismatch);
            }
            cross_unify_pairs(&req.inner.args, &reg.inner.args, arenas, bindings, visited)
        }

        (PyType::Union(a), PyType::Union(b)) => {
            let req = arenas.concrete.unions.get(&a).expect("dangling key");
            let reg = arenas.parametric.unions.get(&b).expect("dangling key");
            if req.inner.variants.len() != reg.inner.variants.len() {
                return Err(UnifyError::LocalMismatch);
            }
            cross_unify_pairs(
                &req.inner.variants,
                &reg.inner.variants,
                arenas,
                bindings,
                visited,
            )
        }

        (PyType::Protocol(a), PyType::Protocol(b)) => {
            let req = arenas.concrete.protocols.get(&a).expect("dangling key");
            let reg = arenas.parametric.protocols.get(&b).expect("dangling key");
            if req.inner.descriptor != reg.inner.descriptor
                || !req.inner.methods.keys().eq(reg.inner.methods.keys())
                || !req.inner.attributes.keys().eq(reg.inner.attributes.keys())
                || !req.inner.properties.keys().eq(reg.inner.properties.keys())
            {
                return Err(UnifyError::LocalMismatch);
            }
            let req_deps: Vec<_> = req
                .inner
                .methods
                .values()
                .chain(req.inner.attributes.values())
                .chain(req.inner.properties.values())
                .chain(req.inner.type_params.iter())
                .copied()
                .collect();
            let reg_deps: Vec<_> = reg
                .inner
                .methods
                .values()
                .chain(reg.inner.attributes.values())
                .chain(reg.inner.properties.values())
                .chain(reg.inner.type_params.iter())
                .copied()
                .collect();
            cross_unify_pairs(&req_deps, &reg_deps, arenas, bindings, visited)
        }

        (PyType::TypedDict(a), PyType::TypedDict(b)) => {
            let req = arenas.concrete.typed_dicts.get(&a).expect("dangling key");
            let reg = arenas.parametric.typed_dicts.get(&b).expect("dangling key");
            if req.inner.descriptor != reg.inner.descriptor
                || !req.inner.attributes.keys().eq(reg.inner.attributes.keys())
            {
                return Err(UnifyError::LocalMismatch);
            }
            let req_deps: Vec<_> = req
                .inner
                .attributes
                .values()
                .chain(req.inner.type_params.iter())
                .copied()
                .collect();
            let reg_deps: Vec<_> = reg
                .inner
                .attributes
                .values()
                .chain(reg.inner.type_params.iter())
                .copied()
                .collect();
            cross_unify_pairs(&req_deps, &reg_deps, arenas, bindings, visited)
        }

        (PyType::Callable(a), PyType::Callable(b)) => {
            let req = arenas.concrete.callables.get(&a).expect("dangling key");
            let reg = arenas.parametric.callables.get(&b).expect("dangling key");
            if !req.inner.params.keys().eq(reg.inner.params.keys())
                || req.inner.type_params.len() != reg.inner.type_params.len()
            {
                return Err(UnifyError::LocalMismatch);
            }
            let req_deps: Vec<_> = req
                .inner
                .params
                .values()
                .chain(std::iter::once(&req.inner.return_type))
                .chain(req.inner.type_params.iter())
                .copied()
                .collect();
            let reg_deps: Vec<_> = reg
                .inner
                .params
                .values()
                .chain(std::iter::once(&reg.inner.return_type))
                .chain(reg.inner.type_params.iter())
                .copied()
                .collect();
            cross_unify_pairs(&req_deps, &reg_deps, arenas, bindings, visited)
        }

        (PyType::LazyRef(a), PyType::LazyRef(b)) => {
            let req = arenas.concrete.lazy_refs.get(&a).expect("dangling key");
            let reg = arenas.parametric.lazy_refs.get(&b).expect("dangling key");
            cross_unify(
                req.inner.target,
                reg.inner.target,
                arenas,
                bindings,
                visited,
            )
        }

        _ => Err(UnifyError::VariantMismatch),
    }
}

// --- Convenience methods ---

impl<S: ArenaFamily> TypeArenas<S> {
    pub(crate) fn cross_unify(
        &self,
        request: PyTypeConcreteKey<S>,
        registration: PyTypeParametricKey<S>,
    ) -> Result<Bindings<S>, UnifyError> {
        let mut visited = HashSet::new();
        cross_unify(
            request,
            registration,
            self,
            Bindings::default(),
            &mut visited,
        )
    }

    pub(crate) fn cross_unify_callable_params(
        &self,
        request: CallableKey<S, Concrete>,
        registration: CallableKey<S, Parametric>,
    ) -> Result<Bindings<S>, UnifyError> {
        let req = self.concrete.callables.get(&request).expect("dangling key");
        let reg = self
            .parametric
            .callables
            .get(&registration)
            .expect("dangling key");
        if !req.inner.params.keys().eq(reg.inner.params.keys()) {
            return Err(UnifyError::LocalMismatch);
        }
        let req_deps: Vec<_> = req.inner.params.values().copied().collect();
        let reg_deps: Vec<_> = reg.inner.params.values().copied().collect();
        let mut visited = HashSet::new();
        cross_unify_pairs(
            &req_deps,
            &reg_deps,
            self,
            Bindings::default(),
            &mut visited,
        )
    }
}
