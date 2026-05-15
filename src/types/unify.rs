use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use super::{
    CallableKey, Concrete, Keyed, Parametric, ProtocolBase, ProtocolKey, PyType, PyTypeConcreteKey,
    PyTypeId, PyTypeParametricKey, Qual, TypeArenas, UnqualifiedMode,
};

#[derive(Default)]
pub(crate) struct Bindings<'ty> {
    pub(crate) type_vars: HashMap<PyTypeId, PyTypeConcreteKey<'ty>>,
    pub(crate) param_specs: HashMap<PyTypeId, PyTypeConcreteKey<'ty>>,
}

#[derive(Debug)]
pub(crate) enum UnifyError {
    VariantMismatch,
    LocalMismatch,
    DepCountMismatch,
    ConflictingBinding,
}

fn cross_unify_pairs<'ty>(
    requests: &[PyTypeConcreteKey<'ty>],
    registrations: &[PyTypeParametricKey<'ty>],
    arenas: &TypeArenas<'ty>,
    bindings: Bindings<'ty>,
    visited: &mut HashSet<(PyTypeConcreteKey<'ty>, PyTypeParametricKey<'ty>)>,
) -> Result<Bindings<'ty>, UnifyError> {
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

fn cross_unify<'ty>(
    request: PyTypeConcreteKey<'ty>,
    registration: PyTypeParametricKey<'ty>,
    arenas: &TypeArenas<'ty>,
    bindings: Bindings<'ty>,
    visited: &mut HashSet<(PyTypeConcreteKey<'ty>, PyTypeParametricKey<'ty>)>,
) -> Result<Bindings<'ty>, UnifyError> {
    // TypeVar binding — keyed by Python TypeVar identity (PyTypeId),
    // not arena slot key. The same logical TypeVar may have different
    // slot keys due to different qualifier contexts (return vs params).
    if let PyType::TypeVar(tv_key) = registration {
        let tv_id = arenas
            .parametric
            .type_vars
            .get(tv_key)
            .inner
            .descriptor
            .id
            .clone();
        return match bindings.type_vars.get(&tv_id) {
            Some(&existing) => {
                if arenas.deep_eq_concrete::<UnqualifiedMode>(existing, request) {
                    Ok(bindings)
                } else {
                    Err(UnifyError::ConflictingBinding)
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
            .get(ps_key)
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

fn cross_unify_known<'ty>(
    request: PyTypeConcreteKey<'ty>,
    registration: PyTypeParametricKey<'ty>,
    arenas: &TypeArenas<'ty>,
    bindings: Bindings<'ty>,
    visited: &mut HashSet<(PyTypeConcreteKey<'ty>, PyTypeParametricKey<'ty>)>,
) -> Result<Bindings<'ty>, UnifyError> {
    match (request, registration) {
        (PyType::Sentinel(a), PyType::Sentinel(b)) => {
            let req = arenas.sentinels.get(a);
            let reg = arenas.sentinels.get(b);
            if req.inner.value != reg.inner.value {
                return Err(UnifyError::LocalMismatch);
            }
            Ok(bindings)
        }

        (PyType::Plain(a), PyType::Plain(b)) => {
            let req = arenas.concrete.plains.get(a);
            let reg = arenas.parametric.plains.get(b);
            if req.inner.descriptor != reg.inner.descriptor {
                return Err(UnifyError::LocalMismatch);
            }
            cross_unify_pairs(&req.inner.args, &reg.inner.args, arenas, bindings, visited)
        }

        (PyType::Class(a), PyType::Class(b)) => {
            let req = arenas.concrete.classes.get(a);
            let reg = arenas.parametric.classes.get(b);
            if req.inner.descriptor != reg.inner.descriptor {
                return Err(UnifyError::LocalMismatch);
            }
            let mut req_deps = req.inner.args.clone();
            let mut reg_deps = reg.inner.args.clone();
            if let (Some(req_init), Some(reg_init)) = (&req.inner.init, &reg.inner.init) {
                if !req_init.params.keys().eq(reg_init.params.keys())
                    || req_init.param_kinds != reg_init.param_kinds
                    || req_init.param_has_default != reg_init.param_has_default
                {
                    return Err(UnifyError::LocalMismatch);
                }
                req_deps.extend(req_init.params.values().copied());
                reg_deps.extend(reg_init.params.values().copied());
            } else if req.inner.init.is_some() != reg.inner.init.is_some() {
                return Err(UnifyError::LocalMismatch);
            }
            cross_unify_pairs(&req_deps, &reg_deps, arenas, bindings, visited)
        }

        (PyType::Union(a), PyType::Union(b)) => {
            let req = arenas.concrete.unions.get(a);
            let reg = arenas.parametric.unions.get(b);
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
            let req = arenas.concrete.protocols.get(a);
            let reg = arenas.parametric.protocols.get(b);
            if req.inner.descriptor != reg.inner.descriptor
                || req.inner.protocol_mro.len() != reg.inner.protocol_mro.len()
                || !req
                    .inner
                    .protocol_mro
                    .iter()
                    .zip(reg.inner.protocol_mro.iter())
                    .all(|(req_base, reg_base)| {
                        req_base.descriptor == reg_base.descriptor
                            && req_base.direct_methods == reg_base.direct_methods
                            && req_base.type_params.len() == reg_base.type_params.len()
                    })
                || req.inner.direct_methods != reg.inner.direct_methods
                || !req.inner.methods.iter().map(|(name, _)| name).eq(reg
                    .inner
                    .methods
                    .iter()
                    .map(|(name, _)| name))
                || !req.inner.attributes.iter().map(|(name, _)| name).eq(reg
                    .inner
                    .attributes
                    .iter()
                    .map(|(name, _)| name))
                || !req.inner.properties.iter().map(|(name, _)| name).eq(reg
                    .inner
                    .properties
                    .iter()
                    .map(|(name, _)| name))
            {
                return Err(UnifyError::LocalMismatch);
            }
            let req_deps: Vec<_> = req
                .inner
                .methods
                .iter()
                .map(|(_, method)| &method.callable)
                .chain(req.inner.attributes.iter().map(|(_, value)| value))
                .chain(req.inner.properties.iter().map(|(_, value)| value))
                .chain(req.inner.type_params.iter())
                .chain(
                    req.inner
                        .protocol_mro
                        .iter()
                        .flat_map(|base| base.type_params.iter()),
                )
                .copied()
                .collect();
            let reg_deps: Vec<_> = reg
                .inner
                .methods
                .iter()
                .map(|(_, method)| &method.callable)
                .chain(reg.inner.attributes.iter().map(|(_, value)| value))
                .chain(reg.inner.properties.iter().map(|(_, value)| value))
                .chain(reg.inner.type_params.iter())
                .chain(
                    reg.inner
                        .protocol_mro
                        .iter()
                        .flat_map(|base| base.type_params.iter()),
                )
                .copied()
                .collect();
            cross_unify_pairs(&req_deps, &reg_deps, arenas, bindings, visited)
        }

        (PyType::TypedDict(a), PyType::TypedDict(b)) => {
            let req = arenas.concrete.typed_dicts.get(a);
            let reg = arenas.parametric.typed_dicts.get(b);
            if req.inner.descriptor != reg.inner.descriptor
                || !req.inner.attributes.iter().map(|(name, _)| name).eq(reg
                    .inner
                    .attributes
                    .iter()
                    .map(|(name, _)| name))
            {
                return Err(UnifyError::LocalMismatch);
            }
            let req_deps: Vec<_> = req
                .inner
                .attributes
                .iter()
                .map(|(_, value)| value)
                .chain(req.inner.type_params.iter())
                .copied()
                .collect();
            let reg_deps: Vec<_> = reg
                .inner
                .attributes
                .iter()
                .map(|(_, value)| value)
                .chain(reg.inner.type_params.iter())
                .copied()
                .collect();
            cross_unify_pairs(&req_deps, &reg_deps, arenas, bindings, visited)
        }

        (PyType::Callable(a), PyType::Callable(b)) => {
            let req = arenas.concrete.callables.get(a);
            let reg = arenas.parametric.callables.get(b);
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
            let req = arenas.concrete.lazy_refs.get(a);
            let reg = arenas.parametric.lazy_refs.get(b);
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

impl<'ty> TypeArenas<'ty> {
    pub(crate) fn cross_unify(
        &self,
        request: PyTypeConcreteKey<'ty>,
        registration: PyTypeParametricKey<'ty>,
    ) -> Result<Bindings<'ty>, UnifyError> {
        let mut visited = HashSet::default();
        cross_unify(
            request,
            registration,
            self,
            Bindings::default(),
            &mut visited,
        )
    }

    pub(crate) fn cross_unify_callable_signature_with_bindings(
        &self,
        request: CallableKey<'ty, Concrete>,
        registration: CallableKey<'ty, Parametric>,
        bindings: Bindings<'ty>,
    ) -> Result<Bindings<'ty>, UnifyError> {
        let req = self.concrete.callables.get(request);
        let reg = self.parametric.callables.get(registration);
        if !req.inner.params.keys().eq(reg.inner.params.keys()) {
            return Err(UnifyError::LocalMismatch);
        }
        if req.inner.param_kinds != reg.inner.param_kinds {
            return Err(UnifyError::LocalMismatch);
        }
        if req.inner.accepts_varargs && !reg.inner.accepts_varargs {
            return Err(UnifyError::LocalMismatch);
        }
        if req.inner.accepts_varkw && !reg.inner.accepts_varkw {
            return Err(UnifyError::LocalMismatch);
        }
        if req.inner.return_wrapper != reg.inner.return_wrapper {
            return Err(UnifyError::LocalMismatch);
        }
        let req_deps: Vec<_> = req
            .inner
            .params
            .values()
            .copied()
            .chain(std::iter::once(req.inner.return_type))
            .collect();
        let reg_deps: Vec<_> = reg
            .inner
            .params
            .values()
            .copied()
            .chain(std::iter::once(reg.inner.return_type))
            .collect();
        let mut visited = HashSet::default();
        cross_unify_pairs(&req_deps, &reg_deps, self, bindings, &mut visited)
    }

    pub(crate) fn cross_unify_protocol_base(
        &self,
        request: &ProtocolBase<Qual<Keyed<'ty>>, Concrete>,
        registration: ProtocolKey<'ty, Parametric>,
    ) -> Result<Bindings<'ty>, UnifyError> {
        let reg = self.parametric.protocols.get(registration);
        if request.descriptor != reg.inner.descriptor
            || request.direct_methods != reg.inner.direct_methods
            || request.type_params.len() != reg.inner.type_params.len()
        {
            return Err(UnifyError::LocalMismatch);
        }

        let mut visited = HashSet::default();
        cross_unify_pairs(
            &request.type_params,
            &reg.inner.type_params,
            self,
            Bindings::default(),
            &mut visited,
        )
    }
}
