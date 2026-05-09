---
title: Limitations
description: Patterns that Inlay does not support and recommended alternatives.
sidebar_position: 4
hide_table_of_contents: true
---

# Limitations

Inlay resolves dependencies into a static graph at compile time. Some runtime patterns cannot be expressed in that model and have to be modeled differently.

## Dependency construction must not read from the globally available compiled context

A constructor that builds a member of the compiled context must not depend on the results of the compilation while it runs.

```python
class Ctx(Protocol):
    a: A
    b: B

class A:
    def __init__(self) -> None:
        self.b = app.b      # reads back into the compiled context

registry = (
    RegistryBuilder()
    .register(A)(A)
    .register(B)(B)
)

@compiled(registry)
def make_ctx() -> Ctx: ...

app = make_ctx()
_ = app.a                   # triggers A.__init__, which reads app.b
```

Inlay materializes context members lazily. Reading `app.b` while `app.a` is still being built creates a runtime ordering dependency that the static dependency graph does not see. It can produce runtime errors on the context proxy.

### Alternative: use real graph edges instead

Take the dependency directly as a constructor argument so it becomes a real graph edge:

```python
class A:
    def __init__(self, b: B) -> None:
        self.b = b
```

If the reference is genuinely cyclic, use `LazyRef[T]` so the cycle is broken explicitly and resolved after construction:

```python
from inlay import LazyRef

class A:
    def __init__(self, b: LazyRef[B]) -> None:
        self.b = b
```

`LazyRef.get()` may only be called after the context is fully built, which is exactly the constraint that makes this safe.
