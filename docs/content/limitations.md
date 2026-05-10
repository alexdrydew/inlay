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

@compiled
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

## Cleanup exits cannot suppress a setup failure that happened before the context value was produced

Multiple `register_method` registrations for the same context-manager method stack into one Inlay context manager. When an inner setup step fails, Inlay runs the cleanup exits of the contexts that were already entered. If one of those cleanup exits returns truthy and suppresses the original failure, Inlay has nothing to give to the body of the user's `with` / `async with`.

```python
class Root(Protocol):
    def open(self) -> AbstractAsyncContextManager[Child]: ...

@final
class OuterManager:
    async def __aenter__(self) -> None: ...
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return True  # suppresses the inner setup failure

@final
class InnerManager:
    async def __aenter__(self) -> None:
        raise RuntimeError('inner setup failed')
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

registry = (
    Registry()
    .register_method(Root, Root.open)(lambda: OuterManager())
    .register_method(Root, Root.open)(lambda: InnerManager())
)
root = compile(Root, registry.build(), default_rules())

async with root.open() as child:    # raises RuntimeError
    use(child)
```

In this case Inlay raises `RuntimeError("async context manager enter did not produce a value")` (or `"context manager enter did not produce a value"` for the sync variant).

The cleanup exits run inside Inlay's synthesized `__aenter__`, before it has produced a context value. A truthy `__aexit__` return normally tells the runtime "exception handled, continue past the `async with`", but here `__aenter__` is still running and has not returned a value, so the only option to skip the body execution is to raise.

### Alternative: do not suppress setup failures in cleanup exits

Cleanup exits in the stack should propagate setup failures by returning falsy (or not returning a value).
