---
title: Cells
description: Live reads and writable dependency values.
sidebar_position: 3
hide_table_of_contents: true
---

# Cells

`ReadCell[T]` defers reading a dependency until `get()` is called:

```python
from inlay import ReadCell

def make_service(config: ReadCell[Config]) -> Service:
    return Service(config)
```

The read is live. Dynamic targets are recomputed on each call, while static targets return their stored value. A cell cannot read its target while that target is still being constructed.

`Cell[T]` adds `set(value)`:

```python
from inlay import Cell

class Root(Protocol):
    config: Cell[Config]

root.config.set(new_config)
assert root.config.get() is new_config
```

Setting a cell updates the target and invalidates cached values that depend on it. Static computed values, fields, and execution inputs can be cell targets. Dynamic values can only be read through `ReadCell`.

## Static policies

Constructor and init rules default to `static_policy='always'`. Custom rule graphs can select:

- `'always'`: the result has stable writable storage;
- `'never'`: the result is dynamic;
- `'if_static_dependencies'`: the result is static only when its eager dependencies are static.

A `ReadCell` dependency does not make an `if_static_dependencies` result dynamic because it is a handle rather than an eager value.

## Aggregates

Synthesized Protocol and TypedDict objects are dynamic aggregates and cannot be replaced as whole Cell targets. Their individual writable members remain assignable through the generated proxy or dictionary.
