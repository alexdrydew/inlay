---
title: Getting Started
description: Getting started with Inlay
sidebar_position: 2
hide_table_of_contents: true
---

# Getting Started

## Install

Inlay requires Python 3.14 or newer.

```bash
pip install inlay
```


## A minimal example

Declare what your code needs as an `AppContext` protocol, register concrete types for interfaces, and let Inlay assemble an implementation:

```python
from typing import Protocol
from abc import ABCMeta
from inlay import Registry, compiled

class BaseDatabase(metaclass=ABCMeta):
    @property
    def url(self) -> str: ...

class PostgresDatabase(BaseDatabase):
    def __init__(self, url: str) -> None:
        self._url = url

    @property
    def url(self) -> str:
        return self._url

class UserService:
    def __init__(self, db: BaseDatabase) -> None:
        self.db = db

class AppContext(Protocol):
    @property
    def users(self) -> UserService: ...

registry = (
    Registry()
    .register(BaseDatabase)(PostgresDatabase)
)

@compiled(registry)
def make_app(url: str) -> AppContext: ...

app = make_app(url='postgres://localhost/app')
assert app.users.db.url == 'postgres://localhost/app'
```

A few things to notice:

* `make_app` has no body. The `@compiled` decorator inspects its signature, solves the dependency graph against `registry`, and replaces it with a generated implementation.
* `url` is a runtime parameter. It flows from the caller into `Database.__init__` because the solver matched the parameter name and type to a constructor argument.
* The resolution happens once, at module import time. If anything is unsatisfiable or ambiguous, `make_app` fails to compile, so the program never starts in a partially-wired state.

## Where to go next

* [Features](features/hierarchical-contexts.md) — explore full capabilities of the library.
* The [`gems-web` example](https://github.com/alexdrydew/inlay/tree/main/examples/gems-web) — a full Starlette application showing modular registries, qualifiers, async transitions, and pluggable backends.
