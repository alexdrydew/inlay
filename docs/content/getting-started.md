---
title: Getting Started
description: Getting started with Inlay
sidebar_position: 2
hide_table_of_contents: true
---

# Getting Started

A quick tour of installing Inlay and assembling your first dependency context. If you haven't yet, the [introduction](/) explains what dependency contexts are and why Inlay represents them as `Protocol` types.

## Install

Inlay requires Python 3.14 or newer.

```bash
pip install inlay
```


## A minimal example

Declare what your code needs as a `Protocol`, register concrete constructors, and let Inlay assemble an implementation:

```python
from typing import Protocol
from inlay import RegistryBuilder, compiled

class Database:
    def __init__(self, url: str) -> None:
        self.url = url

class UserService:
    def __init__(self, db: Database) -> None:
        self.db = db

class AppContext(Protocol):
    @property
    def users(self) -> UserService: ...

registry = (
    RegistryBuilder()
    .register(Database)(Database)
    .register(UserService)(UserService)
)

@compiled(registry)
def make_app(url: str) -> AppContext: ...

app = make_app(url='postgres://localhost/app')
assert app.users.db.url == 'postgres://localhost/app'
```

A few things to notice:

* `make_app` has no body. The `@compiled` decorator inspects its signature, solves the dependency graph against `registry`, and replaces it with a generated implementation.
* `url` is a runtime parameter. It flows from the caller into `Database.__init__` because the solver matched the parameter name and type to a constructor argument it could not satisfy from the registry alone.
* The resolution happens once, at module import time. If anything is unsatisfiable, `make_app` fails to compile, so the program never starts in a partially-wired state.

## Hierarchical contexts

In real applications, some dependencies only exist after a runtime event — a request arrives, a user authenticates, a transaction begins. Imagine a `UserRepository` that scopes database access to a specific user:

```python
class UserRepository:
    def __init__(self, db: Database, user_id: str) -> None:
        self.db = db
        self.user_id = user_id
```

`Database` is fine to wire up at startup, but `user_id` is per-request — there's no sensible value to register globally. We need a way to express "`UserRepository` is only constructible after a user authenticates."

Inlay models this with a **transition**: a method on the parent context that returns a *child* context with extra fields in scope. Let's build it up.

### Declare the authorized scope

First, declare what becomes reachable once a user is authenticated:

```python
class AuthorizedContext(Protocol):
    @property
    def repo(self) -> UserRepository: ...
```

Nothing here mentions `user_id` — that's an implementation detail of `UserRepository`. The child context only declares its user-facing surface.

### Add the transition to the parent

Next, advertise the entry point on the root context:

```python
class AppContext(Protocol):
    def authorize(self, token: str) -> AuthorizedContext: ...
```

This is still just a `Protocol`. We haven't told Inlay how `authorize` works — only that `AppContext` is anything with a method matching this signature.

### Implement the transition

The implementation is just a function that returns the *new fields* the child scope gains over the parent. A `TypedDict` is a clean way to express those fields:

```python
from typing import TypedDict

class AuthorizedFields(TypedDict):
    user_id: str

def authorize(token: str) -> AuthorizedFields:
    # validate token, look up user, etc.
    return {'user_id': 'u-123'}
```

When this function runs, its return value contributes a `user_id: str` into the child context's resolution scope. The child inherits everything from the parent (so `Database` is still available) and adds these new fields on top.

### Wire it up

`register_method` binds the function to the protocol method:

```python
registry = (
    RegistryBuilder()
    .register(Database)(Database)
    .register(UserRepository)(UserRepository)
    .register_method(AppContext, AppContext.authorize)(authorize)
)
```

Inlay can now resolve `AuthorizedContext.repo`: inside the authorized scope it has `Database` (inherited from the parent) and `user_id` (introduced by the transition's return type), which is everything `UserRepository.__init__` requires.

### Compile and call it

The final step is the same as before — compile the root and walk through the transition:

```python
@compiled(registry)
def make_app(url: str) -> AppContext: ...

app = make_app(url='postgres://localhost/app')
authorized = app.authorize(token='...')
assert authorized.repo.user_id == 'u-123'
```

`UserRepository` is constructed *only* once `authorize(...)` is called — up to that point no `user_id` exists, and Inlay never attempts to build it. The same pattern composes recursively: child contexts can declare their own transitions.

## Where to go next

* [How does it work](how-does-it-work.md) — the `compile()` model and what kinds of targets it supports.
* The [`gems-web` example](https://github.com/alexdrydew/inlay/tree/main/examples/gems-web) — a full Starlette application showing modular registries, qualifiers, async transitions, and pluggable backends.
