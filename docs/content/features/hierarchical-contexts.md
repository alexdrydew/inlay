---
title: Hierarchical Contexts
description: Modeling runtime-dependent scopes with child contexts and transitions.
sidebar_position: 1
hide_table_of_contents: true
---

In real applications, some dependencies only exist after a runtime event — a request arrives, a user authenticates, a transaction begins. Imagine a `UserRepository` that scopes database access to a specific user:

```python
class UserRepository:
    def __init__(self, db: BaseDatabase, user_id: str) -> None:
        self.db = db
        self.user_id = user_id
```

`BaseDatabase` is fine to wire up at startup, but `user_id` is per-request — there's no sensible value to register globally. We need a way to express "`UserRepository` is only constructible after a user authenticates."

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
    Registry()
    .register(BaseDatabase)(PostgresDatabase)
    .register_method(AppContext, AppContext.authorize)(authorize)
)
```

Inlay can now resolve `AuthorizedContext.repo`: inside the authorized scope it has `PostgresDatabase` (inherited from the parent) and `user_id` (introduced by the transition's return type), which is everything `UserRepository.__init__` requires.

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
