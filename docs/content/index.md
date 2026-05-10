---
title: Inlay
description: Typed dependency contexts for Python.
sidebar_position: 1
hide_table_of_contents: true
---

# Inlay

Inlay is a Python library for building typed hierarchical dependency contexts.

## What is a dependency context

Just a [`Protocol`](https://typing.python.org/en/latest/spec/protocol.html) type[^1], which declares all of the dependencies that are needed for some part of your program. Here is a very basic example:

```python
class UserHandlerContext(Protocol):
    user_id: str
    email_service: EmailService
    db_client: Database
```

But you need an actual implementation for this type to be useful. This is the role of Inlay library: it provides safe, performant and boilerplate free runtime implementations for any typed contexts using both pre-registered dependencies and values provided at the time of execution.

## How Inlay helps

Now that you want to call `handle_request` you need an instance of `UserHandlerContext`. Inlay offers a way to assemble it from constructible dependencies and values provided at execution:
```python
from inlay import compiled

class EmailService:
    def __init__(self, email_api_key: str):
        ...

class Database:
    def __init__(self, db_uri: str):
        ...

# highlight-start
@compiled
def make_user_ctx(
    user_id: str,
    email_api_key: str,
    db_uri: str,
) -> UserHandlerContext:
    ...  # note: implementation is not required!
# highlight-end

ctx = make_user_ctx(
    user_id="u-123",
    email_api_key="...",
    db_uri="...",
)
handle_request(ctx)
```

Here inlay will generate implementation for `make_user_ctx` in runtime. Classes with typed `__init__` methods can be constructed implicitly, while `user_id`, `email_api_key`, and `db_uri` come from the `make_user_ctx` function call. Because this code is executed very early (during module import) any missing dependencies and/or resolution ambiguities will be caught early. If `compiled` function can be imported it is proven to be type safe.

## Why use dependency contexts

Using protocols to express available dependencies has the following benefits:

* Protocol types are well understood by all Python type checkers, protocols can be extended and intersected by subclassing, made generic, etc.
* Because protocols use structural subtyping rules, your functions and classes can declare only what they actually need.

  ```python
  class NeedsEmail(Protocol):
      email_service: EmailService

  def send_welcome(ctx: NeedsEmail) -> None: ...

  send_welcome(user_handler_context)  # ok: UserHandlerContext is a NeedsEmail
  ```
* Contexts are easy to thread through the call stack.

  ```python
  def handle_request(ctx: UserHandlerContext) -> None:
      register_user(ctx)

  class RegistrationContext(Protocol):
      user_id: str
      email_service: EmailService

  def register_user(ctx: RegistrationContext) -> None:
      print(f"registering {ctx.user_id}")
      send_welcome(ctx)
  ```
* Most of the program is free of any additional dependency injection metadata and libraries.

## But there is more

We used a very basic context in this example, real world applications tends to become much more complex and Inlay supports you through this journey:
* contexts can be hierarchical (in the real world you don't have user id from the beginning), i.e. have methods that return extended contexts (including async methods and context managers);
* contexts can be nested recursively;
* dependency implementations can be made swappable with explicitly configured `Registry`. Registries are modular so common dependency sets can be shared across applications and modules;
* sometimes dependencies can even be circular (with some reasonable restrictions).

[^1]: plain classes and typed dicts are also supported as contexts
