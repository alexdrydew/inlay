---
title: Inlay
description: Typed dependency contexts for Python.
sidebar_position: 1
hide_table_of_contents: true
---

# Inlay

Inlay is a Python library for building typed hierarchical dependency contexts.

## What is a dependency context

Inlay answer is simple: just a [`Protocol`](https://typing.python.org/en/latest/spec/protocol.html) type[^1], which declares all of the dependencies that are needed for some part of your program. Here is a very basic example:

```python
class UserHandlerContext(Protocol):
    user_id: str
    email_service: EmailService
    db_client: Database
```

But you need an actual implementation for this type. This is the role of Inlay library: it provides safe, performant and boilerplate free runtime implementations for any typed contexts using both pre-registered dependencies and values provided at the time of execution.


## Why use dependency contexts

Using protocols to express available dependencies has has the following benefits:

* Protocol types are well understood by all the Python type checkers (they can be extended by subclassing, intersected, made generic, etc).
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
* Actually interesting parts of the program are free of any additional dependency injection metadata and libraries.

## How Inlay helps

Now that you wants to call `handle_request` you need an instance of `UserHandlerContext`. You could supply it manually

```python
class EmailService:
    def __init__(self, api_key: str) -> None: ...

class Database:
    def __init__(self, url: str) -> None: ...

@dataclass
class UserHandlerContextImpl:
    user_id: str
    email_service: EmailService
    db_client: Database

...

handle_request(
    UserHandlerContextImpl(
        user_id="u-123",
        email_service=EmailService(api_key="..."),
        db_client=Database(url="..."),
    )
)
```

Inlay offers a more ergonomic and scalable alternative:
```python
from inlay import RegistryBuilder, compiled

registry = (
    RegistryBuilder()
    .register(EmailService)(EmailService)
    .register(Database)(Database)
)

@compiled(registry)
def make_user_ctx(
    user_id: str,
    api_key: str,
    url: str,
) -> UserHandlerContext:
    ...  # note: implementation is not required!

ctx = make_user_ctx(
    user_id="u-123",
    api_key="...",
    url="...",
)
handle_request(ctx)
```

## But there is more

We used a very basic context in this example, real world applications are much more complex and Inlay supports you through this journey:
* Contexts can be hierarchical (in real world you don't have user id from the beginning)
* Dependency can have ambigious types

[^1]: usual nominal classes and typed dicts are also available

