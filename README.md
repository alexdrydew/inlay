# Inlay

[![Docs](https://github.com/alexdrydew/inlay/actions/workflows/docs.yml/badge.svg)](https://alexdrydew.github.io/inlay/)

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

@compiled
def make_user_ctx(
    user_id: str,
    email_api_key: str,
    db_uri: str,
) -> UserHandlerContext:
    ...  # note: implementation is not required!

ctx = make_user_ctx(
    user_id="u-123",
    email_api_key="...",
    db_uri="...",
)
handle_request(ctx)
```

Here inlay will generate implementation for `make_user_ctx` in runtime. Classes with typed `__init__` methods can be constructed implicitly, while `user_id`, `email_api_key`, and `db_uri` come from the `make_user_ctx` function call. Because this code is executed very early (during module import) any missing dependencies and/or resolution ambiguities will be caught early. If `compiled` function can be imported it is proven to be type safe.


