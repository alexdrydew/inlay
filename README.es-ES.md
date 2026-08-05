

# Inlay

[![Checks](https://github.com/alexdrydew/inlay/actions/workflows/checks.yml/badge.svg)](https://github.com/alexdrydew/inlay/actions/workflows/checks.yml)
[![Docs](https://github.com/alexdrydew/inlay/actions/workflows/docs.yml/badge.svg)](https://alexdrydew.github.io/inlay/)
[![PyPI](https://img.shields.io/pypi/v/inlay.svg)](https://pypi.org/project/inlay/)
[![Python](https://img.shields.io/badge/python-%3E%3D3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Inlay es una biblioteca de Python para crear contextos de dependencia jerárquicos y tipados.

## ¿Qué es un contexto de dependencia?

Simplemente un tipo [`Protocol`](https://typing.python.org/en/latest/spec/protocol.html), que declara todas las dependencias necesarias para alguna parte de tu programa. Aquí hay un ejemplo muy básico:

```python
class UserHandlerContext(Protocol):
    user_id: str
    email_service: EmailService
    db_client: Database
```

Pero necesitas una implementación real para que este tipo sea útil. Este es el rol de la biblioteca Inlay: proporciona implementaciones en tiempo de ejecución seguras, eficientes y sin código repetitivo para cualquier contexto tipado, utilizando tanto dependencias preregistradas como valores proporcionados en el momento de la ejecución.

## ¿Cómo ayuda Inlay?

Ahora que quieres llamar a `handle_request`, necesitas una instancia de `UserHandlerContext`. Inlay ofrece una forma de ensamblarla a partir de dependencias construibles y valores proporcionados en el momento de la ejecución:

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

Aquí, Inlay generará la implementación para `make_user_ctx` en tiempo de ejecución. Las clases con métodos `__init__` tipados pueden construirse implícitamente, mientras que `user_id`, `email_api_key` y `db_uri` provienen de la llamada a la función `make_user_ctx`. Dado que este código se ejecuta muy temprano (durante la importación del módulo), cualquier dependencia faltante y/o ambigüedad de resolución se detectará a tiempo. Si la función `compiled` puede importarse, queda demostrado que es segura en cuanto a tipos.

## Atribuciones

- [Solucionador de traits de Rust Chalk](https://rust-lang.github.io/chalk/book/recursive.html): El solucionador de Inlay está fuertemente inspirado en el solucionador de traits recursivo de Chalk.

## Licencia

Licencia MIT. Consulta [LICENSE](LICENSE) para más detalles.
