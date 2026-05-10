# pyright: reportUnusedParameter=false
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from starlette.applications import Starlette
from starlette.routing import Route

from gems.adapters.auth import DEV_AUTH_REGISTRY, JWK_AUTH_REGISTRY
from gems.adapters.market import MARKET_REGISTRY
from gems.adapters.postgres import POSTGRES_REGISTRY
from gems.adapters.sqlite import SQLITE_REGISTRY
from gems.app import (
    GemsContext,
    State,
    appraise_gem_endpoint,
    create_gem,
    delete_gem,
    get_gem,
    list_gems,
    list_gems_internal,
    update_gem,
)
from inlay import Registry, compiled, qual

registry = Registry().include(MARKET_REGISTRY)

match os.getenv('AUTH_PROVIDER', 'jwk'):
    case 'jwk':
        registry = registry.include(JWK_AUTH_REGISTRY).register_value(
            Annotated[str, qual('jwk_host')]
        )(os.environ['JWKS_URL'])
    case 'dev':
        registry = registry.include(DEV_AUTH_REGISTRY).register_value(
            Annotated[str, qual('dev_user_id')]
        )(os.getenv('DEV_USER_ID', 'dev-user'))
    case auth_provider:
        raise ValueError(f'Unknown auth provider: {auth_provider}')

match os.getenv('STORAGE', 'sqlite'):
    case 'postgres':
        registry = registry.include(POSTGRES_REGISTRY).register_value(
            Annotated[str, qual('psql_uri')]
        )(os.environ['POSTGRES_URI'])
    case 'sqlite':
        registry = registry.include(SQLITE_REGISTRY).register_value(
            Annotated[str, qual('sqlite_path')]
        )(os.getenv('SQLITE_PATH', 'gems.sqlite'))
    case storage:
        raise ValueError(f'Unknown storage: {storage}')


@compiled(registry)
def make_ctx() -> GemsContext: ...


@asynccontextmanager
async def lifespan(_app: Starlette) -> AsyncGenerator[State]:
    yield {'ctx': make_ctx()}


app = Starlette(
    routes=[
        Route('/api/gems', list_gems, methods=['GET']),
        Route('/api/gems', create_gem, methods=['POST']),
        Route('/api/gems/{gem_id:int}', get_gem, methods=['GET']),
        Route('/api/gems/{gem_id:int}', update_gem, methods=['PUT']),
        Route('/api/gems/{gem_id:int}', delete_gem, methods=['DELETE']),
        Route(
            '/api/gems/{gem_id:int}/appraise',
            appraise_gem_endpoint,
            methods=['POST'],
        ),
        Route('/internal/gems', list_gems_internal),
    ],
    lifespan=lifespan,
)
