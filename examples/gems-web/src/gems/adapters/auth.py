from dataclasses import dataclass
from typing import Annotated, TypedDict

import jwt
import niquests
from jwt import PyJWK, PyJWKSet, get_unverified_header
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException

from gems.app import GemsContext
from gems.domain import UserId
from inlay import Registry, qual


class AuthState(TypedDict):
    user_id: UserId


@dataclass
class JwkClient:
    jwks_url: Annotated[str, qual('jwk_host')]

    async def get_jwk(self, kid: str) -> PyJWK:
        data = (
            (await niquests.aget(f'{self.jwks_url}/.well-known/jwks.json'))
            .raise_for_status()
            .json()
        )
        if not isinstance(data, dict):
            raise ValueError('JWK response must be a JSON object')

        try:
            return PyJWKSet.from_dict(data)[kid]
        except KeyError:
            raise ValueError('Invalid JWT token: unknown kid') from None


def _parse_bearer(headers: Headers) -> str:
    match headers.get('authorization', '').split(' ', 1):
        case ['Bearer', token] if token:
            return token
        case _:
            raise HTTPException(
                status_code=401,
                detail='Missing or invalid Authorization header',
                headers={'WWW-Authenticate': 'Bearer'},
            )


@dataclass
class JwkAuthenticator:
    client: JwkClient

    async def validate_headers(self, headers: Headers) -> AuthState:
        token = _parse_bearer(headers)
        unverified_header = get_unverified_header(token)
        kid = unverified_header.get('kid')
        if not isinstance(kid, str):
            raise ValueError('Invalid JWT token: missing kid header')

        decoded = jwt.decode(
            token,
            key=await self.client.get_jwk(kid),
            algorithms=['ES256'],
        )
        user_id = decoded.get('sub')
        if not isinstance(user_id, str):
            raise ValueError('Invalid JWT token: missing sub claim')
        return {'user_id': UserId(user_id)}


@dataclass
class DevAuthenticator:
    user_id: Annotated[str, qual('dev_user_id')]

    async def validate_headers(self, headers: Headers) -> AuthState:
        _ = headers
        return {'user_id': UserId(self.user_id)}


JWK_AUTH_REGISTRY = (
    Registry()
    .register(JwkClient)(JwkClient)
    .register_method(GemsContext, GemsContext.validate_headers)(JwkAuthenticator)
)

DEV_AUTH_REGISTRY = Registry().register_method(
    GemsContext, GemsContext.validate_headers
)(DevAuthenticator)
