from abc import ABCMeta, abstractmethod
from typing import Protocol, TypedDict

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from gems.domain import Gem, GemName, UserId


class GemsRepository(metaclass=ABCMeta):
    @abstractmethod
    async def list_gems(self) -> list[Gem]: ...


class AuthorizedGemsContext(Protocol):
    @property
    def user_id(self) -> UserId: ...

    @property
    def user_gems_repository(self) -> GemsRepository: ...


class GemsContext(Protocol):
    async def validate_headers(self, headers: Headers) -> AuthorizedGemsContext: ...

    def authorize_user(self, user_id: UserId) -> AuthorizedGemsContext: ...


class State(TypedDict):
    ctx: GemsContext


class GemResponse(TypedDict):
    id: int
    name: GemName
    carat: float
    created_at: str
    updated_at: str


async def _list_authorized_gems(authorized_ctx: AuthorizedGemsContext) -> JSONResponse:
    return JSONResponse([
        GemResponse(
            id=gem.id,
            name=gem.name,
            carat=gem.carat,
            created_at=gem.created_at.isoformat(),
            updated_at=gem.updated_at.isoformat(),
        )
        for gem in await authorized_ctx.user_gems_repository.list_gems()
    ])


async def list_gems(request: Request[State]) -> JSONResponse:
    return await _list_authorized_gems(
        await request.state['ctx'].validate_headers(request.headers)
    )


async def list_internal_gems(request: Request[State]) -> JSONResponse:
    match request.query_params.get('user_id'):
        case str() as user_id if user_id:
            return await _list_authorized_gems(
                request.state['ctx'].authorize_user(UserId(user_id))
            )
        case _:
            raise HTTPException(
                status_code=400,
                detail='Missing user_id query parameter',
            )
