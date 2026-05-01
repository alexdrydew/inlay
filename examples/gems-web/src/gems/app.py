from abc import ABCMeta, abstractmethod
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Protocol, TypedDict

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from gems.domain import (
    Gem,
    GemAppraisal,
    GemId,
    GemMarketQuote,
    GemName,
    UserId,
    appraise_gem,
    gem_name,
)


class GemsRepository(metaclass=ABCMeta):
    @abstractmethod
    async def list_gem_ids(self) -> list[GemId]: ...

    @abstractmethod
    async def get_gem(self, gem_id: GemId) -> Gem | None: ...

    @abstractmethod
    async def create_gem(self, name: GemName, carat: Decimal) -> Gem: ...

    @abstractmethod
    async def update_gem(
        self,
        gem_id: GemId,
        name: GemName,
        carat: Decimal,
    ) -> Gem | None: ...

    @abstractmethod
    async def delete_gem(self, gem_id: GemId) -> bool: ...

    @abstractmethod
    async def save_appraisal(
        self,
        gem_id: GemId,
        appraisal: GemAppraisal,
    ) -> Gem | None: ...


class GemMarketData(metaclass=ABCMeta):
    @abstractmethod
    async def quote_for[GemNameT: GemName](
        self,
        name: GemNameT,
    ) -> GemMarketQuote[GemNameT] | None: ...


class AuthorizedGemsContext(Protocol):
    @property
    def user_id(self) -> UserId: ...

    @property
    def user_gems_repository(self) -> GemsRepository: ...

    @property
    def gem_market_data(self) -> GemMarketData: ...


class GemsContext(Protocol):
    async def validate_headers(self, headers: Headers) -> AuthorizedGemsContext: ...

    def authorize_user(self, user_id: UserId) -> AuthorizedGemsContext: ...


class State(TypedDict):
    ctx: GemsContext


class GemResponse(TypedDict):
    id: int
    name: GemName
    carat: str
    appraisal_usd: str | None
    appraisal_source: str | None
    appraised_at: str | None
    created_at: str
    updated_at: str


class GemInput(TypedDict):
    name: GemName
    carat: Decimal


def _gem_response(gem: Gem) -> GemResponse:
    return GemResponse(
        id=gem.id,
        name=gem.name,
        carat=str(gem.carat),
        appraisal_usd=str(gem.appraisal_usd) if gem.appraisal_usd is not None else None,
        appraisal_source=gem.appraisal_source,
        appraised_at=gem.appraised_at.isoformat()
        if gem.appraised_at is not None
        else None,
        created_at=gem.created_at.isoformat(),
        updated_at=gem.updated_at.isoformat(),
    )


def _parse_gem_id(request: Request[State]) -> GemId:
    match request.path_params['gem_id']:
        case int() as gem_id if gem_id > 0:
            return GemId(gem_id)
        case _:
            raise HTTPException(status_code=404, detail='Gem not found')


def _parse_gem_input(data: object) -> GemInput:
    match data:
        case {'name': str(), 'carat': bool()}:
            raise HTTPException(status_code=400, detail='Gem carat must be a number')
        case {'name': str(name), 'carat': int() | float() | str() as carat}:
            try:
                parsed_carat = Decimal(str(carat))
                if not parsed_carat.is_finite():
                    raise ValueError('Gem carat must be finite')
                if parsed_carat <= 0:
                    raise ValueError('Gem carat must be positive')
                return {'name': gem_name(name), 'carat': parsed_carat}
            except InvalidOperation as exc:
                raise HTTPException(
                    status_code=400, detail='Gem carat must be a number'
                ) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        case _:
            raise HTTPException(
                status_code=400,
                detail='Expected JSON body with name and positive carat',
            )


async def _authorized_ctx(request: Request[State]) -> AuthorizedGemsContext:
    return await request.state['ctx'].validate_headers(request.headers)


async def _list_authorized_gem_ids(
    authorized_ctx: AuthorizedGemsContext,
) -> JSONResponse:
    return JSONResponse([
        int(gem_id)
        for gem_id in await authorized_ctx.user_gems_repository.list_gem_ids()
    ])


def _not_found() -> HTTPException:
    return HTTPException(status_code=404, detail='Gem not found')


async def list_gems(request: Request[State]) -> JSONResponse:
    return await _list_authorized_gem_ids(await _authorized_ctx(request))


async def create_gem(request: Request[State]) -> JSONResponse:
    authorized_ctx = await _authorized_ctx(request)
    gem_input = _parse_gem_input(await request.json())
    return JSONResponse(
        _gem_response(
            await authorized_ctx.user_gems_repository.create_gem(
                gem_input['name'],
                gem_input['carat'],
            )
        ),
        status_code=201,
    )


async def get_gem(request: Request[State]) -> JSONResponse:
    match await (await _authorized_ctx(request)).user_gems_repository.get_gem(
        _parse_gem_id(request)
    ):
        case Gem() as gem:
            return JSONResponse(_gem_response(gem))
        case None:
            raise _not_found()


async def update_gem(request: Request[State]) -> JSONResponse:
    authorized_ctx = await _authorized_ctx(request)
    gem_input = _parse_gem_input(await request.json())
    match await authorized_ctx.user_gems_repository.update_gem(
        _parse_gem_id(request),
        gem_input['name'],
        gem_input['carat'],
    ):
        case Gem() as gem:
            return JSONResponse(_gem_response(gem))
        case None:
            raise _not_found()


async def delete_gem(request: Request[State]) -> Response:
    if await (await _authorized_ctx(request)).user_gems_repository.delete_gem(
        _parse_gem_id(request)
    ):
        return Response(status_code=204)
    raise _not_found()


async def appraise_gem_endpoint(request: Request[State]) -> JSONResponse:
    authorized_ctx = await _authorized_ctx(request)
    gem_id = _parse_gem_id(request)
    match await authorized_ctx.user_gems_repository.get_gem(gem_id):
        case Gem() as gem:
            match await authorized_ctx.gem_market_data.quote_for(gem.name):
                case GemMarketQuote() as quote:
                    appraisal = appraise_gem(
                        gem,
                        quote,
                        appraised_at=datetime.now(UTC),
                    )
                case None:
                    raise HTTPException(
                        status_code=503,
                        detail='Market quote unavailable',
                    )
            match await authorized_ctx.user_gems_repository.save_appraisal(
                gem_id,
                appraisal,
            ):
                case Gem() as appraised_gem:
                    return JSONResponse(_gem_response(appraised_gem))
                case None:
                    raise _not_found()
        case None:
            raise _not_found()


async def list_gems_internal(request: Request[State]) -> JSONResponse:
    match request.query_params.get('user_id'):
        case str() as user_id if user_id:
            return await _list_authorized_gem_ids(
                request.state['ctx'].authorize_user(UserId(user_id))
            )
        case _:
            raise HTTPException(
                status_code=400,
                detail='Missing user_id query parameter',
            )
