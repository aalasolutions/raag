"""Tenant context using contextvars."""

from contextvars import ContextVar
from contextlib import asynccontextmanager

_tenant_var: ContextVar[str] = ContextVar("tenant_id", default=None)


def get_current_tenant() -> str | None:
    """Get the current tenant ID from context.

    Returns None if no tenant context is set.
    """
    return _tenant_var.get()


def require_tenant() -> str:
    """Get the current tenant ID, raising RuntimeError if not set.

    Use this in operations that require tenant context.
    """
    tenant = _tenant_var.get()
    if tenant is None:
        raise RuntimeError(
            "No tenant context set. Use async with tenant_context('tenant_id'):"
        )
    return tenant


@asynccontextmanager
async def tenant_context(tenant_id: str):
    """Async context manager for setting tenant context.

    Example:
        async with tenant_context("client_123"):
            # All operations here use "client_123" as tenant_id
            result = await raag.search(...)
    """
    token = _tenant_var.set(tenant_id)
    try:
        yield
    finally:
        _tenant_var.reset(token)
