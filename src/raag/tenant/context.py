"""Tenant context using contextvars."""

import functools
import inspect
import threading
from contextvars import ContextVar

_tenant_var: ContextVar[str] = ContextVar("tenant_id", default=None)
# Tokens of the tenant_context blocks active in this context, innermost last
_entries_var: ContextVar[tuple] = ContextVar("tenant_context_entries", default=())

_EXIT_ERROR = "tenant_context exited out of order or in a different context than it was entered"


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
            "No tenant context set. Use with tenant_context('tenant_id'): "
            "or async with tenant_context('tenant_id'):"
        )
    return tenant


class _TenantContext:
    """Context manager that sets the tenant for both sync and async code.

    An object can be entered again after it exits, but not while it is
    active: blocks that may overlap each need their own tenant_context().
    """

    def __init__(self, tenant_id: str):
        self._tenant_id = tenant_id
        self._active = threading.Lock()
        self._token = None

    def __enter__(self):
        if not self._active.acquire(blocking=False):
            raise RuntimeError(
                "This tenant_context is already active. Call tenant_context() again "
                "for each block that may overlap, such as nested blocks, tasks or threads."
            )
        token = _tenant_var.set(self._tenant_id)
        _entries_var.set(_entries_var.get() + (token,))
        self._token = token
        return self

    def __exit__(self, exc_type, exc, tb):
        token = self._token
        entries = _entries_var.get()
        if token is None or not entries or entries[-1] is not token:
            raise RuntimeError(_EXIT_ERROR)
        try:
            _tenant_var.reset(token)
        except ValueError:
            # A copy of the entering context, such as a child task's
            raise RuntimeError(_EXIT_ERROR) from None
        _entries_var.set(entries[:-1])
        self._token = None
        self._active.release()
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        return self.__exit__(exc_type, exc, tb)

    def __call__(self, func):
        """Decorate a sync or async function. Each call gets a fresh context."""
        tenant_id = self._tenant_id

        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
            raise TypeError(
                "tenant_context cannot decorate a generator: its cleanup can run in "
                "another context when it is garbage collected. Enter tenant_context "
                "in the code that iterates the generator instead."
            )

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with _TenantContext(tenant_id):
                    return await func(*args, **kwargs)
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _TenantContext(tenant_id):
                return func(*args, **kwargs)
        return wrapper


def tenant_context(tenant_id: str) -> _TenantContext:
    """Context manager for setting tenant context.

    Works with ``with``, ``async with``, and as a decorator on sync or
    async functions. Generators cannot be decorated: enter the context in
    the code that iterates them.

    Example:
        async with tenant_context("client_123"):
            result = await raag.search(...)

        with tenant_context("client_123"):
            result = raag.search_sync(...)

        @tenant_context("client_123")
        async def handler():
            return await raag.search(...)
    """
    return _TenantContext(tenant_id)
