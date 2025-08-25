from __future__ import annotations

import asyncio
from typing import Any, Awaitable


def run_async(coro: Awaitable[Any]) -> Any:
    """Run ``coro`` respecting any running event loop.

    If no loop is running in the current thread, ``asyncio.run`` is used. When a
    loop is already running, the coroutine is scheduled on that loop and its
    result is awaited in a thread-safe manner.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
