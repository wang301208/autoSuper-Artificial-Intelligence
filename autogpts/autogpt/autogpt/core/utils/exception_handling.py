from __future__ import annotations

import asyncio
import functools
import logging
import sys
import threading
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)


F = TypeVar("F", bound=Callable[..., Any])
Coro = TypeVar("Coro", bound=Awaitable[Any])


def setup_global_exception_hook() -> None:
    """Register global hooks to log uncaught exceptions."""

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger().error(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception

    def thread_exception_hook(args: threading.ExceptHookArgs) -> None:
        logging.getLogger().error(
            "Unhandled exception in thread %s",
            args.thread.name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = thread_exception_hook

    try:
        loop = asyncio.get_event_loop()

        def asyncio_exception_handler(loop, context):
            exc = context.get("exception")
            if exc:
                logging.getLogger().error(
                    "Unhandled asyncio exception",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            else:
                logging.getLogger().error(
                    f"Unhandled asyncio exception: {context.get('message')}"
                )

        loop.set_exception_handler(asyncio_exception_handler)
    except RuntimeError:
        # No event loop in this thread
        pass


def log_exceptions_in_async(coro: Coro) -> asyncio.Task:
    """Wrap a coroutine in a Task that logs exceptions."""

    task = asyncio.create_task(coro)

    def _done_callback(t: asyncio.Task) -> None:
        try:
            t.result()
        except Exception:
            logger.exception("Unhandled exception in async task")

    task.add_done_callback(_done_callback)
    return task


def run_in_thread_with_logging(target: F, *args, daemon: bool = False, **kwargs) -> threading.Thread:
    """Run a function in a thread and log any exception."""

    def _runner() -> None:
        try:
            target(*args, **kwargs)
        except Exception:
            logger.exception("Unhandled exception in background thread")

    thread = threading.Thread(target=_runner, daemon=daemon, name=getattr(target, "__name__", "thread"))
    thread.start()
    return thread
