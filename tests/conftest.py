import asyncio
import importlib
import os
from collections.abc import Callable

import pytest

from vscode_llm_harness import Backend


def _load_backend(specification: str) -> Backend:
    module_name, separator, factory_name = specification.partition(":")
    if not separator or not module_name or not factory_name:
        raise pytest.UsageError(
            "Backend must be specified as 'module.path:factory_name'."
        )

    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise pytest.UsageError(
            f"Could not import backend module {module_name!r}: {error}"
        ) from error

    factory: Callable[[], Backend] | None = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise pytest.UsageError(
            f"Backend factory {factory_name!r} is missing or not callable."
        )

    backend = factory()
    required_methods = ("list_models", "stream_chat", "count_tokens")
    missing = [name for name in required_methods if not callable(getattr(backend, name, None))]
    if missing:
        raise pytest.UsageError(
            f"Backend is missing required methods: {', '.join(missing)}"
        )
    return backend


@pytest.fixture(scope="session")
def backend() -> Backend:
    specification = os.environ.get(
        "VSCODE_LLM_BACKEND", "examples.echo_backend:create_backend"
    )
    return _load_backend(specification)


@pytest.fixture(scope="session")
def harness_loop():
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


@pytest.fixture(scope="session")
def run_async(harness_loop):
    return harness_loop.run_until_complete


@pytest.fixture(scope="session")
def models(backend: Backend, run_async):
    return run_async(
        backend.list_models(silent=True, cancellation=asyncio.Event())
    )
