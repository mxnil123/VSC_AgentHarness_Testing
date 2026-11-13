import importlib
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from vscode_llm_harness import Backend
from vscode_llm_harness.bridge import BackendBridge, FileBridge


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
    missing = [
        name for name in required_methods if not callable(getattr(backend, name, None))
    ]
    if missing:
        raise pytest.UsageError(
            f"Backend is missing required methods: {', '.join(missing)}"
        )
    return backend


@pytest.fixture(scope="session")
def vscode_bridge():
    specification = os.environ.get(
        "VSCODE_LLM_BACKEND", "examples.echo_backend:create_backend"
    )
    with TemporaryDirectory(prefix="vscode-llm-bridge-") as directory:
        bridge = FileBridge(
            BackendBridge(_load_backend(specification)), Path(directory)
        )
        bridge.start()
        try:
            yield {"directory": directory, "bridge": bridge}
        finally:
            bridge.close()
