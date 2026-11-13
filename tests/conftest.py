import importlib
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from vscode_llm_harness import Backend
from vscode_llm_harness.contracts import (
    ChatOptions,
    JsonContent,
    ModelInfo,
    RequestMessage,
    ResponseData,
    ResponseToolResult,
    TextContent,
)
from vscode_llm_harness.bridge import BackendBridge, FileBridge


_ERROR_SENTINEL = "__vscode_llm_harness_injected_backend_error__"
_DATA_SENTINEL = "__vscode_llm_harness_injected_data_response__"
_TOOL_RESULT_SENTINEL = "__vscode_llm_harness_injected_tool_result_response__"


class _FaultInjectingBackend:
    def __init__(self, backend: Backend):
        self._backend = backend
        self.empty_model_list = False

    async def list_models(self, *, silent, cancellation):
        if self.empty_model_list:
            return ()
        return await self._backend.list_models(
            silent=silent, cancellation=cancellation
        )

    def stream_chat(
        self,
        model: ModelInfo,
        messages: tuple[RequestMessage, ...],
        options: ChatOptions,
        cancellation,
    ):
        text = [
            part.value
            for message in messages
            for part in message.content
            if isinstance(part, TextContent)
        ]
        if _ERROR_SENTINEL in text:
            async def raise_failure() -> None:
                raise RuntimeError(
                    "Injected backend failure for conformance testing."
                )

            async def fail():
                await raise_failure()
                yield None

            return fail()
        if _DATA_SENTINEL in text:
            async def emit_data():
                yield ResponseData("application/octet-stream", b"harness-data")

            return emit_data()
        if _TOOL_RESULT_SENTINEL in text:
            async def emit_tool_result():
                yield ResponseToolResult(
                    "harness-result",
                    (
                        TextContent("harness tool result"),
                        JsonContent({"source": "conformance"}),
                    ),
                )

            return emit_tool_result()
        return self._backend.stream_chat(model, messages, options, cancellation)

    async def count_tokens(self, model, text, cancellation):
        return await self._backend.count_tokens(model, text, cancellation)


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
        test_backend = _FaultInjectingBackend(_load_backend(specification))
        bridge = FileBridge(
            BackendBridge(test_backend), Path(directory)
        )
        bridge.start()
        try:
            yield {
                "directory": directory,
                "bridge": bridge,
                "test_backend": test_backend,
            }
        finally:
            bridge.close()
