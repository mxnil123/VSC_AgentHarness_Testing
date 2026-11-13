import asyncio
import base64
from collections.abc import Mapping, Sequence

import pytest

from vscode_llm_harness import (
    ChatOptions,
    ImageContent,
    ModelInfo,
    RequestMessage,
    ResponseText,
    ResponseToolCall,
    ResponseToolResult,
    TextContent,
    ToolDefinition,
)


def _validate_model(model: ModelInfo) -> None:
    assert isinstance(model.id, str) and model.id.strip()
    assert isinstance(model.name, str) and model.name.strip()
    assert isinstance(model.family, str) and model.family.strip()
    assert isinstance(model.version, str) and model.version.strip()
    assert isinstance(model.max_input_tokens, int) and model.max_input_tokens > 0
    assert isinstance(model.max_output_tokens, int) and model.max_output_tokens > 0
    assert isinstance(model.capabilities.image_input, bool)
    assert isinstance(model.capabilities.tool_calling, (bool, int))
    if isinstance(model.capabilities.tool_calling, int) and not isinstance(
        model.capabilities.tool_calling, bool
    ):
        assert model.capabilities.tool_calling >= 0


def test_model_discovery_returns_valid_unique_models(backend, models):
    assert isinstance(models, Sequence)
    model_ids = []
    for model in models:
        _validate_model(model)
        model_ids.append(model.id)
    assert len(model_ids) == len(set(model_ids)), "Model IDs must be unique per provider."


def test_silent_model_discovery_does_not_require_interaction(backend, run_async):
    discovered = run_async(
        backend.list_models(silent=True, cancellation=asyncio.Event())
    )
    assert isinstance(discovered, Sequence)
    for model in discovered:
        _validate_model(model)


@pytest.mark.parametrize("as_message", [False, True])
def test_token_count_returns_non_negative_integer(
    backend, models, as_message, run_async
):
    if not models:
        pytest.skip("No models are available for token-count tests.")
    value = (
        RequestMessage(role="user", content=(TextContent("Count these tokens."),))
        if as_message
        else "Count these tokens."
    )
    count = run_async(
        backend.count_tokens(models[0], value, asyncio.Event())
    )
    assert isinstance(count, int) and not isinstance(count, bool)
    assert count >= 0


def test_chat_stream_emits_well_formed_response_parts(backend, models, run_async):
    if not models:
        pytest.skip("No models are available for chat tests.")

    async def collect():
        cancellation = asyncio.Event()
        return [
            part
            async for part in backend.stream_chat(
                models[0],
                (RequestMessage(role="user", content=(TextContent("Reply with a short greeting."),)),),
                ChatOptions(),
                cancellation,
            )
        ]

    parts = run_async(asyncio.wait_for(collect(), timeout=30))
    assert parts, "A successful chat request must produce at least one response part."
    has_content = False
    for part in parts:
        if isinstance(part, ResponseText):
            assert isinstance(part.value, str)
            has_content = has_content or bool(part.value)
        elif isinstance(part, ResponseToolCall):
            assert part.call_id and part.name
            assert isinstance(part.arguments, Mapping)
            has_content = True
        elif isinstance(part, ResponseToolResult):
            assert part.call_id
            assert isinstance(part.content, str)
            has_content = has_content or bool(part.content)
        else:
            pytest.fail(f"Unsupported response part: {type(part).__name__}")
    assert has_content, "The response stream must contain non-empty content."


def test_chat_stream_stops_when_cancelled(backend, models, run_async):
    if not models:
        pytest.skip("No models are available for cancellation tests.")

    async def consume_until_cancelled():
        cancellation = asyncio.Event()
        stream = backend.stream_chat(
            models[0],
            (RequestMessage(role="user", content=(TextContent("Continue generating."),)),),
            ChatOptions(),
            cancellation,
        )

        async def consume():
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        if task.done():
            pytest.skip("The response completed before cancellation could be tested.")
        cancellation.set()
        await asyncio.wait_for(task, timeout=2)

    run_async(consume_until_cancelled())


def test_declared_image_support_accepts_image_input(backend, models, run_async):
    model = next((item for item in models if item.capabilities.image_input), None)
    if model is None:
        pytest.skip("No discovered model declares image input support.")

    async def collect():
        image = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lE8AAAAASUVORK5CYII="
        )
        cancellation = asyncio.Event()
        return [
            part
            async for part in backend.stream_chat(
                model,
                (
                    RequestMessage(
                        role="user",
                        content=(
                            TextContent("Describe this image."),
                            ImageContent(media_type="image/png", data=image),
                        ),
                    ),
                ),
                ChatOptions(),
                cancellation,
            )
        ]

    assert run_async(asyncio.wait_for(collect(), timeout=30))


def test_declared_tool_support_accepts_tool_definitions(backend, models, run_async):
    model = next((item for item in models if item.capabilities.tool_calling), None)
    if model is None:
        pytest.skip("No discovered model declares tool calling support.")

    async def collect():
        cancellation = asyncio.Event()
        return [
            part
            async for part in backend.stream_chat(
                model,
                (RequestMessage(role="user", content=(TextContent("Use the test tool."),)),),
                ChatOptions(
                    tools=(
                        ToolDefinition(
                            name="test_tool",
                            description="A harness tool for provider compatibility.",
                            parameters={
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            },
                        ),
                    ),
                    tool_choice="auto",
                ),
                cancellation,
            )
        ]

    assert run_async(asyncio.wait_for(collect(), timeout=30))
