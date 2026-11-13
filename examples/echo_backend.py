import asyncio
import json
from collections.abc import AsyncIterator, Sequence

from vscode_llm_harness import (
    Backend,
    ChatOptions,
    ImageContent,
    ModelCapabilities,
    ModelInfo,
    JsonContent,
    RequestMessage,
    ResponsePart,
    ResponseText,
    ResponseToolCall,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)


_MODELS = (
    ModelInfo(
        id="echo",
        name="Echo",
        family="example",
        version="1",
        max_input_tokens=4096,
        max_output_tokens=1024,
    ),
    ModelInfo(
        id="echo-multimodal",
        name="Echo Multimodal",
        family="example",
        version="1",
        max_input_tokens=8192,
        max_output_tokens=2048,
        capabilities=ModelCapabilities(image_input=True, tool_calling=4),
        tooltip="Deterministic model for image and tool conformance checks.",
        detail="Local test model",
    ),
)


def _content_text(part: object) -> str:
    if isinstance(part, TextContent):
        return part.value
    if isinstance(part, ImageContent):
        return f"[{part.media_type} image]"
    if isinstance(part, JsonContent):
        return json.dumps(part.value, ensure_ascii=False)
    if isinstance(part, ToolCallContent):
        return json.dumps(
            {"tool": part.name, "arguments": dict(part.arguments)},
            ensure_ascii=False,
        )
    if isinstance(part, ToolResultContent):
        if isinstance(part.content, str):
            return part.content
        return "".join(_content_text(item) for item in part.content)
    return ""


class EchoBackend:
    async def list_models(
        self, *, silent: bool, cancellation: asyncio.Event
    ) -> Sequence[ModelInfo]:
        return _MODELS

    async def count_tokens(
        self, model: ModelInfo, text: str | RequestMessage, cancellation: asyncio.Event
    ) -> int:
        value = (
            text
            if isinstance(text, str)
            else "".join(_content_text(part) for part in text.content)
        )
        return (len(value) + 3) // 4

    async def _stream(
        self,
        messages: Sequence[RequestMessage],
        options: ChatOptions,
        cancellation: asyncio.Event,
    ) -> AsyncIterator[ResponsePart]:
        prompt = "".join(
            _content_text(part)
            for message in messages
            for part in message.content
        )
        if options.tools and options.tool_choice == 2:
            tool = options.tools[0]
            yield ResponseToolCall(
                "echo-call",
                tool.name,
                {"value": "echo"},
            )
            return

        if not prompt.strip():
            prompt = "Echo response"
        for offset in range(0, len(prompt), 5):
            if cancellation.is_set():
                return
            try:
                await asyncio.wait_for(cancellation.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                yield ResponseText(prompt[offset : offset + 5])

    def stream_chat(
        self,
        model: ModelInfo,
        messages: Sequence[RequestMessage],
        options: ChatOptions,
        cancellation: asyncio.Event,
    ) -> AsyncIterator[ResponsePart]:
        return self._stream(messages, options, cancellation)


def create_backend() -> Backend:
    return EchoBackend()
