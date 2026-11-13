import asyncio
from collections.abc import AsyncIterator, Sequence

from vscode_llm_harness import (
    Backend,
    ChatOptions,
    ModelCapabilities,
    ModelInfo,
    RequestMessage,
    ResponsePart,
    ResponseText,
    TextContent,
)


class EchoBackend:
    async def list_models(
        self, *, silent: bool, cancellation: asyncio.Event
    ) -> Sequence[ModelInfo]:
        return [
            ModelInfo(
                id="echo",
                name="Echo",
                family="example",
                version="1",
                max_input_tokens=4096,
                max_output_tokens=1024,
                capabilities=ModelCapabilities(),
            )
        ]

    async def count_tokens(
        self, model: ModelInfo, text: str | RequestMessage, cancellation: asyncio.Event
    ) -> int:
        value = text if isinstance(text, str) else "".join(
            part.value for part in text.content if isinstance(part, TextContent)
        )
        return (len(value) + 3) // 4

    async def _stream(
        self,
        messages: Sequence[RequestMessage],
        cancellation: asyncio.Event,
    ) -> AsyncIterator[ResponsePart]:
        prompt = "".join(
            part.value
            for message in messages
            for part in message.content
            if isinstance(part, TextContent)
        )
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
        return self._stream(messages, cancellation)


def create_backend() -> Backend:
    return EchoBackend()
