import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias


@dataclass(frozen=True)
class ModelCapabilities:
    image_input: bool = False
    tool_calling: bool | int = False


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    family: str
    version: str
    max_input_tokens: int
    max_output_tokens: int
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    tooltip: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class TextContent:
    value: str


@dataclass(frozen=True)
class ImageContent:
    media_type: str
    data: bytes


@dataclass(frozen=True)
class ToolCallContent:
    call_id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ToolResultContent:
    call_id: str
    content: str


ContentPart: TypeAlias = (
    TextContent | ImageContent | ToolCallContent | ToolResultContent
)
MessageRole: TypeAlias = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class RequestMessage:
    role: MessageRole
    content: Sequence[ContentPart]
    name: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class ChatOptions:
    max_output_tokens: int | None = None
    temperature: float | None = None
    tools: Sequence[ToolDefinition] = ()
    tool_choice: str | None = None


@dataclass(frozen=True)
class ResponseText:
    value: str


@dataclass(frozen=True)
class ResponseToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ResponseToolResult:
    call_id: str
    content: str


ResponsePart: TypeAlias = ResponseText | ResponseToolCall | ResponseToolResult


class Backend(Protocol):
    async def list_models(
        self, *, silent: bool, cancellation: asyncio.Event
    ) -> Sequence[ModelInfo]:
        """Return models usable by the VS Code model picker."""

    def stream_chat(
        self,
        model: ModelInfo,
        messages: Sequence[RequestMessage],
        options: ChatOptions,
        cancellation: asyncio.Event,
    ) -> AsyncIterator[ResponsePart]:
        """Yield response parts and stop promptly when cancellation is set."""

    async def count_tokens(
        self, model: ModelInfo, text: str | RequestMessage, cancellation: asyncio.Event
    ) -> int:
        """Return an estimated or exact token count for text or one message."""
