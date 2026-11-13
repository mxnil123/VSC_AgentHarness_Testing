import asyncio
import base64
import json
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .contracts import (
    Backend,
    ChatOptions,
    ImageContent,
    ModelInfo,
    RequestMessage,
    ResponseText,
    ResponseToolCall,
    ResponseToolResult,
    TextContent,
    ToolCallContent,
    ToolDefinition,
    ToolResultContent,
)


_END = object()


def _read_content(part: Mapping[str, Any]):
    kind = part.get("type")
    if kind == "text":
        return TextContent(str(part.get("value", "")))
    if kind == "image":
        return ImageContent(
            media_type=str(part["media_type"]),
            data=base64.b64decode(part["data"], validate=True),
        )
    if kind == "tool_call":
        arguments = part.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise ValueError("Tool call arguments must be a JSON object.")
        return ToolCallContent(
            call_id=str(part["call_id"]),
            name=str(part["name"]),
            arguments=arguments,
        )
    if kind == "tool_result":
        return ToolResultContent(
            call_id=str(part["call_id"]),
            content=str(part.get("content", "")),
        )
    raise ValueError(f"Unsupported request content type: {kind!r}")


def _read_message(message: Mapping[str, Any]) -> RequestMessage:
    role = message.get("role")
    if role not in ("system", "user", "assistant"):
        raise ValueError(f"Unsupported message role: {role!r}")
    parts = message.get("content")
    if not isinstance(parts, list):
        raise ValueError("Message content must be an array.")
    return RequestMessage(
        role=role,
        content=tuple(_read_content(part) for part in parts),
        name=message.get("name"),
    )


def _model_to_json(model: ModelInfo) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "family": model.family,
        "version": model.version,
        "maxInputTokens": model.max_input_tokens,
        "maxOutputTokens": model.max_output_tokens,
        "tooltip": model.tooltip,
        "detail": model.detail,
        "capabilities": {
            "imageInput": model.capabilities.image_input,
            "toolCalling": model.capabilities.tool_calling,
        },
    }


def _response_to_json(part: object) -> dict[str, Any]:
    if isinstance(part, ResponseText):
        return {"type": "text", "value": part.value}
    if isinstance(part, ResponseToolCall):
        return {
            "type": "tool_call",
            "call_id": part.call_id,
            "name": part.name,
            "arguments": dict(part.arguments),
        }
    if isinstance(part, ResponseToolResult):
        return {
            "type": "tool_result",
            "call_id": part.call_id,
            "content": part.content,
        }
    raise TypeError(f"Unsupported backend response part: {type(part).__name__}")


async def _next_or_cancel(iterator, cancellation: asyncio.Event):
    next_part = asyncio.create_task(iterator.__anext__())
    cancelled = asyncio.create_task(cancellation.wait())
    done, _ = await asyncio.wait(
        (next_part, cancelled), return_when=asyncio.FIRST_COMPLETED
    )
    if cancelled in done:
        next_part.cancel()
        await asyncio.gather(next_part, return_exceptions=True)
        return None
    cancelled.cancel()
    await asyncio.gather(cancelled, return_exceptions=True)
    try:
        return await next_part
    except StopAsyncIteration:
        return _END


class BackendBridge:
    def __init__(self, backend: Backend):
        self.backend = backend
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="vscode-llm-backend-loop", daemon=True
        )
        self._lock = threading.Lock()
        self._models: dict[str, ModelInfo] = {}
        self._active: dict[str, asyncio.Event] = {}
        self._cancel_requests = 0
        self._active_cancel_requests = 0
        self._model_discovery_requests = 0
        self._loop_thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, awaitable, timeout: float = 30):
        future = asyncio.run_coroutine_threadsafe(awaitable, self._loop)
        return future.result(timeout=timeout)

    async def _new_event(self) -> asyncio.Event:
        return asyncio.Event()

    async def _close_iterator(self, iterator) -> None:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            await close()

    def list_models(self, silent: bool) -> list[dict[str, Any]]:
        with self._lock:
            self._model_discovery_requests += 1
        cancellation = self._run(self._new_event())
        models = self._run(
            self.backend.list_models(silent=silent, cancellation=cancellation)
        )
        with self._lock:
            self._models = {model.id: model for model in models}
        return [_model_to_json(model) for model in models]

    def count_tokens(self, payload: Mapping[str, Any]) -> int:
        with self._lock:
            model = self._models.get(str(payload.get("model_id")))
        if model is None:
            raise ValueError(f"Unknown model ID: {payload.get('model_id')!r}")
        value = payload.get("text")
        text = _read_message(value) if isinstance(value, Mapping) else str(value)
        cancellation = self._run(self._new_event())
        return self._run(self.backend.count_tokens(model, text, cancellation))

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            self._cancel_requests += 1
            cancellation = self._active.get(request_id)
            if cancellation is not None:
                self._active_cancel_requests += 1
        if cancellation is None:
            return False
        self._loop.call_soon_threadsafe(cancellation.set)
        return True

    def cancel_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "cancel_requests": self._cancel_requests,
                "active_cancel_requests": self._active_cancel_requests,
                "model_discovery_requests": self._model_discovery_requests,
            }

    def model_info(self) -> tuple[ModelInfo, ...]:
        with self._lock:
            return tuple(self._models.values())

    def stream_chat(
        self,
        request_id: str,
        payload: Mapping[str, Any],
        report: Callable[[dict[str, Any]], None],
    ) -> None:
        model_id = str(payload.get("model_id", ""))
        with self._lock:
            model = self._models.get(model_id)
        if model is None:
            raise ValueError(f"Unknown model ID: {model_id!r}")
        cancellation = self._run(self._new_event())
        with self._lock:
            self._active[request_id] = cancellation

        messages = tuple(_read_message(item) for item in payload.get("messages", []))
        raw_options = payload.get("options", {})
        raw_tools = raw_options.get("tools", [])
        tools = tuple(
            ToolDefinition(
                name=str(item["name"]),
                description=str(item.get("description", "")),
                parameters=item.get("parameters", {}),
            )
            for item in raw_tools
        )
        options = ChatOptions(
            max_output_tokens=raw_options.get("max_output_tokens"),
            temperature=raw_options.get("temperature"),
            tools=tools,
            tool_choice=raw_options.get("tool_choice"),
        )

        iterator = self.backend.stream_chat(
            model, messages, options, cancellation
        ).__aiter__()
        try:
            while True:
                part = self._run(_next_or_cancel(iterator, cancellation), timeout=120)
                if part is _END or part is None:
                    break
                report(_response_to_json(part))
        finally:
            self._run(self._close_iterator(iterator))
            with self._lock:
                self._active.pop(request_id, None)

    def close(self) -> None:
        with self._lock:
            active = list(self._active.values())
        for cancellation in active:
            self._loop.call_soon_threadsafe(cancellation.set)
        if self._loop.is_running():

            async def cancel_pending():
                current = asyncio.current_task()
                pending = [
                    task
                    for task in asyncio.all_tasks()
                    if task is not current and not task.done()
                ]
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

            asyncio.run_coroutine_threadsafe(
                cancel_pending(), self._loop
            ).result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
        if not self._loop.is_closed():
            self._loop.close()


class FileBridge:
    def __init__(self, backend: BackendBridge, directory: Path):
        self.backend = backend
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._workers: set[threading.Thread] = set()
        self._thread = threading.Thread(
            target=self._watch, name="vscode-llm-file-bridge", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _watch(self) -> None:
        while not self._stopping.is_set():
            for request_path in self.directory.glob("request-*.json"):
                claimed_path = request_path.with_name(
                    request_path.name.replace("request-", "processing-", 1)
                )
                try:
                    request_path.replace(claimed_path)
                except FileNotFoundError:
                    continue
                worker = threading.Thread(
                    target=self._dispatch,
                    args=(claimed_path,),
                    name=f"vscode-llm-request-{claimed_path.stem}",
                    daemon=True,
                )
                with self._lock:
                    self._workers.add(worker)
                worker.start()
            self._stopping.wait(0.01)

    def _write_json(self, path: Path, value: object) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, path)

    def _append_json_line(self, request_id: str, value: object) -> None:
        path = self.directory / f"stream-{request_id}.ndjson"
        encoded = json.dumps(value, ensure_ascii=False).encode("utf-8") + b"\n"
        with path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()

    def _dispatch(self, claimed_path: Path) -> None:
        request_id = claimed_path.stem.removeprefix("processing-")
        response_path = self.directory / f"response-{request_id}.json"
        request = {}
        try:
            request = json.loads(claimed_path.read_text(encoding="utf-8"))
            operation = request.get("operation")
            if operation == "models":
                result = {"models": self.backend.list_models(bool(request["silent"]))}
                self._write_json(response_path, {"ok": True, "result": result})
            elif operation == "count_tokens":
                count = self.backend.count_tokens(request)
                self._write_json(
                    response_path, {"ok": True, "result": {"count": count}}
                )
            elif operation == "chat":
                chat_id = str(request["chat_id"])
                self.backend.stream_chat(
                    chat_id,
                    request,
                    lambda part: self._append_json_line(chat_id, part),
                )
                self._write_json(
                    self.directory / f"status-{chat_id}.json", {"ok": True}
                )
            elif operation == "cancel":
                cancelled = self.backend.cancel(str(request["chat_id"]))
                self._write_json(
                    response_path, {"ok": True, "result": {"cancelled": cancelled}}
                )
            else:
                raise ValueError(f"Unsupported bridge operation: {operation!r}")
        except Exception as error:
            print(f"VS Code file bridge request failed: {error}", flush=True)
            if request.get("operation") == "chat":
                chat_id = str(request.get("chat_id", request_id))
                self._write_json(
                    self.directory / f"status-{chat_id}.json",
                    {"ok": False, "error": str(error)},
                )
            else:
                self._write_json(response_path, {"ok": False, "error": str(error)})
        finally:
            claimed_path.unlink(missing_ok=True)
            with self._lock:
                self._workers.discard(threading.current_thread())

    def cancel_stats(self) -> dict[str, int]:
        return self.backend.cancel_stats()

    def model_info(self) -> tuple[ModelInfo, ...]:
        return self.backend.model_info()

    def close(self) -> None:
        self._stopping.set()
        self._thread.join(timeout=5)
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout=5)
        self.backend.close()
