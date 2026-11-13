# VS Code language model backend harness

This project provides an offline-by-default pytest conformance suite for Python
backends that are adapted to VS Code's Language Model Chat Provider API. It
checks model discovery, model metadata, token counting, streamed response
parts, cancellation, and declared image/tool capabilities.

The harness tests the Python backend boundary, not the VS Code extension
runtime. A VS Code extension still needs to register a
`LanguageModelChatProvider` and translate between the VS Code API types and the
contracts in `vscode_llm_harness`. See the
[official provider guide](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider)
for registration and extension-side details.

## Requirements

- Python 3.10+
- pytest 8+

## Run the included example

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest
```

The example backend is deterministic and requires no API key or network access.

## Test your backend

Implement a factory that returns an object matching the `Backend` protocol in
`vscode_llm_harness/contracts.py`. The backend exposes:

- `async list_models(*, silent: bool, cancellation: asyncio.Event)` returning
  `ModelInfo` objects. Silent discovery must not prompt for credentials or
  configuration.
- `stream_chat(model, messages, options, cancellation)`, an async iterator of
  response parts. It should stop promptly after the supplied
  `asyncio.Event` is set.
- `async count_tokens(model, text, cancellation)` accepting either a string or
  a `RequestMessage`.

The contract types preserve the main VS Code provider concepts: model metadata
and capabilities, text/image/tool request content, streamed text/tool response
parts, token counting, and cancellation. Your extension adapter is responsible
for mapping VS Code's API types and cancellation token to these Python types.

Make your backend importable in the active Python environment, then point the
suite at its zero-argument factory:

```powershell
$env:VSCODE_LLM_BACKEND = "my_backend:create_backend"
pytest
```

Use an import path that Python can resolve from the project directory (for
example, install your backend package in the active environment or add its
source root to `PYTHONPATH`). If no environment variable is set, pytest tests
the included echo backend.

Models declaring image input or tool calling are exercised with corresponding
requests. Models that do not declare a capability are skipped for that
capability-specific check. Network-backed implementations should use a
dedicated test account or a local test endpoint; the general chat conformance
tests issue real requests to every discovered model.

## Test report

The suite skips model-dependent checks when silent discovery returns no models.
Such a run is not evidence that chat, token counting, or cancellation works;
configure the backend to expose at least one test model for those checks.
