# VS Code language model provider runtime harness

This harness tests a Python language-model backend through a real VS Code
Extension Development Host. It does not mock `vscode.lm`: the test extension
registers a `LanguageModelChatProvider`, then the VS Code host discovers the
provider, selects its model, counts tokens, streams a chat response, and sends a
cancellation token. A local Python bridge forwards those VS Code requests to
the configured backend.

The checks are driven by pytest, but pass only when the actual VS Code runtime
starts and the in-host integration checks succeed. Pytest reports model
discovery, token counting, chat streaming, model options, multimodal/tool use,
response-part conversion, cancellation, and backend error propagation as
separate cases; each case starts an isolated Extension Development Host.
See the [VS Code provider API guide](https://code.visualstudio.com/api/extension-guides/ai/language-model-chat-provider)
for the upstream extension contract.

## Requirements

- Python 3.10+
- pytest 8+
- VS Code 1.104 or later, installed locally

The harness uses the installed VS Code CLI, so it does not download a separate
editor build. By default it checks `code` on `PATH`, then the usual Windows
install locations. Set `VSCODE_EXECUTABLE` if VS Code is installed elsewhere.
Each run uses isolated VS Code user-data and extension directories; it does not
modify the user's normal VS Code profile.

## Run the runtime tests

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest
```

The included echo backend is offline and deterministic. It streams several
chunks with a delay so the runtime cancellation test can verify that the
backend was still active when VS Code cancelled the request.

## Test your backend

Implement a factory that returns an object matching the `Backend` protocol in
`vscode_llm_harness/contracts.py`:

- `async list_models(*, silent, cancellation)` returns `ModelInfo` instances.
- `stream_chat(model, messages, options, cancellation)` returns an async
  iterator of `ResponseText`, `ResponseToolCall`, `ResponseToolResult`, and/or
  `ResponseData`.
- `async count_tokens(model, text, cancellation)` accepts a string or
  `RequestMessage`.

Request messages can contain text, image/data, tool-call, and tool-result
parts. `ChatOptions` carries tool definitions, tool mode, common generation
settings, and provider-specific `model_options`. Model metadata includes
image-input and tool-calling capabilities.

The cancellation argument is an `asyncio.Event`. The runtime adapter forwards
VS Code's cancellation token to this event and stops consuming the stream when
it is set.

Make the backend importable from the project directory, then run:

```powershell
$env:VSCODE_LLM_BACKEND = "my_backend:create_backend"
python -m pytest
```

Optionally point to a different VS Code installation or increase the startup
timeout:

```powershell
$env:VSCODE_EXECUTABLE = "C:\Program Files\Microsoft VS Code\Code.exe"
$env:VSCODE_LLM_RUNTIME_TIMEOUT = "240"
python -m pytest
```

`VSCODE_EXECUTABLE` may point to the VS Code CLI (`code.cmd` on Windows), or
to `Code.exe` in its standard installation layout.

The backend factory remains inside pytest. VS Code and pytest exchange bridge
requests through a private temporary directory, avoiding network ports and
keeping model credentials and backend traffic on the local machine.

## What is exercised

The extension-host checks cover:

- Provider activation, multiple model discovery, unique IDs, filtered
  selection by ID/family/version, no-match and empty-list selection, and model
  metadata through VS Code
- Token counting for both strings and `LanguageModelChatMessage` instances
- Chat request and streamed text response for every discovered model
- Forwarding of custom `modelOptions`, tool definitions, and required tool mode
- Image input and tool-call/tool-result message round trips when the backend
  advertises those capabilities (otherwise pytest marks the case skipped)
- Data and tool-result response-part conversion through the real VS Code host
- Cancellation reaching the backend while a response is still active
- Backend error propagation to VS Code and successful requests after an error

This verifies the backend through VS Code's provider runtime, not Copilot's own
models or service. It does not prove that VS Code UI policy, Copilot account
configuration, vendor publishing, or every third-party service integration is
available to end users. VS Code's selected-model API does not expose all
provider metadata, so those fields are checked against the provider metadata
returned by the extension itself. The response-part and error-path cases use
exact test-only sentinel messages intercepted by a transparent pytest backend
wrapper, and the empty-discovery case configures that wrapper to return no
models. Those cases validate the adapter/runtime path rather than model
quality. All other requests are delegated to the configured backend.
