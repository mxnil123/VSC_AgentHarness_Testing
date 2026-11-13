const vscode = require("vscode");
const { randomUUID } = require("crypto");
const fs = require("fs");
const path = require("path");

const vendor = "pytest-harness";

function bridgeDirectory() {
  const value = process.env.LLM_HARNESS_BRIDGE_DIR;
  if (!value) {
    throw new Error("LLM_HARNESS_BRIDGE_DIR is not set.");
  }
  return value;
}

function bridgePath(name) {
  return path.join(bridgeDirectory(), name);
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function submitOperation(operation, values = {}, signal) {
  const requestId = randomUUID();
  const requestPath = bridgePath(`request-${requestId}.json`);
  const temporaryPath = `${requestPath}.tmp`;
  const responsePath = bridgePath(`response-${requestId}.json`);
  await fs.promises.writeFile(
    temporaryPath,
    JSON.stringify({ request_id: requestId, operation, ...values }),
    "utf8"
  );
  await fs.promises.rename(temporaryPath, requestPath);

  for (;;) {
    if (signal?.aborted) {
      throw new Error("Request cancelled.");
    }
    try {
      const result = JSON.parse(await fs.promises.readFile(responsePath, "utf8"));
      await fs.promises.unlink(responsePath);
      if (!result.ok) {
        throw new Error(result.error || `Backend ${operation} request failed.`);
      }
      return result.result;
    } catch (error) {
      if (error.code !== "ENOENT") {
        throw error;
      }
    }
    await delay(20);
  }
}

async function* readChatStream(chatId, signal) {
  const streamPath = bridgePath(`stream-${chatId}.ndjson`);
  const statusPath = bridgePath(`status-${chatId}.json`);
  let offset = 0;
  let buffer = "";

  for (;;) {
    if (signal.aborted) {
      throw new Error("Request cancelled.");
    }

    try {
      const contents = await fs.promises.readFile(streamPath);
      if (contents.length > offset) {
        buffer += contents.subarray(offset).toString("utf8");
        offset = contents.length;
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line) {
            yield JSON.parse(line);
          }
        }
      }
    } catch (error) {
      if (error.code !== "ENOENT") {
        throw error;
      }
    }

    try {
      const status = JSON.parse(await fs.promises.readFile(statusPath, "utf8"));
      if (buffer) {
        yield JSON.parse(buffer);
      }
      if (!status.ok) {
        throw new Error(status.error || "Backend chat request failed.");
      }
      await fs.promises.unlink(statusPath);
      await fs.promises.unlink(streamPath).catch((error) => {
        if (error.code !== "ENOENT") {
          throw error;
        }
      });
      return;
    } catch (error) {
      if (error.code !== "ENOENT") {
        throw error;
      }
    }

    await delay(20);
  }
}

async function removeCancelledChatFiles(chatId) {
  const statusPath = bridgePath(`status-${chatId}.json`);
  const streamPath = bridgePath(`stream-${chatId}.ndjson`);
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    try {
      await fs.promises.access(statusPath);
      await fs.promises.unlink(statusPath);
      await fs.promises.unlink(streamPath).catch((error) => {
        if (error.code !== "ENOENT") {
          throw error;
        }
      });
      return;
    } catch (error) {
      if (error.code !== "ENOENT") {
        throw error;
      }
    }
    await delay(20);
  }
  throw new Error("The backend did not stop after cancellation.");
}

function serializeContentPart(part) {
  if (part instanceof vscode.LanguageModelTextPart) {
    return { type: "text", value: part.value };
  }
  if (part instanceof vscode.LanguageModelDataPart) {
    return {
      type: "data",
      media_type: part.mimeType,
      data: Buffer.from(part.data).toString("base64"),
    };
  }
  if (part instanceof vscode.LanguageModelPromptTsxPart) {
    return { type: "json", value: part.value };
  }
  if (part instanceof vscode.LanguageModelToolCallPart) {
    return {
      type: "tool_call",
      call_id: part.callId,
      name: part.name,
      arguments: part.input,
    };
  }
  if (part instanceof vscode.LanguageModelToolResultPart) {
    return {
      type: "tool_result",
      call_id: part.callId,
      content: part.content.map(serializeContentPart),
    };
  }
  throw new Error(`Unsupported VS Code input content part: ${part?.constructor?.name}`);
}

function serializeRole(role) {
  if (role === vscode.LanguageModelChatMessageRole.User) {
    return "user";
  }
  if (role === vscode.LanguageModelChatMessageRole.Assistant) {
    return "assistant";
  }
  throw new Error(`Unsupported VS Code chat-message role: ${role}`);
}

function serializeMessages(messages) {
  return messages.map((message) => ({
    role: serializeRole(message.role),
    name: message.name,
    content: message.content.map(serializeContentPart),
  }));
}

function serializeTools(tools) {
  return (tools || []).map((tool) => ({
    name: tool.name,
    description: tool.description,
    parameters: tool.inputSchema,
  }));
}

function cancellationSignal(token, controller, onCancel) {
  if (token.isCancellationRequested) {
    controller.abort();
    void onCancel().catch((error) =>
      console.error("Could not forward cancellation to the backend.", error)
    );
    return { dispose() {} };
  }
  return token.onCancellationRequested(() => {
    controller.abort();
    void onCancel().catch((error) =>
      console.error("Could not forward cancellation to the backend.", error)
    );
  });
}

class HarnessProvider {
  constructor() {
    this.models = [];
    this.lastReportedPartTypes = [];
  }

  async provideLanguageModelChatInformation(options, token) {
    const controller = new AbortController();
    const listener = cancellationSignal(token, controller, async () => {});
    try {
      const result = await submitOperation(
        "models",
        { silent: options.silent },
        controller.signal
      );
      this.models = result.models;
      return this.models;
    } finally {
      listener.dispose();
    }
  }

  async provideLanguageModelChatResponse(model, messages, options, progress, token) {
    const chatId = randomUUID();
    const controller = new AbortController();
    let cancelPromise;
    const cancelBackend = () => {
      cancelPromise = submitOperation("cancel", { chat_id: chatId });
      return cancelPromise;
    };
    const listener = cancellationSignal(token, controller, cancelBackend);
    try {
      const values = {
        chat_id: chatId,
        model_id: model.id,
        messages: serializeMessages(messages),
        options: {
          tools: serializeTools(options.tools),
          tool_choice: options.toolMode,
          max_output_tokens: options.modelOptions?.maxOutputTokens,
          temperature: options.modelOptions?.temperature,
          model_options: options.modelOptions || {},
        },
      };
      this.lastReportedPartTypes = [];
      const requestId = randomUUID();
      const requestPath = bridgePath(`request-${requestId}.json`);
      const temporaryPath = `${requestPath}.tmp`;
      await fs.promises.writeFile(
        temporaryPath,
        JSON.stringify({ request_id: requestId, operation: "chat", ...values }),
        "utf8"
      );
      await fs.promises.rename(temporaryPath, requestPath);

      for await (const part of readChatStream(chatId, controller.signal)) {
        this.lastReportedPartTypes.push(reportPart(part, progress));
      }
    } catch (error) {
      if (token.isCancellationRequested) {
        await cancelPromise;
        await removeCancelledChatFiles(chatId);
      } else {
        throw error;
      }
    } finally {
      listener.dispose();
    }
  }

  async provideTokenCount(model, text, token) {
    const controller = new AbortController();
    const listener = cancellationSignal(token, controller, async () => {});
    try {
      const result = await submitOperation(
        "count_tokens",
        {
          model_id: model.id,
          text:
            typeof text === "string"
              ? text
              : {
                  role: serializeRole(text.role),
                  name: text.name,
                  content: text.content.map(serializeContentPart),
                },
        },
        controller.signal
      );
      return result.count;
    } finally {
      listener.dispose();
    }
  }
}

function reportPart(part, progress) {
  if (part.type === "text") {
    progress.report(new vscode.LanguageModelTextPart(part.value));
    return part.type;
  } else if (part.type === "tool_call") {
    progress.report(
      new vscode.LanguageModelToolCallPart(
        part.call_id,
        part.name,
        part.arguments
      )
    );
    return part.type;
  } else if (part.type === "tool_result") {
    const content =
      typeof part.content === "string"
        ? [new vscode.LanguageModelTextPart(part.content)]
        : part.content.map((item) => {
            if (item.type === "text") {
              return new vscode.LanguageModelTextPart(item.value);
            }
            if (item.type === "image" || item.type === "data") {
              return new vscode.LanguageModelDataPart(
                Buffer.from(item.data, "base64"),
                item.media_type
              );
            }
            if (item.type === "json") {
              return new vscode.LanguageModelPromptTsxPart(item.value);
            }
            throw new Error(
              `Unsupported backend tool-result content part: ${item.type}`
            );
          });
    progress.report(
      new vscode.LanguageModelToolResultPart(part.call_id, content)
    );
    return part.type;
  } else if (part.type === "data") {
    progress.report(
      new vscode.LanguageModelDataPart(
        Buffer.from(part.data, "base64"),
        part.media_type
      )
    );
    return part.type;
  } else {
    throw new Error(`Unsupported backend response part: ${part.type}`);
  }
}

function activate(context) {
  const provider = new HarnessProvider();
  context.subscriptions.push(
    vscode.lm.registerLanguageModelChatProvider(vendor, provider),
    vscode.commands.registerCommand(
      "_pytestHarness.getModelInformation",
      () => provider.models
    ),
    vscode.commands.registerCommand(
      "_pytestHarness.getLastReportedPartTypes",
      () => provider.lastReportedPartTypes
    )
  );
}

module.exports = { activate };
