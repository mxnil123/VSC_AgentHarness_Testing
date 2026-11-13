const assert = require("assert");
const fs = require("fs");
const vscode = require("vscode");

const vendor = "pytest-harness";
const results = { tests: [] };

async function check(name, callback) {
  try {
    const outcome = await callback();
    results.tests.push({
      name,
      passed: true,
      ...(outcome && outcome.skipped
        ? { skipped: true, reason: outcome.reason }
        : {}),
    });
    console.log(`PASS ${name}`);
  } catch (error) {
    results.tests.push({
      name,
      passed: false,
      error: error instanceof Error ? error.stack || error.message : String(error),
    });
    console.error(`FAIL ${name}: ${error}`);
  }
}

async function selectModels(selector = {}) {
  return vscode.lm.selectChatModels({ vendor, ...selector });
}

async function selectModel(id) {
  const models = await selectModels({ id });
  assert.strictEqual(models.length, 1, `Expected one selected model for ${id}.`);
  return models[0];
}

async function providerModels() {
  return vscode.commands.executeCommand("_pytestHarness.getModelInformation");
}

async function consume(response) {
  const parts = [];
  for await (const part of response.stream) {
    parts.push(part);
  }
  return parts;
}

function responseText(parts) {
  return parts
    .filter((part) => part instanceof vscode.LanguageModelTextPart)
    .map((part) => part.value)
    .join("");
}

function withTimeout(promise, milliseconds, label) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error(`${label} timed out after ${milliseconds}ms`)),
        milliseconds
      )
    ),
  ]);
}

async function run() {
  let models = [];
  let modelInformation = [];

  await check("discovery", async () => {
    models = await selectModels();
    modelInformation = await providerModels();
    if (process.env.LLM_HARNESS_TEST_CASE === "empty_discovery") {
      assert.strictEqual(models.length, 0);
      assert.strictEqual(modelInformation.length, 0);
      return;
    }
    assert.ok(models.length > 0, "The provider did not contribute a model.");
    assert.strictEqual(modelInformation.length, models.length);

    const ids = modelInformation.map((item) => item.id);
    assert.strictEqual(new Set(ids).size, ids.length, "Model IDs must be unique.");
    for (const info of modelInformation) {
      assert.ok(info.id && info.name && info.family && info.version);
      assert.ok(info.maxInputTokens > 0 && info.maxOutputTokens > 0);
      assert.ok(info.capabilities);
      assert.strictEqual(typeof info.capabilities.imageInput, "boolean");
      assert.ok(
        typeof info.capabilities.toolCalling === "boolean" ||
          Number.isInteger(info.capabilities.toolCalling)
      );
      assert.strictEqual((await selectModels({ id: info.id })).length, 1);
      assert.ok(
        (await selectModels({ family: info.family })).some(
          (model) => model.id === info.id
        )
      );
      assert.ok(
        (await selectModels({ version: info.version })).some(
          (model) => model.id === info.id
        )
      );
    }

    const missing = await selectModels({ id: "__missing_harness_model__" });
    assert.strictEqual(missing.length, 0);
    return undefined;
  });

  switch (process.env.LLM_HARNESS_TEST_CASE) {
    case "discovery":
      break;
    case "empty_discovery":
      await check("empty_discovery", async () => {
        assert.strictEqual(models.length, 0);
        assert.strictEqual(modelInformation.length, 0);
      });
      break;
    case "token_count":
      await check("token_count", async () => {
        for (const model of models) {
          const textCount = await model.countTokens(
            "VS Code runtime token counting",
            new vscode.CancellationTokenSource().token
          );
          const messageCount = await model.countTokens(
            vscode.LanguageModelChatMessage.User([
              new vscode.LanguageModelTextPart("Count this message content."),
            ])
          );
          assert.ok(Number.isInteger(textCount) && textCount >= 0);
          assert.ok(Number.isInteger(messageCount) && messageCount >= 0);
        }
      });
      break;
    case "chat":
      await check("chat", async () => {
        for (const model of models) {
          const response = await model.sendRequest(
            [
              vscode.LanguageModelChatMessage.User(
                "Reply with a short greeting."
              ),
            ],
            {},
            new vscode.CancellationTokenSource().token
          );
          const text = responseText(await consume(response));
          assert.ok(
            text.trim().length > 0,
            `The VS Code chat response from ${model.id} was empty.`
          );
        }
      });
      break;
    case "model_options":
      await check("model_options", async () => {
        const model = models[0];
        const response = await model.sendRequest(
          [vscode.LanguageModelChatMessage.User("Preserve these model options.")],
          {
            modelOptions: {
              maxOutputTokens: 19,
              temperature: 0.125,
              harnessProbe: "preserved",
            },
          },
          new vscode.CancellationTokenSource().token
        );
        assert.ok(responseText(await consume(response)).length > 0);
      });
      break;
    case "image_input":
      await check("image_input", async () => {
        const info = modelInformation.find(
          (item) => item.capabilities.imageInput
        );
        if (!info) {
          return {
            skipped: true,
            reason: "The backend does not advertise image input.",
          };
        }
        const model = await selectModel(info.id);
        const png = Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l3sAAAAASUVORK5CYII=",
          "base64"
        );
        const response = await model.sendRequest(
          [
            vscode.LanguageModelChatMessage.User([
              new vscode.LanguageModelTextPart(
                "Describe the provided conformance-test image."
              ),
              vscode.LanguageModelDataPart.image(png, "image/png"),
            ]),
          ],
          {},
          new vscode.CancellationTokenSource().token
        );
        assert.ok((await consume(response)).length > 0);
      });
      break;
    case "tool_calling":
      await check("tool_calling", async () => {
        const info = modelInformation.find(
          (item) => item.capabilities.toolCalling
        );
        if (!info) {
          return {
            skipped: true,
            reason: "The backend does not advertise tool calling.",
          };
        }
        const model = await selectModel(info.id);
        const tool = {
          name: "lookup_value",
          description: "Return a value for a lookup key.",
          inputSchema: {
            type: "object",
            properties: { key: { type: "string" } },
            required: ["key"],
          },
        };
        const response = await model.sendRequest(
          [vscode.LanguageModelChatMessage.User("Use the lookup tool.")],
          {
            tools: [tool],
            toolMode: vscode.LanguageModelChatToolMode.Required,
          },
          new vscode.CancellationTokenSource().token
        );
        const parts = await consume(response);
        const call = parts.find(
          (part) => part instanceof vscode.LanguageModelToolCallPart
        );
        assert.ok(call, "Required tool mode did not return a tool call.");
        assert.strictEqual(call.name, tool.name);
        assert.ok(call.input && typeof call.input === "object");

        const followUp = await model.sendRequest(
          [
            vscode.LanguageModelChatMessage.Assistant([
              new vscode.LanguageModelToolCallPart(
                call.callId,
                call.name,
                call.input
              ),
            ]),
            vscode.LanguageModelChatMessage.User([
              new vscode.LanguageModelToolResultPart(call.callId, [
                new vscode.LanguageModelTextPart("lookup result"),
                new vscode.LanguageModelPromptTsxPart({
                  source: "conformance",
                }),
              ]),
            ]),
          ],
          {},
          new vscode.CancellationTokenSource().token
        );
        assert.ok(responseText(await consume(followUp)).length > 0);
      });
      break;
    case "response_parts":
      await check("response_parts", async () => {
        const model = models[0];
        const dataResponse = await model.sendRequest(
          [
            vscode.LanguageModelChatMessage.User(
              "__vscode_llm_harness_injected_data_response__"
            ),
          ],
          {},
          new vscode.CancellationTokenSource().token
        );
        const dataParts = await consume(dataResponse);
        const data = dataParts.find(
          (part) => part instanceof vscode.LanguageModelDataPart
        );
        assert.ok(data, "Data response part was not preserved.");
        assert.strictEqual(data.mimeType, "application/octet-stream");
        assert.ok(Buffer.from(data.data).equals(Buffer.from("harness-data")));

        const resultResponse = await model.sendRequest(
          [
            vscode.LanguageModelChatMessage.User(
              "__vscode_llm_harness_injected_tool_result_response__"
            ),
          ],
          {},
          new vscode.CancellationTokenSource().token
        );
        const resultParts = await consume(resultResponse);
        assert.ok(Array.isArray(resultParts));
        const reportedTypes = await vscode.commands.executeCommand(
          "_pytestHarness.getLastReportedPartTypes"
        );
        assert.ok(
          reportedTypes.includes("tool_result"),
          "The provider did not report its tool-result response part."
        );
      });
      break;
    case "cancellation":
      await check("cancellation", async () => {
        const model = models[0];
        const source = new vscode.CancellationTokenSource();
        const response = await model.sendRequest(
          [
            vscode.LanguageModelChatMessage.User(
              "Generate a detailed response with many words. ".repeat(20)
            ),
          ],
          {},
          source.token
        );
        const iterator = response.text[Symbol.asyncIterator]();
        const first = await withTimeout(
          iterator.next(),
          30000,
          "Initial response"
        );
        assert.ok(
          !first.done,
          "The backend completed before cancellation could be tested."
        );
        source.cancel();
        try {
          await withTimeout(iterator.next(), 3000, "Cancelled response");
        } catch (error) {
          if (!String(error).includes("timed out")) {
            throw error;
          }
        } finally {
          source.dispose();
        }
      });
      break;
    case "backend_error":
      await check("backend_error", async () => {
        const model = models[0];
        let surfaced = false;
        try {
          const response = await model.sendRequest(
            [
              vscode.LanguageModelChatMessage.User(
                "__vscode_llm_harness_injected_backend_error__"
              ),
            ],
            {},
            new vscode.CancellationTokenSource().token
          );
          await consume(response);
        } catch (error) {
          surfaced = true;
          assert.ok(
            String(error).includes("Injected backend failure"),
            `Unexpected backend error: ${error}`
          );
        }
        assert.ok(surfaced, "The backend error was swallowed by the provider.");

        const recovery = await model.sendRequest(
          [vscode.LanguageModelChatMessage.User("Recover after the failed request.")],
          {},
          new vscode.CancellationTokenSource().token
        );
        assert.ok(responseText(await consume(recovery)).length > 0);
      });
      break;
    default:
      throw new Error(
        `Unknown LLM_HARNESS_TEST_CASE: ${process.env.LLM_HARNESS_TEST_CASE}`
      );
  }

  const resultPath = process.env.LLM_HARNESS_TEST_RESULTS;
  if (!resultPath) {
    throw new Error("LLM_HARNESS_TEST_RESULTS is not set.");
  }
  console.log(`Writing pytest runtime results to ${resultPath}`);
  fs.writeFileSync(resultPath, JSON.stringify(results), "utf8");

  const failures = results.tests.filter((item) => !item.passed);
  if (failures.length > 0) {
    throw new Error(`${failures.length} VS Code runtime check(s) failed.`);
  }
}

module.exports.run = run;
