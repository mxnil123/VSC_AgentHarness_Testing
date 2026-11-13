const assert = require("assert");
const fs = require("fs");
const vscode = require("vscode");

const vendor = "pytest-harness";
const results = { tests: [] };

async function check(name, callback) {
  try {
    await callback();
    results.tests.push({ name, passed: true });
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

async function selectModel() {
  const models = await vscode.lm.selectChatModels({ vendor });
  assert.ok(models.length > 0, "The provider did not contribute a model.");
  return models[0];
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
  let model;
  await check("discovery", async () => {
    model = await selectModel();
    assert.ok(model.id);
    assert.ok(model.name);
    assert.ok(model.family);
    assert.ok(model.version);
    assert.ok(model.maxInputTokens > 0);
  });

  if (model) {
    switch (process.env.LLM_HARNESS_TEST_CASE) {
      case "discovery":
        break;
      case "token_count":
        await check("token_count", async () => {
          const count = await model.countTokens(
            "VS Code runtime token counting",
            new vscode.CancellationTokenSource().token
          );
          assert.ok(Number.isInteger(count) && count >= 0);
        });
        break;
      case "chat":
        await check("chat", async () => {
          const response = await model.sendRequest(
            [vscode.LanguageModelChatMessage.User("Reply with a short greeting.")],
            {},
            new vscode.CancellationTokenSource().token
          );
          let text = "";
          for await (const fragment of response.text) {
            text += fragment;
          }
          assert.ok(text.trim().length > 0, "The VS Code chat response was empty.");
        });
        break;
      case "cancellation":
        await check("cancellation", async () => {
          const source = new vscode.CancellationTokenSource();
          const response = await model.sendRequest(
            [vscode.LanguageModelChatMessage.User("Generate a detailed response.")],
            {},
            source.token
          );
          const iterator = response.text[Symbol.asyncIterator]();
          const first = await withTimeout(iterator.next(), 30000, "Initial response");
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
      default:
        throw new Error(
          `Unknown LLM_HARNESS_TEST_CASE: ${process.env.LLM_HARNESS_TEST_CASE}`
        );
      }
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
