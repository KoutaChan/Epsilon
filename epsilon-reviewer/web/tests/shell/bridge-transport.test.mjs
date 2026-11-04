import test from "node:test";
import assert from "node:assert/strict";
import { ApiError } from "../../src/api/errors.ts";
import { bridgeTransport } from "../../src/api/bridgeTransport.ts";

const id = "82b7a497-06a6-42a1-8700-8b4b65e83273";
const settings = { resultsDirectory: "C:\\Users\\me\\結果", language: "ja" };

function fakeBridge(overrides = {}) {
  const calls = [];
  const bridge = {
    calls,
    async invoke(command, request, body) {
      calls.push({ command, request, body });
      return { ok: true, response: { command } };
    },
    pathForFile: (file) => file.path ?? "",
    settings: async () => ({ ok: true, value: settings }),
    setLanguage: async (language) => ({
      ok: true,
      value: { ...settings, language },
    }),
    chooseResultsDirectory: async () => ({ ok: true, value: settings }),
    exportResult: async () => ({ ok: true, value: { saved: true } }),
    ...overrides,
  };
  return bridge;
}

async function gzip(text) {
  const stream = new Blob([text])
    .stream()
    .pipeThrough(new CompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

test("commands go to invoke with the whole request and come back as the response", async () => {
  const bridge = fakeBridge();
  const transport = bridgeTransport(bridge);
  assert.equal(transport.platform, "desktop");
  const request = { resultId: id, round: 0, eventIndex: 3, seat: 1, edits: {} };
  assert.deepEqual(await transport.call("scenario", request), {
    command: "scenario",
  });
  assert.deepEqual(bridge.calls, [
    { command: "scenario", request, body: undefined },
  ]);
});

test("record bodies cross IPC as ArrayBuffer", async () => {
  const bridge = fakeBridge();
  const transport = bridgeTransport(bridge);
  const bytes = new TextEncoder().encode("<mjloggm/>");
  await transport.call("importBytes", {}, { body: new Blob([bytes]) });
  await transport.call("importBytes", {}, { body: bytes.buffer });
  const [fromBlob, fromBuffer] = bridge.calls.map((call) => call.body);
  assert.ok(fromBlob instanceof ArrayBuffer);
  assert.deepEqual(new Uint8Array(fromBlob), bytes);
  assert.equal(fromBuffer, bytes.buffer);
});

test("error replies become ApiError with the code and message", async () => {
  const transport = bridgeTransport(
    fakeBridge({
      invoke: async () => ({
        ok: false,
        error: {
          code: "analysis_active",
          message: "An analysis is already queued or running.",
        },
      }),
    }),
  );
  await assert.rejects(
    transport.call("startAnalysis", {
      recordId: id,
      modelId: "m",
      revision: "r",
    }),
    (error) =>
      error instanceof ApiError &&
      error.code === "analysis_active" &&
      error.message === "An analysis is already queued or running." &&
      error.status === 0,
  );
});

test("documents arrive as gzip bytes and are inflated in the page", async () => {
  const document = {
    format: 6,
    resultId: id,
    record: { names: ["東", "南", "西", "北"] },
  };
  const body = await gzip(JSON.stringify(document));
  const transport = bridgeTransport(
    fakeBridge({ invoke: async () => ({ ok: true, body }) }),
  );
  assert.deepEqual(await transport.call("result", { resultId: id }), document);
});

test("a document reply without a readable body is invalid_response", async () => {
  for (const body of [undefined, new TextEncoder().encode("not gzip").buffer]) {
    const transport = bridgeTransport(
      fakeBridge({ invoke: async () => ({ ok: true, body }) }),
    );
    await assert.rejects(
      transport.call("result", { resultId: id }),
      (error) => error instanceof ApiError && error.code === "invalid_response",
    );
  }
});

test("commands the page may not call on the desktop are refused before IPC", async () => {
  const bridge = fakeBridge();
  const transport = bridgeTransport(bridge);
  for (const command of [
    "downloadResult",
    "exportResult",
    "shareLink",
    "sharedResult",
    "sharedScenario",
  ])
    await assert.rejects(transport.call(command, { resultId: id }), TypeError);
  assert.equal(bridge.calls.length, 0);
});

test("an abort during the IPC call rejects with the abort reason", async () => {
  const controller = new AbortController();
  const reason = new DOMException("The review was closed.", "AbortError");
  const transport = bridgeTransport(
    fakeBridge({
      invoke: async () => {
        controller.abort(reason);
        return { ok: true, response: {} };
      },
    }),
  );
  await assert.rejects(
    transport.call("history", {}, { signal: controller.signal }),
    (error) => error === reason,
  );
});

test("bridge operations unwrap their replies", async () => {
  const transport = bridgeTransport(fakeBridge());
  assert.deepEqual(await transport.settings(), settings);
  assert.equal((await transport.setLanguage("en")).language, "en");
  assert.deepEqual(await transport.chooseResultsDirectory(), settings);
  assert.equal(await transport.exportResult(id), true);
  assert.equal(
    transport.pathForFile({ path: "C:\\record.xml" }),
    "C:\\record.xml",
  );

  const failing = bridgeTransport(
    fakeBridge({
      chooseResultsDirectory: async () => ({
        ok: false,
        error: { code: "analysis_active", message: "busy" },
      }),
      exportResult: async () => ({ ok: true, value: { saved: false } }),
    }),
  );
  await assert.rejects(
    failing.chooseResultsDirectory(),
    (error) => error instanceof ApiError && error.code === "analysis_active",
  );
  assert.equal(await failing.exportResult(id), false);
});
