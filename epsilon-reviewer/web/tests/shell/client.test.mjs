import test from "node:test";
import assert from "node:assert/strict";
import { RECORD_SIZE_LIMIT } from "../../../protocol/commands.ts";
import { createApi } from "../../src/api/client.ts";
import { ApiError } from "../../src/api/errors.ts";

const id = "82b7a497-06a6-42a1-8700-8b4b65e83273";

/** テスト用のトランスポート。送ったコマンドを記録し、渡した応答の関数（`respond`）の戻り値をそのまま返す。 */
function fakeTransport(platform, respond = () => undefined, desktop = {}) {
  const calls = [];
  return {
    calls,
    platform,
    async call(command, request, options = {}) {
      calls.push({ command, request, options });
      return respond(command, request);
    },
    ...(platform === "desktop"
      ? {
          pathForFile: () => "",
          settings: async () => ({}),
          setLanguage: async () => ({}),
          chooseResultsDirectory: async () => ({}),
          exportResult: async () => true,
          ...desktop,
        }
      : {}),
  };
}

const documentOf = (resultId) => ({ format: 6, resultId });

test("result keeps the two most recently opened documents", async () => {
  const transport = fakeTransport("web", (_command, { resultId }) =>
    documentOf(resultId),
  );
  const api = createApi(transport);
  const opened = [];
  for (const resultId of ["a", "b", "a", "c", "a", "b", "c"])
    opened.push(await api.result(resultId));
  assert.deepEqual(
    transport.calls.map((call) => call.request.resultId),
    ["a", "b", "c", "b", "c"],
  );
  assert.equal(opened[0], opened[2]);
});

test("deleting a result drops its cached document", async () => {
  const transport = fakeTransport("web", (command, { resultId }) =>
    command === "result" ? documentOf(resultId) : undefined,
  );
  const api = createApi(transport);
  await api.result(id);
  await api.remove(id);
  await api.result(id);
  assert.deepEqual(
    transport.calls.map((call) => call.command),
    ["result", "deleteResult", "result"],
  );
});

test("web export downloads the stored gzip untouched under the result file name", async (t) => {
  const blob = new Blob([new Uint8Array([31, 139, 8, 0])], {
    type: "application/gzip",
  });
  const transport = fakeTransport("web", () => blob);
  const link = { click: t.mock.fn() };
  globalThis.document = {
    createElement: (tag) => (assert.equal(tag, "a"), link),
  };
  t.after(() => delete globalThis.document);
  const created = t.mock.method(
    URL,
    "createObjectURL",
    () => "blob:review-export",
  );
  const revoked = t.mock.method(URL, "revokeObjectURL", () => {});
  const delays = [];
  t.mock.method(globalThis, "setTimeout", (callback, delay) => {
    delays.push(delay);
    callback();
  });

  assert.equal(await createApi(transport).export(id), "downloaded");
  assert.deepEqual(
    transport.calls.map((call) => call.command),
    ["downloadResult"],
  );
  assert.equal(created.mock.calls[0].arguments[0], blob);
  assert.equal(link.href, "blob:review-export");
  assert.equal(link.download, `${id}.epsilon-reviewer.json.gz`);
  assert.equal(link.click.mock.callCount(), 1);
  assert.deepEqual(delays, [1000]);
  assert.deepEqual(revoked.mock.calls[0].arguments, ["blob:review-export"]);
});

test("desktop export uses the save dialog of the bridge", async () => {
  for (const [saved, outcome] of [
    [true, "saved"],
    [false, "cancelled"],
  ]) {
    const transport = fakeTransport("desktop", undefined, {
      exportResult: async (resultId) => (assert.equal(resultId, id), saved),
    });
    assert.equal(await createApi(transport).export(id), outcome);
    assert.equal(transport.calls.length, 0);
  }
});

test("web imports send the File itself as the record body", async () => {
  const transport = fakeTransport("web");
  const file = new File(["<mjloggm/>"], "record.xml");
  await createApi(transport).importFile(file);
  assert.deepEqual(transport.calls, [
    { command: "importBytes", request: {}, options: { body: file } },
  ]);
});

test("desktop imports send the file path, or the bytes when the file has no path", async () => {
  const withPath = new File(["<mjloggm/>"], "record.xml");
  const pasted = new File(["<mjloggm/>"], "pasted.xml");
  const transport = fakeTransport("desktop", undefined, {
    pathForFile: (file) => (file === withPath ? "C:\\牌譜\\record.xml" : ""),
  });
  const api = createApi(transport);
  await api.importFile(withPath);
  await api.importFile(pasted);
  assert.deepEqual(transport.calls, [
    {
      command: "importFile",
      request: { path: "C:\\牌譜\\record.xml" },
      options: {},
    },
    { command: "importBytes", request: {}, options: { body: pasted } },
  ]);
});

test("records over the size limit are refused without sending", async () => {
  for (const platform of ["web", "desktop"]) {
    const transport = fakeTransport(platform);
    const api = createApi(transport);
    await assert.rejects(
      api.importFile({ size: RECORD_SIZE_LIMIT + 1 }),
      (error) => error instanceof ApiError && error.code === "record_too_large",
    );
    assert.equal(transport.calls.length, 0);
    await api.importFile({ size: RECORD_SIZE_LIMIT });
    assert.equal(transport.calls.length, 1);
  }
});

test("malformed share identifiers never become requests", async () => {
  const transport = fakeTransport("web", () => documentOf(id));
  const api = createApi(transport);
  for (const value of [
    "",
    "../history",
    `${id}/source`,
    "not-a-result",
    `${id}?export=1`,
    id.toUpperCase(),
  ])
    await assert.rejects(
      api.sharedResult(value),
      (error) =>
        error instanceof ApiError &&
        error.code === "share_not_found" &&
        error.status === 404,
    );
  assert.equal(transport.calls.length, 0);
  assert.deepEqual(await api.sharedResult(id), documentOf(id));
  assert.deepEqual(
    transport.calls.map((call) => call.command),
    ["sharedResult"],
  );
});

test("start sends only the model key of a catalog model", async () => {
  const transport = fakeTransport("web");
  await createApi(transport).start(id, {
    modelId: "prism",
    revision: "v551",
    displayName: "Epsilon Prism",
    version: "v2",
    available: true,
  });
  assert.deepEqual(transport.calls[0].request, {
    recordId: id,
    modelId: "prism",
    revision: "v551",
  });
});

test("scenario clients send the query to the owner or the share command", async () => {
  const transport = fakeTransport("web", () => ({ candidates: [] }));
  const api = createApi(transport);
  const query = { round: 2, eventIndex: 17, seat: 3, edits: {} };
  const signal = new AbortController().signal;
  await api.scenarioClient(id, "owner").evaluate(query, signal);
  await api.scenarioClient(id, "share").evaluate(query, signal);
  assert.deepEqual(transport.calls, [
    {
      command: "scenario",
      request: { resultId: id, ...query },
      options: { signal },
    },
    {
      command: "sharedScenario",
      request: { resultId: id, ...query },
      options: { signal },
    },
  ]);
});

test("only the desktop exposes its settings", () => {
  assert.equal(createApi(fakeTransport("web")).desktop, null);
  const desktop = fakeTransport("desktop");
  assert.equal(createApi(desktop).desktop, desktop);
  assert.equal(createApi(desktop).platform, "desktop");
});
