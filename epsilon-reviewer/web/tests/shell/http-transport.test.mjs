import test from "node:test";
import assert from "node:assert/strict";
import { COMMANDS } from "../../../protocol/commands.ts";
import { ApiError } from "../../src/api/errors.ts";
import { httpTransport } from "../../src/api/httpTransport.ts";

const id = "82b7a497-06a6-42a1-8700-8b4b65e83273";
const query = { round: 1, eventIndex: 40, seat: 2, edits: { honba: 3 } };
const record = new Blob(["<mjloggm/>"]);

// 操作ごとの HTTP の道。json は本文の JSON、binary は本文が牌譜そのもの、share は Cookie を送らない。
const ROUTES = {
  models: { request: {}, method: "GET", path: "/api/models" },
  history: { request: {}, method: "GET", path: "/api/history" },
  importBytes: {
    request: {},
    method: "POST",
    path: "/api/records",
    binary: true,
  },
  importTenhou: {
    request: { url: "https://tenhou.net/0/?log=2026110412gm-0089-0000-abcd" },
    method: "POST",
    path: "/api/records/tenhou",
    json: { url: "https://tenhou.net/0/?log=2026110412gm-0089-0000-abcd" },
  },
  importStored: {
    request: { resultId: id },
    method: "POST",
    path: `/api/results/${id}/record`,
  },
  startAnalysis: {
    request: { recordId: id, modelId: "prism", revision: "v551" },
    method: "POST",
    path: "/api/analyses",
    json: { recordId: id, modelId: "prism", revision: "v551" },
  },
  job: { request: { jobId: id }, method: "GET", path: `/api/jobs/${id}` },
  cancelJob: {
    request: { jobId: id },
    method: "POST",
    path: `/api/jobs/${id}/cancel`,
  },
  result: {
    request: { resultId: id },
    method: "GET",
    path: `/api/results/${id}`,
  },
  deleteResult: {
    request: { resultId: id },
    method: "DELETE",
    path: `/api/results/${id}`,
  },
  downloadResult: {
    request: { resultId: id },
    method: "GET",
    path: `/api/results/${id}/download`,
  },
  shareLink: {
    request: { resultId: id },
    method: "GET",
    path: `/api/results/${id}/share`,
  },
  sharedResult: {
    request: { resultId: id },
    method: "GET",
    path: `/api/shares/${id}`,
    share: true,
  },
  scenario: {
    request: { resultId: id, ...query },
    method: "POST",
    path: `/api/results/${id}/scenario`,
    json: query,
  },
  sharedScenario: {
    request: { resultId: id, ...query },
    method: "POST",
    path: `/api/shares/${id}/scenario`,
    json: query,
    share: true,
  },
};

function reply(kind) {
  if (kind === "empty") return new Response(null, { status: 204 });
  if (kind === "attachment")
    return new Response(new Uint8Array([31, 139, 8]), {
      headers: { "Content-Type": "application/gzip" },
    });
  return Response.json({ ok: true });
}

test("every HTTP route of the command table is covered", () => {
  assert.deepEqual(
    Object.keys(ROUTES).sort(),
    Object.keys(COMMANDS)
      .filter((name) => COMMANDS[name].http !== null)
      .sort(),
  );
});

for (const [command, route] of Object.entries(ROUTES)) {
  test(`${command} sends ${route.method} ${route.path}`, async (t) => {
    const calls = [];
    t.mock.method(globalThis, "fetch", async (path, init) => {
      calls.push({ path, init });
      return reply(COMMANDS[command].response);
    });
    await httpTransport().call(
      command,
      route.request,
      route.binary ? { body: record } : {},
    );
    assert.equal(calls.length, 1);
    const [{ path, init }] = calls;
    assert.equal(path, route.path);
    assert.equal(init.method, route.method);
    assert.equal(init.credentials, route.share ? "omit" : "same-origin");
    if (route.json) {
      assert.deepEqual(init.headers, { "Content-Type": "application/json" });
      assert.deepEqual(JSON.parse(init.body), route.json);
    } else if (route.binary) {
      assert.deepEqual(init.headers, {
        "Content-Type": "application/octet-stream",
      });
      assert.equal(init.body, record);
    } else {
      assert.equal(init.headers, undefined);
      assert.equal(init.body, undefined);
    }
  });
}

test("desktop-only commands have no HTTP route", async () => {
  for (const command of ["importFile", "exportResult"])
    await assert.rejects(
      httpTransport().call(command, { path: "C:\\record.xml", resultId: id }),
      TypeError,
    );
});

test("responses decode by the command's response kind", async (t) => {
  const transport = httpTransport();
  const document = { format: 6, resultId: id, rounds: [] };
  const fetch = t.mock.method(globalThis, "fetch", async () =>
    Response.json(document),
  );
  assert.deepEqual(await transport.call("result", { resultId: id }), document);

  const bytes = new Uint8Array([31, 139, 8, 0, 1, 2, 3]);
  fetch.mock.mockImplementation(async () => new Response(bytes));
  const blob = await transport.call("downloadResult", { resultId: id });
  assert.ok(blob instanceof Blob);
  assert.deepEqual(new Uint8Array(await blob.arrayBuffer()), bytes);

  fetch.mock.mockImplementation(
    async () => new Response(null, { status: 204 }),
  );
  assert.equal(
    await transport.call("deleteResult", { resultId: id }),
    undefined,
  );
});

test("error envelopes become ApiError with the code, message and status", async (t) => {
  t.mock.method(globalThis, "fetch", async () =>
    Response.json(
      {
        error: {
          code: "share_not_found",
          message: "The shared analysis was not found.",
        },
      },
      { status: 404 },
    ),
  );
  await assert.rejects(
    httpTransport().call("sharedResult", { resultId: id }),
    (error) =>
      error instanceof ApiError &&
      error.code === "share_not_found" &&
      error.status === 404 &&
      error.message === "The shared analysis was not found.",
  );
});

test("odd error bodies become invalid_response with the status", async (t) => {
  const responses = [
    () => new Response(null, { status: 413 }),
    () => new Response("<html>Bad gateway</html>", { status: 502 }),
    () => Response.json({ error: "busy" }, { status: 429 }),
    () =>
      Response.json(
        { error: { code: "no_such_code", message: "?" } },
        { status: 400 },
      ),
    () => Response.json(null, { status: 500 }),
  ];
  const fetch = t.mock.method(globalThis, "fetch");
  for (const create of responses) {
    const response = create();
    fetch.mock.mockImplementation(async () => response);
    await assert.rejects(
      httpTransport().call("history", {}),
      (error) =>
        error instanceof ApiError &&
        error.code === "invalid_response" &&
        error.status === response.status,
    );
  }
});

test("an unreadable success body is invalid_response", async (t) => {
  t.mock.method(
    globalThis,
    "fetch",
    async () => new Response("{", { status: 200 }),
  );
  await assert.rejects(
    httpTransport().call("models", {}),
    (error) =>
      error instanceof ApiError &&
      error.code === "invalid_response" &&
      error.status === 200,
  );
});

test("a network failure is connection_failed", async (t) => {
  t.mock.method(globalThis, "fetch", async () => {
    throw new TypeError("fetch failed");
  });
  await assert.rejects(
    httpTransport().call("history", {}),
    (error) =>
      error instanceof ApiError &&
      error.code === "connection_failed" &&
      error.status === 0,
  );
});

test("aborting rejects with the abort reason, not a connection error", async (t) => {
  const controller = new AbortController();
  const reason = new DOMException("The review was closed.", "AbortError");
  t.mock.method(
    globalThis,
    "fetch",
    (_path, { signal }) =>
      new Promise((_resolve, reject) => {
        assert.equal(signal, controller.signal);
        signal.addEventListener("abort", () => reject(signal.reason), {
          once: true,
        });
      }),
  );
  const pending = httpTransport().call(
    "sharedResult",
    { resultId: id },
    { signal: controller.signal },
  );
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
});
