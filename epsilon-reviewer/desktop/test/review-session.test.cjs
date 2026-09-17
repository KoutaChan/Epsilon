const { test } = require("node:test");
const assert = require("node:assert/strict");
const { mkdtemp, rm } = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const { gzipSync } = require("node:zlib");
const { ResultStore } = require("../dist/result-store.js");
const { resultDocument: result } = require("./result-fixture.cjs");
const { RemoteClient } = require("../dist/remote-client.js");
const {
  ReviewSession,
  parseDesktopApiRequest,
} = require("../dist/review-session.js");

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("jobs remain on their original server and concurrent completion transfers only once", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "review-session-"));
  try {
    const settings = {
      value: {
        serverUrl: "http://first.test",
        resultsDirectory: path.join(root, "results"),
      },
    };
    const store = new ResultStore(path.join(root, "index"));
    const calls = [];
    let released;
    const release = new Promise((resolve) => {
      released = resolve;
    });
    const remote = new RemoteClient(async (url, init) => {
      calls.push([url, init.method, init.body]);
      if (url.endsWith("/api/analyses")) return json({ jobId: "job-1" }, 202);
      if (url.endsWith("/api/jobs/job-1"))
        return json({
          jobId: "job-1",
          status: "completed",
          resultId: "game-1",
        });
      if (url.endsWith("/source"))
        return new Response('{"type":"start_game"}', {
          headers: { "X-File-Name": "game.jsonl", "X-Record-Source": "mjai" },
        });
      if (init.method === "DELETE") {
        released();
        return new Response(null, { status: 204 });
      }
      return new Response(gzipSync(JSON.stringify(result("game-1"))));
    });
    const review = new ReviewSession(settings, store, remote, async () => ({
      status: 200,
      body: null,
    }));
    await review.request({
      path: "/api/analyses",
      method: "POST",
      body: { recordId: "record-1" },
    });
    settings.value.serverUrl = "http://second.test";
    await Promise.all([
      review.request({ path: "/api/jobs/job-1", method: "GET" }),
      review.request({ path: "/api/jobs/job-1", method: "GET" }),
    ]);
    await release;
    assert(calls.every(([url]) => url.startsWith("http://first.test/")));
    assert.equal(
      calls.filter(
        ([url, method]) =>
          url.endsWith("/api/results/game-1/export") && method === "GET",
      ).length,
      1,
    );
    assert.equal(JSON.parse(calls[0][2]).storage, "desktop");
    assert.deepEqual(await store.get("game-1"), result("game-1"));
    assert.equal((await store.getSource("game-1")).recordSource, "mjai");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("a failed source transfer retains the server result and can retry; local history works offline", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "review-retry-"));
  try {
    let failSource = true;
    let offline = false;
    let deletes = 0;
    const settings = {
      value: {
        serverUrl: "http://server.test",
        resultsDirectory: path.join(root, "results"),
      },
    };
    const store = new ResultStore(path.join(root, "index"));
    const remote = new RemoteClient(async (url, init) => {
      if (offline) throw new Error("offline");
      if (url.endsWith("/api/analyses")) return json({ jobId: "job-2" }, 202);
      if (url.endsWith("/api/jobs/job-2"))
        return json({
          jobId: "job-2",
          status: "completed",
          resultId: "game-2",
        });
      if (url.endsWith("/source"))
        return failSource
          ? new Response(null, { status: 503 })
          : new Response("{}", { headers: { "X-File-Name": "game.jsonl" } });
      if (init.method === "DELETE") {
        deletes++;
        return new Response(null, { status: 204 });
      }
      return new Response(gzipSync(JSON.stringify(result("game-2"))));
    });
    const review = new ReviewSession(settings, store, remote, async () => ({
      status: 200,
      body: null,
    }));
    await review.request({ path: "/api/analyses", method: "POST", body: {} });
    await assert.rejects(
      review.request({ path: "/api/jobs/job-2", method: "GET" }),
      (error) => error.code === "source_transfer_failed",
    );
    assert.equal(await store.has("game-2"), false);
    assert.equal(deletes, 0);
    failSource = false;
    await review.request({ path: "/api/jobs/job-2", method: "GET" });
    offline = true;
    const history = await review.request({
      path: "/api/history",
      method: "GET",
    });
    assert.equal(history.body.results[0].resultId, "game-2");
    assert.deepEqual(
      (await review.request({ path: "/api/results/game-2", method: "GET" }))
        .body,
      result("game-2"),
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("IPC validates upload types without copying the ArrayBuffer before transport", async () => {
  const bytes = new TextEncoder().encode("{}").buffer;
  let transported;
  const remote = new RemoteClient(async (_url, init) => {
    transported = init.body;
    return json({ recordId: "record-1" }, 201);
  });
  const review = new ReviewSession(
    { value: { serverUrl: "http://server.test" } },
    null,
    remote,
    null,
  );
  const request = parseDesktopApiRequest({
    path: "/api/records",
    method: "POST",
    fileName: "game.jsonl",
    body: { bytes },
  });
  await review.request(request);
  assert.equal(transported.buffer, bytes);
  await assert.rejects(
    review.request({ ...request, body: { bytes: "invalid" } }),
    (error) => error.code === "invalid_request",
  );
  await assert.rejects(
    review.request({ path: "/api/jobs/foreign", method: "GET" }),
    (error) => error.code === "invalid_request",
  );
  assert.throws(
    () => parseDesktopApiRequest({ path: "/api/history", method: "PATCH" }),
    (error) => error.code === "invalid_request",
  );
});

test("exports the stored gzip archive while offline", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "review-offline-export-"));
  try {
    const store = new ResultStore(path.join(root, "index"));
    const bytes = gzipSync(JSON.stringify(result("export"), null, 2), {
      level: 1,
    });
    await store.saveArchive("export", bytes, path.join(root, "results"));
    let exported;
    const session = new ReviewSession(
      { value: { serverUrl: "http://offline.test" } },
      store,
      new RemoteClient(async () => {
        throw new Error("Network must not be used for local exports.");
      }),
      async (id, archive) => {
        exported = { id, archive };
        return { status: 200, body: { saved: true } };
      },
    );
    assert.deepEqual(
      await session.request({
        path: "/api/results/export/export",
        method: "GET",
      }),
      { status: 200, body: { saved: true } },
    );
    assert.equal(exported.id, "export");
    assert.deepEqual(exported.archive, bytes);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("does not release a server result when its archive identity is wrong", async () => {
  const root = await mkdtemp(
    path.join(os.tmpdir(), "review-archive-mismatch-"),
  );
  try {
    let deletes = 0;
    const store = new ResultStore(path.join(root, "index"));
    const remote = new RemoteClient(async (url, init) => {
      if (init.method === "DELETE") {
        deletes++;
        return new Response(null, { status: 204 });
      }
      if (url.endsWith("/api/analyses")) return json({ jobId: "job" }, 202);
      if (url.endsWith("/api/jobs/job"))
        return json({
          jobId: "job",
          status: "completed",
          resultId: "expected",
        });
      if (url.endsWith("/source"))
        return new Response("{}", { headers: { "X-File-Name": "game.jsonl" } });
      return new Response(gzipSync(JSON.stringify(result("different"))));
    });
    const session = new ReviewSession(
      {
        value: {
          serverUrl: "http://server.test",
          resultsDirectory: path.join(root, "results"),
        },
      },
      store,
      remote,
      async () => ({ status: 200, body: null }),
    );
    await session.request({ path: "/api/analyses", method: "POST", body: {} });
    await assert.rejects(
      session.request({ path: "/api/jobs/job", method: "GET" }),
      (error) => error.code === "result_mismatch",
    );
    assert.equal(deletes, 0);
    assert.deepEqual(await store.list(), []);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
