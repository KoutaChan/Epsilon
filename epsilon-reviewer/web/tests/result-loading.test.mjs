import test from "node:test";
import assert from "node:assert/strict";
import {
  eventSnapshot,
  decisionSnapshot,
  reviewResult,
} from "./reviewFixtures.mjs";

globalThis.window = {};
const { api, ApiError } = await import("../src/api/client.ts");

function serveResult(t, result) {
  window.epsilonReviewer = {
    request: async () => ({ status: 200, body: result }),
  };
  t.after(() => {
    delete window.epsilonReviewer;
  });
}
function isInvalidResult(error) {
  return error instanceof ApiError && error.code === "invalid_result";
}

for (const formatVersion of [undefined, 1, 2, 3, 5]) {
  test(`result loading rejects unsupported format ${formatVersion}`, async (t) => {
    serveResult(
      t,
      reviewResult([eventSnapshot(0, "start_kyoku")], { formatVersion }),
    );
    await assert.rejects(api.result("result-1"), isInvalidResult);
  });
}

test("current results preserve snapshot, probability and scoring data without copies", async (t) => {
  const start = eventSnapshot(0, "start_kyoku");
  const draw = eventSnapshot(1, "tsumo", 0);
  const choice = decisionSnapshot(draw, 0, "TSUMO_AGARI", { eventIndex: 2 });
  const win = eventSnapshot(2, "hora", 0);
  const result = reviewResult([start, draw, choice, win]);
  serveResult(t, result);
  const loaded = await api.result("result-1");
  assert.equal(loaded, result);
  assert.equal(loaded.rounds[0].steps[2].candidates, choice.candidates);
  assert.equal(loaded.rounds[0].steps[3].outcome.details, null);
});

test("HTTP result loading rejects older server output with the localized error code", async (t) => {
  t.mock.method(
    globalThis,
    "fetch",
    async () =>
      new Response(
        JSON.stringify(
          reviewResult([eventSnapshot(0, "start_kyoku")], { formatVersion: 2 }),
        ),
        { status: 200 },
      ),
  );
  await assert.rejects(api.result("result-1"), isInvalidResult);
});

for (const fields of [
  { stateId: undefined },
  { stateId: null },
  { causeEventIndex: undefined },
  { causeEventIndex: 0 },
]) {
  test(`event snapshots require current state links: ${JSON.stringify(fields)}`, async (t) => {
    serveResult(t, reviewResult([eventSnapshot(0, "start_kyoku", -1, fields)]));
    await assert.rejects(api.result("result-1"), isInvalidResult);
  });
}

test("decision snapshots require a numeric cause", async (t) => {
  const draw = eventSnapshot(1, "tsumo", 0);
  serveResult(
    t,
    reviewResult([
      eventSnapshot(0, "start_kyoku"),
      draw,
      decisionSnapshot(draw, 0, "DAHAI", { causeEventIndex: null }),
    ]),
  );
  await assert.rejects(api.result("result-1"), isInvalidResult);
});

for (const fields of [
  { outcome: null },
  { target: null },
  { winningTile: null },
  { details: undefined },
]) {
  test(`win snapshots require recorded outcome fields: ${JSON.stringify(fields)}`, async (t) => {
    const win = eventSnapshot(1, "hora", 0);
    const outcome =
      "outcome" in fields ? fields.outcome : { ...win.outcome, ...fields };
    serveResult(
      t,
      reviewResult([eventSnapshot(0, "start_kyoku"), { ...win, outcome }]),
    );
    await assert.rejects(api.result("result-1"), isInvalidResult);
  });
}

test("desktop export cancellation remains supported", async (t) => {
  serveResult(t, { cancelled: true });
  assert.equal(await api.export("result-1"), false);
});

test("web export downloads the original gzip bytes without parsing or rewriting JSON", async (t) => {
  const { gzipSync } = await import("node:zlib");
  const bytes = gzipSync(
    JSON.stringify(reviewResult([eventSnapshot(0, "start_kyoku")])),
  );
  let downloaded,
    filename,
    clicked = false;
  t.mock.method(globalThis, "fetch", async (path, options) => {
    assert.equal(path, "/api/results/result-1/export");
    assert.equal(options.credentials, "same-origin");
    return new Response(bytes, {
      headers: { "Content-Type": "application/gzip" },
    });
  });
  t.mock.method(URL, "createObjectURL", (blob) => {
    downloaded = blob;
    return "blob:review-export";
  });
  t.mock.method(globalThis, "setTimeout", () => 0);
  globalThis.document = {
    createElement: () => ({
      set href(value) {
        assert.equal(value, "blob:review-export");
      },
      set download(value) {
        filename = value;
      },
      click() {
        clicked = true;
      },
    }),
  };
  t.after(() => {
    delete globalThis.document;
  });
  assert.equal(await api.export("result-1"), true);
  assert.deepEqual(Buffer.from(await downloaded.arrayBuffer()), bytes);
  assert.equal(filename, "result-1.epsilon-reviewer.json.gz");
  assert.equal(clicked, true);
});
