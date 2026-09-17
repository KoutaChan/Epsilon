import test from "node:test";
import assert from "node:assert/strict";
import {
  eventSnapshot,
  decisionSnapshot,
  reviewResult,
} from "./reviewFixtures.mjs";

globalThis.window = {};
const { api, ApiError } = await import("../src/api/client.ts");
const id = "82b7a497-06a6-42a1-8700-8b4b65e83273";

function resultFixture() {
  const draw = eventSnapshot(1, "tsumo", 0);
  return reviewResult(
    [
      eventSnapshot(0, "start_kyoku"),
      draw,
      decisionSnapshot(draw, 0, "DAHAI"),
      eventSnapshot(2, "hora", 0),
    ],
    { resultId: id },
  );
}

test("shared review reads the full replay anonymously through one read-only request", async (t) => {
  const expected = resultFixture();
  const calls = [];
  t.mock.method(globalThis, "fetch", async (path, init) => {
    calls.push({ path, init });
    return Response.json(expected);
  });
  assert.deepEqual(await api.sharedResult(id), expected);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, `/api/shares/${id}`);
  assert.equal(calls[0].init.method, "GET");
  assert.equal(calls[0].init.credentials, "omit");
  assert.equal(calls[0].init.body, undefined);
});

test("owner share link retrieval sends the owner cookie without publishing or changing data", async (t) => {
  const url = `https://review.example.com/?share=${id}`;
  t.mock.method(globalThis, "fetch", async (path, init) => {
    assert.equal(path, `/api/results/${id}/share`);
    assert.equal(init.method, "GET");
    assert.equal(init.credentials, "same-origin");
    assert.equal(init.body, undefined);
    return Response.json({ url });
  });
  assert.deepEqual(await api.shareLink(id), { url });
});

test("malformed share identifiers never become requests for other API routes", async (t) => {
  const fetch = t.mock.method(globalThis, "fetch", () => {
    throw new Error("Invalid share IDs must not issue a request.");
  });
  for (const value of [
    "",
    "../history",
    `${id}/source`,
    "not-a-result",
    `${id}?export=1`,
  ]) {
    await assert.rejects(
      api.sharedResult(value),
      (error) =>
        error instanceof ApiError &&
        error.code === "share_not_found" &&
        error.status === 404,
    );
  }
  assert.equal(fetch.mock.callCount(), 0);
});

test("deleted share links retain the localized public error code", async (t) => {
  t.mock.method(globalThis, "fetch", async () =>
    Response.json(
      {
        error: {
          code: "share_not_found",
          message: "Shared analysis not found.",
        },
      },
      { status: 404 },
    ),
  );
  await assert.rejects(
    api.sharedResult(id),
    (error) =>
      error instanceof ApiError &&
      error.code === "share_not_found" &&
      error.status === 404,
  );
});

test("shared results pass through the same replay format boundary as owned results", async (t) => {
  t.mock.method(globalThis, "fetch", async () =>
    Response.json({
      ...resultFixture(),
      formatVersion: 2,
    }),
  );
  await assert.rejects(
    api.sharedResult(id),
    (error) => error instanceof ApiError && error.code === "invalid_result",
  );
});

test("leaving a shared review aborts its pending download without showing a connection error", async (t) => {
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
  const pending = api.sharedResult(id, controller.signal);
  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
});
