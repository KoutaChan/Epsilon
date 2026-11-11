import assert from "node:assert/strict";
import { PassThrough } from "node:stream";
import { test } from "node:test";
import { setImmediate as tick } from "node:timers/promises";
import { FRAME_BODY_LIMIT } from "../../protocol/stdio.ts";
import { EngineClient } from "../src/engine/client.ts";
import { encodeFrame, FrameDecoder } from "../src/engine/frames.ts";

/** 偽の stdin・stdout につないだ EngineClient を作る。requests には Java の側が受けたフレームが入る。 */
function harness() {
  const input = new PassThrough();
  const output = new PassThrough();
  const requests = [];
  const decoder = new FrameDecoder();
  input.on("data", (chunk) => requests.push(...decoder.push(chunk)));
  const engine = { broken: 0 };
  const client = new EngineClient(input, output, () => {
    engine.broken += 1;
  });
  const send = (header, body) => {
    for (const chunk of encodeFrame(header, body)) output.write(chunk);
  };
  return { client, requests, send, engine, output };
}

test("ids increase from 1 and replies are matched by id in any order", async () => {
  const { client, requests, send } = harness();
  const first = client.request("job", { jobId: "a" });
  const second = client.request(
    "importBytes",
    {},
    new Uint8Array([5, 6]).buffer,
  );
  const third = client.request("history", {});
  await tick();
  assert.deepEqual(
    requests.map(({ header }) => header),
    [
      { id: 1, command: "job", request: { jobId: "a" } },
      { id: 2, command: "importBytes", request: {} },
      { id: 3, command: "history", request: {} },
    ],
  );
  assert.deepEqual(new Uint8Array(requests[1].body), new Uint8Array([5, 6]));
  send({ id: 3, ok: true, response: { results: [] } });
  send({
    id: 1,
    ok: false,
    error: {
      code: "job_not_found",
      message: "The analysis job was not found.",
    },
  });
  send({ id: 2, ok: true, response: { recordId: "r" } });
  assert.deepEqual(await third, { ok: true, response: { results: [] } });
  assert.deepEqual(await first, {
    ok: false,
    error: {
      code: "job_not_found",
      message: "The analysis job was not found.",
    },
  });
  assert.deepEqual(await second, { ok: true, response: { recordId: "r" } });
});

test("the ready event is accepted and a document reply carries its body", async () => {
  const { client, send } = harness();
  send({ event: "ready" });
  const result = client.request("result", { resultId: "x" });
  const removal = client.request("deleteResult", { resultId: "x" });
  await tick();
  send({ id: 1, ok: true }, new Uint8Array([0x1f, 0x8b]).buffer);
  send({ id: 2, ok: true });
  const reply = await result;
  assert.equal(reply.ok, true);
  assert.equal("response" in reply, false);
  assert.deepEqual(new Uint8Array(reply.body), new Uint8Array([0x1f, 0x8b]));
  assert.deepEqual(await removal, { ok: true });
});

test("after a failed event the exit answers pending and later requests with its error", async () => {
  const { client, send } = harness();
  const pending = client.request("models", {});
  const locked = {
    code: "data_locked",
    message: "The data directory is used by another reviewer.",
  };
  send({ event: "failed", error: locked });
  await tick();
  client.close();
  assert.deepEqual(await pending, { ok: false, error: locked });
  assert.deepEqual(await client.request("history", {}), {
    ok: false,
    error: locked,
  });
  assert.deepEqual(client.failure, locked);
});

test("the exit answers pending and later requests with engine_failed", async () => {
  const { client } = harness();
  const pending = client.request("models", {});
  client.close();
  const stopped = {
    ok: false,
    error: { code: "engine_failed", message: "The analysis engine stopped." },
  };
  assert.deepEqual(await pending, stopped);
  assert.deepEqual(await client.request("models", {}), stopped);
});

test("an invalid frame or an unknown reply breaks the engine", async (t) => {
  t.mock.method(console, "error", () => undefined);
  for (const header of [
    { id: 9, ok: true },
    { event: "started" },
    { id: 1, ok: false, error: { code: "nope", message: "" } },
  ]) {
    const { client, send, engine } = harness();
    const pending = client.request("models", {});
    await tick();
    send(header);
    assert.equal((await pending).error.code, "engine_failed");
    assert.equal(engine.broken, 1);
  }
  const { client, output, engine } = harness();
  const pending = client.request("models", {});
  output.write(Buffer.from([0xff, 0xff, 0xff, 0xff]));
  assert.equal((await pending).error.code, "engine_failed");
  assert.equal(engine.broken, 1);
});

test("a request over the frame limit is refused without writing", async () => {
  const { client, requests } = harness();
  const reply = await client.request(
    "importBytes",
    {},
    new ArrayBuffer(FRAME_BODY_LIMIT + 1),
  );
  assert.equal(reply.error.code, "invalid_request");
  await tick();
  assert.equal(requests.length, 0);
});
