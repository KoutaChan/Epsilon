import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile, mkdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { EngineProcess, engineLaunch } from "../src/engine/process.ts";

const FAKE_ENGINE = fileURLToPath(
  new URL("./fake-engine.mjs", import.meta.url),
);

/**
 * 偽のエンジン（node fake-engine.mjs）を本物と同じ環境で起動する EngineProcess を作る。state.mode は起動のたびに
 * 読む。テストの後は、エンジンを止めてから一時フォルダーを消す（子プロセスの作業フォルダーは、動いている間は消せない）。
 */
async function fixture(t, stopGraceMs) {
  const root = await mkdtemp(path.join(os.tmpdir(), "reviewer-engine-"));
  const userData = path.join(root, "結果 user");
  const engine = path.join(root, "engine");
  await mkdir(engine);
  const state = {
    mode: "ready",
    ignoreEof: false,
    spawns: 0,
    results: path.join(root, "保存先"),
  };
  const process_ = new EngineProcess(() => {
    state.spawns += 1;
    const real = engineLaunch(engine, userData, state.results);
    const env = { ...real.env, FAKE_ENGINE: state.mode };
    if (state.ignoreEof) env.FAKE_IGNORE_EOF = "1";
    return { ...real, command: process.execPath, args: [FAKE_ENGINE], env };
  }, stopGraceMs);
  t.after(async () => {
    await process_.stop();
    await rm(root, { recursive: true, force: true, maxRetries: 5 });
  });
  return {
    engine,
    state,
    process_,
    log: path.join(userData, "logs", "engine-stderr.log"),
  };
}

test("the child gets the engine folder, the environment and answers over the pipes", async (t) => {
  const { engine, state, process_ } = await fixture(t);
  process_.start();
  const models = await process_.request("models", {});
  assert.equal(models.ok, true);
  assert.equal(models.response.cwd, engine);
  assert.equal(models.response.data, state.results);
  assert.equal(
    models.response.models,
    path.join(engine, "models", "models.json"),
  );
  assert.equal(models.response.threads, "2");
  const imported = await process_.request(
    "importBytes",
    {},
    new Uint8Array(70000).buffer,
  );
  assert.deepEqual(imported, { ok: true, response: { bytes: 70000 } });
  const result = await process_.request("result", { resultId: "r" });
  assert.deepEqual(
    new Uint8Array(result.body),
    new Uint8Array([0x1f, 0x8b, 8, 0, 1, 2, 3]),
  );
  const [slow, quick] = [
    process_.request("job", { jobId: "slow" }),
    process_.request("job", { jobId: "other" }),
  ];
  assert.equal((await quick).error.code, "job_not_found");
  assert.equal((await slow).response.status, "running");
  assert.equal(state.spawns, 1);
});

test("a crash answers the pending request with engine_failed and the next request respawns", async (t) => {
  const { state, process_ } = await fixture(t);
  const first = await process_.request("models", {});
  const crash = await process_.request("job", { jobId: "crash" });
  assert.deepEqual(crash, {
    ok: false,
    error: { code: "engine_failed", message: "The analysis engine stopped." },
  });
  const again = await process_.request("models", {});
  assert.equal(again.ok, true);
  assert.notEqual(again.response.pid, first.response.pid);
  assert.equal(state.spawns, 2);
});

test("a failed start is answered without respawning until restart", async (t) => {
  const { state, process_ } = await fixture(t);
  state.mode = "locked";
  const locked = {
    code: "data_locked",
    message: "The data directory is used by another reviewer.",
  };
  assert.deepEqual(await process_.request("history", {}), {
    ok: false,
    error: locked,
  });
  assert.deepEqual(await process_.request("history", {}), {
    ok: false,
    error: locked,
  });
  assert.equal(state.spawns, 1);
  // 起動の失敗は、子プロセスが終わったときに覚える。
  const deadline = performance.now() + 5000;
  while (!process_.failed && performance.now() < deadline) await delay(5);
  assert.equal(process_.failed, true);
  state.mode = "ready";
  await process_.restart();
  assert.equal(process_.failed, false);
  assert.equal((await process_.request("models", {})).ok, true);
  assert.equal(state.spawns, 2);
});

test("an engine that cannot start answers engine_failed and is tried again on the next request", async (t) => {
  t.mock.method(console, "error", () => undefined);
  const root = await mkdtemp(path.join(os.tmpdir(), "reviewer-engine-"));
  t.after(() => rm(root, { recursive: true, force: true, maxRetries: 5 }));
  const stopped = {
    ok: false,
    error: { code: "engine_failed", message: "The analysis engine stopped." },
  };
  let spawns = 0;
  const missing = new EngineProcess(() => {
    spawns += 1;
    return engineLaunch(
      path.join(root, "no-engine"),
      path.join(root, "user"),
      root,
    );
  });
  assert.deepEqual(await missing.request("models", {}), stopped);
  assert.deepEqual(await missing.request("models", {}), stopped);
  assert.equal(spawns, 2);
  await writeFile(
    path.join(root, "blocked"),
    "a file where the user data folder should be",
  );
  const unlogged = new EngineProcess(() =>
    engineLaunch(root, path.join(root, "blocked"), root),
  );
  assert.deepEqual(await unlogged.request("models", {}), stopped);
});

test("stop ends stdin, kills a child that keeps running and refuses later requests", async (t) => {
  const { state, process_ } = await fixture(t, 300);
  state.ignoreEof = true;
  assert.equal((await process_.request("models", {})).ok, true);
  const started = performance.now();
  await process_.stop();
  assert.ok(performance.now() - started >= 290);
  assert.equal(
    (await process_.request("models", {})).error.code,
    "engine_failed",
  );
  assert.equal(state.spawns, 1);
});

test("stderr goes to the log, which is emptied at spawn once it exceeds 1 MiB", async (t) => {
  const { log, process_ } = await fixture(t);
  await mkdir(path.dirname(log), { recursive: true });
  await writeFile(log, "x".repeat(1024 * 1024 + 1));
  const first = await process_.request("models", {});
  assert.equal(await readFile(log, "utf8"), `started ${first.response.pid}\n`);
  await process_.restart();
  const second = await process_.request("models", {});
  assert.equal(
    await readFile(log, "utf8"),
    `started ${first.response.pid}\nstarted ${second.response.pid}\n`,
  );
});
