import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { BRIDGE_CHANNELS } from "../../protocol/bridge.ts";
import { registerBridge } from "../src/bridge.ts";
import { Languages } from "../src/language.ts";
import { SettingsStore } from "../src/settings.ts";

const LANGUAGES = Languages.load(
  fileURLToPath(new URL("../../language/", import.meta.url)),
);

/**
 * 偽の ipcMain・エンジン・ウィンドウにつないだブリッジを作る。call で IPC を呼び出す。送信元を信用するかは
 * host.trusted で決まる。
 */
async function bridge(t) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "reviewer-bridge-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const settings = await SettingsStore.open(
    path.join(directory, "settings.json"),
    { resultsDirectory: path.join(directory, "results"), language: "ja" },
    (language) => LANGUAGES.has(language),
  );
  const engine = {
    calls: [],
    restarts: 0,
    failed: false,
    replies: {},
    async request(command, request, body) {
      this.calls.push(
        body === undefined ? { command, request } : { command, request, body },
      );
      const reply = this.replies[command];
      return typeof reply === "function"
        ? reply(request)
        : (reply ?? { ok: true, response: {} });
    },
    async restart() {
      this.restarts += 1;
    },
  };
  const host = {
    trusted: true,
    title: null,
    dialogs: 0,
    directory: null,
    exportPath: null,
    checkSender: () =>
      host.trusted
        ? null
        : {
            code: "invalid_request",
            message: "IPC sender origin is not trusted.",
          },
    chooseDirectory: async () => {
      host.dialogs += 1;
      return host.directory;
    },
    chooseExportPath: async () => {
      host.dialogs += 1;
      return host.exportPath;
    },
    setTitle: (title) => {
      host.title = title;
    },
  };
  const handlers = new Map();
  registerBridge(
    { handle: (channel, listener) => handlers.set(channel, listener) },
    {
      engine,
      settings,
      languages: LANGUAGES,
      host,
    },
  );
  const call = (channel, ...args) => handlers.get(channel)({}, ...args);
  return { directory, settings, engine, host, call, handlers };
}

const invoke = (call, ...args) => call(BRIDGE_CHANNELS.invoke, ...args);

test("every bridge channel is registered", async (t) => {
  const { handlers } = await bridge(t);
  assert.deepEqual(
    [...handlers.keys()].sort(),
    Object.values(BRIDGE_CHANNELS).sort(),
  );
});

test("invoke passes only renderer commands; exportResult and HTTP-only commands are refused", async (t) => {
  const { engine, call } = await bridge(t);
  for (const command of [
    "exportResult",
    "downloadResult",
    "shareLink",
    "sharedScenario",
    "toString",
    "nope",
    7,
  ])
    assert.equal(
      (await invoke(call, command, {})).error.code,
      "invalid_request",
      String(command),
    );
  assert.equal(
    (await invoke(call, "history", null)).error.code,
    "invalid_request",
  );
  assert.equal(
    (await invoke(call, "history", [])).error.code,
    "invalid_request",
  );
  assert.equal(engine.calls.length, 0);
  engine.replies.history = { ok: true, response: { results: [] } };
  assert.deepEqual(await invoke(call, "history", {}), {
    ok: true,
    response: { results: [] },
  });
  assert.deepEqual(engine.calls, [{ command: "history", request: {} }]);
});

test("importFile needs an absolute path and only importBytes carries a body", async (t) => {
  const { engine, call } = await bridge(t);
  for (const request of [{}, { path: "record.mjlog" }, { path: 3 }])
    assert.equal(
      (await invoke(call, "importFile", request)).error.code,
      "invalid_request",
    );
  const file = path.resolve("牌譜.mjlog");
  assert.equal((await invoke(call, "importFile", { path: file })).ok, true);
  assert.equal(
    (await invoke(call, "importBytes", {})).error.code,
    "invalid_request",
  );
  assert.equal(
    (await invoke(call, "importBytes", {}, new Uint8Array(2))).error.code,
    "invalid_request",
  );
  assert.equal(
    (await invoke(call, "models", {}, new ArrayBuffer(1))).error.code,
    "invalid_request",
  );
  const bytes = new ArrayBuffer(3);
  assert.equal((await invoke(call, "importBytes", {}, bytes)).ok, true);
  assert.deepEqual(engine.calls, [
    { command: "importFile", request: { path: file } },
    { command: "importBytes", request: {}, body: bytes },
  ]);
});

test("a document reply reaches the renderer as the same ArrayBuffer", async (t) => {
  const { engine, call } = await bridge(t);
  const gzip = new Uint8Array([0x1f, 0x8b]).buffer;
  engine.replies.result = { ok: true, body: gzip };
  const reply = await invoke(call, "result", { resultId: "r" });
  assert.equal(reply.body, gzip);
});

test("the folder cannot change while an analysis started from the renderer is running", async (t) => {
  const { directory, settings, engine, host, call } = await bridge(t);
  let status = "running";
  engine.replies.startAnalysis = {
    ok: true,
    response: { jobId: "j1", model: {} },
  };
  engine.replies.job = ({ jobId }) => ({
    ok: true,
    response: { jobId, status, progress: 10 },
  });
  await invoke(call, "startAnalysis", {
    recordId: "r",
    modelId: "m",
    revision: "v",
  });
  host.directory = path.join(directory, "他");
  const refused = await call(BRIDGE_CHANNELS.directory);
  assert.equal(refused.error.code, "analysis_active");
  assert.equal(host.dialogs, 0);
  status = "completed";
  await invoke(call, "job", { jobId: "j1" });
  engine.calls.length = 0;
  const changed = await call(BRIDGE_CHANNELS.directory);
  assert.deepEqual(changed, {
    ok: true,
    value: { resultsDirectory: host.directory, language: "ja" },
  });
  assert.deepEqual(engine.calls, []);
  assert.equal(engine.restarts, 1);
  assert.equal(settings.value.resultsDirectory, host.directory);
});

test("a job the renderer stopped polling is checked again, and a cancelled dialog changes nothing", async (t) => {
  const { settings, engine, host, call } = await bridge(t);
  engine.replies.startAnalysis = {
    ok: true,
    response: { jobId: "j2", model: {} },
  };
  await invoke(call, "startAnalysis", {
    recordId: "r",
    modelId: "m",
    revision: "v",
  });
  engine.replies.job = {
    ok: false,
    error: { code: "engine_failed", message: "The analysis engine stopped." },
  };
  const before = settings.value;
  assert.deepEqual(await call(BRIDGE_CHANNELS.directory), {
    ok: true,
    value: before,
  });
  assert.equal(host.dialogs, 1);
  assert.equal(engine.restarts, 0);
  assert.deepEqual(engine.calls.at(-1), {
    command: "job",
    request: { jobId: "j2" },
  });
});

// 取り込んだ牌譜は Java のメモリにだけある。同じフォルダーを選び直しただけで起動し直すと、画面に残る牌譜が使えなくなる。
test("choosing the current folder again restarts the engine only after a failed start", async (t) => {
  const { settings, engine, host, call } = await bridge(t);
  const before = settings.value;
  host.directory = before.resultsDirectory;
  assert.deepEqual(await call(BRIDGE_CHANNELS.directory), {
    ok: true,
    value: before,
  });
  assert.equal(host.dialogs, 1);
  assert.equal(engine.restarts, 0);
  engine.failed = true;
  assert.deepEqual(await call(BRIDGE_CHANNELS.directory), {
    ok: true,
    value: before,
  });
  assert.equal(engine.restarts, 1);
});

test("setLanguage checks the id, saves it and retitles the window", async (t) => {
  const { settings, host, call } = await bridge(t);
  assert.equal(
    (await call(BRIDGE_CHANNELS.language, "../ja")).error.code,
    "invalid_request",
  );
  assert.equal(host.title, null);
  const reply = await call(BRIDGE_CHANNELS.language, "en");
  assert.equal(reply.value.language, "en");
  assert.equal(settings.value.language, "en");
  assert.equal(host.title, "Epsilon AI Reviewer");
  assert.deepEqual(await call(BRIDGE_CHANNELS.settings), {
    ok: true,
    value: settings.value,
  });
});

test("exportResult writes through Java only after the save dialog", async (t) => {
  const { directory, engine, host, call } = await bridge(t);
  assert.deepEqual(await call(BRIDGE_CHANNELS.export, "r1"), {
    ok: true,
    value: { saved: false },
  });
  assert.equal(engine.calls.length, 0);
  host.exportPath = path.join(directory, "r1.epsilon-reviewer.json.gz");
  assert.deepEqual(await call(BRIDGE_CHANNELS.export, "r1"), {
    ok: true,
    value: { saved: true },
  });
  assert.deepEqual(engine.calls, [
    {
      command: "exportResult",
      request: { resultId: "r1", path: host.exportPath },
    },
  ]);
  engine.replies.exportResult = {
    ok: false,
    error: {
      code: "result_not_found",
      message: "The analysis result was not found.",
    },
  };
  assert.equal(
    (await call(BRIDGE_CHANNELS.export, "r1")).error.code,
    "result_not_found",
  );
});

test("untrusted senders and failing handlers get an error reply instead of an exception", async (t) => {
  t.mock.method(console, "error", () => undefined);
  const { engine, host, call } = await bridge(t);
  host.trusted = false;
  for (const channel of Object.values(BRIDGE_CHANNELS))
    assert.equal(
      (await call(channel, "history", {})).error.code,
      "invalid_request",
      channel,
    );
  assert.equal(engine.calls.length, 0);
  host.trusted = true;
  engine.request = async () => {
    throw new Error("pipe closed");
  };
  assert.deepEqual(await invoke(call, "history", {}), {
    ok: false,
    error: { code: "operation_failed", message: "pipe closed" },
  });
});
