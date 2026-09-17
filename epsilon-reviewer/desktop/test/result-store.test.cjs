const { test } = require("node:test");
const assert = require("node:assert/strict");
const {
  mkdtemp,
  readFile,
  writeFile,
  rm,
  readdir,
} = require("node:fs/promises");
const path = require("node:path");
const os = require("node:os");
const { gzipSync } = require("node:zlib");
const { ResultStore } = require("../dist/result-store.js");
const { resultDocument: result } = require("./result-fixture.cjs");

function saveResult(store, document, directory, source) {
  return store.saveArchive(
    document.resultId,
    gzipSync(JSON.stringify(document)),
    directory,
    source,
  );
}

test("restores results and source records after restart without deleting the original file", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "epsilon-reviewer-store-"));
  try {
    const sourceFile = path.join(root, "original.jsonl");
    await writeFile(sourceFile, '{"type":"start_game"}\n');
    const source = {
      fileName: "original.jsonl",
      recordSource: "mjai",
      base64: (await readFile(sourceFile)).toString("base64"),
    };
    const index = path.join(root, "index");
    await saveResult(
      new ResultStore(index),
      result("game-1"),
      path.join(root, "results"),
      source,
    );
    const restarted = new ResultStore(index);
    assert.deepEqual(await restarted.get("game-1"), result("game-1"));
    assert.deepEqual(await restarted.getSource("game-1"), source);
    assert.equal((await restarted.list())[0].metadata.finalScores[0], 32000);
    await restarted.remove("game-1");
    assert.deepEqual(await restarted.list(), []);
    assert.deepEqual(await readdir(path.join(root, "results")), []);
    assert.equal(await readFile(sourceFile, "utf8"), '{"type":"start_game"}\n');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("retains concurrent saves and opens existing results after a directory change", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "epsilon-reviewer-store-"));
  try {
    const store = new ResultStore(path.join(root, "index"));
    await Promise.all(
      Array.from({ length: 8 }, (_, i) =>
        saveResult(
          store,
          result(`game-${i}`),
          path.join(root, i < 4 ? "first" : "second"),
        ),
      ),
    );
    assert.equal((await store.list()).length, 8);
    for (let i = 0; i < 8; i++)
      assert.equal((await store.get(`game-${i}`)).resultId, `game-${i}`);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects invalid IDs and corrupt history without replacing the index", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "epsilon-reviewer-store-"));
  try {
    const store = new ResultStore(root);
    await assert.rejects(saveResult(store, result("../outside"), root), /ID/);
    await assert.rejects(store.remove("../outside"), /ID/);
    await writeFile(path.join(root, "history-v4.json"), "{broken");
    await assert.rejects(store.list(), SyntaxError);
    await assert.rejects(saveResult(store, result("valid"), root), SyntaxError);
    assert.equal(
      await readFile(path.join(root, "history-v4.json"), "utf8"),
      "{broken",
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("removes uncommitted result and source files after an index write failure", async (context) => {
  const fs = require("node:fs/promises");
  const root = await mkdtemp(
    path.join(os.tmpdir(), "epsilon-reviewer-save-fault-"),
  );
  try {
    const index = path.join(root, "index");
    const directory = path.join(root, "results");
    const rename = fs.rename;
    const fault = context.mock.method(fs, "rename", async (from, to) => {
      if (to === path.join(index, "history-v4.json"))
        throw Object.assign(new Error("index write failed"), {
          code: "EACCES",
        });
      return rename(from, to);
    });
    const source = { fileName: "original.jsonl", base64: "e30=" };
    await assert.rejects(
      saveResult(new ResultStore(index), result("failed"), directory, source),
      /index write failed/,
    );
    assert.equal((await readdir(directory)).length, 2);
    fault.mock.restore();
    const restarted = new ResultStore(index);
    assert.deepEqual(await restarted.list(), []);
    assert.deepEqual(await readdir(directory), []);
    assert.deepEqual(await readdir(index), []);
    await saveResult(restarted, result("failed"), directory, source);
    assert.equal((await restarted.list()).length, 1);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("retains committed results across restart and makes duplicate saves idempotent", async (context) => {
  const fs = require("node:fs/promises");
  const root = await mkdtemp(
    path.join(os.tmpdir(), "epsilon-reviewer-commit-fault-"),
  );
  try {
    const index = path.join(root, "index");
    const directory = path.join(root, "results");
    const unlink = fs.unlink;
    const fault = context.mock.method(fs, "unlink", async (file) => {
      if (file === path.join(index, "pending-v4.json"))
        throw Object.assign(new Error("cleanup failed"), { code: "EACCES" });
      return unlink(file);
    });
    const source = { fileName: "original.jsonl", base64: "e30=" };
    await assert.rejects(
      saveResult(
        new ResultStore(index),
        result("committed"),
        directory,
        source,
      ),
      /cleanup failed/,
    );
    fault.mock.restore();
    const restarted = new ResultStore(index);
    assert.deepEqual(await restarted.get("committed"), result("committed"));
    assert.deepEqual(await restarted.getSource("committed"), source);
    await saveResult(
      restarted,
      result("committed"),
      path.join(root, "new-directory"),
      source,
    );
    assert.equal((await restarted.list()).length, 1);
    assert.equal((await readdir(directory)).length, 2);
    await assert.rejects(fs.access(path.join(root, "new-directory")), {
      code: "ENOENT",
    });
    assert.deepEqual(await readdir(index), ["history-v4.json"]);
    await assert.rejects(
      saveResult(
        restarted,
        { ...result("committed"), createdAt: "2030-01-01T00:00:00Z" },
        directory,
      ),
      /overwrite/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("keeps result files when deleting the history entry fails", async (context) => {
  const fs = require("node:fs/promises");
  const root = await mkdtemp(
    path.join(os.tmpdir(), "epsilon-reviewer-delete-index-"),
  );
  try {
    const index = path.join(root, "index");
    const directory = path.join(root, "results");
    const source = { fileName: "original.jsonl", base64: "e30=" };
    const store = new ResultStore(index);
    await saveResult(store, result("retained"), directory, source);
    const rename = fs.rename;
    const fault = context.mock.method(fs, "rename", async (from, to) => {
      if (to === path.join(index, "history-v4.json"))
        throw Object.assign(new Error("index write failed"), {
          code: "EACCES",
        });
      return rename(from, to);
    });
    await assert.rejects(store.remove("retained"), /index write failed/);
    fault.mock.restore();
    assert.deepEqual(await store.get("retained"), result("retained"));
    assert.deepEqual(await store.getSource("retained"), source);
    assert.equal((await store.list()).length, 1);
    assert.equal((await readdir(directory)).length, 2);
    assert.deepEqual(await readdir(index), ["history-v4.json"]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("finishes committed deletions after restart when source cleanup fails", async (context) => {
  const fs = require("node:fs/promises");
  const root = await mkdtemp(
    path.join(os.tmpdir(), "epsilon-reviewer-delete-source-"),
  );
  try {
    const index = path.join(root, "index");
    const directory = path.join(root, "results");
    const original = path.join(root, "original.jsonl");
    await writeFile(original, '{"type":"start_game"}\n');
    const store = new ResultStore(index);
    await saveResult(store, result("removed"), directory, {
      fileName: "original.jsonl",
      base64: (await readFile(original)).toString("base64"),
    });
    const unlink = fs.unlink;
    const fault = context.mock.method(fs, "unlink", async (file) => {
      if (file.endsWith(".source.json"))
        throw Object.assign(new Error("source locked"), { code: "EACCES" });
      return unlink(file);
    });
    await assert.rejects(store.remove("removed"), /source locked/);
    assert.deepEqual(
      JSON.parse(await readFile(path.join(index, "history-v4.json"), "utf8"))
        .entries,
      [],
    );
    assert.deepEqual(await readdir(directory), [
      "removed.epsilon-reviewer.json.gz.source.json",
    ]);
    fault.mock.restore();
    const restarted = new ResultStore(index);
    assert.deepEqual(await restarted.list(), []);
    await assert.rejects(restarted.get("removed"), /history/);
    assert.deepEqual(await readdir(directory), []);
    assert.deepEqual(await readdir(index), ["history-v4.json"]);
    assert.equal(await readFile(original, "utf8"), '{"type":"start_game"}\n');
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("does not overwrite or remove files outside history with the same name", async () => {
  const fs = require("node:fs/promises");
  const root = await mkdtemp(
    path.join(os.tmpdir(), "epsilon-reviewer-collision-"),
  );
  try {
    const directory = path.join(root, "results");
    await fs.mkdir(directory);
    const original = path.join(directory, "collision.epsilon-reviewer.json.gz");
    await writeFile(original, "external content");
    const store = new ResultStore(path.join(root, "index"));
    await assert.rejects(
      saveResult(store, result("collision"), directory),
      /overwrite/,
    );
    assert.deepEqual(await store.list(), []);
    assert.equal(await readFile(original, "utf8"), "external content");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("preserves current scored results after restart", async () => {
  const root = await mkdtemp(
    path.join(os.tmpdir(), "epsilon-reviewer-versioned-"),
  );
  try {
    const scored = result("scored");
    scored.rounds[0].steps[0] = {
      ...scored.rounds[0].steps[0],
      eventType: "hora",
      actor: 0,
      outcome: {
        type: "hora",
        actor: 0,
        target: 1,
        winningTile: "3m",
        deltas: [5800, -5800, 0, 0],
        reason: null,
        details: {
          han: 3,
          fu: 30,
          points: 5800,
          yakuman: 0,
          yaku: [{ code: "RIICHI", han: 1, yakuman: 0 }],
          uraIndicators: ["3p"],
        },
      },
    };
    const index = path.join(root, "index");
    const store = new ResultStore(index);
    await saveResult(store, scored, path.join(root, "results"));
    const restarted = new ResultStore(index);
    assert.deepEqual(await restarted.get("scored"), scored);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

for (const version of [1, 2, 3, 5]) {
  test(`rejects unsupported result version ${version} before creating storage files`, async () => {
    const root = await mkdtemp(
      path.join(os.tmpdir(), "epsilon-reviewer-unsupported-save-"),
    );
    try {
      const store = new ResultStore(path.join(root, "index"));
      const unsupported = result("unsupported");
      unsupported.formatVersion = version;
      await assert.rejects(
        saveResult(store, unsupported, path.join(root, "results")),
        (error) => error.code === "invalid_result",
      );
      assert.deepEqual(await readdir(root), []);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  test(`rejects stored result version ${version} while preserving its file and history entry`, async () => {
    const root = await mkdtemp(
      path.join(os.tmpdir(), "epsilon-reviewer-unsupported-read-"),
    );
    try {
      const index = path.join(root, "index");
      const directory = path.join(root, "results");
      const store = new ResultStore(index);
      await saveResult(store, result("unsupported"), directory);
      await saveResult(store, result("current"), directory);
      const file = path.join(directory, "unsupported.epsilon-reviewer.json.gz");
      const unsupported = result("unsupported");
      unsupported.formatVersion = version;
      const original = gzipSync(JSON.stringify(unsupported));
      await writeFile(file, original);
      const originalHistory = await readFile(
        path.join(index, "history-v4.json"),
        "utf8",
      );
      const restarted = new ResultStore(index);
      await assert.rejects(
        restarted.get("unsupported"),
        (error) => error.code === "invalid_result",
      );
      assert.deepEqual(await restarted.get("current"), result("current"));
      assert.equal((await restarted.list()).length, 2);
      assert.deepEqual(await readFile(file), original);
      assert.equal(
        await readFile(path.join(index, "history-v4.json"), "utf8"),
        originalHistory,
      );
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
}

test("stores and exports the exact server archive without recompression", async () => {
  const root = await mkdtemp(
    path.join(os.tmpdir(), "epsilon-reviewer-archive-"),
  );
  try {
    const document = result("archive");
    const bytes = gzipSync(JSON.stringify(document, null, 2), { level: 1 });
    const store = new ResultStore(path.join(root, "index"));
    await store.saveArchive("archive", bytes, path.join(root, "results"));
    assert.deepEqual(await store.readArchive("archive"), bytes);
    assert.deepEqual(await store.get("archive"), document);
    assert.deepEqual(
      await readFile(
        path.join(root, "results", "archive.epsilon-reviewer.json.gz"),
      ),
      bytes,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("rejects corrupt archives and mismatched IDs before creating storage files", async () => {
  const root = await mkdtemp(
    path.join(os.tmpdir(), "epsilon-reviewer-archive-input-"),
  );
  try {
    const store = new ResultStore(path.join(root, "index"));
    for (const bytes of [
      Buffer.from("plain JSON"),
      gzipSync("{broken"),
      gzipSync("null"),
      gzipSync(
        JSON.stringify({ ...result("archive"), rounds: [{ steps: [] }] }),
      ),
    ]) {
      await assert.rejects(
        store.saveArchive("archive", bytes, root),
        (error) => error.code === "invalid_result",
      );
    }
    await assert.rejects(
      store.saveArchive(
        "archive",
        gzipSync(JSON.stringify(result("other"))),
        root,
      ),
      (error) => error.code === "result_mismatch",
    );
    assert.deepEqual(await readdir(root), []);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("starts a separate current-format index without changing earlier history files", async () => {
  const root = await mkdtemp(
    path.join(os.tmpdir(), "epsilon-reviewer-fresh-index-"),
  );
  try {
    const oldIndex = JSON.stringify({
      version: 1,
      entries: [{ file: "old-result.json" }],
    });
    await writeFile(path.join(root, "history.json"), oldIndex);
    await writeFile(path.join(root, "pending.json"), "old pending transaction");
    const store = new ResultStore(root);
    assert.deepEqual(await store.list(), []);
    await saveResult(store, result("current"), path.join(root, "results"));
    assert.equal(
      await readFile(path.join(root, "history.json"), "utf8"),
      oldIndex,
    );
    assert.equal(
      await readFile(path.join(root, "pending.json"), "utf8"),
      "old pending transaction",
    );
    assert.equal((await store.list()).length, 1);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
