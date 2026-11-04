import assert from "node:assert/strict";
import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { SettingsStore } from "../src/settings.ts";

const isLanguage = (language) => language === "ja" || language === "en";

async function folder(t) {
  const directory = await mkdtemp(path.join(os.tmpdir(), "reviewer-settings-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  return {
    directory,
    file: path.join(directory, "settings.json"),
    defaults: { resultsDirectory: path.join(directory, "結果"), language: "ja" },
  };
}

test("a missing file gives the defaults without writing", async (t) => {
  const { directory, file, defaults } = await folder(t);
  const settings = await SettingsStore.open(file, defaults, isLanguage);
  assert.deepEqual(settings.value, defaults);
  assert.deepEqual(await readdir(directory), []);
});

test("an unreadable or invalid file is replaced by the defaults", async (t) => {
  const { directory, file, defaults } = await folder(t);
  const invalid = [
    "{",
    "[]",
    JSON.stringify({ resultsDirectory: "relative", language: "ja" }),
    JSON.stringify({ resultsDirectory: directory, language: "../unknown" }),
    JSON.stringify({ resultsDirectory: directory, language: "en", serverUrl: "http://x" }),
  ];
  for (const text of invalid) {
    await writeFile(file, text);
    const settings = await SettingsStore.open(file, defaults, isLanguage);
    assert.deepEqual(settings.value, defaults, text);
    assert.equal(await readFile(file, "utf8"), JSON.stringify(defaults) + "\n");
  }
  assert.deepEqual(await readdir(directory), ["settings.json"]);
});

test("concurrent updates commit in order and survive a restart", async (t) => {
  const { directory, file, defaults } = await folder(t);
  const settings = await SettingsStore.open(file, defaults, isLanguage);
  const [first, second] = await Promise.all([
    settings.update({ language: "en" }),
    settings.update({ resultsDirectory: path.join(directory, "新しい") }),
  ]);
  assert.deepEqual(first, { ...defaults, language: "en" });
  assert.deepEqual(second, { resultsDirectory: path.join(directory, "新しい"), language: "en" });
  assert.deepEqual((await SettingsStore.open(file, defaults, isLanguage)).value, second);
});

test("a failed write keeps the current value", async (t) => {
  const { directory, defaults } = await folder(t);
  const blocked = path.join(directory, "blocked");
  const settings = await SettingsStore.open(path.join(blocked, "settings.json"), defaults, isLanguage);
  await writeFile(blocked, "a file where the folder should be");
  await assert.rejects(settings.update({ language: "en" }));
  assert.deepEqual(settings.value, defaults);
  assert.deepEqual(await settings.update({}).catch(() => "rejected"), "rejected");
});
