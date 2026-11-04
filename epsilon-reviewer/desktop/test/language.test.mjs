import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { Languages } from "../src/language.ts";

const LANGUAGE = fileURLToPath(new URL("../../language/", import.meta.url));

test("the shipped files merge per language and every desktop key resolves", () => {
  const languages = Languages.load(LANGUAGE);
  assert.equal(languages.message("en", "desktop.exportTitle"), "Save analysis result");
  assert.equal(languages.message("en", "app.title"), "Epsilon AI Reviewer");
  assert.equal(languages.message("ja", "app.title"), "Epsilon AI レビュアー");
  for (const key of ["exportTitle", "exportFilter", "directoryTitle", "startupTitle", "startupFailure"])
    for (const language of ["ja", "en"])
      assert.ok(languages.message(language, `desktop.${key}`).length > 0);
});

test("a key missing in a language falls back to ja, and locales pick the closest id", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "reviewer-language-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  await mkdir(path.join(directory, "ja"));
  await mkdir(path.join(directory, "en"));
  await writeFile(path.join(directory, "ja", "shell.json"), JSON.stringify({ "app.title": "題" }));
  await writeFile(
    path.join(directory, "ja", "desktop.json"),
    JSON.stringify({ "desktop.exportTitle": "保存", "desktop.directoryTitle": "保存先" }),
  );
  await writeFile(path.join(directory, "en", "desktop.json"), JSON.stringify({ "desktop.exportTitle": "Save" }));
  const languages = Languages.load(directory);
  assert.equal(languages.message("en", "desktop.exportTitle"), "Save");
  assert.equal(languages.message("en", "desktop.directoryTitle"), "保存先");
  assert.equal(languages.message("en", "app.title"), "題");
  assert.throws(() => languages.message("en", "desktop.startupTitle"), /Missing language key/);
  assert.equal(languages.select("en-US"), "en");
  assert.equal(languages.select("en"), "en");
  assert.equal(languages.select("fr-FR"), "ja");
  assert.equal(languages.select(""), "ja");
  assert.equal(languages.has("../ja"), false);
});

test("a language folder without ja is refused", async (t) => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "reviewer-language-"));
  t.after(() => rm(directory, { recursive: true, force: true }));
  await mkdir(path.join(directory, "en"));
  assert.throws(() => Languages.load(directory), /ja is missing/);
  assert.throws(() => Languages.load(path.join(directory, "none")), { code: "ENOENT" });
});
