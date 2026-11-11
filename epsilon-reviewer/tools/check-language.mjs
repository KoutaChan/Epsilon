import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { CLIENT_ERROR_CODES, SERVER_ERROR_CODES } from "../protocol/errors.ts";

// language/<言語>/<ファイル>.json を確かめる。ja を基準に、どの言語にも同じファイルと同じキーがあり、文言の {名前} もそろっていること。
// キーの先頭の語で、そのキーを置くファイルが決まる。ほかのファイルの語で始まるキーは置けない。errors.json のキーはエラーコードと 1 対 1 に対応する。
const OWNERS = {
  "shell.json": [
    "language",
    "app",
    "common",
    "source",
    "home",
    "model",
    "history",
    "confirm",
    "progress",
    "storage",
    "settings",
    "share",
  ],
  "replay.json": [
    "replay",
    "candidate",
    "scenario",
    "grade",
    "event",
    "outcome",
  ],
  "mahjong.json": ["mahjong", "tile", "action", "yaku"],
  "desktop.json": ["desktop"],
  "errors.json": ["errors"],
};
const root = new URL("../language/", import.meta.url);
const read = async (language, file) =>
  JSON.parse(await readFile(new URL(`${language}/${file}`, root), "utf8"));
const placeholders = (message) =>
  [...new Set(message.match(/\{\w+\}/g) ?? [])].sort();

const languages = (await readdir(root, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)
  .sort();
assert.ok(languages.includes("ja"), "language/ja is missing.");
const expectedFiles = Object.keys(OWNERS).sort();
let total = 0;
for (const file of expectedFiles) {
  const source = await read("ja", file);
  const keys = Object.keys(source).sort();
  for (const key of keys)
    assert.ok(
      OWNERS[file].includes(key.split(".")[0]),
      `Key outside the file's prefixes: ja/${file}:${key}`,
    );
  total += keys.length;
  for (const language of languages) {
    const messages = await read(language, file);
    assert.deepEqual(
      Object.keys(messages).sort(),
      keys,
      `Keys differ: ${language}/${file}`,
    );
    for (const key of keys) {
      assert.equal(
        typeof messages[key],
        "string",
        `Invalid message: ${language}/${file}:${key}`,
      );
      assert.ok(
        messages[key].trim(),
        `Empty message: ${language}/${file}:${key}`,
      );
      assert.deepEqual(
        placeholders(messages[key]),
        placeholders(source[key]),
        `Placeholder mismatch: ${language}/${file}:${key}`,
      );
    }
  }
}
for (const language of languages) {
  const files = (await readdir(new URL(`${language}/`, root))).sort();
  assert.deepEqual(files, expectedFiles, `Files differ: language/${language}`);
}
const errorKeys = Object.keys(await read("ja", "errors.json")).sort();
const codes = [...SERVER_ERROR_CODES, ...CLIENT_ERROR_CODES]
  .map((code) => `errors.${code}`)
  .sort();
assert.deepEqual(
  errorKeys,
  codes,
  "errors.json must match protocol/errors.ts.",
);
console.log(
  `Validated ${languages.length} languages, ${expectedFiles.length} files and ${total} message keys.`,
);
