import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";

const directory = new URL("../language/", import.meta.url);
const files = (await readdir(directory)).filter((file) =>
  file.endsWith(".json"),
);
const source = JSON.parse(
  await readFile(new URL("ja.json", directory), "utf8"),
);
const keys = Object.keys(source).sort();
const placeholders = (message) =>
  [...new Set(message.match(/\{\w+\}/g) ?? [])].sort();
for (const file of files) {
  const messages = JSON.parse(await readFile(new URL(file, directory), "utf8"));
  assert.deepEqual(
    Object.keys(messages).sort(),
    keys,
    `Locale keys differ: ${file}`,
  );
  for (const key of keys) {
    assert.equal(
      typeof messages[key],
      "string",
      `Invalid message: ${file}:${key}`,
    );
    assert.ok(messages[key].trim(), `Empty message: ${file}:${key}`);
    assert.deepEqual(
      placeholders(messages[key]),
      placeholders(source[key]),
      `Placeholder mismatch: ${file}:${key}`,
    );
  }
}
console.log(
  `Validated ${files.length} language files and ${keys.length} message keys.`,
);
