import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { runnerImport } from "vite";
import {
  CLIENT_ERROR_CODES,
  SERVER_ERROR_CODES,
} from "../../../protocol/errors.ts";

// catalog.ts は import.meta.glob で言語のファイルを集めるので、Vite のモジュールランナーで読み込む。
const web = fileURLToPath(new URL("../../", import.meta.url));
const { module: catalog } = await runnerImport("/src/language/catalog.ts", {
  root: web,
  configFile: false,
  logLevel: "silent",
});
const read = async (language, file) =>
  JSON.parse(
    await readFile(
      new URL(`../../../language/${language}/${file}`, import.meta.url),
      "utf8",
    ),
  );

test("languages are listed in folder order with their own names", () => {
  assert.deepEqual(catalog.languages, [
    { id: "en", name: "English" },
    { id: "ja", name: "日本語" },
  ]);
  assert.equal(catalog.isLanguage("ja"), true);
  assert.equal(catalog.isLanguage("fr"), false);
  assert.throws(() => catalog.translator("fr"), RangeError);
});

test("the page reads shell, replay, mahjong and errors but not desktop", async () => {
  const t = catalog.translator("ja");
  for (const file of [
    "shell.json",
    "replay.json",
    "mahjong.json",
    "errors.json",
  ]) {
    const messages = await read("ja", file);
    for (const key of Object.keys(messages))
      assert.equal(t(key), messages[key]);
  }
  const [desktopKey] = Object.keys(await read("ja", "desktop.json"));
  assert.equal(t(desktopKey), desktopKey);
});

test("placeholders take the given values and keep the token when a value is missing", () => {
  const ja = catalog.translator("ja");
  const en = catalog.translator("en");
  assert.equal(ja("common.player", { number: 3 }), "プレイヤー 3");
  assert.equal(en("common.player", { number: 3 }), "Player 3");
  assert.equal(ja("common.player"), "プレイヤー {number}");
  assert.equal(ja("common.player", { other: 1 }), "プレイヤー {number}");
  assert.equal(
    ja("history.analysedAt", { date: "{number}" }),
    "解析：{number}",
  );
  assert.equal(
    en("confirm.summary", { source: "Tenhou", count: 12 }),
    "Tenhou · Hands: 12",
  );
});

test("inherited names are not placeholder values", () => {
  const t = catalog.translator("en");
  assert.equal(t("common.player", {}), "Player {number}");
  assert.equal(
    t("common.player", Object.create({ number: 7 })),
    "Player {number}",
  );
});

test("errors map to their message key and anything else to operation_failed", () => {
  for (const code of [...SERVER_ERROR_CODES, ...CLIENT_ERROR_CODES]) {
    assert.equal(catalog.errorKey({ code }), `errors.${code}`);
    assert.notEqual(
      catalog.translator("en")(`errors.${code}`),
      `errors.${code}`,
    );
  }
  for (const error of [
    null,
    undefined,
    "busy",
    new Error("boom"),
    { code: "no_such_code" },
    new DOMException("closed", "AbortError"),
  ])
    assert.equal(catalog.errorKey(error), "errors.operation_failed");
});
