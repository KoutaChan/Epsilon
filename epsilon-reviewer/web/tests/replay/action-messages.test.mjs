import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { ACTION_TYPES } from "../../../protocol/result.ts";
import { actionMessageKeys } from "../../src/language/actionMessages.ts";

const languageDirectory = new URL("../../../language/", import.meta.url);

async function messages(language) {
  const files = await Promise.all(
    ["replay.json", "mahjong.json"].map(async (file) =>
      JSON.parse(
        await readFile(
          new URL(`${language}/${file}`, languageDirectory),
          "utf8",
        ),
      ),
    ),
  );
  return Object.assign({}, ...files);
}

test("every action emitted by the Java engine has an explicit display message", async () => {
  const source = await readFile(
    new URL(
      "../../../../epsilon-mahjong/src/main/java/com/epsilon/mahjong/action/Action.java",
      import.meta.url,
    ),
    "utf8",
  );
  const types = source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .match(/enum Type\b[^{]*\{([^;]*);/)?.[1];
  assert.ok(types, "The Java action type enum was not found.");
  const actions = [...types.matchAll(/([A-Z_]+)(?:\(RuleFlag\.([A-Z_]+)\))?/g)]
    .map((match) => match[1]);
  assert.ok(actions.length > 0, "No Java action types were found.");
  assert.deepEqual(Object.keys(actionMessageKeys).sort(), actions.sort());
  assert.deepEqual(Object.keys(actionMessageKeys), [...ACTION_TYPES]);
});

test("all candidate actions render named labels in every shipped language", async () => {
  const languages = (await readdir(languageDirectory, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name);
  assert.ok(languages.includes("en") && languages.includes("ja"));
  for (const language of languages) {
    const catalog = await messages(language);
    for (const [type, key] of Object.entries(actionMessageKeys)) {
      assert.equal(
        typeof catalog[key],
        "string",
        `Missing candidate label: ${language}:${type}`,
      );
      assert.ok(
        catalog[key].trim(),
        `Empty candidate label: ${language}:${type}`,
      );
    }
  }
});

test("special candidates display their distinct action names", async () => {
  const catalog = await messages("en");
  const labels = [
    "RIICHI_DAHAI",
    "TSUMO_AGARI",
    "RON_AGARI",
    "KYUSHU_KYUHAI",
  ].map((type) => catalog[actionMessageKeys[type]]);
  assert.deepEqual(labels, ["Riichi", "Tsumo", "Ron", "Nine Terminals"]);
});

test("kan action labels distinguish claiming, concealing and extending a meld", async () => {
  for (const language of ["ja", "en"]) {
    const catalog = await messages(language);
    assert.equal(
      new Set(
        ["daiminkan", "ankan", "kakan"].map(
          (action) => catalog[`action.${action}`],
        ),
      ).size,
      3,
    );
  }
});

test("English tile draws, drawn hands and winning calls remain distinct", async () => {
  const catalog = await messages("en");
  assert.equal(catalog["replay.draw"], "Draw");
  assert.equal(catalog["event.ryukyoku"], "Drawn hand");
  assert.equal(catalog["action.tsumo"], "Tsumo");
  assert.equal(catalog["outcome.tsumo"], catalog["action.tsumo"]);
  assert.equal(catalog["outcome.ron"], catalog["action.ron"]);
});
