import test from "node:test";
import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { actionMessageKeys } from "../src/language/actionMessages.ts";

const languageDirectory = new URL("../../language/", import.meta.url);

test("every action emitted by the Java engine has an explicit display message", async () => {
  const source = await readFile(
    new URL(
      "../../../epsilon-common/src/main/java/com/epsilon/core/Action.java",
      import.meta.url,
    ),
    "utf8",
  );
  const actions = [...source.matchAll(/^\s*([A-Z_]+)\(Group\./gm)].map(
    (match) => match[1],
  );
  assert.ok(actions.length > 0, "No Java action types were found.");
  assert.deepEqual(Object.keys(actionMessageKeys).sort(), actions.sort());
});

test("all candidate actions render named labels in every shipped language", async () => {
  const files = (await readdir(languageDirectory)).filter((file) =>
    file.endsWith(".json"),
  );
  for (const file of files) {
    const messages = JSON.parse(
      await readFile(new URL(file, languageDirectory), "utf8"),
    );
    for (const [type, key] of Object.entries(actionMessageKeys)) {
      assert.equal(
        typeof messages[key],
        "string",
        `Missing candidate label: ${file}:${type}`,
      );
      assert.ok(messages[key].trim(), `Empty candidate label: ${file}:${type}`);
    }
  }
});

test("special candidates display their distinct action names", async () => {
  const messages = JSON.parse(
    await readFile(new URL("en.json", languageDirectory), "utf8"),
  );
  const labels = [
    "RIICHI_DAHAI",
    "TSUMO_AGARI",
    "RON_AGARI",
    "KYUSHU_KYUHAI",
  ].map((type) => messages[actionMessageKeys[type]]);
  assert.deepEqual(labels, [
    "Riichi",
    "Tsumo",
    "Ron",
    "Nine Terminal Initial Draw",
  ]);
});

test("kan action labels distinguish claiming, concealing and extending a meld", async () => {
  for (const file of ["ja.json", "en.json"]) {
    const messages = JSON.parse(
      await readFile(new URL(file, languageDirectory), "utf8"),
    );
    assert.equal(
      new Set(
        ["daiminkan", "ankan", "kakan"].map(
          (action) => messages[`action.${action}`],
        ),
      ).size,
      3,
    );
  }
});

test("English tile draws, drawn hands and winning calls remain distinct", async () => {
  const messages = JSON.parse(
    await readFile(new URL("en.json", languageDirectory), "utf8"),
  );
  const draw = messages["replay.draw"];
  const drawnHand = messages["event.ryukyoku"];
  const tsumo = messages["action.tsumo"];
  assert.equal(draw, "Draw");
  assert.equal(drawnHand, "Drawn hand");
  assert.equal(tsumo, "Tsumo");
  assert.equal(messages["outcome.tsumo"], tsumo);
  assert.equal(messages["outcome.ron"], messages["action.ron"]);
});
