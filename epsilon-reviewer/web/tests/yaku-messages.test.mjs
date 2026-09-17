import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { yakuLabel } from "../src/language/yakuMessages.ts";

const languages = ["ja", "en"].map((language) =>
  JSON.parse(
    readFileSync(
      new URL(`../../language/${language}.json`, import.meta.url),
      "utf8",
    ),
  ),
);

test("recorded normal and bonus yaku use localized names in each shipped language", () => {
  for (const messages of languages) {
    const translate = (key) => messages[key].replace(/\{(\w+)\}/g, "");
    for (const code of [
      "RIICHI",
      "YAKUHAI_SEAT_E",
      "YAKUHAI_ROUND_S",
      "SUUANKOU_TANKI",
      "DORA",
      "URADORA",
      "AKADORA",
    ]) {
      assert.equal(yakuLabel(code, translate), messages[`yaku.${code}`]);
    }
  }
});

test("unrecognized source yaku cannot be mistaken for inherited message keys", () => {
  for (const messages of languages) {
    const translate = (key) => messages[key].replace(/\{(\w+)\}/g, "");
    for (const code of [
      "TENHOU_999",
      "MJAI_future_yaku",
      "toString",
      "constructor",
      "__proto__",
    ]) {
      assert.equal(yakuLabel(code, translate), messages["outcome.unknownYaku"]);
    }
  }
});
