import test from "node:test";
import assert from "node:assert/strict";
import {
  dateLabel,
  modelKey,
  modelLabel,
  playerName,
  roundName,
  seatName,
  sourceLabel,
  windName,
} from "../../src/language/format.ts";

/** テスト用の Translate。訳さずに、キーの後ろに値の JSON をつないで返す（値が無ければキーだけ）。 */
const t = (key, values) => (values ? `${key}${JSON.stringify(values)}` : key);

test("model keys and labels", () => {
  const model = {
    modelId: "prism",
    revision: "v551",
    displayName: "Prism",
    version: "v2",
  };
  assert.equal(modelKey(model), '["prism","v551"]');
  assert.equal(modelKey({ ...model, displayName: "other" }), modelKey(model));
  assert.equal(modelLabel(model), "Prism v2");
  assert.equal(modelLabel({ ...model, version: "" }), "Prism");
});

test("sources and players", () => {
  assert.equal(sourceLabel("tenhou", t), "source.tenhou");
  assert.equal(sourceLabel("majsoul", t), "source.majsoul");
  assert.equal(playerName("Kota", 2, t), "Kota");
  assert.equal(playerName("", 2, t), 'common.player{"number":3}');
});

test("winds, seats relative to the dealer and rounds", () => {
  assert.equal(windName(0, t), "mahjong.east");
  assert.equal(windName(5, t, true), "mahjong.southShort");
  const four = { dealer: 2, seats: 4 };
  const three = { dealer: 2, seats: 3 };
  assert.equal(
    seatName(0, { dealer: 0, seats: 4 }, t),
    'mahjong.seat{"wind":"mahjong.east"}',
  );
  assert.equal(seatName(1, four, t), 'mahjong.seat{"wind":"mahjong.north"}');
  assert.equal(seatName(1, three, t), 'mahjong.seat{"wind":"mahjong.west"}');
  assert.equal(
    roundName({ roundIndex: 5, seats: 4 }, t),
    'mahjong.round{"wind":"mahjong.south","number":2}',
  );
  assert.equal(
    roundName({ roundIndex: 3, seats: 4 }, t, true),
    'mahjong.roundShort{"wind":"mahjong.eastShort","number":4}',
  );
  assert.equal(
    roundName({ roundIndex: 5, seats: 3 }, t),
    'mahjong.round{"wind":"mahjong.south","number":3}',
  );
  assert.equal(
    roundName({ roundIndex: 6, seats: 3 }, t),
    'mahjong.round{"wind":"mahjong.west","number":1}',
  );
});

test("dates use the language's month, day and time, and unreadable dates say so", () => {
  const date = "2026-11-04T05:05:55.421Z";
  const expected = (locale) =>
    new Intl.DateTimeFormat(locale, {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(date));
  assert.equal(dateLabel(date, "ja", t), expected("ja"));
  assert.equal(dateLabel(date, "en", t), expected("en"));
  assert.equal(dateLabel(date, "ja", t), expected("ja"));
  assert.equal(dateLabel("not a date", "ja", t), "common.unknownDate");
});
