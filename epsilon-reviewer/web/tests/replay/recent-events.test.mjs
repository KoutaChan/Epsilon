import test from "node:test";
import assert from "node:assert/strict";
import { describeReplayEvent } from "../../src/features/replay/model/recentEvents.ts";
import { discard, meld, hora, playerSnapshot, snapshot } from "./fixtures.mjs";

function playerEvent(type, fields, eventFields = {}) {
  return snapshot(type, 1, {
    players: [0, 1, 2, 3].map((seat) =>
      playerSnapshot(seat, seat === 1 ? fields : {}),
    ),
    ...eventFields,
  });
}

test("draw images respect the selected perspective and all-hands visibility", () => {
  const event = playerEvent("tsumo", { draw: "5pr" });
  for (const [reveal, seat, tile] of [
    [true, 0, "5pr"],
    [false, 1, "5pr"],
    [false, 0, "back"],
  ]) {
    const details = describeReplayEvent(event, null, reveal, seat);
    assert.equal(details.action, "replay.draw");
    assert.deepEqual(details.tiles, [tile]);
    assert.equal(details.player, event.players[1]);
  }
});

for (const [riichi, tsumogiri, action] of [
  [false, false, "action.dahai"],
  [false, true, "action.dahaiTsumogiri"],
  [true, false, "action.riichi"],
  [true, true, "action.riichiTsumogiri"],
]) {
  test(`discard images identify riichi=${riichi} and tsumogiri=${tsumogiri}`, () => {
    const event = playerEvent("dahai", {
      river: [discard("E"), discard("5mr", { riichi, tsumogiri })],
    });
    const details = describeReplayEvent(event, null, false, 0);
    assert.equal(details.action, action);
    assert.deepEqual(details.tiles, ["5mr"]);
  });
}

for (const [type, tiles] of [
  ["CHI", ["4s", "5sr", "6s"]],
  ["PON", ["E", "E", "E"]],
  ["DAIMINKAN", ["P", "P", "P", "P"]],
  ["ANKAN", ["9m", "9m", "9m", "9m"]],
]) {
  test(`${type} shows every tile of the newly formed meld`, () => {
    const newest = meld(type, tiles);
    const event = playerEvent(type.toLowerCase(), {
      melds: [meld("PON", ["7p", "7p", "7p"]), newest],
    });
    const details = describeReplayEvent(event, null, false, 0);
    assert.equal(details.action, `action.${type.toLowerCase()}`);
    assert.equal(details.tiles, newest.tiles);
  });
}

test("a call after set-aside norths shows the new meld, and a kita shows one north", () => {
  const pon = meld("PON", ["P", "P", "P"]);
  const call = describeReplayEvent(
    playerEvent("pon", { melds: [pon], nuki: 2 }),
    null,
    false,
    0,
  );
  assert.equal(call.tiles, pon.tiles);
  const kita = describeReplayEvent(
    playerEvent("kita", { melds: [pon], nuki: 3 }),
    null,
    false,
    0,
  );
  assert.deepEqual([kita.action, kita.tiles], ["action.nuki", ["N"]]);
});

test("added kan identifies the upgraded pon even before an older kan in the meld list", () => {
  const olderKan = meld("KAKAN", ["2p", "2p", "2p", "2p"], { addedIndex: 1 });
  const before = playerEvent("tsumo", {
    draw: "5mr",
    melds: [meld("PON", ["5m", "5m", "5m"]), olderKan],
  });
  const addedKan = meld("KAKAN", ["5m", "5mr", "5m", "5m"], { addedIndex: 1 });
  const after = playerEvent("kakan", { melds: [addedKan, olderKan] });
  const details = describeReplayEvent(after, before, false, 0);
  assert.equal(details.action, "action.kakan");
  assert.equal(details.tiles, addedKan.tiles);
});

test("dora events show only the newly revealed indicator", () => {
  const event = snapshot("dora", -1, { dora: ["9m", "F"] });
  assert.deepEqual(describeReplayEvent(event, null, false, 0), {
    player: null,
    action: "event.dora",
    tiles: ["F"],
  });
});

for (const [target, action] of [
  [1, "action.tsumo"],
  [2, "action.ron"],
]) {
  test(`${action} shows the recorded red winning tile even with other hands hidden`, () => {
    const event = playerEvent("hora", {}, { outcome: hora(1, target, "5sr") });
    const details = describeReplayEvent(event, null, false, 0);
    assert.equal(details.action, action);
    assert.deepEqual(details.tiles, ["5sr"]);
  });
}

test("non-tile events have translated labels without invented tile images", () => {
  for (const type of [
    "start_kyoku",
    "end_kyoku",
    "end_game",
    "reach",
    "reach_accepted",
    "none",
    "ryukyoku",
  ]) {
    const details = describeReplayEvent(snapshot(type), null, true, 0);
    assert.equal(details.action, `event.${type}`);
    assert.deepEqual(details.tiles, []);
  }
});
