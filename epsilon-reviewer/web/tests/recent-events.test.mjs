import test from "node:test";
import assert from "node:assert/strict";
import { describeReplayEvent } from "../src/features/replay/recentEvents.ts";
import { eventSnapshot, playerSnapshot } from "./reviewFixtures.mjs";

function playerEvent(type, fields, eventFields = {}) {
  return eventSnapshot(5, type, 1, {
    players: [0, 1, 2, 3].map((seat) =>
      playerSnapshot(seat, seat === 1 ? fields : {}),
    ),
    ...eventFields,
  });
}

function meld(type, tiles) {
  return Object.freeze({
    type,
    tiles: Object.freeze(tiles),
    from: 2,
    calledIndex: 0,
    addedIndex: type === "KAKAN" ? 1 : -1,
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
      river: [
        { tile: "E", riichi: false, tsumogiri: false, called: false },
        { tile: "5mr", riichi, tsumogiri, called: false },
      ],
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

test("added kan identifies the upgraded pon even before an older kan in the meld list", () => {
  const olderKan = meld("KAKAN", ["2p", "2p", "2p", "2p"]);
  const before = playerEvent("tsumo", {
    draw: "5mr",
    melds: [meld("PON", ["5m", "5m", "5m"]), olderKan],
  });
  const addedKan = meld("KAKAN", ["5m", "5mr", "5m", "5m"]);
  const after = playerEvent("kakan", { melds: [addedKan, olderKan] });
  const details = describeReplayEvent(after, before, false, 0);
  assert.equal(details.action, "action.kakan");
  assert.equal(details.tiles, addedKan.tiles);
});

test("dora events show only the newly revealed indicator", () => {
  const event = eventSnapshot(4, "dora", -1, { dora: ["9m", "F"] });
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
    const event = playerEvent(
      "hora",
      {},
      {
        outcome: {
          type: "hora",
          actor: 1,
          target,
          winningTile: "5sr",
          deltas: null,
          details: null,
        },
      },
    );
    const details = describeReplayEvent(event, null, false, 0);
    assert.equal(details.action, action);
    assert.deepEqual(details.tiles, ["5sr"]);
  });
}

test("non-tile events have translated labels without invented tile images", () => {
  for (const type of [
    "start_game",
    "start_kyoku",
    "end_kyoku",
    "end_game",
    "reach",
    "reach_accepted",
    "none",
    "ryukyoku",
  ]) {
    const details = describeReplayEvent(eventSnapshot(0, type), null, true, 0);
    assert.equal(details.action, `event.${type}`);
    assert.deepEqual(details.tiles, []);
  }
});
