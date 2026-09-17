import test from "node:test";
import assert from "node:assert/strict";
import { buildPlaybackPositions } from "../src/features/replay/timeline.ts";
import {
  eventSnapshot as event,
  decisionSnapshot as decision,
} from "./reviewFixtures.mjs";

test("a draw shows its AI choice immediately and advances directly to the discard", () => {
  const start = event(0, "start_kyoku");
  const draw = event(1, "tsumo", 0);
  const choice = decision(draw, 0, "DAHAI", { eventIndex: 2 });
  const discard = event(2, "dahai", 0);
  const nextDraw = event(3, "tsumo", 1);
  const nextChoice = decision(nextDraw, 1, "DAHAI", { eventIndex: 4 });
  const raw = Object.freeze([
    start,
    draw,
    choice,
    discard,
    nextDraw,
    nextChoice,
  ]);
  const positions = buildPlaybackPositions(raw);
  assert.deepEqual(
    positions.map((p) => p.event.eventType),
    ["start_kyoku", "tsumo", "dahai", "tsumo"],
  );
  assert.equal(positions[1].snapshot, choice);
  assert.equal(positions[1].decisions[0], choice);
  assert.equal(positions[3].decisions[0], nextChoice);
  assert.equal(raw.length, 6);
});

test("riichi declaration stays on the draw and settlement belongs to the discard response", () => {
  const draw = event(1, "tsumo", 0);
  const riichiChoice = decision(draw, 0, "RIICHI_DAHAI", { eventIndex: 3 });
  const discard = event(3, "dahai", 0);
  const accepted = event(4, "reach_accepted", 0, { kyotaku: 1 });
  const response = decision(discard, 2, "PASS", {
    eventIndex: 5,
    stateId: accepted.stateId,
    kyotaku: 1,
  });
  const positions = buildPlaybackPositions([
    event(0, "start_kyoku"),
    draw,
    event(2, "reach", 0),
    riichiChoice,
    discard,
    accepted,
    response,
    event(5, "tsumo", 1),
  ]);
  assert.equal(positions.length, 4);
  assert.equal(positions[1].event, draw);
  assert.equal(positions[1].decisions[0], riichiChoice);
  assert.equal(positions[2].event, discard);
  assert.equal(positions[2].snapshot, response);
  assert.equal(positions[2].snapshot.kyotaku, 1);
  assert.equal(positions[2].decisions[0], response);
});

test("multiple ron winners share one settlement without hiding any discard response", () => {
  const discard = event(1, "dahai", 0);
  const pass = decision(discard, 1, "PASS", { eventIndex: 2 });
  const firstRon = decision(discard, 2, "RON_AGARI", { eventIndex: 3 });
  const secondRon = decision(discard, 3, "RON_AGARI", { eventIndex: 4 });
  const firstWin = event(3, "hora", 2, {
    outcome: {
      type: "hora",
      actor: 2,
      target: 0,
      winningTile: "E",
      deltas: null,
      details: null,
    },
  });
  const secondWin = event(4, "hora", 3, {
    outcome: {
      type: "hora",
      actor: 3,
      target: 0,
      winningTile: "E",
      deltas: null,
      details: null,
    },
  });
  const settled = event(5, "end_kyoku");
  const positions = buildPlaybackPositions([
    event(0, "start_kyoku"),
    discard,
    pass,
    event(2, "none", 1),
    firstRon,
    secondRon,
    firstWin,
    secondWin,
    settled,
  ]);
  assert.deepEqual(
    positions.map((p) => p.kind),
    ["table", "table", "round-result"],
  );
  assert.deepEqual(positions[1].decisions, [pass, firstRon, secondRon]);
  for (const choice of [pass, firstRon, secondRon])
    assert.equal(
      positions[1].decisions.find((d) => d.actor === choice.actor).candidates,
      choice.candidates,
    );
  assert.equal(positions[2].event, firstWin);
  assert.deepEqual(positions[2].outcomeEvents, [firstWin, secondWin]);
  assert.equal(positions[2].snapshot, settled);
  assert.deepEqual(positions[2].decisions, []);
});

test("a dora update after chankan decisions preserves the earlier decision board", () => {
  const kan = event(1, "kakan", 0);
  const response = decision(kan, 1, "PASS", { eventIndex: 3 });
  const dora = event(2, "dora", -1, { dora: ["1m", "2p"] });
  const positions = buildPlaybackPositions([
    event(0, "start_kyoku"),
    kan,
    response,
    dora,
    event(3, "tsumo", 0),
  ]);
  assert.equal(positions[1].snapshot, response);
  assert.equal(positions[1].decisions[0], response);
  assert.equal(positions[2].snapshot, dora);
  assert.deepEqual(positions[2].decisions, []);
});

test("dora announced before a draw decision updates that draw without a blank stop", () => {
  const draw = event(1, "tsumo", 0);
  const dora = event(2, "dora", -1, { dora: ["1m", "2p"] });
  const choice = decision(draw, 0, "DAHAI", {
    eventIndex: 3,
    stateId: dora.stateId,
    dora: dora.dora,
  });
  const positions = buildPlaybackPositions([
    event(0, "start_kyoku"),
    draw,
    dora,
    choice,
  ]);
  assert.equal(positions.length, 2);
  assert.equal(positions[1].event, draw);
  assert.equal(positions[1].snapshot, choice);
  assert.equal(positions[1].decisions[0], choice);
  assert.deepEqual(positions[1].snapshot.dora, ["1m", "2p"]);
});

test("round settlement and final match scores have distinct stops without bookkeeping screens", () => {
  const settlement = event(1, "hora", 0);
  const roundEnd = event(2, "end_kyoku");
  const final = event(3, "end_game");
  const positions = buildPlaybackPositions(
    [event(0, "start_kyoku"), settlement, roundEnd, final],
    true,
  );
  assert.deepEqual(
    positions.map((p) => p.kind),
    ["table", "round-result", "match-result"],
  );
  assert.equal(positions[1].event, settlement);
  assert.equal(positions[1].snapshot, roundEnd);
  assert.equal(positions[2].event, final);
  assert.equal(positions[2].snapshot, final);
  assert.equal(positions[2].outcomeEvents, positions[1].outcomeEvents);
});

test("a draw or abort has one reviewable round result", () => {
  const draw = event(2, "ryukyoku");
  const positions = buildPlaybackPositions([
    event(0, "start_kyoku"),
    event(1, "tsumo", 0),
    draw,
    event(3, "end_kyoku"),
  ]);
  assert.deepEqual(
    positions.map((p) => p.kind),
    ["table", "table", "round-result"],
  );
  assert.equal(positions[2].outcomeEvents[0], draw);
});

test("a last recorded round can show its scores without establishing a completed match", () => {
  const settlement = event(1, "ryukyoku");
  const positions = buildPlaybackPositions(
    [event(0, "start_kyoku"), settlement, event(2, "end_kyoku")],
    true,
  );
  assert.equal(positions.at(-1).kind, "match-result");
  assert.equal(positions.at(-1).snapshot, positions.at(-2).snapshot);
});

test("an unfinished log never invents a settlement or match result", () => {
  const draw = event(1, "tsumo", 0);
  const choice = decision(draw, 0, "DAHAI", { eventIndex: 2 });
  const positions = buildPlaybackPositions(
    [event(0, "start_kyoku"), draw, choice],
    true,
  );
  assert.deepEqual(
    positions.map((p) => p.kind),
    ["table", "table"],
  );
  assert.equal(positions[1].snapshot, choice);
});

test("explicit decision causes attach delayed callbacks to their actual board", () => {
  const start = event(0, "start_kyoku");
  const draw = event(1, "tsumo", 0);
  const discard = event(2, "dahai", 0);
  const choice = decision(draw, 0, "DAHAI", { eventIndex: 2 });
  const positions = buildPlaybackPositions([start, draw, discard, choice]);
  assert.equal(positions[1].decisions[0], choice);
  assert.equal(positions[1].snapshot, choice);
  assert.deepEqual(positions[2].decisions, []);
});

test("explicit causes preserve separate decisions before and after a dora announcement", () => {
  const kan = event(1, "kakan", 0);
  const before = decision(kan, 1, "PASS", { eventIndex: 3 });
  const dora = event(2, "dora", -1, { dora: ["1m", "2p"] });
  const after = decision(kan, 2, "PASS", {
    eventIndex: 3,
    stateId: dora.stateId,
    dora: dora.dora,
  });
  const positions = buildPlaybackPositions([
    event(0, "start_kyoku"),
    kan,
    before,
    dora,
    after,
  ]);
  assert.equal(positions[1].decisions[0], before);
  assert.deepEqual(positions[1].snapshot.dora, ["1m"]);
  assert.equal(positions[2].decisions[0], after);
  assert.deepEqual(positions[2].snapshot.dora, ["1m", "2p"]);
});

test("a decision with an unavailable cause cannot attach itself to the current board", () => {
  const start = event(0, "start_kyoku");
  const absent = event(9, "tsumo", 0);
  assert.throws(
    () => buildPlaybackPositions([start, decision(absent, 0, "DAHAI")]),
    /unavailable replay event/,
  );
});
