import test from "node:test";
import assert from "node:assert/strict";
import { decodeResult } from "../../src/features/replay/model/decodeResult.ts";
import { ReviewFormatError } from "../../src/features/replay/model/formatError.ts";
import { createStateReader } from "../../src/features/replay/model/stateReader.ts";
import { buildPlaybackPositions } from "../../src/features/replay/model/timeline.ts";
import {
  ACTIONS,
  decision,
  discardAction,
  event,
  hora,
  resultDocument,
  round,
} from "./fixtures.mjs";

const choose = (seat, fields) =>
  decision(seat, [[discardAction("1m"), 1]], { chosen: 0, ...fields });
const pass = (seat, fields) =>
  decision(seat, [[ACTIONS.pass, 1]], { chosen: 0, ...fields });
const ron = (seat, fields) =>
  decision(seat, [[ACTIONS.ron, 1]], { chosen: 0, ...fields });
const doraDelta = { dora: [1, 0, "2p"] };

function decode(steps) {
  const model = decodeResult(resultDocument([round(steps)]));
  return { model, steps: model.rounds[0].steps };
}

test("a draw shows its AI choice immediately and advances directly to the discard", () => {
  const { steps } = decode([
    event("start_kyoku"),
    event("tsumo", { actor: 0 }),
    choose(0),
    event("dahai", { actor: 0 }),
    event("tsumo", { actor: 1 }),
    choose(1),
  ]);
  const positions = buildPlaybackPositions(steps);
  assert.deepEqual(
    positions.map((p) => p.event.eventType),
    ["start_kyoku", "tsumo", "dahai", "tsumo"],
  );
  assert.equal(positions[1].snapshot, steps[2]);
  assert.equal(positions[1].decisions[0], steps[2]);
  assert.equal(positions[3].decisions[0], steps[5]);
  assert.equal(steps.length, 6);
});

test("riichi declaration stays on the draw and settlement belongs to the discard response", () => {
  const { model, steps } = decode([
    event("start_kyoku"),
    event("tsumo", { actor: 0 }),
    event("reach", { actor: 0 }),
    choose(0, { cause: 1, resolve: 3 }),
    event("dahai", { actor: 0 }),
    event("reach_accepted", { actor: 0, delta: { kyotaku: 1 } }),
    pass(2, { cause: 3, resolve: 5 }),
    event("tsumo", { actor: 1 }),
  ]);
  const positions = buildPlaybackPositions(steps);
  assert.equal(positions.length, 4);
  assert.equal(positions[1].event, steps[1]);
  assert.equal(positions[1].decisions[0], steps[3]);
  assert.equal(positions[2].event, steps[4]);
  assert.equal(positions[2].snapshot, steps[6]);
  assert.equal(
    createStateReader(model).readSnapshot(0, positions[2].snapshot).kyotaku,
    1,
  );
  assert.equal(positions[2].decisions[0], steps[6]);
});

test("multiple ron winners share one settlement without hiding any discard response", () => {
  const { steps } = decode([
    event("start_kyoku"),
    event("dahai", { actor: 0 }),
    pass(1),
    event("none", { actor: 1 }),
    ron(2, { cause: 1, resolve: 3 }),
    ron(3, { cause: 1, resolve: 4 }),
    event("hora", { actor: 2, outcome: hora(2, 0, "E") }),
    event("hora", { actor: 3, outcome: hora(3, 0, "E") }),
    event("end_kyoku"),
  ]);
  const [, , passing, , firstRon, secondRon, firstWin, secondWin, settled] =
    steps;
  const positions = buildPlaybackPositions(steps);
  assert.deepEqual(
    positions.map((p) => p.kind),
    ["table", "table", "round-result"],
  );
  assert.deepEqual(positions[1].decisions, [passing, firstRon, secondRon]);
  for (const choice of [passing, firstRon, secondRon])
    assert.equal(
      positions[1].decisions.find((d) => d.actor === choice.actor).set,
      choice.set,
    );
  assert.equal(positions[2].event, firstWin);
  assert.deepEqual(positions[2].outcomeEvents, [firstWin, secondWin]);
  assert.equal(positions[2].snapshot, settled);
  assert.deepEqual(positions[2].decisions, []);
});

test("a dora update after chankan decisions preserves the earlier decision board", () => {
  const { steps } = decode([
    event("start_kyoku"),
    event("kakan", { actor: 0 }),
    pass(1, { resolve: 3 }),
    event("dora", { delta: doraDelta }),
    event("tsumo", { actor: 0 }),
  ]);
  const positions = buildPlaybackPositions(steps);
  assert.equal(positions[1].snapshot, steps[2]);
  assert.equal(positions[1].decisions[0], steps[2]);
  assert.equal(positions[2].snapshot, steps[3]);
  assert.deepEqual(positions[2].decisions, []);
});

test("dora announced before a draw decision updates that draw without a blank stop", () => {
  const { model, steps } = decode([
    event("start_kyoku"),
    event("tsumo", { actor: 0 }),
    event("dora", { delta: doraDelta }),
    choose(0, { cause: 1, resolve: 3 }),
  ]);
  const positions = buildPlaybackPositions(steps);
  assert.equal(positions.length, 2);
  assert.equal(positions[1].event, steps[1]);
  assert.equal(positions[1].snapshot, steps[3]);
  assert.equal(positions[1].decisions[0], steps[3]);
  assert.deepEqual(
    createStateReader(model).readSnapshot(0, positions[1].snapshot).dora,
    ["1m", "2p"],
  );
});

test("round settlement and final match scores have distinct stops without bookkeeping screens", () => {
  const { steps } = decode([
    event("start_kyoku"),
    event("hora", { actor: 0, outcome: hora(0, 0) }),
    event("end_kyoku"),
    event("end_game"),
  ]);
  const [, settlement, roundEnd, final] = steps;
  const positions = buildPlaybackPositions(steps, true);
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
  const { steps } = decode([
    event("start_kyoku"),
    event("tsumo", { actor: 0 }),
    event("ryukyoku", { outcome: { type: "ryukyoku" } }),
    event("end_kyoku"),
  ]);
  const positions = buildPlaybackPositions(steps);
  assert.deepEqual(
    positions.map((p) => p.kind),
    ["table", "table", "round-result"],
  );
  assert.equal(positions[2].outcomeEvents[0], steps[2]);
});

test("a last recorded round can show its scores without establishing a completed match", () => {
  const { steps } = decode([
    event("start_kyoku"),
    event("ryukyoku", { outcome: { type: "ryukyoku" } }),
    event("end_kyoku"),
  ]);
  const positions = buildPlaybackPositions(steps, true);
  assert.equal(positions.at(-1).kind, "match-result");
  assert.equal(positions.at(-1).snapshot, positions.at(-2).snapshot);
});

test("an unfinished log never invents a settlement or match result", () => {
  const { steps } = decode([
    event("start_kyoku"),
    event("tsumo", { actor: 0 }),
    choose(0),
  ]);
  const positions = buildPlaybackPositions(steps, true);
  assert.deepEqual(
    positions.map((p) => p.kind),
    ["table", "table"],
  );
  assert.equal(positions[1].snapshot, steps[2]);
});

test("explicit decision causes attach delayed callbacks to their actual board", () => {
  const { steps } = decode([
    event("start_kyoku"),
    event("tsumo", { actor: 0 }),
    event("dahai", { actor: 0 }),
    choose(0, { cause: 1 }),
  ]);
  const positions = buildPlaybackPositions(steps);
  assert.equal(positions[1].decisions[0], steps[3]);
  assert.equal(positions[1].snapshot, steps[3]);
  assert.deepEqual(positions[2].decisions, []);
});

test("explicit causes preserve separate decisions before and after a dora announcement", () => {
  const { model, steps } = decode([
    event("start_kyoku"),
    event("kakan", { actor: 0 }),
    pass(1, { resolve: 3 }),
    event("dora", { delta: doraDelta }),
    pass(2, { cause: 1, resolve: 3 }),
  ]);
  const reader = createStateReader(model);
  const positions = buildPlaybackPositions(steps);
  assert.equal(positions[1].decisions[0], steps[2]);
  assert.deepEqual(reader.readSnapshot(0, positions[1].snapshot).dora, ["1m"]);
  assert.equal(positions[2].decisions[0], steps[4]);
  assert.deepEqual(reader.readSnapshot(0, positions[2].snapshot).dora, [
    "1m",
    "2p",
  ]);
});

test("a decision caused by an event without its own board cannot attach to the current one", () => {
  const { steps } = decode([
    event("start_kyoku"),
    event("tsumo", { actor: 0 }),
    event("reach", { actor: 0 }),
    choose(0),
  ]);
  assert.throws(
    () => buildPlaybackPositions(steps),
    (error) =>
      error instanceof ReviewFormatError &&
      /unavailable replay event/.test(error.message),
  );
});
