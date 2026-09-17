import test from "node:test";
import assert from "node:assert/strict";
import { playbackReducer } from "../src/features/replay/playback.ts";
import { buildPlaybackPositions } from "../src/features/replay/timeline.ts";
import { eventSnapshot } from "./reviewFixtures.mjs";

const initial = {
  roundIndex: 0,
  stepIndex: 0,
  seat: 2,
  reveal: true,
  playing: false,
  animateDraw: false,
};

function makeRound(events, includeMatchResult = false) {
  return buildPlaybackPositions(
    events.map((eventType, index) =>
      eventSnapshot(
        index,
        eventType,
        ["tsumo", "dahai", "hora"].includes(eventType) ? 0 : -1,
      ),
    ),
    includeMatchResult,
  );
}
const timelines = Object.freeze([
  makeRound(["start_kyoku", "tsumo", "dahai", "hora", "end_kyoku"]),
  makeRound(["start_kyoku", "tsumo", "dahai", "ryukyoku", "end_kyoku"]),
  makeRound(["start_kyoku", "tsumo", "hora", "end_kyoku", "end_game"], true),
]);
const step = (state, delta, automatic = false) =>
  playbackReducer(state, { type: "step", delta, timelines, automatic });

function location(state) {
  return [
    state.roundIndex,
    state.stepIndex,
    timelines[state.roundIndex][state.stepIndex].kind,
  ];
}

test("manual navigation stops at the beginning and pauses playback", () => {
  const previous = step({ ...initial, playing: true }, -1);
  assert.equal(previous.stepIndex, 0);
  assert.equal(previous.playing, false);
  assert.equal(previous.animateDraw, false);
  const next = step(initial, 1);
  assert.equal(next.stepIndex, 1);
  assert.equal(next.animateDraw, true);
});

test("manual next enters the next recorded round and previous returns to its settlement", () => {
  const settlement = { ...initial, stepIndex: 3, reveal: false };
  const next = step(settlement, 1);
  assert.deepEqual(location(next), [1, 0, "table"]);
  assert.equal(next.seat, 2);
  assert.equal(next.reveal, false);
  assert.equal(next.playing, false);
  const previous = step(next, -1);
  assert.deepEqual(location(previous), [0, 3, "round-result"]);
  assert.equal(previous.animateDraw, false);
});

test("automatic playback pauses on every settlement and cannot start from result screens", () => {
  for (const roundIndex of [0, 1, 2]) {
    const settlementIndex = timelines[roundIndex].findIndex(
      (position) => position.kind === "round-result",
    );
    const settlement = step(
      { ...initial, roundIndex, stepIndex: settlementIndex - 1, playing: true },
      1,
      true,
    );
    assert.deepEqual(location(settlement), [
      roundIndex,
      settlementIndex,
      "round-result",
    ]);
    assert.equal(settlement.playing, false);
    assert.equal(settlement.animateDraw, false);
    assert.equal(
      playbackReducer(settlement, { type: "play", timelines }).playing,
      false,
    );
    assert.deepEqual(location(step(settlement, 1, true)), location(settlement));
  }
});

test("playback can resume after manually continuing into the next round", () => {
  const nextRound = step({ ...initial, stepIndex: 3 }, 1);
  const playing = playbackReducer(nextRound, { type: "play", timelines });
  assert.equal(playing.playing, true);
  const nextDraw = step(playing, 1, true);
  assert.deepEqual(location(nextDraw), [1, 1, "table"]);
  assert.equal(nextDraw.playing, true);
  assert.equal(nextDraw.animateDraw, true);
});

test("the final round settlement leads to a distinct match result and supports returning", () => {
  const finalSettlement = { ...initial, roundIndex: 2, stepIndex: 2 };
  const matchResult = step(finalSettlement, 1);
  assert.deepEqual(location(matchResult), [2, 3, "match-result"]);
  assert.equal(matchResult.playing, false);
  assert.equal(matchResult.animateDraw, false);
  assert.deepEqual(location(step(matchResult, 1)), location(matchResult));
  assert.equal(
    playbackReducer(matchResult, { type: "play", timelines }).playing,
    false,
  );
  assert.deepEqual(location(step(matchResult, -1)), location(finalSettlement));
  assert.deepEqual(location(step(step(matchResult, -1), -1)), [2, 1, "table"]);
});

test("a full match walks forward and backward through each recorded position exactly once", () => {
  const expected = timelines.flatMap((positions, roundIndex) =>
    positions.map((position, stepIndex) => [
      roundIndex,
      stepIndex,
      position.kind,
    ]),
  );
  let state = initial;
  const forward = [location(state)];
  for (let count = 1; count < expected.length; count++) {
    state = step(state, 1);
    forward.push(location(state));
  }
  assert.deepEqual(forward, expected);
  const backward = [location(state)];
  for (let count = 1; count < expected.length; count++) {
    state = step(state, -1);
    backward.push(location(state));
  }
  assert.deepEqual(backward, expected.toReversed());
});

test("round selection and timeline seeking leave results without resetting perspective", () => {
  const final = { ...initial, roundIndex: 2, stepIndex: 3, reveal: false };
  const sought = playbackReducer(final, { type: "seek", index: 1 });
  assert.deepEqual(location(sought), [2, 1, "table"]);
  assert.equal(sought.animateDraw, false);
  assert.equal(sought.reveal, false);
  assert.equal(
    playbackReducer(sought, { type: "play", timelines }).playing,
    true,
  );
  const selected = playbackReducer(final, { type: "round", index: 1 });
  assert.deepEqual(selected, {
    roundIndex: 1,
    stepIndex: 0,
    seat: 2,
    reveal: false,
    playing: false,
    animateDraw: false,
  });
});

test("perspective changes preserve settlement position and do not replay draw animation", () => {
  const state = playbackReducer(
    { ...initial, stepIndex: 3, playing: true, animateDraw: true },
    { type: "seat", seat: 1 },
  );
  assert.equal(state.stepIndex, 3);
  assert.equal(state.seat, 1);
  assert.equal(state.playing, false);
  assert.equal(state.animateDraw, false);
});
