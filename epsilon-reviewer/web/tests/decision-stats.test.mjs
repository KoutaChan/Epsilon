import test from "node:test";
import assert from "node:assert/strict";
import { summarizePlayerDecisions } from "../src/features/replay/decisionStats.ts";

function candidate(
  actionId,
  type,
  tiles,
  probability,
  chosen = false,
  tsumogiri = false,
) {
  return Object.freeze({
    actionId,
    type,
    tiles: Object.freeze(tiles),
    probability,
    chosen,
    tsumogiri,
  });
}

function decision(actor, candidates) {
  return Object.freeze({
    index: 1,
    eventIndex: 2,
    stateId: 3,
    eventType: "DAHAI",
    actor,
    decision: true,
    causeEventIndex: 1,
    candidates: Object.freeze(candidates),
    outcome: null,
  });
}

function round(...steps) {
  return Object.freeze({ steps: Object.freeze(steps) });
}

const empty = { reviewed: 0, matched: 0, badMoves: 0 };

for (const type of ["DAHAI", "RIICHI_DAHAI"]) {
  test(`${type} combines drawn and held copies before ranking either chosen variant`, () => {
    for (const drawnChosen of [false, true]) {
      const input = Object.freeze([
        round(
          decision(0, [
            candidate(1, type, ["1m"], 0.4),
            candidate(2, type, ["2m"], 0.31, !drawnChosen),
            candidate(3, type, ["2m"], 0.29, drawnChosen, true),
          ]),
        ),
      ]);
      assert.deepEqual(summarizePlayerDecisions(input), [
        { reviewed: 1, matched: 1, badMoves: 0 },
        empty,
        empty,
        empty,
      ]);
    }
  });
}

test("discard aggregation uses candidates beyond the first three", () => {
  const counts = summarizePlayerDecisions([
    round(
      decision(0, [
        candidate(1, "DAHAI", ["1m"], 0.35),
        candidate(2, "DAHAI", ["2m"], 0.25),
        candidate(3, "DAHAI", ["3m"], 0.2, true),
        candidate(4, "DAHAI", ["3m"], 0.2, false, true),
      ]),
    ),
  ]);
  assert.deepEqual(counts[0], { reviewed: 1, matched: 1, badMoves: 0 });
});

test("a recorded choice below the first three candidates is still reviewed", () => {
  const counts = summarizePlayerDecisions([
    round(
      decision(0, [
        candidate(1, "DAHAI", ["1m"], 0.4),
        candidate(2, "DAHAI", ["2m"], 0.3),
        candidate(3, "DAHAI", ["3m"], 0.26),
        candidate(4, "DAHAI", ["4m"], 0.04, true),
      ]),
    ),
  ]);
  assert.deepEqual(counts[0], { reviewed: 1, matched: 0, badMoves: 1 });
});

test("only a mismatching choice strictly below five percent is a bad move", () => {
  for (const [probability, badMoves] of [
    [0.05 - 1e-10, 1],
    [0.05, 0],
    [0.05 + 1e-10, 0],
  ]) {
    const counts = summarizePlayerDecisions([
      round(
        decision(0, [
          candidate(1, "DAHAI", ["1m"], 1 - probability),
          candidate(2, "DAHAI", ["2m"], probability, true),
        ]),
      ),
    ]);
    assert.deepEqual(counts[0], { reviewed: 1, matched: 0, badMoves });
  }
});

test("split five-percent probability is not a bad move because of addition rounding", () => {
  assert.ok(0.005 + 0.045 < 0.05);
  const counts = summarizePlayerDecisions([
    round(
      decision(0, [
        candidate(1, "DAHAI", ["1m"], 0.95),
        candidate(2, "DAHAI", ["2m"], 0.005),
        candidate(3, "DAHAI", ["2m"], 0.045, true, true),
      ]),
    ),
  ]);
  assert.deepEqual(counts[0], { reviewed: 1, matched: 0, badMoves: 0 });
});

test("a best tied choice matches despite binary64 addition rounding", () => {
  assert.ok(0.1 + 0.2 > 0.3);
  const counts = summarizePlayerDecisions([
    round(
      decision(0, [
        candidate(1, "DAHAI", ["1m"], 0.1),
        candidate(2, "DAHAI", ["1m"], 0.2, false, true),
        candidate(3, "DAHAI", ["2m"], 0.3, true),
        candidate(4, "DAHAI", ["3m"], 0.2),
        candidate(5, "DAHAI", ["4m"], 0.2),
      ]),
    ),
  ]);
  assert.deepEqual(counts[0], { reviewed: 1, matched: 1, badMoves: 0 });
});

test("a genuine probability difference is not rounded into a tie", () => {
  const counts = summarizePlayerDecisions([
    round(
      decision(0, [
        candidate(1, "DAHAI", ["1m"], 0.5 + 1e-10),
        candidate(2, "DAHAI", ["2m"], 0.5 - 1e-10, true),
      ]),
    ),
  ]);
  assert.deepEqual(counts[0], { reviewed: 1, matched: 0, badMoves: 0 });
});

test("a tied maximum below five percent is not a bad move", () => {
  const tiles = [
    "1m",
    "9m",
    "1p",
    "9p",
    "1s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
  ];
  const candidates = tiles.map((tile, index) =>
    candidate(index, "DAHAI", [tile], 0.04),
  );
  for (let index = 0; index < 12; index++)
    candidates.push(
      candidate(13 + index, "RIICHI_DAHAI", [tiles[index]], 0.04, index === 11),
    );
  const counts = summarizePlayerDecisions([round(decision(0, candidates))]);
  assert.deepEqual(counts[0], { reviewed: 1, matched: 1, badMoves: 0 });
});

test("forced choices are excluded after discard aggregation", () => {
  const counts = summarizePlayerDecisions([
    round(
      decision(0, [candidate(1, "PASS", [], 1, true)]),
      decision(0, [
        candidate(1, "DAHAI", ["5mr"], 0.6),
        candidate(2, "DAHAI", ["5mr"], 0.4, true, true),
      ]),
    ),
  ]);
  assert.deepEqual(counts, [empty, empty, empty, empty]);
});

test("unobserved decisions and applied events do not contribute", () => {
  const event = Object.freeze({
    ...decision(0, [
      candidate(1, "DAHAI", ["1m"], 0.98),
      candidate(2, "DAHAI", ["2m"], 0.02, true),
    ]),
    decision: false,
    causeEventIndex: null,
    eventType: "dahai",
  });
  const counts = summarizePlayerDecisions([
    round(
      event,
      decision(1, []),
      decision(2, [
        candidate(1, "DAHAI", ["1m"], 0.98),
        candidate(2, "DAHAI", ["2m"], 0.02),
      ]),
    ),
  ]);
  assert.deepEqual(counts, [empty, empty, empty, empty]);
});

test("red tiles and riichi discards remain distinct choices", () => {
  const counts = summarizePlayerDecisions([
    round(
      decision(0, [
        candidate(1, "DAHAI", ["5m"], 0.96),
        candidate(2, "DAHAI", ["5mr"], 0.04, true),
      ]),
      decision(0, [
        candidate(1, "DAHAI", ["5mr"], 0.96),
        candidate(2, "RIICHI_DAHAI", ["5mr"], 0.04, true),
      ]),
    ),
  ]);
  assert.deepEqual(counts[0], { reviewed: 2, matched: 0, badMoves: 2 });
});

test("different chii sequences and passing remain separate choices", () => {
  const counts = summarizePlayerDecisions([
    round(
      decision(0, [
        candidate(1, "CHI", ["3m", "4m", "5mr"], 0.04, true),
        candidate(2, "CHI", ["4m", "5mr", "6m"], 0.7),
        candidate(3, "PASS", [], 0.26),
      ]),
    ),
  ]);
  assert.deepEqual(counts[0], { reviewed: 1, matched: 0, badMoves: 1 });
});

test("pon choices retain red-tile consumption and remain separate from passing", () => {
  const counts = summarizePlayerDecisions([
    round(
      decision(0, [
        candidate(1, "PON", ["5m", "5mr", "5m"], 0.04, true),
        candidate(2, "PON", ["5m", "5m", "5m"], 0.7),
        candidate(3, "PASS", [], 0.26),
      ]),
    ),
  ]);
  assert.deepEqual(counts[0], { reviewed: 1, matched: 0, badMoves: 1 });
});

test("winning and abortive-draw choices remain separate from passing or discarding", () => {
  for (const candidates of [
    [
      candidate(1, "RON_AGARI", ["E"], 0.04, true),
      candidate(2, "PASS", [], 0.96),
    ],
    [
      candidate(1, "TSUMO_AGARI", ["E"], 0.04, true),
      candidate(2, "DAHAI", ["E"], 0.96),
    ],
    [
      candidate(1, "KYUSHU_KYUHAI", [], 0.04, true),
      candidate(2, "DAHAI", ["E"], 0.96),
    ],
  ]) {
    const counts = summarizePlayerDecisions([round(decision(0, candidates))]);
    assert.deepEqual(counts[0], { reviewed: 1, matched: 0, badMoves: 1 });
  }
});

test("all seats accumulate independently across rounds", () => {
  const match = (seat) =>
    decision(seat, [
      candidate(1, "PASS", [], 0.6, true),
      candidate(2, "RON_AGARI", ["E"], 0.4),
    ]);
  const mistake = (seat) =>
    decision(seat, [
      candidate(1, "PASS", [], 0.97),
      candidate(2, "RON_AGARI", ["E"], 0.03, true),
    ]);
  const counts = summarizePlayerDecisions(
    Object.freeze([
      round(match(0), mistake(1), match(3)),
      round(mistake(0), match(2), match(3)),
    ]),
  );
  assert.deepEqual(counts, [
    { reviewed: 2, matched: 1, badMoves: 1 },
    { reviewed: 1, matched: 0, badMoves: 1 },
    { reviewed: 1, matched: 1, badMoves: 0 },
    { reviewed: 2, matched: 2, badMoves: 0 },
  ]);
});
