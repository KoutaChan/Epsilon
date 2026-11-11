import test from "node:test";
import assert from "node:assert/strict";
import { summarizePlayerDecisions } from "../../src/features/replay/model/decisionStats.ts";
import { ACTIONS, discardAction } from "./fixtures.mjs";

// 候補は [行動 ID, 値] の行。chosen は選んだ行。価値を推定するモデルは値が損失、しないモデルは確率。
const dahai = (tile, value, chosen = false, tsumogiri = false) => ({
  row: [discardAction(tile, { tsumogiri }), value],
  chosen,
});
const riichi = (tile, value, chosen = false, tsumogiri = false) => ({
  row: [discardAction(tile, { tsumogiri, riichi: true }), value],
  chosen,
});
const action = (id, value, chosen = false, tiles) => ({
  row: tiles ? [id, value, tiles] : [id, value],
  chosen,
});
const discardOf = { DAHAI: dahai, RIICHI_DAHAI: riichi };

function decision(actor, entries) {
  const chosen = entries.findIndex((entry) => entry.chosen);
  return {
    decision: true,
    actor,
    set: {
      candidates: entries.map((entry) => entry.row),
      ...(chosen >= 0 ? { chosen } : {}),
    },
  };
}
const policy = (...rounds) => ({ temperature: undefined, rounds });
const valued = (...rounds) => ({ temperature: 0.002, rounds });
const round = (...steps) => ({ seats: 4, steps });

const empty = { reviewed: 0, matched: 0, badMoves: 0, meanLoss: null };

for (const type of ["DAHAI", "RIICHI_DAHAI"]) {
  test(`${type} combines drawn and held copies before ranking either chosen variant`, () => {
    const discard = discardOf[type];
    for (const drawnChosen of [false, true]) {
      const input = policy(
        round(
          decision(0, [
            discard("1m", 0.4),
            discard("2m", 0.31, !drawnChosen),
            discard("2m", 0.29, drawnChosen, true),
          ]),
        ),
      );
      assert.deepEqual(summarizePlayerDecisions(input), [
        { reviewed: 1, matched: 1, badMoves: 0, meanLoss: null },
        empty,
        empty,
        empty,
      ]);
    }
  });
}

test("discard aggregation uses candidates beyond the first three", () => {
  const counts = summarizePlayerDecisions(
    policy(
      round(
        decision(0, [
          dahai("1m", 0.35),
          dahai("2m", 0.25),
          dahai("3m", 0.2, true),
          dahai("3m", 0.2, false, true),
        ]),
      ),
    ),
  );
  assert.deepEqual(counts[0], {
    reviewed: 1,
    matched: 1,
    badMoves: 0,
    meanLoss: null,
  });
});

test("a recorded choice below the first three candidates is still reviewed", () => {
  const counts = summarizePlayerDecisions(
    policy(
      round(
        decision(0, [
          dahai("1m", 0.4),
          dahai("2m", 0.3),
          dahai("3m", 0.26),
          dahai("4m", 0.04, true),
        ]),
      ),
    ),
  );
  assert.deepEqual(counts[0], {
    reviewed: 1,
    matched: 0,
    badMoves: 1,
    meanLoss: null,
  });
});

test("only a mismatching choice strictly below five percent is a bad move", () => {
  for (const [probability, badMoves] of [
    [0.05 - 1e-10, 1],
    [0.05, 0],
    [0.05 + 1e-10, 0],
  ]) {
    const counts = summarizePlayerDecisions(
      policy(
        round(
          decision(0, [
            dahai("1m", 1 - probability),
            dahai("2m", probability, true),
          ]),
        ),
      ),
    );
    assert.deepEqual(counts[0], {
      reviewed: 1,
      matched: 0,
      badMoves,
      meanLoss: null,
    });
  }
});

test("split five-percent probability is not a bad move because of addition rounding", () => {
  assert.ok(0.005 + 0.045 < 0.05);
  const counts = summarizePlayerDecisions(
    policy(
      round(
        decision(0, [
          dahai("1m", 0.95),
          dahai("2m", 0.005),
          dahai("2m", 0.045, true, true),
        ]),
      ),
    ),
  );
  assert.deepEqual(counts[0], {
    reviewed: 1,
    matched: 0,
    badMoves: 0,
    meanLoss: null,
  });
});

test("a best tied choice matches despite binary64 addition rounding", () => {
  assert.ok(0.1 + 0.2 > 0.3);
  const counts = summarizePlayerDecisions(
    policy(
      round(
        decision(0, [
          dahai("1m", 0.1),
          dahai("1m", 0.2, false, true),
          dahai("2m", 0.3, true),
          dahai("3m", 0.2),
          dahai("4m", 0.2),
        ]),
      ),
    ),
  );
  assert.deepEqual(counts[0], {
    reviewed: 1,
    matched: 1,
    badMoves: 0,
    meanLoss: null,
  });
});

test("a genuine probability difference is not rounded into a tie", () => {
  const counts = summarizePlayerDecisions(
    policy(
      round(
        decision(0, [dahai("1m", 0.5 + 1e-10), dahai("2m", 0.5 - 1e-10, true)]),
      ),
    ),
  );
  assert.deepEqual(counts[0], {
    reviewed: 1,
    matched: 0,
    badMoves: 0,
    meanLoss: null,
  });
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
  const entries = tiles.map((tile) => dahai(tile, 0.04));
  for (let index = 0; index < 12; index++)
    entries.push(riichi(tiles[index], 0.04, index === 11));
  const counts = summarizePlayerDecisions(policy(round(decision(0, entries))));
  assert.deepEqual(counts[0], {
    reviewed: 1,
    matched: 1,
    badMoves: 0,
    meanLoss: null,
  });
});

test("forced choices are excluded after discard aggregation", () => {
  const counts = summarizePlayerDecisions(
    policy(
      round(
        decision(0, [action(ACTIONS.pass, 1, true)]),
        decision(0, [dahai("5mr", 0.6), dahai("5mr", 0.4, true, true)]),
      ),
    ),
  );
  assert.deepEqual(counts, [empty, empty, empty, empty]);
});

test("unobserved decisions and applied events do not contribute", () => {
  const counts = summarizePlayerDecisions(
    policy(
      round(
        { decision: false, actor: 0 },
        decision(2, [dahai("1m", 0.98), dahai("2m", 0.02)]),
      ),
    ),
  );
  assert.deepEqual(counts, [empty, empty, empty, empty]);
});

test("red tiles and riichi discards remain distinct choices", () => {
  const counts = summarizePlayerDecisions(
    policy(
      round(
        decision(0, [dahai("5m", 0.96), dahai("5mr", 0.04, true)]),
        decision(0, [dahai("5mr", 0.96), riichi("5mr", 0.04, true)]),
      ),
    ),
  );
  assert.deepEqual(counts[0], {
    reviewed: 2,
    matched: 0,
    badMoves: 2,
    meanLoss: null,
  });
});

test("different chii sequences and passing remain separate choices", () => {
  const counts = summarizePlayerDecisions(
    policy(
      round(
        decision(0, [
          action(ACTIONS.chi(0), 0.04, true, ["3m", "4m", "5mr"]),
          action(ACTIONS.chi(1), 0.7, false, ["4m", "5mr", "6m"]),
          action(ACTIONS.pass, 0.26),
        ]),
      ),
    ),
  );
  assert.deepEqual(counts[0], {
    reviewed: 1,
    matched: 0,
    badMoves: 1,
    meanLoss: null,
  });
});

test("pon choices retain red-tile consumption and remain separate from passing", () => {
  const counts = summarizePlayerDecisions(
    policy(
      round(
        decision(0, [
          action(ACTIONS.pon(0), 0.04, true, ["5m", "5mr", "5m"]),
          action(ACTIONS.pon(1), 0.7, false, ["5m", "5m", "5m"]),
          action(ACTIONS.pass, 0.26),
        ]),
      ),
    ),
  );
  assert.deepEqual(counts[0], {
    reviewed: 1,
    matched: 0,
    badMoves: 1,
    meanLoss: null,
  });
});

test("winning and abortive-draw choices remain separate from passing or discarding", () => {
  for (const entries of [
    [action(ACTIONS.ron, 0.04, true), action(ACTIONS.pass, 0.96)],
    [action(ACTIONS.tsumo, 0.04, true), dahai("E", 0.96)],
    [action(ACTIONS.kyushu, 0.04, true), dahai("E", 0.96)],
  ]) {
    const counts = summarizePlayerDecisions(
      policy(round(decision(0, entries))),
    );
    assert.deepEqual(counts[0], {
      reviewed: 1,
      matched: 0,
      badMoves: 1,
      meanLoss: null,
    });
  }
});

test("all seats accumulate independently across rounds", () => {
  const match = (seat) =>
    decision(seat, [action(ACTIONS.pass, 0.6, true), action(ACTIONS.ron, 0.4)]);
  const mistake = (seat) =>
    decision(seat, [
      action(ACTIONS.pass, 0.97),
      action(ACTIONS.ron, 0.03, true),
    ]);
  const counts = summarizePlayerDecisions(
    policy(
      round(match(0), mistake(1), match(3)),
      round(mistake(0), match(2), match(3)),
    ),
  );
  assert.deepEqual(counts, [
    { reviewed: 2, matched: 1, badMoves: 1, meanLoss: null },
    { reviewed: 1, matched: 0, badMoves: 1, meanLoss: null },
    { reviewed: 1, matched: 1, badMoves: 0, meanLoss: null },
    { reviewed: 2, matched: 2, badMoves: 0, meanLoss: null },
  ]);
});

test("value decisions match only the best value and grade mistakes at the threshold", () => {
  for (const [loss, matched, badMoves] of [
    [0, 1, 0],
    [1e-12, 0, 0],
    [0.002 - 1e-9, 0, 0],
    [0.002, 0, 0],
    [0.005 - 1e-9, 0, 0],
    [0.005, 0, 1],
  ]) {
    const counts = summarizePlayerDecisions(
      valued(round(decision(0, [dahai("1m", 0), dahai("2m", loss, true)]))),
    );
    assert.deepEqual(counts[0], {
      reviewed: 1,
      matched,
      badMoves,
      meanLoss: loss,
    });
  }
});

test("value decisions ignore the display probabilities", () => {
  const input = valued(
    round(decision(0, [dahai("1m", 0.001), dahai("2m", 0, true)])),
  );
  input.rounds[0].steps[0].set.probabilities = [0.99, 0.01];
  assert.deepEqual(summarizePlayerDecisions(input)[0], {
    reviewed: 1,
    matched: 1,
    badMoves: 0,
    meanLoss: 0,
  });
});

for (const type of ["DAHAI", "RIICHI_DAHAI"]) {
  test(`${type} value takes the better of drawn and held copies of the chosen tile`, () => {
    const discard = discardOf[type];
    for (const drawnChosen of [false, true]) {
      const counts = summarizePlayerDecisions(
        valued(
          round(
            decision(0, [
              discard("1m", 0.002),
              discard("2m", 0.03, !drawnChosen),
              discard("2m", 0, drawnChosen, true),
            ]),
          ),
        ),
      );
      assert.deepEqual(counts[0], {
        reviewed: 1,
        matched: 1,
        badMoves: 0,
        meanLoss: 0,
      });
    }
  });
}

test("value decisions keep red fives distinct from plain fives", () => {
  const counts = summarizePlayerDecisions(
    valued(
      round(
        decision(0, [
          dahai("5m", 0),
          dahai("5mr", 0.025, true),
          dahai("5mr", 0.03, false, true),
        ]),
      ),
    ),
  );
  assert.deepEqual(counts[0], {
    reviewed: 1,
    matched: 0,
    badMoves: 1,
    meanLoss: 0.025,
  });
});

test("forced value decisions are excluded after discard aggregation", () => {
  const counts = summarizePlayerDecisions(
    valued(
      round(
        decision(0, [dahai("5m", 0, true), dahai("5m", 0.01, false, true)]),
      ),
    ),
  );
  assert.deepEqual(counts, [empty, empty, empty, empty]);
});

test("mean loss covers only the value decisions of a valued model", () => {
  const counts = summarizePlayerDecisions(
    valued(
      round(
        decision(0, [
          action(ACTIONS.pass, 0),
          action(ACTIONS.ron, 0.004, true),
        ]),
        decision(0, [
          action(ACTIONS.pass, 0, true),
          action(ACTIONS.ron, 0.002),
        ]),
        decision(1, [
          action(ACTIONS.pass, 0),
          action(ACTIONS.ron, 0.006, true),
        ]),
      ),
    ),
  );
  assert.deepEqual(counts, [
    { reviewed: 2, matched: 1, badMoves: 0, meanLoss: 0.002 },
    { reviewed: 1, matched: 0, badMoves: 1, meanLoss: 0.006 },
    empty,
    empty,
  ]);
});
