import test from "node:test";
import assert from "node:assert/strict";
import { decodeResult } from "../../src/features/replay/model/decodeResult.ts";
import { ReviewFormatError } from "../../src/features/replay/model/formatError.ts";
import {
  ACTIONS,
  decision,
  discard,
  discardAction,
  event,
  hora,
  player,
  resultDocument,
  round,
  table,
} from "./fixtures.mjs";

function sample() {
  const start = table({
    players: [0, 1, 2, 3].map((seat) =>
      player(seat === 0 ? { hand: ["1m", "2m", "3m"] } : {}),
    ),
  });
  return resultDocument(
    [
      round(
        [
          event("start_kyoku"),
          event("tsumo", {
            actor: 0,
            delta: { players: [{ seat: 0, draw: "5pr" }], remaining: 59 },
          }),
          decision(
            0,
            [
              [discardAction("5pr", { tsumogiri: true }), 0.75],
              [discardAction("1m"), 0.25],
            ],
            { chosen: 0 },
          ),
          event("dahai", {
            actor: 0,
            delta: {
              players: [
                {
                  seat: 0,
                  draw: null,
                  river: [0, 0, discard("5pr", { tsumogiri: true })],
                },
              ],
            },
          }),
          decision(1, [
            [ACTIONS.chi(3), 0.4, ["3p", "4p", "5pr"]],
            [ACTIONS.pass, 0.6],
          ]),
          decision(2, [[ACTIONS.ron, 1]], { cause: 2, resolve: 4 }),
          event("none", { actor: 1 }),
          event("hora", { actor: 2, outcome: hora(2, 0, "5pr") }),
        ],
        { roundIndex: 4, honba: 1, dealer: 3, firstEvent: 1, start },
      ),
    ],
    {
      record: {
        source: "tenhou",
        names: ["A", "B", "", "D"],
        finalScores: [1, 2, 3, 4],
        roundCount: 1,
      },
    },
  );
}

const invalid = (error) =>
  error instanceof ReviewFormatError && error.code === "invalid_result";

test("a document decodes in one pass and borrows its arrays and objects", () => {
  const document = sample();
  const model = decodeResult(document);
  assert.equal(model.resultId, document.resultId);
  assert.equal(model.createdAt, document.createdAt);
  assert.equal(model.model, document.model);
  assert.equal(model.record, document.record);
  assert.equal(model.temperature, undefined);
  const [decoded] = model.rounds;
  const source = document.rounds[0];
  assert.deepEqual(
    [decoded.index, decoded.roundIndex, decoded.honba, decoded.dealer],
    [0, 4, 1, 3],
  );
  assert.equal(decoded.start, source.start);
  assert.deepEqual(decoded.deltas, [
    source.steps[1].delta,
    source.steps[3].delta,
  ]);
  assert.equal(decoded.deltas[0], source.steps[1].delta);
  assert.deepEqual(
    decoded.steps.map((step) =>
      step.decision
        ? [
            "decision",
            step.index,
            step.actor,
            step.causeEventIndex,
            step.eventIndex,
            step.frame,
          ]
        : [
            "event",
            step.index,
            step.eventType,
            step.actor,
            step.eventIndex,
            step.frame,
          ],
    ),
    [
      ["event", 0, "start_kyoku", -1, 1, 0],
      ["event", 1, "tsumo", 0, 2, 1],
      ["decision", 2, 0, 2, 3, 1],
      ["event", 3, "dahai", 0, 3, 2],
      ["decision", 4, 1, 3, 4, 2],
      ["decision", 5, 2, 2, 4, 2],
      ["event", 6, "none", 1, 4, 2],
      ["event", 7, "hora", 2, 5, 2],
    ],
  );
  assert.equal(decoded.steps[2].set, source.steps[2]);
  assert.equal(decoded.steps[7].outcome, source.steps[7].outcome);
  assert.equal(decoded.steps[3].outcome, null);
});

test("a valued document keeps its temperature and accepts value losses above one", () => {
  const document = sample();
  document.temperature = 0.002;
  document.rounds[0].steps[2].candidates = [
    [discardAction("1m"), 0],
    [discardAction("2m"), 1.5],
  ];
  document.rounds[0].steps[4].candidates = [[ACTIONS.pass, 0]];
  document.rounds[0].steps[5].candidates = [[ACTIONS.ron, 0]];
  assert.equal(decodeResult(document).temperature, 0.002);
});

for (const [name, mutate] of [
  ["an unsupported format", (d) => (d.format = 5)],
  ["a missing format", (d) => delete d.format],
  ["three names", (d) => d.record.names.pop()],
  ["no rounds", (d) => (d.rounds = [])],
  ["a round without steps", (d) => (d.rounds[0].steps = [])],
  [
    "a round that starts with a decision",
    (d) => d.rounds[0].steps.splice(0, 2),
  ],
  [
    "a step that is an event and a decision",
    (d) => (d.rounds[0].steps[1].seat = 0),
  ],
  ["a step that is neither", (d) => delete d.rounds[0].steps[1].event],
  ["an unknown event", (d) => (d.rounds[0].steps[1].event = "start_game")],
  [
    "an invalid hand tile",
    (d) => (d.rounds[0].start.players[0].hand[0] = "0m"),
  ],
  [
    "a red tile that is not a five",
    (d) => (d.rounds[0].start.players[0].hand[0] = "1mr"),
  ],
  [
    "an invalid draw",
    (d) => (d.rounds[0].steps[1].delta.players[0].draw = "back"),
  ],
  ["an invalid dora", (d) => (d.rounds[0].start.dora = ["X"])],
  ["three players", (d) => d.rounds[0].start.players.pop()],
  [
    "a river mark written as false",
    (d) => (d.rounds[0].steps[3].delta.players[0].river[2].riichi = false),
  ],
  [
    "a splice past the end",
    (d) =>
      (d.rounds[0].steps[3].delta.players[0].river = [1, 0, discard("1m")]),
  ],
  [
    "a splice removing past the end",
    (d) => (d.rounds[0].steps[1].delta.players[0].hand = [2, 2]),
  ],
  [
    "a negative splice",
    (d) => (d.rounds[0].steps[1].delta.players[0].hand = [-1, 0]),
  ],
  ["a splice without a count", (d) => (d.rounds[0].steps[1].delta.dora = [0])],
  [
    "an invalid spliced tile",
    (d) => (d.rounds[0].steps[1].delta.dora = [1, 0, "9z"]),
  ],
  [
    "player changes out of seat order",
    (d) => (d.rounds[0].steps[1].delta.players = [{ seat: 2 }, { seat: 1 }]),
  ],
  [
    "a repeated player change",
    (d) => (d.rounds[0].steps[1].delta.players = [{ seat: 1 }, { seat: 1 }]),
  ],
  ["a cause before the round", (d) => (d.rounds[0].steps[2].cause = 0)],
  ["a cause after the decision", (d) => (d.rounds[0].steps[2].cause = 3)],
  ["no candidates", (d) => (d.rounds[0].steps[2].candidates = [])],
  ["a chosen candidate out of range", (d) => (d.rounds[0].steps[2].chosen = 2)],
  [
    "probabilities of another length",
    (d) => (d.rounds[0].steps[2].probabilities = [1]),
  ],
  [
    "an action ID out of range",
    (d) => (d.rounds[0].steps[2].candidates[0][0] = 372),
  ],
  ["a call without tiles", (d) => d.rounds[0].steps[4].candidates[0].pop()],
  [
    "a discard with tiles",
    (d) => d.rounds[0].steps[2].candidates[0].push(["1m"]),
  ],
  [
    "an invalid call tile",
    (d) => (d.rounds[0].steps[4].candidates[0][2] = ["3p", "4p", "5p5"]),
  ],
  [
    "a probability above one",
    (d) => (d.rounds[0].steps[2].candidates[0][1] = 1.5),
  ],
  [
    "a value that is not a number",
    (d) => (d.rounds[0].steps[2].candidates[0][1] = "0.5"),
  ],
  [
    "an outcome on a discard",
    (d) => (d.rounds[0].steps[3].outcome = { type: "ryukyoku" }),
  ],
  ["a win without its outcome", (d) => delete d.rounds[0].steps[7].outcome],
  [
    "a win with a draw outcome",
    (d) => (d.rounds[0].steps[7].outcome = { type: "ryukyoku" }),
  ],
  [
    "an invalid winning tile",
    (d) => (d.rounds[0].steps[7].outcome.winningTile = "?"),
  ],
  [
    "invalid ura indicators",
    (d) => (d.rounds[0].steps[7].outcome.details = { uraIndicators: ["5z"] }),
  ],
  [
    "a meld with an invalid called tile",
    (d) =>
      (d.rounds[0].start.players[1].melds = [
        {
          type: "PON",
          tiles: ["E", "E", "E"],
          from: 0,
          calledIndex: 3,
          addedIndex: -1,
        },
      ]),
  ],
  [
    "a closed kan with a source",
    (d) =>
      (d.rounds[0].start.players[1].melds = [
        {
          type: "ANKAN",
          tiles: ["E", "E", "E", "E"],
          from: 0,
          calledIndex: -1,
          addedIndex: -1,
        },
      ]),
  ],
  ["a non-positive temperature", (d) => (d.temperature = 0)],
  ["a null final score list", (d) => (d.record.finalScores = null)],
])
  test(`decoding rejects ${name}`, () => {
    const document = sample();
    mutate(document);
    assert.throws(() => decodeResult(document), invalid);
  });

test("a value loss must not be negative in a valued document", () => {
  const document = sample();
  document.temperature = 0.002;
  document.rounds[0].steps[2].candidates[0][1] = -0.001;
  assert.throws(() => decodeResult(document), invalid);
});

test("failures inside a round name the round and the step", () => {
  const document = sample();
  document.rounds[0].steps[4].candidates = [];
  assert.throws(
    () => decodeResult(document),
    (error) =>
      invalid(error) &&
      error.message === "rounds[0].steps[4]: A decision has no candidates.",
  );
});
