import test from "node:test";
import assert from "node:assert/strict";
import { actionOf, tileCopies } from "../../../protocol/result.ts";
import { decodeResult } from "../../src/features/replay/model/decodeResult.ts";
import { ReviewFormatError } from "../../src/features/replay/model/formatError.ts";
import { displaySeats } from "../../src/features/replay/model/seats.ts";
import {
  layoutPlayerRow,
  measurePlayerRowWidth,
} from "../../src/features/replay/renderer/tileLayout.ts";
import { event, player, resultDocument, round, table } from "./fixtures.mjs";

// 三人打ちの解析結果。席は 3 つで、北抜き（kita）の手順がある。抜いた北は副露の後ろに NUKI として並び、行動 372 が北抜き。

const NUKI = { type: "NUKI", tiles: ["N"], from: -1, calledIndex: -1, addedIndex: -1 };
const PON = { type: "PON", tiles: ["P", "P", "P"], from: 2, calledIndex: 0, addedIndex: -1 };

function threePlayer(fields = {}) {
  return resultDocument(
    [
      round(
        [
          event("start_kyoku"),
          event("kita", {
            actor: 1,
            delta: { players: [{ seat: 1, melds: [0, 0, NUKI] }] },
          }),
          { seat: 2, candidates: [[369, 0.5], [370, 0.5]] },
          event("none", { actor: 2 }),
        ],
        { roundIndex: 6, dealer: 0, start: table({ players: [0, 1, 2].map(() => player()) }) },
      ),
    ],
    {
      record: { source: "tenhou", names: ["A", "B", "C"], finalScores: [35000, 35000, 35000], roundCount: 1 },
      ...fields,
    },
  );
}

test("a three-player document decodes with three seats, kita steps and extracted norths", () => {
  const decoded = decodeResult(threePlayer());
  const [first] = decoded.rounds;
  assert.equal(first.seats, 3);
  assert.equal(first.steps[1].eventType, "kita");
  assert.deepEqual(first.deltas[0].players[0].melds, [0, 0, NUKI]);
});

test("a three-player document rejects a fourth seat", () => {
  const document = threePlayer();
  document.rounds[0].steps[2].seat = 3;
  assert.throws(() => decodeResult(document), ReviewFormatError);
  const fourTables = threePlayer();
  fourTables.rounds[0].start = table();
  assert.throws(() => decodeResult(fourTables), ReviewFormatError);
});

test("the north extraction is action 372", () => {
  assert.deepEqual(actionOf(372), { type: "NUKI", tsumogiri: false, tile: null });
  assert.throws(() => actionOf(373), RangeError);
});

test("three players leave the opposite seat empty and have no 2m to 8m", () => {
  assert.deepEqual(displaySeats(4), [0, 1, 2, 3]);
  assert.deepEqual(displaySeats(3), [0, 1, null, 2]);
  assert.equal(tileCopies("5m", 3), 0);
  assert.equal(tileCopies("5mr", 3), 0);
  assert.equal(tileCopies("9m", 3), 4);
  assert.equal(tileCopies("5pr", 3), 1);
  assert.equal(tileCopies("5m"), 3);
});

test("extracted norths sit beside the melds without shortening the hand", () => {
  const withNorth = layoutPlayerRow([PON, NUKI]);
  assert.equal(withNorth.melds.length, 4);
  assert.equal(withNorth.melds[3].meld.type, "NUKI");
  assert.equal(withNorth.melds[3].sideways, false);
  assert.ok(withNorth.melds[3].x < withNorth.melds[0].x);
  assert.ok(measurePlayerRowWidth([PON, NUKI]) > measurePlayerRowWidth([PON]));
  assert.equal(
    layoutPlayerRow([PON, NUKI]).handLeft <= layoutPlayerRow([PON]).handLeft,
    true,
  );
});
