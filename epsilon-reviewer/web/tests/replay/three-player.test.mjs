import test from "node:test";
import assert from "node:assert/strict";
import { actionOf, tileCopies } from "../../../protocol/result.ts";
import { decodeResult } from "../../src/features/replay/model/decodeResult.ts";
import { ReviewFormatError } from "../../src/features/replay/model/formatError.ts";
import { displaySeats } from "../../src/features/replay/model/seats.ts";
import { createStateReader } from "../../src/features/replay/model/stateReader.ts";
import { local, tableDimensions } from "../../src/features/replay/renderer/projection.ts";
import {
  layoutPlayerRow,
  measurePlayerRowWidth,
  nukiTilePosition,
} from "../../src/features/replay/renderer/tileLayout.ts";
import { event, player, resultDocument, round, table } from "./fixtures.mjs";

// 三人打ちの解析結果。席は 3 つで、北抜き（kita）の手順がある。抜いた北は副露に入れず、席ごとの数（nuki）で持ち、行動 372 が北抜き。

const PON = { type: "PON", tiles: ["P", "P", "P"], from: 2, calledIndex: 0, addedIndex: -1 };
const OLD_NUKI_MELD = { type: "NUKI", tiles: ["N"], from: -1, calledIndex: -1, addedIndex: -1 };

/** 席 1 がポンしてから北を 2 回抜き、2 回目の北に席 2 がロンできる局。 */
function threePlayer(fields = {}) {
  return resultDocument(
    [
      round(
        [
          event("start_kyoku"),
          event("pon", { actor: 1, delta: { players: [{ seat: 1, melds: [0, 0, PON] }] } }),
          event("kita", { actor: 1, delta: { players: [{ seat: 1, nuki: 1 }] } }),
          event("kita", { actor: 1, delta: { players: [{ seat: 1, nuki: 2 }] } }),
          { seat: 2, candidates: [[369, 0.5], [370, 0.5]] },
          event("hora", {
            actor: 2,
            outcome: { type: "hora", actor: 2, target: 1, winningTile: "N" },
          }),
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

test("a three-player document decodes with three seats and keeps the norths apart from the melds", () => {
  const model = decodeResult(threePlayer());
  const [first] = model.rounds;
  assert.equal(first.seats, 3);
  assert.equal(first.steps[2].eventType, "kita");
  assert.deepEqual(first.deltas[1].players[0], { seat: 1, nuki: 1 });
  const reader = createStateReader(model);
  const afterPon = reader.readSnapshot(0, first.steps[1]).players[1];
  assert.deepEqual([afterPon.melds, afterPon.nuki], [[PON], 0]);
  const ron = reader.readSnapshot(0, first.steps[5]);
  assert.deepEqual([ron.players[1].melds, ron.players[1].nuki], [[PON], 2]);
  assert.equal(ron.players[0].nuki, 0);
  assert.equal(ron.outcome.winningTile, "N");
});

test("a three-player document rejects a fourth seat, a north written as a meld and impossible counts", () => {
  const document = threePlayer();
  document.rounds[0].steps[4].seat = 3;
  assert.throws(() => decodeResult(document), ReviewFormatError);
  const fourTables = threePlayer();
  fourTables.rounds[0].start = table();
  assert.throws(() => decodeResult(fourTables), ReviewFormatError);
  const meld = threePlayer();
  meld.rounds[0].steps[2].delta = { players: [{ seat: 1, melds: [1, 0, OLD_NUKI_MELD] }] };
  assert.throws(() => decodeResult(meld), /Unsupported meld/);
  for (const nuki of [5, -1, 1.5, "1"]) {
    const count = threePlayer();
    count.rounds[0].steps[2].delta.players[0].nuki = nuki;
    assert.throws(() => decodeResult(count), ReviewFormatError, String(nuki));
  }
  const missing = threePlayer();
  delete missing.rounds[0].start.players[0].nuki;
  assert.throws(() => decodeResult(missing), ReviewFormatError);
  assert.throws(() => decodeResult(threePlayer({ format: 6 })), /Unsupported result format/);
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

test("only real melds shorten the hand: a pon moves the centred hand by one and a half tiles", () => {
  const open = layoutPlayerRow([PON]);
  assert.equal(open.melds.length, 3);
  assert.ok(Math.abs(open.handLeft - layoutPlayerRow([]).handLeft - 1.5 * 1.012) < 1e-9);
  assert.ok(measurePlayerRowWidth([PON]) > measurePlayerRowWidth([]));
});

test("set-aside norths lie in the corner between the river and the next seat's river, apart from both", () => {
  for (const layout of ["portrait", "desktop", "landscape"]) {
    const d = tableDimensions(layout);
    const half = d.riverWidth / 2;
    const depth = (d.riverWidth * 2) / 3;
    const riverRight = d.riverStart + 5 * d.pitchX + half;
    // 下家（east）の河は、卓の座標で y が河の始まりの牌の手前の縁より奥にある。
    const nextRiverNear = local("east", d.riverStart - half, 0).y;
    const positions = [0, 1, 2, 3].map((index) => nukiTilePosition(index, d));
    assert.ok(positions[0].x - half > riverRight + d.riverWidth / 3, layout);
    for (const [index, { x, y }] of positions.entries()) {
      assert.equal(y, -d.riverRadius - 2 * d.pitchY, layout);
      if (index > 0) assert.ok(Math.abs(x - positions[index - 1].x - d.pitchX) < 1e-9, layout);
      assert.ok(y + depth < nextRiverNear - d.riverWidth, layout);
      assert.ok(x + half < d.opponentRadius - depth, layout);
    }
  }
});
