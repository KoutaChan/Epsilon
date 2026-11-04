import test from "node:test";
import assert from "node:assert/strict";
import { displayProbabilities } from "../../../protocol/result.ts";
import {
  listedCandidates,
  readCandidates,
} from "../../src/features/replay/model/candidates.ts";
import { ACTIONS, discardAction } from "./fixtures.mjs";

test("action IDs give the type, tsumogiri and discard tile while calls keep their own tiles", () => {
  const set = {
    candidates: [
      [discardAction("5m"), 0.3],
      [discardAction("5m", { tsumogiri: true }), 0.2],
      [discardAction("5mr"), 0.1],
      [discardAction("5sr", { tsumogiri: true }), 0.05],
      [discardAction("E", { riichi: true }), 0.05],
      [discardAction("C", { riichi: true, tsumogiri: true }), 0.05],
      [ACTIONS.chi(5), 0.05, ["3p", "4p", "5pr"]],
      [ACTIONS.kakan(1), 0.05, ["9s", "9s", "9s", "9s"]],
      [ACTIONS.tsumo, 0.04],
      [ACTIONS.ron, 0.04],
      [ACTIONS.pass, 0.04],
      [ACTIONS.kyushu, 0.03],
    ],
    chosen: 2,
  };
  const candidates = readCandidates(set, undefined);
  assert.deepEqual(
    candidates.map(({ type, tiles, tsumogiri }) => [type, tiles, tsumogiri]),
    [
      ["DAHAI", ["5m"], false],
      ["DAHAI", ["5m"], true],
      ["DAHAI", ["5mr"], false],
      ["DAHAI", ["5sr"], true],
      ["RIICHI_DAHAI", ["E"], false],
      ["RIICHI_DAHAI", ["C"], true],
      ["CHI", ["3p", "4p", "5pr"], false],
      ["KAKAN", ["9s", "9s", "9s", "9s"], false],
      ["TSUMO_AGARI", [], false],
      ["RON_AGARI", [], false],
      ["PASS", [], false],
      ["KYUSHU_KYUHAI", [], false],
    ],
  );
  assert.equal(candidates[6].tiles, set.candidates[6][2]);
  assert.deepEqual(
    candidates.map((candidate) => candidate.actionId),
    set.candidates.map((row) => row[0]),
  );
  assert.deepEqual(
    candidates.map((candidate) => candidate.chosen),
    candidates.map((_, index) => index === 2),
  );
  // 価値を推定しないモデルの値はそのまま確率で、損失は無い。
  assert.deepEqual(
    candidates.map((candidate) => candidate.probability),
    set.candidates.map((row) => row[1]),
  );
  assert.ok(candidates.every((candidate) => candidate.valueLoss === null));
});

test("value losses derive the display probabilities from the document temperature", () => {
  const set = {
    candidates: [
      [discardAction("1m"), 0],
      [discardAction("2m"), 0.0012],
      [discardAction("3m"), 0.0051],
    ],
  };
  const candidates = readCandidates(set, 0.002);
  assert.deepEqual(
    candidates.map((candidate) => candidate.valueLoss),
    [0, 0.0012, 0.0051],
  );
  assert.deepEqual(
    candidates.map((candidate) => candidate.probability),
    displayProbabilities([0, 0.0012, 0.0051], 0.002),
  );
  assert.ok(candidates.every((candidate) => !candidate.chosen));
});

test("written probabilities replace the derived ones", () => {
  const set = {
    candidates: [
      [discardAction("1m"), 0],
      [discardAction("2m"), 0.01],
    ],
    probabilities: [0.6, 0.4],
    chosen: 1,
  };
  const candidates = readCandidates(set, 0.002);
  assert.deepEqual(
    candidates.map((candidate) => [candidate.probability, candidate.valueLoss]),
    [
      [0.6, 0],
      [0.4, 0.01],
    ],
  );
  assert.equal(candidates[1].chosen, true);
});

test("the list shows the top three and the actual choice below them, except while comparing a scenario", () => {
  const set = {
    candidates: ["1m", "2m", "3m", "4m", "5m"].map((tile, index) => [
      discardAction(tile),
      0.3 - index * 0.05,
    ]),
    chosen: 4,
  };
  const candidates = readCandidates(set, undefined);
  const listed = listedCandidates(candidates, false);
  assert.deepEqual(listed.top, candidates.slice(0, 3));
  assert.equal(listed.outside, candidates[4]);
  assert.deepEqual(listedCandidates(candidates, true), {
    top: candidates.slice(0, 3),
    outside: null,
  });
  // 実際の選択が上位にあれば、末尾には足さない。
  const inside = readCandidates({ ...set, chosen: 1 }, undefined);
  assert.equal(listedCandidates(inside, false).outside, null);
});

test("each decision builds its candidates once", () => {
  const set = { candidates: [[ACTIONS.pass, 1]] };
  const first = readCandidates(set, undefined);
  assert.equal(readCandidates(set, undefined), first);
  assert.notEqual(
    readCandidates({ candidates: [[ACTIONS.pass, 1]] }, undefined),
    first,
  );
});
