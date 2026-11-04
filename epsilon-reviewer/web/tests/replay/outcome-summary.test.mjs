import test from "node:test";
import assert from "node:assert/strict";
import { decodeResult } from "../../src/features/replay/model/decodeResult.ts";
import {
  collectWinningHands,
  hasRecordedMatchEnd,
  rankFinalPlayers,
  roundScoreChanges,
} from "../../src/features/replay/model/outcomeSummary.ts";
import {
  event,
  hora,
  playerSnapshot,
  resultDocument,
  round,
  snapshot,
} from "./fixtures.mjs";

const concealed = Object.freeze([
  "1m",
  "2m",
  "3m",
  "4p",
  "5p",
  "6p",
  "7s",
  "8s",
  "9s",
  "E",
  "E",
  "F",
  "F",
]);
function makePlayer(seat, fields = {}) {
  return playerSnapshot(seat, { hand: concealed, ...fields });
}
function makeSnapshot(eventType, actor = -1, overrides = {}, fields = {}) {
  return snapshot(eventType, actor, {
    players: Object.freeze(
      [0, 1, 2, 3].map((seat) => makePlayer(seat, overrides[seat])),
    ),
    ...fields,
  });
}
function makeWin(actor, target, winningTile, overrides = {}, details) {
  return makeSnapshot("hora", actor, overrides, {
    outcome: Object.freeze(hora(actor, target, winningTile, details)),
  });
}

test("recorded ron identifies its source and winning tile without consulting prior events", () => {
  const win = makeWin(0, 1, "8p");
  const [winner] = collectWinningHands([win]);
  assert.equal(winner.player, win.players[0]);
  assert.equal(winner.method, "ron");
  assert.equal(winner.target, 1);
  assert.equal(winner.winningTile, "8p");
  assert.equal(winner.hand, concealed);
});

test("recorded tsumo uses the separated winning tile and borrows the thirteen-tile hand", () => {
  const win = makeWin(2, 2, "5sr", { 2: { draw: "5sr" } });
  const [winner] = collectWinningHands([win]);
  assert.equal(winner.method, "tsumo");
  assert.equal(winner.target, 2);
  assert.equal(winner.winningTile, "5sr");
  assert.equal(winner.hand, concealed);
});

test("a matching tile already in the hand is preserved beside the recorded red winning tile", () => {
  const hand = Object.freeze([...concealed.slice(0, 12), "5p"]);
  const win = makeWin(0, 2, "5pr", { 0: { hand } });
  const [winner] = collectWinningHands([win]);
  assert.equal(winner.target, 2);
  assert.equal(winner.winningTile, "5pr");
  assert.equal(winner.hand, hand);
  assert.equal(winner.hand.length, 13);
  assert.equal(winner.hand.filter((tile) => tile === "5p").length, 2);
});

test("double ron retains each winner and the recorded shared source tile", () => {
  const firstWin = makeWin(0, 3, "C");
  const secondWin = makeWin(2, 3, "C");
  const winners = collectWinningHands(Object.freeze([firstWin, secondWin]));
  assert.deepEqual(
    winners.map((winner) => [
      winner.player.seat,
      winner.method,
      winner.target,
      winner.winningTile,
    ]),
    [
      [0, "ron", 3, "C"],
      [2, "ron", 3, "C"],
    ],
  );
  assert.equal(winners[0].player, firstWin.players[0]);
  assert.equal(winners[1].player, secondWin.players[2]);
});

test("recorded chankan preserves the red winning tile and the winner's exposed melds", () => {
  const melds = Object.freeze([
    Object.freeze({
      type: "PON",
      tiles: Object.freeze(["E", "E", "E"]),
      from: 3,
      calledIndex: 0,
      addedIndex: -1,
    }),
  ]);
  const hand = Object.freeze(concealed.slice(0, 10));
  const win = makeWin(0, 1, "5pr", { 0: { hand, melds } });
  const [winner] = collectWinningHands([win]);
  assert.equal(winner.method, "ron");
  assert.equal(winner.target, 1);
  assert.equal(winner.winningTile, "5pr");
  assert.equal(winner.hand, hand);
  assert.equal(winner.player.melds, melds);
});

test("unrecorded scoring details remain unknown", () => {
  const [winner] = collectWinningHands([makeWin(0, 1, "E")]);
  assert.equal(winner.details, null);
});

test("recorded scoring and ura indicators are borrowed without modification", () => {
  const details = Object.freeze({
    han: 3,
    fu: 40,
    points: 5200,
    yaku: Object.freeze([
      { code: "RIICHI", han: 1, yakuman: 0 },
      { code: "URADORA", han: 2, yakuman: 0 },
    ]),
    uraIndicators: Object.freeze(["9m"]),
  });
  const [winner] = collectWinningHands([makeWin(0, 1, "E", {}, details)]);
  assert.equal(winner.details, details);
  assert.equal(winner.details.uraIndicators, details.uraIndicators);
});

test("known empty ura indicators remain distinct from unrecorded indicators", () => {
  const winners = collectWinningHands([
    makeWin(0, 1, "E", {}, Object.freeze({ uraIndicators: Object.freeze([]) })),
    makeWin(2, 1, "E", {}, Object.freeze({})),
  ]);
  assert.deepEqual(winners[0].details.uraIndicators, []);
  assert.equal(winners[1].details.uraIndicators, undefined);
});

test("drawn rounds have no fabricated winning hands", () => {
  assert.deepEqual(collectWinningHands([makeSnapshot("ryukyoku")]), []);
});

test("round score change includes riichi deposits and final settlement rather than only the winning payment", () => {
  const start = makeSnapshot("start_kyoku");
  const riichi = makeSnapshot("reach_accepted", 0, {
    0: { score: 24000, riichi: true },
  });
  const win = makeSnapshot(
    "hora",
    1,
    { 0: { score: 20100 }, 1: { score: 29900 } },
    { outcome: Object.freeze(hora(1, 0, "E")) },
  );
  assert.deepEqual(roundScoreChanges(start, win), [-4900, 4900, 0, 0]);
  assert.equal(start.players[0].score, 25000);
  assert.equal(riichi.players[0].score, 24000);
});

test("final score overrides determine ranking and tied scores retain a shared rank without mutating players", () => {
  const players = Object.freeze([
    makePlayer(0, { score: 34000 }),
    makePlayer(1, { score: 26000 }),
    makePlayer(2, { score: 23000 }),
    makePlayer(3, { score: 17000 }),
  ]);
  const finalScores = Object.freeze([24000, 38000, 38000, 0]);
  const ranking = rankFinalPlayers(players, finalScores);
  assert.deepEqual(
    ranking.map(({ player, score, rank }) => [player.seat, score, rank]),
    [
      [1, 38000, 1],
      [2, 38000, 1],
      [0, 24000, 3],
      [3, 0, 4],
    ],
  );
  assert.deepEqual(
    players.map((player) => player.seat),
    [0, 1, 2, 3],
  );
  assert.deepEqual(
    players.map((player) => player.score),
    [34000, 26000, 23000, 17000],
  );
  assert.equal(ranking[0].player, players[1]);
  assert.deepEqual(finalScores, [24000, 38000, 38000, 0]);
});

test("final ranking uses settled scores when the imported record has no final score override", () => {
  const players = Object.freeze([
    makePlayer(0, { score: -1200 }),
    makePlayer(1, { score: 37100 }),
    makePlayer(2, { score: 43000 }),
    makePlayer(3, { score: 21100 }),
  ]);
  assert.deepEqual(
    rankFinalPlayers(players, undefined).map(({ player, score, rank }) => [
      player.seat,
      score,
      rank,
    ]),
    [
      [2, 43000, 1],
      [1, 37100, 2],
      [3, 21100, 3],
      [0, -1200, 4],
    ],
  );
});

function recordedMatch(kinds, finalScores) {
  const document = resultDocument([
    round(
      kinds.map((kind) =>
        kind === "hora"
          ? event(kind, { actor: 0, outcome: hora(0, 0) })
          : kind === "ryukyoku"
            ? event(kind, { outcome: { type: "ryukyoku" } })
            : event(kind),
      ),
    ),
  ]);
  if (finalScores) document.record.finalScores = finalScores;
  return decodeResult(document);
}

test("a recorded final score establishes the match ending even without an end_game step", () => {
  assert.equal(
    hasRecordedMatchEnd(
      recordedMatch(
        ["start_kyoku", "hora", "end_kyoku"],
        [35000, 28000, 22000, 15000],
      ),
    ),
    true,
  );
});

test("an explicit end_game establishes the match ending when the record has no final scores", () => {
  assert.equal(
    hasRecordedMatchEnd(
      recordedMatch(["start_kyoku", "hora", "end_kyoku", "end_game"]),
    ),
    true,
  );
});

test("a final stored round settlement alone does not establish that the match ended", () => {
  assert.equal(
    hasRecordedMatchEnd(
      recordedMatch(["start_kyoku", "ryukyoku", "end_kyoku"]),
    ),
    false,
  );
});
