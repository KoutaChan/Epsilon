import test from "node:test";
import assert from "node:assert/strict";
import {
  collectWinningHands,
  hasRecordedMatchEnd,
  roundScoreChanges,
  rankFinalPlayers,
} from "../src/features/replay/outcomeSummary.ts";
import {
  eventSnapshot,
  playerSnapshot,
  reviewRound as makeRound,
  reviewResult,
} from "./reviewFixtures.mjs";

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
let eventIndex = 0;
function makePlayer(seat, fields = {}) {
  return playerSnapshot(seat, { hand: concealed, ...fields });
}
function makeSnapshot(eventType, actor = -1, overrides = {}, fields = {}) {
  return eventSnapshot(eventIndex++, eventType, actor, {
    players: Object.freeze(
      [0, 1, 2, 3].map((seat) => makePlayer(seat, overrides[seat])),
    ),
    ...fields,
  });
}
function makeWin(actor, target, winningTile, overrides = {}, details = null) {
  return makeSnapshot("hora", actor, overrides, {
    outcome: Object.freeze({
      type: "hora",
      actor,
      target,
      winningTile,
      deltas: null,
      details,
    }),
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
    yakuman: 0,
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
  const absent = Object.freeze({
    han: null,
    fu: null,
    points: null,
    yakuman: null,
    yaku: null,
    uraIndicators: Object.freeze([]),
  });
  const unknown = Object.freeze({ ...absent, uraIndicators: null });
  const winners = collectWinningHands([
    makeWin(0, 1, "E", {}, absent),
    makeWin(2, 1, "E", {}, unknown),
  ]);
  assert.deepEqual(winners[0].details.uraIndicators, []);
  assert.equal(winners[1].details.uraIndicators, null);
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
    {
      outcome: Object.freeze({
        type: "hora",
        actor: 1,
        target: 0,
        winningTile: "E",
        details: null,
        deltas: Object.freeze([-3900, 4900, 0, 0]),
      }),
    },
  );
  const round = makeRound([start, riichi, win]);
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
    rankFinalPlayers(players, null).map(({ player, score, rank }) => [
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

test("a recorded final score establishes the match ending even without an end_game snapshot", () => {
  const result = reviewResult([], {
    metadata: Object.freeze({
      finalScores: Object.freeze([35000, 28000, 22000, 15000]),
    }),
    rounds: Object.freeze([
      makeRound([
        makeSnapshot("start_kyoku"),
        makeSnapshot("hora", 0),
        makeSnapshot("end_kyoku"),
      ]),
    ]),
  });
  assert.equal(hasRecordedMatchEnd(result), true);
});

test("an explicit end_game establishes the match ending when the metadata has no final scores", () => {
  const result = reviewResult([], {
    metadata: Object.freeze({ finalScores: null }),
    rounds: Object.freeze([
      makeRound([
        makeSnapshot("start_kyoku"),
        makeSnapshot("hora", 0),
        makeSnapshot("end_kyoku"),
        makeSnapshot("end_game"),
      ]),
    ]),
  });
  assert.equal(hasRecordedMatchEnd(result), true);
});

test("a final stored round settlement alone does not establish that the match ended", () => {
  const result = reviewResult([], {
    metadata: Object.freeze({ finalScores: null }),
    rounds: Object.freeze([
      makeRound([
        makeSnapshot("start_kyoku"),
        makeSnapshot("ryukyoku"),
        makeSnapshot("end_kyoku"),
      ]),
    ]),
  });
  assert.equal(hasRecordedMatchEnd(result), false);
});
