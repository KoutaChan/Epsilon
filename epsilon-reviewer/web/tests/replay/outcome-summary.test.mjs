import { meld } from "./fixtures.mjs";
import test from "node:test";
import assert from "node:assert/strict";
import {
  collectWinningHands,
  rankFinalPlayers,
  roundScoreChanges,
} from "../../src/features/replay/model/outcomeSummary.ts";
import { hora, playerSnapshot, snapshot } from "./fixtures.mjs";

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

test("a ron on a set-aside north keeps the winner's norths apart from its melds", () => {
  const pon = meld("PON", ["P", "P", "P"]);
  const win = makeWin(0, 1, "N", {
    0: { hand: concealed.slice(0, 10), melds: [pon], nuki: 2 },
    1: { nuki: 1 },
  });
  const [winner] = collectWinningHands([win]);
  assert.equal(winner.method, "ron");
  assert.equal(winner.winningTile, "N");
  assert.deepEqual(winner.player.melds, [pon]);
  assert.equal(winner.player.nuki, 2);
  assert.equal(win.players[winner.target].nuki, 1);
});

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

// 天鳳の順位は重ならない。同点は起家に近い席（席の番号の小さい方）が上。
test("final score overrides determine ranking and tied scores rank in seat order without mutating players", () => {
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
      [2, 38000, 2],
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
