import test from "node:test";
import assert from "node:assert/strict";
import {
  NO_EDITS,
  STICK_LIMIT,
  applyEdits,
  canEditHand,
  canEditRiver,
  compareCandidates,
  handOrigin,
  isCall,
  listChanges,
  revert,
  setDora,
  setHand,
  setRiver,
  setScore,
  setSticks,
  tableSnapshot,
  tileChoices,
} from "../../src/features/replay/scenario/scenario.ts";
import { discard, playerSnapshot } from "./fixtures.mjs";

const HAND = [
  "5m",
  "5m",
  "5m",
  "3p",
  "4p",
  "5pr",
  "7s",
  "7s",
  "8s",
  "9s",
  "W",
  "W",
  "P",
];
const VIEWER = 2;
const DECISION = { decision: true, actor: VIEWER };

// 見ている席は 2。上家（1）の 2s は対面（0）に鳴かれている。
function position() {
  return {
    index: 10,
    decision: true,
    eventIndex: 57,
    causeEventIndex: 56,
    actor: VIEWER,
    frame: 3,
    set: { candidates: [[370, 1]] },
    players: [
      playerSnapshot(0, {
        river: [discard("1m"), discard("9p", { tsumogiri: true })],
        melds: [
          {
            type: "CHI",
            tiles: ["1s", "2s", "3s"],
            from: 1,
            calledIndex: 1,
            addedIndex: -1,
          },
        ],
      }),
      playerSnapshot(1, {
        river: [discard("E"), discard("2s", { called: true })],
      }),
      playerSnapshot(2, { river: [discard("N")], hand: HAND, draw: "6p" }),
      playerSnapshot(3, {
        river: [
          discard("6s", { riichi: true }),
          discard("C", { tsumogiri: true }),
        ],
        riichi: true,
        score: 24000,
      }),
    ],
    dora: ["4m"],
    honba: 1,
    kyotaku: 1,
    remaining: 50,
    activeSeat: VIEWER,
  };
}
const find = (choices, tile) =>
  choices.flat().find((choice) => choice.tile === tile);

test("edits apply to a copy and leave the shared snapshot untouched", () => {
  const original = position();
  const frozen = structuredClone(original);
  const unchanged = applyEdits(original, NO_EDITS, VIEWER);
  assert.equal(unchanged.snapshot, original);
  assert.deepEqual(
    unchanged.handOrigins,
    HAND.map((_, index) => index),
  );
  let edits = setRiver(NO_EDITS, original, 3, 0, {
    tile: "5sr",
    tsumogiri: true,
  });
  edits = setDora(edits, original, 0, "N");
  edits = setScore(edits, original, 0, 31000);
  edits = setSticks(edits, original, "honba", 0);
  edits = setSticks(edits, original, "kyotaku", 2);
  const edited = applyEdits(original, edits, VIEWER).snapshot;
  assert.deepEqual(original, frozen);
  assert.deepEqual(edited.players[3].river[0], {
    tile: "5sr",
    tsumogiri: true,
    riichi: true,
  });
  assert.equal(edited.players[3].river[1], original.players[3].river[1]);
  assert.equal(edited.players[1], original.players[1]);
  assert.equal(edited.players[0].river, original.players[0].river);
  assert.deepEqual(edited.dora, ["N"]);
  assert.deepEqual(
    edited.players.map((entry) => entry.score),
    [31000, 25000, 25000, 24000],
  );
  assert.equal(edited.honba, 0);
  assert.equal(edited.kyotaku, 2);
  assert.equal(edited.eventIndex, 57);
  const tedashi = applyEdits(
    original,
    setRiver(NO_EDITS, original, 0, 1, { tsumogiri: false }),
    VIEWER,
  ).snapshot;
  assert.deepEqual(tedashi.players[0].river[1], { tile: "9p" });
});

test("edits keep one canonical form and vanish when restored", () => {
  const original = position();
  let edits = setRiver(NO_EDITS, original, 3, 1, { tile: "F" });
  edits = setRiver(edits, original, 0, 0, { tsumogiri: true });
  assert.deepEqual(edits, {
    rivers: [
      { seat: 0, index: 0, tile: "1m", tsumogiri: true },
      { seat: 3, index: 1, tile: "F", tsumogiri: true },
    ],
  });
  // 同じ局面は同じ JSON になり、解析の結果の取り違えを防ぐ。
  const reordered = setRiver(
    setRiver(NO_EDITS, original, 0, 0, { tsumogiri: true }),
    original,
    3,
    1,
    { tile: "F" },
  );
  assert.equal(JSON.stringify(reordered), JSON.stringify(edits));
  edits = setRiver(edits, original, 3, 1, { tile: "C" });
  edits = revert(edits, original, { kind: "river", seat: 0, index: 0 });
  assert.equal(edits, NO_EDITS);
  assert.equal(setScore(NO_EDITS, original, 3, 24000), NO_EDITS);
  assert.equal(setSticks(NO_EDITS, original, "honba", 1), NO_EDITS);
  assert.equal(setSticks(NO_EDITS, original, "kyotaku", -5).kyotaku, 0);
  assert.equal(
    setSticks(NO_EDITS, original, "honba", STICK_LIMIT + 9).honba,
    STICK_LIMIT,
  );
});

test("hand edits keep the original positions in index order and drop unchanged tiles", () => {
  const original = position();
  let edits = setHand(NO_EDITS, original, VIEWER, 13, "7p");
  edits = setHand(edits, original, VIEWER, 12, "1m");
  edits = setHand(edits, original, VIEWER, 0, "E");
  assert.deepEqual(edits.hand, [
    { index: 0, tile: "E" },
    { index: 12, tile: "1m" },
    { index: 13, tile: "7p" },
  ]);
  const reordered = setHand(
    setHand(
      setHand(NO_EDITS, original, VIEWER, 0, "E"),
      original,
      VIEWER,
      13,
      "7p",
    ),
    original,
    VIEWER,
    12,
    "1m",
  );
  assert.equal(JSON.stringify(reordered), JSON.stringify(edits));
  edits = setHand(edits, original, VIEWER, 0, "5m");
  assert.deepEqual(
    edits.hand.map((edit) => edit.index),
    [12, 13],
  );
  edits = revert(edits, original, { kind: "hand", index: 13 });
  edits = revert(edits, original, { kind: "hand", index: 12 });
  assert.equal(edits, NO_EDITS);
  // 手牌の変更は河・ドラの後、点数の前に並ぶ。
  const mixed = setScore(
    setHand(setDora(NO_EDITS, original, 0, "N"), original, VIEWER, 12, "1m"),
    original,
    0,
    30000,
  );
  assert.deepEqual(Object.keys(mixed), ["dora", "hand", "scores"]);
});

test("a hand edit re-sorts the viewer's hand and maps each shown tile to its original position", () => {
  const original = position();
  let edits = setHand(NO_EDITS, original, VIEWER, 12, "1m");
  let board = applyEdits(original, edits, VIEWER);
  const viewer = board.snapshot.players[VIEWER];
  assert.deepEqual(viewer.hand, ["1m", ...HAND.slice(0, 12)]);
  assert.equal(viewer.draw, "6p");
  assert.deepEqual(
    board.handOrigins,
    [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  );
  assert.equal(board.snapshot.players[1], original.players[1]);
  assert.equal(viewer.river, original.players[VIEWER].river);
  // 後の変更も元の位置を指し、ツモ牌はツモ牌のまま。
  edits = setHand(edits, original, VIEWER, 3, "E");
  edits = setHand(edits, original, VIEWER, 13, "1p");
  board = applyEdits(original, edits, VIEWER);
  assert.deepEqual(board.snapshot.players[VIEWER].hand, [
    "1m",
    "5m",
    "5m",
    "5m",
    "4p",
    "5pr",
    "7s",
    "7s",
    "8s",
    "9s",
    "E",
    "W",
    "W",
  ]);
  assert.equal(board.snapshot.players[VIEWER].draw, "1p");
  assert.deepEqual(
    board.handOrigins,
    [12, 0, 1, 2, 4, 5, 6, 7, 8, 9, 3, 10, 11],
  );
  assert.equal(handOrigin(board.handOrigins, 10), 3);
  assert.equal(handOrigin(board.handOrigins, 13), 13);
  assert.deepEqual(original.players[VIEWER].hand, HAND);
});

test("changes are listed by river, dora, hand, score, honba and kyotaku", () => {
  const original = position();
  let edits = setSticks(NO_EDITS, original, "kyotaku", 0);
  edits = setScore(edits, original, 1, 27000);
  edits = setHand(edits, original, VIEWER, 13, "7p");
  edits = setDora(edits, original, 0, "5pr");
  edits = setRiver(edits, original, 1, 0, { tile: "S" });
  edits = setSticks(edits, original, "honba", 3);
  assert.deepEqual(
    listChanges(edits, original, VIEWER).map((change) => [
      change.kind,
      change.from,
      change.to,
    ]),
    [
      [
        "river",
        original.players[1].river[0],
        { seat: 1, index: 0, tile: "S", tsumogiri: false },
      ],
      ["dora", "4m", "5pr"],
      ["hand", "6p", "7p"],
      ["score", 25000, 27000],
      ["honba", 1, 3],
      ["kyotaku", 1, 0],
    ],
  );
});

test("every river tile that was not called can change, the viewer's own too", () => {
  const original = position();
  assert.equal(canEditRiver(original, 3, 0), true);
  assert.equal(canEditRiver(original, VIEWER, 0), true);
  assert.equal(canEditRiver(original, 1, 1), false);
  assert.equal(canEditRiver(original, 1, 7), false);
});

test("the viewer's own river keeps the recorded discard style and counts as seen", () => {
  const original = position();
  const edits = setRiver(NO_EDITS, original, VIEWER, 0, { tile: "S" });
  assert.deepEqual(edits.rivers, [
    { seat: VIEWER, index: 0, tile: "S", tsumogiri: false },
  ]);
  const board = applyEdits(original, edits, VIEWER);
  assert.deepEqual(board.snapshot.players[VIEWER].river, [{ tile: "S" }]);
  assert.equal(board.snapshot.players[VIEWER].hand, original.players[VIEWER].hand);
  // 自分の河の牌も、ほかの所の牌の一覧で見えている牌として数える。
  const choices = tileChoices(board, VIEWER, { kind: "river", seat: 0, index: 0 });
  assert.equal(find(choices, "S").visible, 1);
  assert.equal(find(choices, "N").visible, 0);
});

test("the hand changes only at a decision of the viewer before riichi", () => {
  const original = position();
  assert.equal(canEditHand(original, VIEWER, DECISION), true);
  assert.equal(canEditHand(original, VIEWER, null), false);
  assert.equal(canEditHand(original, 3, { decision: true, actor: 3 }), false);
});

test("tile choices stop at four visible copies and one red five", () => {
  const original = position();
  const board = applyEdits(original, NO_EDITS, VIEWER);
  const target = { kind: "river", seat: 3, index: 1 };
  const choices = tileChoices(board, VIEWER, target);
  assert.deepEqual(find(choices, "5m"), {
    tile: "5m",
    visible: 3,
    current: false,
    disabled: false,
  });
  assert.equal(find(choices, "5pr").disabled, true);
  assert.equal(find(choices, "5p").disabled, false);
  // 鳴かれた 2s は副露の側だけで数える。
  assert.equal(find(choices, "2s").visible, 1);
  assert.equal(find(choices, "C").current, true);
  const crowded = applyEdits(
    original,
    setDora(NO_EDITS, original, 0, "5m"),
    VIEWER,
  );
  assert.equal(find(tileChoices(crowded, VIEWER, target), "5m").disabled, true);
  // 変える所の牌そのものは数えないので、今の牌は選び直せる。
  const own = tileChoices(crowded, VIEWER, { kind: "dora", index: 0 });
  assert.equal(find(own, "5m").current, true);
  assert.equal(find(own, "5m").disabled, false);
});

test("tile choices for the hand skip the edited spot and count the edited hand", () => {
  const original = position();
  const board = applyEdits(
    original,
    setHand(NO_EDITS, original, VIEWER, 12, "1m"),
    VIEWER,
  );
  const spot = tileChoices(board, VIEWER, { kind: "hand", index: 12 });
  assert.equal(find(spot, "1m").current, true);
  assert.equal(find(spot, "1m").visible, 1);
  assert.equal(find(spot, "P").visible, 0);
  // 赤の 5 は今の牌なら選び直せ、ほかの所からは選べない。
  const red = tileChoices(board, VIEWER, { kind: "hand", index: 5 });
  assert.equal(find(red, "5pr").current, true);
  assert.equal(find(red, "5pr").disabled, false);
  assert.equal(
    find(tileChoices(board, VIEWER, { kind: "hand", index: 4 }), "5pr")
      .disabled,
    true,
  );
  const draw = tileChoices(board, VIEWER, { kind: "hand", index: 13 });
  assert.equal(find(draw, "6p").current, true);
  assert.equal(find(draw, "6p").visible, 0);
  const fourth = applyEdits(
    original,
    setHand(NO_EDITS, original, VIEWER, 12, "5m"),
    VIEWER,
  );
  assert.equal(
    find(
      tileChoices(fourth, VIEWER, { kind: "river", seat: 3, index: 1 }),
      "5m",
    ).disabled,
    true,
  );
});

test("a returned table keeps the decision and the player names", () => {
  const original = position();
  const table = {
    players: original.players.map(({ seat, name, ...player }) => ({
      ...player,
      score: player.score - 1000,
    })),
    dora: original.dora,
    honba: original.honba,
    kyotaku: original.kyotaku,
    remaining: 49,
    activeSeat: 2,
  };
  const shown = tableSnapshot(original, table);
  assert.equal(shown.eventIndex, 57);
  assert.equal(shown.decision, true);
  assert.equal(shown.remaining, 49);
  assert.deepEqual(
    shown.players.map((player) => [player.seat, player.name, player.score]),
    [
      [0, "Player 1", 24000],
      [1, "Player 2", 24000],
      [2, "Player 3", 24000],
      [3, "Player 4", 23000],
    ],
  );
  assert.deepEqual(
    ["CHI", "PON", "DAIMINKAN", "PASS"].map((type) => isCall({ type })),
    [true, true, false, false],
  );
});

test("candidates are compared by action and rank", () => {
  const candidate = (type, tiles, valueLoss, extra = {}) => ({
    actionId: 0,
    type,
    tiles,
    probability: 0.2,
    valueLoss,
    chosen: false,
    tsumogiri: false,
    ...extra,
  });
  const original = [
    candidate("DAHAI", ["8p"], 0),
    candidate("DAHAI", ["9m"], 0.003),
    candidate("DAHAI", ["5s"], 0.009),
  ];
  const edited = [
    candidate("DAHAI", ["5s"], 0),
    candidate("DAHAI", ["8p"], 0.001),
    candidate("DAHAI", ["9m"], 0.004, { tsumogiri: true }),
  ];
  const rows = compareCandidates(original, edited);
  assert.deepEqual(rows.get(edited[0]), { previous: original[2], rank: 3 });
  assert.deepEqual(rows.get(edited[1]), { previous: original[0], rank: 1 });
  // ツモ切りは別の行動として扱う。
  assert.equal(rows.get(edited[2]), null);
});
