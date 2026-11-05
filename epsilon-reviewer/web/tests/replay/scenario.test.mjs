import test from "node:test";
import assert from "node:assert/strict";
import {
  NO_EDITS,
  STICK_LIMIT,
  applyEdits,
  canEditHand,
  canEditRiver,
  canRevert,
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
/** 席 3 の河の 1 番（C）の牌の一覧。 */
const riverChoices = (original, edits) =>
  tileChoices(applyEdits(original, edits, VIEWER), VIEWER, {
    kind: "river",
    seat: 3,
    index: 1,
  });

/** 河を「牌 牌* …」（* はツモ切り）から作る。 */
const river = (text) =>
  text
    .split(" ")
    .map((tile) =>
      tile.endsWith("*")
        ? discard(tile.slice(0, -1), { tsumogiri: true })
        : discard(tile),
    );
const SWAPPER = 3;
const swapSpot = (index) => ({ kind: "river", seat: SWAPPER, index });

// 席 3 の手番の局面。8 萬は席 0 の河の 2 枚と席 3 の河の 6 番・13 番で 4 枚とも見えている。
function swapPosition() {
  return {
    ...position(),
    actor: SWAPPER,
    activeSeat: SWAPPER,
    players: [
      playerSnapshot(0, { river: river("8m 8m") }),
      playerSnapshot(1, { river: river("5m") }),
      playerSnapshot(2),
      playerSnapshot(SWAPPER, {
        river: river("W 1p C 1m 9p 7s 8m* 2s 1m* 1m* W* 5m* 8p 8m"),
        hand: [
          "2m",
          "3m",
          "4m",
          "5mr",
          "6m",
          "2p",
          "3p",
          "4p",
          "6s",
          "7s",
          "8s",
          "N",
          "N",
        ],
        draw: "P",
      }),
    ],
    dora: ["3s"],
  };
}

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
  assert.equal(
    board.snapshot.players[VIEWER].hand,
    original.players[VIEWER].hand,
  );
  // 自分の河の牌も、ほかの所の牌の一覧で見えている牌として数える。
  const choices = tileChoices(board, VIEWER, {
    kind: "river",
    seat: 0,
    index: 0,
  });
  assert.equal(find(choices, "S").visible, 1);
  assert.equal(find(choices, "N").visible, 0);
});

test("the hand changes only at a decision of the viewer before riichi", () => {
  const original = position();
  assert.equal(canEditHand(original, VIEWER, DECISION), true);
  assert.equal(canEditHand(original, VIEWER, null), false);
  assert.equal(canEditHand(original, 3, { decision: true, actor: 3 }), false);
});

test("tile choices stop at three plain fives and one red five of a suit", () => {
  const original = position();
  // 赤でない 5 萬の 3 枚が手にあり、赤の 5 萬は見えていない。
  let choices = riverChoices(original, NO_EDITS);
  assert.deepEqual(find(choices, "5m"), {
    tile: "5m",
    visible: 3,
    current: false,
    disabled: true,
  });
  assert.deepEqual(find(choices, "5mr"), {
    tile: "5mr",
    visible: 0,
    current: false,
    disabled: false,
  });
  // 手にある赤の 5 筒は、赤でない 5 筒と別に数える。
  assert.equal(find(choices, "5pr").disabled, true);
  assert.deepEqual(find(choices, "5p"), {
    tile: "5p",
    visible: 0,
    current: false,
    disabled: false,
  });
  // 手の 5 萬の 1 枚を赤にすると、赤の 5 萬は選べず、赤でない 5 萬は選べる。
  choices = riverChoices(
    original,
    setHand(NO_EDITS, original, VIEWER, 0, "5mr"),
  );
  assert.deepEqual(find(choices, "5mr"), {
    tile: "5mr",
    visible: 1,
    current: false,
    disabled: true,
  });
  assert.deepEqual(find(choices, "5m"), {
    tile: "5m",
    visible: 2,
    current: false,
    disabled: false,
  });
});

test("other tiles stop at four copies and a called tile counts once", () => {
  const original = position();
  const choices = riverChoices(original, NO_EDITS);
  // 鳴かれた 2s は副露の側だけで数える。
  assert.equal(find(choices, "2s").visible, 1);
  assert.equal(find(choices, "C").current, true);
  // 手の西 2 枚に、ドラ表示牌と上家の河の 1 枚を足すと 4 枚になる。
  let edits = setDora(NO_EDITS, original, 0, "W");
  assert.equal(find(riverChoices(original, edits), "W").disabled, false);
  // ドラ表示牌の所は、その所の西を数えず、今の牌として選び直せる。
  const dora = tileChoices(applyEdits(original, edits, VIEWER), VIEWER, {
    kind: "dora",
    index: 0,
  });
  assert.deepEqual(find(dora, "W"), {
    tile: "W",
    visible: 2,
    current: true,
    disabled: false,
  });
  edits = setRiver(edits, original, 1, 0, { tile: "W" });
  assert.deepEqual(find(riverChoices(original, edits), "W"), {
    tile: "W",
    visible: 4,
    current: false,
    disabled: true,
  });
});

test("own discards can swap a tile whose four copies are all visible", () => {
  const original = swapPosition();
  const choices = (edits, index) =>
    tileChoices(applyEdits(original, edits, SWAPPER), SWAPPER, swapSpot(index));
  // 元の局面では 8 萬が 4 枚とも見えているので、11 番の 5 萬を 8 萬にはできない。
  assert.equal(find(choices(NO_EDITS, 11), "8m").disabled, true);
  assert.equal(find(choices(NO_EDITS, 13), "5m").disabled, false);
  // 13 番の 8 萬を 5 萬にした局面では見えている 8 萬が 3 枚になり、11 番を 8 萬にできる。
  let edits = setRiver(NO_EDITS, original, SWAPPER, 13, { tile: "5m" });
  assert.deepEqual(find(choices(edits, 11), "8m"), {
    tile: "8m",
    visible: 3,
    current: false,
    disabled: false,
  });
  edits = setRiver(edits, original, SWAPPER, 11, { tile: "8m" });
  assert.deepEqual(edits.rivers, [
    { seat: SWAPPER, index: 11, tile: "8m", tsumogiri: true },
    { seat: SWAPPER, index: 13, tile: "5m", tsumogiri: false },
  ]);
});

test("a spot is restored only when its original tile can be chosen again", () => {
  const original = swapPosition();
  const restorable = (edits, target) =>
    canRevert(applyEdits(original, edits, SWAPPER), original, SWAPPER, target);
  let edits = setRiver(NO_EDITS, original, SWAPPER, 13, { tile: "5m" });
  edits = setRiver(edits, original, SWAPPER, 11, { tile: "8m" });
  // 入れ替えた後は、13 番だけを 8 萬に戻すと 8 萬が 5 枚見えるので戻せず、11 番の 5 萬は戻せる。
  assert.equal(restorable(edits, swapSpot(13)), false);
  assert.equal(restorable(edits, swapSpot(11)), true);
  assert.equal(restorable(edits, { kind: "score", seat: 0 }), true);
  // 11 番を戻した後は、13 番も戻せる。
  edits = revert(edits, original, swapSpot(11));
  assert.equal(restorable(edits, swapSpot(13)), true);
  // 手牌も同じ。手の 5 萬を 8 萬にして河に赤でない 5 萬を足すと、戻す 5 萬の写しが残らない。
  const base = position();
  const hand = setRiver(setHand(NO_EDITS, base, VIEWER, 0, "8m"), base, 3, 1, {
    tile: "5m",
  });
  const spot = { kind: "hand", index: 0 };
  assert.equal(
    canRevert(applyEdits(base, hand, VIEWER), base, VIEWER, spot),
    false,
  );
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
  // 手の 5 萬の所は、ほかの 2 枚だけを数えるので 5 萬のまま選べる。
  const five = tileChoices(board, VIEWER, { kind: "hand", index: 0 });
  assert.deepEqual(find(five, "5m"), {
    tile: "5m",
    visible: 2,
    current: true,
    disabled: false,
  });
  // ほかの所からは変えた手牌も数え、赤でない 5 萬は 3 枚とも手にあるので置けない。
  const other = tileChoices(board, VIEWER, { kind: "hand", index: 3 });
  assert.equal(find(other, "1m").visible, 2);
  assert.equal(find(other, "5m").disabled, true);
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
