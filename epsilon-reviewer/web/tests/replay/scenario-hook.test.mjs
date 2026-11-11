import test from "node:test";
import assert from "node:assert/strict";
import { useScenario } from "../../src/features/replay/scenario/useScenario.ts";
import { playerSnapshot } from "./fixtures.mjs";
import { mountHook } from "./hooks.mjs";

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

/** 見ている席（席 0）が判断する位置。usePlayback が返すもののうち、useScenario が読むものだけを持つ。 */
function position(eventIndex, viewer) {
  return {
    model: { temperature: 1 },
    roundIndex: 0,
    seat: 0,
    decision: { decision: true, actor: 0, eventIndex },
    snapshot: {
      players: [
        playerSnapshot(0, viewer),
        playerSnapshot(1),
        playerSnapshot(2),
        playerSnapshot(3),
      ],
      dora: ["1m"],
      honba: 0,
      kyotaku: 0,
    },
    candidates: [],
    dispatch() {},
  };
}

/** 自分のツモの判断（手牌 13 枚とツモ牌）。 */
const DRAW = position(41, { hand: HAND, draw: "1s" });
/** その 1 つ前の、上家の打牌への応答の判断（手牌 13 枚、ツモ牌なし）。 */
const RESPONSE = position(40, { hand: HAND, draw: null });
const CLIENT = { evaluate: () => new Promise(() => {}) };

async function mount(t, playback) {
  const hook = await mountHook((value) => useScenario(value, CLIENT), playback);
  t.after(hook.unmount);
  await hook.act(() => hook.current.toggle());
  return hook;
}

// 選んでいる所は、その判断の局面の中の位置を指す。別の判断の描画に残ると、編集の吹き出しがその局面に無い牌を読む。
test("moving to another decision drops the selection before anything is drawn with it", async (t) => {
  const hook = await mount(t, DRAW);
  await hook.act(() => hook.current.select({ kind: "hand", index: 13 }));
  assert.deepEqual(hook.current.selection, { kind: "hand", index: 13 });
  const drawn = hook.seen.length;
  await hook.render(RESPONSE);
  assert.ok(hook.seen.length > drawn);
  for (const scenario of hook.seen.slice(drawn)) {
    assert.equal(scenario.editing, true);
    assert.equal(scenario.selection, null);
  }
  await hook.render(DRAW);
  assert.equal(hook.current.selection, null);
});

// 元の局面を見ているかも、その判断だけのもの。変更の無い判断へ持ち越すと、目のボタンも卓も押せず、編集へ戻れない。
test("showing the original board at one decision does not stop editing at another", async (t) => {
  const hook = await mount(t, DRAW);
  await hook.act(() => hook.current.setDora(0, "2m"));
  await hook.act(() => hook.current.showBoard("original"));
  assert.equal(hook.current.board, "original");
  assert.equal(hook.current.editing, false);
  await hook.render(RESPONSE);
  assert.equal(hook.current.changes.length, 0);
  assert.equal(hook.current.board, "edited");
  assert.equal(hook.current.editing, true);
  assert.notEqual(hook.current.branchTo, null);
  await hook.render(DRAW);
  assert.equal(hook.current.changes.length, 1);
  assert.equal(hook.current.board, "edited");
});
