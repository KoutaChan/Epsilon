import test from "node:test";
import assert from "node:assert/strict";
import { usePlaybackInput } from "../../src/features/replay/usePlaybackInput.ts";
import { mountHook } from "./hooks.mjs";

// usePlaybackInput は、イベントの対象が操作できる要素の中かを Element で見分ける。
globalThis.Element = class {};

/** 卓の要素の代わりを置いてフックを動かす。send は卓の上の wheel を 1 つ送り、moves には動かした向きが順に入る。 */
async function table(t) {
  const moves = [];
  let wheel;
  const surface = {
    addEventListener(type, listener) {
      if (type === "wheel") wheel = listener;
    },
    removeEventListener() {},
  };
  const hook = await mountHook(usePlaybackInput, {
    surface: { current: surface },
    playback: { move: (delta) => moves.push(delta), togglePlay() {} },
    menuOpen: false,
    closeMenu() {},
  });
  t.after(hook.unmount);
  const send = (deltaY, timeStamp, deltaMode = 0) =>
    wheel({
      deltaX: 0,
      deltaY,
      deltaMode,
      timeStamp,
      target: null,
      preventDefault() {},
    });
  return { moves, send };
}

test("every notch of a mouse wheel moves one position, however fast it turns", async (t) => {
  const { moves, send } = await table(t);
  for (const time of [0, 400, 430, 460]) send(100, time);
  assert.deepEqual(moves, [1, 1, 1, 1]);
  for (const time of [1000, 1030, 1060]) send(-100, time);
  assert.deepEqual(moves.slice(4), [-1, -1, -1]);
  // ページを拡大していると 1 刻みの量は小さくなり、行で数えるブラウザーは 1 刻みを 3 行で送る。
  for (const time of [2000, 2030, 2060]) send(80, time);
  for (const time of [3000, 3030]) send(3, time, 1);
  assert.deepEqual(moves.slice(7), [1, 1, 1, 1, 1]);
});

// タッチパッドや慣性スクロールは、1 回の操作で小さな wheel を数十個送る。
test("a touchpad swipe moves by its distance, not once per event", async (t) => {
  const { moves, send } = await table(t);
  for (let event = 0; event < 40; event++) send(3, event * 8);
  // 120px。始めに 1 つ、その後は 50px ごとに 1 つ。
  assert.deepEqual(moves, [1, 1, 1]);
});

test("a swipe after a pause moves at once, and turning back needs its own distance", async (t) => {
  const { moves, send } = await table(t);
  send(3, 0);
  send(3, 1000);
  assert.deepEqual(moves, [1, 1]);
  send(-3, 1008);
  assert.deepEqual(moves, [1, 1]);
  for (let event = 1; event <= 16; event++) send(-3, 1008 + event * 8);
  assert.deepEqual(moves, [1, 1, -1]);
});
