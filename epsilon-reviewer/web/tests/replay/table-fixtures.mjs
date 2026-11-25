import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import { PLAYER_ROW_WIDTH } from "../../src/features/replay/renderer/tileLayout.ts";
import { createComposition } from "../../src/features/replay/renderer/composition.ts";
import { createTileLayer } from "../../src/features/replay/renderer/tiles.ts";
import { createResources } from "../../src/features/replay/renderer/resources.ts";
import { discard, meld } from "./fixtures.mjs";

export const SEATS = ["south", "east", "north", "west"];

export const EPSILON = 1e-9;

export const SCREEN_SIZES = [
  [1920, 1080, "desktop"],
  [1366, 768, "desktop"],
  [844, 390, "landscape"],
  [667, 375, "landscape"],
  [390, 844, "portrait"],
  [360, 800, "portrait"],
];

// 操作バーの大きさは table.css の値をそのまま読む（もしものボタンを含む）。

const css = readFileSync(
  new URL("../../src/features/replay/table.css", import.meta.url),
  "utf8",
).replace(/\/\*[\s\S]*?\*\//g, "");
/** セレクターが一致する CSS のルールの宣言を集める。セレクターの空白はまとめてから比べ、同じプロパティは後のルールの値で上書きする。 */

function style(selector) {
  const result = {};
  for (const [, selectors, body] of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (selectors.replace(/\s+/g, " ").trim() !== selector) continue;
    for (const declaration of body.split(";")) {
      const colon = declaration.indexOf(":");
      if (colon > 0)
        result[declaration.slice(0, colon).trim()] = declaration
          .slice(colon + 1)
          .trim();
    }
  }
  assert.ok(Object.keys(result).length > 0, `No rule for ${selector}`);
  return result;
}

const layoutRule = (layout, part) =>
  style(`#epsilon-review-material .em-device[data-layout="${layout}"] ${part}`);
/** calc(Npx * var(--em-pc-ui-scale)) の N を返す。 */

const scaled = (value) =>
  Number(/^calc\(([\d.]+)px \* var\(--em-pc-ui-scale\)\)$/.exec(value)[1]);

const pixels = (value) => Number(/^([\d.]+)px$/.exec(value)[1]);

const desktopIcon = layoutRule("desktop", ":is(.em-icon, .em-info > summary)");

export const DESKTOP_BAR = {
  units:
    scaled(layoutRule("desktop", ".em-round-trigger").width) +
    scaled(layoutRule("desktop", ".em-seat-trigger").width) +
    2 * scaled(layoutRule("desktop", ".em-icon[data-step]").width) +
    scaled(desktopIcon.width) +
    scaled(layoutRule("desktop", ".em-icon.em-scenario-toggle").width) +
    scaled(desktopIcon.width),
  height: scaled(layoutRule("desktop", ".em-playback").height),
};

const mobileBar = style(
  '#epsilon-review-material .em-device:not([data-layout="desktop"]) .em-playback',
);

export const PHONE_BARS = Object.fromEntries(
  ["landscape", "portrait"].map((layout) => {
    const bar = { ...mobileBar, ...layoutRule(layout, ".em-playback") };
    return [
      layout,
      {
        width: pixels(bar.width),
        height: pixels(bar.height),
        right: pixels(bar.right),
        top: bar.top === undefined ? null : pixels(bar.top),
      },
    ];
  }),
);

export function makeMeld(type, calledIndex = -1) {
  const size = type === "CHI" || type === "PON" ? 3 : 4;
  const tiles = Array.from({ length: size }, (_, i) => String(i + 1) + "m");
  return meld(type, tiles, {
    calledIndex,
    addedIndex: type === "KAKAN" ? 3 : -1,
  });
}

export const VARIANTS = [
  makeMeld("CHI", 0),
  ...[0, 1, 2].map((index) => makeMeld("PON", index)),
  ...[0, 1, 3].map((index) => makeMeld("DAIMINKAN", index)),
  makeMeld("ANKAN"),
  ...[0, 1, 2].map((index) => makeMeld("KAKAN", index)),
];

export const MELD_ROWS = [
  [],
  ...VARIANTS.flatMap((meld) =>
    [1, 2, 3, 4].map((count) =>
      Object.freeze(
        Array.from({ length: count }, () =>
          makeMeld(meld.type, meld.calledIndex),
        ),
      ),
    ),
  ),
  [makeMeld("CHI", 0), makeMeld("PON", 2), makeMeld("KAKAN", 1)],
  [
    makeMeld("CHI", 0),
    makeMeld("ANKAN"),
    makeMeld("KAKAN", 0),
    makeMeld("PON", 1),
  ],
  [
    makeMeld("DAIMINKAN", 3),
    makeMeld("KAKAN", 2),
    makeMeld("CHI", 0),
    makeMeld("PON", 0),
  ],
];

function makeComposition(rowWidth = PLAYER_ROW_WIDTH) {
  const resources = createResources(),
    material = resources.keep(new THREE.MeshBasicMaterial());
  const table = {
    resources,
    pieces: new THREE.Group(),
    handGroup: new THREE.Group(),
    materials: { black: material, ivory: material, back: material },
    riverBody: [material, material],
    tsumogiriBody: [material, material],
    camera: new THREE.PerspectiveCamera(55, 1, 0.1, 160),
    handCamera: new THREE.PerspectiveCamera(32, 1, 0.1, 100),
    landscapeFrame: new THREE.Group(),
    feltSurface: new THREE.Group(),
    fabricTex: { repeat: new THREE.Vector2() },
    centerGroup: new THREE.Group(),
  };
  table.camera.up.set(0, 0, 1);
  table.handCamera.up.set(0, 0, 1);
  const composition = createComposition(table, rowWidth);
  return {
    table,
    composition,
    /** カメラを合わせて、操作バーの画面上の矩形を返す。 */
    fit(width, height, mode) {
      if (mode === "desktop") {
        const w = Math.round((DESKTOP_BAR.units * width) / 1440),
          h = Math.round((DESKTOP_BAR.height * width) / 1440);
        const anchors = composition.fit(width, height, mode, h);
        return { x: (width - w) / 2, y: anchors.playbackTop, w, h };
      }
      const bar = PHONE_BARS[mode];
      const anchors = composition.fit(width, height, mode, bar.height);
      return {
        x: width - bar.width - bar.right,
        y: bar.top ?? anchors.playbackTop,
        w: bar.width,
        h: bar.height,
      };
    },
    dispose: () => resources.dispose(),
  };
}

export function makeTileProjectionFixture(rowWidth = PLAYER_ROW_WIDTH) {
  const fixture = makeComposition(rowWidth),
    previousDocument = globalThis.document;
  // 形のテストでは影の画像の canvas とツモのラベルだけを用意し、GPU では描かない。
  globalThis.document = {
    createElement: () => ({
      getContext: () => ({ beginPath() {}, roundRect() {}, fill() {} }),
      remove() {},
    }),
  };
  const layer = createTileLayer(
    { append() {} },
    fixture.table,
    { get: () => fixture.table.materials.ivory },
    rowWidth,
  );
  return {
    ...fixture,
    layer,
    dispose() {
      fixture.dispose();
      if (previousDocument === undefined) delete globalThis.document;
      else globalThis.document = previousDocument;
    },
  };
}

export function makeProjectionSnapshot(meldsBySeat, drawnSeat) {
  return {
    players: meldsBySeat.map((melds, seat) => ({
      score: 25000,
      riichi: false,
      melds,
      hand: Array(13 - 3 * melds.length).fill("1m"),
      draw: seat === drawnSeat ? "2p" : null,
      river: Array.from({ length: 18 }, (_, index) =>
        discard("3s", { riichi: index === 17 }),
      ),
    })),
  };
}

export function buildFrame(
  fixture,
  mode,
  meldsBySeat,
  viewSeat,
  drawnSeat,
  reveal,
  animateDraw = true,
) {
  return fixture.layer.update({
    snapshot: makeProjectionSnapshot(meldsBySeat, drawnSeat),
    event: { eventType: "tsumo", actor: drawnSeat },
    viewSeat,
    reveal,
    layout: mode,
    labels: { draw: "Draw" },
    animateDraw,
  });
}
