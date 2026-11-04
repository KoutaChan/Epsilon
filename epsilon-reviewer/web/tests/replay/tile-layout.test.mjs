import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as THREE from "three";
import {
  concealedTileX,
  layoutPlayerRow,
  measurePlayerRowWidth,
  PLAYER_ROW_CENTER,
  PLAYER_ROW_WIDTH,
} from "../../src/features/replay/renderer/tileLayout.ts";
import {
  local,
  tableDimensions,
  screenBounds,
} from "../../src/features/replay/renderer/projection.ts";
import { createComposition } from "../../src/features/replay/renderer/composition.ts";
import {
  createTileLayer,
  positionDrawArrival,
} from "../../src/features/replay/renderer/tiles.ts";
import { createResources } from "../../src/features/replay/renderer/resources.ts";
import {
  handTargets,
  riverTargets,
  sameTargets,
} from "../../src/features/replay/renderer/targets.ts";
import { discard } from "./fixtures.mjs";

const SEATS = ["south", "east", "north", "west"];
const EPSILON = 1e-9;
const SCREEN_SIZES = [
  [1920, 1080, "desktop"],
  [1366, 768, "desktop"],
  [844, 390, "landscape"],
  [667, 375, "landscape"],
  [390, 844, "portrait"],
  [360, 800, "portrait"],
];

// 操作の帯の大きさは table.css の値をそのまま読む（もしもの押し釦を含む）。
const css = readFileSync(
  new URL("../../src/features/replay/table.css", import.meta.url),
  "utf8",
).replace(/\/\*[\s\S]*?\*\//g, "");
/** 選択子（空白をまとめた形）の規則の宣言を、後の規則が勝つ順で合わせる。 */
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
/** calc(Npx * var(--em-pc-ui-scale)) の N。 */
const scaled = (value) =>
  Number(/^calc\(([\d.]+)px \* var\(--em-pc-ui-scale\)\)$/.exec(value)[1]);
const pixels = (value) => Number(/^([\d.]+)px$/.exec(value)[1]);
const desktopIcon = layoutRule("desktop", ":is(.em-icon, .em-info > summary)");
const DESKTOP_BAR = {
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
const PHONE_BARS = Object.fromEntries(
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

test("the control sizes include the what-if button on every layout", () => {
  assert.deepEqual(DESKTOP_BAR, { units: 742, height: 62 });
  assert.deepEqual(PHONE_BARS, {
    landscape: { width: 308, height: 44, right: 8, top: null },
    portrait: { width: 176, height: 88, right: 10, top: 16 },
  });
});

function makeMeld(type, calledIndex = -1) {
  const size = type === "CHI" || type === "PON" ? 3 : 4;
  return Object.freeze({
    type,
    calledIndex,
    addedIndex: type === "KAKAN" ? 3 : -1,
    tiles: Object.freeze(Array.from({ length: size }, (_, i) => `${i + 1}m`)),
  });
}
const VARIANTS = [
  makeMeld("CHI", 0),
  ...[0, 1, 2].map((index) => makeMeld("PON", index)),
  ...[0, 1, 3].map((index) => makeMeld("DAIMINKAN", index)),
  makeMeld("ANKAN"),
  ...[0, 1, 2].map((index) => makeMeld("KAKAN", index)),
];
const MELD_ROWS = [
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

function tileBounds(seat, x, y, width, height, placement) {
  const corners = [-width / 2, width / 2].flatMap((dx) =>
    [-height / 2, height / 2].map((dy) => local(seat, x + dx, y + dy)),
  );
  return {
    minX: Math.min(...corners.map((p) => p.x)),
    maxX: Math.max(...corners.map((p) => p.x)),
    minY: Math.min(...corners.map((p) => p.y)),
    maxY: Math.max(...corners.map((p) => p.y)),
    seat,
    placement,
  };
}
function boundsOverlap(a, b) {
  return (
    a.minX < b.maxX - EPSILON &&
    a.maxX > b.minX + EPSILON &&
    a.minY < b.maxY - EPSILON &&
    a.maxY > b.minY + EPSILON
  );
}
function assertSeparatedTiles(tiles, label) {
  for (let i = 0; i < tiles.length; i++)
    for (let j = i + 1; j < tiles.length; j++)
      assert.equal(
        boundsOverlap(tiles[i], tiles[j]),
        false,
        `${label}: tiles ${i} and ${j} overlap`,
      );
}
function meldFootprints(row, seat, scale = 1, radius = 0) {
  return row.melds.map((p) =>
    tileBounds(
      seat,
      p.x * scale,
      p.y * scale - radius,
      (p.sideways ? 4 / 3 : 1) * scale,
      (p.sideways ? 1 : 4 / 3) * scale,
      p,
    ),
  );
}
function handFootprints(
  row,
  meldCount,
  seat,
  standing,
  phase,
  scale = 1,
  radius = 0,
) {
  const count = 13 - 3 * meldCount,
    length = count + (phase === "before-draw" ? 0 : 1);
  return Array.from({ length }, (_, index) =>
    tileBounds(
      seat,
      concealedTileX(
        index,
        phase === "after-draw" && index === count,
        row.handLeft,
      ) * scale,
      -radius,
      scale,
      (standing ? 0.61 : 4 / 3) * scale,
    ),
  );
}
function riverFootprints(seat, dimensions) {
  const w = dimensions.riverWidth;
  return Array.from({ length: 18 }, (_, index) => {
    const riichi = index === 17;
    return tileBounds(
      seat,
      dimensions.riverStart +
        (index % 6) * dimensions.pitchX +
        (riichi ? w / 6 : 0),
      -dimensions.riverRadius - Math.floor(index / 6) * dimensions.pitchY,
      riichi ? (w * 4) / 3 : w,
      riichi ? w : (w * 4) / 3,
    );
  });
}

test("concealed tiles and base melds stay in one bounded row with added kans adjacent through draw and call phases", () => {
  const left = PLAYER_ROW_CENTER - PLAYER_ROW_WIDTH / 2;
  const right = PLAYER_ROW_CENTER + PLAYER_ROW_WIDTH / 2;
  for (const melds of MELD_ROWS) {
    const row = layoutPlayerRow(melds),
      exposed = meldFootprints(row, "south");
    assert.ok(row.width > 0 && row.width <= PLAYER_ROW_WIDTH);
    for (const phase of [
      "before-draw",
      "after-draw",
      ...(melds.length ? ["after-call"] : []),
    ]) {
      const hand = handFootprints(row, melds.length, "south", true, phase);
      const tiles = [...hand, ...exposed];
      assertSeparatedTiles(tiles, `${melds.map((m) => m.type)} ${phase}`);
      for (const tile of tiles) {
        assert.ok(
          tile.minX >= left - EPSILON,
          `${phase}: tile exceeds the left row envelope`,
        );
        assert.ok(
          tile.maxX <= right + EPSILON,
          `${phase}: tile exceeds the right row envelope`,
        );
        assert.ok(
          Math.abs((tile.minY + tile.maxY) / 2 - (tile.placement?.y ?? 0)) <
            EPSILON,
          "only an added kan tile may move toward the center from the base row",
        );
      }
      if (exposed.length) {
        const gap =
          Math.min(...exposed.map((tile) => tile.minX)) -
          Math.max(...hand.map((tile) => tile.maxX));
        assert.ok(
          gap >= 0.5 - EPSILON,
          `${phase}: only ${gap} units separate the draw and meld`,
        );
      }
      if (phase === "after-draw") {
        const gap = hand.at(-1).minX - hand.at(-2).maxX;
        assert.ok(
          gap > 0.25 && gap < 0.75,
          "the draw must remain adjacent but visibly separated",
        );
      }
    }
  }
});

test("meld placement preserves every physical tile with added kans beside their called tiles", () => {
  for (const melds of MELD_ROWS) {
    const row = layoutPlayerRow(melds);
    assert.equal(
      row.melds.length,
      melds.reduce((sum, meld) => sum + meld.tiles.length, 0),
    );
    assertSeparatedTiles(meldFootprints(row, "south"), "one-row meld groups");
    for (const meld of melds) {
      const own = row.melds.filter((p) => p.meld === meld);
      assert.deepEqual(
        own.map((p) => p.index).sort(),
        meld.tiles.map((_, index) => index),
      );
      for (const placement of own)
        assert.equal(
          placement.sideways,
          placement.index === meld.calledIndex || placement.added,
        );
      const base = own.filter((placement) => !placement.added);
      for (let index = 1; index < base.length; index++)
        assert.ok(
          base[index].x > base[index - 1].x,
          "reversing meld groups must preserve the tile order inside each group",
        );
      const added = own.find((p) => p.added);
      if (meld.type === "KAKAN") {
        const called = own.find((p) => p.index === meld.calledIndex);
        assert.equal(added.x, called.x);
        assert.equal(added.index, meld.addedIndex);
        const gap = added.y - called.y - 1;
        assert.ok(
          gap > 0 && gap < 0.08,
          "the two sideways tiles need a narrow visible gap",
        );
      } else assert.equal(added, undefined);
      for (const placement of own)
        if (!placement.added) assert.equal(placement.y, 0);
    }
    for (let index = 1; index < melds.length; index++) {
      const previous = meldFootprints(
        { melds: row.melds.filter((p) => p.meld === melds[index - 1]) },
        "south",
      );
      const next = meldFootprints(
        { melds: row.melds.filter((p) => p.meld === melds[index]) },
        "south",
      );
      assert.ok(
        Math.max(...next.map((p) => p.maxX)) <
          Math.min(...previous.map((p) => p.minX)),
        "each newer meld must remain separately readable to the left",
      );
    }
  }
});

test("initial hands center their reserved draw slot independently of future meld widths", () => {
  const wideMelds = Array.from({ length: 4 }, () => makeMeld("DAIMINKAN", 3));
  const rowWidths = [
    measurePlayerRowWidth([]) + 0.4,
    measurePlayerRowWidth(wideMelds) + 0.4,
    PLAYER_ROW_WIDTH,
  ];
  const initial = layoutPlayerRow([], rowWidths[0]);
  for (const rowWidth of rowWidths) {
    const row = layoutPlayerRow([], rowWidth);
    assert.equal(
      row.handLeft,
      initial.handLeft,
      "future meld space must not shift the initial hand",
    );
    const hand = handFootprints(row, 0, "south", true, "after-draw");
    const left = hand[0].minX,
      right = hand.at(-1).maxX;
    assert.ok(
      Math.abs((left + right) / 2 - PLAYER_ROW_CENTER) < EPSILON,
      "the thirteen tiles and reserved draw slot must be centered together",
    );
    assert.ok(left >= PLAYER_ROW_CENTER - rowWidth / 2 - EPSILON);
    assert.ok(right <= PLAYER_ROW_CENTER + rowWidth / 2 + EPSILON);
  }
});

test("appending one through four mixed melds leaves older groups fixed at the row's right edge", () => {
  const melds = Object.freeze([
    makeMeld("PON", 2),
    makeMeld("CHI", 0),
    makeMeld("ANKAN"),
    makeMeld("DAIMINKAN", 1),
  ]);
  const measuredWidth = measurePlayerRowWidth(melds) + 0.4;
  for (const rowWidth of [PLAYER_ROW_WIDTH, measuredWidth]) {
    const right = PLAYER_ROW_CENTER + rowWidth / 2;
    let previous = layoutPlayerRow([], rowWidth);
    for (let count = 1; count <= melds.length; count++) {
      const row = layoutPlayerRow(melds.slice(0, count), rowWidth);
      assert.deepEqual(
        row.melds.filter((placement) => placement.meld !== melds[count - 1]),
        previous.melds,
        "a newly called group must not move or rotate any older tile",
      );
      const bounds = meldFootprints(row, "south");
      assert.ok(
        Math.abs(Math.max(...bounds.map((tile) => tile.maxX)) - right) <
          EPSILON,
        "the oldest group must remain anchored at the fixed right edge",
      );
      if (count > 1) {
        const added = bounds.filter(
          (tile) => tile.placement.meld === melds[count - 1],
        );
        const older = bounds.filter(
          (tile) => tile.placement.meld !== melds[count - 1],
        );
        assert.ok(
          Math.max(...added.map((tile) => tile.maxX)) <
            Math.min(...older.map((tile) => tile.minX)),
          "the new group must occupy space to the left of every older group",
        );
      }
      previous = row;
    }
  }
});

test("upgrading a pon within multiple melds preserves every existing tile and the hand origin", () => {
  for (const calledIndex of [0, 1, 2]) {
    for (const ponIndex of [0, 1, 2, 3]) {
      const pon = makeMeld("PON", calledIndex);
      const kan = makeMeld("KAKAN", calledIndex);
      const melds = [makeMeld("CHI", 0), makeMeld("ANKAN"), makeMeld("PON", 1)];
      melds.splice(ponIndex, 0, pon);
      const upgraded = melds.map((meld) => (meld === pon ? kan : meld));
      for (const rowWidth of [
        PLAYER_ROW_WIDTH,
        measurePlayerRowWidth(melds) + 0.4,
      ]) {
        const before = layoutPlayerRow(melds, rowWidth);
        const after = layoutPlayerRow(upgraded, rowWidth);
        assert.equal(
          after.handLeft,
          before.handLeft,
          "adding a kan tile moved the concealed hand",
        );
        assert.equal(
          after.width,
          before.width,
          "adding a kan tile widened the base row",
        );
        for (let groupIndex = 0; groupIndex < melds.length; groupIndex++) {
          const oldTiles = before.melds.filter(
            (tile) => tile.meld === melds[groupIndex],
          );
          const newTiles = after.melds.filter(
            (tile) => tile.meld === upgraded[groupIndex] && !tile.added,
          );
          assert.equal(newTiles.length, oldTiles.length);
          for (let tileIndex = 0; tileIndex < oldTiles.length; tileIndex++) {
            const oldTile = oldTiles[tileIndex],
              newTile = newTiles[tileIndex];
            assert.equal(
              newTile.index,
              oldTile.index,
              "a base tile changed its display order",
            );
            assert.equal(
              newTile.x,
              oldTile.x,
              "a base tile moved horizontally during added kan",
            );
            assert.equal(
              newTile.y,
              oldTile.y,
              "a base tile moved vertically during added kan",
            );
            assert.equal(
              newTile.sideways,
              oldTile.sideways,
              "added kan changed the original call direction",
            );
          }
        }
        const added = after.melds.find((tile) => tile.added);
        const called = after.melds.find(
          (tile) => tile.meld === kan && tile.index === calledIndex,
        );
        assert.equal(added.x, called.x);
        assert.ok(
          added.y > called.y,
          "the added tile must stay beside its original called tile",
        );
      }
    }
  }
});

test("combined rows of every seat stay clear of neighboring hands and full rivers", () => {
  for (const mode of ["desktop", "landscape", "portrait"]) {
    const d = tableDimensions(mode),
      rivers = SEATS.flatMap((seat) => riverFootprints(seat, d));
    for (const melds of MELD_ROWS) {
      const row = layoutPlayerRow(melds);
      for (const standing of [false, true]) {
        const rows = SEATS.flatMap((seat) => [
          ...handFootprints(
            row,
            melds.length,
            seat,
            standing,
            "after-draw",
            d.opponentScale,
            d.opponentRadius,
          ),
          ...meldFootprints(row, seat, d.opponentScale, d.opponentRadius),
        ]);
        assertSeparatedTiles(rows, `${mode}: neighboring combined player rows`);
        for (const tile of rows)
          for (const river of rivers)
            assert.equal(
              boundsOverlap(tile, river),
              false,
              `${mode}: ${tile.seat} row enters a river`,
            );
      }
    }
  }
});

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
    /** 当てはめて、操作の帯の画面の矩形を返す。 */
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
function makeTileProjectionFixture(rowWidth = PLAYER_ROW_WIDTH) {
  const fixture = makeComposition(rowWidth),
    previousDocument = globalThis.document;
  // 形の試験は影の絵の canvas とツモの札だけを用意し、GPU では描かない。
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
function makeProjectionSnapshot(meldsBySeat, drawnSeat) {
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
function screenRectanglesOverlap(a, b) {
  return (
    a.x < b.x + b.w - EPSILON &&
    a.x + a.w > b.x + EPSILON &&
    a.y < b.y + b.h - EPSILON &&
    a.y + a.h > b.y + EPSILON
  );
}
function buildFrame(
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
function glyphPoints(group, camera, width, height) {
  group.updateWorldMatrix(true, true);
  const glyph = group.children[0].children.find(
    (mesh) => mesh.geometry.type === "PlaneGeometry",
  );
  if (!glyph) return [];
  return [0, 1, 3, 2].map((index) => {
    const p = new THREE.Vector3()
      .fromBufferAttribute(glyph.geometry.attributes.position, index)
      .applyMatrix4(glyph.matrixWorld)
      .project(camera);
    return new THREE.Vector2(((p.x + 1) * width) / 2, ((1 - p.y) * height) / 2);
  });
}
function polygonArea(points) {
  return (
    Math.abs(
      points.reduce((sum, p, i) => {
        const next = points[(i + 1) % points.length];
        return sum + p.x * next.y - next.x * p.y;
      }, 0),
    ) / 2
  );
}

test("rivers carry the riichi marker past called tiles without changing replay data", () => {
  const plain = discard("3s");
  const riichi = discard("4s", { riichi: true });
  const calledRiichi = discard("4s", { riichi: true, called: true });
  const calledDiscard = discard("3s", { called: true });
  const cases = [
    { name: "ordinary river", river: [plain, plain], sideways: -1 },
    { name: "declaration", river: [plain, riichi, plain], sideways: 1 },
    { name: "called declaration", river: [plain, calledRiichi], sideways: -1 },
    {
      name: "next discard",
      river: [plain, calledRiichi, plain, plain],
      sideways: 1,
    },
    {
      name: "successive calls",
      river: [calledDiscard, plain, calledRiichi, calledDiscard, plain, plain],
      sideways: 1,
    },
    {
      name: "end of row",
      river: [...Array(5).fill(plain), calledRiichi, plain, plain],
      sideways: 5,
    },
    {
      name: "start of next row",
      river: [...Array(6).fill(plain), calledRiichi, plain, plain],
      sideways: 6,
    },
    { name: "rewind before riichi", river: [plain, plain], sideways: -1 },
  ];
  const fixture = makeTileProjectionFixture();
  try {
    for (const mode of ["desktop", "landscape", "portrait"]) {
      const dimensions = tableDimensions(mode);
      for (const scenario of cases) {
        const river = Object.freeze(scenario.river);
        const snapshot = makeProjectionSnapshot([[], [], [], []], -1);
        for (const player of snapshot.players) player.river = river;
        fixture.layer.update({
          snapshot,
          event: { eventType: "dahai", actor: 0 },
          viewSeat: 0,
          reveal: true,
          layout: mode,
          labels: { draw: "Draw" },
          animateDraw: false,
        });
        for (const seat of SEATS) {
          const tiles = fixture.table.pieces.children.filter(
            (group) =>
              group.isGroup &&
              group.userData.kind === "river" &&
              group.userData.seat === seat,
          );
          const context = `${mode}: ${seat}: ${scenario.name}`;
          const baseAngle = local(seat, 0, 0).angle;
          assert.equal(
            tiles.length,
            river.filter((tile) => !tile.called).length,
            context,
          );
          for (const [index, tile] of tiles.entries()) {
            const sideways = index === scenario.sideways;
            assert.ok(
              Math.abs(
                tile.rotation.z - baseAngle - (sideways ? Math.PI / 2 : 0),
              ) < EPSILON,
              `${context}: discard ${index} has the wrong orientation`,
            );
            if (index % 6 === 0) continue;
            const previousWidth = index - 1 === scenario.sideways ? 4 / 3 : 1;
            const width = sideways ? 4 / 3 : 1;
            assert.ok(
              tile.position.distanceTo(tiles[index - 1].position) >=
                (dimensions.riverWidth * (previousWidth + width)) / 2 - EPSILON,
              `${context}: discard ${index} overlaps the previous tile`,
            );
          }
        }
      }
    }
  } finally {
    fixture.dispose();
  }
});

test("initial tile positions along every seat stay centered across record widths, viewpoint, draws and reveal settings", () => {
  const wideMelds = Array.from({ length: 4 }, () => makeMeld("DAIMINKAN", 3));
  const rowWidths = [
    measurePlayerRowWidth([]) + 0.4,
    measurePlayerRowWidth(wideMelds) + 0.4,
    PLAYER_ROW_WIDTH,
  ];
  const centeredRow = layoutPlayerRow([]);
  for (const rowWidth of rowWidths) {
    const fixture = makeTileProjectionFixture(rowWidth);
    try {
      for (const mode of ["desktop", "landscape", "portrait"])
        for (let viewSeat = 0; viewSeat < 4; viewSeat++)
          for (const drawnSeat of [-1, 0, 1, 2, 3])
            for (const reveal of [false, true]) {
              buildFrame(
                fixture,
                mode,
                [[], [], [], []],
                viewSeat,
                drawnSeat,
                reveal,
                false,
              );
              for (const [relativeSeat, seat] of SEATS.entries()) {
                const parent =
                  relativeSeat === 0
                    ? fixture.table.handGroup
                    : fixture.table.pieces;
                const hand = parent.children.filter(
                  (group) =>
                    group.isGroup &&
                    group.userData.kind === "hand" &&
                    group.userData.seat === seat,
                );
                const hasDraw = (viewSeat + relativeSeat) % 4 === drawnSeat;
                assert.equal(hand.length, 13 + Number(hasDraw));
                const axis = local(seat, 1, 0);
                for (const [index, group] of hand.entries()) {
                  // 卓の半径とカメラは対局の幅ごとに違ってよい。
                  const alongSeat =
                    (group.position.x * axis.x + group.position.y * axis.y) /
                    group.scale.x;
                  const expected = concealedTileX(
                    index,
                    index === 13,
                    centeredRow.handLeft,
                  );
                  assert.ok(
                    Math.abs(alongSeat - expected) < EPSILON,
                    `${mode}, row ${rowWidth}, view ${viewSeat}, ${seat}: an initial tile shifted along its seat`,
                  );
                }
              }
            }
    } finally {
      fixture.dispose();
    }
  }
});

test("viewed melds share the hand camera and scale with only added kans extending inward", () => {
  const fixture = makeTileProjectionFixture();
  try {
    for (const [width, height, mode] of SCREEN_SIZES) {
      fixture.fit(width, height, mode);
      for (const melds of MELD_ROWS) {
        buildFrame(
          fixture,
          mode,
          Array.from({ length: 4 }, () => melds),
          0,
          0,
          true,
          false,
        );
        const front = fixture.table.handGroup.children.filter((g) => g.isGroup);
        const hand = front.filter((g) => g.userData.kind === "hand");
        const exposed = front.filter((g) => g.userData.kind === "meld");
        assert.equal(hand.length, 14 - 3 * melds.length);
        assert.equal(
          exposed.length,
          melds.reduce((sum, meld) => sum + meld.tiles.length, 0),
        );
        const placements = layoutPlayerRow(melds).melds;
        for (const [index, group] of exposed.entries()) {
          assert.equal(group.userData.foreground, true);
          assert.equal(group.userData.seat, "south");
          assert.deepEqual(
            group.scale.toArray(),
            hand[0].scale.toArray(),
            "meld tiles must use the same physical scale as the hand",
          );
          assert.equal(
            group.position.y,
            hand[0].position.y + placements[index].y,
            "only the added tile may occupy the adjacent foreground row",
          );
        }
        const tableTiles = fixture.table.pieces.children.filter(
          (g) => g.isGroup,
        );
        assert.ok(
          tableTiles.every(
            (g) => g.userData.seat !== "south" || g.userData.kind === "river",
          ),
          "the viewed player must not retain a detached table meld row",
        );
        for (const seat of ["east", "north", "west"]) {
          const row = tableTiles.filter(
            (g) => g.userData.seat === seat && g.userData.kind !== "river",
          );
          const scale = tableDimensions(mode).opponentScale;
          let meldIndex = 0;
          for (const group of row) {
            assert.deepEqual(group.scale.toArray(), [scale, scale, scale]);
            const placement =
              group.userData.kind === "meld" ? placements[meldIndex++] : null;
            const radial =
              seat === "north" ? group.position.y : Math.abs(group.position.x);
            const expected =
              tableDimensions(mode).opponentRadius -
              (placement?.y ?? 0) * scale;
            assert.ok(
              Math.abs(radial - expected) < EPSILON,
              "only added kans may move inward from the concealed and called tile row",
            );
          }
        }
      }
    }
  } finally {
    fixture.dispose();
  }
});

test("fixed cameras and felt contain complete rows including hidden hands and added kans", () => {
  const fixture = makeTileProjectionFixture();
  try {
    for (const [width, height, mode] of SCREEN_SIZES) {
      fixture.fit(width, height, mode);
      const { feltSurface } = fixture.table;
      const felt = {
        minX: -14.025 * feltSurface.scale.x,
        maxX: 14.025 * feltSurface.scale.x,
        minY: feltSurface.position.y - 20.025 * feltSurface.scale.y,
        maxY: feltSurface.position.y + 20.025 * feltSurface.scale.y,
      };
      for (const melds of [
        [],
        ...VARIANTS.map((m) =>
          Array.from({ length: 4 }, () => makeMeld(m.type, m.calledIndex)),
        ),
      ])
        for (const reveal of [true, false]) {
          buildFrame(
            fixture,
            mode,
            Array.from({ length: 4 }, () => melds),
            0,
            0,
            reveal,
            false,
          );
          for (const group of fixture.table.pieces.children.filter(
            (g) => g.isGroup && g.userData.kind !== "river",
          )) {
            group.updateWorldMatrix(true, true);
            const box = new THREE.Box3().setFromObject(group);
            assert.ok(
              box.min.x >= felt.minX &&
                box.max.x <= felt.maxX &&
                box.min.y >= felt.minY &&
                box.max.y <= felt.maxY,
              `${width}x${height}: a player row extends beyond the felt`,
            );
            const rect = screenBounds(
              [group],
              fixture.table.camera,
              width,
              height,
            );
            assert.ok(
              rect.x >= -EPSILON &&
                rect.x + rect.w <= width + EPSILON &&
                rect.y >= -EPSILON &&
                rect.y + rect.h <= height + EPSILON,
              `${width}x${height}: a standing tile or added kan leaves the table camera`,
            );
          }
        }
    }
  } finally {
    fixture.dispose();
  }
});

test("foreground hands and melds avoid rivers, other players and controls throughout draw animation", () => {
  const fixture = makeTileProjectionFixture();
  const meldsBySeat = [
    [],
    [makeMeld("CHI", 0), makeMeld("KAKAN", 1), makeMeld("PON", 2)],
    [makeMeld("PON", 1)],
    Array.from({ length: 4 }, () => makeMeld("DAIMINKAN", 3)),
  ];
  try {
    for (const [width, height, mode] of SCREEN_SIZES) {
      const controls = fixture.fit(width, height, mode);
      for (let viewSeat = 0; viewSeat < 4; viewSeat++)
        for (let drawnSeat = 0; drawnSeat < 4; drawnSeat++)
          for (const reveal of [true, false]) {
            const { arrival } = buildFrame(
              fixture,
              mode,
              meldsBySeat,
              viewSeat,
              drawnSeat,
              reveal,
            );
            const tableTiles = fixture.table.pieces.children.filter(
              (g) => g.isGroup,
            );
            for (const remaining of [0, 1]) {
              positionDrawArrival(arrival, remaining);
              const tableBoxes = tableTiles.map((g) =>
                screenBounds([g], fixture.table.camera, width, height),
              );
              for (const box of tableBoxes)
                assert.equal(
                  screenRectanglesOverlap(box, controls),
                  false,
                  `${width}x${height}: controls cover a table tile`,
                );
              if (mode === "landscape")
                for (const group of tableTiles.filter(
                  (g) => g.userData.seat === "west",
                ))
                  assert.ok(
                    screenBounds([group], fixture.table.camera, width, height)
                      .x >= 156,
                    `${width}x${height}: a left player row enters the AI candidate column`,
                  );
              const frontBoxes = fixture.table.handGroup.children
                .filter((g) => g.isGroup)
                .map((g) =>
                  screenBounds([g], fixture.table.handCamera, width, height),
                );
              for (const box of frontBoxes) {
                assert.ok(
                  box.x >= -EPSILON &&
                    box.x + box.w <= width + EPSILON &&
                    box.y >= -EPSILON &&
                    box.y + box.h <= height + EPSILON,
                  `${width}x${height}: a foreground hand or meld leaves the viewport`,
                );
                assert.equal(
                  screenRectanglesOverlap(box, controls),
                  false,
                  `${width}x${height}: controls cover a foreground tile`,
                );
                for (const other of tableBoxes)
                  assert.equal(
                    screenRectanglesOverlap(box, other),
                    false,
                    `${width}x${height}, seat ${viewSeat}: a foreground tile covers a table tile`,
                  );
              }
              const gap =
                Math.min(...frontBoxes.map((b) => b.y)) -
                Math.max(...tableBoxes.map((b) => b.y + b.h));
              assert.ok(
                gap >= 6,
                `${width}x${height}: only ${gap.toFixed(2)}px separate the table and foreground row`,
              );
            }
          }
    }
  } finally {
    fixture.dispose();
  }
});

function traceGlyphInterior(glyph, camera, renderedTiles, raycaster) {
  const origin = camera.getWorldPosition(new THREE.Vector3());
  const samples = [];
  // 二つの三角形の境目を避けて、図柄の内側を見る。
  for (const x of [-0.38, 0.017, 0.36])
    for (const y of [-0.34, 0.029, 0.4]) {
      const target = new THREE.Vector3(x * 0.9, y * 1.2, 0).applyMatrix4(
        glyph.matrixWorld,
      );
      const projected = target.clone().project(camera);
      raycaster.set(origin, target.sub(origin).normalize());
      samples.push({
        x,
        y,
        projected,
        nearest: raycaster.intersectObjects(renderedTiles, true)[0]?.object,
      });
    }
  return samples;
}

test("all called and added kan glyphs remain unoccluded at every seat and screen size", () => {
  const fixture = makeTileProjectionFixture();
  const raycaster = new THREE.Raycaster();
  try {
    for (const [width, height, mode] of SCREEN_SIZES) {
      fixture.fit(width, height, mode);
      for (const calledIndex of [0, 1, 2]) {
        const pon = makeMeld("PON", calledIndex);
        const kan = makeMeld("KAKAN", calledIndex);
        buildFrame(
          fixture,
          mode,
          SEATS.map(() => [pon]),
          0,
          0,
          true,
          false,
        );
        const previous = new Map(
          SEATS.map((seat) => {
            const parent =
              seat === "south" ? fixture.table.handGroup : fixture.table.pieces;
            const tiles = parent.children.filter(
              (group) =>
                group.isGroup &&
                group.userData.kind === "meld" &&
                group.userData.seat === seat,
            );
            return [
              seat,
              tiles.map((group) => ({
                position: group.position.toArray(),
                rotation: group.rotation.toArray(),
                scale: group.scale.toArray(),
              })),
            ];
          }),
        );
        buildFrame(
          fixture,
          mode,
          SEATS.map(() => [kan]),
          0,
          0,
          true,
          false,
        );
        for (const seat of SEATS) {
          const foreground = seat === "south";
          const parent = foreground
            ? fixture.table.handGroup
            : fixture.table.pieces;
          const camera = foreground
            ? fixture.table.handCamera
            : fixture.table.camera;
          parent.updateWorldMatrix(true, true);
          const renderedTiles = parent.children.filter(
            (group) => group.isGroup,
          );
          const meld = renderedTiles.filter(
            (group) =>
              group.userData.kind === "meld" && group.userData.seat === seat,
          );
          const label = `${width}x${height}, ${seat}, called ${calledIndex}`;
          assert.equal(meld.length, 4, label);
          for (let index = 0; index < 3; index++) {
            assert.deepEqual(
              meld[index].position.toArray(),
              previous.get(seat)[index].position,
              `${label}: adding a tile moved the original pon`,
            );
            assert.deepEqual(
              meld[index].rotation.toArray(),
              previous.get(seat)[index].rotation,
              `${label}: adding a tile changed the called direction`,
            );
            assert.deepEqual(
              meld[index].scale.toArray(),
              previous.get(seat)[index].scale,
              `${label}: adding a tile resized the original pon`,
            );
          }
          const called = meld[calledIndex];
          const added = meld[kan.addedIndex];
          assert.equal(called.position.z, 0.085, label);
          assert.equal(
            added.position.z,
            called.position.z,
            `${label}: the added tile must rest beside the called tile`,
          );
          const delta = added.position.clone().sub(called.position);
          const inward = local(seat, 0, 1);
          assert.ok(
            delta.dot(new THREE.Vector3(inward.x, inward.y, 0)) > 0,
            `${label}: the added tile must extend toward the center`,
          );
          for (const [kind, group] of [
            ["called", called],
            ["added", added],
          ]) {
            const glyph = group.children[0].children.find(
              (mesh) => mesh.geometry.type === "PlaneGeometry",
            );
            assert.ok(glyph, `${label}: ${kind} tile has no glyph`);
            for (const sample of traceGlyphInterior(
              glyph,
              camera,
              renderedTiles,
              raycaster,
            )) {
              assert.ok(
                Math.abs(sample.projected.x) <= 1 &&
                  Math.abs(sample.projected.y) <= 1 &&
                  Math.abs(sample.projected.z) <= 1,
                `${label}: ${kind} glyph leaves the visible camera`,
              );
              assert.equal(
                sample.nearest?.uuid,
                glyph.uuid,
                `${label}: ${kind} glyph is occluded at ${sample.x},${sample.y}`,
              );
            }
          }
          const position = added.position.clone();
          try {
            // 加槓牌を上に重ねる積み方をこの試験の中だけで作り、鳴いた牌の図柄が隠れるのを見つけられることを確かめる。
            added.position.copy(called.position);
            added.position.z += 0.61 * added.scale.z;
            parent.updateWorldMatrix(true, true);
            const glyph = called.children[0].children.find(
              (mesh) => mesh.geometry.type === "PlaneGeometry",
            );
            assert.ok(
              traceGlyphInterior(glyph, camera, renderedTiles, raycaster).some(
                (sample) => sample.nearest !== glyph,
              ),
              `${label}: the visibility test must detect the old stack covering the called glyph`,
            );
          } finally {
            added.position.copy(position);
            parent.updateWorldMatrix(true, true);
          }
        }
      }
    }
  } finally {
    fixture.dispose();
  }
});

test("foreground meld faces stay readable beside standing hand tiles including rotated calls and added kans", () => {
  const fixture = makeTileProjectionFixture();
  try {
    for (const [width, height, mode] of SCREEN_SIZES) {
      fixture.fit(width, height, mode);
      for (const meld of [
        makeMeld("PON", 0),
        makeMeld("KAKAN", 1),
        makeMeld("ANKAN"),
      ]) {
        buildFrame(
          fixture,
          mode,
          Array.from({ length: 4 }, () => [meld]),
          0,
          0,
          true,
          false,
        );
        const front = fixture.table.handGroup.children.filter((g) => g.isGroup);
        const hand = front.find((g) => g.userData.kind === "hand");
        const handArea = polygonArea(
          glyphPoints(hand, fixture.table.handCamera, width, height),
        );
        const exposed = front.filter((g) => g.userData.kind === "meld");
        let hidden = 0;
        for (const group of exposed) {
          const points = glyphPoints(
            group,
            fixture.table.handCamera,
            width,
            height,
          );
          if (!points.length) {
            hidden++;
            continue;
          }
          const edges = points.map((p, i) =>
            p.distanceTo(points[(i + 1) % points.length]),
          );
          assert.ok(
            Math.min(...edges) / Math.max(...edges) >= 0.35,
            `${width}x${height}: a called tile face is nearly edge-on`,
          );
          const ratio = handArea / polygonArea(points);
          assert.ok(
            ratio >= 0.5 && ratio <= 2,
            `${width}x${height}: hand/meld projected face area ratio is ${ratio.toFixed(2)}`,
          );
        }
        assert.equal(
          hidden,
          meld.type === "ANKAN" ? 2 : 0,
          "only closed-kan outer tiles should hide their faces",
        );
      }
    }
  } finally {
    fixture.dispose();
  }
});

test("standing foreground faces retain readable portrait proportions from every viewpoint while hidden opponents stay upright", () => {
  const meldsBySeat = [
    [],
    [makeMeld("PON", 1)],
    [makeMeld("CHI", 0), makeMeld("KAKAN", 2)],
    [makeMeld("ANKAN")],
  ];
  const rowWidth = Math.max(...meldsBySeat.map(measurePlayerRowWidth)) + 0.4;
  const fixture = makeTileProjectionFixture(rowWidth);
  try {
    for (const [width, height, mode] of SCREEN_SIZES) {
      fixture.fit(width, height, mode);
      for (let viewSeat = 0; viewSeat < 4; viewSeat++)
        for (const reveal of [true, false]) {
          buildFrame(
            fixture,
            mode,
            meldsBySeat,
            viewSeat,
            viewSeat,
            reveal,
            false,
          );
          const hand = fixture.table.handGroup.children.filter(
            (group) => group.isGroup && group.userData.kind === "hand",
          );
          const central = hand[Math.floor(hand.length / 2)];
          const points = glyphPoints(
            central,
            fixture.table.handCamera,
            width,
            height,
          );
          assert.equal(
            points.length,
            4,
            "the viewed hand must always reveal its faces",
          );
          const faceWidth =
            (points[0].distanceTo(points[1]) +
              points[2].distanceTo(points[3])) /
            2;
          const faceHeight =
            (points[1].distanceTo(points[2]) +
              points[3].distanceTo(points[0])) /
            2;
          assert.ok(
            faceHeight / faceWidth >= 1.2,
            `${width}x${height}, seat ${viewSeat}: the standing face is compressed to aspect ${(faceHeight / faceWidth).toFixed(3)}`,
          );
          if (mode === "desktop") {
            const river = fixture.table.pieces.children.filter(
              (group) =>
                group.isGroup &&
                group.userData.kind === "river" &&
                group.userData.seat === "south",
            )[12];
            const riverPoints = glyphPoints(
              river,
              fixture.table.camera,
              width,
              height,
            );
            const riverWidth =
              (riverPoints[0].distanceTo(riverPoints[1]) +
                riverPoints[2].distanceTo(riverPoints[3])) /
              2;
            assert.ok(
              faceWidth / riverWidth >= 1.15 && faceWidth / riverWidth <= 1.4,
              `${width}x${height}: foreground/nearest river face width ratio is ${(faceWidth / riverWidth).toFixed(2)}`,
            );
          }
          if (mode === "landscape")
            assert.ok(
              faceWidth >= 28,
              `${width}x${height}, seat ${viewSeat}: only ${faceWidth.toFixed(1)}px remain for a foreground face`,
            );
          for (const group of hand) {
            const faceNormal = new THREE.Vector3(0, 0, 1).applyQuaternion(
              group.children[0].quaternion,
            );
            assert.ok(
              Math.abs(faceNormal.z) < 0.5,
              "readability must preserve a standing hand rather than laying its faces flat",
            );
          }
          const otherHands = fixture.table.pieces.children.filter(
            (group) => group.isGroup && group.userData.kind === "hand",
          );
          assert.ok(otherHands.length > 0);
          for (const group of otherHands) {
            const body = group.children[0];
            const faceNormal = new THREE.Vector3(0, 0, 1).applyQuaternion(
              body.quaternion,
            );
            const hasGlyph = body.children.some(
              (mesh) => mesh.geometry.type === "PlaneGeometry",
            );
            assert.equal(hasGlyph, reveal);
            if (!reveal)
              assert.ok(
                Math.abs(faceNormal.z) < EPSILON,
                "concealed opponents must remain fully upright with hidden faces",
              );
          }
        }
    }
  } finally {
    fixture.dispose();
  }
});

test("changing meld counts, viewpoint and reveal settings keeps both camera projections fixed", () => {
  const fixture = makeTileProjectionFixture();
  try {
    for (const [width, height, mode] of SCREEN_SIZES) {
      fixture.fit(width, height, mode);
      const cameras = () =>
        [fixture.table.camera, fixture.table.handCamera].flatMap((camera) => [
          ...camera.matrixWorld.elements,
          ...camera.projectionMatrix.elements,
        ]);
      const original = cameras();
      for (const melds of [
        [],
        [makeMeld("PON", 0)],
        Array.from({ length: 4 }, () => makeMeld("DAIMINKAN", 3)),
      ])
        for (let viewSeat = 0; viewSeat < 4; viewSeat++)
          for (const reveal of [true, false]) {
            buildFrame(
              fixture,
              mode,
              Array.from({ length: 4 }, () => melds),
              viewSeat,
              viewSeat,
              reveal,
              false,
            );
            fixture.fit(width, height, mode);
            assert.deepEqual(
              cameras(),
              original,
              "a replay update moved a camera",
            );
          }
    }
  } finally {
    fixture.dispose();
  }
});

test("a repeated fit with the same viewport, layout and controls reuses the cameras", () => {
  const fixture = makeTileProjectionFixture();
  try {
    const first = fixture.composition.fit(1366, 768, "desktop", 59);
    const projection = fixture.table.camera.projectionMatrix.elements.slice();
    fixture.table.camera.projectionMatrix.elements[8] = 99;
    assert.equal(fixture.composition.fit(1366, 768, "desktop", 59), first);
    assert.equal(fixture.table.camera.projectionMatrix.elements[8], 99);
    const refitted = fixture.composition.fit(1366, 768, "desktop", 60);
    assert.notEqual(refitted, first);
    assert.notEqual(fixture.table.camera.projectionMatrix.elements[8], 99);
    fixture.composition.fit(1366, 768, "desktop", 59);
    assert.deepEqual(
      fixture.table.camera.projectionMatrix.elements,
      projection,
    );
  } finally {
    fixture.dispose();
  }
});

test("a record-wide measured row contains every draw and later call without moving the cameras", () => {
  const records = [
    [],
    [makeMeld("PON", 1)],
    [makeMeld("CHI", 0), makeMeld("KAKAN", 2)],
    [makeMeld("CHI", 0), makeMeld("PON", 2), makeMeld("ANKAN")],
    Array.from({ length: 4 }, () => makeMeld("DAIMINKAN", 3)),
  ];
  for (const finalMelds of records) {
    const stages = Array.from({ length: finalMelds.length + 1 }, (_, count) =>
      finalMelds.slice(0, count),
    );
    const rowWidth = Math.max(...stages.map(measurePlayerRowWidth)) + 0.4;
    const fixture = makeTileProjectionFixture(rowWidth);
    try {
      for (const [width, height, mode] of SCREEN_SIZES) {
        const controls = fixture.fit(width, height, mode);
        const cameras = () =>
          [fixture.table.camera, fixture.table.handCamera].flatMap((camera) => [
            ...camera.matrixWorld.elements,
            ...camera.projectionMatrix.elements,
          ]);
        const original = cameras();
        const handEnvelope = screenBounds(
          [fixture.composition.handEnvelope],
          fixture.table.handCamera,
          width,
          height,
        );
        const fitsInside = (box, container) =>
          box.x >= container.x - EPSILON &&
          box.x + box.w <= container.x + container.w + EPSILON &&
          box.y >= container.y - EPSILON &&
          box.y + box.h <= container.y + container.h + EPSILON;
        const screen = { x: 0, y: 0, w: width, h: height };
        for (const melds of stages)
          for (const reveal of [true, false]) {
            const { arrival } = buildFrame(
              fixture,
              mode,
              Array.from({ length: 4 }, () => melds),
              0,
              0,
              reveal,
            );
            fixture.fit(width, height, mode);
            assert.deepEqual(
              cameras(),
              original,
              "a later call changed the camera selected for the whole record",
            );
            for (const remaining of [0, 1]) {
              positionDrawArrival(arrival, remaining);
              const front = fixture.table.handGroup.children
                .filter((group) => group.isGroup)
                .map((group) =>
                  screenBounds(
                    [group],
                    fixture.table.handCamera,
                    width,
                    height,
                  ),
                );
              const table = fixture.table.pieces.children
                .filter((group) => group.isGroup)
                .map((group) =>
                  screenBounds([group], fixture.table.camera, width, height),
                );
              for (const box of [...front, ...table]) {
                assert.ok(
                  fitsInside(box, screen),
                  `${width}x${height}, row ${rowWidth}: a tile leaves the viewport`,
                );
                assert.equal(
                  screenRectanglesOverlap(box, controls),
                  false,
                  `${width}x${height}, row ${rowWidth}: controls cover a tile`,
                );
              }
              for (const box of front) {
                assert.ok(
                  fitsInside(box, handEnvelope),
                  `${width}x${height}, row ${rowWidth}: a foreground tile leaves its reserved hand envelope`,
                );
                for (const other of table)
                  assert.equal(
                    screenRectanglesOverlap(box, other),
                    false,
                    `${width}x${height}, row ${rowWidth}: the foreground covers a table tile`,
                  );
              }
            }
          }
      }
    } finally {
      fixture.dispose();
    }
  }
});

test("only the seats whose player, last-discard mark or view changed are rebuilt", () => {
  const fixture = makeTileProjectionFixture();
  try {
    const snapshot = makeProjectionSnapshot([[], [], [], []], -1);
    const frame = {
      snapshot,
      event: { eventType: "dahai", actor: 1 },
      viewSeat: 0,
      reveal: true,
      layout: "desktop",
      labels: { draw: "Draw" },
      animateDraw: false,
    };
    assert.equal(fixture.layer.update(frame).rebuilt, true);
    const groups = () =>
      Object.fromEntries(
        SEATS.map((seat) => [
          seat,
          [
            ...fixture.table.pieces.children,
            ...fixture.table.handGroup.children,
          ].filter((group) => group.isGroup && group.userData.seat === seat),
        ]),
      );
    const before = groups();
    assert.equal(fixture.layer.update(frame).rebuilt, false);
    // 席 2 だけツモ牌が増えた局面。印は席 1 のまま。
    const drawn = {
      ...snapshot,
      players: snapshot.players.map((entry, seat) =>
        seat === 2 ? { ...entry, draw: "2p" } : entry,
      ),
    };
    const next = fixture.layer.update({
      ...frame,
      snapshot: drawn,
      event: { eventType: "tsumo", actor: 2 },
      animateDraw: true,
    });
    const after = groups();
    assert.equal(next.rebuilt, true);
    assert.equal(after.south[0], before.south[0]);
    assert.equal(after.west[0], before.west[0]);
    assert.notEqual(after.north[0], before.north[0]);
    // 席 1 の最後の打牌の印が外れるので、席 1 も作り直す。
    assert.notEqual(after.east[0], before.east[0]);
    assert.equal(
      next.arrival,
      after.north.filter((group) => group.userData.kind === "hand").at(-1),
    );
    assert.equal(next.annotations.length, 1);
    assert.equal(next.annotations[0].group, next.arrival);
  } finally {
    fixture.dispose();
  }
});

/** 点を含む最後の（上に重なる）多角形の位置。凸の多角形だけを扱う。 */
function topmostShape(shapes, x, y) {
  for (let index = shapes.length - 1; index >= 0; index--) {
    const points = shapes[index].points;
    let inside = true;
    for (let i = 0; i < points.length && inside; i++) {
      const [ax, ay] = points[i],
        [bx, by] = points[(i + 1) % points.length];
      inside = (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= 0;
    }
    if (inside) return index;
  }
  return -1;
}

test("hand targets pick the same tile as a ray from the hand camera", () => {
  const raycaster = new THREE.Raycaster(),
    ndc = new THREE.Vector2();
  for (const [width, height, mode] of SCREEN_SIZES) {
    const fixture = makeTileProjectionFixture();
    try {
      fixture.fit(width, height, mode);
      const built = buildFrame(
        fixture,
        mode,
        [[], [], [], []],
        1,
        1,
        true,
        false,
      );
      const camera = fixture.table.handCamera;
      const shapes = handTargets(built.hand, camera, width, height);
      const pieceOf = new Map();
      for (const piece of built.hand)
        piece.group.traverse((object) => pieceOf.set(object, piece.index));
      const xs = shapes.flatMap((shape) => shape.points.map(([x]) => x)),
        ys = shapes.flatMap((shape) => shape.points.map(([, y]) => y));
      // 画面の画素ごとに、光線が最初に当たる牌と、上に重なる当たり枠の牌を比べる。
      let tiles = 0,
        outside = 0,
        neighbor = 0;
      for (
        let y = Math.floor(Math.min(...ys)) - 4;
        y <= Math.max(...ys) + 4;
        y++
      )
        for (
          let x = Math.floor(Math.min(...xs)) - 4;
          x <= Math.max(...xs) + 4;
          x++
        ) {
          ndc.set(((x + 0.5) / width) * 2 - 1, 1 - ((y + 0.5) / height) * 2);
          raycaster.setFromCamera(ndc, camera);
          const hit = raycaster.intersectObjects(
            built.hand.map((piece) => piece.group),
            true,
          )[0];
          const truth = hit ? pieceOf.get(hit.object) : -1;
          const shape = topmostShape(shapes, x + 0.5, y + 0.5);
          const picked = shape < 0 ? -1 : shapes[shape].index;
          if (truth >= 0) tiles++;
          if (picked === truth) continue;
          if (truth >= 0 && picked >= 0) neighbor++;
          else outside++;
        }
      // 外れてよいのは、牌の丸めた角と縁の 1 画素ほどだけ。
      const size = `${width}x${height}`;
      assert.ok(outside / tiles < 0.03, `${size}: ${outside} of ${tiles}`);
      assert.ok(neighbor / tiles < 0.01, `${size}: ${neighbor} of ${tiles}`);
    } finally {
      fixture.dispose();
    }
  }
});

test("the viewer's hand pieces carry display positions with the draw last, and targets compare by value", () => {
  const fixture = makeTileProjectionFixture();
  try {
    const controls = fixture.fit(1366, 768, "desktop");
    assert.ok(controls.w > 0);
    const built = buildFrame(
      fixture,
      "desktop",
      [[], [], [], []],
      1,
      1,
      true,
      false,
    );
    assert.deepEqual(
      built.hand.map((piece) => piece.index),
      Array.from({ length: 14 }, (_, index) => index),
    );
    assert.ok(built.hand.every((piece) => piece.group.userData.foreground));
    assert.equal(built.river.length, 4 * 18);
    assert.deepEqual(
      [...new Set(built.river.map((piece) => piece.seat))].sort(),
      [0, 1, 2, 3],
    );
    const targets = () => ({
      rivers: riverTargets(built.river, fixture.table.camera, 1366, 768),
      hands: handTargets(built.hand, fixture.table.handCamera, 1366, 768),
      scores: [],
    });
    const first = targets();
    assert.equal(first.hands.length, 14);
    for (const shape of [...first.rivers, ...first.hands])
      for (const [x, y] of shape.points) {
        assert.equal(Math.round(x * 10) / 10, x);
        assert.equal(Math.round(y * 10) / 10, y);
      }
    assert.equal(sameTargets(first, targets()), true);
    assert.equal(sameTargets(first, null), false);
    assert.equal(sameTargets(null, null), true);
    built.hand[0].group.position.y -= 0.5;
    assert.equal(sameTargets(first, targets()), false);
  } finally {
    fixture.dispose();
  }
});
