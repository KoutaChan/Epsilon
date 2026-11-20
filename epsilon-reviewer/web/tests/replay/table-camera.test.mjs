import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { measurePlayerRowWidth } from "../../src/features/replay/renderer/tileLayout.ts";
import { screenBounds } from "../../src/features/replay/renderer/projection.ts";
import { positionDrawArrival } from "../../src/features/replay/renderer/tiles.ts";
import { meld } from "./fixtures.mjs";
import {
  EPSILON,
  SCREEN_SIZES,
  VARIANTS,
  buildFrame,
  makeMeld,
  makeTileProjectionFixture,
} from "./table-fixtures.mjs";

function screenRectanglesOverlap(a, b) {
  return (
    a.x < b.x + b.w - EPSILON &&
    a.x + a.w > b.x + EPSILON &&
    a.y < b.y + b.h - EPSILON &&
    a.y + a.h > b.y + EPSILON
  );
}

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
