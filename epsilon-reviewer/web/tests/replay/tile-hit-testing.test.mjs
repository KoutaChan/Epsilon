import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { measurePlayerRowWidth } from "../../src/features/replay/renderer/tileLayout.ts";
import { local } from "../../src/features/replay/renderer/projection.ts";
import {
  handTargets,
  riverTargets,
  sameTargets,
} from "../../src/features/replay/renderer/targets.ts";
import { meld } from "./fixtures.mjs";
import {
  EPSILON,
  SCREEN_SIZES,
  SEATS,
  buildFrame,
  makeMeld,
  makeTileProjectionFixture,
} from "./table-fixtures.mjs";

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

function traceGlyphInterior(glyph, camera, renderedTiles, raycaster) {
  const origin = camera.getWorldPosition(new THREE.Vector3());
  const samples = [];
  // 2 つの三角形の境目を避けて、図柄の内側を見る。
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
            // このテストの中だけで加槓牌を真上に重ねて置き、鳴いた牌の図柄が隠れたら見つけられることを確かめる。
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
      // 画面のピクセルごとに、光線が最初に当たる牌と、いちばん上に重なる当たり判定の牌を比べる。
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
      // 外れてよいのは、牌の角の丸みと縁の 1 ピクセルほどだけ。
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
