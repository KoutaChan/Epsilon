import * as THREE from "three";
import { PLAYER_ROW_CENTER, PLAYER_ROW_WIDTH } from "./tileLayout.ts";
import type { Layout, SeatSlot } from "./types.ts";

const angles: Readonly<Record<SeatSlot, number>> = {
  south: 0,
  east: Math.PI / 2,
  north: Math.PI,
  west: -Math.PI / 2,
};
const worldBox = new THREE.Box3();
const point = new THREE.Vector3();
const eye = new THREE.Vector3();

export interface TableDimensions {
  readonly riverWidth: number;
  readonly pitchX: number;
  readonly pitchY: number;
  readonly centerWidth: number;
  readonly riverRadius: number;
  readonly riverOuter: number;
  readonly riverStart: number;
  readonly opponentScale: number;
  readonly opponentRadius: number;
}
/** 正規化した画面座標（-1..1）での範囲。 */
export interface Bounds {
  readonly minX: number;
  readonly minY: number;
  readonly maxX: number;
  readonly maxY: number;
}
/** 卓の左上から測った、画面上の矩形（CSS の px）。 */
export interface ScreenRect {
  readonly x: number;
  readonly y: number;
  readonly w: number;
  readonly h: number;
}

export function tableDimensions(
  mode: Layout,
  rowWidth = PLAYER_ROW_WIDTH,
): TableDimensions {
  const landscape = mode === "landscape",
    riverWidth = landscape ? 1.55 : 1.4,
    pitchX = landscape ? 1.575 : 1.425,
    pitchY = landscape ? 2.115 : 1.915;
  const centerWidth = 6 * riverWidth,
    riverRadius =
      centerWidth / 2 + (riverWidth * 2) / 3 + (landscape ? 0.75 : 0.3);
  const riverOuter = riverRadius + 2 * pitchY + (riverWidth * 2) / 3,
    opponentScale = riverWidth,
    opponentRadius = Math.max(
      riverOuter + (2 / 3) * opponentScale + (landscape ? 1.25 : 0.35),
      (PLAYER_ROW_CENTER + rowWidth / 2 + 2 / 3) * opponentScale + 0.3,
    );
  return {
    riverWidth,
    pitchX,
    pitchY,
    centerWidth,
    riverRadius,
    riverOuter,
    riverStart: centerWidth / 2 - riverWidth / 2 - 5 * pitchX,
    opponentScale,
    opponentRadius,
  };
}

/** 席から見た座標（x は右、y は奥）を卓の座標に直す。その席の向き（ラジアン）も一緒に返す。 */
export function local(seat: SeatSlot, x: number, y: number) {
  const a = angles[seat];
  return {
    x: x * Math.cos(a) - y * Math.sin(a),
    y: x * Math.sin(a) + y * Math.cos(a),
    angle: a,
  };
}

export function pointBounds(
  objects: readonly THREE.Object3D[],
  cam: THREE.Camera,
): Bounds {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const o of objects) {
    o.updateWorldMatrix(true, true);
    const b = worldBox.setFromObject(o);
    for (const x of [b.min.x, b.max.x])
      for (const y of [b.min.y, b.max.y])
        for (const z of [b.min.z, b.max.z]) {
          const p = point.set(x, y, z).project(cam);
          minX = Math.min(minX, p.x);
          maxX = Math.max(maxX, p.x);
          minY = Math.min(minY, p.y);
          maxY = Math.max(maxY, p.y);
        }
  }
  return { minX, minY, maxX, maxY };
}

export function screenBounds(
  objects: readonly THREE.Object3D[],
  camera: THREE.Camera,
  width: number,
  height: number,
): ScreenRect {
  const b = pointBounds(objects, camera);
  return {
    x: ((b.minX + 1) * width) / 2,
    y: ((1 - b.maxY) * height) / 2,
    w: ((b.maxX - b.minX) * width) / 2,
    h: ((b.maxY - b.minY) * height) / 2,
  };
}

export function projectPoint(
  p: THREE.Vector3,
  camera: THREE.Camera,
  width: number,
  height: number,
) {
  point.copy(p).project(camera);
  return { x: ((point.x + 1) * width) / 2, y: ((1 - point.y) * height) / 2 };
}

/**
 * オブジェクトが画面に映る輪郭（凸包）と、カメラからその中心までの距離を返す。輪郭は、メッシュごとの
 * バウンディングボックスの 8 隅を、箱の向きのまま画面に写して求める。そのため、傾けて立てた手牌でも、
 * ワールド座標の軸にそろえた箱と違って牌の外の角を含まない。
 * 並んだ牌の輪郭が重なる所では、目に近い牌が隣の牌の側面を隠す。手牌はどれも奥行きが同じなので、距離の近い方を
 * 上に重ねれば、見えている牌を指せる。
 */
export function screenHull(
  object: THREE.Object3D,
  camera: THREE.Camera,
  width: number,
  height: number,
) {
  object.updateWorldMatrix(true, true);
  worldBox.makeEmpty();
  const points: [number, number][] = [];
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const geometry = child.geometry as THREE.BufferGeometry;
    if (!geometry.boundingBox) geometry.computeBoundingBox();
    const { min, max } = geometry.boundingBox!;
    for (const x of [min.x, max.x])
      for (const y of [min.y, max.y])
        for (const z of [min.z, max.z]) {
          point.set(x, y, z).applyMatrix4(child.matrixWorld);
          worldBox.expandByPoint(point);
          point.project(camera);
          points.push([
            ((point.x + 1) * width) / 2,
            ((1 - point.y) * height) / 2,
          ]);
        }
  });
  const distance = worldBox
    .getCenter(point)
    .distanceTo(eye.setFromMatrixPosition(camera.matrixWorld));
  return { points: convexHull(points), distance };
}

// 単調鎖法で凸包を求める。返す点は時計回り（画面の y は下向き）に並ぶ。
function convexHull(points: [number, number][]): [number, number][] {
  const sorted = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (
    o: [number, number],
    a: [number, number],
    b: [number, number],
  ) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const chain = (list: [number, number][]) => {
    const hull: [number, number][] = [];
    for (const p of list) {
      while (hull.length >= 2 && cross(hull.at(-2)!, hull.at(-1)!, p) <= 0)
        hull.pop();
      hull.push(p);
    }
    return hull.slice(0, -1);
  };
  return [...chain(sorted), ...chain(sorted.reverse())];
}
