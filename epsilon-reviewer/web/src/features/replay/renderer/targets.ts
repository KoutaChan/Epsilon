import * as THREE from "three";
import { projectPoint, screenHull } from "./projection.ts";
import type { HandPiece, RiverPiece } from "./tiles.ts";
import type { ScreenPoint, TableTargets } from "./types.ts";
import { displaySeats } from "../model/seats.ts";

// もしもで押せる所を卓と同じカメラで写し、画面上の多角形にする。DOM の当たり判定はこの形に合わせて重ねる。

const corner = new THREE.Vector3();
const rounded = ([x, y]: readonly [number, number]): ScreenPoint => [
  Math.round(x * 10) / 10,
  Math.round(y * 10) / 10,
];

/** 牌の輪郭をカメラから遠い順に並べる。近い牌の当たり判定が上に重なる。 */
function outlines<P extends { readonly group: THREE.Object3D }>(
  pieces: readonly P[],
  camera: THREE.Camera,
  width: number,
  height: number,
) {
  return pieces
    .map((piece) => ({
      piece,
      ...screenHull(piece.group, camera, width, height),
    }))
    .sort((a, b) => b.distance - a.distance)
    .map(({ piece, points }) => ({ piece, points: points.map(rounded) }));
}

export function riverTargets(
  pieces: readonly RiverPiece[],
  camera: THREE.Camera,
  width: number,
  height: number,
): TableTargets["rivers"] {
  return outlines(pieces, camera, width, height).map(({ piece, points }) => ({
    seat: piece.seat,
    index: piece.index,
    points,
  }));
}

/** 見ている席の手牌の輪郭。手牌用のカメラで写す。 */
export function handTargets(
  pieces: readonly HandPiece[],
  handCamera: THREE.Camera,
  width: number,
  height: number,
): TableTargets["hands"] {
  return outlines(pieces, handCamera, width, height).map(
    ({ piece, points }) => ({ index: piece.index, points }),
  );
}

// 中央の表示で点数を描く枠の 4 隅（表示を描くときと同じ位置）。座標は 768 px 四方の canvas の中心が原点で、席の向きに回す前のもの（y は下向き）。
const SCORE_CORNERS = [
  [-177, 210],
  [177, 210],
  [177, 296],
  [-177, 296],
] as const;
const CANVAS = 768;
/**
 * 卓の中央の表示にある、席ごとの点数の枠（4 隅）。並びは見ている席から反時計回り（自家・下家・対面・上家）。三人打ちでは対面の枠が無い。
 */
export function scoreTargets(
  display: THREE.Mesh<THREE.PlaneGeometry>,
  camera: THREE.Camera,
  width: number,
  height: number,
  viewSeat: number,
  seats: number,
): TableTargets["scores"] {
  display.updateWorldMatrix(true, false);
  const size = display.geometry.parameters.width;
  return displaySeats(seats).flatMap((offset, position) => {
    if (offset === null) return [];
    // 点数は表示の位置ごとに -90° ずつ回して描いてある。
    const angle = (-position * Math.PI) / 2;
    const points = SCORE_CORNERS.map(([x, y]) => {
      const cx = x * Math.cos(angle) - y * Math.sin(angle);
      const cy = x * Math.sin(angle) + y * Math.cos(angle);
      corner.set((cx / CANVAS) * size, (-cy / CANVAS) * size, 0);
      const p = projectPoint(
        display.localToWorld(corner),
        camera,
        width,
        height,
      );
      return rounded([p.x, p.y]);
    });
    return [{ seat: (viewSeat + offset) % seats, points }];
  });
}

function samePoints(
  a: readonly ScreenPoint[],
  b: readonly ScreenPoint[],
): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index++) {
    const p = a[index]!,
      q = b[index]!;
    if (p[0] !== q[0] || p[1] !== q[1]) return false;
  }
  return true;
}

function sameShapes<S extends { readonly points: readonly ScreenPoint[] }>(
  a: readonly S[],
  b: readonly S[],
  sameKey: (left: S, right: S) => boolean,
): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index++) {
    const left = a[index]!,
      right = b[index]!;
    if (!sameKey(left, right) || !samePoints(left.points, right.points))
      return false;
  }
  return true;
}

/** 2 つの当たり判定が同じか。指す牌や席と、座標の数値で比べる。 */
export function sameTargets(
  a: TableTargets | null,
  b: TableTargets | null,
): boolean {
  if (a === null || b === null) return a === b;
  return (
    sameShapes(
      a.rivers,
      b.rivers,
      (left, right) => left.seat === right.seat && left.index === right.index,
    ) &&
    sameShapes(a.hands, b.hands, (left, right) => left.index === right.index) &&
    sameShapes(a.scores, b.scores, (left, right) => left.seat === right.seat)
  );
}
