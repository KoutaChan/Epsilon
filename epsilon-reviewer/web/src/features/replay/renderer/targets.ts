import * as THREE from "three";
import { projectPoint, screenHull } from "./projection.ts";
import type { HandPiece, RiverPiece } from "./tiles.ts";
import type { ScreenPoint, TableTargets } from "./types.ts";

// もしもで押せる所を、卓と同じカメラで画面の多角形にする。DOM の当たり枠はこの形に重ねる。

const corner = new THREE.Vector3();
const rounded = ([x, y]: readonly [number, number]): ScreenPoint => [
  Math.round(x * 10) / 10,
  Math.round(y * 10) / 10,
];

/** 牌の輪郭を目から遠い牌から並べ、近い牌の当たり枠が上に重なるようにする。 */
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

/** 見ている席の手牌の輪郭。手の場面のカメラで写す。 */
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

// paintDisplay が点数を描く枠。768 四方の絵の中心から見た、席を回す前の座標（下向きの y）。
const SCORE_CORNERS = [
  [-177, 210],
  [177, 210],
  [177, 296],
  [-177, 296],
] as const;
const CANVAS = 768;
/** 中央の表示の点数の枠（4 隅）。並びは見ている席から反時計回り（自家・下家・対面・上家）。 */
export function scoreTargets(
  display: THREE.Mesh<THREE.PlaneGeometry>,
  camera: THREE.Camera,
  width: number,
  height: number,
  viewSeat: number,
): TableTargets["scores"] {
  display.updateWorldMatrix(true, false);
  const size = display.geometry.parameters.width;
  return [0, 1, 2, 3].map((offset) => {
    // 席ごとに -90° ずつ回して描いている。
    const angle = (-offset * Math.PI) / 2;
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
    return { seat: (viewSeat + offset) % 4, points };
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

/** 当たり枠が同じ形か（数で比べる）。 */
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
