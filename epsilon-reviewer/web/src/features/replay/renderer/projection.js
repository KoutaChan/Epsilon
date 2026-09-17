import * as THREE from "three";
import { PLAYER_ROW_WIDTH, PLAYER_ROW_CENTER } from "./tileLayout.js";
const angles = {
  south: 0,
  east: Math.PI / 2,
  north: Math.PI,
  west: -Math.PI / 2,
};
const worldBox = new THREE.Box3();
const point = new THREE.Vector3();
export function tableDimensions(mode, rowWidth = PLAYER_ROW_WIDTH) {
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
export function local(seat, x, y) {
  const a = angles[seat];
  return {
    x: x * Math.cos(a) - y * Math.sin(a),
    y: x * Math.sin(a) + y * Math.cos(a),
    angle: a,
  };
}
export function pointBounds(objects, cam) {
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

export function screenBounds(objects, camera, width, height) {
  const b = pointBounds(objects, camera);
  return {
    x: ((b.minX + 1) * width) / 2,
    y: ((1 - b.maxY) * height) / 2,
    w: ((b.maxX - b.minX) * width) / 2,
    h: ((b.maxY - b.minY) * height) / 2,
  };
}
export function projectPoint(p, camera, width, height) {
  point.copy(p).project(camera);
  return { x: ((point.x + 1) * width) / 2, y: ((1 - point.y) * height) / 2 };
}
