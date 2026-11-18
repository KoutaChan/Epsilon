import { TILE_BASE_Z, TILE_DEPTH, LANDSCAPE_RIVER_SCALE, SURFACE_REFERENCE_RADIUS, DESKTOP_REFERENCE_WIDTH } from "./tableMetrics.ts";
import * as THREE from "three";
import {
  local,
  pointBounds,
  tableDimensions,
  type Bounds,
} from "./projection.ts";
import type { TableScene } from "./scene.ts";
import { PLAYER_ROW_CENTER, PLAYER_ROW_WIDTH } from "./tileLayout.ts";
import type { Layout, SeatSlot } from "./types.ts";

/** 卓を画面に収めた後の各部の位置。それぞれ .em-device の CSS 変数（px）に書く。 */
export interface Anchors {
  /** --em-hand-top: 手前の手牌の上端。 */
  readonly handTop: number;
  /** --em-playback-top: 操作バーの上端。 */
  readonly playbackTop: number;
  /** --em-review-top: 縦長の画面で、卓の下の一覧の上端。 */
  readonly reviewTop: number;
  /** --em-hud-top: ドラ表示ともしもの欄の上端。 */
  readonly hudTop: number;
}

export interface Composition {
  readonly cameraEnvelope: Readonly<Record<SeatSlot, THREE.Mesh>>;
  readonly handEnvelope: THREE.Mesh;
  readonly rowWidth: number;
  /** 卓と手牌のカメラを画面に収め、各部の位置を返す。画面の大きさ・配置・操作バーの高さが前と同じなら、何もせず前の位置を返す。 */
  fit(
    width: number,
    height: number,
    mode: Layout,
    playbackHeight: number,
  ): Anchors;
}

export function createComposition(
  tableScene: TableScene,
  rowWidth = PLAYER_ROW_WIDTH,
): Composition {
  const {
    camera,
    handCamera,
    landscapeFrame,
    feltSurface,
    fabricTex,
    centerGroup,
  } = tableScene;
  const { keep } = tableScene.resources;
  const riverEdge = new THREE.Vector3();
  let width = 0,
    height = 0;
  let fitted: {
    width: number;
    height: number;
    mode: Layout;
    playbackHeight: number;
    anchors: Anchors;
  } | null = null;
  // 各席の河・手牌・加槓牌・ツモ牌を収める箱。
  const geometry = keep(new THREE.BoxGeometry(1, 1, 1));
  const envelope = () => new THREE.Mesh(geometry, tableScene.materials.black);
  const cameraEnvelope = {
    south: envelope(), east: envelope(), north: envelope(), west: envelope(),
  };
  const handEnvelope = new THREE.Mesh(
    keep(new THREE.BoxGeometry(rowWidth, 1.78, 1.48)),
    tableScene.materials.black,
  );
  handEnvelope.position.set(PLAYER_ROW_CENTER, -0.2, 0.74 + TILE_BASE_Z);
  function positionProjection(
    cam: THREE.PerspectiveCamera,
    b: Bounds,
    top: number,
    shiftLeft = 0,
  ) {
    cam.projectionMatrix.elements[8] =
      (b.minX + b.maxX) / 2 + (2 * shiftLeft) / width;
    cam.projectionMatrix.elements[9] = b.maxY - (1 - (2 * top) / height);
    cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
  }
  function fitComposition(
    w: number,
    h: number,
    mode: Layout,
    playbackHeight: number,
  ): Anchors {
    width = w;
    height = h;
    landscapeFrame.visible = mode !== "portrait";
    // 手牌と副露を一列に並べても卓の縁にかからないよう、牌の列の幅に合わせて卓面を広げる。
    const tableExpansion =
      tableDimensions(mode, rowWidth).opponentRadius / SURFACE_REFERENCE_RADIUS;
    landscapeFrame.scale.set(tableExpansion, tableExpansion, 1);
    feltSurface.position.y = -6 * tableExpansion;
    feltSurface.scale.set(
      mode === "portrait" ? 4 : tableExpansion,
      mode === "portrait" ? 6 : tableExpansion,
      1,
    );
    fabricTex.repeat.set(
      mode === "portrait" ? 12 : 3,
      mode === "portrait" ? 18 : 3,
    );
    const dimensions = tableDimensions(mode, rowWidth),
      centerScale = dimensions.centerWidth / 5.65;
    centerGroup.scale.set(centerScale, centerScale, 1);
    const s = dimensions.opponentScale,
      riverDepth = TILE_DEPTH * (mode === "landscape" ? LANDSCAPE_RIVER_SCALE : 1),
      front = cameraEnvelope.south;
    front.scale.set(
      5 * dimensions.pitchX + dimensions.riverWidth,
      2 * dimensions.pitchY + (4 / 3) * dimensions.riverWidth,
      riverDepth,
    );
    front.position.set(
      dimensions.riverStart + 2.5 * dimensions.pitchX,
      -dimensions.riverRadius - dimensions.pitchY,
      TILE_BASE_Z + riverDepth / 2,
    );
    (["north", "west", "east"] as const).forEach((seat) => {
      const m = cameraEnvelope[seat],
        p = local(seat, PLAYER_ROW_CENTER * s, -dimensions.opponentRadius),
        depth = (4 / 3) * s + 0.5;
      m.scale.set(rowWidth * s, (4 / 3) * s, depth);
      m.rotation.z = p.angle;
      m.position.set(p.x, p.y, TILE_BASE_Z + depth / 2);
    });
    const elevation =
      ((mode === "portrait" ? 68 : mode === "landscape" ? 44 : 48) * Math.PI) /
      180;
    const handElevation = (40 * Math.PI) / 180,
      handTarget = new THREE.Vector3(0.22, 0, 0.72);
    camera.fov = mode === "landscape" ? 28 : 55;
    camera.aspect = handCamera.aspect = width / height;
    let distance = 18,
      handDistance = 14,
      tableShift = 0,
      b!: Bounds,
      hb!: Bounds;
    const topSafe =
      mode === "portrait"
        ? 98
        : mode === "landscape"
          ? 28
          : Math.max(28, (26 * width) / DESKTOP_REFERENCE_WIDTH + 8);
    const bottomSafe =
      mode === "portrait" ? 300 : mode === "landscape" ? 26 : 40;
    const between =
      mode === "portrait"
        ? 21 + Math.max(8, Math.min(14, width * 0.025))
        : playbackHeight + (mode === "landscape" ? 12 : 16);
    const availableHandWidth = width - (mode === "desktop" ? width * 0.1 : 24),
      nearRiverY = -dimensions.riverRadius - 2 * dimensions.pitchY,
      riverFaceZ = TILE_BASE_Z + TILE_DEPTH * (mode === "landscape" ? LANDSCAPE_RIVER_SCALE : 1);
    const setTable = () => {
      camera.position.set(
        0,
        -1 - Math.cos(elevation) * distance,
        Math.sin(elevation) * distance,
      );
      camera.lookAt(0, -1, 0);
      camera.updateMatrixWorld();
      camera.updateProjectionMatrix();
      return pointBounds(Object.values(cameraEnvelope), camera);
    };
    const setHand = () => {
      handCamera.position.set(
        0.22,
        -Math.cos(handElevation) * handDistance,
        0.72 + Math.sin(handElevation) * handDistance,
      );
      handCamera.lookAt(handTarget);
      handCamera.updateMatrixWorld();
      handCamera.updateProjectionMatrix();
      return pointBounds([handEnvelope], handCamera);
    };
    // 卓と手牌のカメラを一緒に合わせて収める。手前の牌の大きさは河の牌から決め、画面が大きいだけでは大きくしない。
    for (let i = 0; i < 400; i++) {
      b = setTable();
      const riverLeft = riverEdge
        .set(-dimensions.riverWidth / 2, nearRiverY, riverFaceZ)
        .project(camera).x;
      const riverRight = riverEdge
        .set(dimensions.riverWidth / 2, nearRiverY, riverFaceZ)
        .project(camera).x;
      const riverPixels = ((riverRight - riverLeft) * width) / 2;
      // 手前の手牌は自分の河より一回り大きい牌幅にして、立てた牌を読みやすくする。
      const handPixels = Math.min(
        availableHandWidth,
        rowWidth * Math.max(40, riverPixels * 1.37),
      );
      const handWidth = (handPixels * 2) / width;
      hb = setHand();
      for (let adjustment = 0; adjustment < 6; adjustment++) {
        const projectedWidth = hb.maxX - hb.minX;
        if (Math.abs(projectedWidth - handWidth) * width < 0.1) break;
        handDistance *= projectedWidth / handWidth;
        hb = setHand();
      }
      if (mode === "landscape") {
        const west = pointBounds([cameraEnvelope.west], camera);
        const centeredLeft =
          ((west.minX + 1 - (b.minX + b.maxX) / 2) * width) / 2;
        tableShift = Math.min(6, centeredLeft - 156);
      }
      const total =
        ((b.maxY - b.minY + hb.maxY - hb.minY) * height) / 2 + between;
      if (
        b.maxX - b.minX <= 1.91 &&
        total <= height - topSafe - bottomSafe &&
        (mode !== "landscape" ||
          ((1 + (b.maxX - b.minX) / 2) * width) / 2 - tableShift <= width - 8)
      )
        break;
      distance += 0.25;
    }
    const tableHeight = ((b.maxY - b.minY) * height) / 2,
      handHeight = ((hb.maxY - hb.minY) * height) / 2,
      total = tableHeight + between + handHeight;
    const tableTop =
      mode === "landscape"
        ? topSafe
        : topSafe + Math.max(0, (height - topSafe - bottomSafe - total) / 2);
    positionProjection(camera, b, tableTop, tableShift);
    positionProjection(handCamera, hb, tableTop + tableHeight + between);
    return {
      handTop: tableTop + tableHeight + between,
      playbackTop: tableTop + tableHeight + (mode === "desktop" ? 8 : 4),
      reviewTop: tableTop + total + 44,
      hudTop: mode === "portrait" ? Math.max(16, tableTop - 88) : 16,
    };
  }
  return {
    cameraEnvelope,
    handEnvelope,
    rowWidth,
    fit(w, h, mode, playbackHeight) {
      if (
        fitted?.width !== w ||
        fitted.height !== h ||
        fitted.mode !== mode ||
        fitted.playbackHeight !== playbackHeight
      )
        fitted = {
          width: w,
          height: h,
          mode,
          playbackHeight,
          anchors: fitComposition(w, h, mode, playbackHeight),
        };
      return fitted.anchors;
    },
  };
}
