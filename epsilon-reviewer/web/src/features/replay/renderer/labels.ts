import * as THREE from "three";
import type { Composition } from "./composition.ts";
import {
  local,
  projectPoint,
  screenBounds,
  tableDimensions,
  type ScreenRect,
} from "./projection.ts";
import type { TableScene } from "./scene.ts";
import { tileData, type DrawAnnotation } from "./tiles.ts";
import type { Layout, SeatSlot, TableHost } from "./types.ts";

export interface Labels {
  /** ツモ牌のラベルを牌の位置に合わせる。ツモ牌が届く動きの間は毎フレーム呼ぶ。 */
  animate(): void;
  /**
   * ツモ牌のラベルと名札を置き、候補欄の上端を決める。
   *
   * @param handTop 手前の手牌の上端（px）
   */
  update(
    width: number,
    height: number,
    layout: Layout,
    uiScale: number,
    handTop: number,
    draws: readonly DrawAnnotation[],
  ): void;
}

export function createLabels(
  host: TableHost,
  tableScene: TableScene,
  composition: Composition,
): Labels {
  const { camera, handCamera } = tableScene;
  const { cameraEnvelope, handEnvelope, rowWidth } = composition;
  const { board, device, dora: doraElement, playerLabels } = host;
  const vector = new THREE.Vector3();
  let width = 0,
    height = 0,
    annotations: readonly DrawAnnotation[] = [],
    mode: Layout = "desktop";
  const project = (p: THREE.Vector3, cam: THREE.Camera = camera) =>
    projectPoint(p, cam, width, height);
  function drawLabelPosition(
    rect: ScreenRect,
    seat: SeatSlot,
    labelWidth: number,
    labelHeight: number,
  ) {
    let x = rect.x + rect.w / 2,
      y = rect.y + rect.h + 5;
    // ラベルはツモ牌の手前に置き、同じ列の副露から離す。
    if (seat === "north") {
      x = rect.x - labelWidth / 2 - 5;
      y = rect.y + (rect.h - labelHeight) / 2;
    } else if (seat === "east") {
      y = rect.y - labelHeight - 5;
    }
    return {
      x: Math.max(labelWidth / 2 + 3, Math.min(width - labelWidth / 2 - 3, x)),
      y: Math.max(3, Math.min(height - labelHeight - 3, y)),
    };
  }
  function layoutAnnotations() {
    for (const a of annotations) {
      const { seat, foreground } = tileData(a.group);
      const rect = screenBounds(
        [a.group],
        foreground ? handCamera : camera,
        width,
        height,
      );
      const p = drawLabelPosition(
        rect,
        seat,
        a.element.offsetWidth,
        a.element.offsetHeight,
      );
      a.element.style.left = p.x + "px";
      a.element.style.top = p.y + "px";
    }
  }
  function layoutPlayerLabels(u: number, handTop: number) {
    const bounds = (objects: readonly THREE.Object3D[], cam = camera) =>
      screenBounds(objects, cam, width, height);
    const boxes: Record<SeatSlot, ScreenRect> = {
      south: bounds([handEnvelope], handCamera),
      north: bounds([cameraEnvelope.children[1]!]),
      west: bounds([cameraEnvelope.children[2]!]),
      east: bounds([cameraEnvelope.children[3]!]),
    };
    const hostRect = board.getBoundingClientRect();
    const domBox = (el: Element): ScreenRect => {
      const b = el.getBoundingClientRect();
      return {
        x: b.x - hostRect.x,
        y: b.y - hostRect.y,
        w: b.width,
        h: b.height,
      };
    };
    // 名札が避ける場所は、今の牌の枚数によらず、河とツモ枠が最も広がったときの範囲で取る。
    const projectedBox = (
      w: number,
      h: number,
      d: number,
      seat: SeatSlot,
      x: number,
      y: number,
      z: number,
      cam: THREE.Camera = camera,
    ): ScreenRect => {
      let left = Infinity,
        top = Infinity,
        right = -Infinity,
        bottom = -Infinity;
      for (const dx of [-w / 2, w / 2])
        for (const dy of [-h / 2, h / 2])
          for (const dz of [0, d]) {
            const q = local(seat, x + dx, y + dy);
            const p = project(vector.set(q.x, q.y, z + dz), cam);
            left = Math.min(left, p.x);
            right = Math.max(right, p.x);
            top = Math.min(top, p.y);
            bottom = Math.max(bottom, p.y);
          }
      return { x: left, y: top, w: right - left, h: bottom - top };
    };
    const dimensions = tableDimensions(mode, rowWidth),
      blocked: ScreenRect[] = Object.values(boxes);
    for (const seat of ["south", "east", "north", "west"] as const) {
      for (let i = 0; i < 18; i++)
        blocked.push(
          projectedBox(
            dimensions.riverWidth,
            (dimensions.riverWidth * 4) / 3,
            0.61 * (mode === "landscape" ? 1.4 : 1),
            seat,
            dimensions.riverStart + (i % 6) * dimensions.pitchX,
            -dimensions.riverRadius - Math.floor(i / 6) * dimensions.pitchY,
            0.085,
          ),
        );
      if (seat !== "south")
        for (const lift of [0, 0.5]) {
          const b = projectedBox(
            dimensions.opponentScale,
            0.61 * dimensions.opponentScale,
            (4 / 3) * dimensions.opponentScale,
            seat,
            7.058 * dimensions.opponentScale,
            -dimensions.opponentRadius,
            0.085 + lift,
          );
          const p = drawLabelPosition(b, seat, 42, 22);
          blocked.push({ x: p.x - 21, y: p.y, w: 42, h: 22 });
        }
    }
    blocked.push({
      x: boxes.south.x,
      y: boxes.south.y - 29,
      w: boxes.south.w,
      h: 29,
    });
    for (const { element } of annotations) blocked.push(domBox(element));
    blocked.push(domBox(doraElement));
    for (const element of host.controls) blocked.push(domBox(element));
    // もしもの欄は開いている間しか無いので、配置のたびに探す。
    const panel = host.panel();
    if (panel) blocked.push(domBox(panel));
    const intersects = (a: ScreenRect, b: ScreenRect) =>
      a.x < b.x + b.w + 2 &&
      a.x + a.w > b.x - 2 &&
      a.y < b.y + b.h + 2 &&
      a.y + a.h > b.y - 2;
    const dora = domBox(doraElement),
      playback = domBox(host.playback);
    for (const seat of ["north", "west", "east", "south"] as const) {
      const el = playerLabels[seat],
        b = boxes[seat],
        vertical = mode === "portrait" && (seat === "west" || seat === "east"),
        labelW = vertical
          ? 12
          : mode === "desktop"
            ? (seat === "north" ? 110 : 150) * u
            : mode === "portrait" && seat === "north"
              ? 80
              : mode === "portrait"
                ? 100
                : seat === "north"
                  ? 90
                  : 108,
        labelH = vertical
          ? 92
          : mode === "desktop"
            ? 26 * u
            : mode === "landscape" && seat === "south"
              ? 18
              : 20;
      el.style.width = labelW + "px";
      el.style.height = labelH + "px";
      let targets: (readonly [number, number])[] = [];
      if (seat === "north")
        targets =
          mode === "portrait"
            ? [
                [playback.x + 4, playback.y + 52],
                [8, b.y - 24],
                [width - labelW - 8, b.y - 24],
              ]
            : [
                [b.x + b.w / 2 - labelW / 2, b.y - labelH - 8],
                [
                  Math.max(dora.x + dora.w + 8, b.x - labelW - 24),
                  b.y + b.h / 2 - labelH / 2,
                ],
                [b.x + b.w + 32, b.y + b.h / 2 - labelH / 2],
                [width / 2 - labelW / 2, b.y + b.h + 8],
                [width - labelW - 8, 8],
                [dora.x + dora.w + 8, 8],
              ];
      if (seat === "west")
        targets =
          mode === "portrait"
            ? [
                [8, b.y - 34],
                [8, b.y + b.h + 8],
                [8, b.y - 50],
              ]
            : [
                ...(mode === "landscape"
                  ? [[8, dora.y + dora.h + 8] as const]
                  : []),
                [b.x - labelW - 10, b.y + b.h / 2 - labelH / 2],
                [b.x - labelW - 10, b.y + b.h * 0.75 - labelH / 2],
                [b.x - labelW - 10, b.y + b.h * 0.25 - labelH / 2],
              ];
      if (seat === "east")
        targets =
          mode === "portrait"
            ? [
                [width - labelW - 8, b.y - 12],
                [width - labelW - 8, b.y + b.h + 8],
                [width - labelW - 8, b.y - 50],
              ]
            : [
                [b.x + b.w + 10, b.y + b.h / 2 - labelH / 2],
                [b.x + b.w + 10, b.y + b.h * 0.75 - labelH / 2],
                [b.x + b.w + 10, b.y + b.h * 0.25 - labelH / 2],
                // 右がもしもの欄でふさがっているときは、手牌の列の下か上に置く。
                [b.x + b.w / 2 - labelW / 2, b.y + b.h + 8],
                [b.x + b.w / 2 - labelW / 2, b.y - labelH - 8],
              ];
      if (seat === "south")
        targets = [
          [b.x, b.y + b.h + 4],
          [b.x + labelW + 24, b.y + b.h + 4],
          [width / 2 - labelW / 2, b.y + b.h + 4],
        ];
      if (vertical)
        targets = [
          [seat === "west" ? 2 : width - 14, b.y + b.h / 2 - labelH / 2],
        ];
      const candidates = targets
        .flatMap(([x, y]) =>
          [0, -8, 8, -16, 16, -24, 24].map((dy) => [x, y + dy] as const),
        )
        .map(([x, y]): ScreenRect => ({
          x: Math.max(
            vertical ? 2 : 6,
            Math.min(width - labelW - (vertical ? 2 : 6), x),
          ),
          y: Math.max(4, Math.min(height - labelH - 4, y)),
          w: labelW,
          h: labelH,
        }));
      const chosen =
        candidates.find((p) => !blocked.some((b) => intersects(p, b))) ||
        candidates[0]!;
      el.style.left = chosen.x + "px";
      el.style.top = chosen.y + "px";
      blocked.push(chosen);
    }
    // 候補欄は上家（左の席）の名札より下に置き、卓の投影は動かさない。横長の小さな画面で手牌の上に十分な高さが
    // 無いときも、見出しが画面の上で切れないようにする。
    if (mode !== "portrait") {
      const name = domBox(playerLabels.west),
        top =
          mode === "desktop"
            ? Math.max(height * 0.35, name.y + name.h + 12 * u)
            : Math.max(
                4,
                Math.min(
                  name.y + name.h + 6,
                  handTop - (host.candidates()?.offsetHeight ?? 0) - 8,
                ),
              );
      device.style.setProperty("--em-action-candidates-top", top + "px");
    }
  }
  return {
    animate: layoutAnnotations,
    update(w, h, layout, uiScale, handTop, draws) {
      width = w;
      height = h;
      mode = layout;
      annotations = draws;
      layoutAnnotations();
      layoutPlayerLabels(uiScale, handTop);
    },
  };
}
