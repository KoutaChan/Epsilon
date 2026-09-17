import * as THREE from "three";
import {
  local,
  tableDimensions,
  screenBounds,
  projectPoint,
} from "./projection.js";

export function createLabels(root, board, tableScene, composition) {
  const { camera, handCamera } = tableScene;
  const { cameraEnvelope, handEnvelope, rowWidth } = composition;
  const device = root.querySelector(".em-device");
  const doraElement = root.querySelector(".em-dora-hud");
  const playbackElement = root.querySelector(".em-playback");
  const playerElements = Object.fromEntries(
    ["south", "east", "north", "west"].map((seat) => [
      seat,
      root.querySelector("[data-player-label=" + seat + "]"),
    ]),
  );
  const controlElements = root.querySelectorAll(
    ".em-dora-hud,.em-transport>button,.em-round-trigger,.em-seat-trigger,.em-info>summary",
  );
  const vector = new THREE.Vector3();
  let width = 0,
    height = 0,
    annotations = [],
    mode;
  const project = (p, cam = camera) => projectPoint(p, cam, width, height);
  function drawLabelPosition(rect, seat, labelWidth, labelHeight) {
    let x = rect.x + rect.w / 2,
      y = rect.y + rect.h + 5;
    // ツモ牌の手前へ置き、同じ列の副露とラベルを離す。
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
      const rect = screenBounds(
        [a.g],
        a.g.userData.foreground ? handCamera : camera,
        width,
        height,
      );
      const p = drawLabelPosition(
        rect,
        a.g.userData.seat,
        a.e.offsetWidth,
        a.e.offsetHeight,
      );
      a.e.style.left = p.x + "px";
      a.e.style.top = p.y + "px";
    }
  }
  function layoutPlayerLabels() {
    const u = Number(device.style.getPropertyValue("--em-pc-ui-scale"));
    const bounds = (objects, cam = camera) =>
      screenBounds(objects, cam, width, height);
    const boxes = {
      south: bounds([handEnvelope], handCamera),
      north: bounds([cameraEnvelope.children[1]]),
      west: bounds([cameraEnvelope.children[2]]),
      east: bounds([cameraEnvelope.children[3]]),
    };
    const host = board.getBoundingClientRect();
    const domBox = (el) => {
      const b = el.getBoundingClientRect();
      return {
        x: b.x - host.x,
        y: b.y - host.y,
        w: b.width,
        h: b.height,
      };
    };
    // 名札の位置は実際の残り牌数に依存させず、最大の河とツモ枠を予約する。
    const projectedBox = (w, h, d, seat, x, y, z, cam = camera) => {
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
      blocked = Object.values(boxes);
    for (const seat of ["south", "east", "north", "west"]) {
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
    for (const { e } of annotations) blocked.push(domBox(e));
    for (const element of controlElements) blocked.push(domBox(element));
    const intersects = (a, b) =>
      a.x < b.x + b.w + 2 &&
      a.x + a.w > b.x - 2 &&
      a.y < b.y + b.h + 2 &&
      a.y + a.h > b.y - 2;
    const dora = domBox(doraElement),
      playback = domBox(playbackElement);
    for (const seat of ["north", "west", "east", "south"]) {
      const el = playerElements[seat],
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
      let targets;
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
                ...(mode === "landscape" ? [[8, dora.y + dora.h + 8]] : []),
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
        .flatMap(([x, y]) => [
          [x, y],
          [x, y - 8],
          [x, y + 8],
          [x, y - 16],
          [x, y + 16],
          [x, y - 24],
          [x, y + 24],
        ])
        .map(([x, y]) => ({
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
        candidates[0];
      el.style.left = chosen.x + "px";
      el.style.top = chosen.y + "px";
      blocked.push(chosen);
    }
    // 候補欄は左家の名札より下へ置き、卓の投影は変えない。
    if (mode !== "portrait") {
      const name = domBox(playerElements.west),
        top =
          mode === "desktop"
            ? Math.max(height * 0.35, name.y + name.h + 12 * u)
            : Math.min(
                name.y + name.h + 6,
                Number.parseFloat(
                  device.style.getPropertyValue("--em-hand-top"),
                ) -
                  root.querySelector(".em-action-candidates").offsetHeight -
                  8,
              );
      root.style.setProperty("--em-action-candidates-top", top + "px");
    }
  }
  return {
    animate: layoutAnnotations,
    update(w, h, layout, draws) {
      width = w;
      height = h;
      mode = layout;
      annotations = draws;
      layoutAnnotations();
      layoutPlayerLabels();
    },
  };
}
