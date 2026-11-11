import * as THREE from "three";
import { createComposition, type Anchors } from "./composition.ts";
import { createLabels } from "./labels.ts";
import { createTableScene } from "./scene.ts";
import {
  handTargets,
  riverTargets,
  sameTargets,
  scoreTargets,
} from "./targets.ts";
import { loadTileTextures } from "./textures.ts";
import {
  createTileLayer,
  positionDrawArrival,
  type TileFrame,
} from "./tiles.ts";
import type {
  TableEvents,
  TableFrame,
  TableHost,
  TableRenderer,
  TableTargets,
} from "./types.ts";

/** ツモ牌が届くアニメーションの長さ（ms）。 */
const ARRIVAL_MS = 300;
/** 描画するピクセル数の上限（縦 × 横）。 */
const PIXEL_BUDGET = 12000000;

// 局面は呼び出し元から受け取ったものを書き換えずに使う。ここで持つのは GPU のリソースと表示用の canvas だけ。
export function createTableRenderer(
  host: TableHost,
  rowWidth: number,
  events: TableEvents,
): TableRenderer {
  const { root, board, device } = host;
  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    antialias: true,
    powerPreference: "low-power",
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.setClearColor(0x07121e, 1);

  const presentation = document.createElement("canvas");
  const context = presentation.getContext("2d", { alpha: false })!;
  presentation.className = "em-physical-canvas";
  presentation.setAttribute("aria-hidden", "true");
  board.prepend(presentation);
  const overlay = document.createElement("div");
  overlay.className = "em-scene-labels";
  overlay.setAttribute("aria-hidden", "true");
  board.append(overlay);

  const table = createTableScene(renderer);
  const composition = createComposition(table, rowWidth);
  const annotations = createLabels(host, table, composition);
  const textures = loadTileTextures(renderer, schedule, reportError);
  const tiles = createTileLayer(overlay, table, textures, rowWidth);
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let current: TableFrame | null = null;
  let tileFrame: TileFrame | null = null;
  let anchors: Anchors | null = null;
  let disposed = false,
    lost = false,
    raf = 0;
  let width = 0,
    height = 0,
    arrival: THREE.Group | null = null,
    arrivalStart = 0;
  let densityQuery: MediaQueryList;
  // もしもの欄の大きさが変わったら、名札を置き直して欄を避ける。
  const panelObserver = new ResizeObserver(schedule);
  let observedPanel: HTMLElement | null = null,
    emittedTargets: TableTargets | null = null;

  function reportError(error: unknown) {
    events.onError(error instanceof Error ? error : new Error(String(error)));
  }

  function resizePresentation() {
    const scale = devicePixelRatio;
    const w = Math.max(1, Math.round(width * scale));
    const h = Math.max(1, Math.round(height * scale));
    if (presentation.width !== w || presentation.height !== h) {
      presentation.width = w;
      presentation.height = h;
    }
    return scale;
  }

  function render() {
    if (arrival && reducedMotion.matches) {
      positionDrawArrival(arrival, 0);
      arrival = null;
    }
    if (arrival) {
      const progress = Math.min(
        1,
        (performance.now() - arrivalStart) / ARRIVAL_MS,
      );
      positionDrawArrival(arrival, 1 - progress);
      if (progress < 1) raf = requestAnimationFrame(render);
      else {
        arrival = null;
        // ツモ牌が届いた位置で、手牌の当たり判定を作り直す。
        emitTargets();
      }
    }
    annotations.animate();
    renderer.autoClear = true;
    renderer.render(table.scene, table.camera);
    renderer.autoClear = false;
    renderer.clearDepth();
    renderer.render(table.handScene, table.handCamera);
    renderer.autoClear = true;
    resizePresentation();
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(
      renderer.domElement,
      0,
      0,
      presentation.width,
      presentation.height,
    );
    root.dataset.ready = "true";
  }

  function writeAnchors(next: Anchors) {
    if (next === anchors) return;
    anchors = next;
    device.style.setProperty("--em-hand-top", next.handTop + "px");
    device.style.setProperty("--em-playback-top", next.playbackTop + "px");
    device.style.setProperty("--em-review-top", next.reviewTop + "px");
    device.style.setProperty("--em-hud-top", next.hudTop + "px");
  }

  function sync() {
    if (!current || lost) return;
    width = board.clientWidth;
    height = board.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(
      Math.max(3, Math.min(4, devicePixelRatio * 1.5)),
      Math.sqrt(PIXEL_BUDGET / (width * height)),
      renderer.capabilities.maxTextureSize / width,
      renderer.capabilities.maxTextureSize / height,
    );
    if (renderer.getPixelRatio() !== ratio) renderer.setPixelRatio(ratio);
    renderer.setSize(width, height, false);
    writeAnchors(
      composition.fit(
        width,
        height,
        current.layout,
        host.playback.offsetHeight,
      ),
    );
    board.setAttribute("aria-busy", String(textures.pending > 0));
    if (textures.pending) {
      const scale = resizePresentation();
      context.fillStyle = "#0b203e";
      context.fillRect(0, 0, presentation.width, presentation.height);
      context.fillStyle = "#c2d5e3";
      context.font = Math.round(14 * scale) + "px sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(
        current.labels.loadingTiles,
        presentation.width / 2,
        presentation.height / 2,
      );
      return;
    }
    const frame = tiles.update(current);
    tileFrame = frame;
    if (frame.rebuilt) {
      // 前のツモ牌がまだ届く途中なら、届いた位置に置く。その席を作り直したなら、その牌はもう消えている。
      if (arrival) positionDrawArrival(arrival, 0);
      arrival = frame.arrival;
      arrivalStart = performance.now();
    }
    for (const annotation of frame.annotations)
      if (annotation.element.textContent !== current.labels.draw)
        annotation.element.textContent = current.labels.draw;
    watchPanel();
    annotations.update(
      width,
      height,
      current.layout,
      current.uiScale,
      anchors!.handTop,
      frame.annotations,
    );
    table.paintDisplay(current);
    render();
    emitTargets();
  }

  function watchPanel() {
    const panel = host.panel();
    if (panel === observedPanel) return;
    if (observedPanel) panelObserver.unobserve(observedPanel);
    if (panel) panelObserver.observe(panel);
    observedPanel = panel;
  }

  // 当たり判定は、形が変わったときだけ送る。
  function emitTargets() {
    if (!current || !tileFrame) return;
    const targets: TableTargets | null = current.targets
      ? {
          rivers: riverTargets(tileFrame.river, table.camera, width, height),
          hands: handTargets(tileFrame.hand, table.handCamera, width, height),
          scores: scoreTargets(
            table.display,
            table.camera,
            width,
            height,
            current.viewSeat,
            current.snapshot.players.length,
          ),
        }
      : null;
    if (sameTargets(targets, emittedTargets)) return;
    emittedTargets = targets;
    events.onTargets(targets);
  }

  function schedule() {
    if (disposed) return;
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => {
      try {
        sync();
      } catch (error) {
        reportError(error);
      }
    });
  }

  function watchDensity() {
    densityQuery = matchMedia("(resolution: " + devicePixelRatio + "dppx)");
    densityQuery.addEventListener("change", densityChanged, { once: true });
  }
  function densityChanged() {
    watchDensity();
    schedule();
  }
  function contextLost(event: Event) {
    event.preventDefault();
    lost = true;
    cancelAnimationFrame(raf);
    reportError(new Error("The graphics context was lost. Reopen the table."));
  }
  function contextRestored() {
    lost = false;
    schedule();
  }

  watchDensity();
  const resizeObserver = new ResizeObserver(schedule);
  resizeObserver.observe(board);
  renderer.domElement.addEventListener("webglcontextlost", contextLost);
  renderer.domElement.addEventListener("webglcontextrestored", contextRestored);

  return {
    update(frame) {
      current = frame;
      schedule();
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      panelObserver.disconnect();
      densityQuery.removeEventListener("change", densityChanged);
      renderer.domElement.removeEventListener("webglcontextlost", contextLost);
      renderer.domElement.removeEventListener(
        "webglcontextrestored",
        contextRestored,
      );
      textures.dispose();
      table.resources.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      presentation.remove();
      overlay.remove();
      delete root.dataset.ready;
    },
  };
}
