import * as THREE from "three";
import { createTableScene } from "./renderer/scene.js";
import { loadTileTextures } from "./renderer/textures.js";
import { createTileLayer, positionDrawArrival } from "./renderer/tiles.js";
import { createComposition } from "./renderer/composition.js";
import { createLabels } from "./renderer/labels.js";

// 局面DTOは借用し、GPU資源と表示フレームだけをこのインスタンスが所有する。
/** @param {HTMLElement} root @param {number} rowWidth @returns {import("./renderer/types").TableRenderer} */
export function createTableRenderer(root, rowWidth) {
  const board = root.querySelector(".em-board");
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
  const context = presentation.getContext("2d", { alpha: false });
  presentation.className = "em-physical-canvas";
  presentation.setAttribute("aria-hidden", "true");
  board.prepend(presentation);
  const overlay = document.createElement("div");
  overlay.className = "em-scene-labels";
  overlay.setAttribute("aria-hidden", "true");
  board.append(overlay);

  const table = createTableScene(renderer);
  const composition = createComposition(root, table, rowWidth);
  const annotations = createLabels(root, board, table, composition);
  const textures = loadTileTextures(renderer, schedule, reportError);
  const tiles = createTileLayer(overlay, table, textures, rowWidth);
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  let current, built, tileFrame;
  let disposed = false,
    lost = false,
    raf = 0;
  let width = 0,
    height = 0,
    arrival = null,
    arrivalStart = 0;
  let densityQuery;

  function reportError(error) {
    root.dispatchEvent(
      new CustomEvent("review-render-error", { detail: error.message }),
    );
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
      const progress = Math.min(1, (performance.now() - arrivalStart) / 300);
      positionDrawArrival(arrival, 1 - progress);
      if (progress < 1) raf = requestAnimationFrame(render);
      else arrival = null;
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

  function sync() {
    if (!current || lost) return;
    width = board.clientWidth;
    height = board.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(
      Math.max(3, Math.min(4, devicePixelRatio * 1.5)),
      Math.sqrt(12000000 / (width * height)),
      renderer.capabilities.maxTextureSize / width,
      renderer.capabilities.maxTextureSize / height,
    );
    if (renderer.getPixelRatio() !== ratio) renderer.setPixelRatio(ratio);
    renderer.setSize(width, height, false);
    composition.fit(width, height, current.layout);
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
    if (
      !built ||
      built.snapshot !== current.snapshot ||
      built.event !== current.event ||
      built.viewSeat !== current.viewSeat ||
      built.reveal !== current.reveal ||
      built.layout !== current.layout
    ) {
      tileFrame = tiles.build(current);
      arrival = tileFrame.arrival;
      arrivalStart = performance.now();
      built = current;
    } else {
      for (const annotation of tileFrame.annotations)
        annotation.e.textContent = current.labels.draw;
    }
    annotations.update(width, height, current.layout, tileFrame.annotations);
    table.paintDisplay(current);
    render();
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
  function contextLost(event) {
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
