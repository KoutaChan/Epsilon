import * as THREE from "three";
import { TILE_CODES } from "@epsilon-reviewer/protocol/tiles";
import { tileAsset } from "../../../components/tileAssets.ts";

/** 牌の画像 1 枚の読み込みを待つ時間（ms）。 */
const TEXTURE_TIMEOUT_MS = 12000;

export interface TileTextures {
  /** 牌の記号に対応する、画像のマテリアルを返す。全ての画像を読み終えた後にだけ呼ぶこと。 */
  get(tile: string): THREE.MeshBasicMaterial;
  readonly pending: number;
  dispose(): void;
}

// 全ての牌の画像を最初に一度だけ読み込む。途中の局面へ移っても、画像を読み直したりカメラを置き直したりしない。
export function loadTileTextures(
  renderer: THREE.WebGLRenderer,
  changed: () => void,
  failed: (error: Error) => void,
): TileTextures {
  const materials = new Map<string, THREE.MeshBasicMaterial>();
  const timers = new Set<ReturnType<typeof setTimeout>>();
  const loader = new THREE.TextureLoader();
  const anisotropy = renderer.capabilities.getMaxAnisotropy();
  let disposed = false;
  let remaining = TILE_CODES.length;

  for (const tile of TILE_CODES) {
    const url = new URL(tileAsset(tile), document.baseURI).href;
    let settled = false;
    const timeout = setTimeout(
      () => finish(new Error("Tile texture request timed out: " + tile)),
      TEXTURE_TIMEOUT_MS,
    );
    timers.add(timeout);
    function finish(error?: Error) {
      if (settled || disposed) return;
      settled = true;
      clearTimeout(timeout);
      timers.delete(timeout);
      if (error) failed(error);
      else {
        remaining--;
        if (remaining === 0) changed();
      }
    }
    loader.load(
      url,
      (texture) => {
        if (disposed || settled) {
          texture.dispose();
          return;
        }
        const glyph = document.createElement("canvas");
        glyph.width = 768;
        glyph.height = 1024;
        const context = glyph.getContext("2d")!;
        context.imageSmoothingEnabled = true;
        context.imageSmoothingQuality = "high";
        context.drawImage(texture.image, 0, 0, glyph.width, glyph.height);
        texture.image = glyph;
        texture.needsUpdate = true;
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = anisotropy;
        materials.set(
          tile,
          new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            alphaTest: 0.01,
            depthWrite: false,
            toneMapped: false,
            polygonOffset: true,
            polygonOffsetFactor: -2,
          }),
        );
        finish();
      },
      undefined,
      () => finish(new Error("Tile texture could not be loaded: " + tile)),
    );
  }
  return {
    get: (tile) => materials.get(tile)!,
    get pending() {
      return remaining;
    },
    dispose() {
      disposed = true;
      for (const timeout of timers) clearTimeout(timeout);
      for (const material of materials.values()) {
        material.map!.dispose();
        material.dispose();
      }
      materials.clear();
    },
  };
}
