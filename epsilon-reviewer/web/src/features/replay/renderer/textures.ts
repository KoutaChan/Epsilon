import * as THREE from "three";
import { ASSET_TILES, tileAssetName } from "../tileAssets.ts";

/** 1 枚の絵を待つ時間。 */
const TEXTURE_TIMEOUT_MS = 12000;

export interface TileTextures {
  /** 牌の記号の絵の材質。全ての絵を読み終えた後だけ呼ぶ。 */
  get(tile: string): THREE.MeshBasicMaterial;
  readonly pending: number;
  dispose(): void;
}

// 全牌を一度だけ読み込む。途中局面への移動で画像ロードやカメラ再配置を起こさない。
export function loadTileTextures(
  renderer: THREE.WebGLRenderer,
  changed: () => void,
  failed: (error: Error) => void,
): TileTextures {
  const materials = new Map<string, THREE.MeshBasicMaterial>();
  const timers = new Set<ReturnType<typeof setTimeout>>();
  const loader = new THREE.TextureLoader();
  const base = new URL("./tiles/Regular/", document.baseURI);
  const anisotropy = renderer.capabilities.getMaxAnisotropy();
  let disposed = false;
  let remaining = ASSET_TILES.length;

  for (const tile of ASSET_TILES) {
    const name = tileAssetName(tile);
    const url = new URL(name + ".svg", base).href;
    let settled = false;
    const timeout = setTimeout(
      () => finish(new Error("Tile texture request timed out: " + name)),
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
      () => finish(new Error("Tile texture could not be loaded: " + name)),
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
