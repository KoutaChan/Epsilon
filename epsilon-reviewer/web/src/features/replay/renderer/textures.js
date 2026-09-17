import * as THREE from "three";

const TILE_ASSETS = [
  ["E", "Ton"],
  ["S", "Nan"],
  ["W", "Shaa"],
  ["N", "Pei"],
  ["P", "Haku"],
  ["F", "Hatsu"],
  ["C", "Chun"],
];
for (const [suit, code] of [
  ["Man", "m"],
  ["Pin", "p"],
  ["Sou", "s"],
]) {
  for (let n = 1; n <= 9; n++) TILE_ASSETS.push([n + code, suit + n]);
  TILE_ASSETS.push(["0" + code, suit + "5-Dora"]);
}

// 全牌を一度だけ読み込む。途中局面への移動で画像ロードやカメラ再配置を起こさない。
export function loadTileTextures(renderer, changed, failed) {
  const materials = new Map();
  const timers = new Set();
  const loader = new THREE.TextureLoader();
  const base = new URL("./tiles/Regular/", document.baseURI);
  const anisotropy = renderer.capabilities.getMaxAnisotropy();
  let disposed = false;
  let remaining = TILE_ASSETS.length;

  for (const [tile, name] of TILE_ASSETS) {
    const url = new URL(name + ".svg", base).href;
    let settled = false;
    const timeout = setTimeout(
      () => finish(new Error("Tile texture request timed out: " + name)),
      12000,
    );
    timers.add(timeout);
    function finish(error) {
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
        const context = glyph.getContext("2d");
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
    get: (tile) => materials.get(tile.includes("r") ? "0" + tile[1] : tile),
    get pending() {
      return remaining;
    },
    dispose() {
      disposed = true;
      for (const timeout of timers) clearTimeout(timeout);
      for (const material of materials.values()) {
        material.map.dispose();
        material.dispose();
      }
      materials.clear();
    },
  };
}
