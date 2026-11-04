import * as THREE from "three";

export interface Disposable {
  dispose(): void;
}

export interface Resources {
  /** 破棄をこの持ち主に任せる。 */
  keep<T extends Disposable>(resource: T): T;
  /** 角の丸い箱。形は寸法ごとに 1 つを使い回し、影を落として受ける。 */
  mesh(
    w: number,
    h: number,
    d: number,
    r: number,
    b: number,
    material: THREE.Material | THREE.Material[],
    x: number,
    y: number,
    z: number,
    parent: THREE.Object3D,
  ): THREE.Mesh;
  dispose(): void;
}

// 描画に使う共有 geometry / material / texture は生成元が一度だけ破棄する。
export function createResources(): Resources {
  const owned = new Set<Disposable>();
  const geometry = new Map<string, THREE.ExtrudeGeometry>();
  const keep = <T extends Disposable>(resource: T): T => {
    owned.add(resource);
    return resource;
  };
  function rounded(w: number, h: number, d: number, r: number, b: number) {
    const id = [w, h, d, r, b].join(",");
    const cached = geometry.get(id);
    if (cached) return cached;
    const x = -w / 2 + b,
      y = -h / 2 + b,
      ww = w - 2 * b,
      hh = h - 2 * b,
      rr = Math.max(0.001, r - b),
      s = new THREE.Shape();
    s.moveTo(x + rr, y);
    s.lineTo(x + ww - rr, y);
    s.quadraticCurveTo(x + ww, y, x + ww, y + rr);
    s.lineTo(x + ww, y + hh - rr);
    s.quadraticCurveTo(x + ww, y + hh, x + ww - rr, y + hh);
    s.lineTo(x + rr, y + hh);
    s.quadraticCurveTo(x, y + hh, x, y + hh - rr);
    s.lineTo(x, y + rr);
    s.quadraticCurveTo(x, y, x + rr, y);
    const g = new THREE.ExtrudeGeometry(s, {
      depth: Math.max(0.001, d - 2 * b),
      bevelEnabled: true,
      bevelThickness: b,
      bevelSize: b,
      bevelSegments: 3,
      steps: 1,
      curveSegments: 5,
    });
    g.translate(0, 0, b);
    geometry.set(id, keep(g));
    return g;
  }
  return {
    keep,
    mesh(w, h, d, r, b, material, x, y, z, parent) {
      const m = new THREE.Mesh(rounded(w, h, d, r, b), material);
      m.position.set(x, y, z);
      m.castShadow = m.receiveShadow = true;
      parent.add(m);
      return m;
    },
    dispose() {
      for (const resource of owned) resource.dispose();
      owned.clear();
      geometry.clear();
    },
  };
}
