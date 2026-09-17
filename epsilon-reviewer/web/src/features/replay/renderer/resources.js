import * as THREE from "three";

// 描画に使う共有 geometry / material / texture は生成元が一度だけ破棄する。
export function createResources() {
  const owned = new Set();
  const geometry = new Map();
  const keep = (resource) => {
    owned.add(resource);
    return resource;
  };
  function rounded(w, h, d, r, b) {
    const id = [w, h, d, r, b].join(",");
    if (geometry.has(id)) return geometry.get(id);
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
  function mesh(w, h, d, r, b, mat, x = 0, y = 0, z = 0, parent) {
    const m = new THREE.Mesh(rounded(w, h, d, r, b), mat);
    m.position.set(x, y, z);
    m.castShadow = m.receiveShadow = true;
    parent.add(m);
    return m;
  }
  return {
    keep,
    mesh,
    dispose() {
      for (const resource of owned) resource.dispose();
      owned.clear();
      geometry.clear();
    },
  };
}
