import * as THREE from "three";
import { createResources } from "./resources.js";

// 卓面と前景手牌は照明を共有せず、同じ牌形状をそれぞれのカメラで描く。
export function createTableScene(renderer) {
  const resources = createResources();
  const { keep } = resources;
  function mesh(w, h, d, r, b, material, x = 0, y = 0, z = 0, parent = table) {
    return resources.mesh(w, h, d, r, b, material, x, y, z, parent);
  }
  const scene = new THREE.Scene(),
    camera = new THREE.PerspectiveCamera(55, 1, 0.1, 160),
    table = new THREE.Group(),
    pieces = new THREE.Group();
  scene.add(table, pieces);
  camera.up.set(0, 0, 1);
  // The reference uses a readable foreground hand. It shares the tile mesh and
  // materials, but has its own close camera so mobile faces remain legible.
  const handScene = new THREE.Scene(),
    handCamera = new THREE.PerspectiveCamera(32, 1, 0.1, 100),
    handGroup = new THREE.Group();
  handScene.add(handGroup);
  handCamera.up.set(0, 0, 1);
  const handLight = new THREE.DirectionalLight(0xfff8ed, 2.5);
  handLight.position.set(-9, -8, 18);
  handLight.castShadow = true;
  keep(handLight.shadow);
  handLight.shadow.mapSize.set(1024, 1024);
  Object.assign(handLight.shadow.camera, {
    left: -11,
    right: 11,
    top: 7,
    bottom: -7,
    near: 1,
    far: 40,
  });
  handLight.shadow.normalBias = 0.018;
  handLight.shadow.bias = -0.0001;
  handScene.add(handLight);
  const contact = new THREE.Mesh(
    keep(new THREE.PlaneGeometry(40, 12)),
    keep(new THREE.ShadowMaterial({ opacity: 0.26 })),
  );
  contact.position.set(0, -3, 0.07);
  contact.receiveShadow = true;
  handScene.add(contact);
  const handHemi = new THREE.HemisphereLight(0xf2f8ff, 0x30455a, 2.5);
  handHemi.position.set(0, 0, 1);
  handScene.add(handHemi);
  const hemi = new THREE.HemisphereLight(0xf0f7ff, 0x283d53, 2.6);
  hemi.position.set(0, 0, 1);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xfff8e9, 2.6);
  key.position.set(-16, 7, 38);
  key.castShadow = true;
  keep(key.shadow);
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, {
    left: -22,
    right: 22,
    top: 22,
    bottom: -22,
    near: 1,
    far: 85,
  });
  key.shadow.bias = -0.0001;
  key.shadow.normalBias = 0.017;
  key.shadow.radius = 2;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xbad9ff, 0.9);
  fill.position.set(14, -15, 22);
  scene.add(fill);
  const materials = {
    ivory: keep(
      new THREE.MeshPhysicalMaterial({
        color: 0xf4f4ec,
        roughness: 0.25,
        clearcoat: 0.6,
        clearcoatRoughness: 0.23,
      }),
    ),
    back: keep(
      new THREE.MeshPhysicalMaterial({
        color: 0x70c7e2,
        roughness: 0.28,
        clearcoat: 0.65,
        clearcoatRoughness: 0.22,
      }),
    ),
    rail: keep(
      new THREE.MeshPhysicalMaterial({
        color: 0x30373e,
        roughness: 0.35,
        metalness: 0.25,
        clearcoat: 0.25,
      }),
    ),
    trim: keep(
      new THREE.MeshStandardMaterial({
        color: 0x73808c,
        roughness: 0.3,
        metalness: 0.6,
      }),
    ),
    case: keep(
      new THREE.MeshStandardMaterial({
        color: 0x48535f,
        roughness: 0.43,
        metalness: 0.28,
      }),
    ),
    black: keep(
      new THREE.MeshPhysicalMaterial({
        color: 0x050e19,
        roughness: 0.55,
        clearcoat: 0,
        specularIntensity: 0.12,
        envMapIntensity: 0.02,
      }),
    ),
  };
  const riverFace = keep(
    new THREE.MeshPhysicalMaterial({
      color: 0xf5f2e8,
      roughness: 0.4,
      clearcoat: 0.2,
      clearcoatRoughness: 0.35,
    }),
  );
  const riverSide = keep(
    new THREE.MeshPhysicalMaterial({
      color: 0xaab4b8,
      roughness: 0.48,
      clearcoat: 0.12,
      clearcoatRoughness: 0.4,
    }),
  );
  const riverBody = [riverFace, riverSide];
  // ツモ切りは牌本体をグレーにし、図柄と赤牌の色はそのまま保つ。
  const tsumogiriBody = [
    keep(
      new THREE.MeshPhysicalMaterial({
        color: 0x858b92,
        roughness: 0.4,
        clearcoat: 0.2,
        clearcoatRoughness: 0.35,
      }),
    ),
    keep(
      new THREE.MeshPhysicalMaterial({
        color: 0x68717b,
        roughness: 0.48,
        clearcoat: 0.12,
        clearcoatRoughness: 0.4,
      }),
    ),
  ];
  // Small original textile pattern; no photograph is enlarged across the tabletop.
  const fabric = document.createElement("canvas");
  fabric.width = fabric.height = 512;
  const fc = fabric.getContext("2d");
  let seed = 9271;
  const rnd = () => (seed = (1664525 * seed + 1013904223) >>> 0) / 4294967296;
  function paintFabric(color) {
    const rgb = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16)),
      d = fc.createImageData(512, 512);
    seed = 9271;
    for (let i = 0; i < d.data.length; i += 4) {
      const n = (rnd() + rnd() - 1) * 7;
      d.data[i] = rgb[0] + n;
      d.data[i + 1] = rgb[1] + n;
      d.data[i + 2] = rgb[2] + n;
      d.data[i + 3] = 255;
    }
    fc.putImageData(d, 0, 0);
    fc.lineWidth = 0.5;
    for (let i = 0; i < 13000; i++) {
      const x = rnd() * 512,
        y = rnd() * 512;
      fc.strokeStyle = rnd() > 0.5 ? "#93b3d010" : "#05132a1a";
      fc.beginPath();
      fc.moveTo(x, y);
      fc.lineTo(x + 1.5, y + 0.4);
      fc.stroke();
    }
  }
  paintFabric("#0b203e");
  const fabricTex = keep(new THREE.CanvasTexture(fabric));
  fabricTex.colorSpace = THREE.SRGBColorSpace;
  fabricTex.wrapS = fabricTex.wrapT = THREE.RepeatWrapping;
  fabricTex.repeat.set(3, 3);
  fabricTex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const felt = keep(
    new THREE.MeshStandardMaterial({
      map: fabricTex,
      bumpMap: fabricTex,
      bumpScale: 0.025,
      roughness: 1,
    }),
  );
  const feltSurface = mesh(28.05, 40.05, 0.1, 0.4, 0.018, felt, 0, -6, -0.025);
  // The landscape felt and its frame share the same near and far edges.
  const landscapeFrame = new THREE.Group();
  table.add(landscapeFrame);
  mesh(
    30.7,
    42.7,
    1.2,
    0.65,
    0.18,
    materials.rail,
    0,
    -6,
    -1.15,
    landscapeFrame,
  );
  mesh(
    28.6,
    40.6,
    0.17,
    0.5,
    0.05,
    materials.black,
    0,
    -6,
    -0.12,
    landscapeFrame,
  );
  for (const x of [-14.85, 14.85]) {
    mesh(
      1.32,
      40.5,
      0.52,
      0.35,
      0.13,
      materials.rail,
      x,
      -6,
      -0.05,
      landscapeFrame,
    );
    mesh(
      0.075,
      39.9,
      0.07,
      0.035,
      0.014,
      materials.trim,
      Math.sign(x) * 14.2,
      -6,
      0.12,
      landscapeFrame,
    );
  }
  for (const y of [14.85, -26.85]) {
    mesh(
      28.5,
      1.32,
      0.52,
      0.35,
      0.13,
      materials.rail,
      0,
      y,
      -0.05,
      landscapeFrame,
    );
    mesh(
      27.9,
      0.075,
      0.07,
      0.035,
      0.014,
      materials.trim,
      0,
      y > 0 ? 14.2 : -26.2,
      0.12,
      landscapeFrame,
    );
  }
  const seamMat = keep(
    new THREE.LineBasicMaterial({
      color: 0x4d7599,
      transparent: true,
      opacity: 0.23,
    }),
  );
  for (const r of [7.65, 7.77]) {
    const pts = [
      [-r, -r],
      [r, -r],
      [r, r],
      [-r, r],
      [-r, -r],
    ].map(([x, y]) => new THREE.Vector3(x, y, 0.09));
    table.add(
      new THREE.Line(
        keep(new THREE.BufferGeometry()).setFromPoints(pts),
        seamMat,
      ),
    );
  }
  const displayCanvas = document.createElement("canvas");
  displayCanvas.width = displayCanvas.height = 768;
  const dc = displayCanvas.getContext("2d"),
    displayTex = keep(new THREE.CanvasTexture(displayCanvas));
  displayTex.colorSpace = THREE.SRGBColorSpace;
  displayTex.anisotropy = 8;
  const centerGroup = new THREE.Group();
  table.add(centerGroup);
  mesh(5.65, 5.65, 0.35, 0.28, 0.075, materials.black, 0, 0, 0.06, centerGroup);
  mesh(5.5, 5.5, 0.32, 0.23, 0.055, materials.case, 0, 0, 0.22, centerGroup);
  const display = new THREE.Mesh(
    keep(new THREE.PlaneGeometry(5.25, 5.25)),
    keep(new THREE.MeshBasicMaterial({ map: displayTex, toneMapped: false })),
  );
  display.position.z = 0.557;
  centerGroup.add(display);
  for (const a of [0, Math.PI / 2, Math.PI, Math.PI * 1.5]) {
    const g = new THREE.Group();
    g.rotation.z = a;
    centerGroup.add(g);
    mesh(2.45, 0.17, 0.08, 0.075, 0.025, materials.trim, 0, -2.53, 0.56, g);
  }
  function paintDisplay({ snapshot, round, viewSeat, labels }) {
    const active = snapshot.activeSeat;
    dc.clearRect(0, 0, 768, 768);
    dc.fillStyle = "#414c59";
    dc.fillRect(0, 0, 768, 768);
    dc.strokeStyle = "#8b969e";
    dc.lineWidth = 2;
    dc.strokeRect(4, 4, 760, 760);
    [0, 1, 2, 3].forEach((offset, i) => {
      const seat = (viewSeat + offset) % 4;
      const player = snapshot.players.find((p) => p.seat === seat);
      dc.save();
      dc.translate(384, 384);
      dc.rotate((-i * Math.PI) / 2);
      dc.fillStyle = active === seat ? "#9dd7e8" : "#697784";
      dc.fillRect(-316, 233, 75, 70);
      dc.fillStyle = active === seat ? "#112c40" : "#e1e8ec";
      dc.font = "bold 56px sans-serif";
      dc.textAlign = "center";
      dc.fillText(labels.winds[(seat - round.dealer + 4) % 4], -279, 287);
      dc.fillStyle = "#081322";
      dc.fillRect(-177, 210, 354, 86);
      dc.fillStyle = "#f3d35e";
      dc.font = "bold 76px monospace";
      dc.fillText(String(player.score), 0, 280);
      dc.strokeStyle = "#242e38";
      dc.lineWidth = 4;
      dc.beginPath();
      dc.moveTo(-375, 375);
      dc.lineTo(-177, 177);
      dc.stroke();
      dc.restore();
    });
    dc.fillStyle = "#222c39";
    dc.fillRect(185, 185, 398, 398);
    dc.strokeStyle = "#6e7b89";
    dc.lineWidth = 5;
    dc.strokeRect(185, 185, 398, 398);
    dc.fillStyle = "#071626";
    dc.fillRect(224, 224, 320, 320);
    dc.fillStyle = "#87e7f4";
    dc.font = "bold 64px sans-serif";
    dc.textAlign = "center";
    dc.fillText(labels.round, 384, 320);
    dc.font = "bold 112px monospace";
    dc.fillText(String(snapshot.remaining), 384, 442);
    dc.fillStyle = "#9ab9c8";
    dc.font = "36px sans-serif";
    dc.fillText(labels.remaining, 384, 495);
    displayTex.needsUpdate = true;
  }
  return {
    resources,
    mesh,
    scene,
    camera,
    table,
    pieces,
    handScene,
    handCamera,
    handGroup,
    materials,
    riverBody,
    tsumogiriBody,
    landscapeFrame,
    feltSurface,
    fabricTex,
    centerGroup,
    paintDisplay,
  };
}
