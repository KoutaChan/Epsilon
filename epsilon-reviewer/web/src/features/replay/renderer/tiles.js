import * as THREE from "three";
import { tableDimensions, local } from "./projection.js";
import {
  PLAYER_ROW_WIDTH,
  concealedTileX,
  layoutPlayerRow,
} from "./tileLayout.js";

// 自家は操作バーへ持ち上げず手前から入れ、他家は卓上で持ち上げる。
export function positionDrawArrival(group, remaining) {
  const { foreground, baseY, baseZ } = group.userData;
  const offset = remaining ** 2 * 0.5;
  group.position.y = baseY - (foreground ? offset : 0);
  group.position.z = baseZ + (foreground ? 0 : offset);
}

const SEATS = ["south", "east", "north", "west"];
// 手牌はDTO生成時に理牌済み。局面の配列を借用し、複製や再ソートを行わない。
export function createTileLayer(
  overlay,
  scene,
  textures,
  rowWidth = PLAYER_ROW_WIDTH,
) {
  const { mesh, pieces, handGroup, materials, riverBody, tsumogiriBody } =
    scene;
  const { keep } = scene.resources;
  const glyphGeometry = keep(new THREE.PlaneGeometry(0.9, 1.2));
  const ringGeometry = keep(new THREE.RingGeometry(0.69, 0.72, 40));
  const ringMaterial = keep(
    new THREE.MeshBasicMaterial({
      color: 0xa7e8f9,
      transparent: true,
      opacity: 0.75,
      depthWrite: false,
    }),
  );
  const discardGeometry = keep(new THREE.PlaneGeometry(1.07, 1.4));
  const discardMaterial = keep(
    new THREE.MeshBasicMaterial({
      color: 0xffd67a,
      transparent: true,
      opacity: 0.6,
    }),
  );
  const contactCanvas = document.createElement("canvas");
  contactCanvas.width = 128;
  contactCanvas.height = 160;
  const context = contactCanvas.getContext("2d");
  context.filter = "blur(5px)";
  context.fillStyle = "#0009";
  context.beginPath();
  context.roundRect(12, 12, 104, 136, 10);
  context.fill();
  const contactTexture = keep(new THREE.CanvasTexture(contactCanvas));
  const contactGeometry = keep(new THREE.PlaneGeometry(1.14, 1.48));
  const contactMaterial = keep(
    new THREE.MeshBasicMaterial({
      map: contactTexture,
      transparent: true,
      opacity: 0.38,
      depthWrite: false,
      toneMapped: false,
    }),
  );
  let annotations = [],
    landscape = false;

  function tile({
    value,
    x = 0,
    y = 0,
    z = 0.085,
    angle = 0,
    standing = false,
    river = false,
    tsumogiri = false,
    seat = "south",
    foreground = false,
    kind,
  }) {
    const back = value === "back";
    const group = new THREE.Group(),
      body = new THREE.Group();
    const depth = 0.61,
      split = back ? 0.43 : 0.17;
    group.add(body);
    mesh(
      1,
      4 / 3,
      split,
      0.065,
      0.02,
      tsumogiri ? tsumogiriBody[1] : back ? materials.ivory : materials.back,
      0,
      0,
      0,
      body,
    );
    mesh(
      1,
      4 / 3,
      depth - split,
      0.065,
      0.028,
      tsumogiri
        ? tsumogiriBody
        : back
          ? materials.back
          : river && landscape
            ? riverBody
            : materials.ivory,
      0,
      0,
      split - 0.008,
      body,
    );
    if (!back) {
      const glyph = new THREE.Mesh(glyphGeometry, textures.get(value));
      glyph.position.z = depth - 0.004;
      body.add(glyph);
    }
    if (standing) {
      // 自家は立った形を保ちつつ牌面をカメラへ向け、副露の俯角と両立させる。
      const angle = foreground ? (68 * Math.PI) / 180 : Math.PI / 2;
      body.rotation.x = back ? -angle : angle;
      body.position.set(
        0,
        (back ? -1 : 1) * (depth / 2) * Math.sin(angle),
        (2 / 3) * Math.sin(angle),
      );
    }
    group.rotation.z = angle;
    group.position.set(x, y, z);
    group.userData = { seat, foreground, kind, baseY: y, baseZ: z };
    (foreground ? handGroup : pieces).add(group);
    return group;
  }

  function drawLabel(group, text) {
    const element = document.createElement("span");
    element.className = "em-world-draw";
    element.textContent = text;
    overlay.append(element);
    annotations.push({ g: group, e: element });
    if (group.userData.foreground) return;
    const ring = new THREE.Mesh(ringGeometry, ringMaterial);
    ring.scale.set(group.scale.x, group.scale.y * 1.15, 1);
    ring.position.set(group.position.x, group.position.y, 0.09);
    pieces.add(ring);
  }

  function build(frame) {
    const { snapshot, event, viewSeat, reveal, layout, labels, animateDraw } =
      frame;
    const dimensions = tableDimensions(layout, rowWidth);
    landscape = layout === "landscape";
    pieces.clear();
    handGroup.clear();
    overlay.replaceChildren();
    annotations = [];
    let arrival = null;
    for (let relativeSeat = 0; relativeSeat < 4; relativeSeat++) {
      const seat = SEATS[relativeSeat],
        foreground = relativeSeat === 0;
      const player = snapshot.players.find(
        (p) => p.seat === (viewSeat + relativeSeat) % 4,
      );
      const scale = foreground ? 1 : dimensions.opponentScale;
      const hand = player.hand;
      const row = layoutPlayerRow(player.melds, rowWidth);
      for (let i = 0; i < hand.length + (player.draw ? 1 : 0); i++) {
        const drawn = i === hand.length;
        const concealed = !foreground && !reveal;
        const x = concealedTileX(i, drawn, row.handLeft);
        const group = tile({
          value: concealed ? "back" : drawn ? player.draw : hand[i],
          ...(foreground
            ? { x, y: 0 }
            : local(seat, x * scale, -dimensions.opponentRadius)),
          seat,
          foreground,
          standing: foreground || concealed,
          kind: "hand",
        });
        group.scale.setScalar(scale);
        if (drawn) {
          drawLabel(group, labels.draw);
          if (
            animateDraw &&
            event.eventType.toLowerCase() === "tsumo" &&
            event.actor === player.seat
          )
            arrival = group;
        }
      }

      let riverIndex = 0,
        sidewaysOffset = 0;
      for (const discarded of player.river) {
        if (discarded.called) continue;
        const i = riverIndex++;
        if (i % 6 === 0) sidewaysOffset = 0;
        const extra = discarded.riichi ? dimensions.riverWidth / 3 : 0;
        const position = local(
          seat,
          dimensions.riverStart +
            (i % 6) * dimensions.pitchX +
            sidewaysOffset +
            extra / 2,
          -dimensions.riverRadius - Math.floor(i / 6) * dimensions.pitchY,
        );
        sidewaysOffset += extra;
        const group = tile({
          value: discarded.tile,
          ...position,
          angle: position.angle + (discarded.riichi ? Math.PI / 2 : 0),
          river: true,
          kind: "river",
          tsumogiri: discarded.tsumogiri,
          seat,
        });
        group.scale.set(
          dimensions.riverWidth,
          dimensions.riverWidth,
          landscape ? 1.4 : 1,
        );
        if (landscape) {
          const contact = new THREE.Mesh(contactGeometry, contactMaterial);
          contact.rotation.z = group.rotation.z;
          contact.position.set(
            group.position.x + 0.025,
            group.position.y - 0.035,
            0.075,
          );
          contact.scale.set(dimensions.riverWidth, dimensions.riverWidth, 1);
          pieces.add(contact);
        }
        if (
          discarded === player.river.at(-1) &&
          event.eventType.toLowerCase() === "dahai" &&
          event.actor === player.seat
        ) {
          const marker = new THREE.Mesh(discardGeometry, discardMaterial);
          marker.position.set(group.position.x, group.position.y, 0.08);
          marker.rotation.z = group.rotation.z;
          marker.scale.set(dimensions.riverWidth, dimensions.riverWidth, 1);
          pieces.add(marker);
        }
      }

      // 自家も他家も、手牌と同じ列・縮尺・カメラで副露を描く。
      for (const placement of row.melds) {
        const { meld, index, x, y, sideways } = placement;
        const position = foreground
          ? { x, y, angle: 0 }
          : local(seat, x * scale, -dimensions.opponentRadius + y * scale);
        const group = tile({
          value:
            meld.type === "ANKAN" &&
            (index === 0 || index === meld.tiles.length - 1)
              ? "back"
              : meld.tiles[index],
          ...position,
          angle: position.angle + (sideways ? Math.PI / 2 : 0),
          seat,
          foreground,
          kind: "meld",
        });
        group.scale.setScalar(scale);
        const contact = new THREE.Mesh(contactGeometry, contactMaterial);
        contact.rotation.z = group.rotation.z;
        contact.position.set(position.x + 0.025, position.y - 0.035, 0.075);
        contact.scale.set(scale, scale, 1);
        (foreground ? handGroup : pieces).add(contact);
      }
    }
    return { annotations, arrival };
  }
  return { build };
}
