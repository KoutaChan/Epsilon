import * as THREE from "three";
import type { PlayerState } from "../../../../../protocol/result.ts";
import { local, tableDimensions, type TableDimensions } from "./projection.ts";
import type { TableScene } from "./scene.ts";
import type { TileTextures } from "./textures.ts";
import {
  PLAYER_ROW_WIDTH,
  concealedTileX,
  layoutPlayerRow,
} from "./tileLayout.ts";
import type { Layout, SeatSlot, TableFrame } from "./types.ts";

/** 牌の組（Group）の userData。 */
export interface TileData {
  readonly seat: SeatSlot;
  readonly foreground: boolean;
  readonly kind: "hand" | "river" | "meld";
  readonly baseY: number;
  readonly baseZ: number;
}

/** ツモ牌の札（DOM）と、その牌。 */
export interface DrawAnnotation {
  readonly group: THREE.Group;
  readonly element: HTMLElement;
}
/** 河の 1 枚。seat は絶対の席、index は river[] の位置（鳴かれた牌も数える）。 */
export interface RiverPiece {
  readonly group: THREE.Group;
  readonly seat: number;
  readonly index: number;
}
/** 見ている席の手牌の 1 枚。index は表示の位置で、ツモ牌は hand.length。 */
export interface HandPiece {
  readonly group: THREE.Group;
  readonly index: number;
}

export interface TileFrame {
  /** この更新で作り直した席があるか。 */
  readonly rebuilt: boolean;
  /** 作り直した席で、今届いたツモ牌。 */
  readonly arrival: THREE.Group | null;
  readonly annotations: readonly DrawAnnotation[];
  readonly river: readonly RiverPiece[];
  readonly hand: readonly HandPiece[];
}

interface TileOptions {
  value: string;
  x?: number;
  y?: number;
  z?: number;
  angle?: number;
  standing?: boolean;
  river?: boolean;
  tsumogiri?: boolean;
  seat?: SeatSlot;
  foreground?: boolean;
  kind: TileData["kind"];
}

/** 1 席ぶんの描いた物。player が同じ物で描き方も同じなら作り直さない。 */
interface SeatTiles {
  readonly player: PlayerState;
  /** 最後の打牌の印を付けたか。 */
  readonly marked: boolean;
  readonly objects: readonly THREE.Object3D[];
  readonly annotation: DrawAnnotation | null;
  readonly arrival: THREE.Group | null;
  readonly river: readonly RiverPiece[];
  readonly hand: readonly HandPiece[];
}

const SEATS = ["south", "east", "north", "west"] as const;

export const tileData = (group: THREE.Object3D) => group.userData as TileData;

// 自家は操作バーへ持ち上げず手前から入れ、他家は卓上で持ち上げる。
export function positionDrawArrival(group: THREE.Object3D, remaining: number) {
  const { foreground, baseY, baseZ } = tileData(group);
  const offset = remaining ** 2 * 0.5;
  group.position.y = baseY - (foreground ? offset : 0);
  group.position.z = baseZ + (foreground ? 0 : offset);
}

// 手牌は文書の時点で理牌済み。局面の配列を借用し、複製や再ソートを行わない。
export function createTileLayer(
  overlay: HTMLElement,
  scene: TableScene,
  textures: Pick<TileTextures, "get">,
  rowWidth = PLAYER_ROW_WIDTH,
) {
  const { pieces, handGroup, materials, riverBody, tsumogiriBody } = scene;
  const { keep, mesh } = scene.resources;
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
  const context = contactCanvas.getContext("2d")!;
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
  /** 見ている席から数えた席ごとの描いた物。 */
  const seats: (SeatTiles | null)[] = [null, null, null, null];
  let drawn: { viewSeat: number; reveal: boolean; layout: Layout } | null =
    null;
  let landscape = false;

  function tile(
    {
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
    }: TileOptions,
    objects: THREE.Object3D[],
  ) {
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
      tsumogiri ? tsumogiriBody[1]! : back ? materials.ivory : materials.back,
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
    group.userData = {
      seat,
      foreground,
      kind,
      baseY: y,
      baseZ: z,
    } satisfies TileData;
    (foreground ? handGroup : pieces).add(group);
    objects.push(group);
    return group;
  }

  function place(
    parent: THREE.Object3D,
    object: THREE.Object3D,
    objects: THREE.Object3D[],
  ) {
    parent.add(object);
    objects.push(object);
  }

  function drawLabel(
    group: THREE.Group,
    text: string,
    objects: THREE.Object3D[],
  ): DrawAnnotation {
    const element = document.createElement("span");
    element.className = "em-world-draw";
    element.textContent = text;
    overlay.append(element);
    if (!tileData(group).foreground) {
      const ring = new THREE.Mesh(ringGeometry, ringMaterial);
      ring.scale.set(group.scale.x, group.scale.y * 1.15, 1);
      ring.position.set(group.position.x, group.position.y, 0.09);
      place(pieces, ring, objects);
    }
    return { group, element };
  }

  function buildSeat(
    relativeSeat: number,
    player: PlayerState,
    seatIndex: number,
    marked: boolean,
    frame: TableFrame,
    dimensions: TableDimensions,
  ): SeatTiles {
    const { event, reveal, labels, animateDraw } = frame;
    const seat = SEATS[relativeSeat]!,
      foreground = relativeSeat === 0;
    const objects: THREE.Object3D[] = [];
    const river: RiverPiece[] = [];
    const hand: HandPiece[] = [];
    let annotation: DrawAnnotation | null = null;
    let arrival: THREE.Group | null = null;
    const scale = foreground ? 1 : dimensions.opponentScale;
    const row = layoutPlayerRow(player.melds, rowWidth);
    const count = player.hand.length + (player.draw ? 1 : 0);
    for (let i = 0; i < count; i++) {
      const drawnTile = i === player.hand.length;
      const concealed = !foreground && !reveal;
      const x = concealedTileX(i, drawnTile, row.handLeft);
      const group = tile(
        {
          value: concealed
            ? "back"
            : drawnTile
              ? player.draw!
              : player.hand[i]!,
          ...(foreground
            ? { x, y: 0 }
            : local(seat, x * scale, -dimensions.opponentRadius)),
          seat,
          foreground,
          standing: foreground || concealed,
          kind: "hand",
        },
        objects,
      );
      group.scale.setScalar(scale);
      if (foreground) hand.push({ group, index: i });
      if (drawnTile) {
        annotation = drawLabel(group, labels.draw, objects);
        if (
          animateDraw &&
          event.eventType === "tsumo" &&
          event.actor === seatIndex
        )
          arrival = group;
      }
    }

    let drawnIndex = 0,
      sidewaysOffset = 0,
      pendingRiichiMarker = false;
    for (const [riverIndex, discarded] of player.river.entries()) {
      // 宣言牌が鳴かれた場合は、次に河へ残る牌を横向きにする。
      pendingRiichiMarker ||= discarded.riichi === true;
      if (discarded.called) continue;
      const sideways = pendingRiichiMarker;
      pendingRiichiMarker = false;
      const i = drawnIndex++;
      if (i % 6 === 0) sidewaysOffset = 0;
      const extra = sideways ? dimensions.riverWidth / 3 : 0;
      const position = local(
        seat,
        dimensions.riverStart +
          (i % 6) * dimensions.pitchX +
          sidewaysOffset +
          extra / 2,
        -dimensions.riverRadius - Math.floor(i / 6) * dimensions.pitchY,
      );
      sidewaysOffset += extra;
      const group = tile(
        {
          value: discarded.tile,
          ...position,
          angle: position.angle + (sideways ? Math.PI / 2 : 0),
          river: true,
          kind: "river",
          tsumogiri: discarded.tsumogiri === true,
          seat,
        },
        objects,
      );
      group.scale.set(
        dimensions.riverWidth,
        dimensions.riverWidth,
        landscape ? 1.4 : 1,
      );
      // もしもの当たり枠は river[] の位置（鳴かれた牌も数える）と絶対の席で牌を指す。
      river.push({ group, seat: seatIndex, index: riverIndex });
      if (landscape) {
        const contact = new THREE.Mesh(contactGeometry, contactMaterial);
        contact.rotation.z = group.rotation.z;
        contact.position.set(
          group.position.x + 0.025,
          group.position.y - 0.035,
          0.075,
        );
        contact.scale.set(dimensions.riverWidth, dimensions.riverWidth, 1);
        place(pieces, contact, objects);
      }
      if (marked && discarded === player.river.at(-1)) {
        const marker = new THREE.Mesh(discardGeometry, discardMaterial);
        marker.position.set(group.position.x, group.position.y, 0.08);
        marker.rotation.z = group.rotation.z;
        marker.scale.set(dimensions.riverWidth, dimensions.riverWidth, 1);
        place(pieces, marker, objects);
      }
    }

    // 自家も他家も、手牌と同じ列・縮尺・カメラで副露を描く。
    for (const placement of row.melds) {
      const { meld, index, x, y, sideways } = placement;
      const position = foreground
        ? { x, y, angle: 0 }
        : local(seat, x * scale, -dimensions.opponentRadius + y * scale);
      const group = tile(
        {
          value:
            meld.type === "ANKAN" &&
            (index === 0 || index === meld.tiles.length - 1)
              ? "back"
              : meld.tiles[index]!,
          ...position,
          angle: position.angle + (sideways ? Math.PI / 2 : 0),
          seat,
          foreground,
          kind: "meld",
        },
        objects,
      );
      group.scale.setScalar(scale);
      const contact = new THREE.Mesh(contactGeometry, contactMaterial);
      contact.rotation.z = group.rotation.z;
      contact.position.set(position.x + 0.025, position.y - 0.035, 0.075);
      contact.scale.set(scale, scale, 1);
      place(foreground ? handGroup : pieces, contact, objects);
    }
    return { player, marked, objects, annotation, arrival, river, hand };
  }

  function remove(entry: SeatTiles) {
    for (const object of entry.objects) object.removeFromParent();
    entry.annotation?.element.remove();
  }

  /** 局面を描く。player の物・最後の打牌の印・視点・全員の手牌・配置が変わった席だけ作り直す。 */
  function update(frame: TableFrame): TileFrame {
    const { snapshot, event, viewSeat, reveal, layout } = frame;
    const all =
      drawn === null ||
      drawn.viewSeat !== viewSeat ||
      drawn.reveal !== reveal ||
      drawn.layout !== layout;
    drawn = { viewSeat, reveal, layout };
    landscape = layout === "landscape";
    const dimensions = tableDimensions(layout, rowWidth);
    let rebuilt = false;
    let arrival: THREE.Group | null = null;
    for (let relativeSeat = 0; relativeSeat < 4; relativeSeat++) {
      const seatIndex = (viewSeat + relativeSeat) % 4;
      const player = snapshot.players[seatIndex]!;
      const marked = event.eventType === "dahai" && event.actor === seatIndex;
      const current = seats[relativeSeat];
      if (
        !all &&
        current &&
        current.player === player &&
        current.marked === marked
      )
        continue;
      if (current) remove(current);
      const next = buildSeat(
        relativeSeat,
        player,
        seatIndex,
        marked,
        frame,
        dimensions,
      );
      seats[relativeSeat] = next;
      rebuilt = true;
      arrival ??= next.arrival;
    }
    const built = seats as readonly SeatTiles[];
    return {
      rebuilt,
      arrival,
      annotations: built.flatMap((entry) =>
        entry.annotation ? [entry.annotation] : [],
      ),
      river: built.flatMap((entry) => entry.river),
      hand: built[0]!.hand,
    };
  }

  return { update };
}
