import type { EventKind, TableState } from "../../../../../protocol/result.ts";

export type Layout = "desktop" | "portrait" | "landscape";
/** 見ている席から数えた席の場所。自家・下家・対面・上家の順。 */
export type SeatSlot = "south" | "east" | "north" | "west";

export interface TableLabels {
  readonly remaining: string;
  readonly loadingTiles: string;
  readonly draw: string;
  /** 局の短い名前。 */
  readonly round: string;
  /** 東南西北の短い名前。 */
  readonly winds: readonly string[];
}

export interface TableFrame {
  /** 卓に描く局面（もしもの写しのことがある）。players の位置が絶対の席。 */
  readonly snapshot: TableState;
  /** 今の位置の出来事。ツモの到着と最後の打牌の印だけに使う。 */
  readonly event: { readonly eventType: EventKind; readonly actor: number };
  readonly dealer: number;
  readonly viewSeat: number;
  readonly reveal: boolean;
  readonly animateDraw: boolean;
  readonly layout: Layout;
  /** PC の操作の縮尺（画面の幅 / 1440）。 */
  readonly uiScale: number;
  readonly labels: TableLabels;
  /** もしもの当たり枠を onTargets で送るか。 */
  readonly targets: boolean;
}

/** 卓の左上から見た画面の点（CSS の px、0.1 px に丸める）。 */
export type ScreenPoint = readonly [number, number];
/** もしもで押せる所の画面の多角形。席は絶対の席。 */
export interface TableTargets {
  /** 河の牌の輪郭。奥から手前の順で、手前の牌が上に重なる。 */
  readonly rivers: readonly {
    readonly seat: number;
    readonly index: number;
    readonly points: readonly ScreenPoint[];
  }[];
  /** 見ている席の手牌とツモ牌の輪郭（手の場面のカメラ）。index は表示の位置で、ツモ牌は hand.length。 */
  readonly hands: readonly {
    readonly index: number;
    readonly points: readonly ScreenPoint[];
  }[];
  /** 中央の表示の点数の枠。並びは自家・下家・対面・上家。 */
  readonly scores: readonly {
    readonly seat: number;
    readonly points: readonly ScreenPoint[];
  }[];
}

/** 描画が読み書きする React の要素。描画は class で DOM を探さない。 */
export interface TableHost {
  /** #epsilon-review-material。描いた後に data-ready を付ける。 */
  readonly root: HTMLElement;
  /** .em-board。描画の canvas と名札の層をここに置く。 */
  readonly board: HTMLElement;
  /** .em-device。カメラの当てはめの位置を CSS 変数で書く。 */
  readonly device: HTMLElement;
  readonly playerLabels: Readonly<Record<SeatSlot, HTMLElement>>;
  /** .em-playback。高さでカメラを当てはめ、位置で名札を置く。 */
  readonly playback: HTMLElement;
  /** .em-dora-hud。名札はこれを避け、位置の基準にもする。 */
  readonly dora: HTMLElement;
  /** 名札が避ける再生の操作（進む・戻るなどのボタンとメニューの札）。 */
  readonly controls: ReadonlySet<Element>;
  /** もしもの操作の欄。開いていなければ null。 */
  panel(): HTMLElement | null;
  /** 候補の欄。縦の画面では null。 */
  candidates(): HTMLElement | null;
}

export interface TableEvents {
  /** 牌の絵の読み込み・描画・WebGL の文脈の失敗。 */
  onError(error: Error): void;
  /** もしもの当たり枠。形が変わった時だけ呼ぶ。求めていない間は null。 */
  onTargets(targets: TableTargets | null): void;
}

export interface TableRenderer {
  update(frame: TableFrame): void;
  dispose(): void;
}
