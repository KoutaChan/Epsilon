import type { EventKind, TableState } from "../../../../../protocol/result.ts";

export type Layout = "desktop" | "portrait" | "landscape";
/** 卓の上の席の場所。見ている席から数えて、自家・下家・対面・上家の順。 */
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
  /** 卓に描く局面。もしもで変えた写しのこともある。`players` の添字は絶対の席。 */
  readonly snapshot: TableState;
  /** 今の位置のイベント。ツモ牌が届く動きと最後の打牌の印にだけ使う。 */
  readonly event: { readonly eventType: EventKind; readonly actor: number };
  readonly dealer: number;
  readonly viewSeat: number;
  readonly reveal: boolean;
  readonly animateDraw: boolean;
  readonly layout: Layout;
  /** PC の画面での操作部品の縮尺（画面の幅 / 1440）。 */
  readonly uiScale: number;
  readonly labels: TableLabels;
  /** もしもの当たり判定を作って送るか（`onTargets` で受け取る）。 */
  readonly targets: boolean;
}

/** 卓の左上を原点にした画面上の点（CSS の px、0.1 px 単位に丸める）。 */
export type ScreenPoint = readonly [number, number];
/** もしもで押せる所の、画面上の多角形。席は絶対の席。 */
export interface TableTargets {
  /** 河の牌の輪郭。奥から手前の順で、手前の牌が上に重なる。 */
  readonly rivers: readonly {
    readonly seat: number;
    readonly index: number;
    readonly points: readonly ScreenPoint[];
  }[];
  /** 見ている席の手牌とツモ牌の輪郭（手牌用のカメラで写す）。`index` は表示の並びでの位置（0 から数える）で、ツモ牌は手牌の枚数（`hand.length`）。 */
  readonly hands: readonly {
    readonly index: number;
    readonly points: readonly ScreenPoint[];
  }[];
  /** 卓の中央の表示にある点数の枠。並びは自家・下家・対面・上家。 */
  readonly scores: readonly {
    readonly seat: number;
    readonly points: readonly ScreenPoint[];
  }[];
}

/** 描画が読み書きする、React が作った要素。描画は class 名で DOM を探さない。 */
export interface TableHost {
  /** #epsilon-review-material。描いた後に data-ready を付ける。 */
  readonly root: HTMLElement;
  /** .em-board。描画の canvas と名札の層をここに置く。 */
  readonly board: HTMLElement;
  /** .em-device。カメラを合わせた後の各部の位置を、ここの CSS 変数に書く。 */
  readonly device: HTMLElement;
  readonly playerLabels: Readonly<Record<SeatSlot, HTMLElement>>;
  /** .em-playback。この高さに合わせてカメラを置き、この位置を基準に名札を置く。 */
  readonly playback: HTMLElement;
  /** .em-dora-hud。名札はこれを避け、位置の基準にもする。 */
  readonly dora: HTMLElement;
  /** 名札が避ける再生の操作部品（進む・戻るなどのボタンと、メニューのラベル）。 */
  readonly controls: ReadonlySet<Element>;
  /** もしもの操作の欄。開いていなければ null。 */
  panel(): HTMLElement | null;
  /** 候補の欄。縦長の画面では null。 */
  candidates(): HTMLElement | null;
}

export interface TableEvents {
  /** 牌の画像の読み込み、描画、WebGL のコンテキストの失敗を受け取る。 */
  onError(error: Error): void;
  /** もしもの当たり判定を受け取る。呼ぶのは形が変わったときだけ。当たり判定を求められていない間は null を渡す。 */
  onTargets(targets: TableTargets | null): void;
}

export interface TableRenderer {
  update(frame: TableFrame): void;
  dispose(): void;
}
