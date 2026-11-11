package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.tile.Tiles;

/**
 * 判断の時点の直前のイベント。局の公開情報が 1 つだけ持ち、イベントごとに書き換える。そのため読む側が使えるのは、卓を次に
 * 進める呼び出しまで。牌は種類で表し、無ければ {@link #NONE}。
 */
public final class TurnEvent {

  public enum Kind {
    /** チー・ポンの後と局の始め。 */
    NONE,
    /** ツモ（嶺上と北抜きの補充を含む）。 */
    DRAW,
    DISCARD,
    /** 加槓・暗槓の宣言。 */
    KAN,
    /** 北抜き。牌は北。 */
    NUKI
  }

  /** PRISM の語彙が名前と並びの順に依るので、どちらも変えない。 */
  public enum KanKind {
    ANKAN,
    KAKAN,
    DAIMINKAN
  }

  /** 席や牌が無いことを表す値。 */
  public static final int NONE = -1;

  private Kind kind = Kind.NONE;
  private int seat = NONE;
  private int tile = NONE;
  private boolean red;
  private boolean rinshan;
  private boolean doraPending;
  private KanKind kanKind;

  public Kind kind() {
    return kind;
  }

  public int seat() {
    return seat;
  }

  /** 牌の種類。イベントが無いときと、見る席から伏せて見える他家のツモでは {@link #NONE}。 */
  public int tile() {
    return tile;
  }

  /** 牌が赤の 5 なら true。加槓では、足した牌が赤なら true。 */
  public boolean red() {
    return red;
  }

  /** ツモが嶺上か北抜きの補充なら true。 */
  public boolean rinshan() {
    return rinshan;
  }

  /** ツモ・槓・北抜きの時点で、大明槓・加槓の後めくりの槓ドラがまだめくられていなければ true。 */
  public boolean doraPending() {
    return doraPending;
  }

  /** 槓の種類。槓の宣言でなければ null。 */
  public KanKind kanKind() {
    return kanKind;
  }

  void clear() {
    set(Kind.NONE, NONE, NONE, false, false, false, null);
  }

  void setDraw(int seat, int tile, boolean red, boolean rinshan, boolean doraPending) {
    set(Kind.DRAW, seat, tile, red, rinshan, doraPending, null);
  }

  void setDiscard(int seat, int tile, boolean red) {
    set(Kind.DISCARD, seat, tile, red, false, false, null);
  }

  void setKan(int seat, int tile, boolean red, KanKind kanKind, boolean doraPending) {
    set(Kind.KAN, seat, tile, red, false, doraPending, kanKind);
  }

  void setNuki(int seat, boolean doraPending) {
    set(Kind.NUKI, seat, Tiles.PEI, false, false, doraPending, null);
  }

  void copyFrom(TurnEvent other) {
    set(other.kind, other.seat, other.tile, other.red, other.rinshan, other.doraPending, other.kanKind);
  }

  private void set(
      Kind kind, int seat, int tile, boolean red, boolean rinshan, boolean doraPending, KanKind kanKind) {
    this.kind = kind;
    this.seat = seat;
    this.tile = tile;
    this.red = red;
    this.rinshan = rinshan;
    this.doraPending = doraPending;
    this.kanKind = kanKind;
  }
}
