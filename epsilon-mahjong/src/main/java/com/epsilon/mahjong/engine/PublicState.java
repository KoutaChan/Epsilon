package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 局の公開情報。{@link Engine} と {@link SeatTable} がそれぞれ 1 つ持ち、同じ手順で書き換える。席は起家を 0 とする絶対席。
 * 席のマスクでは、ビット s が席 s を表す。
 *
 * <p>書き換えるメソッドは公開のイベントごとに 1 つある。どれも、イベントの記録、見えている枚数、河の種類、立直者の現物、直前の
 * イベントをまとめて進める。見えている枚数は、河・面子・めくった表示牌・抜いた北の枚数の和。鳴かれた牌は河の側で 1 回だけ数え、
 * 面子では数えない。河の種類には鳴かれた牌も入る。立直者の現物は、その席の立直が成立した後に誰かが打った牌の種類。
 */
final class PublicState {

  private final Variant variant;
  private final EventLog events;
  private final Dora dora;
  private final TurnEvent last = new TurnEvent();
  private final int[] scores;
  private final byte[] visible = new byte[Tiles.KINDS];
  private final long[] riverKinds;
  private final long[] riichiSafe;
  private int round;
  private int honba;
  private int kyotaku;
  private int current;
  private int liveLeft;
  private int rinshanDrawn;
  private int kans;
  private int riichiMask;
  private int doubleRiichiMask;
  private int ippatsuMask;

  PublicState(Variant variant) {
    this.variant = variant;
    events = new EventLog(variant);
    dora = new Dora(variant);
    scores = new int[variant.seats()];
    riverKinds = new long[variant.seats()];
    riichiSafe = new long[variant.seats()];
  }

  Variant variant() {
    return variant;
  }

  EventLog events() {
    return events;
  }

  Dora dora() {
    return dora;
  }

  TurnEvent last() {
    return last;
  }

  int round() {
    return round;
  }

  int dealer() {
    return variant.dealer(round);
  }

  int honba() {
    return honba;
  }

  int kyotaku() {
    return kyotaku;
  }

  int score(int seat) {
    return scores[seat];
  }

  /** 手番の席（直前にツモか鳴きをした席）。 */
  int current() {
    return current;
  }

  /** 王牌を除いた山の残り枚数。ツモでも嶺上牌のツモでも 1 減る。 */
  int liveLeft() {
    return liveLeft;
  }

  /** 嶺上牌をツモった回数。槓と北抜きの後の補充を数える。 */
  int rinshanDrawn() {
    return rinshanDrawn;
  }

  /** 場の槓の数。 */
  int kans() {
    return kans;
  }

  int visible(int kind) {
    return visible[kind];
  }

  long riverKinds(int seat) {
    return riverKinds[seat];
  }

  long riichiSafe(int seat) {
    return riichiSafe[seat];
  }

  /** 立直が成立した席のマスク。 */
  int riichiMask() {
    return riichiMask;
  }

  RiichiState riichi(int seat) {
    int bit = 1 << Objects.checkIndex(seat, scores.length);
    if ((doubleRiichiMask & bit) != 0) {
      return RiichiState.DOUBLE_RIICHI;
    }
    return (riichiMask & bit) != 0 ? RiichiState.RIICHI : RiichiState.NONE;
  }

  boolean ippatsu(int seat) {
    return (ippatsuMask & 1 << Objects.checkIndex(seat, scores.length)) != 0;
  }

  /**
   * 卓の状態から局を始める。手番は親になる。局の始めの表示牌は、この後 {@link #reveal} でめくる。ルールの違う卓と、終局した卓には
   * IllegalArgumentException を投げる。
   */
  void start(TableState table) {
    if (table.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + table.variant());
    }
    if (table.gameOver()) {
      throw new IllegalArgumentException("No round starts from a finished game: " + table);
    }
    round = table.round();
    honba = table.honba();
    kyotaku = table.kyotaku();
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = table.score(seat);
    }
    current = variant.dealer(round);
    liveLeft = variant.liveWall();
    rinshanDrawn = 0;
    kans = 0;
    riichiMask = 0;
    doubleRiichiMask = 0;
    ippatsuMask = 0;
    events.clear();
    dora.clear();
    last.clear();
    Arrays.fill(visible, (byte) 0);
    Arrays.fill(riverKinds, 0);
    Arrays.fill(riichiSafe, 0);
  }

  /** 次のドラ表示牌をめくる。保留していた槓ドラがあれば、その保留は消える。 */
  void reveal(int physicalId) {
    dora.reveal(physicalId, events.size());
    see(Tiles.kind(physicalId));
  }

  /** 明槓・加槓の槓ドラを後めくりにして、保留にする。 */
  void deferDora() {
    dora.defer();
  }

  /**
   * ツモを書く。引いた牌が分からないツモ（1 席から見た卓での他家のツモ）では、物理 id に {@link TurnEvent#NONE} を渡す。
   * 山が残っていなければ IllegalStateException を投げる。
   */
  void draw(int seat, int physicalId, boolean rinshan) {
    Objects.checkIndex(seat, scores.length);
    boolean known = physicalId != TurnEvent.NONE;
    if (known) {
      requireTile(physicalId);
    }
    if (liveLeft == 0) {
      throw new IllegalStateException("The live wall is exhausted");
    }
    liveLeft--;
    if (rinshan) {
      rinshanDrawn++;
    }
    current = seat;
    int kind = known ? Tiles.kind(physicalId) : TurnEvent.NONE;
    last.setDraw(seat, kind, known && variant.tiles().red(physicalId), rinshan, dora.pending());
  }

  void discard(int seat, int physicalId, boolean tsumogiri, boolean riichi) {
    int kind = Tiles.kind(requireTile(physicalId));
    boolean red = variant.tiles().red(physicalId);
    events.appendDiscard(seat, kind, red, tsumogiri, riichi);
    see(kind);
    long bit = Tiles.bit(kind);
    riverKinds[seat] |= bit;
    for (int mask = riichiMask; mask != 0; mask &= mask - 1) {
      riichiSafe[Integer.numberOfTrailingZeros(mask)] |= bit;
    }
    last.setDiscard(seat, kind, red);
  }

  /** チー・ポン・大明槓を書く。鳴いた牌は河で数えてあるので、見えている枚数には手から出した牌だけを足す。手番は鳴いた席へ移る。 */
  void call(int seat, int meldId) {
    Meld meld = Meld.of(meldId);
    if (meld.source() == Meld.RelativeSource.NONE) {
      throw new IllegalArgumentException("Not a call: " + meld);
    }
    events.appendMeld(seat, meldId);
    boolean skipped = false;
    for (int index = 0; index < meld.size(); index++) {
      int kind = meld.tile(index);
      if (!skipped && kind == meld.called()) {
        skipped = true;
      } else {
        see(kind);
      }
    }
    if (meld.type().kan()) {
      kans++;
    }
    current = seat;
    last.clear();
  }

  void ankan(int seat, int meldId) {
    Meld meld = Meld.of(meldId);
    if (meld.type() != Meld.Type.ANKAN) {
      throw new IllegalArgumentException("Not an ankan: " + meld);
    }
    events.appendMeld(seat, meldId);
    for (int index = 0; index < meld.size(); index++) {
      see(meld.tile(index));
    }
    kans++;
    last.setKan(seat, meld.base(), false, TurnEvent.KanKind.ANKAN, dora.pending());
  }

  void kakan(int seat, int meldId) {
    events.appendKakan(seat, meldId);
    Meld meld = Meld.of(meldId);
    see(meld.base());
    kans++;
    boolean red = meld.aka() == Meld.AkaSource.ADDED_KAN_TILE;
    last.setKan(seat, meld.base(), red, TurnEvent.KanKind.KAKAN, dora.pending());
  }

  /** 北抜きを書く。抜いた北は河に入らず、抜いた時点で見えている枚数に足す（加槓の牌と同じ）。 */
  void nuki(int seat) {
    events.appendNuki(seat);
    see(Tiles.PEI);
    last.setNuki(seat, dora.pending());
  }

  /** 立直の成立を書く。供託を 1 本払い、一発が付く。既に立直している席なら IllegalStateException を投げる。 */
  void riichi(int seat, boolean doubleRiichi) {
    int bit = 1 << Objects.checkIndex(seat, scores.length);
    if ((riichiMask & bit) != 0) {
      throw new IllegalStateException("Seat " + seat + " is already in riichi");
    }
    riichiMask |= bit;
    if (doubleRiichi) {
      doubleRiichiMask |= bit;
    }
    ippatsuMask |= bit;
    kyotaku++;
    scores[seat] -= Variant.DEPOSIT;
  }

  void clearIppatsu() {
    ippatsuMask = 0;
  }

  void clearIppatsu(int seat) {
    ippatsuMask &= ~(1 << Objects.checkIndex(seat, scores.length));
  }

  void copyFrom(PublicState other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    events.copyFrom(other.events);
    dora.copyFrom(other.dora);
    last.copyFrom(other.last);
    System.arraycopy(other.scores, 0, scores, 0, scores.length);
    System.arraycopy(other.visible, 0, visible, 0, visible.length);
    System.arraycopy(other.riverKinds, 0, riverKinds, 0, riverKinds.length);
    System.arraycopy(other.riichiSafe, 0, riichiSafe, 0, riichiSafe.length);
    round = other.round;
    honba = other.honba;
    kyotaku = other.kyotaku;
    current = other.current;
    liveLeft = other.liveLeft;
    rinshanDrawn = other.rinshanDrawn;
    kans = other.kans;
    riichiMask = other.riichiMask;
    doubleRiichiMask = other.doubleRiichiMask;
    ippatsuMask = other.ippatsuMask;
  }

  /** その種類の見えている枚数を 1 増やす。ルールで決まる枚数を超えるなら、同じ牌を 2 度数えたので IllegalStateException を投げる。 */
  private void see(int kind) {
    if (visible[kind] == variant.tiles().copies(kind)) {
      throw new IllegalStateException("More than " + visible[kind] + " tiles of kind " + kind + " are visible");
    }
    visible[kind]++;
  }

  private int requireTile(int physicalId) {
    if (!variant.tiles().contains(physicalId)) {
      throw new IllegalArgumentException(variant + " has no physical tile " + physicalId);
    }
    return physicalId;
  }
}
