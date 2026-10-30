package com.epsilon.prism.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.RoundLimits;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputSchema.PublicEvent;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationSchema.EventKind;
import java.util.Arrays;
import java.util.Objects;

/**
 * 記録 O の公開の出来事を、Tokenizer が何度も読む形にほどく。判断ごとに 1 回だけ O を読み（出来事ごとに記録を 1 回、
 * {@link Observation#readEvent}）、出来事ごとの値と、席ごとの打牌・立直・今の副露と、槓ドラの表示を挟んだ公開の位置を持つ。割り当てをしない。
 *
 * <p>番号は 2 種類ある。O の出来事の番号（打牌・副露・加槓。エンジンの公開の出来事の番号）と、公開の位置（O の出来事の間に槓ドラの表示を
 * {@link Observation#kanDoraEvent} の決まりで挟んだ順。入力の出来事トークンの順）。席は観測席から見た相対席。
 */
final class EventTable {

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int KAN_DORA = Limits.MAX_INDICATORS - 1;
  private static final int NONE = -1;
  private static final PublicEvent[] MELD_EVENTS = meldEvents();

  private final PublicEvent[] kind = new PublicEvent[RoundLimits.MAX_PUBLIC_EVENTS];
  private final int[] author = new int[RoundLimits.MAX_PUBLIC_EVENTS];
  private final int[] tile = new int[RoundLimits.MAX_PUBLIC_EVENTS];
  private final boolean[] aka = new boolean[RoundLimits.MAX_PUBLIC_EVENTS];
  private final boolean[] tsumogiri = new boolean[RoundLimits.MAX_PUBLIC_EVENTS];
  private final boolean[] riichi = new boolean[RoundLimits.MAX_PUBLIC_EVENTS];
  private final int[] calledBy = new int[RoundLimits.MAX_PUBLIC_EVENTS];
  private final int[] discardIndex = new int[RoundLimits.MAX_PUBLIC_EVENTS];
  private final Meld[] meld = new Meld[RoundLimits.MAX_PUBLIC_EVENTS];
  private final int[] link = new int[RoundLimits.MAX_PUBLIC_EVENTS];
  private final int[] position = new int[RoundLimits.MAX_PUBLIC_EVENTS];
  private final int[] entity = new int[Limits.MAX_EVENT_TOKENS];
  private final int[] kanEvent = new int[RoundLimits.MAX_KANS];
  private final int[] indicatorAfter = new int[KAN_DORA];
  private final int[] discards = new int[SEATS];
  private final int[] lastDiscard = new int[SEATS];
  private final int[] riichiDeclaration = new int[SEATS];
  private final int[] meldCount = new int[SEATS];
  private final Meld[] melds = new Meld[SEATS * RoundLimits.MAX_MELDS];
  private final int[] meldEvent = new int[SEATS * RoundLimits.MAX_MELDS];
  private final Observation.EventRecord record = new Observation.EventRecord();
  private int events;
  private int kans;
  private int indicators;
  private int publicEvents;

  /**
   * O の出来事を読む。
   *
   * @param o 判断の記録
   * @throws IllegalStateException O が出来事の決まり（加槓の元のポン、槓ドラの表示の順）と合わないとき
   */
  void load(Observation o) {
    events = o.events();
    kans = 0;
    Arrays.fill(discards, 0);
    Arrays.fill(lastDiscard, NONE);
    Arrays.fill(riichiDeclaration, NONE);
    Arrays.fill(meldCount, 0);
    for (int e = 0; e < events; e++) {
      o.readEvent(e, record);
      int who = record.author();
      author[e] = who;
      if (record.kind() == EventKind.DISCARD) {
        readDiscard(e, who);
      } else {
        readMeld(e, who);
      }
    }
    indicators = o.kanDoraIndicators();
    if (indicators > kans) {
      throw new IllegalStateException(indicators + " kan dora indicators for " + kans + " kans");
    }
    for (int k = 0; k < indicators; k++) {
      indicatorAfter[k] = o.kanDoraEvent(k);
      if (k > 0 && indicatorAfter[k] < indicatorAfter[k - 1]) {
        throw new IllegalStateException("kan dora indicators are not in public order");
      }
    }
    int next = 0;
    int at = 0;
    for (int e = 0; e < events; e++) {
      position[e] = at;
      entity[at++] = e;
      while (next < indicators && indicatorAfter[next] == e) {
        entity[at++] = -1 - next;
        next++;
      }
    }
    publicEvents = at;
  }

  private void readDiscard(int e, int who) {
    kind[e] = PublicEvent.DISCARD;
    tile[e] = record.discardTile();
    aka[e] = record.discardAka();
    tsumogiri[e] = record.discardTsumogiri();
    riichi[e] = record.discardRiichi();
    calledBy[e] = record.discardCalledBy();
    discardIndex[e] = discards[who]++;
    meld[e] = null;
    link[e] = NONE;
    lastDiscard[who] = e;
    if (riichi[e] && riichiDeclaration[who] == NONE) {
      riichiDeclaration[who] = e;
    }
  }

  private void readMeld(int e, int who) {
    Meld created = record.meld();
    kind[e] = MELD_EVENTS[created.type().ordinal()];
    tile[e] = created.baseTileType();
    aka[e] = false;
    tsumogiri[e] = false;
    riichi[e] = false;
    calledBy[e] = NONE;
    discardIndex[e] = NONE;
    meld[e] = created;
    if (created.type() == Meld.Type.KAKAN) {
      int slot = ponSlot(who, created.baseTileType());
      link[e] = meldEvent[slot];
      melds[slot] = created;
    } else {
      int slot = who * RoundLimits.MAX_MELDS + meldCount[who]++;
      melds[slot] = created;
      meldEvent[slot] = e;
      // チー・ポン・大明槓は鳴かれた打牌の直後の番号（O の決まり）。暗槓は繋がる出来事がない。
      link[e] = created.type() == Meld.Type.ANKAN ? NONE : e - 1;
    }
    if (created.isKan()) {
      kanEvent[kans++] = e;
    }
  }

  private int ponSlot(int who, int baseTile) {
    for (int slot = who * RoundLimits.MAX_MELDS, end = slot + meldCount[who]; slot < end; slot++) {
      if (melds[slot].type() == Meld.Type.PON && melds[slot].baseTileType() == baseTile) {
        return slot;
      }
    }
    throw new IllegalStateException("kakan without its pon: tile " + baseTile);
  }

  /** O の出来事の数。 */
  int events() {
    return events;
  }

  /** 公開の出来事の数（O の出来事 ＋ 槓ドラの表示）。 */
  int publicEvents() {
    return publicEvents;
  }

  /** 成立した槓の数。 */
  int kans() {
    return kans;
  }

  /** 槓ドラの表示の数（初めのドラ表示を除く）。 */
  int indicators() {
    return indicators;
  }

  /** O の出来事 e の種類。 */
  PublicEvent kind(int e) {
    return kind[Objects.checkIndex(e, events)];
  }

  /** O の出来事 e の作者の相対席。 */
  int author(int e) {
    return author[Objects.checkIndex(e, events)];
  }

  /** O の出来事 e の牌（打牌の牌、副露の最小の牌）。 */
  int tile(int e) {
    return tile[Objects.checkIndex(e, events)];
  }

  /** 打牌が赤（副露は false）。 */
  boolean aka(int e) {
    return aka[Objects.checkIndex(e, events)];
  }

  /** 打牌がツモ切り（副露は false）。 */
  boolean tsumogiri(int e) {
    return tsumogiri[Objects.checkIndex(e, events)];
  }

  /** 打牌が立直の宣言牌（副露は false）。 */
  boolean riichi(int e) {
    return riichi[Objects.checkIndex(e, events)];
  }

  /** 打牌を鳴いた相対席（鳴かれていない打牌と副露は −1）。 */
  int calledBy(int e) {
    return calledBy[Objects.checkIndex(e, events)];
  }

  /** 打牌の作者の何打目か（0 始まり。副露は −1）。 */
  int discardIndex(int e) {
    return discardIndex[Objects.checkIndex(e, events)];
  }

  /** 副露・加槓の出来事の時点の面子（打牌は null）。 */
  Meld meld(int e) {
    return meld[Objects.checkIndex(e, events)];
  }

  /** 繋がる O の出来事（チー・ポン・大明槓は鳴かれた打牌、加槓は元のポン。ほかは −1）。 */
  int link(int e) {
    return link[Objects.checkIndex(e, events)];
  }

  /** O の出来事 e の公開の位置（e が −1 なら −1）。 */
  int position(int e) {
    return e == NONE ? NONE : position[Objects.checkIndex(e, events)];
  }

  /** 公開の位置 p の出来事。O の出来事なら番号（0 以上）、槓ドラの表示 k なら −1 − k。 */
  int entity(int p) {
    return entity[Objects.checkIndex(p, publicEvents)];
  }

  /** k 番目（0 始まり、出来事の順）の槓の O の出来事。 */
  int kanEvent(int k) {
    return kanEvent[Objects.checkIndex(k, kans)];
  }

  /** 席の打牌の数。 */
  int discards(int seat) {
    return discards[seat];
  }

  /** 席の最後の打牌の O の出来事（なしは −1）。 */
  int lastDiscard(int seat) {
    return lastDiscard[seat];
  }

  /** 席の立直宣言の打牌の O の出来事（なしは −1）。 */
  int riichiDeclaration(int seat) {
    return riichiDeclaration[seat];
  }

  /** 席の今の副露の数（暗槓を含み、加槓はポンを置き換える）。 */
  int meldCount(int seat) {
    return meldCount[seat];
  }

  /** 席の今の副露の index 番目（成立の順。加槓したポンは加槓）。 */
  Meld meld(int seat, int index) {
    return melds[seat * RoundLimits.MAX_MELDS + Objects.checkIndex(index, meldCount[seat])];
  }

  private static PublicEvent[] meldEvents() {
    Meld.Type[] types = Meld.Type.values();
    PublicEvent[] events = new PublicEvent[types.length];
    for (Meld.Type type : types) {
      events[type.ordinal()] =
          switch (type) {
            case CHI -> PublicEvent.CHI;
            case PON -> PublicEvent.PON;
            case DAIMINKAN -> PublicEvent.DAIMINKAN;
            case ANKAN -> PublicEvent.ANKAN;
            case KAKAN -> PublicEvent.KAKAN;
          };
    }
    return events;
  }
}
