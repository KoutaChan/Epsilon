package com.epsilon.prism.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.VariantScoped;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 判断の記録にある公開のイベントを、入力のライターが何度も読める形に展開する。記録は判断ごとに 1 回だけ読む（イベントごとに
 * {@link Observation#readEvent} を 1 回）。持つのは、イベントごとの値、席ごとの打牌・立直・今の副露、槓ドラの表示を挟んだ公開の位置。
 * メモリを割り当てない。
 *
 * <p>番号は 2 種類ある。1 つは記録のイベントの番号で、打牌・副露・加槓・北抜きに振られる（エンジンの公開のイベントの番号と同じ）。もう 1 つは
 * 公開の位置で、記録のイベントの間に槓ドラの表示を {@link Observation#kanDoraEvent} の決まりで挟んだ順になる（入力のイベントのトークンの順）。
 * 席は自分から見た相対の席で表す。加槓の元になったポンのイベントの番号を持つために、席ごとの面子もここで数える。
 */
public final class EventTable {

  /** 公開のイベントの種類。四人打ち・三人打ちの語彙は、そのルールで起きる種類だけを宣言の順に並べる。 */
  public enum PublicEvent implements VariantScoped {
    DISCARD,
    CHI(RuleFlag.CHI),
    PON,
    DAIMINKAN,
    ANKAN,
    KAKAN,
    /** 北抜き（牌は北）。 */
    NUKI(RuleFlag.NUKI),
    /**
     * 槓ドラの表示。めくられた位置に置く（暗槓は槓の直後、大明槓・加槓は槓をした席の次の打牌か次の槓の後）。最初のドラ表示はイベントに
     * しない。
     */
    DORA_INDICATOR;

    private final RuleFlag requires;

    PublicEvent() {
      this(null);
    }

    PublicEvent(RuleFlag requires) {
      this.requires = requires;
    }

    @Override
    public RuleFlag requires() {
      return requires;
    }
  }

  private static final int NONE = -1;
  private static final PublicEvent[] MELD_EVENTS = meldEvents();

  private final PublicEvent[] kind;
  private final int[] author;
  private final int[] tile;
  private final boolean[] aka;
  private final boolean[] tsumogiri;
  private final boolean[] riichi;
  private final int[] calledBy;
  private final int[] discardIndex;
  private final Meld[] meld;
  private final int[] link;
  private final int[] position;
  private final int[] entity;
  private final int[] kanEvent = new int[Variant.MAX_KANS];
  private final int[] indicatorAfter = new int[WinContext.MAX_INDICATORS - 1];
  private final int[] discards;
  private final int[] lastDiscard;
  private final int[] riichiDeclaration;
  private final int[] meldCount;
  private final Meld[] melds;
  private final int[] meldEvent;
  private final Observation.EventRecord record = new Observation.EventRecord();
  private int events;
  private int kans;
  private int indicators;
  private int publicEvents;

  public EventTable(Variant variant) {
    int seats = variant.seats();
    int capacity = variant.maxPublicEvents();
    kind = new PublicEvent[capacity];
    author = new int[capacity];
    tile = new int[capacity];
    aka = new boolean[capacity];
    tsumogiri = new boolean[capacity];
    riichi = new boolean[capacity];
    calledBy = new int[capacity];
    discardIndex = new int[capacity];
    meld = new Meld[capacity];
    link = new int[capacity];
    position = new int[capacity];
    entity = new int[capacity + indicatorAfter.length];
    discards = new int[seats];
    lastDiscard = new int[seats];
    riichiDeclaration = new int[seats];
    meldCount = new int[seats];
    melds = new Meld[seats * Variant.MAX_MELDS];
    meldEvent = new int[seats * Variant.MAX_MELDS];
  }

  /**
   * 判断の記録のイベントを読む。
   *
   * @param o 判断の記録
   * @throws IllegalStateException 記録がイベントの決まりに合わないとき（加槓の元のポンが無い、槓ドラの表示の順がおかしいなど）
   */
  public void load(Observation o) {
    begin(o);
    for (int e = 0; e < events; e++) {
      read(o, e);
    }
    placeIndicators(o);
  }

  /**
   * 判断の記録のイベントを読み、同じ 1 回の読み取りで牌の枚数と席ごとの公開情報も作る。作る値は、それぞれの {@code load} と同じ。
   *
   * @param o 判断の記録
   * @param tiles 作り直す牌の枚数
   * @param seats 作り直す席ごとの公開情報
   * @throws IllegalStateException 記録がイベントの決まりに合わないとき（加槓の元のポンが無い、槓ドラの表示の順がおかしいなど）、または
   *     枚数・面子が上限を超えるとき
   */
  public void load(Observation o, TileCounts tiles, SeatFacts seats) {
    begin(o);
    tiles.begin(o);
    seats.begin(o);
    for (int e = 0; e < events; e++) {
      read(o, e);
      tiles.add(record);
      seats.add(record);
    }
    tiles.finish(o);
    seats.finish(o);
    placeIndicators(o);
  }

  private void begin(Observation o) {
    events = o.events();
    kans = 0;
    Arrays.fill(discards, 0);
    Arrays.fill(lastDiscard, NONE);
    Arrays.fill(riichiDeclaration, NONE);
    Arrays.fill(meldCount, 0);
  }

  private void read(Observation o, int e) {
    o.readEvent(e, record);
    int who = record.author();
    author[e] = who;
    switch (record.kind()) {
      case DISCARD -> readDiscard(e, who);
      case MELD -> readMeld(e, who);
      case NUKI -> readNuki(e);
    }
  }

  /** 槓ドラの表示を、めくられた公開の位置に挟む。 */
  private void placeIndicators(Observation o) {
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

  private void readNuki(int e) {
    kind[e] = PublicEvent.NUKI;
    tile[e] = Tiles.PEI;
    aka[e] = false;
    tsumogiri[e] = false;
    riichi[e] = false;
    calledBy[e] = NONE;
    discardIndex[e] = NONE;
    meld[e] = null;
    link[e] = NONE;
  }

  private void readMeld(int e, int who) {
    Meld created = record.meld();
    kind[e] = MELD_EVENTS[created.type().ordinal()];
    tile[e] = created.base();
    aka[e] = false;
    tsumogiri[e] = false;
    riichi[e] = false;
    calledBy[e] = NONE;
    discardIndex[e] = NONE;
    meld[e] = created;
    if (created.type() == Meld.Type.KAKAN) {
      int slot = ponSlot(who, created.base());
      link[e] = meldEvent[slot];
      melds[slot] = created;
    } else {
      int slot = who * Variant.MAX_MELDS + meldCount[who]++;
      melds[slot] = created;
      meldEvent[slot] = e;
      // 記録では、チー・ポン・大明槓は鳴かれた打牌のすぐ次の番号になる。暗槓には繋がるイベントがない。
      link[e] = created.type() == Meld.Type.ANKAN ? NONE : e - 1;
    }
    if (created.type().kan()) {
      kanEvent[kans++] = e;
    }
  }

  private int ponSlot(int who, int base) {
    for (int slot = who * Variant.MAX_MELDS, end = slot + meldCount[who]; slot < end; slot++) {
      if (melds[slot].type() == Meld.Type.PON && melds[slot].base() == base) {
        return slot;
      }
    }
    throw new IllegalStateException("kakan without its pon: tile " + base);
  }

  /** 記録のイベントの数。 */
  public int events() {
    return events;
  }

  /** 公開のイベントの数（記録のイベント ＋ 槓ドラの表示）。 */
  public int publicEvents() {
    return publicEvents;
  }

  /** 成立した槓の数。 */
  public int kans() {
    return kans;
  }

  /** 槓ドラの表示の数（最初のドラ表示は数えない）。 */
  public int indicators() {
    return indicators;
  }

  /** 記録のイベント {@code e} の種類。 */
  public PublicEvent kind(int e) {
    return kind[Objects.checkIndex(e, events)];
  }

  /** 記録のイベント {@code e} を起こした席（相対の席）。 */
  public int author(int e) {
    return author[Objects.checkIndex(e, events)];
  }

  /** 記録のイベント {@code e} の牌（打牌は捨てた牌、副露は最も小さい牌、北抜きは北）。 */
  public int tile(int e) {
    return tile[Objects.checkIndex(e, events)];
  }

  /** 打牌が赤か（副露と北抜きは false）。 */
  public boolean aka(int e) {
    return aka[Objects.checkIndex(e, events)];
  }

  /** 打牌がツモ切りか（副露と北抜きは false）。 */
  public boolean tsumogiri(int e) {
    return tsumogiri[Objects.checkIndex(e, events)];
  }

  /** 打牌が立直の宣言牌か（副露と北抜きは false）。 */
  public boolean riichi(int e) {
    return riichi[Objects.checkIndex(e, events)];
  }

  /** 打牌を鳴いた席（相対の席）。鳴かれていない打牌と、副露・北抜きは −1。 */
  public int calledBy(int e) {
    return calledBy[Objects.checkIndex(e, events)];
  }

  /** 打牌が、その席の何打目か（0 始まり）。副露と北抜きは −1。 */
  public int discardIndex(int e) {
    return discardIndex[Objects.checkIndex(e, events)];
  }

  /** 副露・加槓のイベントでできた面子（打牌と北抜きは null）。 */
  public Meld meld(int e) {
    return meld[Objects.checkIndex(e, events)];
  }

  /** 繋がる記録のイベント。チー・ポン・大明槓は鳴かれた打牌、加槓は元のポンで、ほかは −1。 */
  public int link(int e) {
    return link[Objects.checkIndex(e, events)];
  }

  /** 記録のイベント {@code e} の公開の位置（{@code e} が −1 なら −1）。 */
  public int position(int e) {
    return e == NONE ? NONE : position[Objects.checkIndex(e, events)];
  }

  /** 公開の位置 {@code p} にあるもの。記録のイベントならその番号（0 以上）、k 番目の槓ドラの表示なら −1 − k。 */
  public int entity(int p) {
    return entity[Objects.checkIndex(p, publicEvents)];
  }

  /** k 番目（0 始まり、起きた順）の槓の、記録のイベントの番号。 */
  public int kanEvent(int k) {
    return kanEvent[Objects.checkIndex(k, kans)];
  }

  /** その席が打牌した数。 */
  public int discards(int seat) {
    return discards[seat];
  }

  /** その席の最後の打牌の、記録のイベントの番号（無ければ −1）。 */
  public int lastDiscard(int seat) {
    return lastDiscard[seat];
  }

  /** その席の立直の宣言牌の、記録のイベントの番号（無ければ −1）。 */
  public int riichiDeclaration(int seat) {
    return riichiDeclaration[seat];
  }

  /** その席の今の副露の数（暗槓を含む。加槓はポンを置き換えるので数は増えない）。 */
  public int meldCount(int seat) {
    return meldCount[seat];
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
