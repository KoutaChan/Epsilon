package com.epsilon.prism.luck;

import static com.epsilon.prism.luck.LuckSchema.DISCARD;
import static com.epsilon.prism.luck.LuckSchema.HAND;
import static com.epsilon.prism.luck.LuckSchema.HEADER;
import static com.epsilon.prism.luck.LuckSchema.MELD;
import static com.epsilon.prism.luck.LuckSchema.SEAT;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.prism.luck.LuckSchema.DiscardBits;
import com.epsilon.prism.luck.LuckSchema.HandBits;
import com.epsilon.prism.luck.LuckSchema.HeaderBits;
import com.epsilon.prism.luck.LuckSchema.MeldBits;
import com.epsilon.prism.luck.LuckSchema.SeatBits;
import com.epsilon.prism.luck.LuckSchema.Section;
import java.util.Arrays;
import java.util.Objects;

/**
 * 観測 O°（{@link LuckSchema}）の読み手。{@link #bind} で別の O° に付け替えて使い回し、読むときに割り当てをしない（flyweight）。
 * 付け替えるので、読むスレッドごとに 1 つ持つ。
 *
 * <p>席は行為者から見た相対席（0 行為者、1 下家、2 対面、3 上家）で返し、絶対席は {@link #absoluteSeat} で戻す。O° に書かない値
 * （場風・門前・第一ツモ・槓の数・プール）もここで出す。プールは出来事の前に全員から見えない牌（4 − 公開 − 4 席の手）で、結果はこの上で
 * 一様に出る。
 *
 * <p>結果（{@link #outcomeTile}・{@link #outcomeAka}・{@link #outcome}）は学習の選びにだけ使い、網の入力には入れない。
 */
public final class LuckObservation {

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final LuckEvent[] EVENTS = LuckEvent.values();
  private static final RiichiState[] RIICHI_STATES = RiichiState.values();

  private final int[] start = new int[LuckSchema.SECTIONS.length];
  private final int[] firstMeld = new int[SEATS + 1];
  private final int[] firstDiscard = new int[SEATS + 1];
  private final int[] pool = new int[Tile.NUM_TILE_TYPES];
  private long[] words;
  private int base;
  private int longs;
  private boolean pooled;
  private int poolAka;
  private int poolTotal;

  /**
   * 読む O° を付け替える。
   *
   * @param words O° を置いた配列
   * @param base O° の先頭
   * @return この読み手
   * @throws IndexOutOfBoundsException O° が配列に収まらないとき
   */
  public LuckObservation bind(long[] words, int base) {
    this.words = Objects.requireNonNull(words, "words");
    this.base = base;
    Objects.checkFromIndexSize(
        base, LuckSchema.SEAT_START + SEAT.longs(SEATS), words.length);
    for (int seat = 0; seat < SEATS; seat++) {
      firstMeld[seat + 1] = firstMeld[seat] + seat(seat, SeatBits.MELDS);
      firstDiscard[seat + 1] = firstDiscard[seat] + seat(seat, SeatBits.DISCARDS);
    }
    longs = LuckSchema.place(firstMeld[SEATS], firstDiscard[SEATS], start);
    Objects.checkFromIndexSize(base, longs, words.length);
    pooled = false;
    return this;
  }

  /** O° の {@code long} の数。 */
  public int longs() {
    return longs;
  }

  public LuckEvent event() {
    return EVENTS[header(HeaderBits.EVENT)];
  }

  /** 行為者（絶対席）。 */
  public int actor() {
    return header(HeaderBits.ACTOR);
  }

  /** 相対席を絶対席に戻す。 */
  public int absoluteSeat(int seat) {
    return (actor() + Objects.checkIndex(seat, SEATS)) % SEATS;
  }

  /** 親の相対席。 */
  public int dealer() {
    return header(HeaderBits.DEALER);
  }

  /** 局（東 1 局 0 … 西 4 局 11）。 */
  public int kyoku() {
    return header(HeaderBits.KYOKU);
  }

  /** 場風の牌（{@link Tile#TON} … {@link Tile#PEI}）。局から出す（4 局ごとに次の風）。 */
  public int roundWindTile() {
    return Tile.TON + kyoku() / SEATS;
  }

  public int honba() {
    return header(HeaderBits.HONBA);
  }

  /** 供託の本数。 */
  public int kyotaku() {
    return header(HeaderBits.KYOTAKU);
  }

  /** 王牌を除く山の残り（出来事の前）。 */
  public int remainingWall() {
    return header(HeaderBits.WALL);
  }

  /** 嶺上待ち（槓をして、その嶺上のツモがまだ）。 */
  public boolean rinshanPending() {
    return header(HeaderBits.RINSHAN_PENDING) != 0;
  }

  /** 出来事の前に開いているドラ表示の数（初めの表示を含む）。 */
  public int indicators() {
    return header(HeaderBits.INDICATORS);
  }

  /** 表示の順の {@code index} 番目のドラ表示の牌。 */
  public int indicatorTile(int index) {
    return HEADER.get(
        words, base, 0, HeaderBits.INDICATOR, Objects.checkIndex(index, indicators()));
  }

  /** 表示の順の {@code index} 番目のドラ表示が赤。 */
  public boolean indicatorAka(int index) {
    return HEADER.get(
            words, base, 0, HeaderBits.INDICATOR_AKA, Objects.checkIndex(index, indicators()))
        != 0;
  }

  /** 結果の牌（34 種の番号）。網の入力には入れない。 */
  public int outcomeTile() {
    return header(HeaderBits.OUTCOME_TILE);
  }

  /** 結果の牌が赤。網の入力には入れない。 */
  public boolean outcomeAka() {
    return header(HeaderBits.OUTCOME_AKA) != 0;
  }

  /** 結果の番号（{@link LuckSchema#outcome}）。網の入力には入れない。 */
  public int outcome() {
    return LuckSchema.outcome(outcomeTile(), outcomeAka());
  }

  /** 相対席 {@code seat} の点。 */
  public int score(int seat) {
    return LuckSchema.decodePoints(seat(seat, SeatBits.SCORE));
  }

  /** 相対席 {@code seat} の立直の状態。 */
  public RiichiState riichi(int seat) {
    return RIICHI_STATES[seat(seat, SeatBits.RIICHI)];
  }

  /** 相対席 {@code seat} に一発がある。 */
  public boolean ippatsu(int seat) {
    return seat(seat, SeatBits.IPPATSU) != 0;
  }

  /** 相対席 {@code seat} の見逃しの振聴（同巡の振聴と立直後の振聴。捨て牌の振聴は河から出す）。 */
  public boolean furiten(int seat) {
    return seat(seat, SeatBits.FURITEN) != 0;
  }

  /** 相対席 {@code seat} の手（副露を除く）の色ごとの赤（AkaTileMask：bit 0 萬、1 筒、2 索）。 */
  public int handAkaMask(int seat) {
    return seat(seat, SeatBits.HAND_AKA);
  }

  /** 相対席 {@code seat} の副露（暗槓を含む）の数。 */
  public int meldCount(int seat) {
    return seat(seat, SeatBits.MELDS);
  }

  /** 相対席 {@code seat} の河の打牌の数（鳴かれた打牌を含む）。 */
  public int discardCount(int seat) {
    return seat(seat, SeatBits.DISCARDS);
  }

  /** 相対席 {@code seat} が門前（暗槓のほかに副露がない）。 */
  public boolean menzen(int seat) {
    for (int index = 0, count = meldCount(seat); index < count; index++) {
      if (!meld(seat, index).preservesMenzen()) {
        return false;
      }
    }
    return true;
  }

  /** 相対席 {@code seat} がまだ打牌していない（第一ツモの旗。エンジンの {@code isFirstDraw} と同じ）。 */
  public boolean firstDraw(int seat) {
    return discardCount(seat) == 0;
  }

  /** 4 席の槓の数。 */
  public int kans() {
    int kans = 0;
    for (int seat = 0; seat < SEATS; seat++) {
      for (int index = 0, count = meldCount(seat); index < count; index++) {
        if (meld(seat, index).isKan()) {
          kans++;
        }
      }
    }
    return kans;
  }

  /** 相対席 {@code seat} の手（副露を除く）の牌 {@code tile} の枚数。 */
  public int handCount(int seat, int tile) {
    return HAND.get(
        words,
        start(Section.HAND),
        Objects.checkIndex(seat, SEATS),
        HandBits.COUNT,
        Objects.checkIndex(tile, Tile.NUM_TILE_TYPES));
  }

  /** 相対席 {@code seat} の {@code index} 番目の副露（共有の表の面子）。 */
  public Meld meld(int seat, int index) {
    return Meld.fromCanonicalId(meldField(seat, index, MeldBits.MELD));
  }

  /** 副露ができた公開の出来事の番号（加槓は元のポンの番号）。 */
  public int meldEvent(int seat, int index) {
    return meldField(seat, index, MeldBits.EVENT);
  }

  /** 加槓の公開の出来事の番号（加槓でなければ −1）。 */
  public int meldKakanEvent(int seat, int index) {
    return meldField(seat, index, MeldBits.KAKAN_EVENT);
  }

  /** 相対席 {@code seat} の河の {@code index} 番目の打牌の牌。 */
  public int discardTile(int seat, int index) {
    return discardField(seat, index, DiscardBits.TILE);
  }

  /** 打牌が赤。 */
  public boolean discardAka(int seat, int index) {
    return discardField(seat, index, DiscardBits.AKA) != 0;
  }

  /** 打牌がツモ切り。 */
  public boolean discardTsumogiri(int seat, int index) {
    return discardField(seat, index, DiscardBits.TSUMOGIRI) != 0;
  }

  /** 打牌が立直の宣言牌。 */
  public boolean discardRiichi(int seat, int index) {
    return discardField(seat, index, DiscardBits.RIICHI) != 0;
  }

  /** 打牌が鳴かれた（牌は鳴いた席の面子にある）。 */
  public boolean discardCalled(int seat, int index) {
    return discardField(seat, index, DiscardBits.CALLED) != 0;
  }

  /** 打牌の公開の出来事の番号。 */
  public int discardEvent(int seat, int index) {
    return discardField(seat, index, DiscardBits.EVENT);
  }

  /**
   * 出来事の前に全員から見えない牌 {@code tile} の枚数（赤を含む）を返す。4 − 4 席の手 − 副露 − 鳴かれていない河 − 開いた表示。
   *
   * @param tile 牌
   * @return 枚数
   * @throws IllegalStateException O° が数え方と合わない（枚数が負か、赤を二度数えた）とき
   */
  public int pool(int tile) {
    scanPool();
    return pool[Objects.checkIndex(tile, Tile.NUM_TILE_TYPES)];
  }

  /** 全員から見えない赤 5 の色のマスク（AkaTileMask）。 */
  public int poolAkaMask() {
    scanPool();
    return poolAka;
  }

  /** プールの枚数の和 N。 */
  public int poolTotal() {
    scanPool();
    return poolTotal;
  }

  /**
   * 結果 {@code outcome}（{@link LuckSchema#outcome}）のプールの枚数を返す。結果の確率はこれを {@link #poolTotal} で割ったもの。
   *
   * @param outcome 結果の番号
   * @return 枚数
   */
  public int poolOf(int outcome) {
    int tile = LuckSchema.outcomeTile(outcome);
    boolean akaUnseen = AkaTileMask.containsTile(poolAkaMask(), tile);
    if (LuckSchema.outcomeAka(outcome)) {
      return akaUnseen ? 1 : 0;
    }
    return pool(tile) - (akaUnseen ? 1 : 0);
  }

  /** プールを 1 回だけ数える（付け替えると数え直す）。 */
  private void scanPool() {
    if (pooled) {
      return;
    }
    Arrays.fill(pool, Tile.TILES_PER_TYPE);
    int aka = LuckSchema.AKA_MASK;
    for (int seat = 0; seat < SEATS; seat++) {
      for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
        pool[tile] -= handCount(seat, tile);
      }
      int handAka = handAkaMask(seat);
      if ((aka & handAka) != handAka) {
        throw new IllegalStateException("aka counted twice in the hand of seat " + seat);
      }
      aka &= ~handAka;
      for (int index = 0, count = meldCount(seat); index < count; index++) {
        Meld meld = meld(seat, index);
        for (int at = 0; at < meld.size(); at++) {
          pool[meld.tileAt(at)]--;
        }
        if (meld.containsAkaTile()) {
          aka = seen(aka, meld.akaTileType());
        }
      }
      for (int index = 0, count = discardCount(seat); index < count; index++) {
        if (discardCalled(seat, index)) {
          continue;
        }
        int tile = discardTile(seat, index);
        pool[tile]--;
        if (discardAka(seat, index)) {
          aka = seen(aka, tile);
        }
      }
    }
    for (int index = 0, count = indicators(); index < count; index++) {
      int tile = indicatorTile(index);
      pool[tile]--;
      if (indicatorAka(index)) {
        aka = seen(aka, tile);
      }
    }
    int total = 0;
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      if (pool[tile] < 0) {
        throw new IllegalStateException("negative unseen count of tile " + Tile.name(tile));
      }
      if (pool[tile] == 0 && AkaTileMask.containsTile(aka, tile)) {
        throw new IllegalStateException("unseen aka " + Tile.name(tile) + " has no unseen tile");
      }
      total += pool[tile];
    }
    poolAka = aka;
    poolTotal = total;
    pooled = true;
  }

  /** 見えた赤 5 を外す（二度見えたら O° の誤り）。 */
  private static int seen(int aka, int tile) {
    if (!AkaTileMask.containsTile(aka, tile)) {
      throw new IllegalStateException("aka " + Tile.name(tile) + " counted twice");
    }
    return AkaTileMask.excludeTile(aka, tile);
  }

  private int start(Section section) {
    return base + start[section.ordinal()];
  }

  private int header(HeaderBits field) {
    return HEADER.get(words, base, 0, field);
  }

  private int seat(int seat, SeatBits field) {
    return SEAT.get(words, base + LuckSchema.SEAT_START, Objects.checkIndex(seat, SEATS), field);
  }

  private int meldField(int seat, int index, MeldBits field) {
    int count = meldCount(seat);
    int record = firstMeld[seat] + Objects.checkIndex(index, count);
    return MELD.get(words, start(Section.MELD), record, field);
  }

  private int discardField(int seat, int index, DiscardBits field) {
    int count = discardCount(seat);
    int record = firstDiscard[seat] + Objects.checkIndex(index, count);
    return DISCARD.get(words, start(Section.DISCARD), record, field);
  }
}
