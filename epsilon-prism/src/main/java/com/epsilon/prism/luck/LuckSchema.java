package com.epsilon.prism.luck;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.RoundLimits;
import com.epsilon.core.Tile;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.prism.Limits;
import com.epsilon.prism.codec.BitField;
import com.epsilon.prism.codec.BitLayout;
import com.epsilon.prism.codec.SchemaDigest;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * 運の網 Ω の観測 O°（偶然の出来事の直前の全ての状態と、その出来事）の形を宣言する。
 *
 * <p>O° は {@code long[]} の上に節（{@link Section}）を宣言の順に続けて置く。節ごとに欄の enum を宣言し、配置は {@link BitLayout} が
 * その enum から一度だけ作る。副露と河の記録の数は席の節に席ごとに書き、節の位置はその数から {@link #place} で出す。
 *
 * <p>O° が持つのは出来事の前の一次の事実だけ: 局・本場・供託・親・山の残り・嶺上待ち・表示牌（種と赤）、席ごとの点・立直（ダブルを
 * 含む）・一発・見逃しの振聴・手の赤、4 席の手の 34 種の枚数、副露（面子の正準 ID と出来事の番号）、河（牌・赤・ツモ切り・立直宣言・
 * 鳴かれ・出来事の番号）、出来事の種類・行為者・結果（牌の種と赤）。場風・門前・第一ツモ・槓の数・プールは O° から出せるので書かず、
 * {@link LuckObservation} が出す。
 *
 * <p>山の並び・裏ドラ・まだ開いていない表示牌は持たない。持つと結果が「プールの上で一様」でなくなり、c の平均 0 が壊れる。結果の欄
 * （{@link HeaderBits#OUTCOME_TILE}・{@link HeaderBits#OUTCOME_AKA}）は学習の選びにだけ使い、網の入力（トークン）には入れない。
 *
 * <p>席は行為者から見た相対席（0 行為者、1 下家、2 対面、3 上家）で書き、行為者だけを絶対席で書く。牌は 34 種の番号、点は
 * {@link #POINT_UNIT} 点単位で書く。範囲は規則の上限（{@link Limits}）から取り、範囲の外の値は書くときに例外にする。版の番号は持たず、
 * 形を変えれば {@link #fingerprint()} が変わる。
 */
public final class LuckSchema {

  /** 点を書く単位。 */
  public static final int POINT_UNIT = 100;

  /** 結果の数（34 種の赤でない牌と、萬・筒・索の赤 5）。番号は {@link #outcome}。 */
  public static final int OUTCOMES = Tile.NUM_TILE_TYPES + Limits.AKA_TILES;

  /** 点（{@link #POINT_UNIT} 点単位）の符号つきの幅。±327 万点まで書ける。 */
  private static final int POINT_BITS = 16;

  /** 本場の上限。規則に上限はないので O° の表し方で決め、超える局は例外にする。 */
  private static final int MAX_HONBA = 255;

  /** 色ごとの赤のマスク（AkaTileMask：bit 0 萬、1 筒、2 索）の最大。 */
  static final int AKA_MASK = (1 << Limits.AKA_TILES) - 1;

  /** 節（宣言の順）。 */
  static final Section[] SECTIONS = Section.values();

  static final BitLayout<HeaderBits> HEADER = BitLayout.of(HeaderBits.class, Long.SIZE);

  static final BitLayout<SeatBits> SEAT = BitLayout.of(SeatBits.class, Long.SIZE);

  static final BitLayout<HandBits> HAND = BitLayout.of(HandBits.class, Long.SIZE);

  /** 副露の配置（{@code EventRecords} が書く）。 */
  public static final BitLayout<MeldBits> MELD = BitLayout.of(MeldBits.class, Integer.SIZE);

  /** 河の打牌の配置（{@code EventRecords} が書く）。 */
  public static final BitLayout<DiscardBits> DISCARD =
      BitLayout.of(DiscardBits.class, Integer.SIZE);

  /** 席の節の先頭（頭の直後。頭と席は大きさが決まっているので、記録の数を読む前に引ける）。 */
  static final int SEAT_START = HEADER.longs(1);

  /** O° の {@code long} の最大（4 席の副露 16 と、1 局の打牌の最大の河）。 */
  public static final int MAX_LONGS =
      total(GameState.NUM_PLAYERS * RoundLimits.MAX_MELDS, RoundLimits.MAX_DISCARDS);

  static {
    if (SECTIONS[0] != Section.HEADER || SECTIONS[1] != Section.SEAT) {
      throw new IllegalStateException("the header and the seats must come first");
    }
  }

  private LuckSchema() {}

  /**
   * O° の指紋を返す。標本の目録に書き、読むときに照らす。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /**
   * O° の {@code long} の数を返す。副露と河の記録の数は席の節から読む。O° を写すときは、先頭からこの数の {@code long} を写す。
   *
   * @param words O° を置いた配列
   * @param base O° の先頭
   * @return {@code long} の数
   */
  public static int longs(long[] words, int base) {
    int melds = 0;
    int discards = 0;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      melds += SEAT.get(words, base + SEAT_START, seat, SeatBits.MELDS);
      discards += SEAT.get(words, base + SEAT_START, seat, SeatBits.DISCARDS);
    }
    return total(melds, discards);
  }

  /**
   * 結果の番号を返す。赤でない牌は牌の番号、赤 5 は {@code 34 + 色}。
   *
   * @param tile 牌（34 種の番号）
   * @param aka 赤
   * @return 結果の番号（{@link #OUTCOMES} 未満）
   * @throws IllegalArgumentException 赤のない牌を赤としたとき
   */
  public static int outcome(int tile, boolean aka) {
    Objects.checkIndex(tile, Tile.NUM_TILE_TYPES);
    if (!aka) {
      return tile;
    }
    if (!Tile.canBeAka(tile)) {
      throw new IllegalArgumentException("tile " + tile + " has no aka");
    }
    return Tile.NUM_TILE_TYPES + Tile.suitOf(tile);
  }

  /**
   * 結果の牌（34 種の番号）を返す。
   *
   * @param outcome 結果の番号
   * @return 牌
   */
  public static int outcomeTile(int outcome) {
    Objects.checkIndex(outcome, OUTCOMES);
    return outcome < Tile.NUM_TILE_TYPES
        ? outcome
        : Tile.M5 + (outcome - Tile.NUM_TILE_TYPES) * Tile.TILE_TYPES_PER_SUIT;
  }

  /**
   * 結果が赤 5 なら true を返す。
   *
   * @param outcome 結果の番号
   * @return 赤なら true
   */
  public static boolean outcomeAka(int outcome) {
    return Objects.checkIndex(outcome, OUTCOMES) >= Tile.NUM_TILE_TYPES;
  }

  /**
   * 節の先頭（O° の先頭からの相対の添字）を {@code start} に入れ、O° の {@code long} の数を返す。
   *
   * @param melds 副露の記録の数（4 席の和）
   * @param discards 河の記録の数（4 席の和）
   * @param start 節の先頭を入れる配列（{@link Section} の ordinal で引く）
   * @return O° の {@code long} の数
   */
  static int place(int melds, int discards, int[] start) {
    int at = 0;
    for (Section section : SECTIONS) {
      start[section.ordinal()] = at;
      at += section.longs(melds, discards);
    }
    return at;
  }

  /** O° の {@code long} の数。 */
  static int total(int melds, int discards) {
    int at = 0;
    for (Section section : SECTIONS) {
      at += section.longs(melds, discards);
    }
    return at;
  }

  /**
   * 点を {@link #POINT_UNIT} 点単位にする。
   *
   * @param points 点
   * @return 書く値
   * @throws IllegalArgumentException 単位で割り切れないとき
   */
  static int encodePoints(int points) {
    if (points % POINT_UNIT != 0) {
      throw new IllegalArgumentException(
          "points are not a multiple of " + POINT_UNIT + ": " + points);
    }
    return points / POINT_UNIT;
  }

  /** {@link #encodePoints} で書いた値を点に戻す。 */
  static int decodePoints(int stored) {
    return stored * POINT_UNIT;
  }

  /** O° の節。宣言の順に {@code long} の境で続けて置く。 */
  public enum Section {
    /** 局面と出来事の頭（1 個）。 */
    HEADER,
    /** 席（行為者から見た相対席の順に 4 個）。副露と河の記録の数を持つ。 */
    SEAT,
    /** 手（副露を除く。相対席の順に 4 個）。 */
    HAND,
    /** 副露（相対席の順、席の中は副露の順）。 */
    MELD,
    /** 河の打牌（相対席の順、席の中は河の順）。 */
    DISCARD;

    /** この節の {@code long} の数。 */
    int longs(int melds, int discards) {
      return switch (this) {
        case HEADER -> LuckSchema.HEADER.longs(1);
        case SEAT -> LuckSchema.SEAT.longs(GameState.NUM_PLAYERS);
        case HAND -> LuckSchema.HAND.longs(GameState.NUM_PLAYERS);
        case MELD -> LuckSchema.MELD.longs(melds);
        case DISCARD -> LuckSchema.DISCARD.longs(discards);
      };
    }
  }

  /**
   * 欄の値の範囲と繰り返し。幅は {@link BitLayout} が範囲から出す。
   *
   * @param min 最小の値（なしがありうる値は −1）
   * @param max 最大の値
   * @param count 同じ範囲の値を並べる数（牌ごと・表示ごと）
   * @param values 値の enum（enum の ordinal を書く欄だけ。ほかは null）。指紋はここから値の enum を集める
   */
  record Width(int min, int max, int count, Class<? extends Enum<?>> values) {

    /** 2 値（0 は偽、1 は真）。 */
    static Width flag() {
      return upTo(1);
    }

    /** 0〜max の数・番号。 */
    static Width upTo(int max) {
      return range(0, max);
    }

    /** min〜max の数。 */
    static Width range(int min, int max) {
      return new Width(min, max, 1, null);
    }

    /** なしがありうる 0〜max の番号（なしは −1）。 */
    static Width optional(int max) {
      return range(-1, max);
    }

    /** enum の ordinal。 */
    static Width ordinal(Class<? extends Enum<?>> type) {
      return new Width(0, type.getEnumConstants().length - 1, 1, type);
    }

    /** {@code bits} ビットの 2 の補数の範囲の符号つきの数。 */
    static Width signed(int bits) {
      return range(-(1 << (bits - 1)), (1 << (bits - 1)) - 1);
    }

    /** 同じ範囲の値を count 個並べる。 */
    Width times(int count) {
      return new Width(min, max, count, values);
    }
  }

  /** O° の欄。範囲は {@link Width} が持つ。 */
  sealed interface Field extends BitField
      permits HeaderBits, SeatBits, HandBits, MeldBits, DiscardBits {

    /** 範囲と繰り返し。 */
    Width width();

    @Override
    default int min() {
      return width().min();
    }

    @Override
    default int max() {
      return width().max();
    }

    @Override
    default int count() {
      return width().count();
    }
  }

  /** 頭（1 個、64 bit の単位）。 */
  enum HeaderBits implements Field {
    EVENT(Width.ordinal(LuckEvent.class)),
    /** 行為者（絶対席。O° の中でこの欄だけが絶対席）。 */
    ACTOR(Width.upTo(GameState.NUM_PLAYERS - 1)),
    /** 親の相対席。 */
    DEALER(Width.upTo(GameState.NUM_PLAYERS - 1)),
    /** 局（東 1 局 0 … 西 4 局 11）。場風は局から出す。 */
    KYOKU(Width.upTo(HanchanProgression.WEST_END_KYOKU)),
    HONBA(Width.upTo(MAX_HONBA)),
    /** 供託の本数。 */
    KYOTAKU(Width.upTo(Limits.MAX_KYOTAKU)),
    /** 王牌を除く山の残り（出来事の前）。 */
    WALL(Width.upTo(RoundLimits.MAX_WALL)),
    /**
     * 嶺上待ち（槓をして、その嶺上のツモがまだ）。嶺上のツモでは真、生きた山のツモでは偽。槓ドラの表示では、嶺上のツモの前に開く表示
     * （暗槓の表示と、次の槓で開く遅れた表示）で真、打牌の後に開く表示で偽。
     */
    RINSHAN_PENDING(Width.flag()),
    /** 出来事の前に開いているドラ表示の数（初めの表示を含むので 1 以上）。 */
    INDICATORS(Width.range(1, Limits.MAX_INDICATORS)),
    /** 表示の順のドラ表示の牌（数より後は 0）。 */
    INDICATOR(Width.upTo(Tile.NUM_TILE_TYPES - 1).times(Limits.MAX_INDICATORS)),
    /** 表示の順のドラ表示が赤（数より後は 0）。 */
    INDICATOR_AKA(Width.flag().times(Limits.MAX_INDICATORS)),
    /** 結果の牌（引いた牌か、開いた表示牌の種）。網の入力には入れない。 */
    OUTCOME_TILE(Width.upTo(Tile.NUM_TILE_TYPES - 1)),
    /** 結果の牌が赤。網の入力には入れない。 */
    OUTCOME_AKA(Width.flag());

    private final Width width;

    HeaderBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 席（行為者から見た相対席の順に 4 個、64 bit の単位）。 */
  enum SeatBits implements Field {
    /** 点（{@link #POINT_UNIT} 点単位）。 */
    SCORE(Width.signed(POINT_BITS)),
    /** 立直の状態（ダブル立直を含む）。 */
    RIICHI(Width.ordinal(RiichiState.class)),
    IPPATSU(Width.flag()),
    /** 見逃しの振聴（同巡の振聴と立直後の振聴を兼ねる。捨て牌の振聴は河から出せるので含まない）。 */
    FURITEN(Width.flag()),
    /** 手（副露を除く）の色ごとの赤。 */
    HAND_AKA(Width.upTo(AKA_MASK)),
    /** 副露（暗槓を含む）の数（{@link Section#MELD} のこの席の記録の数）。 */
    MELDS(Width.upTo(RoundLimits.MAX_MELDS)),
    /** 河の打牌の数（鳴かれた打牌を含む。{@link Section#DISCARD} のこの席の記録の数）。 */
    DISCARDS(Width.upTo(RoundLimits.MAX_DISCARDS));

    private final Width width;

    SeatBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 手（副露を除く。相対席の順に 4 個、64 bit の単位）。 */
  enum HandBits implements Field {
    /** 牌ごとの枚数（繰り返しの番号は牌）。 */
    COUNT(Width.upTo(Tile.TILES_PER_TYPE).times(Tile.NUM_TILE_TYPES));

    private final Width width;

    HandBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 副露（32 bit の単位）。 */
  public enum MeldBits implements Field {
    /** 面子の正準 ID（種類・牌・元・赤の由来。加槓は加槓の面子）。 */
    MELD(Width.upTo(Limits.MELD_IDS - 1)),
    /** 面子ができた公開の出来事の番号（加槓は元のポンの番号）。 */
    EVENT(Width.upTo(RoundLimits.MAX_PUBLIC_EVENTS - 1)),
    /** 加槓の公開の出来事の番号（加槓でなければ −1）。 */
    KAKAN_EVENT(Width.optional(RoundLimits.MAX_PUBLIC_EVENTS - 1));

    private final Width width;

    MeldBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 河の打牌（32 bit の単位）。 */
  public enum DiscardBits implements Field {
    TILE(Width.upTo(Tile.NUM_TILE_TYPES - 1)),
    AKA(Width.flag()),
    TSUMOGIRI(Width.flag()),
    /** 立直の宣言牌。 */
    RIICHI(Width.flag()),
    /** 鳴かれた（牌は鳴いた席の面子にあり、その面子の出来事の番号はこの打牌の次）。 */
    CALLED(Width.flag()),
    /** 公開の出来事の番号（打牌・副露・加槓の共通の連番）。 */
    EVENT(Width.upTo(RoundLimits.MAX_PUBLIC_EVENTS - 1));

    private final Width width;

    DiscardBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 遅延で作る（初期化の順の罠を避ける）。 */
  private static final class Fingerprint {

    private static final String VALUE = describe().hex();

    /** 節の順、全ての配置（欄の範囲と位置）、欄が使う値の enum、面子の符号、牌の数・点の単位・結果の数。 */
    private static SchemaDigest describe() {
      SchemaDigest digest = SchemaDigest.named("epsilon-prism-v2-luck");
      digest.values(Section.class);
      digest.layout(HEADER).layout(SEAT).layout(HAND).layout(MELD).layout(DISCARD);
      Set<Class<? extends Enum<?>>> values = new LinkedHashSet<>();
      for (Class<?> type : SchemaDigest.sorted(Field.class.getPermittedSubclasses())) {
        for (Object constant : type.getEnumConstants()) {
          Class<? extends Enum<?>> enumType = ((Field) constant).width().values();
          if (enumType != null) {
            values.add(enumType);
          }
        }
      }
      for (Class<? extends Enum<?>> type : values) {
        digest.values(type);
      }
      return digest
          .values(Meld.Type.class)
          .values(Meld.RelativeSource.class)
          .values(Meld.AkaSource.class)
          .with("meldIds", Limits.MELD_IDS)
          .with("tiles", Tile.NUM_TILE_TYPES)
          .with("pointUnit", POINT_UNIT)
          .with("outcomes", OUTCOMES);
    }
  }
}
