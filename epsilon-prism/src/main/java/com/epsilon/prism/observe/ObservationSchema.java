package com.epsilon.prism.observe;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.ScoreMath;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.RoundLimits;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.WinSettlementProjection.SettlementAssumption;
import com.epsilon.prism.Limits;
import com.epsilon.prism.codec.BitField;
import com.epsilon.prism.codec.BitLayout;
import com.epsilon.prism.codec.SchemaDigest;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * 記録 O（判断の時点の一次の事実）の形を宣言する。
 *
 * <p>O は {@code long[]} の上に節（{@link Section}）を宣言の順に続けて置く。節ごとに欄の enum を宣言し、配置は {@link BitLayout} が
 * その enum から一度だけ作る。可変の節の記録の数は頭に書き、節の位置は頭の数から {@link #place} で一度だけ出す。O はどこでも
 * {@code long[]} のまま持ち、写すときは {@link #longs} の数の {@code long} を {@link System#arraycopy} で写す（バイト列への書き出しは
 * 行の codec が行う）。
 *
 * <p>O が持つのは {@code PublicObservation} が返す一次の値と合法手、エンジンの判定（チー・ポンの後の打牌、即時の和了の精算）だけ。
 * ほかの欄から出せる値（場風・初巡・鳴きの有無・第一打の前・ツモ牌）と、後の手・見える枚数・向聴・待ち・安全・支払い・双子・槓ドラの
 * 表示の位置は書かず、{@link Observation} が O から出す。
 *
 * <p>席は観測席から見た相対席（0 自分、1 下家、2 対面、3 上家）で書く。網の入力が相対席で並ぶので、読む側は写さずに使える。観測席だけは
 * 絶対席で書き、絶対席が要るとき（同点の順位）は {@link Observation#absoluteSeat} で戻す。牌は 34 種の番号、点は {@link #POINT_UNIT}
 * 点単位で書く。
 *
 * <p>欄は値の範囲（{@link Width}）を宣言し、範囲は規則の上限（{@link Limits}）から取る。範囲の外の値は書くときに例外にし、切り捨てない。
 * なしがありうる値は −1 をなしにする。版の番号は持たず、形を変えれば {@link #fingerprint()} が変わる。
 */
public final class ObservationSchema {

  /** 点を書く単位。 */
  public static final int POINT_UNIT = 100;

  /** 点（{@link #POINT_UNIT} 点単位）の符号つきの幅。±327 万点まで書ける。 */
  private static final int POINT_BITS = 16;

  /**
   * 本場の上限。規則に上限はないので O の表し方で決め、超える局は例外にする。局末の記録の次局の本場も、次局の O に入る値なので同じ上限を
   * 使う。
   */
  public static final int MAX_HONBA = 255;

  /** 符号にそのまま書ける符の最大。 */
  private static final int MAX_FU = ScoreMath.decodeFuCode(Limits.MAX_FU_CODE);

  /** 色ごとの赤のマスク（AkaTileMask：bit 0 萬、1 筒、2 索）の最大。 */
  private static final int AKA_MASK = (1 << Limits.AKA_TILES) - 1;

  /** 節（宣言の順）。 */
  static final Section[] SECTIONS = Section.values();

  static final BitLayout<HeaderBits> HEADER = BitLayout.of(HeaderBits.class, Long.SIZE);

  static final BitLayout<SeatBits> SEAT = BitLayout.of(SeatBits.class, Integer.SIZE);

  static final BitLayout<HandBits> HAND = BitLayout.of(HandBits.class, Long.SIZE);

  /** 出来事の共通の頭の配置。 */
  static final BitLayout<EventBits> EVENT = BitLayout.of(EventBits.class, Short.SIZE);

  static final BitLayout<DiscardBits> DISCARD = BitLayout.following(EVENT, DiscardBits.class);

  /** 副露・加槓の出来事の配置。 */
  static final BitLayout<MeldBits> MELD = BitLayout.following(EVENT, MeldBits.class);

  static final BitLayout<LegalBits> LEGAL = BitLayout.of(LegalBits.class, Short.SIZE);

  /** 鳴いた後の打牌の配置。 */
  static final BitLayout<CallDiscardBits> CALL_DISCARD =
      BitLayout.of(CallDiscardBits.class, Short.SIZE);

  /** 即時の和了の配置。 */
  static final BitLayout<WinBits> WIN = BitLayout.of(WinBits.class, Long.SIZE);

  static {
    if (DISCARD.units() != EVENT.units() || MELD.units() != EVENT.units()) {
      throw new IllegalStateException("event records of each kind must have the same length");
    }
  }

  private ObservationSchema() {}

  /**
   * 記録の指紋を返す。行の目録に書き、読むときに照らす。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /**
   * O の {@code long} の数を返す。記録の数は頭から読む。O を写すときは、先頭からこの数の {@code long} を写す。
   *
   * @param words O を置いた配列
   * @param base O の先頭
   * @return {@code long} の数
   */
  public static int longs(long[] words, int base) {
    int longs = 0;
    for (Section section : SECTIONS) {
      longs += section.longs(words, base);
    }
    return longs;
  }

  /**
   * 節の先頭（{@code base} からの相対の添字）を {@code start} に入れ、O の {@code long} の数を返す。記録の数は頭から読む。
   *
   * @param words O を置いた配列
   * @param base O の先頭
   * @param start 節の先頭を入れる配列（{@link Section} の ordinal で引く）
   * @return O の {@code long} の数
   */
  static int place(long[] words, int base, int[] start) {
    int at = 0;
    for (Section section : SECTIONS) {
      start[section.ordinal()] = at;
      at += section.longs(words, base);
    }
    return at;
  }

  /**
   * 合法手がチー・ポンの後の打牌の記録（{@link Section#CALL_DISCARD}）を持つなら true を返す。
   *
   * @param type 手の種類
   * @return チー・ポンなら true
   */
  static boolean hasCallDiscards(Action.Type type) {
    return type == Action.Type.CHI || type == Action.Type.PON;
  }

  /**
   * 合法手が即時の和了の記録（{@link Section#WIN}）を持つなら true を返す。
   *
   * @param type 手の種類
   * @return ロン・ツモなら true
   */
  static boolean hasWin(Action.Type type) {
    return type == Action.Type.RON_AGARI || type == Action.Type.TSUMO_AGARI;
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

  /**
   * 符を符号（{@link ScoreMath#encodeFuCode}）にする。110 符を超える手（槓が重なった手で、3 翻以上でしか起きない）は 110 符にまとめる。
   * 3 翻 70 符で満貫なので支払いは変わらない（Major の DecisionWinPointFactsEncoder と同じ）。
   *
   * @param han 翻（裏なし）
   * @param fu 符（役満は 0）
   * @return 書く値
   * @throws IllegalArgumentException 符号にない符のとき（3 翻未満で 110 符を超えるときを含む）
   */
  static int encodeFu(int han, int fu) {
    return han >= 3 && fu > MAX_FU ? Limits.MAX_FU_CODE : ScoreMath.encodeFuCode(fu);
  }

  /** O の節。宣言の順に {@code long} の境で続けて置く。 */
  public enum Section {
    /** 局面の頭（1 個）。可変の節の記録の数と遷移の数を持つ。 */
    HEADER,
    /** 席（観測席から見た相対席の順に 4 個）。 */
    SEAT,
    /** 自分の手（1 個）。 */
    HAND,
    /** 公開の出来事（出来事の番号の順に、頭の数だけ）。 */
    EVENT,
    /** 合法手（合法手の順に、頭の数だけ）。 */
    LEGAL,
    /** チー・ポンの後の打牌（その合法手の順にまとめて、頭の数だけ）。 */
    CALL_DISCARD,
    /** 即時の和了（ロン・ツモの合法手の順に、頭の数だけ）。 */
    WIN;

    /** この節の {@code long} の数（記録の数は頭から読む）。 */
    int longs(long[] words, int base) {
      return switch (this) {
        case HEADER -> ObservationSchema.HEADER.longs(1);
        case SEAT -> ObservationSchema.SEAT.longs(GameState.NUM_PLAYERS);
        case HAND -> ObservationSchema.HAND.longs(1);
        case EVENT -> ObservationSchema.EVENT.longs(header(words, base, HeaderBits.EVENTS));
        case LEGAL -> ObservationSchema.LEGAL.longs(header(words, base, HeaderBits.LEGAL_ACTIONS));
        case CALL_DISCARD ->
            ObservationSchema.CALL_DISCARD.longs(header(words, base, HeaderBits.CALL_DISCARDS));
        case WIN -> ObservationSchema.WIN.longs(header(words, base, HeaderBits.WINS));
      };
    }

    private static int header(long[] words, int base, HeaderBits count) {
      return ObservationSchema.HEADER.get(words, base, 0, count);
    }
  }

  /** 公開の出来事の記録の種類。 */
  public enum EventKind {
    /** 打牌（立直の宣言牌を含む）。 */
    DISCARD,
    /** 副露（チー・ポン・大明槓・暗槓）と加槓。面子の種類は面子が持つ。 */
    MELD
  }

  /** 判断の直前の出来事（{@code PublicObservation.turnEvent()} の種類）。入力の局の欄もこの enum を使う。 */
  public enum LastEvent {
    /** ツモ（山か嶺上）。 */
    DRAW,
    DISCARD,
    /** 暗槓・加槓の宣言（槍槓の応答の元）。 */
    KAN_ATTEMPT
  }

  /**
   * 欄の値の範囲と繰り返し。幅は {@link BitLayout} が範囲から出す。
   *
   * @param min 最小の値（なしがありうる値は −1）
   * @param max 最大の値
   * @param count 同じ範囲の値を並べる数（牌ごと・席ごと）
   * @param values 値の enum（enum の ordinal を書く欄だけ。ほかは null）。指紋はここから値の enum を集めるので、一覧を手で書かない
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
      return new Width(0, last(type), 1, type);
    }

    /** なしがありうる enum の ordinal（なしは −1）。 */
    static Width optional(Class<? extends Enum<?>> type) {
      return new Width(-1, last(type), 1, type);
    }

    /** {@code bits} ビットの 2 の補数の範囲の符号つきの数。 */
    static Width signed(int bits) {
      return range(-(1 << (bits - 1)), (1 << (bits - 1)) - 1);
    }

    /** 同じ範囲の値を count 個並べる。 */
    Width times(int count) {
      return new Width(min, max, count, values);
    }

    private static int last(Class<? extends Enum<?>> type) {
      return type.getEnumConstants().length - 1;
    }
  }

  /** O の欄。範囲は {@link Width} が持つ。 */
  sealed interface Field extends BitField
      permits HeaderBits,
          SeatBits,
          HandBits,
          EventBits,
          DiscardBits,
          MeldBits,
          LegalBits,
          CallDiscardBits,
          WinBits {

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
    DECISION_KIND(Width.ordinal(EngineDecisionKind.class)),
    /** 観測席（絶対席。O の中でこの欄だけが絶対席）。 */
    OBSERVER(Width.upTo(GameState.NUM_PLAYERS - 1)),
    /** 親の相対席。 */
    DEALER(Width.upTo(GameState.NUM_PLAYERS - 1)),
    /** 局（東 1 局 0 … 西 4 局 11）。場風は局から出す。 */
    KYOKU(Width.upTo(HanchanProgression.WEST_END_KYOKU)),
    HONBA(Width.upTo(MAX_HONBA)),
    /** 供託の本数。 */
    KYOTAKU(Width.upTo(Limits.MAX_KYOTAKU)),
    /** 王牌を除く山の残り。 */
    WALL(Width.upTo(RoundLimits.MAX_WALL)),
    /** 自分の見逃しの振聴（同巡の振聴と立直後の振聴を兼ねる。捨て牌の振聴は含まない）。エンジンの判定で、出来事からは出せない。 */
    FURITEN_TEMPORARY(Width.flag()),
    /** 直前の出来事（なしは −1）。 */
    LAST_EVENT(Width.optional(LastEvent.class)),
    /** 直前の出来事の相対席（なしは −1）。 */
    LAST_EVENT_SEAT(Width.optional(GameState.NUM_PLAYERS - 1)),
    /** 直前の出来事の牌（なしと他家のツモは −1）。 */
    LAST_EVENT_TILE(Width.optional(Tile.NUM_TILE_TYPES - 1)),
    /** 直前の出来事の牌が赤。 */
    LAST_EVENT_AKA(Width.flag()),
    /** 直前のツモが嶺上。 */
    LAST_EVENT_RINSHAN(Width.flag()),
    /** 直前の槓の宣言の種類（直前の出来事が槓の宣言でなければ −1）。 */
    LAST_KAN_KIND(Width.optional(TurnEvent.KanKind.class)),
    /** 槓ドラの表示が後に延びている（直前の出来事の doraRevealPending）。 */
    DEFERRED_DORA(Width.flag()),
    /** 自分の手（副露を除き、ツモ牌を含む）の色ごとの赤。 */
    HAND_AKA(Width.upTo(AKA_MASK)),
    /** ドラ表示の数（初めの表示は局の始めに開くので 1 以上）。 */
    INDICATORS(Width.range(1, Limits.MAX_INDICATORS)),
    /** 表示の順のドラ表示の牌（数より後は 0）。 */
    INDICATOR(Width.upTo(Tile.NUM_TILE_TYPES - 1).times(Limits.MAX_INDICATORS)),
    /** ドラ表示に出た色ごとの赤（エンジンは表示の赤を色ごとにしか持たない）。 */
    INDICATOR_AKA(Width.upTo(AKA_MASK)),
    /** 出来事の数（EVENT の記録の数）。 */
    EVENTS(Width.upTo(RoundLimits.MAX_PUBLIC_EVENTS)),
    /** 合法手の数（LEGAL の記録の数）。 */
    LEGAL_ACTIONS(Width.range(1, Limits.MAX_LEGAL_ACTIONS)),
    /** 遷移の数（{@link Observation#countTransitions}。O に節はなく、使う側が前置和に使う）。 */
    TRANSITIONS(Width.upTo(Limits.MAX_TRANSITIONS)),
    /** チー・ポンの後の打牌の数（CALL_DISCARD の記録の数）。 */
    CALL_DISCARDS(Width.upTo(Limits.MAX_TRANSITIONS)),
    /** 即時の和了の数（WIN の記録の数）。 */
    WINS(Width.upTo(Limits.MAX_LEGAL_ACTIONS));

    private final Width width;

    HeaderBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 席（観測席から見た相対席の順に 4 個、32 bit の単位）。第一打の前は出来事から出す。 */
  enum SeatBits implements Field {
    /** 点（{@link #POINT_UNIT} 点単位）。 */
    SCORE(Width.signed(POINT_BITS)),
    RIICHI(Width.ordinal(RiichiState.class)),
    /** 一発がある。エンジンの判定で、出来事からは出せない。 */
    IPPATSU(Width.flag());

    private final Width width;

    SeatBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 自分の手（1 個、64 bit の単位）。 */
  enum HandBits implements Field {
    /** 牌ごとの枚数（副露を除き、ツモ牌を含む。繰り返しの番号は牌）。 */
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

  /** 出来事の共通の頭（出来事の番号の順、16 bit の単位）。種類ごとの欄が続く。 */
  enum EventBits implements Field {
    KIND(Width.ordinal(EventKind.class)),
    /** 作者の相対席。 */
    AUTHOR(Width.upTo(GameState.NUM_PLAYERS - 1));

    private final Width width;

    EventBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 打牌の出来事（共通の頭に続く）。 */
  enum DiscardBits implements Field {
    TILE(Width.upTo(Tile.NUM_TILE_TYPES - 1)),
    AKA(Width.flag()),
    TSUMOGIRI(Width.flag()),
    /** 立直の宣言牌。 */
    RIICHI(Width.flag()),
    /** この打牌を鳴いた相対席（鳴かれていなければ −1）。鳴いた副露は次の番号の出来事。 */
    CALLED_BY(Width.optional(GameState.NUM_PLAYERS - 1));

    private final Width width;

    DiscardBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 副露・加槓の出来事（共通の頭に続く）。 */
  enum MeldBits implements Field {
    /** その出来事の時点の面子の正準 ID（後に加槓したポンも、ポンの出来事はポンの ID）。 */
    MELD(Width.upTo(Limits.MELD_IDS - 1));

    private final Width width;

    MeldBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 合法手（合法手の順、16 bit の単位）。 */
  enum LegalBits implements Field {
    /** 手（Action#toIndex）。 */
    ACTION(Width.upTo(Action.ACTION_SPACE_SIZE - 1)),
    /** チー・ポンの後の打牌の数（ほかの手は 0）。 */
    CALL_DISCARDS(Width.upTo(Limits.MAX_CALL_DISCARDS));

    private final Width width;

    LegalBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** チー・ポンの後の打牌（16 bit の単位）。 */
  enum CallDiscardBits implements Field {
    /** 打牌の手（Action#toIndex）。 */
    ACTION(Width.upTo(Action.ACTION_SPACE_SIZE - 1));

    private final Width width;

    CallDiscardBits(Width width) {
      this.width = width;
    }

    @Override
    public Width width() {
      return width;
    }
  }

  /** 即時の和了（64 bit の単位）。エンジンの WinSettlementProjection の写しで、裏ドラを含まない。 */
  enum WinBits implements Field {
    /** 翻（役満は 0）。 */
    HAN(Width.upTo(Limits.MAX_WIN_HAN)),
    /** 符の符号（{@link ObservationSchema#encodeFu}。役満は 0）。 */
    FU_CODE(Width.upTo(Limits.MAX_FU_CODE)),
    /** 役満の倍率。 */
    YAKUMAN(Width.upTo(Limits.MAX_YAKUMAN)),
    /** 包が成り立つ。 */
    PAO(Width.flag()),
    /** 立直していて、裏ドラの資格がある。 */
    URA_ELIGIBLE(Width.flag()),
    /** 和了で開きうる裏ドラ表示の数。 */
    URA_INDICATORS(Width.upTo(Limits.MAX_INDICATORS)),
    /** 順位を出した精算の仮定。 */
    SETTLEMENT(Width.ordinal(SettlementAssumption.class)),
    /** 和了の後の順位（0 始まり。繰り返しの番号は相対席）。 */
    RANK(Width.upTo(GameState.NUM_PLAYERS - 1).times(GameState.NUM_PLAYERS)),
    /** 点の移動（{@link #POINT_UNIT} 点単位、ロンは本場なし。繰り返しの番号は相対席）。 */
    PAYMENT(Width.signed(POINT_BITS).times(GameState.NUM_PLAYERS));

    private final Width width;

    WinBits(Width width) {
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

    /** 節の順、全ての配置（欄の範囲と位置）、欄が使う値の enum、enum でない符号（面子の ID・手の番号）、牌の数と点の単位。 */
    private static SchemaDigest describe() {
      SchemaDigest digest = SchemaDigest.named("epsilon-prism-v2-observation");
      digest.values(Section.class);
      digest.layout(HEADER).layout(SEAT).layout(HAND).layout(EVENT).layout(DISCARD).layout(MELD);
      digest.layout(LEGAL).layout(CALL_DISCARD).layout(WIN);
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
          .values(Action.Type.class)
          .values(Action.TileSelection.class)
          .with("actions", Action.ACTION_SPACE_SIZE)
          .with("tiles", Tile.NUM_TILE_TYPES)
          .with("pointUnit", POINT_UNIT);
    }
  }
}
