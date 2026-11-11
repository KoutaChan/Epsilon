package com.epsilon.agent.observe;

import com.epsilon.agent.observe.codec.AkaMask;
import com.epsilon.agent.observe.codec.BitField;
import com.epsilon.agent.observe.codec.BitLayout;
import com.epsilon.agent.observe.codec.SchemaDigest;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.ActionSection;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.engine.WinSettlementProjection.SettlementAssumption;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.VariantScoped;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.Tiles;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;

/**
 * 記録 O の形。O は判断の時点の値を加工せずに持つ。形はルールごとに 1 つあり（{@link #of}）、席の数・行動の数・語彙の大きさで
 * 決まるフィールドの幅は、ルール（{@code Variant}）から取る。
 *
 * <p>O は {@code long[]} の上に、セクション（{@link Section}）を宣言の順に続けて置く。セクションごとにフィールドの enum を宣言し、
 * フィールドの配置は {@link BitLayout} がその enum から一度だけ決める。長さの変わるセクションは記録の数をヘッダーに書き、
 * セクションの位置はその数から {@link #place} で求める。O はどこでも {@code long[]} のまま持ち、コピーするときは {@link #longs} が
 * 返す数だけ {@code long} をコピーする。
 *
 * <p>O に入れるのは、判断する席から見た局面（SeatView）が返す値そのもの、合法手、ルールの関数の結果（チー・ポンの後の打牌、選べる
 * 和了の精算）だけ。ほかのフィールドから出せる値（場風・第一巡かどうか・副露の有無・第一打の前かどうか・ツモ牌・槓ドラの表示の
 * 位置・同値な打牌の代表・遷移の数の内訳）は書かず、{@link Observation} が O から求める。
 *
 * <p>席は観測席から見た相対席（0 が自分、1 が下家、…、S − 1 が上家）で書く。観測席だけは絶対席で書く。牌は種類で、点は
 * {@link #POINT_UNIT} 点単位で書く。enum の値は、ルールごとの語彙（Vocabulary）の中の位置で書く。フィールドは値の範囲を宣言し、
 * 範囲の外の値は切り捨てずに、書くときに例外にする。無いことがありうる値は −1 で「無し」を表す。バージョン番号は持たず、形を
 * 変えればフィンガープリント（{@link #fingerprint}）が変わる。四人打ちのフィンガープリントは今あるコホートとスナップショットに
 * 書いてあるので、変えるとそれらを読めなくなる。
 *
 * <p>北抜きは、北抜きのあるルールの語彙にだけある値（イベントの種類の北抜きと、直前のイベントの北抜き）で書く。語彙はルール
 * ごとに作るので、四人打ちの形は変わらない。席ごとの抜いた北の数は書かず、イベントから求める。
 */
public final class ObservationSchema {

  /** 点を書く単位。 */
  public static final int POINT_UNIT = 100;

  /**
   * 本場の上限。ルールには上限が無いので、O の表し方で決めた。超える局は書くときに例外にする。局末の記録に書く次局の本場も、次局の
   * O に入る値なので同じ上限を使う。
   */
  public static final int MAX_HONBA = 255;

  /** 1 つの判断での遷移の数の上限。ルールの上限ではなく O の表し方で決めた値で、超える判断は書くときに例外にする。 */
  public static final int MAX_TRANSITIONS = 192;

  /** 1 回のチー・ポンの後に選べる打牌の数の上限。手牌の 13 枚から鳴きに使う 2 枚を除いた、残りの 11 枚に 1 つずつ。 */
  public static final int MAX_CALL_DISCARDS = Variant.HAND - (Meld.Type.PON.tiles() - 1);

  /** 合法手の数の上限（エンジンが 1 つの席の判断に持てる数の上限）。 */
  static final int MAX_LEGAL = Decisions.MAX_LEGAL;

  /** 点（{@link #POINT_UNIT} 点単位）を書く、符号つきのフィールドのビット数。±327 万点まで書ける。 */
  private static final int POINT_BITS = 16;

  /** コードで表せる符の上限。 */
  private static final int MAX_FU = 110;

  /** 30 符以上の符をコードにするときの単位（30〜110 符のコードは 符 / 10）。 */
  private static final int FU_UNIT = 10;

  private static final int PINFU_TSUMO_FU = 20;
  private static final int PINFU_TSUMO_CODE = 1;
  private static final int CHIITOITSU_FU = 25;
  private static final int CHIITOITSU_CODE = 2;
  private static final int MIN_DIVIDED_FU = 30;

  /** 符を 110 符にまとめてよい翻の下限（3 翻 70 符で満貫）。 */
  private static final int FOLDED_FU_HAN = 3;

  /** 符のコード（{@link #encodeFu}）の上限。110 符のコード。 */
  public static final int MAX_FU_CODE = MAX_FU / FU_UNIT;

  /** 役満の倍率の上限（全ての役満が重なったとき。天鳳は 1 役 1 倍）。 */
  public static final int MAX_YAKUMAN = yakumanCount();

  /** 役満でない役の翻の和（役ごとに、門前と副露のうち大きい方を足す）。 */
  private static final int MAX_YAKU_HAN = yakuHan();

  private static final Section[] SECTIONS = Section.values();
  private static final ObservationSchema[] BY_VARIANT = schemas();

  private final Variant variant;
  final BitLayout<HeaderBits> header;
  final BitLayout<SeatBits> seat;
  final BitLayout<HandBits> hand;
  final BitLayout<EventBits> event;
  final BitLayout<DiscardBits> discard;
  final BitLayout<MeldBits> meld;
  final BitLayout<LegalBits> legal;
  final BitLayout<CallDiscardBits> callDiscard;
  final BitLayout<WinBits> win;
  final Vocabulary<EngineDecisionKind> decisionKinds;
  final Vocabulary<LastEvent> lastEvents;
  final Vocabulary<TurnEvent.KanKind> kanKinds;
  final Vocabulary<RiichiState> riichiStates;
  final Vocabulary<EventKind> eventKinds;
  final Vocabulary<SettlementAssumption> settlements;
  private final String fingerprint;

  private ObservationSchema(Variant variant) {
    this.variant = variant;
    header = BitLayout.of(HeaderBits.class, Long.SIZE, variant);
    seat = BitLayout.of(SeatBits.class, Integer.SIZE, variant);
    hand = BitLayout.of(HandBits.class, Long.SIZE, variant);
    event = BitLayout.of(EventBits.class, eventUnit(variant), variant);
    discard = BitLayout.following(event, DiscardBits.class);
    meld = BitLayout.following(event, MeldBits.class);
    legal = BitLayout.of(LegalBits.class, Short.SIZE, variant);
    callDiscard = BitLayout.of(CallDiscardBits.class, Short.SIZE, variant);
    win = BitLayout.of(WinBits.class, Long.SIZE, variant);
    if (discard.units() != event.units() || meld.units() != event.units()) {
      throw new IllegalStateException("Event records of each kind must have the same length");
    }
    decisionKinds = vocabulary(HeaderBits.DECISION_KIND, EngineDecisionKind.class);
    lastEvents = vocabulary(HeaderBits.LAST_EVENT, LastEvent.class);
    kanKinds = vocabulary(HeaderBits.LAST_KAN_KIND, TurnEvent.KanKind.class);
    riichiStates = vocabulary(SeatBits.RIICHI, RiichiState.class);
    eventKinds = vocabulary(EventBits.KIND, EventKind.class);
    settlements = vocabulary(WinBits.SETTLEMENT, SettlementAssumption.class);
    fingerprint = describe().hex();
  }

  public static ObservationSchema of(Variant variant) {
    return BY_VARIANT[variant.ordinal()];
  }

  /**
   * フィンガープリントから、O を読むルールを決める。コホートやスナップショットに書いたフィンガープリントを渡す。
   *
   * @param fingerprint {@link #fingerprint} の値
   * @return フィンガープリントが合うルール。どのルールの O とも合わなければ空
   */
  public static Optional<Variant> variantOf(String fingerprint) {
    Objects.requireNonNull(fingerprint, "fingerprint");
    for (ObservationSchema schema : BY_VARIANT) {
      if (schema.fingerprint.equals(fingerprint)) {
        return Optional.of(schema.variant);
      }
    }
    return Optional.empty();
  }

  public Variant variant() {
    return variant;
  }

  /**
   * フィンガープリント。形の記述の SHA-256 を取り、先頭 8 バイトを 16 進で書いたもの。コホートのマニフェストに書き、読むときに
   * 照らし合わせる。
   */
  public String fingerprint() {
    return fingerprint;
  }

  /** 配列の指定した位置から始まる O の長さ（{@code long} の数）を返す。記録の数はヘッダーから読む。 */
  public int longs(long[] words, int base) {
    int longs = 0;
    for (Section section : SECTIONS) {
      longs += longs(section, words, base);
    }
    return longs;
  }

  /**
   * セクションごとに先頭の位置を求める。記録の数はヘッダーから読む。
   *
   * @param base O の先頭の位置
   * @param start 書く先。セクションの ordinal の位置に、O の先頭からの位置を書く
   * @return O の長さ（{@code long} の数）
   */
  int place(long[] words, int base, int[] start) {
    int at = 0;
    for (Section section : SECTIONS) {
      start[section.ordinal()] = at;
      at += longs(section, words, base);
    }
    return at;
  }

  /** セクションの数（{@link #place} に渡す配列の長さ）。 */
  static int sections() {
    return SECTIONS.length;
  }

  /** O の長さ（{@code long} の数）の上限。長さの変わるセクションを、ヘッダーのフィールドが表せる上限の数まで埋めたときの長さ。 */
  int maxLongs() {
    return header.longs(1)
        + seat.longs(variant.seats())
        + hand.longs(1)
        + event.longs(header.max(HeaderBits.EVENTS))
        + legal.longs(header.max(HeaderBits.LEGAL_ACTIONS))
        + callDiscard.longs(header.max(HeaderBits.CALL_DISCARDS))
        + win.longs(header.max(HeaderBits.WINS));
  }

  /**
   * フィンガープリントの元になる記述を作る。セクションの順、全てのフィールドの配置、フィールドが使う語彙を入れる。さらに、面子と
   * 行動の番号を組み立てる enum の語彙と番号の総数、牌の種類の数、点の単位を入れる。
   */
  SchemaDigest describe() {
    SchemaDigest digest = SchemaDigest.named("epsilon-prism-v2-observation");
    digest.values(Vocabulary.of(Section.class, variant));
    digest.layout(header).layout(seat).layout(hand).layout(event).layout(discard).layout(meld);
    digest.layout(legal).layout(callDiscard).layout(win);
    Map<Class<?>, Vocabulary<?>> values = new LinkedHashMap<>();
    for (Class<?> type : SchemaDigest.sorted(Field.class.getPermittedSubclasses())) {
      for (Object constant : type.getEnumConstants()) {
        Vocabulary<?> vocabulary = ((Field) constant).width(variant).values();
        if (vocabulary != null) {
          values.putIfAbsent(vocabulary.type(), vocabulary);
        }
      }
    }
    values.values().forEach(digest::values);
    return digest
        .values(Vocabulary.of(Meld.Type.class, variant))
        .values(Vocabulary.of(Meld.RelativeSource.class, variant))
        .values(Vocabulary.of(Meld.AkaSource.class, variant))
        .with("meldIds", Meld.IDS)
        .values(Vocabulary.of(Action.Type.class, variant))
        .values(Vocabulary.of(Action.TileSelection.class, variant))
        .with("actions", ActionSection.space(variant))
        .with("tiles", Tiles.KINDS)
        .with("pointUnit", POINT_UNIT);
  }

  /** その種類の合法手がチー・ポンの後の打牌の記録（{@link Section#CALL_DISCARD}）を持つなら true。 */
  static boolean hasCallDiscards(Action.Type type) {
    return type == Action.Type.CHI || type == Action.Type.PON;
  }

  /** その種類の合法手が和了の記録（{@link Section#WIN}）を持つなら true。 */
  static boolean hasWin(Action.Type type) {
    return type == Action.Type.RON_AGARI || type == Action.Type.TSUMO_AGARI;
  }

  /** 点を {@link #POINT_UNIT} 点単位にする。割り切れなければ IllegalArgumentException を投げる。 */
  static int encodePoints(int points) {
    if (points % POINT_UNIT != 0) {
      throw new IllegalArgumentException("Points are not a multiple of " + POINT_UNIT + ": " + points);
    }
    return points / POINT_UNIT;
  }

  static int decodePoints(int stored) {
    return stored * POINT_UNIT;
  }

  /**
   * 符を、フィールドに書くコードにする。役満の 0 符は 0、20 符は 1、25 符は 2、30〜110 符は 符 / 10 にする。110 符を超える手（槓が
   * 重なった手で、3 翻以上でしか起きない）は 110 符にまとめる。3 翻 70 符で満貫なので、払いは変わらない。コードの無い符（3 翻未満で
   * 110 符を超える手を含む）は IllegalArgumentException を投げる。
   */
  static int encodeFu(int han, int fu) {
    if (han >= FOLDED_FU_HAN && fu > MAX_FU) {
      return MAX_FU_CODE;
    }
    if (fu == 0) {
      return 0;
    }
    if (fu == PINFU_TSUMO_FU) {
      return PINFU_TSUMO_CODE;
    }
    if (fu == CHIITOITSU_FU) {
      return CHIITOITSU_CODE;
    }
    if (fu < MIN_DIVIDED_FU || fu > MAX_FU || fu % FU_UNIT != 0) {
      throw new IllegalArgumentException("No fu code for " + fu + " fu at " + han + " han");
    }
    return fu / FU_UNIT;
  }

  private int longs(Section section, long[] words, int base) {
    return switch (section) {
      case HEADER -> header.longs(1);
      case SEAT -> seat.longs(variant.seats());
      case HAND -> hand.longs(1);
      case EVENT -> event.longs(header.get(words, base, 0, HeaderBits.EVENTS));
      case LEGAL -> legal.longs(header.get(words, base, 0, HeaderBits.LEGAL_ACTIONS));
      case CALL_DISCARD -> callDiscard.longs(header.get(words, base, 0, HeaderBits.CALL_DISCARDS));
      case WIN -> win.longs(header.get(words, base, 0, HeaderBits.WINS));
    };
  }

  /** enum の位置を書くフィールドの語彙を返す。フィールドがその enum の語彙を持たなければ IllegalStateException を投げる。 */
  private <E extends Enum<E>> Vocabulary<E> vocabulary(Field field, Class<E> type) {
    Vocabulary<?> values = field.width(variant).values();
    if (values == null || values.type() != type) {
      throw new IllegalStateException(field + " does not hold the vocabulary of " + type.getSimpleName());
    }
    @SuppressWarnings("unchecked")
    Vocabulary<E> typed = (Vocabulary<E>) values;
    return typed;
  }

  /**
   * 和了の翻（裏ドラを数えない）の上限。役満でない役の翻の和に、ドラの上限を足す。ドラは表示牌 1 枚につき 4 翻（同じ種類の 4 枚。
   * 抜いた北も含む）、赤はルールにある赤の 5 の枚数、抜いた北は 1 枚につき 1 翻と数える。
   */
  private static int maxWinHan(Variant variant) {
    int reds = Long.bitCount(variant.tiles().redKinds());
    int nuki = variant.has(RuleFlag.NUKI) ? Tiles.COPIES : 0;
    return MAX_YAKU_HAN + WinContext.MAX_INDICATORS * Tiles.COPIES + reds + nuki;
  }

  private static int yakumanCount() {
    int count = 0;
    for (ScoringYaku yaku : ScoringYaku.values()) {
      if (yaku.yakuman()) {
        count++;
      }
    }
    return count;
  }

  private static int yakuHan() {
    int han = 0;
    for (ScoringYaku yaku : ScoringYaku.values()) {
      if (!yaku.yakuman()) {
        han += Math.max(yaku.han(true), yaku.han(false));
      }
    }
    return han;
  }

  /**
   * イベントの記録の単位のビット数。打牌と副露の記録がどちらも 1 単位に収まる、最も小さい単位にする。四人打ちは 16 bit、イベントの
   * 種類が 3 つになる北抜きのあるルールは 32 bit。
   */
  private static int eventUnit(Variant variant) {
    for (int unit = Short.SIZE; unit <= Long.SIZE; unit *= 2) {
      BitLayout<EventBits> head = BitLayout.of(EventBits.class, unit, variant);
      if (BitLayout.following(head, DiscardBits.class).units() == 1
          && BitLayout.following(head, MeldBits.class).units() == 1) {
        return unit;
      }
    }
    throw new IllegalStateException("Event records of " + variant + " do not fit in one long");
  }

  private static ObservationSchema[] schemas() {
    Variant[] variants = Variant.values();
    ObservationSchema[] schemas = new ObservationSchema[variants.length];
    for (Variant variant : variants) {
      schemas[variant.ordinal()] = new ObservationSchema(variant);
    }
    return schemas;
  }

  /** O のセクション。宣言の順に、{@code long} の切れ目にそろえて続けて置く。 */
  public enum Section {
    /** ヘッダー（1 個）。長さの変わるセクションの記録の数と、遷移の数を持つ。 */
    HEADER,
    /** 席（観測席から見た相対席の順に S 個）。 */
    SEAT,
    /** 自分の純手牌（1 個）。 */
    HAND,
    /** 公開のイベント（イベントの番号の順に、ヘッダーにある数だけ）。 */
    EVENT,
    /** 合法手（合法手の順に、ヘッダーにある数だけ）。 */
    LEGAL,
    /** チー・ポンの後の打牌（チー・ポンの合法手の順にまとめて、ヘッダーにある数だけ）。 */
    CALL_DISCARD,
    /** この判断で選べる和了（ロン・ツモの合法手の順に、ヘッダーにある数だけ）。 */
    WIN
  }

  /** 公開のイベントの記録の種類。 */
  public enum EventKind implements VariantScoped {
    /** 打牌（立直の宣言牌を含む）。 */
    DISCARD(null),
    /** 副露（チー・ポン・大明槓・暗槓）と加槓。面子の種別は面子が持つ。 */
    MELD(null),
    /** 北抜き。種類と席だけを持つ（抜くのはいつも北）。 */
    NUKI(RuleFlag.NUKI);

    private final RuleFlag requires;

    EventKind(RuleFlag requires) {
      this.requires = requires;
    }

    @Override
    public RuleFlag requires() {
      return requires;
    }
  }

  /** 判断の直前のイベント。ネットワークの入力でも、局のフィールドにこの enum を使う。 */
  public enum LastEvent implements VariantScoped {
    /** ツモ（山か嶺上）。 */
    DRAW(null),
    DISCARD(null),
    /** 加槓・暗槓の宣言。加槓なら、他家に槍槓の応答を求める。 */
    KAN_ATTEMPT(null),
    /** 北抜き。他家に、抜いた北への応答を求める。 */
    NUKI(RuleFlag.NUKI);

    private final RuleFlag requires;

    LastEvent(RuleFlag requires) {
      this.requires = requires;
    }

    @Override
    public RuleFlag requires() {
      return requires;
    }
  }

  /**
   * フィールドの値の範囲と、繰り返しの数。
   *
   * @param values enum の位置を書くフィールドの語彙（ほかのフィールドは null）。フィンガープリントはここから語彙を集めるので、
   *     語彙の一覧を手で書かない
   */
  record Width(int min, int max, int count, Vocabulary<?> values) {

    /** 2 値（0 は偽、1 は真）。 */
    static final Width FLAG = upTo(1);

    /** 0 から上限までの数・番号。 */
    static Width upTo(int max) {
      return range(0, max);
    }

    static Width range(int min, int max) {
      return new Width(min, max, 1, null);
    }

    /** 無いことがありうる、0 から上限までの番号（無いときは −1）。 */
    static Width optional(int max) {
      return range(-1, max);
    }

    /** 席の番号（0〜S − 1）。 */
    static Width seat(Variant variant) {
      return upTo(variant.seats() - 1);
    }

    /** 語彙の中の位置。 */
    static Width of(Vocabulary<?> values) {
      return new Width(0, values.size() - 1, 1, values);
    }

    /** 無いことがありうる、語彙の中の位置（無いときは −1）。 */
    static Width optional(Vocabulary<?> values) {
      return new Width(-1, values.size() - 1, 1, values);
    }

    /** 指定したビット数の 2 の補数で表せる、符号つきの数。 */
    static Width signed(int bits) {
      return range(-(1 << (bits - 1)), (1 << (bits - 1)) - 1);
    }

    /** 同じ範囲の値を、指定した数だけ並べる。 */
    Width times(int count) {
      return new Width(min, max, count, values);
    }
  }

  /** O のフィールド。ルールごとの範囲は {@link Width} が持つ。 */
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

    Width width(Variant variant);

    @Override
    default int min(Variant variant) {
      return width(variant).min();
    }

    @Override
    default int max(Variant variant) {
      return width(variant).max();
    }

    @Override
    default int count(Variant variant) {
      return width(variant).count();
    }
  }

  /** ヘッダー（1 個、64 bit の単位）。 */
  enum HeaderBits implements Field {
    DECISION_KIND(v -> Width.of(Vocabulary.of(EngineDecisionKind.class, v))),
    /** 観測席（絶対席。O の中で絶対席を書くのはこのフィールドだけ）。 */
    OBSERVER(Width::seat),
    /** 親の相対席。 */
    DEALER(Width::seat),
    /** 局の通し番号（東 1 局が 0 で、西場の最後の局まで）。場風は局の番号から求める。 */
    KYOKU(v -> Width.upTo(v.finalRound())),
    HONBA(v -> Width.upTo(MAX_HONBA)),
    /** 供託の本数。上限は、全ての席が配給原点を全て供託に出したときの本数。 */
    KYOTAKU(v -> Width.upTo(v.seats() * v.startPoints() / Variant.DEPOSIT)),
    /** 王牌を除いた山の残りの枚数。 */
    WALL(v -> Width.upTo(v.liveWall())),
    /**
     * 自分が見逃しで振聴になっている（同巡内と立直の後の両方を含み、河による振聴は含まない）。エンジンが判定する値で、イベントからは
     * 出せない。
     */
    FURITEN_TEMPORARY(v -> Width.FLAG),
    /** 直前のイベント（無ければ −1）。 */
    LAST_EVENT(v -> Width.optional(Vocabulary.of(LastEvent.class, v))),
    /** 直前のイベントの席（相対席。無ければ −1）。 */
    LAST_EVENT_SEAT(v -> Width.optional(v.seats() - 1)),
    /** 直前のイベントの牌（無いときと、他家のツモは −1）。 */
    LAST_EVENT_TILE(v -> Width.optional(Tiles.KINDS - 1)),
    LAST_EVENT_AKA(v -> Width.FLAG),
    /** 直前のツモが嶺上。 */
    LAST_EVENT_RINSHAN(v -> Width.FLAG),
    /** 直前の槓の宣言の種類（直前のイベントが槓の宣言でなければ −1）。 */
    LAST_KAN_KIND(v -> Width.optional(Vocabulary.of(TurnEvent.KanKind.class, v))),
    /** 大明槓・加槓の後めくりの槓ドラを、まだめくっていない。 */
    DEFERRED_DORA(v -> Width.FLAG),
    /** 自分の純手牌の赤を、色ごとのビットで表した値（{@link AkaMask}。bit 0 が萬、1 が筒、2 が索）。 */
    HAND_AKA(v -> Width.upTo(AkaMask.ALL)),
    /** ドラ表示牌の数（最初の表示牌は局の始めにめくるので 1 以上）。 */
    INDICATORS(v -> Width.range(1, WinContext.MAX_INDICATORS)),
    /** ドラ表示牌の種類をめくった順に並べたもの（表示牌の数より後ろは 0）。 */
    INDICATOR(v -> Width.upTo(Tiles.KINDS - 1).times(WinContext.MAX_INDICATORS)),
    /** ドラ表示牌に出た赤を、色ごとのビットで表した値（ビットの位置は {@link #HAND_AKA} と同じ）。どの表示牌が赤かは書かない。 */
    INDICATOR_AKA(v -> Width.upTo(AkaMask.ALL)),
    /** イベントの数（{@link Section#EVENT} の記録の数）。 */
    EVENTS(v -> Width.upTo(v.maxPublicEvents())),
    /** 合法手の数（{@link Section#LEGAL} の記録の数）。 */
    LEGAL_ACTIONS(v -> Width.range(1, MAX_LEGAL)),
    /** 遷移の数（{@link Observation#countTransitions}）。遷移のセクションは O に無い。使う側が累積和を取るのに使う。 */
    TRANSITIONS(v -> Width.upTo(MAX_TRANSITIONS)),
    /** チー・ポンの後の打牌の数（{@link Section#CALL_DISCARD} の記録の数）。 */
    CALL_DISCARDS(v -> Width.upTo(MAX_TRANSITIONS)),
    /** この判断で選べる和了の数（{@link Section#WIN} の記録の数）。 */
    WINS(v -> Width.upTo(MAX_LEGAL));

    private final Function<Variant, Width> width;

    HeaderBits(Function<Variant, Width> width) {
      this.width = width;
    }

    @Override
    public Width width(Variant variant) {
      return width.apply(variant);
    }
  }

  /** 席（観測席から見た相対席の順に S 個、32 bit の単位）。第一打の前かどうかは書かず、イベントから求める。 */
  enum SeatBits implements Field {
    /** 点（{@link ObservationSchema#POINT_UNIT} 点単位）。 */
    SCORE(v -> Width.signed(POINT_BITS)),
    RIICHI(v -> Width.of(Vocabulary.of(RiichiState.class, v))),
    /** 一発がある。エンジンが判定する値で、イベントからは出せない。 */
    IPPATSU(v -> Width.FLAG);

    private final Function<Variant, Width> width;

    SeatBits(Function<Variant, Width> width) {
      this.width = width;
    }

    @Override
    public Width width(Variant variant) {
      return width.apply(variant);
    }
  }

  /** 自分の純手牌（1 個、64 bit の単位）。 */
  enum HandBits implements Field {
    /** 種類ごとの枚数（副露を除き、ツモ牌を含む。繰り返しの番号が牌の種類）。 */
    COUNT(v -> Width.upTo(Tiles.COPIES).times(Tiles.KINDS));

    private final Function<Variant, Width> width;

    HandBits(Function<Variant, Width> width) {
      this.width = width;
    }

    @Override
    public Width width(Variant variant) {
      return width.apply(variant);
    }
  }

  /** イベントの記録のうち、種類に依らない先頭部分（イベントの番号の順、単位は {@link #eventUnit}）。この後に種類ごとのフィールドが続く。 */
  enum EventBits implements Field {
    KIND(v -> Width.of(Vocabulary.of(EventKind.class, v))),
    /** イベントを起こした席（相対席）。 */
    AUTHOR(Width::seat);

    private final Function<Variant, Width> width;

    EventBits(Function<Variant, Width> width) {
      this.width = width;
    }

    @Override
    public Width width(Variant variant) {
      return width.apply(variant);
    }
  }

  /** 打牌のイベントのフィールド（共通の先頭部分に続く）。 */
  enum DiscardBits implements Field {
    TILE(v -> Width.upTo(Tiles.KINDS - 1)),
    AKA(v -> Width.FLAG),
    TSUMOGIRI(v -> Width.FLAG),
    /** 立直の宣言牌。 */
    RIICHI(v -> Width.FLAG),
    /** この打牌を鳴いた席（相対席。鳴かれていなければ −1）。鳴いた副露は次の番号のイベントになる。 */
    CALLED_BY(v -> Width.optional(v.seats() - 1));

    private final Function<Variant, Width> width;

    DiscardBits(Function<Variant, Width> width) {
      this.width = width;
    }

    @Override
    public Width width(Variant variant) {
      return width.apply(variant);
    }
  }

  /** 副露・加槓のイベントのフィールド（共通の先頭部分に続く）。 */
  enum MeldBits implements Field {
    /** そのイベントの時点の面子を表す、正準の番号。後で加槓したポンも、ポンのイベントではポンの番号で書く。 */
    MELD(v -> Width.upTo(Meld.IDS - 1));

    private final Function<Variant, Width> width;

    MeldBits(Function<Variant, Width> width) {
      this.width = width;
    }

    @Override
    public Width width(Variant variant) {
      return width.apply(variant);
    }
  }

  /** 合法手（合法手の順、16 bit の単位）。 */
  enum LegalBits implements Field {
    /** 行動の番号。 */
    ACTION(v -> Width.upTo(ActionSection.space(v) - 1)),
    /** チー・ポンの後の打牌の数（ほかの手は 0）。 */
    CALL_DISCARDS(v -> Width.upTo(MAX_CALL_DISCARDS));

    private final Function<Variant, Width> width;

    LegalBits(Function<Variant, Width> width) {
      this.width = width;
    }

    @Override
    public Width width(Variant variant) {
      return width.apply(variant);
    }
  }

  /** チー・ポンの後の打牌（16 bit の単位）。 */
  enum CallDiscardBits implements Field {
    /** 打牌の行動の番号。 */
    ACTION(v -> Width.upTo(ActionSection.space(v) - 1));

    private final Function<Variant, Width> width;

    CallDiscardBits(Function<Variant, Width> width) {
      this.width = width;
    }

    @Override
    public Width width(Variant variant) {
      return width.apply(variant);
    }
  }

  /** この判断で選べる和了（64 bit の単位）。WinSettlementProjection の値をコピーしたもので、裏ドラを数えない。 */
  enum WinBits implements Field {
    /** 翻（役満は 0）。 */
    HAN(v -> Width.upTo(maxWinHan(v))),
    /** 符のコード（{@link ObservationSchema#encodeFu}。役満は 0）。 */
    FU_CODE(v -> Width.upTo(MAX_FU_CODE)),
    /** 役満の倍率。 */
    YAKUMAN(v -> Width.upTo(MAX_YAKUMAN)),
    /** 包が成り立つ。 */
    PAO(v -> Width.FLAG),
    /** 立直していて、裏ドラが乗りうる。 */
    URA_ELIGIBLE(v -> Width.FLAG),
    /** 和了でめくりうる裏ドラ表示牌の数（今めくってあるドラ表示牌の数。立直していない和了でも書く）。 */
    URA_INDICATORS(v -> Width.upTo(WinContext.MAX_INDICATORS)),
    /** 順位を付けるときに置いた、精算の仮定。 */
    SETTLEMENT(v -> Width.of(Vocabulary.of(SettlementAssumption.class, v))),
    /** 和了の後の順位（0 始まり。繰り返しの番号は相対席）。 */
    RANK(v -> Width.seat(v).times(v.seats())),
    /** 点の移動（{@link ObservationSchema#POINT_UNIT} 点単位。ロンは本場を含まない。繰り返しの番号は相対席）。 */
    PAYMENT(v -> Width.signed(POINT_BITS).times(v.seats()));

    private final Function<Variant, Width> width;

    WinBits(Function<Variant, Width> width) {
      this.width = width;
    }

    @Override
    public Width width(Variant variant) {
      return width.apply(variant);
    }
  }
}
