package com.epsilon.prism3p.input;

import static com.epsilon.prism3p.input.ThreePlayerFormat.oneOf;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.agent.observe.codec.SchemaDigest;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.EventTable.PublicEvent;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.FieldSpec.BoundedField;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.Ids;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Composition;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.PassGrade;
import com.epsilon.prism.input.SuitCanonicalizer.SuitKey;
import com.epsilon.prism.input.TileSafety.Safety;
import com.epsilon.prism.input.TileTokens;
import com.epsilon.prism.input.TokenLayout;
import com.epsilon.prism.input.WaitProjection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 三人打ちの PRISM のネットワークが読む入力のフィールドを宣言する。三人打ちの卓（3 席・牌 27 種類・北抜き・ツモ損・返し点 40,000 点）に合わせて
 * 一から決めたフィールドで、書き方の決まりは四人打ちの入力と同じ。
 *
 * <p>フィールドはトークンの種類（局・席・牌・イベント・遷移）ごとに enum で宣言し、定数ごとに書き方・語彙・繰り返しの数（{@link FieldSpec}）
 * を持たせる。列の位置、埋め込みの名前空間、転送の幅は、{@link TokenLayout} と {@link #FORMAT} が enum から一度だけ作る。値 0 はいつも
 * 「なし・0 個・偽」を表す。enum の値は語彙の中の位置 + 1 で、牌・席・0 始まりの番号は番号 + 1 で書く（{@link Ids}）。牌の番号は、色を
 * 正準化した後の牌のトークンの番号（{@link TileTokens}）。萬子は 1 と 9 だけで、色の正準化では筒子と索子だけを入れ替える。
 *
 * <p>{@link #fingerprint()} は、フィールドの名前・順・書き方・語彙・繰り返しの数、数値と厳密な値のフィールドを割る数、値の enum の名前と並び、
 * 手の種類ごとの扱い、容量とテンソルのアラインメント、効用の名前と順位ごとの 3 つの値、色の正準化のキーと入れ替える色から作る。バージョン番号は
 * 持たない。
 */
public final class InputSchema {

  /** イベントのトークンの容量。バッチの各行に取る数で、三人打ちのイベントのトークンの最大 91 個が入る。 */
  public static final int EVENT_CAPACITY = 96;

  /** 本場と供託のフィールドの上限。これ以上の値は全てこの値として書く。 */
  public static final int STICK_SATURATION = 16;

  /** この手で和了したときの翻のフィールドの上限（数え役満の 13 翻）。これ以上の翻はこの値として書く。 */
  public static final int HAN_SATURATION = 13;

  /** 半荘が終わったときの効用と、待ちの射影に使う順位ごとの効用。 */
  public static final RankUtility UTILITY = RankUtility.TENHOU;

  /** 点差と得点のフィールドを割る数（10,000 点）。 */
  private static final float POINT_SCALE = 10_000f;

  /** 生きている枚数のフィールドを割る数。 */
  private static final float LIVE_TILE_SCALE = 16f;

  /** 局のフィールドの配置。 */
  public static final TokenLayout<RoundField> ROUND = TokenLayout.of(RoundField.class);

  /** 席のフィールドの配置。 */
  public static final TokenLayout<SeatField> SEAT = TokenLayout.of(SeatField.class);

  /** 牌のフィールドの配置。 */
  public static final TokenLayout<TileField> TILE = TokenLayout.of(TileField.class);

  /** イベントのフィールドの配置。 */
  public static final TokenLayout<EventField> EVENT = TokenLayout.of(EventField.class);

  /** 遷移のフィールドの配置。 */
  public static final TokenLayout<TransitionField> TRANSITION = TokenLayout.of(TransitionField.class);

  /** 関係の属性の配置。 */
  public static final TokenLayout<RelationAttribute> RELATION = TokenLayout.of(RelationAttribute.class);

  /** 三人打ちの入力の形式。 */
  public static final InputFormat FORMAT = new Format();

  private InputSchema() {}

  /**
   * 入力のフィンガープリントを返す。model.json に書き、モデルを開くときに照合する。
   *
   * @return SHA-256 の先頭 8 バイトの 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /** 埋め込みで読むカテゴリのフィールド。トークンの種類ごとに 1 つの enum にする。 */
  public sealed interface CategoryField extends BoundedField
      permits RoundField, SeatField, TileField, EventField, TransitionField {}

  /** BF16 で送る数値のフィールド。値は {@code scale()} で割ってから BF16 に丸める。 */
  public sealed interface ScalarField extends InputFormat.Scalar
      permits RoundScalar, SeatScalar, EventScalar, TransitionScalar {

    @Override
    default Tensor tensor() {
      return switch (this) {
        case RoundScalar round -> Tensor.ROUND_SCALAR;
        case SeatScalar seat -> Tensor.SEAT_SCALAR;
        case EventScalar event -> Tensor.EVENT_SCALAR;
        case TransitionScalar transition -> Tensor.TRANSITION_SCALAR;
      };
    }
  }

  /** FP32 のままヘッドへ渡す厳密な値のフィールド（打牌の値とゲートのヘッドに直接渡す値）。値は {@code scale()} で割ってから送る。 */
  public sealed interface ExactField extends InputFormat.Exact permits DiscardExact, WinExact, RiichiExact {

    @Override
    default Tensor tensor() {
      return switch (this) {
        case DiscardExact discard -> Tensor.DISCARD_EXACT;
        case WinExact win -> Tensor.ACTION_EXACT;
        case RiichiExact riichi -> Tensor.ACTION_EXACT;
      };
    }
  }

  /** 場風と自風（三人打ちに北家はない）。 */
  public enum Wind {
    EAST,
    SOUTH,
    WEST;

    private static final Wind[] VALUES = values();

    /**
     * 風の牌から風を返す。
     *
     * @param windTile 風の牌の種類（東〜西）
     * @return 風
     */
    public static Wind ofTile(int windTile) {
      return VALUES[Objects.checkIndex(windTile - Tiles.TON, VALUES.length)];
    }
  }

  /** 遷移の後の振聴の種類。 */
  public enum Furiten {
    /** 待ちが自分の河にある。 */
    DISCARD,
    /** 同巡の見逃し（次の自分の打牌まで）。 */
    TEMPORARY,
    /** 立直後の見逃し（局の終わりまで）。 */
    RIICHI_PERSISTENT
  }

  /** 副露の色。筒子と索子は正準化した色の順で表す。萬子の副露は 1・9 の刻子と槓子だけ。 */
  public enum OpenSuit {
    /** 正準化した 1 つ目の色（筒子か索子）と字牌だけ。 */
    FIRST,
    /** 正準化した 2 つ目の色と字牌だけ。 */
    SECOND,
    /** 萬子と字牌だけ。 */
    MANZU,
    /** 字牌だけ。 */
    HONORS
  }

  /**
   * 待ち牌の分類。その遷移の立直の状態でロンしたときの翻で分ける（裏ドラなし、抜きドラを含む）。待ちでない牌は、キーのコードのこの桁を
   * 0 にする。
   */
  public enum WaitClass {
    /** ツモでしか和了できない。 */
    TSUMO_ONLY,
    /** ロンで 1〜2 翻。 */
    HAN_1_2,
    /** ロンで 3〜5 翻。 */
    HAN_3_5,
    /** ロンで 6 翻以上（役満を含む）。 */
    HAN_6_PLUS
  }

  /** 価値グラフでの手の扱い。ゲートになるのは立直・ポンと大明槓・暗槓と加槓・ツモ・ロン・九種九牌・北抜きの 7 つ。 */
  public enum ActionRole implements InputFormat.Role {
    /** 打牌（打牌の値のヘッドが値を出す）。 */
    DISCARD(false, Composition.DISCARD_VALUE, false),
    /** 立直（同じ牌を切るダマを基準にしたゲート）。 */
    RIICHI(true, Composition.RIICHI_GATE, false),
    /** ポン・大明槓。 */
    CALL(true, Composition.RESPONSE_GATE, false),
    /** 暗槓・加槓。 */
    KAN(true, Composition.TURN_GATE, false),
    TSUMO(true, Composition.TURN_GATE, true),
    /** ロン和了（槍槓と、抜いた北へのロンを含む）。 */
    RON(true, Composition.RESPONSE_GATE, true),
    KYUSHU(true, Composition.TURN_GATE, true),
    /** 北抜き（手番のゲート）。 */
    NUKI(true, Composition.TURN_GATE, false),
    /** 見送り（他家の応答の基準。値は 0）。 */
    PASS(false, Composition.ZERO, false);

    private final boolean gate;
    private final Composition composition;
    private final boolean endsHand;

    ActionRole(boolean gate, Composition composition, boolean endsHand) {
      this.gate = gate;
      this.composition = composition;
      this.endsHand = endsHand;
    }

    @Override
    public boolean gate() {
      return gate;
    }

    @Override
    public Composition composition() {
      return composition;
    }

    @Override
    public boolean endsHand() {
      return endsHand;
    }

    /**
     * 手の種類から扱いを返す。default を書かないので、手の種類を足すとコンパイルエラーになる。
     *
     * @param type 手の種類
     * @return 扱い
     * @throws IllegalArgumentException 三人打ちの入力に無い種類（チー）のとき
     */
    public static ActionRole of(Action.Type type) {
      return switch (type) {
        case DAHAI -> ActionRole.DISCARD;
        case RIICHI_DAHAI -> ActionRole.RIICHI;
        case PON, DAIMINKAN -> ActionRole.CALL;
        case ANKAN, KAKAN -> ActionRole.KAN;
        case TSUMO_AGARI -> ActionRole.TSUMO;
        case RON_AGARI -> ActionRole.RON;
        case KYUSHU_KYUHAI -> ActionRole.KYUSHU;
        case NUKI -> ActionRole.NUKI;
        case PASS -> ActionRole.PASS;
        case CHI -> throw new IllegalArgumentException("The three-player input has no role for " + type);
      };
    }
  }

  /** 局のトークン（1 行に 1 個）のフィールド。 */
  public enum RoundField implements CategoryField {
    DECISION_KIND(oneOf(EngineDecisionKind.class)),
    ROUND_WIND(oneOf(Wind.class)),
    /** 局（東 1 局 = 1 … 西 3 局 = 9）。 */
    KYOKU(FieldSpec.ids(ThreePlayerFormat.VARIANT.finalRound() + 1)),
    /** 本場（{@link InputSchema#STICK_SATURATION} 以上は同じ値にする。厳密な値はラダーと射影が使う）。 */
    HONBA(FieldSpec.count(STICK_SATURATION)),
    /** 供託の本数（上限は本場と同じ）。 */
    KYOTAKU(FieldSpec.count(STICK_SATURATION)),
    /** 自分の残りツモの回数（この先に鳴きと北抜きがないとき）。 */
    DRAWS_LEFT(FieldSpec.count(ThreePlayerFormat.VARIANT.maxDrawsPerSeat())),
    /**
     * 成立した槓の数（3 席の合計。四槓の判定に使う）。席のフィールドの KANS の合計と同じ値だが、四槓を局のトークンから直接読めるように
     * 重ねて持つ。
     */
    KANS(FieldSpec.count(Variant.MAX_KANS)),
    /** 今のツモが海底、または今の打牌が河底になる。 */
    LAST_LIVE_TILE(FieldSpec.flag()),
    /** まだ全員が第一打を終えていない。 */
    FIRST_CYCLE(FieldSpec.flag()),
    /** この局で鳴き（暗槓を含む）が一度でもあった。北抜きは数えない。 */
    CALL_MADE(FieldSpec.flag()),
    /** 南 3 局から西 2 局までで、トップが返し点（40,000 点）に届いていない（このままなら半荘が続く）。 */
    WEST_POSSIBLE(FieldSpec.flag()),
    /** 自分が捨て牌で振聴している（手が 13 枚の判断だけ）。 */
    FURITEN_DISCARD(FieldSpec.flag()),
    /** 自分が同巡の見逃しで振聴している。 */
    FURITEN_TEMPORARY(FieldSpec.flag()),
    /** 自分が立直後の見逃しで振聴している。 */
    FURITEN_RIICHI(FieldSpec.flag()),
    /** 直前のイベント（記録と同じ enum。北抜きを含む）。 */
    LAST_EVENT(oneOf(ObservationSchema.LastEvent.class)),
    /** 直前のイベントの席（相対席）。 */
    LAST_EVENT_SEAT(FieldSpec.ids(ThreePlayerFormat.SEATS)),
    /** 直前のイベントの牌。 */
    LAST_EVENT_TILE(FieldSpec.ids(ThreePlayerFormat.TILES)),
    /** 直前のイベントの牌が赤。 */
    LAST_EVENT_AKA(FieldSpec.flag()),
    /** 直前のツモが嶺上牌（北抜きの補充を含む）。 */
    LAST_EVENT_RINSHAN(FieldSpec.flag()),
    /** 槍槓の対象の槓の種類。 */
    LAST_KAN_KIND(oneOf(TurnEvent.KanKind.class)),
    /** 槓ドラの表示を保留している（後めくり）。 */
    DEFERRED_DORA(FieldSpec.flag());

    private final FieldSpec spec;

    RoundField(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** 席のトークン（観測席から見た相対席の順に 3 個）のフィールド。 */
  public enum SeatField implements CategoryField {
    /** 観測席から見た相対席（自分 = 1、下家 = 2、上家 = 3）。 */
    RELATIVE_SEAT(FieldSpec.ids(ThreePlayerFormat.SEATS)),
    /** 自風（東が親）。 */
    WIND(oneOf(Wind.class)),
    /** 今の順位（同点は起家に近い席が上）。 */
    RANK(FieldSpec.ids(ThreePlayerFormat.SEATS)),
    RIICHI(oneOf(RiichiState.class)),
    IPPATSU(FieldSpec.flag()),
    MENZEN(FieldSpec.flag()),
    /** まだ第一打を打っていない。 */
    BEFORE_FIRST_DISCARD(FieldSpec.flag()),
    /** 副露の数（暗槓を含む。北抜きは含まない）。 */
    MELDS(FieldSpec.count(Variant.MAX_MELDS)),
    KANS(FieldSpec.count(Variant.MAX_KANS)),
    /** 抜いた北の数（1 枚ごとに抜きドラ 1）。 */
    NUKI(FieldSpec.count(Tiles.COPIES)),
    DISCARDS(FieldSpec.count(ThreePlayerFormat.VARIANT.maxDiscards())),
    /** 見えている役牌の刻子・槓子の翻の合計。 */
    OPEN_YAKUHAI_HAN(FieldSpec.count(Limits.MAX_OPEN_YAKUHAI_HAN)),
    /** 副露の色（1 つの色と字牌だけのとき。混ざれば 0）。 */
    OPEN_SUIT(oneOf(OpenSuit.class)),
    /** 副露があり、どれも断么九と両立する。 */
    OPEN_TANYAO(FieldSpec.flag()),
    /** 副露のドラ（表ドラと赤）。 */
    OPEN_DORA(FieldSpec.count(Limits.maxDora(ThreePlayerFormat.VARIANT))),
    /** 流し満貫の可能性が残っている。 */
    NAGASHI(FieldSpec.flag()),
    /** 自分がこの席から直撃のロンで和了して、この席を抜くのに要る打点の格（自分の席は 0）。 */
    PASS_BY_DIRECT_RON(oneOf(PassGrade.class)),
    /** 自分がもう 1 人の他家からのロンで和了して、この席を抜くのに要る打点の格。 */
    PASS_BY_OTHER_RON(oneOf(PassGrade.class)),
    /** 自分がツモで和了して、この席を抜くのに要る打点の格。 */
    PASS_BY_TSUMO(oneOf(PassGrade.class)),
    /** この席が自分から直撃のロンで和了して、自分を抜くのに要る打点の格。 */
    PASSED_BY_RON(oneOf(PassGrade.class)),
    /** この席がツモで和了して、自分を抜くのに要る打点の格。 */
    PASSED_BY_TSUMO(oneOf(PassGrade.class));

    private final FieldSpec spec;

    SeatField(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** 牌のトークン（色を正準化した後の牌のトークンの順に 27 個）のフィールド。 */
  public enum TileField implements CategoryField {
    TILE(FieldSpec.ids(ThreePlayerFormat.TILES)),
    /** 自分の手にある枚数（ツモ牌を含む）。 */
    HAND(FieldSpec.count(Tiles.COPIES)),
    /** 自分の手にこの種類の赤がある。 */
    HAND_AKA(FieldSpec.flag()),
    /**
     * 今ツモった牌。局のフィールドにある直前のイベント（自分のツモ）の牌と同じ値だが、手出しとツモ切りを牌のトークンから直接読めるように
     * 重ねて持つ。
     */
    DRAWN(FieldSpec.flag()),
    /** 誰からも見える枚数（河・副露・抜いた北・ドラ表示牌）。 */
    VISIBLE(FieldSpec.count(Tiles.COPIES)),
    /** 自分から見えない枚数（4 − 誰からも見える枚数 − 自分の手にある枚数）。 */
    UNSEEN(FieldSpec.count(Tiles.COPIES)),
    /**
     * この牌 1 枚が持つドラの数（この牌をドラにする表示牌の数）。北は抜けば 1 枚ごとに抜きドラ 1 になるので、北のトークンは表示牌の数 + 1
     * にする。
     */
    DORA(FieldSpec.count(Tiles.COPIES + 1)),
    /** この種類の赤がまだ見えていない（河・副露・ドラ表示牌・自分の手のどこにもない）。ドラ表示牌の赤は、このフィールドだけが表す。 */
    UNSEEN_AKA(FieldSpec.flag()),
    /** 自分の河にある。 */
    OWN_RIVER(FieldSpec.flag()),
    /** 下家・上家それぞれに対する安全度。 */
    SAFETY(oneOf(Safety.class).times(ThreePlayerFormat.SEATS - 1)),
    /** 自分・下家・上家にとっての役牌の翻（0〜2）。 */
    YAKUHAI(FieldSpec.count(2).times(ThreePlayerFormat.SEATS)),
    /** 今の手の待ち牌（手が 13 枚の判断だけ）。 */
    CURRENT_WAIT(FieldSpec.flag()),
    /** 今の手の受け入れ牌（手が 13 枚の判断だけ）。 */
    CURRENT_UKEIRE(FieldSpec.flag());

    private final FieldSpec spec;

    TileField(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** イベントのトークン（公開された順に 0〜91 個。バッチで取る数は {@link InputSchema#EVENT_CAPACITY}）のフィールド。 */
  public enum EventField implements CategoryField {
    KIND(oneOf(PublicEvent.class)),
    /** イベントを起こした席（相対席）。 */
    AUTHOR(FieldSpec.ids(ThreePlayerFormat.SEATS)),
    /** 牌（打牌した牌、副露の最も小さい牌、抜いた北、ドラ表示牌）。 */
    TILE(FieldSpec.ids(ThreePlayerFormat.TILES)),
    /** 打牌した牌が赤。ドラ表示のイベントでは 0 にする（表示牌の赤は、牌のフィールドの UNSEEN_AKA が表す）。 */
    AKA(FieldSpec.flag()),
    TSUMOGIRI(FieldSpec.flag()),
    RIICHI_DECLARATION(FieldSpec.flag()),
    /** この打牌を鳴いた席（相対席）。 */
    CALLED_BY(FieldSpec.ids(ThreePlayerFormat.SEATS)),
    /** 副露で鳴いた牌。 */
    CALLED_TILE(FieldSpec.ids(ThreePlayerFormat.TILES)),
    /** 鳴いた牌を出した席（相対席）。 */
    SOURCE(FieldSpec.ids(ThreePlayerFormat.SEATS)),
    /** 副露に含まれる赤の出どころ。 */
    AKA_SOURCE(oneOf(Meld.AkaSource.class)),
    /** その席の何打目か（打牌だけ）。 */
    DISCARD_INDEX(FieldSpec.ids(ThreePlayerFormat.VARIANT.maxDiscards())),
    /** イベントを起こした席が、この時点で立直している。 */
    AUTHOR_IN_RIICHI(FieldSpec.flag());

    private final FieldSpec spec;

    EventField(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** 遷移（手を選んだ後の局面 1 つ。遷移が同じになる手のうち、代表でない手は代表の遷移を指す）のフィールド。 */
  public enum TransitionField implements CategoryField {
    /** 手の種類（続き・文脈・鳴いた牌を出した席は、これと局のフィールドから決まるので持たない）。 */
    ACTION(oneOf(Action.Type.class)),
    /** 主な牌（打牌する牌・鳴いた牌・槓の牌・抜く北）。 */
    MAIN_TILE(FieldSpec.ids(ThreePlayerFormat.TILES)),
    /** 赤を使うかと、ツモ切りかの選び方。 */
    TILE_SELECTION(oneOf(Action.TileSelection.class)),
    /** 鳴いた後に捨てる牌。 */
    CALL_DISCARD(FieldSpec.ids(ThreePlayerFormat.TILES)),
    /** 鳴いた後に赤を捨てる。 */
    CALL_DISCARD_AKA(FieldSpec.flag()),
    /** 遷移後の手が門前。 */
    MENZEN(FieldSpec.flag()),
    /** 遷移後の手の向聴数（3 つの形の最小。和了と九種九牌の遷移は 0）。 */
    SHANTEN(FieldSpec.ids(Limits.MAX_SHANTEN + 1)),
    /** 遷移後の手の一般形の向聴数。 */
    STANDARD_SHANTEN(FieldSpec.ids(Limits.MAX_STANDARD_SHANTEN + 1)),
    /** 遷移後の手の七対子の向聴数（副露があれば 0）。 */
    CHIITOI_SHANTEN(FieldSpec.ids(Limits.MAX_CHIITOI_SHANTEN + 1)),
    /** 遷移後の手の国士無双の向聴数（副露があれば 0）。 */
    KOKUSHI_SHANTEN(FieldSpec.ids(Limits.MAX_KOKUSHI_SHANTEN + 1)),
    /** 受け入れ牌の種類の数。 */
    UKEIRE_KINDS(FieldSpec.count(ThreePlayerFormat.TILES)),
    /** 形の上の待ち牌の種類の数。 */
    WAIT_KINDS(FieldSpec.count(Limits.MAX_WAIT_KINDS)),
    /** 遷移後の振聴。 */
    FURITEN(oneOf(Furiten.class)),
    /** 遷移後の手・副露・抜いた北のドラの数（表ドラ・赤・抜きドラ。裏ドラは含まない）。 */
    DORA(FieldSpec.count(Limits.maxDora(ThreePlayerFormat.VARIANT))),
    /** この手で和了したときの翻（裏ドラなし。{@link InputSchema#HAN_SATURATION} 以上は同じ値にし、役満は 0）。 */
    WIN_HAN(FieldSpec.count(HAN_SATURATION)),
    /** この手で和了したときの符の番号（記録と同じ番号）。 */
    WIN_FU(FieldSpec.count(ObservationSchema.MAX_FU_CODE)),
    /** この手で和了したときの役満の倍数。 */
    WIN_YAKUMAN(FieldSpec.count(ObservationSchema.MAX_YAKUMAN)),
    /** 立直して和了したときにめくる裏ドラ表示牌の数。 */
    URA_INDICATORS(FieldSpec.count(Limits.MAX_INDICATORS));

    private final FieldSpec spec;

    TransitionField(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /**
   * 関係のバイアスを作る属性（トークンごとに u8 で、埋め込まない）。関係のバイアスのコード（{@code RelationClass}、7 つのセグメント）はここから作る。
   *
   * <p>牌と安全度は、牌・イベントのフィールドにもある。関係のクラスをトークンの種類によらず同じ列から作れるように、ここにも重ねて持つ。
   */
  public enum RelationAttribute implements BoundedField {
    /** 牌（牌とイベントのトークン）。 */
    TILE(FieldSpec.ids(ThreePlayerFormat.TILES)),
    /** トークンが属する席（相対席。席とイベントのトークン）。 */
    OWNER(FieldSpec.ids(ThreePlayerFormat.SEATS)),
    /** イベントの番号（局のトークンでは、今までのイベントの数）。 */
    EVENT(FieldSpec.ids(ThreePlayerFormat.MAX_EVENTS + 1)),
    /** その席が立直を宣言したイベントの番号。 */
    OWNER_RIICHI_EVENT(FieldSpec.ids(ThreePlayerFormat.MAX_EVENTS)),
    /** その席の最後の打牌のイベントの番号。 */
    OWNER_LAST_DISCARD_EVENT(FieldSpec.ids(ThreePlayerFormat.MAX_EVENTS)),
    /** つながるイベントの番号（鳴かれた打牌、加槓の元のポン、槓ドラをめくった槓）。 */
    LINK(FieldSpec.ids(ThreePlayerFormat.MAX_EVENTS)),
    /** 下家・上家それぞれに対する安全度（牌のトークン）。 */
    SAFETY(oneOf(Safety.class).times(ThreePlayerFormat.SEATS - 1));

    private final FieldSpec spec;

    RelationAttribute(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** 局の数値のフィールド。 */
  public enum RoundScalar implements ScalarField {
    /** 王牌を除いた山の残り枚数。 */
    WALL(ThreePlayerFormat.VARIANT.liveWall());

    private final float scale;

    RoundScalar(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }
  }

  /** 席の数値のフィールド。 */
  public enum SeatScalar implements ScalarField {
    SCORE(100_000f),
    /** 自分との点差（この席 − 自分）。 */
    GAP_TO_SELF(POINT_SCALE);

    private final float scale;

    SeatScalar(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }
  }

  /** イベントの数値のフィールド。 */
  public enum EventScalar implements ScalarField {
    /** 今からどれだけ前か（後に続く公開のイベントの数）。 */
    AGE(EVENT_CAPACITY);

    private final float scale;

    EventScalar(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }
  }

  /** 遷移の数値のフィールド。 */
  public enum TransitionScalar implements ScalarField {
    /** 受け入れ牌の生きている枚数。 */
    LIVE_UKEIRE(ThreePlayerFormat.TILES),
    /** 1 ツモ以内に向聴が進む確率。 */
    HIT_WITHIN_1(1f),
    /** 2 ツモ以内に向聴が進む確率。 */
    HIT_WITHIN_2(1f),
    /** 3 ツモ以内に向聴が進む確率。 */
    HIT_WITHIN_3(1f),
    /** 立直しなくてもロンで役がつく待ち牌の、生きている枚数。 */
    LIVE_RON_YAKU_WAITS(LIVE_TILE_SCALE),
    /** 立直しなくてもツモで役がつく待ち牌の、生きている枚数。 */
    LIVE_TSUMO_YAKU_WAITS(LIVE_TILE_SCALE);

    private final float scale;

    TransitionScalar(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }
  }

  /** 打牌の値のヘッドに直接渡す遷移ごとの値（FP32）。このヘッドが線形に読み、候補の層も同じ列を BF16 で読む。値は三人打ちの精算（ツモ損あり）で作る。 */
  public enum DiscardExact implements ExactField {
    /** 聴牌なら 1。 */
    TENPAI(1f),
    /** 待ち牌の生きている枚数。 */
    LIVE_WAITS(LIVE_TILE_SCALE),
    /** ロンで和了して半荘が終わる割合（待ち牌と放銃する席の組ごとに、生きている枚数で重み付けする）。 */
    RON_ENDS(1f),
    /** ロンで和了して半荘が終わるときの効用の平均。 */
    RON_UTILITY(1f),
    /** ロンで和了したときの得点の平均。 */
    RON_POINTS(POINT_SCALE),
    /** ツモで和了して半荘が終わる割合。 */
    TSUMO_ENDS(1f),
    /** ツモで和了して半荘が終わるときの効用の平均。 */
    TSUMO_UTILITY(1f),
    /** ツモで和了したときの得点の平均（ツモ損あり）。 */
    TSUMO_POINTS(POINT_SCALE);

    private final float scale;

    DiscardExact(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }

    /**
     * このフィールドに入れる、待ちの射影の値の種類を返す。
     *
     * @return 待ちの射影の値の種類
     */
    public WaitProjection.Field projected() {
      return switch (this) {
        case TENPAI -> WaitProjection.Field.TENPAI;
        case LIVE_WAITS -> WaitProjection.Field.LIVE_WAITS;
        case RON_ENDS -> WaitProjection.Field.RON_ENDS;
        case RON_UTILITY -> WaitProjection.Field.RON_UTILITY;
        case RON_POINTS -> WaitProjection.Field.RON_POINTS;
        case TSUMO_ENDS -> WaitProjection.Field.TSUMO_ENDS;
        case TSUMO_UTILITY -> WaitProjection.Field.TSUMO_UTILITY;
        case TSUMO_POINTS -> WaitProjection.Field.TSUMO_POINTS;
      };
    }
  }

  /** ロンとツモのゲートに直接渡す手ごとの値（FP32）。点は三人打ちの精算（ツモ損あり）で求める。 */
  public enum WinExact implements ExactField {
    /** 半荘が終わる。 */
    ENDS(1f),
    /** 半荘が終わるときの効用（続くなら 0）。 */
    UTILITY(1f),
    /** 和了の後に 1 位。 */
    RANK_1(1f),
    /** 和了の後に 2 位。 */
    RANK_2(1f),
    /** 和了の後に 3 位。 */
    RANK_3(1f),
    /** 得点（裏ドラなし）。 */
    POINTS(POINT_SCALE),
    RENCHAN(1f),
    /** 包が成り立つ。 */
    PAO(1f),
    /** 裏ドラが乗りうる（立直している）。 */
    URA_ELIGIBLE(1f);

    private final float scale;

    WinExact(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }
  }

  /**
   * 立直のゲートに直接渡す値（FP32）。牌 x を切る立直と、同じ牌 x を切るダマとで、打牌の値のヘッドに渡す値の差を取る。手の列の先頭から置く。割る数は
   * 差を取る元のフィールドと同じ。
   */
  public enum RiichiExact implements ExactField {
    /** ロンで半荘が終わる割合の差。 */
    RON_ENDS(DiscardExact.RON_ENDS),
    /** ロンで終わるときの効用の差。 */
    RON_UTILITY(DiscardExact.RON_UTILITY),
    /** ロンの得点の差。 */
    RON_POINTS(DiscardExact.RON_POINTS),
    /** ツモで半荘が終わる割合の差。 */
    TSUMO_ENDS(DiscardExact.TSUMO_ENDS),
    /** ツモで終わるときの効用の差。 */
    TSUMO_UTILITY(DiscardExact.TSUMO_UTILITY),
    /** ツモの得点の差。 */
    TSUMO_POINTS(DiscardExact.TSUMO_POINTS);

    private final DiscardExact source;

    RiichiExact(DiscardExact source) {
      this.source = source;
    }

    /**
     * 差を取る元のフィールドを返す。
     *
     * @return 立直とダマで差を取るフィールド
     */
    public DiscardExact source() {
      return source;
    }

    @Override
    public float scale() {
      return source.scale();
    }
  }

  /** 遷移ごとのキーのコードの領域。キーは局 1・席 3・牌 27 の順に並び、コードは 3 つの領域を通した通し番号。 */
  public enum KeyBlock implements InputFormat.KeyRange {
    /** 局のキー（コードは 1 通り）。 */
    ROUND(1, 1),
    /** 席のキー。コードは相対席 3 通り × 鳴く牌・ロンする牌を出した席か 2 通り。 */
    SEAT(ThreePlayerFormat.SEATS, ThreePlayerFormat.SEATS * 2),
    /** 牌のキー。コードは遷移後の枚数 5 通り × 受け入れ牌か 2 通り × 待ちの分類 5 通り（待ちでない 0 を含む）× 捨てる牌か 2 通り。 */
    TILE(ThreePlayerFormat.TILES, (Tiles.COPIES + 1) * 2 * (WaitClass.values().length + 1) * 2);

    /** キーの総数（遷移の行の列の数）。 */
    public static final int KEY_COUNT;

    /** コードの総数（Ek・Ev の行の数）。 */
    public static final int CODE_COUNT;

    private static final int WAIT_DIGITS = WaitClass.values().length + 1;
    private static final Vocabulary<WaitClass> WAIT_CLASSES =
        Vocabulary.of(WaitClass.class, ThreePlayerFormat.VARIANT);
    private static final int[] FIRST_KEY;
    private static final int[] FIRST_CODE;

    static {
      KeyBlock[] blocks = values();
      FIRST_KEY = new int[blocks.length];
      FIRST_CODE = new int[blocks.length];
      int key = 0;
      int code = 0;
      for (KeyBlock block : blocks) {
        FIRST_KEY[block.ordinal()] = key;
        FIRST_CODE[block.ordinal()] = code;
        key += block.keys;
        code += block.codes;
      }
      if (code > FieldSpec.MAX_VOCABULARY) {
        throw new IllegalStateException("key codes exceed u8: " + code);
      }
      KEY_COUNT = key;
      CODE_COUNT = code;
    }

    private final int keys;
    private final int codes;

    KeyBlock(int keys, int codes) {
      this.keys = keys;
      this.codes = codes;
    }

    @Override
    public int keys() {
      return keys;
    }

    @Override
    public int codes() {
      return codes;
    }

    @Override
    public int firstKey() {
      return FIRST_KEY[ordinal()];
    }

    @Override
    public int firstCode() {
      return FIRST_CODE[ordinal()];
    }

    /**
     * 局のキーのコードを返す。
     *
     * @return コード
     */
    public static int roundCode() {
      return FIRST_CODE[ROUND.ordinal()];
    }

    /**
     * 席のキーのコードを返す。
     *
     * @param relativeSeat 観測席から見た相対席
     * @param involved この手に関わる席（鳴く牌・ロンする牌を出した席）なら true
     * @return コード
     */
    public static int seatCode(int relativeSeat, boolean involved) {
      return FIRST_CODE[SEAT.ordinal()]
          + Objects.checkIndex(relativeSeat, ThreePlayerFormat.SEATS) * 2
          + (involved ? 1 : 0);
    }

    /**
     * 牌のキーのコードを返す。
     *
     * @param afterCount 遷移後の手にあるこの牌の枚数
     * @param ukeire 遷移後の手の受け入れ牌なら true
     * @param wait 待ちの分類（待ちでなければ null）
     * @param discarded この手で捨てる牌なら true
     * @return コード
     */
    public static int tileCode(int afterCount, boolean ukeire, WaitClass wait, boolean discarded) {
      int code = Objects.checkIndex(afterCount, Tiles.COPIES + 1);
      code = code * 2 + (ukeire ? 1 : 0);
      code = code * WAIT_DIGITS + Ids.of(WAIT_CLASSES, wait);
      code = code * 2 + (discarded ? 1 : 0);
      return FIRST_CODE[TILE.ordinal()] + code;
    }
  }

  /** この宣言から作る三人打ちの入力の形式。 */
  private static final class Format implements InputFormat {

    private static final List<ActionRole> ROLES = List.of(ActionRole.values());

    /** キーの領域の大きさ。バッチの配置を作るときに判断ごとに読むので、1 つを使い回す。 */
    private static final Keys KEYS =
        new Keys(KeyBlock.ROUND.keys(), KeyBlock.SEAT.keys(), KeyBlock.TILE.keys(), KeyBlock.CODE_COUNT);

    @Override
    public Variant variant() {
      return ThreePlayerFormat.VARIANT;
    }

    @Override
    public RankUtility utility() {
      return UTILITY;
    }

    @Override
    public String fingerprint() {
      return InputSchema.fingerprint();
    }

    @Override
    public TokenLayout<?> round() {
      return ROUND;
    }

    @Override
    public TokenLayout<?> seat() {
      return SEAT;
    }

    @Override
    public TokenLayout<?> tile() {
      return TILE;
    }

    @Override
    public TokenLayout<?> event() {
      return EVENT;
    }

    @Override
    public TokenLayout<?> transition() {
      return TRANSITION;
    }

    @Override
    public TokenLayout<?> relation() {
      return RELATION;
    }

    @Override
    public int width(Tensor tensor) {
      return switch (tensor) {
        case ROUND_CATEGORY -> ROUND.width();
        case ROUND_SCALAR -> RoundScalar.values().length;
        case SEAT_CATEGORY -> SEAT.width();
        case SEAT_SCALAR -> SeatScalar.values().length;
        case TILE_CATEGORY -> TILE.width();
        case EVENT_CATEGORY -> EVENT.width();
        case EVENT_SCALAR -> EventScalar.values().length;
        case RELATION -> RELATION.width();
        case TRANSITION_CATEGORY -> TRANSITION.width();
        case TRANSITION_SCALAR -> TransitionScalar.values().length;
        case TRANSITION_CODE -> KeyBlock.KEY_COUNT;
        case DISCARD_EXACT -> DiscardExact.values().length;
        case ACTION_COLUMN -> ActionColumn.values().length;
        case ACTION_EXACT -> Math.max(WinExact.values().length, RiichiExact.values().length);
        case ROW_COLUMN -> RowColumn.values().length;
      };
    }

    @Override
    public int eventCapacity() {
      return EVENT_CAPACITY;
    }

    @Override
    public Keys keys() {
      return KEYS;
    }

    @Override
    public List<ActionRole> roles() {
      return ROLES;
    }

    @Override
    public ActionRole role(Action.Type type) {
      return ActionRole.of(type);
    }

    @Override
    public ActionRole discard() {
      return ActionRole.DISCARD;
    }

    @Override
    public ActionRole pass() {
      return ActionRole.PASS;
    }

    @Override
    public int relationColumn(RelationRole attribute, int repeat) {
      return RELATION.column(attribute(attribute), repeat);
    }

    @Override
    public int relationRepeats(RelationRole attribute) {
      return attribute(attribute).spec().repeats();
    }

    @Override
    public int transitionColumn(TransitionRole field) {
      return TRANSITION.column(
          switch (field) {
            case MAIN_TILE -> TransitionField.MAIN_TILE;
            case CALL_DISCARD -> TransitionField.CALL_DISCARD;
          });
    }

    @Override
    public RowWriter writer() {
      return new Writer();
    }

    private static RelationAttribute attribute(RelationRole attribute) {
      return switch (attribute) {
        case TILE -> RelationAttribute.TILE;
        case OWNER -> RelationAttribute.OWNER;
        case EVENT -> RelationAttribute.EVENT;
        case OWNER_RIICHI_EVENT -> RelationAttribute.OWNER_RIICHI_EVENT;
        case OWNER_LAST_DISCARD_EVENT -> RelationAttribute.OWNER_LAST_DISCARD_EVENT;
        case LINK -> RelationAttribute.LINK;
        case SAFETY -> RelationAttribute.SAFETY;
      };
    }
  }

  /** 三人打ちの入力を 1 行ずつ書くライター。 */
  private static final class Writer implements InputFormat.RowWriter {

    private final Tokenizer tokenizer = new Tokenizer();
    private final InputWriter writer = new InputWriter();

    @Override
    public void write(Observation o, HostInputs batch, int row) {
      tokenizer.write(o, writer.bind(batch.input(), row));
    }

    @Override
    public int canonicalTile(int physicalTile) {
      return tokenizer.canonicalTile(physicalTile);
    }
  }

  /** フィンガープリントを、初めて使うときに作る。クラスの初期化の順によって、まだ作られていない値を読むのを避けるため。 */
  private static final class Fingerprint {

    private static final String VALUE = digest().hex();

    private static SchemaDigest digest() {
      SchemaDigest digest = SchemaDigest.named("epsilon-prism-3p-input");
      Map<Class<?>, Vocabulary<?>> valueTypes = new LinkedHashMap<>();
      for (Class<?> type : SchemaDigest.sorted(CategoryField.class.getPermittedSubclasses())) {
        appendBounded(digest, type, valueTypes);
      }
      appendBounded(digest, RelationAttribute.class, valueTypes);
      valueTypes.values().forEach(digest::values);
      for (Class<?> type : SchemaDigest.sorted(ScalarField.class.getPermittedSubclasses())) {
        StringBuilder scalars = new StringBuilder();
        for (Object constant : type.getEnumConstants()) {
          scalars.append(constant).append('/').append(((ScalarField) constant).scale()).append(',');
        }
        digest.with(SchemaDigest.name(type), scalars);
      }
      for (Class<?> type : SchemaDigest.sorted(ExactField.class.getPermittedSubclasses())) {
        StringBuilder exact = new StringBuilder();
        for (Object constant : type.getEnumConstants()) {
          exact.append(constant).append('/').append(((ExactField) constant).scale()).append(',');
        }
        digest.with(SchemaDigest.name(type), exact);
      }
      Variant variant = ThreePlayerFormat.VARIANT;
      digest
          .values(Vocabulary.of(WaitClass.class, variant))
          .values(Vocabulary.of(ActionRole.class, variant))
          .values(Vocabulary.of(ActionColumn.class, variant))
          .values(Vocabulary.of(RowColumn.class, variant))
          .values(Vocabulary.of(SuitKey.class, variant));
      StringBuilder roles = new StringBuilder();
      for (Action.Type type : Vocabulary.of(Action.Type.class, variant).constants()) {
        roles.append(type).append('>').append(ActionRole.of(type)).append(',');
      }
      StringBuilder gates = new StringBuilder();
      for (ActionRole role : ActionRole.values()) {
        gates.append(role).append('=').append(role.gate()).append(',');
      }
      StringBuilder tensors = new StringBuilder();
      for (Tensor tensor : Tensor.values()) {
        tensors.append(tensor).append(':').append(tensor.axis()).append(':');
        tensors.append(tensor.element()).append(',');
      }
      StringBuilder grades = new StringBuilder();
      for (PassGrade grade : PassGrade.values()) {
        grades.append(grade).append('=');
        if (grade.scored()) {
          grades.append(grade.basePoints());
        } else {
          grades.append('-');
        }
        grades.append(',');
      }
      StringBuilder blocks = new StringBuilder();
      for (KeyBlock block : KeyBlock.values()) {
        blocks.append(block).append('=').append(block.keys()).append('x');
        blocks.append(block.codes()).append(',');
      }
      StringBuilder utility = new StringBuilder(UTILITY.name());
      for (int rank = 0; rank < variant.seats(); rank++) {
        utility.append(rank == 0 ? '=' : ',').append(UTILITY.value(variant, rank));
      }
      StringBuilder tiles = new StringBuilder();
      TileTokens tokens = TileTokens.of(variant);
      for (int token = 0; token < tokens.count(); token++) {
        tiles.append(tokens.kind(token)).append(',');
      }
      return digest
          .with("roles", roles)
          .with("gates", gates)
          .with(SchemaDigest.name(Tensor.class), tensors)
          .with(SchemaDigest.name(PassGrade.class), grades)
          .with(SchemaDigest.name(KeyBlock.class), blocks)
          .with("tiles", tiles)
          .with("movableSuits", tokens.movableSuits())
          .with("events", EVENT_CAPACITY)
          .with("actions", Limits.MAX_LEGAL_ACTIONS)
          .with("transitions", Limits.MAX_TRANSITIONS)
          .with("alignment", BatchPlan.TENSOR_ALIGNMENT)
          .with("utility", utility);
    }

    /** フィールドごとに名前・書き方・語彙・繰り返しの数を書き足し、値の enum の語彙を集める。 */
    private static void appendBounded(
        SchemaDigest digest, Class<?> type, Map<Class<?>, Vocabulary<?>> valueTypes) {
      StringBuilder fields = new StringBuilder();
      for (Object constant : type.getEnumConstants()) {
        FieldSpec spec = ((BoundedField) constant).spec();
        fields.append(constant).append(':').append(spec.coding()).append(':');
        fields.append(spec.vocabulary()).append('x').append(spec.repeats()).append(',');
        if (spec.values() != null) {
          valueTypes.putIfAbsent(spec.values().type(), spec.values());
        }
      }
      digest.with(SchemaDigest.name(type), fields);
    }
  }
}
