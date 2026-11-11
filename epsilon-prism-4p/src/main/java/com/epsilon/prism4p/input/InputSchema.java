package com.epsilon.prism4p.input;

import static com.epsilon.prism.input.FieldSpec.oneOf;

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
import com.epsilon.prism.FourPlayerFormat;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.EventTable;
import com.epsilon.prism.input.EventTable.PublicEvent;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.FieldSpec.BoundedField;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.Ids;
import com.epsilon.prism.input.InputBlob;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Composition;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.PassGrade;
import com.epsilon.prism.input.SuitCanonicalizer;
import com.epsilon.prism.input.SuitCanonicalizer.SuitKey;
import com.epsilon.prism.input.TileSafety;
import com.epsilon.prism.input.TileSafety.Safety;
import com.epsilon.prism.input.TokenLayout;
import com.epsilon.prism.input.WaitProjection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * PRISM のネットワークが読む入力のフィールドを宣言する。
 *
 * <p>フィールドはトークンの型（局・席・牌・イベント・遷移）ごとの enum に置き、定数ごとに値の書き方・語彙・繰り返し（{@link FieldSpec}）を
 * 持たせる。列・埋め込みの名前空間・転送の幅は、{@link TokenLayout} と {@link #FORMAT} が enum から一度だけ作る。ライター・ネットワーク・
 * HIP の表はそれを引き、位置を手で並べない。
 *
 * <p>値の書き方は 1 通りにそろえる。0 は常に「なし・0 個・偽」を表す。enum の値は語彙の中の位置 + 1、牌・席・0 始まりの番号は番号 + 1 で
 * 書く（{@link Ids}）。このクラスで宣言する値の enum には NONE を置かない。語彙の大きさはルールから数えた上限（{@link Limits}）で決め、
 * 範囲外の値は切り捨てずに例外にする。上限でまとめるフィールドは、まとめ方を Javadoc に書き、変換は {@link Ids#saturated} だけで行う。
 * 数値のフィールドと厳密なフィールドは、正規化の定数（scale）をフィールド自身が持つ。
 *
 * <p>同じ事実は 1 か所にだけ置く。重ねて置くのは読む所が違うフィールドだけで、その理由を Javadoc に書く。
 *
 * <p>{@link #fingerprint()} は次のものから作る。フィールドの名前・順・書き方・語彙・繰り返し、数値のフィールドと厳密なフィールドの正規化、
 * 値の enum（このクラスのものと外のもの）の名前と順、手の種類から扱いへの対応、容量とテンソルの位置のそろえ方、効用、色の正規化のキー。フィールドの
 * enum は sealed の permits から集めるので、足し忘れがない。バージョン番号は持たず、フィンガープリントが合わなければ止めるだけにする。
 */
public final class InputSchema {

  /** バッチの 1 行が持てるイベントのトークンの数。イベントのトークンの最大数 {@link Limits#MAX_EVENT_TOKENS} が入る。 */
  public static final int EVENT_CAPACITY = 128;

  /** 本場・供託のフィールドで、この値以上を同じ値にまとめる。 */
  public static final int STICK_SATURATION = 16;

  /** この手で和了するときの翻のフィールドで、この値（数え役満）以上を同じ値にまとめる。 */
  public static final int HAN_SATURATION = 13;

  /** 終局の効用と、待ちの射影に使う効用。 */
  public static final RankUtility UTILITY = RankUtility.TENHOU;

  /** 点差と得点のフィールドを割る数（1 万点）。 */
  private static final float POINT_SCALE = 10_000f;

  /** 生きた枚数のフィールドを割る数。 */
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
  public static final TokenLayout<TransitionField> TRANSITION =
      TokenLayout.of(TransitionField.class);

  /** 関係の属性の配置。 */
  public static final TokenLayout<RelationAttribute> RELATION =
      TokenLayout.of(RelationAttribute.class);

  /** 四人打ちの入力形式。 */
  public static final InputFormat FORMAT = new Format();

  private InputSchema() {}

  /**
   * ネットワークの入力のフィンガープリントを返す。model.json に書き、モデルを開くときに照合する。
   *
   * @return SHA-256 の先頭 8 バイトの 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /** 埋め込みで読むカテゴリのフィールド。トークンの型ごとに 1 つの enum。 */
  public sealed interface CategoryField extends BoundedField
      permits RoundField, SeatField, TileField, EventField, TransitionField {}

  /** BF16 で送る数値のフィールド。値は scale で割ってから丸める。 */
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

  /** FP32 のままヘッドへ渡す厳密なフィールド（打牌の値のヘッドとゲートのヘッドが読む）。値は scale で割ってから送る。 */
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

  public enum Wind {
    EAST,
    SOUTH,
    WEST,
    NORTH;

    private static final Wind[] VALUES = values();

    /**
     * 風の牌から風を返す。
     *
     * @param windTile 風の牌の種類（東〜北）
     * @return 風
     */
    public static Wind ofTile(int windTile) {
      return VALUES[Objects.checkIndex(windTile - Tiles.TON, VALUES.length)];
    }
  }

  /** 遷移の後の振聴。 */
  public enum Furiten {
    /** 待ちが自分の河にある。 */
    DISCARD,
    /** 同巡の見逃し（次の自分の打牌まで）。 */
    TEMPORARY,
    /** 立直後の見逃し（局の終わりまで）。 */
    RIICHI_PERSISTENT
  }

  /** 副露の色（色を正規化した順）。 */
  public enum OpenSuit {
    /** 第 1 の色と字牌だけ。 */
    FIRST,
    /** 第 2 の色と字牌だけ。 */
    SECOND,
    /** 第 3 の色と字牌だけ。 */
    THIRD,
    /** 字牌だけ。 */
    HONORS
  }

  /** 待ち牌の分類（その遷移の立直の状態でロンしたときの翻、裏なし）。待ちでない牌は、牌のキーのコードでこの桁を 0 にする。 */
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

  /** 価値グラフでの手の扱い。ゲートのヘッドが値を出すのは、立直・鳴き・槓・ツモ・ロン・九種九牌の 6 つ。 */
  public enum ActionRole implements InputFormat.Role {
    /** 打牌（値は打牌の値のヘッドが出す）。 */
    DISCARD(false, Composition.DISCARD_VALUE, false),
    /** 立直（同じ牌 x をダマで切る手を基準にしたゲート）。 */
    RIICHI(true, Composition.RIICHI_GATE, false),
    /** チー・ポン・大明槓。 */
    CALL(true, Composition.RESPONSE_GATE, false),
    /** 暗槓・加槓。 */
    KAN(true, Composition.TURN_GATE, false),
    TSUMO(true, Composition.TURN_GATE, true),
    /** ロン和了（槍槓を含む）。 */
    RON(true, Composition.RESPONSE_GATE, true),
    KYUSHU(true, Composition.TURN_GATE, true),
    /** 見送り（応答の基準。値は 0）。 */
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
     * 手の種類から扱いを返す。switch に default を書かないので、手の種類を足すとコンパイルが通らなくなる。
     *
     * @param type 手の種類
     * @return 手の扱い
     * @throws IllegalArgumentException 四人打ちの入力に無い種類（抜き）のとき
     */
    public static ActionRole of(Action.Type type) {
      return switch (type) {
        case DAHAI -> ActionRole.DISCARD;
        case RIICHI_DAHAI -> ActionRole.RIICHI;
        case CHI, PON, DAIMINKAN -> ActionRole.CALL;
        case ANKAN, KAKAN -> ActionRole.KAN;
        case TSUMO_AGARI -> ActionRole.TSUMO;
        case RON_AGARI -> ActionRole.RON;
        case KYUSHU_KYUHAI -> ActionRole.KYUSHU;
        case PASS -> ActionRole.PASS;
        case NUKI -> throw new IllegalArgumentException("The four-player input has no role for " + type);
      };
    }
  }

  /** 局のトークン（1 行に 1 個）のフィールド。 */
  public enum RoundField implements CategoryField {
    DECISION_KIND(oneOf(EngineDecisionKind.class)),
    ROUND_WIND(oneOf(Wind.class)),
    /** 局（東 1 局 = 1 … 西場の最後の局）。 */
    KYOKU(FieldSpec.ids(FourPlayerFormat.VARIANT.finalRound() + 1)),
    /** 本場（{@link InputSchema#STICK_SATURATION} 以上はまとめる。厳密な値はラダーと待ちの射影が使う）。 */
    HONBA(FieldSpec.count(STICK_SATURATION)),
    /** 供託の本数（本場と同じく、{@link InputSchema#STICK_SATURATION} 以上はまとめる）。 */
    KYOTAKU(FieldSpec.count(STICK_SATURATION)),
    /** 鳴きがないときの、自分の残りのツモの数。 */
    DRAWS_LEFT(FieldSpec.count(FourPlayerFormat.VARIANT.maxDrawsPerSeat())),
    /**
     * 成立した槓の数（4 席の和。四槓と嶺上の残りを決める）。席のフィールドの KANS の和と同じ事実だが、四槓・嶺上の判定を局のトークンから
     * 直接読めるように重ねて置く。
     */
    KANS(FieldSpec.count(Variant.MAX_KANS)),
    /** 今のツモが海底、または今の打牌が河底になる。 */
    LAST_LIVE_TILE(FieldSpec.flag()),
    /** 全員の第一打の前。 */
    FIRST_CYCLE(FieldSpec.flag()),
    /** この局で鳴き（暗槓を含む）が一度でもあった。 */
    CALL_MADE(FieldSpec.flag()),
    /** 南 4 局以降で、トップが 30000 点に届いていない（このままなら半荘が続く）。 */
    WEST_POSSIBLE(FieldSpec.flag()),
    /** 自分は捨て牌で振聴（13 枚の判断だけ）。 */
    FURITEN_DISCARD(FieldSpec.flag()),
    /** 自分は同巡の見逃しで振聴。 */
    FURITEN_TEMPORARY(FieldSpec.flag()),
    /** 自分は立直後の見逃しで振聴。 */
    FURITEN_RIICHI(FieldSpec.flag()),
    /** 直前のイベント（判断の記録と同じ enum）。 */
    LAST_EVENT(oneOf(ObservationSchema.LastEvent.class)),
    /** 直前のイベントを起こした席（相対席）。 */
    LAST_EVENT_SEAT(FieldSpec.ids(FourPlayerFormat.SEATS)),
    /** 直前のイベントの牌。 */
    LAST_EVENT_TILE(FieldSpec.ids(Tiles.KINDS)),
    /** 直前のイベントの牌が赤。 */
    LAST_EVENT_AKA(FieldSpec.flag()),
    /** 直前のツモが嶺上。 */
    LAST_EVENT_RINSHAN(FieldSpec.flag()),
    /** 槍槓の対象になった槓の種類。 */
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

  /** 席のトークン（観測席から見た相対席の順に 4 個）のフィールド。 */
  public enum SeatField implements CategoryField {
    /** 観測席から見た相対席（自分 = 1、下家 = 2、対面 = 3、上家 = 4）。 */
    RELATIVE_SEAT(FieldSpec.ids(FourPlayerFormat.SEATS)),
    /** 自風（東が親）。 */
    WIND(oneOf(Wind.class)),
    /** 今の順位（同点は起家に近い席が上）。 */
    RANK(FieldSpec.ids(FourPlayerFormat.SEATS)),
    RIICHI(oneOf(RiichiState.class)),
    IPPATSU(FieldSpec.flag()),
    MENZEN(FieldSpec.flag()),
    /** 第一打の前。 */
    BEFORE_FIRST_DISCARD(FieldSpec.flag()),
    /** 副露の数（暗槓を含む）。 */
    MELDS(FieldSpec.count(Variant.MAX_MELDS)),
    KANS(FieldSpec.count(Variant.MAX_KANS)),
    DISCARDS(FieldSpec.count(FourPlayerFormat.VARIANT.maxDiscards())),
    /** 公開の役牌の刻子・槓子の翻の和。 */
    OPEN_YAKUHAI_HAN(FieldSpec.count(Limits.MAX_OPEN_YAKUHAI_HAN)),
    /** 副露の色（1 つの色と字牌だけのとき。混ざれば 0）。 */
    OPEN_SUIT(oneOf(OpenSuit.class)),
    /** 副露があり、全て断么と両立する。 */
    OPEN_TANYAO(FieldSpec.flag()),
    /** 副露があり、全て刻子か槓子。 */
    OPEN_TRIPLETS(FieldSpec.flag()),
    /** 副露のドラ（表ドラと赤）。 */
    OPEN_DORA(FieldSpec.count(Limits.maxDora(FourPlayerFormat.VARIANT))),
    /** 流し満貫の形が残っている。 */
    NAGASHI(FieldSpec.flag()),
    /** 自分がこの席からの直撃のロンで、この席を抜くのに要る格（自分の席は 0）。 */
    PASS_BY_DIRECT_RON(oneOf(PassGrade.class)),
    /** 自分が他家からのロンで、この席を抜くのに要る格。 */
    PASS_BY_OTHER_RON(oneOf(PassGrade.class)),
    /** 自分がツモで、この席を抜くのに要る格。 */
    PASS_BY_TSUMO(oneOf(PassGrade.class)),
    /** この席が自分からの直撃のロンで、自分を抜くのに要る格。 */
    PASSED_BY_RON(oneOf(PassGrade.class)),
    /** この席がツモで、自分を抜くのに要る格。 */
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

  /** 牌のトークン（色を正規化した順に 34 個）のフィールド。 */
  public enum TileField implements CategoryField {
    TILE(FieldSpec.ids(Tiles.KINDS)),
    /** 自分の手の枚数（ツモ牌を含む）。 */
    HAND(FieldSpec.count(Tiles.COPIES)),
    /** 自分の手に赤がある。 */
    HAND_AKA(FieldSpec.flag()),
    /**
     * 今ツモった牌の種類。局のフィールドにある直前のイベント（自分のツモ）の牌と同じ事実だが、手出しとツモ切りを牌のトークンから直接
     * 読めるように重ねて置く。
     */
    DRAWN(FieldSpec.flag()),
    /** 公開で見える枚数（河・副露・ドラ表示）。 */
    VISIBLE(FieldSpec.count(Tiles.COPIES)),
    /** 自分から見えない枚数（4 − 公開 − 自分の手）。 */
    UNSEEN(FieldSpec.count(Tiles.COPIES)),
    /** ドラの倍率（この牌を指す表示の数）。 */
    DORA(FieldSpec.count(Tiles.COPIES)),
    /** この種類の赤がまだ見えていない（河・副露・ドラ表示・自分の手のどれにも出ていない）。ドラ表示の赤はこのフィールドだけが持つ。 */
    UNSEEN_AKA(FieldSpec.flag()),
    /** 自分の河にある。 */
    OWN_RIVER(FieldSpec.flag()),
    /** 下家・対面・上家のそれぞれに対する安全度。 */
    SAFETY(oneOf(Safety.class).times(FourPlayerFormat.SEATS - 1)),
    /** 自分・下家・対面・上家のそれぞれにとっての役牌の翻（0〜2）。 */
    YAKUHAI(FieldSpec.count(2).times(FourPlayerFormat.SEATS)),
    /** 今の手の形の待ち（13 枚の判断だけ）。 */
    CURRENT_WAIT(FieldSpec.flag()),
    /** 今の手の受け入れ（13 枚の判断だけ）。 */
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

  /**
   * イベントのトークン（公開の順に 0〜{@link Limits#MAX_EVENT_TOKENS} 個。バッチの 1 行に {@link InputSchema#EVENT_CAPACITY} 個まで
   * 入る）のフィールド。
   */
  public enum EventField implements CategoryField {
    KIND(oneOf(PublicEvent.class)),
    /** イベントを起こした席（相対席）。 */
    AUTHOR(FieldSpec.ids(FourPlayerFormat.SEATS)),
    /** 牌（打牌の牌、副露の最小の牌、ドラ表示牌）。 */
    TILE(FieldSpec.ids(Tiles.KINDS)),
    /**
     * 打牌の牌が赤。ドラ表示のイベントでは 0 にし、表示牌の赤は牌のフィールドの UNSEEN_AKA が持つ。判断の記録は表示牌の赤を色ごとに
     * しか持たず、同じ色の 5 の表示牌が 2 枚あると、どちらが赤か決まらないため。
     */
    AKA(FieldSpec.flag()),
    TSUMOGIRI(FieldSpec.flag()),
    RIICHI_DECLARATION(FieldSpec.flag()),
    /** この打牌を鳴いた相対席。 */
    CALLED_BY(FieldSpec.ids(FourPlayerFormat.SEATS)),
    /** 副露で鳴いた牌。 */
    CALLED_TILE(FieldSpec.ids(Tiles.KINDS)),
    /** 副露で鳴いた牌を出した席（相対席）。 */
    SOURCE(FieldSpec.ids(FourPlayerFormat.SEATS)),
    /** 副露の赤の由来。 */
    AKA_SOURCE(oneOf(Meld.AkaSource.class)),
    /** 打った席の何打目か（打牌のイベントだけ）。 */
    DISCARD_INDEX(FieldSpec.ids(FourPlayerFormat.VARIANT.maxDiscards())),
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

  /** 遷移（手を打った後の 1 局面。同じ手のうち代表でないものは、代表の遷移を指す）のフィールド。 */
  public enum TransitionField implements CategoryField {
    /** 手の種類（続き・文脈・元の席は、これと局のフィールドから決まるので持たない）。 */
    ACTION(oneOf(Action.Type.class)),
    /** 主牌（打牌・鳴いた牌・槓の牌）。 */
    MAIN_TILE(FieldSpec.ids(Tiles.KINDS)),
    /** 赤とツモ切りの選択。 */
    TILE_SELECTION(oneOf(Action.TileSelection.class)),
    /** 鳴いた後に捨てる牌。 */
    CALL_DISCARD(FieldSpec.ids(Tiles.KINDS)),
    /** 鳴いた後に赤を捨てる。 */
    CALL_DISCARD_AKA(FieldSpec.flag()),
    /** 遷移の後の手が門前。 */
    MENZEN(FieldSpec.flag()),
    /** 遷移の後の手の最小の向聴（和了・九種九牌の遷移は 0）。 */
    SHANTEN(FieldSpec.ids(Limits.MAX_SHANTEN + 1)),
    /** 遷移の後の手の通常形の向聴。 */
    STANDARD_SHANTEN(FieldSpec.ids(Limits.MAX_STANDARD_SHANTEN + 1)),
    /** 遷移の後の手の七対子の向聴（副露があれば 0）。 */
    CHIITOI_SHANTEN(FieldSpec.ids(Limits.MAX_CHIITOI_SHANTEN + 1)),
    /** 遷移の後の手の国士の向聴（副露があれば 0）。 */
    KOKUSHI_SHANTEN(FieldSpec.ids(Limits.MAX_KOKUSHI_SHANTEN + 1)),
    /** 受け入れの種類の数。 */
    UKEIRE_KINDS(FieldSpec.count(Tiles.KINDS)),
    /** 形の上の待ちの種類の数。 */
    WAIT_KINDS(FieldSpec.count(Limits.MAX_WAIT_KINDS)),
    /** 遷移の後の振聴。 */
    FURITEN(oneOf(Furiten.class)),
    /** 遷移の後の手と副露のドラ（表ドラと赤、裏なし）。 */
    DORA(FieldSpec.count(Limits.maxDora(FourPlayerFormat.VARIANT))),
    /** この手で和了するときの翻（裏なし。{@link InputSchema#HAN_SATURATION} 以上はまとめ、役満は 0）。 */
    WIN_HAN(FieldSpec.count(HAN_SATURATION)),
    /** この手で和了するときの符の番号（判断の記録と同じ符の番号）。 */
    WIN_FU(FieldSpec.count(ObservationSchema.MAX_FU_CODE)),
    /** この手で和了するときの役満の倍率。 */
    WIN_YAKUMAN(FieldSpec.count(ObservationSchema.MAX_YAKUMAN)),
    /** 立直して和了したときに開く裏ドラ表示の数。 */
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
   * 関係のバイアスを作る属性（トークンごとに u8 で、埋め込まない）。関係のバイアスの 7 つのクラス（{@code RelationClass}）はここから作る。
   *
   * <p>牌と安全度は、牌・イベントのフィールドにもある。トークンの型によらず同じ列から関係のクラスを作れるように、ここにも重ねて置く。
   */
  public enum RelationAttribute implements BoundedField {
    /** 牌（牌・イベントのトークン）。 */
    TILE(FieldSpec.ids(Tiles.KINDS)),
    /** トークンの席（相対席）。席のトークンではその席、イベントのトークンではイベントを起こした席。 */
    OWNER(FieldSpec.ids(FourPlayerFormat.SEATS)),
    /** イベントの番号（局のトークンでは、今までのイベントの数）。 */
    EVENT(FieldSpec.ids(Limits.MAX_EVENT_TOKENS + 1)),
    /** トークンの席が立直を宣言したイベントの番号。 */
    OWNER_RIICHI_EVENT(FieldSpec.ids(Limits.MAX_EVENT_TOKENS)),
    /** トークンの席の最後の打牌のイベントの番号。 */
    OWNER_LAST_DISCARD_EVENT(FieldSpec.ids(Limits.MAX_EVENT_TOKENS)),
    /** つながるイベントの番号（鳴かれた打牌、加槓の元のポン、槓ドラを開いた槓）。 */
    LINK(FieldSpec.ids(Limits.MAX_EVENT_TOKENS)),
    /** 下家・対面・上家のそれぞれに対する安全度（牌のトークン）。 */
    SAFETY(oneOf(Safety.class).times(FourPlayerFormat.SEATS - 1));

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
    /** 王牌を除く山の残り枚数。 */
    WALL(FourPlayerFormat.VARIANT.liveWall());

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
    /** 自分との点差（この席の点 − 自分の点）。 */
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
    /** 今からの距離（このイベントより後にある公開のイベントの数）。 */
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
    /** 受け入れの生きた枚数。 */
    LIVE_UKEIRE(Tiles.KINDS),
    /** 1 ツモ以内に向聴が進む確率。 */
    HIT_WITHIN_1(1f),
    /** 2 ツモ以内に向聴が進む確率。 */
    HIT_WITHIN_2(1f),
    /** 3 ツモ以内に向聴が進む確率。 */
    HIT_WITHIN_3(1f),
    /** 立直なしでロンに役がある待ちの生きた枚数。 */
    LIVE_RON_YAKU_WAITS(LIVE_TILE_SCALE),
    /** 立直なしでツモに役がある待ちの生きた枚数。 */
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

  /** 打牌の値のヘッドへ渡す厳密なフィールド（遷移ごと、FP32）。このヘッドが線形に読み、候補層も同じ列を BF16 で読む。 */
  public enum DiscardExact implements ExactField {
    /** 聴牌なら 1。 */
    TENPAI(1f),
    /** 待ちの生きた枚数。 */
    LIVE_WAITS(LIVE_TILE_SCALE),
    /** ロン和了で半荘が終わる割合（待ちと、その牌を出す席の組ごとに、生きた枚数で重みを付ける）。 */
    RON_ENDS(1f),
    /** ロン和了で半荘が終わるときの効用の平均。 */
    RON_UTILITY(1f),
    /** ロン和了の得点の平均。 */
    RON_POINTS(POINT_SCALE),
    /** ツモ和了で半荘が終わる割合。 */
    TSUMO_ENDS(1f),
    /** ツモ和了で半荘が終わるときの効用の平均。 */
    TSUMO_UTILITY(1f),
    /** ツモ和了の得点の平均。 */
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
     * @return 待ちの射影のフィールド
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

  /** ロン・ツモのゲートのヘッドへ渡す厳密なフィールド（手ごと、FP32）。 */
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
    /** 和了の後に 4 位。 */
    RANK_4(1f),
    /** 得点（裏なし）。 */
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
   * 立直のゲートのヘッドへ渡す厳密なフィールド（FP32）。値は、牌 x を立直して切る手とダマで切る手で、打牌の値のヘッドへ渡すフィールドの差を取ったもの。
   * 手の列の先頭から置く。正規化は、差を取る、打牌の値のフィールドと同じ。
   */
  public enum RiichiExact implements ExactField {
    /** ロンで終わる割合の差。 */
    RON_ENDS(DiscardExact.RON_ENDS),
    /** ロンの効用の差。 */
    RON_UTILITY(DiscardExact.RON_UTILITY),
    /** ロンの得点の差。 */
    RON_POINTS(DiscardExact.RON_POINTS),
    /** ツモで終わる割合の差。 */
    TSUMO_ENDS(DiscardExact.TSUMO_ENDS),
    /** ツモの効用の差。 */
    TSUMO_UTILITY(DiscardExact.TSUMO_UTILITY),
    /** ツモの得点の差。 */
    TSUMO_POINTS(DiscardExact.TSUMO_POINTS);

    private final DiscardExact source;

    RiichiExact(DiscardExact source) {
      this.source = source;
    }

    /**
     * 差を取る、打牌の値のフィールドを返す。
     *
     * @return 立直 x とダマ x で差を取るフィールド
     */
    public DiscardExact source() {
      return source;
    }

    @Override
    public float scale() {
      return source.scale();
    }
  }

  /** 遷移ごとのキーの範囲と、キーが取るコードの数。キーは局 1・席 4・牌 34 の順に並ぶ。コードは局・席・牌の範囲をまたいだ通し番号にする。 */
  public enum KeyBlock implements InputFormat.KeyRange {
    /** 局のキー（コードは 1 つ）。 */
    ROUND(1, 1),
    /** 席のキー。コードは、相対席 4 × この手を打つ席かその相手か 2。 */
    SEAT(FourPlayerFormat.SEATS, FourPlayerFormat.SEATS * 2),
    /** 牌のキー。コードは、遷移の後の枚数 5 × 受け入れか 2 × 待ちの分類 5（待ちでない 0 を含む）× 捨てる牌か 2。 */
    TILE(
        Tiles.KINDS,
        (Tiles.COPIES + 1) * 2 * (WaitClass.values().length + 1) * 2);

    /** キーの総数（遷移の行の列の数）。 */
    public static final int KEY_COUNT;

    /** コードの総数（Ek・Ev の行の数）。 */
    public static final int CODE_COUNT;

    private static final int WAIT_DIGITS = WaitClass.values().length + 1;
    private static final Vocabulary<WaitClass> WAIT_CLASSES =
        Vocabulary.of(WaitClass.class, FourPlayerFormat.VARIANT);
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
     * @param involved この手を打つ席か、その相手の席なら true
     * @return コード
     */
    public static int seatCode(int relativeSeat, boolean involved) {
      return FIRST_CODE[SEAT.ordinal()]
          + Objects.checkIndex(relativeSeat, FourPlayerFormat.SEATS) * 2
          + (involved ? 1 : 0);
    }

    /**
     * 牌のキーのコードを返す。
     *
     * @param afterCount 遷移の後の手にある、この牌の枚数
     * @param ukeire 遷移の後の手の受け入れなら true
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

  /** この宣言を読む四人打ちの入力形式。 */
  private static final class Format implements InputFormat {

    private static final List<ActionRole> ROLES = List.of(ActionRole.values());

    /** キーの範囲。バッチの配置を決めるときに判断ごとに読むので、1 つを使い回す。 */
    private static final Keys KEYS =
        new Keys(
            KeyBlock.ROUND.keys(), KeyBlock.SEAT.keys(), KeyBlock.TILE.keys(), KeyBlock.CODE_COUNT);

    @Override
    public Variant variant() {
      return FourPlayerFormat.VARIANT;
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

  /** 四人打ちの 1 行分のライター。 */
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

  /** 初めて使うときに作る（クラスの初期化の順で値が欠けるのを避ける）。 */
  private static final class Fingerprint {

    private static final Set<Class<?>> MOVED =
        Set.of(
            Tensor.class,
            RowColumn.class,
            ActionColumn.class,
            PublicEvent.class,
            Safety.class,
            SuitKey.class,
            PassGrade.class);

    private static final String VALUE = digest().hex();

    private static SchemaDigest digest() {
      SchemaDigest digest = SchemaDigest.named("epsilon-prism-v2-input");
      Map<Class<?>, Vocabulary<?>> valueTypes = new LinkedHashMap<>();
      for (Class<?> type : SchemaDigest.sorted(CategoryField.class.getPermittedSubclasses())) {
        appendBounded(digest, type, valueTypes);
      }
      appendBounded(digest, RelationAttribute.class, valueTypes);
      valueTypes.values().forEach(vocabulary -> values(digest, vocabulary));
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
      digest.values(FourPlayerFormat.values(WaitClass.class)).values(FourPlayerFormat.values(ActionRole.class));
      values(digest, FourPlayerFormat.values(ActionColumn.class));
      values(digest, FourPlayerFormat.values(RowColumn.class));
      values(digest, FourPlayerFormat.values(SuitKey.class));
      StringBuilder roles = new StringBuilder();
      for (Action.Type type : Vocabulary.of(Action.Type.class, FourPlayerFormat.VARIANT).constants()) {
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
      return digest
          .with("roles", roles)
          .with("gates", gates)
          .with(name(Tensor.class), tensors)
          .with(name(PassGrade.class), grades)
          .with(SchemaDigest.name(KeyBlock.class), blocks)
          .with("events", EVENT_CAPACITY)
          .with("actions", Limits.MAX_LEGAL_ACTIONS)
          .with("transitions", Limits.MAX_TRANSITIONS)
          .with("alignment", BatchPlan.TENSOR_ALIGNMENT)
          .with("utility", UTILITY.name());
    }

    /**
     * 型の名前を返す。このクラスの外へ移した型は、フィンガープリントを変えないように、移す前の入れ子の名前で書く。
     */
    private static String name(Class<?> type) {
      return MOVED.contains(type) ? "InputSchema$" + type.getSimpleName() : SchemaDigest.name(type);
    }

    /** 語彙の型の名前と値の並びを書く（{@link SchemaDigest#values} と同じ形）。 */
    private static void values(SchemaDigest digest, Vocabulary<?> vocabulary) {
      digest.with(name(vocabulary.type()), vocabulary);
    }

    /** フィールドごとに名前・書き方・語彙・繰り返しを書き、値の enum（このクラスのものも外のものも）を集める。 */
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
