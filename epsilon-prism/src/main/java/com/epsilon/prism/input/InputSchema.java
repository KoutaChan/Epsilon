package com.epsilon.prism.input;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.RoundLimits;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.prism.Limits;
import com.epsilon.prism.codec.SchemaDigest;
import com.epsilon.prism.observe.ObservationSchema;
import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * PRISM の網が読む入力の欄を宣言する。
 *
 * <p>欄はトークンの型（局・席・牌・出来事・遷移）ごとの enum に置き、定数ごとに書き方・語彙・繰り返し（{@link FieldSpec}）を持たせる。列・
 * 埋め込みの名前空間・転送の幅は {@link TokenLayout} と {@link Tensor} が enum から一度だけ作る。書き手・網・HIP の表はそれを引き、番地を
 * 手で並べない。
 *
 * <p>値の決まりは 1 つだけにする。0 は常に「なし・0 個・偽」で、enum の値は {@code ordinal() + 1}、牌・席・0 始まりの番号は番号 + 1 で書く
 * （{@link Ids}）。自前の値の enum には NONE を置かない。語彙は規則から数えた上限（{@link Limits}）で取り、範囲外の値は例外にして切り捨て
 * ない。まとめる欄は Javadoc にまとめ方を書き、{@link Ids#saturated} だけで変換する。数値の欄と厳密な欄は、正規化の定数（scale）を欄が
 * 持つ。
 *
 * <p>同じ事実は 1 つの場所に置く。重ねるのは読む所が違う欄だけで、その理由を Javadoc に書く。
 *
 * <p>{@link #fingerprint()} は、欄の名前・順・書き方・語彙・繰り返し、数値の欄と厳密な欄の正規化、値の enum と外の enum の名前と順、手の
 * 種類から役割への写し方、容量と区画のそろえ、効用、色の正準化の鍵から作る。欄の enum は sealed の permits から集めるので、足し忘れが
 * ない。版の番号は持たない。指紋が合わなければ止めるだけにする。
 */
public final class InputSchema {

  /** 値は u8 で送るので、語彙はこれ以下。 */
  public static final int MAX_VOCABULARY = 256;

  /** 出来事トークンの容量（出来事トークンの最大 {@link Limits#MAX_EVENT_TOKENS} を入れる、塊の中の行ごとの枠）。 */
  public static final int EVENT_CAPACITY = 128;

  /** 本場・供託の欄がまとめ始める値（この値以上は同じ値にする）。 */
  public static final int STICK_SATURATION = 16;

  /** 即時の和了の翻の欄がまとめ始める値（数え役満）。 */
  public static final int HAN_SATURATION = 13;

  /** 塊の中の各テンソルの区画の先頭をそろえるバイト数。HostBatch と HIP の表はこれを引く。 */
  public static final int TENSOR_ALIGNMENT = 256;

  /** 終局の効用と待ちの射影に使う効用。 */
  public static final EpsilonUtilityProfile UTILITY = EpsilonUtilityProfile.TENHOU;

  /** 点差と得点の欄を割る数（1 万点）。 */
  private static final float POINT_SCALE = 10_000f;

  /** 生きた枚数の欄を割る数。 */
  private static final float LIVE_TILE_SCALE = 16f;

  /** 局の欄の配置。 */
  public static final TokenLayout<RoundField> ROUND = TokenLayout.of(RoundField.class);

  /** 席の欄の配置。 */
  public static final TokenLayout<SeatField> SEAT = TokenLayout.of(SeatField.class);

  /** 牌の欄の配置。 */
  public static final TokenLayout<TileField> TILE = TokenLayout.of(TileField.class);

  /** 出来事の欄の配置。 */
  public static final TokenLayout<EventField> EVENT = TokenLayout.of(EventField.class);

  /** 遷移の欄の配置。 */
  public static final TokenLayout<TransitionField> TRANSITION =
      TokenLayout.of(TransitionField.class);

  /** 関係の属性の配置。 */
  public static final TokenLayout<RelationAttribute> RELATION =
      TokenLayout.of(RelationAttribute.class);

  private InputSchema() {}

  /**
   * 網の入力の指紋を返す。model.json に書き、開くときに照らす。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  public static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /** 指紋の記述を返す（試験と診断のため）。 */
  static String descriptor() {
    return Fingerprint.DESCRIPTOR;
  }

  /** 語彙で値の範囲が決まる欄（埋め込む欄と、関係の属性）。 */
  public interface BoundedField {

    /**
     * 語彙と繰り返しを返す。
     *
     * @return 語彙と繰り返し
     */
    FieldSpec spec();
  }

  /** 埋め込みで読むカテゴリの欄。型ごとに 1 つの enum。 */
  public sealed interface CategoryField extends BoundedField
      permits RoundField, SeatField, TileField, EventField, TransitionField {}

  /** BF16 で送る数値の欄。値は scale で割ってから丸める。 */
  public sealed interface ScalarField
      permits RoundScalar, SeatScalar, EventScalar, TransitionScalar {

    /**
     * 正規化の定数を返す。
     *
     * @return 値を割る数
     */
    float scale();
  }

  /** FP32 のまま頭へ渡す厳密な欄（B と門の道）。値は scale で割ってから送る。 */
  public sealed interface ExactField permits DiscardExact, WinExact, RiichiExact {

    /**
     * 正規化の定数を返す。
     *
     * @return 値を割る数
     */
    float scale();
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
     * @param windTile 風の牌（{@link Tile#TON}〜{@link Tile#PEI}）
     * @return 風
     */
    public static Wind ofTile(int windTile) {
      return VALUES[Objects.checkIndex(windTile - Tile.TON, VALUES.length)];
    }
  }

  /** 相手から見た安全。振聴の規則で厳密に決まるものだけ。複数に当たるときは宣言の順で先のもの。 */
  public enum Safety {
    /** その席の河にある。 */
    GENBUTSU,
    /** その席の立直より後に通った。 */
    PASSED_AFTER_RIICHI,
    /** その席の最後の打牌より後に通った。 */
    PASSED_THIS_CYCLE
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

  public enum PublicEvent {
    DISCARD,
    CHI,
    PON,
    DAIMINKAN,
    ANKAN,
    KAKAN,
    /** 槓ドラの表示。公開された位置に置く（暗槓は槓の直後、大明槓・加槓は作者の次の打牌か槓の後）。初めのドラ表示は出来事にしない。 */
    DORA_INDICATOR
  }

  /** 副露の色（正準の色の順）。 */
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

  /** 待ち牌の類（その遷移の立直の状態でのロンの翻、裏なし）。待ちでない牌は符号の桁 0。 */
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

  /**
   * 追い越しに要る最小の和了の格。段は基本点で、本場・供託を含む点の移動は {@code WinPointDeltas} が足す。NOT_NEEDED と UNREACHABLE は
   * 和了の点を持たない（{@link #scored()} が false で、{@link #basePoints()} は例外にする）。
   */
  public enum PassGrade {
    /** 和了しなくても既に上にいる。 */
    NOT_NEEDED,
    HAN1_FU30(240),
    HAN1_FU40(320),
    HAN2_FU30(480),
    HAN2_FU40(640),
    HAN3_FU30(960),
    HAN3_FU40(1280),
    HAN4_FU30(1920),
    MANGAN(2000),
    HANEMAN(3000),
    BAIMAN(4000),
    SANBAIMAN(6000),
    YAKUMAN(8000),
    /** 役満でも届かない。 */
    UNREACHABLE;

    /** 和了の点を持たない段の基本点。 */
    private static final int UNSCORED = 0;

    private final int basePoints;

    PassGrade() {
      this(UNSCORED);
    }

    PassGrade(int basePoints) {
      this.basePoints = basePoints;
    }

    /**
     * 和了の点を持つ段（NOT_NEEDED と UNREACHABLE のほか）なら true を返す。
     *
     * @return 基本点を持つなら true
     */
    public boolean scored() {
      return basePoints != UNSCORED;
    }

    /**
     * 基本点を返す。
     *
     * @return 基本点
     * @throws IllegalStateException 和了の点を持たない段（NOT_NEEDED・UNREACHABLE）のとき
     */
    public int basePoints() {
      if (!scored()) {
        throw new IllegalStateException(name() + " has no base points");
      }
      return basePoints;
    }
  }

  /** 価値グラフでの手の役割。門は RIICHI・CALL・KAN・TSUMO・RON・KYUSHU の 6 つ。 */
  public enum ActionRole {
    /** 打牌（B が値を出す）。 */
    DISCARD(false),
    /** 立直（ダマ x を基準にした門）。 */
    RIICHI(true),
    /** チー・ポン・大明槓。 */
    CALL(true),
    /** 暗槓・加槓。 */
    KAN(true),
    TSUMO(true),
    /** ロン和了（槍槓を含む）。 */
    RON(true),
    KYUSHU(true),
    /** 見送り（応答の基準。値は 0）。 */
    PASS(false);

    private final boolean gate;

    ActionRole(boolean gate) {
      this.gate = gate;
    }

    /**
     * 門の頭が値を出す手なら true を返す。
     *
     * @return 門なら true
     */
    public boolean gate() {
      return gate;
    }

    /**
     * 手の種類から役割を返す。default を書かないので、種類を足すと組み立てで落ちる。
     *
     * @param type 手の種類
     * @return 役割
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
      };
    }
  }

  /**
   * 色の正準化の鍵。萬・筒・索の並べ替えは、宣言の順に鍵を比べ、先の鍵で決まらなければ次の鍵を見る。SuitCanonicalizer はこの順で比べ、
   * 指紋は宣言の順から作る。
   */
  public enum SuitKey {
    /** 自分の手。 */
    HAND,
    MELDS,
    INDICATORS,
    /** 出来事の列。 */
    EVENTS,
    /** 物理の色（ほかの鍵が全て同じときの決め手）。 */
    PHYSICAL
  }

  /** 局のトークン（行に 1 個）の欄。 */
  public enum RoundField implements CategoryField {
    DECISION_KIND(FieldSpec.of(EngineDecisionKind.class)),
    ROUND_WIND(FieldSpec.of(Wind.class)),
    /** 局（東 1 局 = 1 … 西 4 局 = 12）。 */
    KYOKU(FieldSpec.ids(HanchanProgression.WEST_END_KYOKU + 1)),
    /** 本場（{@link InputSchema#STICK_SATURATION} 以上はまとめる。厳密な値は梯子と射影が使う）。 */
    HONBA(FieldSpec.count(STICK_SATURATION)),
    /** 供託の本数（同上）。 */
    KYOTAKU(FieldSpec.count(STICK_SATURATION)),
    /** 自分の残りツモ（鳴きがないとき）。 */
    DRAWS_LEFT(FieldSpec.count(Limits.MAX_DRAWS_PER_SEAT)),
    /**
     * 成立した槓の数（4 席の和。四槓と嶺上の残りを決める）。席の欄の KANS の和と同じ事実だが、四槓・嶺上の判定を局のトークンで直に読む
     * ために重ねる。
     */
    KANS(FieldSpec.count(RoundLimits.MAX_KANS)),
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
    /** 直前の出来事（O と同じ enum）。 */
    LAST_EVENT(FieldSpec.of(ObservationSchema.LastEvent.class)),
    /** 直前の出来事の相対席。 */
    LAST_EVENT_SEAT(FieldSpec.ids(GameState.NUM_PLAYERS)),
    /** 直前の出来事の牌。 */
    LAST_EVENT_TILE(FieldSpec.ids(Tile.NUM_TILE_TYPES)),
    /** 直前の出来事の牌が赤。 */
    LAST_EVENT_AKA(FieldSpec.flag()),
    /** 直前のツモが嶺上。 */
    LAST_EVENT_RINSHAN(FieldSpec.flag()),
    /** 槍槓の元の槓の種類。 */
    LAST_KAN_KIND(FieldSpec.of(TurnEvent.KanKind.class)),
    /** 槓ドラの表示が後に延びている。 */
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

  /** 席のトークン（観測席から見た相対席の順に 4 個）の欄。 */
  public enum SeatField implements CategoryField {
    /** 観測席から見た相対席（自分 = 1、下家 = 2、対面 = 3、上家 = 4）。 */
    RELATIVE_SEAT(FieldSpec.ids(GameState.NUM_PLAYERS)),
    /** 自風（東が親）。 */
    WIND(FieldSpec.of(Wind.class)),
    /** 今の順位（同点は ScoreRanking の席順）。 */
    RANK(FieldSpec.ids(GameState.NUM_PLAYERS)),
    RIICHI(FieldSpec.of(RiichiState.class)),
    IPPATSU(FieldSpec.flag()),
    MENZEN(FieldSpec.flag()),
    /** 第一打の前。 */
    BEFORE_FIRST_DISCARD(FieldSpec.flag()),
    /** 副露の数（暗槓を含む）。 */
    MELDS(FieldSpec.count(RoundLimits.MAX_MELDS)),
    KANS(FieldSpec.count(RoundLimits.MAX_KANS)),
    DISCARDS(FieldSpec.count(RoundLimits.MAX_DISCARDS)),
    /** 公開の役牌の刻子・槓子の翻の和。 */
    OPEN_YAKUHAI_HAN(FieldSpec.count(Limits.MAX_OPEN_YAKUHAI_HAN)),
    /** 副露の色（1 つの色と字牌だけのとき。混ざれば 0）。 */
    OPEN_SUIT(FieldSpec.of(OpenSuit.class)),
    /** 副露があり、全て断么と両立する。 */
    OPEN_TANYAO(FieldSpec.flag()),
    /** 副露があり、全て刻子か槓子。 */
    OPEN_TRIPLETS(FieldSpec.flag()),
    /** 副露のドラ（表ドラと赤）。 */
    OPEN_DORA(FieldSpec.count(Limits.MAX_DORA)),
    /** 流し満貫の形が残っている。 */
    NAGASHI(FieldSpec.flag()),
    /** 自分がこの席からの直撃のロンで、この席を抜くのに要る格（自分の席は 0）。 */
    PASS_BY_DIRECT_RON(FieldSpec.of(PassGrade.class)),
    /** 自分が他家からのロンで、この席を抜くのに要る格。 */
    PASS_BY_OTHER_RON(FieldSpec.of(PassGrade.class)),
    /** 自分がツモで、この席を抜くのに要る格。 */
    PASS_BY_TSUMO(FieldSpec.of(PassGrade.class)),
    /** この席が自分からの直撃のロンで、自分を抜くのに要る格。 */
    PASSED_BY_RON(FieldSpec.of(PassGrade.class)),
    /** この席がツモで、自分を抜くのに要る格。 */
    PASSED_BY_TSUMO(FieldSpec.of(PassGrade.class));

    private final FieldSpec spec;

    SeatField(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** 牌のトークン（正準の色の順に 34 個）の欄。 */
  public enum TileField implements CategoryField {
    TILE(FieldSpec.ids(Tile.NUM_TILE_TYPES)),
    /** 自分の手の枚数（ツモ牌を含む）。 */
    HAND(FieldSpec.count(Tile.TILES_PER_TYPE)),
    /** 自分の手に赤がある。 */
    HAND_AKA(FieldSpec.flag()),
    /**
     * 今ツモった牌の種類。局の欄の直前の出来事（自分のツモ）の牌と同じ事実だが、手出しとツモ切りを牌のトークンから直に読むために重ねる。
     */
    DRAWN(FieldSpec.flag()),
    /** 公開で見える枚数（河・副露・ドラ表示）。 */
    VISIBLE(FieldSpec.count(Tile.TILES_PER_TYPE)),
    /** 自分から見えない枚数（4 − 公開 − 自分の手）。 */
    UNSEEN(FieldSpec.count(Tile.TILES_PER_TYPE)),
    /** ドラの倍率（この牌を指す表示の数）。 */
    DORA(FieldSpec.count(Tile.TILES_PER_TYPE)),
    /** この種の赤がまだ見えていない（河・副露・ドラ表示・自分の手のどれにも出ていない）。ドラ表示の赤はここだけが持つ。 */
    UNSEEN_AKA(FieldSpec.flag()),
    /** 自分の河にある。 */
    OWN_RIVER(FieldSpec.flag()),
    /** 下家・対面・上家から見た安全。 */
    SAFETY(FieldSpec.of(Safety.class).times(GameState.NUM_PLAYERS - 1)),
    /** 自分・下家・対面・上家にとっての役牌の翻（0〜2）。 */
    YAKUHAI(FieldSpec.count(2).times(GameState.NUM_PLAYERS)),
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

  /** 出来事のトークン（公開の順に 0〜{@link Limits#MAX_EVENT_TOKENS} 個、塊の枠は {@link InputSchema#EVENT_CAPACITY}）の欄。 */
  public enum EventField implements CategoryField {
    KIND(FieldSpec.of(PublicEvent.class)),
    /** 作者の相対席。 */
    AUTHOR(FieldSpec.ids(GameState.NUM_PLAYERS)),
    /** 牌（打牌の牌、副露の最小の牌、ドラ表示の牌）。 */
    TILE(FieldSpec.ids(Tile.NUM_TILE_TYPES)),
    /**
     * 打牌の牌が赤。ドラ表示の出来事では 0 にする（O は表示の赤を色ごとにしか持たず、同じ色の 5 の表示が 2 枚あるとどちらが赤か決まらない
     * ので、牌の欄の UNSEEN_AKA が持つ）。
     */
    AKA(FieldSpec.flag()),
    TSUMOGIRI(FieldSpec.flag()),
    RIICHI_DECLARATION(FieldSpec.flag()),
    /** この打牌を鳴いた相対席。 */
    CALLED_BY(FieldSpec.ids(GameState.NUM_PLAYERS)),
    /** 副露で鳴いた牌。 */
    CALLED_TILE(FieldSpec.ids(Tile.NUM_TILE_TYPES)),
    /** 副露の元の相対席。 */
    SOURCE(FieldSpec.ids(GameState.NUM_PLAYERS)),
    /** 副露の赤の由来。 */
    AKA_SOURCE(FieldSpec.of(Meld.AkaSource.class)),
    /** 作者の何打目か（打牌だけ）。 */
    DISCARD_INDEX(FieldSpec.ids(RoundLimits.MAX_DISCARDS)),
    /** 作者がこの時点で立直している。 */
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

  /** 遷移（手を打った後の 1 局面。代表でない双子は代表を指す）の欄。 */
  public enum TransitionField implements CategoryField {
    /** 手の種類（続き・文脈・元の席はこれと局の欄から決まるので持たない）。 */
    ACTION(FieldSpec.of(Action.Type.class)),
    /** 主牌（打牌・鳴いた牌・槓の牌）。 */
    MAIN_TILE(FieldSpec.ids(Tile.NUM_TILE_TYPES)),
    /** 赤とツモ切りの選び。 */
    TILE_SELECTION(FieldSpec.of(Action.TileSelection.class)),
    /** 鳴いた後に捨てる牌。 */
    CALL_DISCARD(FieldSpec.ids(Tile.NUM_TILE_TYPES)),
    /** 鳴いた後に赤を捨てる。 */
    CALL_DISCARD_AKA(FieldSpec.flag()),
    /** 後の手が門前。 */
    MENZEN(FieldSpec.flag()),
    /** 後の手の最小の向聴（和了・九種の遷移は 0）。 */
    SHANTEN(FieldSpec.ids(Limits.MAX_SHANTEN + 1)),
    /** 後の手の通常形の向聴。 */
    STANDARD_SHANTEN(FieldSpec.ids(Limits.MAX_STANDARD_SHANTEN + 1)),
    /** 後の手の七対子の向聴（副露があれば 0）。 */
    CHIITOI_SHANTEN(FieldSpec.ids(Limits.MAX_CHIITOI_SHANTEN + 1)),
    /** 後の手の国士の向聴（副露があれば 0）。 */
    KOKUSHI_SHANTEN(FieldSpec.ids(Limits.MAX_KOKUSHI_SHANTEN + 1)),
    /** 受け入れの種類の数。 */
    UKEIRE_KINDS(FieldSpec.count(Tile.NUM_TILE_TYPES)),
    /** 形の待ちの種類の数。 */
    WAIT_KINDS(FieldSpec.count(Limits.MAX_WAIT_KINDS)),
    /** 後の振聴。 */
    FURITEN(FieldSpec.of(Furiten.class)),
    /** 後の手と副露のドラ（表ドラと赤、裏なし）。 */
    DORA(FieldSpec.count(Limits.MAX_DORA)),
    /** 即時の和了の翻（裏なし。{@link InputSchema#HAN_SATURATION} 以上はまとめ、役満は 0）。 */
    WIN_HAN(FieldSpec.count(HAN_SATURATION)),
    /** 即時の和了の符の符号（ScoreMath.encodeFuCode）。 */
    WIN_FU(FieldSpec.count(Limits.MAX_FU_CODE)),
    /** 即時の和了の役満の倍率。 */
    WIN_YAKUMAN(FieldSpec.count(Limits.MAX_YAKUMAN)),
    /** 立直の和了で開く裏ドラ表示の数。 */
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
   * 関係の偏りを作る属性（トークンごとに u8、埋め込まない）。関係の偏りの 7 区画（{@code RelationClass}）はここから作る。
   *
   * <p>牌と安全は牌・出来事の欄にもある。関係の類を型によらず同じ列から作るために重ねる。
   */
  public enum RelationAttribute implements BoundedField {
    /** 牌（牌・出来事のトークン）。 */
    TILE(FieldSpec.ids(Tile.NUM_TILE_TYPES)),
    /** 持ち主の相対席（席・出来事のトークン）。 */
    OWNER(FieldSpec.ids(GameState.NUM_PLAYERS)),
    /** 出来事の番号（局のトークンは今の出来事の数）。 */
    EVENT(FieldSpec.ids(Limits.MAX_EVENT_TOKENS + 1)),
    /** 持ち主の立直宣言の出来事の番号。 */
    OWNER_RIICHI_EVENT(FieldSpec.ids(Limits.MAX_EVENT_TOKENS)),
    /** 持ち主の最後の打牌の出来事の番号。 */
    OWNER_LAST_DISCARD_EVENT(FieldSpec.ids(Limits.MAX_EVENT_TOKENS)),
    /** 繋がる出来事の番号（鳴かれた打牌、加槓の元のポン、槓ドラを開いた槓）。 */
    LINK(FieldSpec.ids(Limits.MAX_EVENT_TOKENS)),
    /** 下家・対面・上家から見た安全（牌のトークン）。 */
    SAFETY(FieldSpec.of(Safety.class).times(GameState.NUM_PLAYERS - 1));

    private final FieldSpec spec;

    RelationAttribute(FieldSpec spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec() {
      return spec;
    }
  }

  /** 局の数値の欄。 */
  public enum RoundScalar implements ScalarField {
    /** 王牌を除く山の残り。 */
    WALL(RoundLimits.MAX_WALL);

    private final float scale;

    RoundScalar(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }
  }

  /** 席の数値の欄。 */
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

  /** 出来事の数値の欄。 */
  public enum EventScalar implements ScalarField {
    /** 今からの距離（公開の出来事の数）。 */
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

  /** 遷移の数値の欄。 */
  public enum TransitionScalar implements ScalarField {
    /** 受け入れの生きた枚数。 */
    LIVE_UKEIRE(Tile.NUM_TILE_TYPES),
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

  /** B の道（遷移ごと、FP32）。B の頭が線形に読み、候補層も同じ列を BF16 で読む。 */
  public enum DiscardExact implements ExactField {
    /** 聴牌なら 1。 */
    TENPAI(1f),
    /** 待ちの生きた枚数。 */
    LIVE_WAITS(LIVE_TILE_SCALE),
    /** ロンの和了で半荘が終わる割合（待ちと元を生きた枚数で重み付け）。 */
    RON_ENDS(1f),
    /** ロンの和了で終わるときの効用の平均。 */
    RON_UTILITY(1f),
    /** ロンの和了の得点の平均。 */
    RON_POINTS(POINT_SCALE),
    /** ツモの和了で半荘が終わる割合。 */
    TSUMO_ENDS(1f),
    /** ツモの和了で終わるときの効用の平均。 */
    TSUMO_UTILITY(1f),
    /** ツモの和了の得点の平均。 */
    TSUMO_POINTS(POINT_SCALE);

    private final float scale;

    DiscardExact(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }
  }

  /** RON・TSUMO の門の道（手ごと、FP32）。 */
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
    /** 裏ドラの資格がある（立直している）。 */
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

  /** RIICHI の門の道（立直 x とダマ x の B の道の差、FP32）。手の列の先頭から置く。正規化は差を取る B の道の欄と同じ。 */
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
     * 差を取る B の道の欄を返す。
     *
     * @return 立直 x とダマ x で差を取る欄
     */
    public DiscardExact source() {
      return source;
    }

    @Override
    public float scale() {
      return source.scale();
    }
  }

  /** 手ごとの索引の列（INT32、埋め込まない）。番号は 0 始まりで、なしは −1。 */
  public enum ActionColumn {
    ROW,
    FIRST_TRANSITION,
    /** 遷移の数（代表でない双子は 0）。 */
    TRANSITIONS,
    /** 役割（ActionRole の ordinal）。 */
    ROLE,
    /** 基準の手（立直 x はダマ x、応答は PASS、ほかは −1）。 */
    BASE,
    /** 双子の代表の手（代表は自分）。 */
    REPRESENTATIVE
  }

  /** 行ごとの索引の列（INT32、埋め込まない）。 */
  public enum RowColumn {
    FIRST_EVENT,
    EVENTS,
    FIRST_ACTION,
    ACTIONS,
    FIRST_TRANSITION,
    TRANSITIONS,
    /** 判断の種類（EngineDecisionKind の ordinal）。K7 が価値グラフの形を選ぶために、局の欄と重ねて持つ。 */
    DECISION_KIND
  }

  /** 遷移ごとの鍵の符号の区画。鍵は局 1・席 4・牌 34 の順で、符号は区画の先頭からの通し番号。 */
  public enum KeyBlock {
    /** 局の鍵（1 通り）。 */
    ROUND(1, 1),
    /** 席の鍵：相対席 4 × この手の元か 2。 */
    SEAT(GameState.NUM_PLAYERS, GameState.NUM_PLAYERS * 2),
    /** 牌の鍵：後の枚数 5 × 受け入れか 2 × 待ちの類 5（待ちでない 0 を含む）× 捨てる牌か 2。 */
    TILE(
        Tile.NUM_TILE_TYPES,
        (Tile.TILES_PER_TYPE + 1) * 2 * (WaitClass.values().length + 1) * 2);

    /** 鍵の総数（遷移の行の列の数）。 */
    public static final int KEY_COUNT;

    /** 符号の総数（Ek・Ev の行の数）。 */
    public static final int CODE_COUNT;

    private static final int WAIT_DIGITS = WaitClass.values().length + 1;
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
      if (code > MAX_VOCABULARY) {
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

    /**
     * この区画の鍵の数を返す。
     *
     * @return 鍵の数
     */
    public int keys() {
      return keys;
    }

    /**
     * この区画の符号の数を返す。
     *
     * @return 符号の数
     */
    public int codes() {
      return codes;
    }

    /**
     * この区画の最初の鍵の列を返す。
     *
     * @return 鍵の列
     */
    public int firstKey() {
      return FIRST_KEY[ordinal()];
    }

    /**
     * この区画の最初の符号を返す。
     *
     * @return 符号
     */
    public int firstCode() {
      return FIRST_CODE[ordinal()];
    }

    /**
     * 局の鍵の符号を返す。
     *
     * @return 符号
     */
    public static int roundCode() {
      return FIRST_CODE[ROUND.ordinal()];
    }

    /**
     * 席の鍵の符号を返す。
     *
     * @param relativeSeat 観測席から見た相対席
     * @param involved この手の元か相手なら true
     * @return 符号
     */
    public static int seatCode(int relativeSeat, boolean involved) {
      return FIRST_CODE[SEAT.ordinal()]
          + Objects.checkIndex(relativeSeat, GameState.NUM_PLAYERS) * 2
          + (involved ? 1 : 0);
    }

    /**
     * 牌の鍵の符号を返す。
     *
     * @param afterCount 後の手のこの牌の枚数
     * @param ukeire 後の手の受け入れなら true
     * @param wait 待ちの類（null なら待ちでない）
     * @param discarded この手で捨てる牌なら true
     * @return 符号
     */
    public static int tileCode(int afterCount, boolean ukeire, WaitClass wait, boolean discarded) {
      int code = Objects.checkIndex(afterCount, Tile.TILES_PER_TYPE + 1);
      code = code * 2 + (ukeire ? 1 : 0);
      code = code * WAIT_DIGITS + Ids.of(wait);
      code = code * 2 + (discarded ? 1 : 0);
      return FIRST_CODE[TILE.ordinal()] + code;
    }
  }

  /** テンソルの行の軸。 */
  public enum Axis {
    /** 判断の行。 */
    ROW,
    /** 行ごとに 4 個の席トークン。 */
    SEAT,
    /** 行ごとに 34 個の牌トークン。 */
    TILE,
    /** 行ごとに 0〜EVENT_CAPACITY 個の出来事トークン。 */
    EVENT,
    /** 局・席・牌・出来事の全てのトークン。 */
    TOKEN,
    TRANSITION,
    ACTION
  }

  /** 要素の型。 */
  public enum Element {
    U8(1),
    /** bfloat16（最近接偶数で丸める）。 */
    BF16(2),
    FP32(4),
    I32(4);

    private final int bytes;

    Element(int bytes) {
      this.bytes = bytes;
    }

    /**
     * 1 要素のバイト数を返す。
     *
     * @return バイト数
     */
    public int bytes() {
      return bytes;
    }
  }

  /** 1 塊で送るテンソル。区画はこの順に並べ、各区画の先頭を {@link InputSchema#TENSOR_ALIGNMENT} B にそろえる。 */
  public enum Tensor {
    ROUND_CATEGORY(Axis.ROW, Element.U8),
    ROUND_SCALAR(Axis.ROW, Element.BF16),
    SEAT_CATEGORY(Axis.SEAT, Element.U8),
    SEAT_SCALAR(Axis.SEAT, Element.BF16),
    TILE_CATEGORY(Axis.TILE, Element.U8),
    EVENT_CATEGORY(Axis.EVENT, Element.U8),
    EVENT_SCALAR(Axis.EVENT, Element.BF16),
    RELATION(Axis.TOKEN, Element.U8),
    TRANSITION_CATEGORY(Axis.TRANSITION, Element.U8),
    TRANSITION_SCALAR(Axis.TRANSITION, Element.BF16),
    /** 遷移ごとの鍵の符号。 */
    TRANSITION_CODE(Axis.TRANSITION, Element.U8),
    /** B の道。 */
    DISCARD_EXACT(Axis.TRANSITION, Element.FP32),
    ACTION_COLUMN(Axis.ACTION, Element.I32),
    /** 門の道（RON・TSUMO は WinExact、RIICHI は RiichiExact を先頭から。ほかは 0）。 */
    ACTION_EXACT(Axis.ACTION, Element.FP32),
    ROW_COLUMN(Axis.ROW, Element.I32);

    private final Axis axis;
    private final Element element;

    Tensor(Axis axis, Element element) {
      this.axis = axis;
      this.element = element;
    }

    /**
     * 行の軸を返す。
     *
     * @return 軸
     */
    public Axis axis() {
      return axis;
    }

    /**
     * 要素の型を返す。
     *
     * @return 要素の型
     */
    public Element element() {
      return element;
    }

    /**
     * 軸の 1 つあたりの要素の数を返す。塊の準備のときに呼ぶ（熱い道では呼ばない）。
     *
     * @return 要素の数
     */
    public int width() {
      return switch (this) {
        case ROUND_CATEGORY -> InputSchema.ROUND.width();
        case ROUND_SCALAR -> RoundScalar.values().length;
        case SEAT_CATEGORY -> InputSchema.SEAT.width();
        case SEAT_SCALAR -> SeatScalar.values().length;
        case TILE_CATEGORY -> InputSchema.TILE.width();
        case EVENT_CATEGORY -> InputSchema.EVENT.width();
        case EVENT_SCALAR -> EventScalar.values().length;
        case RELATION -> InputSchema.RELATION.width();
        case TRANSITION_CATEGORY -> InputSchema.TRANSITION.width();
        case TRANSITION_SCALAR -> TransitionScalar.values().length;
        case TRANSITION_CODE -> KeyBlock.KEY_COUNT;
        case DISCARD_EXACT -> DiscardExact.values().length;
        case ACTION_COLUMN -> ActionColumn.values().length;
        case ACTION_EXACT -> Math.max(WinExact.values().length, RiichiExact.values().length);
        case ROW_COLUMN -> RowColumn.values().length;
      };
    }
  }

  /** 遅延で作る（初期化の順の罠を避ける）。 */
  private static final class Fingerprint {

    private static final String DESCRIPTOR;
    private static final String VALUE;

    static {
      SchemaDigest digest = digest();
      DESCRIPTOR = digest.toString();
      VALUE = digest.hex();
    }

    private static SchemaDigest digest() {
      SchemaDigest digest = SchemaDigest.named("epsilon-prism-v2-input");
      Set<Class<? extends Enum<?>>> valueTypes = new LinkedHashSet<>();
      for (Class<?> type : SchemaDigest.sorted(CategoryField.class.getPermittedSubclasses())) {
        appendBounded(digest, type, valueTypes);
      }
      appendBounded(digest, RelationAttribute.class, valueTypes);
      for (Class<? extends Enum<?>> type : valueTypes) {
        digest.values(type);
      }
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
      digest.values(WaitClass.class).values(ActionRole.class);
      digest.values(ActionColumn.class).values(RowColumn.class).values(SuitKey.class);
      StringBuilder roles = new StringBuilder();
      for (Action.Type type : Action.Type.values()) {
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
          .with(SchemaDigest.name(Tensor.class), tensors)
          .with(SchemaDigest.name(PassGrade.class), grades)
          .with(SchemaDigest.name(KeyBlock.class), blocks)
          .with("events", EVENT_CAPACITY)
          .with("actions", Limits.MAX_LEGAL_ACTIONS)
          .with("transitions", Limits.MAX_TRANSITIONS)
          .with("alignment", TENSOR_ALIGNMENT)
          .with("utility", UTILITY.name());
    }

    /** 欄ごとに名前・書き方・語彙・繰り返しを書き、値の enum（自前も外も）を集める。 */
    private static void appendBounded(
        SchemaDigest digest, Class<?> type, Set<Class<? extends Enum<?>>> valueTypes) {
      StringBuilder fields = new StringBuilder();
      for (Object constant : type.getEnumConstants()) {
        FieldSpec spec = ((BoundedField) constant).spec();
        fields.append(constant).append(':').append(spec.coding()).append(':');
        fields.append(spec.vocabulary()).append('x').append(spec.repeats()).append(',');
        if (spec.values() != null) {
          valueTypes.add(spec.values());
        }
      }
      digest.with(SchemaDigest.name(type), fields);
    }
  }
}
