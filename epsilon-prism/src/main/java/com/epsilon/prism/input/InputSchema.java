package com.epsilon.prism.input;

import static com.epsilon.prism.input.FieldSpec.oneOf;

import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.VariantScoped;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.EventTable.PublicEvent;
import com.epsilon.prism.input.FieldSpec.BoundedField;
import com.epsilon.prism.input.InputBlob.Composition;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.TileSafety.Safety;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/** ルールの入力フィールド、語彙、列の並びとキーの範囲。 */
public final class InputSchema {
  private InputSchema() {}

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

  public static final InputFormat FOUR_PLAYER = new InputFormat(Variant.TENHOU_4P, Set.of());
  public static final InputFormat THREE_PLAYER = new InputFormat(Variant.TENHOU_3P, Set.of());

  /** ルールの基本入力形式を返す。 */
  public static InputFormat of(Variant variant) {
    return switch (variant) {
      case TENHOU_4P -> FOUR_PLAYER;
      case TENHOU_3P -> THREE_PLAYER;
    };
  }

  private interface Scoped extends VariantScoped {
    @Override
    default RuleFlag requires() {
      return null;
    }
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
  public sealed interface ExactField extends InputFormat.Exact
      permits DiscardExact, WinExact, RiichiExact {

    @Override
    default Tensor tensor() {
      return switch (this) {
        case DiscardExact discard -> Tensor.DISCARD_EXACT;
        case WinExact win -> Tensor.ACTION_EXACT;
        case RiichiExact riichi -> Tensor.ACTION_EXACT;
      };
    }
  }

  public enum Wind implements Scoped {
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

    @Override
    public boolean in(Variant variant) {
      return ordinal() < variant.seats();
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

  /** 副露の色（色を正規化した順）。入れ替えない数牌の色（三人打ちの萬子）は、正規化した色とは別の値で書く。 */
  public enum OpenSuit implements Scoped {
    /** 第 1 の色と字牌だけ。 */
    FIRST,
    /** 第 2 の色と字牌だけ。 */
    SECOND,
    /** 第 3 の色と字牌だけ。 */
    THIRD,
    /** 萬子と字牌だけ（萬子を入れ替えないルール）。 */
    MANZU,
    /** 字牌だけ。 */
    HONORS;

    @Override
    public boolean in(Variant variant) {
      return switch (this) {
        case MANZU -> !TileTokens.of(variant).movable(Tiles.Group.MAN);
        case THIRD -> Integer.bitCount(TileTokens.of(variant).movableSuits()) == Tiles.NUMBER_SUITS;
        default -> true;
      };
    }
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

  /** 価値グラフでの手の扱い。北抜きは北を抜くルールに限る。 */
  public enum ActionRole implements InputFormat.Role, Scoped {
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
    /** 北抜き。 */
    NUKI(true, Composition.TURN_GATE, false),
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
        case NUKI -> ActionRole.NUKI;
      };
    }

    @Override
    public boolean in(Variant variant) {
      return this != NUKI || variant.has(RuleFlag.NUKI);
    }
  }

  /** 局のトークン（1 行に 1 個）のフィールド。 */
  public enum RoundField implements CategoryField, InputColumn.Field {
    DECISION_KIND(
        v -> oneOf(EngineDecisionKind.class, v),
        (r, c) -> c.value(0, r.observation.decisionKind())),
    ROUND_WIND(
        v -> oneOf(Wind.class, v),
        (r, c) -> c.value(0, Wind.ofTile(r.observation.roundWindTile()))),
    /** 局（東 1 局 = 1 … 西場の最後の局）。 */
    KYOKU(v -> FieldSpec.ids(v.finalRound() + 1), (r, c) -> c.id(0, r.observation.kyoku())),
    /** 本場（{@link InputSchema#STICK_SATURATION} 以上はまとめる。厳密な値はラダーと待ちの射影が使う）。 */
    HONBA(
        v -> FieldSpec.count(STICK_SATURATION),
        (r, c) -> c.count(0, Ids.saturated(r.observation.honba(), STICK_SATURATION))),
    /** 供託の本数（本場と同じく、{@link InputSchema#STICK_SATURATION} 以上はまとめる）。 */
    KYOTAKU(
        v -> FieldSpec.count(STICK_SATURATION),
        (r, c) -> c.count(0, Ids.saturated(r.observation.kyotaku(), STICK_SATURATION))),
    /** 鳴きがないときの、自分の残りのツモの数。 */
    DRAWS_LEFT(
        v -> FieldSpec.count(v.maxDrawsPerSeat()),
        (r, c) -> c.count(0, r.drawsLeft(r.observation))),
    /**
     * 成立した槓の数（4 席の和。四槓と嶺上の残りを決める）。席のフィールドの KANS の和と同じ事実だが、四槓・嶺上の判定を局のトークンから
     * 直接読めるように重ねて置く。
     */
    KANS(v -> FieldSpec.count(Variant.MAX_KANS), (r, c) -> c.count(0, r.events.kans())),
    /** 今のツモが海底、または今の打牌が河底になる。 */
    LAST_LIVE_TILE(v -> FieldSpec.flag(), (r, c) -> c.flag(0, r.observation.remainingWall() == 0)),
    /** 全員の第一打の前。 */
    FIRST_CYCLE(v -> FieldSpec.flag(), (r, c) -> c.flag(0, r.observation.firstCycle())),
    /** この局で鳴き（暗槓を含む）が一度でもあった。 */
    CALL_MADE(v -> FieldSpec.flag(), (r, c) -> c.flag(0, r.observation.callMade())),
    /** 南 4 局以降で、トップが 30000 点に届いていない（このままなら半荘が続く）。 */
    WEST_POSSIBLE(v -> FieldSpec.flag(), (r, c) -> c.flag(0, r.westPossible(r.observation))),
    /** 自分は捨て牌で振聴（13 枚の判断だけ）。 */
    FURITEN_DISCARD(
        v -> FieldSpec.flag(), (r, c) -> c.flag(0, (r.currentWaits & r.safety.river(0)) != 0L)),
    /** 自分は同巡の見逃しで振聴。 */
    FURITEN_TEMPORARY(
        v -> FieldSpec.flag(),
        (r, c) -> c.flag(0, r.furitenTemporary && r.selfRiichi == RiichiState.NONE)),
    /** 自分は立直後の見逃しで振聴。 */
    FURITEN_RIICHI(
        v -> FieldSpec.flag(),
        (r, c) -> c.flag(0, r.furitenTemporary && r.selfRiichi != RiichiState.NONE)),
    /** 直前のイベント（判断の記録と同じ enum）。 */
    LAST_EVENT(
        v -> oneOf(ObservationSchema.LastEvent.class, v),
        (r, c) -> c.value(0, r.observation.lastEvent())),
    /** 直前のイベントを起こした席（相対席）。 */
    LAST_EVENT_SEAT(
        v -> FieldSpec.ids(v.seats()), (r, c) -> c.id(0, r.observation.lastEventSeat())),
    /** 直前のイベントの牌。 */
    LAST_EVENT_TILE(
        v -> FieldSpec.ids(TileTokens.of(v).count()),
        (r, c) -> c.id(0, r.token(r.observation.lastEventTile()))),
    /** 直前のイベントの牌が赤。 */
    LAST_EVENT_AKA(v -> FieldSpec.flag(), (r, c) -> c.flag(0, r.observation.lastEventAka())),
    /** 直前のツモが嶺上。 */
    LAST_EVENT_RINSHAN(
        v -> FieldSpec.flag(), (r, c) -> c.flag(0, r.observation.lastEventRinshan())),
    /** 槍槓の対象になった槓の種類。 */
    LAST_KAN_KIND(
        v -> oneOf(TurnEvent.KanKind.class, v), (r, c) -> c.value(0, r.observation.lastKanKind())),
    /** 槓ドラの表示を保留している（後めくり）。 */
    DEFERRED_DORA(v -> FieldSpec.flag(), (r, c) -> c.flag(0, r.observation.deferredDora()));

    private final InputColumn.Source source;

    @Override
    public InputColumn.Source source() {
      return source;
    }

    private final Function<Variant, FieldSpec> spec;

    RoundField(Function<Variant, FieldSpec> spec, InputColumn.Source source) {
      this.source = source;
      this.spec = spec;
    }

    @Override
    public FieldSpec spec(Variant variant) {
      return spec.apply(variant);
    }
  }

  /** 席のトークン（観測席から見た相対席の順）のフィールド。 */
  public enum SeatField implements CategoryField, Scoped {
    /** 観測席から見た相対席（自分 = 1）。 */
    RELATIVE_SEAT(v -> FieldSpec.ids(v.seats())),
    /** 自風（東が親）。 */
    WIND(v -> oneOf(Wind.class, v)),
    /** 今の順位（同点は起家に近い席が上）。 */
    RANK(v -> FieldSpec.ids(v.seats())),
    RIICHI(v -> oneOf(RiichiState.class, v)),
    IPPATSU(v -> FieldSpec.flag()),
    MENZEN(v -> FieldSpec.flag()),
    /** 第一打の前。 */
    BEFORE_FIRST_DISCARD(v -> FieldSpec.flag()),
    /** 副露の数（暗槓を含む）。 */
    MELDS(v -> FieldSpec.count(Variant.MAX_MELDS)),
    KANS(v -> FieldSpec.count(Variant.MAX_KANS)),
    /** 抜いた北の数。 */
    NUKI(v -> FieldSpec.count(Tiles.COPIES)),
    DISCARDS(v -> FieldSpec.count(v.maxDiscards())),
    /** 公開の役牌の刻子・槓子の翻の和。 */
    OPEN_YAKUHAI_HAN(v -> FieldSpec.count(Limits.MAX_OPEN_YAKUHAI_HAN)),
    /** 副露の色（1 つの色と字牌だけのとき。混ざれば 0）。 */
    OPEN_SUIT(v -> oneOf(OpenSuit.class, v)),
    /** 副露があり、全て断么と両立する。 */
    OPEN_TANYAO(v -> FieldSpec.flag()),
    /** 副露があり、全て刻子か槓子。チーの無いルールでは副露の数で決まるので持たない。 */
    OPEN_TRIPLETS(v -> FieldSpec.flag()),
    /** 副露のドラ（表ドラと赤）。 */
    OPEN_DORA(v -> FieldSpec.count(Limits.maxDora(v))),
    /** 流し満貫の形が残っている。 */
    NAGASHI(v -> FieldSpec.flag()),
    /** 自分がこの席からの直撃のロンで、この席を抜くのに要る格（自分の席は 0）。 */
    PASS_BY_DIRECT_RON(v -> oneOf(PassGrade.class, v)),
    /** 自分が他家からのロンで、この席を抜くのに要る格。 */
    PASS_BY_OTHER_RON(v -> oneOf(PassGrade.class, v)),
    /** 自分がツモで、この席を抜くのに要る格。 */
    PASS_BY_TSUMO(v -> oneOf(PassGrade.class, v)),
    /** この席が自分からの直撃のロンで、自分を抜くのに要る格。 */
    PASSED_BY_RON(v -> oneOf(PassGrade.class, v)),
    /** この席がツモで、自分を抜くのに要る格。 */
    PASSED_BY_TSUMO(v -> oneOf(PassGrade.class, v));

    private final Function<Variant, FieldSpec> spec;

    SeatField(Function<Variant, FieldSpec> spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec(Variant variant) {
      return spec.apply(variant);
    }

    @Override
    public boolean in(Variant variant) {
      return switch (this) {
        case NUKI -> variant.has(RuleFlag.NUKI);
        case OPEN_TRIPLETS -> variant.has(RuleFlag.CHI);
        default -> true;
      };
    }
  }

  /** 牌のトークン（正準化した色の順）のフィールド。 */
  public enum TileField implements InputColumn.Field, CategoryField {
    TILE(
        v -> FieldSpec.ids(TileTokens.of(v).count()),
        (r, c) -> {
          for (int token = 0; token < r.tileCount; token++) c.id(token, token);
        }),
    /** 自分の手の枚数（ツモ牌を含む）。 */
    HAND(
        v -> FieldSpec.count(Tiles.COPIES),
        (r, c) -> {
          for (int token = 0; token < r.tileCount; token++)
            c.count(token, r.tiles.hand(r.physicalOf[token]));
        }),
    /** 自分の手に赤がある。 */
    HAND_AKA(v -> FieldSpec.flag(), (r, c) -> c.flags(r, r.tiles.handRed())),
    /**
     * 今ツモった牌の種類。局のフィールドにある直前のイベント（自分のツモ）の牌と同じ事実だが、手出しとツモ切りを牌のトークンから直接
     * 読めるように重ねて置く。
     */
    DRAWN(v -> FieldSpec.flag(), (r, c) -> c.flags(r, r.drawnMask())),
    /** 公開で見える枚数（河・副露・ドラ表示）。 */
    VISIBLE(
        v -> FieldSpec.count(Tiles.COPIES),
        (r, c) -> {
          for (int token = 0; token < r.tileCount; token++)
            c.count(token, r.tiles.visible(r.physicalOf[token]));
        }),
    /** 自分から見えない枚数（4 − 公開 − 自分の手）。 */
    UNSEEN(
        v -> FieldSpec.count(Tiles.COPIES),
        (r, c) -> {
          for (int token = 0; token < r.tileCount; token++)
            c.count(token, r.tiles.unseen(r.physicalOf[token]));
        }),
    /** この牌を指すドラ表示の数。北抜きがあるルールでは北に 1 を足す。 */
    DORA(
        v -> FieldSpec.count(Tiles.COPIES + (v.has(RuleFlag.NUKI) ? 1 : 0)),
        (r, c) -> {
          for (int token = 0; token < r.tileCount; token++)
            c.count(
                token,
                r.tiles.dora(r.physicalOf[token]) + (token == r.tokens.nukiDoraToken() ? 1 : 0));
        }),
    /** この種類の赤がまだ見えていない（河・副露・ドラ表示・自分の手のどれにも出ていない）。ドラ表示の赤はこのフィールドだけが持つ。 */
    UNSEEN_AKA(v -> FieldSpec.flag(), (r, c) -> c.flags(r, r.tiles.unseenRed())),
    /** 自分の河にある。 */
    OWN_RIVER(v -> FieldSpec.flag(), (r, c) -> c.flags(r, r.safety.river(0))),
    /** 各相手に対する安全度。 */
    SAFETY(v -> oneOf(Safety.class, v).times(v.seats() - 1), (r, c) -> r.safety(c, 0)),
    /** 各席にとっての役牌の翻（0〜2）。 */
    YAKUHAI(v -> FieldSpec.count(2).times(v.seats()), RowFacts::yakuhai),
    /** 今の手の形の待ち（13 枚の判断だけ）。 */
    CURRENT_WAIT(v -> FieldSpec.flag(), (r, c) -> c.flags(r, r.currentWaits)),
    /** 今の手の受け入れ（13 枚の判断だけ）。 */
    CURRENT_UKEIRE(v -> FieldSpec.flag(), (r, c) -> c.flags(r, r.currentUkeire));

    private final InputColumn.Source source;

    @Override
    public InputColumn.Source source() {
      return source;
    }

    private final Function<Variant, FieldSpec> spec;

    TileField(Function<Variant, FieldSpec> spec, InputColumn.Source source) {
      this.source = source;
      this.spec = spec;
    }

    @Override
    public FieldSpec spec(Variant variant) {
      return spec.apply(variant);
    }
  }

  /**
   * イベントのトークン（公開の順に 0〜{@link Limits#MAX_EVENT_TOKENS} 個。バッチの 1 行に {@link InputFormat#eventCapacity} 個まで
   * 入る）のフィールド。
   */
  public enum EventField implements InputColumn.Field, CategoryField {
    KIND(
        v -> oneOf(PublicEvent.class, v),
        (r, c) -> {
          for (int p = r.fromPosition; p < r.events.publicEvents(); p++) {
            int e = r.events.entity(p);
            c.value(p, e < 0 ? PublicEvent.DORA_INDICATOR : r.events.kind(e));
          }
        }),
    /** イベントを起こした席（相対席）。 */
    AUTHOR(
        v -> FieldSpec.ids(v.seats()),
        (r, c) -> {
          for (int e = r.fromRecord; e < r.events.events(); e++)
            c.id(r.events.position(e), r.events.author(e));
        }),
    /** 打牌・副露の最小牌・ドラ表示牌。基本形式は正準化した牌、公開順の形式は物理牌。 */
    TILE(
        v -> FieldSpec.ids(TileTokens.of(v).count()),
        (r, c) -> {
          for (int p = 0; p < r.events.publicEvents(); p++) c.id(p, r.eventTile(p));
        },
        (r, c) -> r.eventTiles(c, 0)),
    /**
     * 打牌の牌が赤。ドラ表示のイベントでは 0 にし、表示牌の赤は牌のフィールドの UNSEEN_AKA が持つ。判断の記録は表示牌の赤を色ごとに
     * しか持たず、同じ色の 5 の表示牌が 2 枚あると、どちらが赤か決まらないため。
     */
    AKA(
        v -> FieldSpec.flag(),
        (r, c) -> {
          for (int e = r.fromRecord; e < r.events.events(); e++)
            c.flag(r.events.position(e), r.events.aka(e));
        }),
    TSUMOGIRI(
        v -> FieldSpec.flag(),
        (r, c) -> {
          for (int e = r.fromRecord; e < r.events.events(); e++)
            c.flag(r.events.position(e), r.events.tsumogiri(e));
        }),
    RIICHI_DECLARATION(
        v -> FieldSpec.flag(),
        (r, c) -> {
          for (int seats = r.events.declaredSeats(); seats != 0; seats &= seats - 1) {
            int position = r.declaration[Integer.numberOfTrailingZeros(seats)];
            if (position >= r.fromPosition) c.flag(position, true);
          }
        }),
    /** この打牌を鳴いた相対席。公開順の形式では、鳴いた席を副露イベントに置く。 */
    CALLED_BY(
        v -> FieldSpec.ids(v.seats()),
        (r, c) -> {
          for (int e = r.fromRecord; e < r.events.events(); e++)
            c.id(r.events.position(e), r.events.calledBy(e));
        },
        (r, c) -> {
          for (int i = r.fromMeld; i < r.events.meldEntries(); i++) {
            int e = r.events.meldEntry(i);
            switch (r.events.kind(e)) {
              case CHI, PON, DAIMINKAN -> c.id(r.events.position(e), r.events.author(e));
              default -> {}
            }
          }
        }),
    /** 副露で鳴いた牌。基本形式は正準化した牌、公開順の形式は物理牌。 */
    CALLED_TILE(
        v -> FieldSpec.ids(TileTokens.of(v).count()),
        (r, c) -> {
          for (int i = 0; i < r.events.meldEntries(); i++) {
            int e = r.events.meldEntry(i);
            Meld meld = r.events.meld(e);
            c.id(r.events.position(e), r.token(meld.called()));
          }
        },
        (r, c) -> {
          for (int i = r.fromMeld; i < r.events.meldEntries(); i++) {
            int e = r.events.meldEntry(i);
            Meld meld = r.events.meld(e);
            c.id(r.events.position(e), r.tokens.token(meld.called()));
          }
        }),
    /** 副露で鳴いた牌を出した席（相対席）。 */
    SOURCE(
        v -> FieldSpec.ids(v.seats()),
        (r, c) -> {
          for (int i = r.fromMeld; i < r.events.meldEntries(); i++) {
            int e = r.events.meldEntry(i);
            Meld meld = r.events.meld(e);
            if (meld.source() != Meld.RelativeSource.NONE)
              c.id(r.events.position(e), meld.source().seat(r.events.author(e), r.seatCount));
          }
        }),
    /** 副露の赤の由来。 */
    AKA_SOURCE(
        v -> oneOf(Meld.AkaSource.class, v),
        (r, c) -> {
          for (int i = r.fromMeld; i < r.events.meldEntries(); i++) {
            int e = r.events.meldEntry(i);
            Meld meld = r.events.meld(e);
            c.value(r.events.position(e), meld.aka());
          }
        }),
    /** 打った席の何打目か（打牌のイベントだけ）。 */
    DISCARD_INDEX(
        v -> FieldSpec.ids(v.maxDiscards()),
        (r, c) -> {
          for (int e = r.fromRecord; e < r.events.events(); e++)
            c.id(r.events.position(e), r.events.discardIndex(e));
        }),
    /** イベントを起こした席が、この時点で立直している。 */
    AUTHOR_IN_RIICHI(
        v -> FieldSpec.flag(),
        (r, c) -> {
          if (r.events.declaredSeats() == 0) return;
          for (int e = r.fromRecord; e < r.events.events(); e++) {
            int declaration = r.events.riichiDeclaration(r.events.author(e));
            c.flag(r.events.position(e), declaration >= 0 && declaration < e);
          }
        });

    private final InputColumn.Source source;
    private final InputColumn.Source publicOrder;

    @Override
    public InputColumn.Source source(InputFormat format) {
      return format.has(InputFeature.PUBLIC_ORDER_EVENTS) ? publicOrder : source;
    }

    @Override
    public InputColumn.Source source() {
      return source;
    }

    private final Function<Variant, FieldSpec> spec;

    EventField(Function<Variant, FieldSpec> spec, InputColumn.Source source) {
      this(spec, source, source);
    }

    EventField(
        Function<Variant, FieldSpec> spec,
        InputColumn.Source source,
        InputColumn.Source publicOrder) {
      this.publicOrder = publicOrder;
      this.source = source;
      this.spec = spec;
    }

    @Override
    public FieldSpec spec(Variant variant) {
      return spec.apply(variant);
    }
  }

  /** 遷移（手を打った後の 1 局面。同じ手のうち代表でないものは、代表の遷移を指す）のフィールド。 */
  public enum TransitionField implements CategoryField {
    /** 手の種類（続き・文脈・元の席は、これと局のフィールドから決まるので持たない）。 */
    ACTION(v -> oneOf(Action.Type.class, v)),
    /** 主牌（打牌・鳴いた牌・槓の牌）。 */
    MAIN_TILE(v -> FieldSpec.ids(TileTokens.of(v).count())),
    /** 赤とツモ切りの選択。 */
    TILE_SELECTION(v -> oneOf(Action.TileSelection.class, v)),
    /** 鳴いた後に捨てる牌。 */
    CALL_DISCARD(v -> FieldSpec.ids(TileTokens.of(v).count())),
    /** 鳴いた後に赤を捨てる。 */
    CALL_DISCARD_AKA(v -> FieldSpec.flag()),
    /** 遷移の後の手が門前。 */
    MENZEN(v -> FieldSpec.flag()),
    /** 遷移の後の手の最小の向聴（和了・九種九牌の遷移は 0）。 */
    SHANTEN(v -> FieldSpec.ids(Limits.MAX_SHANTEN + 1)),
    /** 遷移の後の手の通常形の向聴。 */
    STANDARD_SHANTEN(v -> FieldSpec.ids(Limits.MAX_STANDARD_SHANTEN + 1)),
    /** 遷移の後の手の七対子の向聴（副露があれば 0）。 */
    CHIITOI_SHANTEN(v -> FieldSpec.ids(Limits.MAX_CHIITOI_SHANTEN + 1)),
    /** 遷移の後の手の国士の向聴（副露があれば 0）。 */
    KOKUSHI_SHANTEN(v -> FieldSpec.ids(Limits.MAX_KOKUSHI_SHANTEN + 1)),
    /** 受け入れの種類の数。 */
    UKEIRE_KINDS(v -> FieldSpec.count(TileTokens.of(v).count())),
    /** 形の上の待ちの種類の数。 */
    WAIT_KINDS(v -> FieldSpec.count(Limits.MAX_WAIT_KINDS)),
    /** 遷移の後の振聴。 */
    FURITEN(v -> oneOf(Furiten.class, v)),
    /** 遷移の後の手と副露のドラ（表ドラと赤、裏なし）。 */
    DORA(v -> FieldSpec.count(Limits.maxDora(v))),
    /** この手で和了するときの翻（裏なし。{@link InputSchema#HAN_SATURATION} 以上はまとめ、役満は 0）。 */
    WIN_HAN(v -> FieldSpec.count(HAN_SATURATION)),
    /** この手で和了するときの符の番号（判断の記録と同じ符の番号）。 */
    WIN_FU(v -> FieldSpec.count(ObservationSchema.MAX_FU_CODE)),
    /** この手で和了するときの役満の倍率。 */
    WIN_YAKUMAN(v -> FieldSpec.count(ObservationSchema.MAX_YAKUMAN)),
    /** 立直して和了したときに開く裏ドラ表示の数。 */
    URA_INDICATORS(v -> FieldSpec.count(Limits.MAX_INDICATORS));

    private final Function<Variant, FieldSpec> spec;

    TransitionField(Function<Variant, FieldSpec> spec) {
      this.spec = spec;
    }

    @Override
    public FieldSpec spec(Variant variant) {
      return spec.apply(variant);
    }
  }

  /**
   * 関係のバイアスを作る属性（トークンごとに u8 で、埋め込まない）。関係のバイアスの 7 つのクラス（{@code RelationClass}）はここから作る。
   *
   * <p>牌と安全度は、牌・イベントのフィールドにもある。トークンの型によらず同じ列から関係のクラスを作れるように、ここにも重ねて置く。
   */
  public enum RelationAttribute implements InputColumn.Field {
    /** 牌・イベントの牌番号。基本形式は正準化した牌、公開順の形式は物理牌。 */
    TILE(
        v -> FieldSpec.ids(TileTokens.of(v).count()),
        (r, c) -> {
          for (int tile = 0; tile < r.tileCount; tile++) c.id(r.firstTile + tile, tile);
          for (int p = 0; p < r.events.publicEvents(); p++) c.id(r.firstEvent + p, r.eventTile(p));
        },
        (r, c) -> {
          for (int t = 0; t < r.tileCount; t++)
            c.id(r.firstTile + t, r.tokens.token(r.physicalOf[t]));
          r.eventTiles(c, r.firstEvent);
        }),
    /** トークンの席（相対席）。席のトークンではその席、イベントのトークンではイベントを起こした席。 */
    OWNER(
        v -> FieldSpec.ids(v.seats()),
        (r, c) -> {
          for (int seat = 0; seat < r.seatCount; seat++) c.id(r.firstSeat + seat, seat);
          for (int e = r.fromRecord; e < r.events.events(); e++)
            c.id(r.firstEvent + r.events.position(e), r.events.author(e));
        }),
    /** イベントの番号（局のトークンでは、今までのイベントの数）。 */
    EVENT(
        v -> FieldSpec.ids((v.maxPublicEvents() + Limits.MAX_INDICATORS - 1) + 1),
        (r, c) -> {
          c.id(0, r.events.publicEvents());
          for (int p = r.fromPosition; p < r.events.publicEvents(); p++) c.id(r.firstEvent + p, p);
        }),
    /** 席の立直宣言の位置。公開順の形式では、そのイベントまでに判明した位置を使う。 */
    OWNER_RIICHI_EVENT(
        v -> FieldSpec.ids((v.maxPublicEvents() + Limits.MAX_INDICATORS - 1)),
        (r, c) -> {
          if (r.events.declaredSeats() != 0) r.ownerEvents(c, r.declaration);
        },
        (r, c) -> {
          if (r.events.declaredSeats() != 0) r.prefixEvents(c, true);
        }),
    /** 席の最後の打牌の位置。公開順の形式では、そのイベントまでの最後の打牌を使う。 */
    OWNER_LAST_DISCARD_EVENT(
        v -> FieldSpec.ids((v.maxPublicEvents() + Limits.MAX_INDICATORS - 1)),
        (r, c) -> r.ownerEvents(c, r.lastDiscard),
        (r, c) -> r.prefixEvents(c, false)),
    /** つながるイベントの番号（鳴かれた打牌、加槓の元のポン、槓ドラを開いた槓）。 */
    LINK(
        v -> FieldSpec.ids((v.maxPublicEvents() + Limits.MAX_INDICATORS - 1)),
        (r, c) -> {
          if (r.events.meldEntries() == 0) return;
          for (int p = r.fromPosition; p < r.events.publicEvents(); p++) {
            int e = r.events.entity(p);
            int link = e < 0 ? r.events.kanEvent(-1 - e) : r.events.link(e);
            c.id(r.firstEvent + p, r.events.position(link));
          }
        }),
    /** 各相手に対する安全度（牌のトークン）。 */
    SAFETY(v -> oneOf(Safety.class, v).times(v.seats() - 1), (r, c) -> r.safety(c, r.firstTile));

    private final InputColumn.Source source;
    private final InputColumn.Source publicOrder;

    @Override
    public InputColumn.Source source(InputFormat format) {
      return format.has(InputFeature.PUBLIC_ORDER_EVENTS) ? publicOrder : source;
    }

    @Override
    public InputColumn.Source source() {
      return source;
    }

    private final Function<Variant, FieldSpec> spec;

    RelationAttribute(Function<Variant, FieldSpec> spec, InputColumn.Source source) {
      this(spec, source, source);
    }

    RelationAttribute(
        Function<Variant, FieldSpec> spec,
        InputColumn.Source source,
        InputColumn.Source publicOrder) {
      this.publicOrder = publicOrder;
      this.source = source;
      this.spec = spec;
    }

    @Override
    public FieldSpec spec(Variant variant) {
      return spec.apply(variant);
    }
  }

  interface RowScalar {
    void write(RowFacts row, RowInputWriter out);
  }

  /** 局の数値のフィールド。 */
  public enum RoundScalar implements ScalarField, RowScalar {
    /** 王牌を除く山の残り枚数。 */
    WALL;

    @Override
    public void write(RowFacts row, RowInputWriter out) {
      out.scalar(this, row.observation.remainingWall());
    }

    @Override
    public float scale(Variant variant) {
      return variant.liveWall();
    }
  }

  /** 席の数値のフィールド。 */
  public enum SeatScalar implements ScalarField, RowScalar {
    SCORE(100_000f, (row, seat) -> row.observation.score(seat)),
    /** 自分との点差（この席の点 − 自分の点）。 */
    GAP_TO_SELF(POINT_SCALE, (row, seat) -> row.observation.score(seat) - row.observation.score(0));

    private final float scale;

    @FunctionalInterface
    private interface SeatValue {
      int value(RowFacts row, int seat);
    }

    private final SeatValue source;

    @Override
    public void write(RowFacts row, RowInputWriter out) {
      for (int seat = 0; seat < row.seatCount; seat++) {
        out.scalar(seat, this, source.value(row, seat));
      }
    }

    SeatScalar(float scale, SeatValue source) {
      this.source = source;
      this.scale = scale;
    }

    @Override
    public float scale(Variant variant) {
      return scale;
    }
  }

  /** イベントの数値のフィールド。 */
  public enum EventScalar implements ScalarField, RowScalar {
    /** 今からの距離。公開順の形式では先頭からの位置。 */
    AGE;

    @Override
    public void write(RowFacts row, RowInputWriter out) {
      out.eventAges(row.events.publicEvents());
    }

    @Override
    public float scale(Variant variant) {
      return InputRules.of(variant).eventCapacity;
    }
  }

  /** 遷移の数値のフィールド。 */
  public enum TransitionScalar implements ScalarField {
    /** 受け入れの生きた枚数。 */
    LIVE_UKEIRE {
      @Override
      public float scale(Variant variant) {
        return TileTokens.of(variant).count();
      }
    },
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

    TransitionScalar() {
      this(1f);
    }

    TransitionScalar(float scale) {
      this.scale = scale;
    }

    @Override
    public float scale(Variant variant) {
      return scale;
    }
  }

  /** 打牌の値のヘッドへ渡す厳密なフィールド（遷移ごと、FP32）。このヘッドが線形に読み、候補層も同じ列を BF16 で読む。 */
  public enum DiscardExact implements ExactField {
    /** 聴牌なら 1。 */
    TENPAI(1f, WaitProjection.Field.TENPAI),
    /** 待ちの生きた枚数。 */
    LIVE_WAITS(LIVE_TILE_SCALE, WaitProjection.Field.LIVE_WAITS),
    /** ロン和了で半荘が終わる割合（待ちと、その牌を出す席の組ごとに、生きた枚数で重みを付ける）。 */
    RON_ENDS(1f, WaitProjection.Field.RON_ENDS),
    /** ロン和了で半荘が終わるときの効用の平均。 */
    RON_UTILITY(1f, WaitProjection.Field.RON_UTILITY),
    /** ロン和了の得点の平均。 */
    RON_POINTS(POINT_SCALE, WaitProjection.Field.RON_POINTS),
    /** ツモ和了で半荘が終わる割合。 */
    TSUMO_ENDS(1f, WaitProjection.Field.TSUMO_ENDS),
    /** ツモ和了で半荘が終わるときの効用の平均。 */
    TSUMO_UTILITY(1f, WaitProjection.Field.TSUMO_UTILITY),
    /** ツモ和了の得点の平均。 */
    TSUMO_POINTS(POINT_SCALE, WaitProjection.Field.TSUMO_POINTS);

    private final float scale;

    private final WaitProjection.Field projected;

    DiscardExact(float scale, WaitProjection.Field projected) {
      this.projected = projected;
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
      return projected;
    }
  }

  /** ロン・ツモのゲートのヘッドへ渡す厳密なフィールド（手ごと、FP32）。 */
  public enum WinExact implements ExactField, Scoped {
    /** 半荘が終わる。 */
    ENDS(1f, w -> w.ends() ? 1f : 0f),
    /** 半荘が終わるときの効用（続くなら 0）。 */
    UTILITY(1f, w -> w.utility()),
    /** 和了の後に 1 位。 */
    RANK_1(1f, w -> w.rank() == 0 ? 1f : 0f),
    /** 和了の後に 2 位。 */
    RANK_2(1f, w -> w.rank() == 1 ? 1f : 0f),
    /** 和了の後に 3 位。 */
    RANK_3(1f, w -> w.rank() == 2 ? 1f : 0f),
    /** 和了の後に 4 位。 */
    RANK_4(1f, w -> w.rank() == 3 ? 1f : 0f),
    /** 得点（裏なし）。 */
    POINTS(POINT_SCALE, w -> w.points()),
    RENCHAN(1f, w -> w.renchan() ? 1f : 0f),
    /** 包が成り立つ。 */
    PAO(1f, w -> w.pao() ? 1f : 0f),
    /** 裏ドラが乗りうる（立直している）。 */
    URA_ELIGIBLE(1f, w -> w.uraEligible() ? 1f : 0f);

    private final float scale;

    @FunctionalInterface
    private interface Source {
      float value(WinFacts facts);
    }

    private final Source source;

    float value(WinFacts facts) {
      return source.value(facts);
    }

    WinExact(float scale, Source source) {
      this.source = source;
      this.scale = scale;
    }

    @Override
    public float scale() {
      return scale;
    }

    @Override
    public boolean in(Variant variant) {
      return this != RANK_4 || variant.seats() >= 4;
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

  static final class TileKeys {
    static final CodeSpace<TileKeyCode.Digit> SPACE =
        TileKeyCode.space(WaitClass.values().length + 1);
  }

  /** 遷移のキーとコードを並べる順。 */
  public enum KeyBlock {
    ROUND,
    SEAT,
    TILE;

    int keys(Variant variant) {
      return switch (this) {
        case ROUND -> 1;
        case SEAT -> variant.seats();
        case TILE -> TileTokens.of(variant).count();
      };
    }

    int codes(Variant variant) {
      return switch (this) {
        case ROUND -> 1;
        case SEAT -> variant.seats() * 2;
        case TILE -> TileKeys.SPACE.size();
      };
    }
  }
}
