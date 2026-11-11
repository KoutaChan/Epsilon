package com.epsilon.classic.input;

import static org.testng.Assert.assertEquals;

import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicFields.ActionFloat;
import com.epsilon.classic.input.ClassicFields.ActionInt;
import com.epsilon.classic.input.ClassicFields.ActionRoute;
import com.epsilon.classic.input.ClassicFields.ActionTransitionFloat;
import com.epsilon.classic.input.ClassicFields.ActionTransitionInt;
import com.epsilon.classic.input.ClassicFields.ActionTransitionWaitFloat;
import com.epsilon.classic.input.ClassicFields.BoundaryFloat;
import com.epsilon.classic.input.ClassicFields.MeldFloat;
import com.epsilon.classic.input.ClassicFields.MeldInt;
import com.epsilon.classic.input.ClassicFields.PlayerFloat;
import com.epsilon.classic.input.ClassicFields.PlayerInt;
import com.epsilon.classic.input.ClassicFields.RiverFloat;
import com.epsilon.classic.input.ClassicFields.RiverInt;
import com.epsilon.classic.input.ClassicFields.RoundFloat;
import com.epsilon.classic.input.ClassicFields.RoundInt;
import com.epsilon.classic.input.ClassicFields.TileFloat;
import com.epsilon.classic.input.ClassicFields.TileInt;
import com.epsilon.classic.input.ClassicFields.WinFact;
import com.epsilon.classic.input.ClassicSchema.FieldLayout;
import java.util.ArrayList;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 入力のフィールドの名前・並び・幅と、テンソルの並びと 1 単位あたりの要素の数が変わらないことを確かめる。変換した重みの埋め込みの表と
 * 線形の層は、入力がこの並びであることを前提にしている。
 */
public class ClassicSchemaTest {

  private static final ClassicSchema NANO = ClassicSchema.of(Architecture.NANO);
  private static final ClassicSchema MAJOR = ClassicSchema.of(Architecture.MAJOR);

  /** model.json に書く入力のフィンガープリント。変換したチェックポイントは、この値が合うときだけ開ける。 */
  private static final String NANO_FINGERPRINT = "d73c0a95f93c02cb";

  private static final String MAJOR_FINGERPRINT = "68d5bfefcf3e09cc";

  @Test
  public void roundFieldsKeepTheirNamesOrderAndWidths() {
    String expected =
        "[PLAYER_SEAT:5, CURRENT_PLAYER_RELATIVE_SEAT:5, SOURCE_PLAYER_RELATIVE_SEAT:5, KYOKU_INDEX:256,"
            + " DEALER_RELATIVE_SEAT:5, BAKAZE:35, JIKAZE:35, HONBA:512, KYOTAKU:512, WALL_REMAINING:123,"
            + " TURN_NUMBER:128, TOTAL_KAN:5, DORA_INDICATOR_COUNT:6, ALL_LAST:2, INITIAL_DISCARD_CYCLE:2,"
            + " SELF_BEFORE_FIRST_DISCARD:2, UNINTERRUPTED_FIRST_DRAW:2, FIRST_TURN_CALL_OCCURRED:2,"
            + " LAST_LIVE_TILE:2, EVENT_TYPE:5, EVENT_KAN_KIND:4, EVENT_PLAYER_RELATIVE_SEAT:5, EVENT_TILE:35,"
            + " EVENT_IS_AKA:2, EVENT_DRAW_SOURCE:3, EVENT_RINSHAN:2, EVENT_DEFERRED_DORA:2]";
    assertBoth(RoundInt.class, expected);
    assertBoth(
        RoundFloat.class,
        "[HONBA:/10.0, KYOTAKU:/10.0, WALL_REMAINING:/70.0, TURN:/18.0, SELF_SCORE:/100000.0,"
            + " SCORE_LEAD:/100000.0, SCORE_TO_FIRST:/100000.0, SCORE_TO_FOURTH:/100000.0]");
  }

  @Test
  public void playerFieldsKeepTheirNamesOrderAndWidths() {
    assertBoth(
        PlayerInt.class,
        "[RELATIVE_SEAT:5, ABSOLUTE_SEAT:5, JIKAZE:35, RIICHI_STATUS:4, IPPATSU:2, BEFORE_FIRST_DISCARD:2,"
            + " SELF_TEMPORARY_FURITEN:2, SELF_PERMANENT_FURITEN:2, MENZEN:2, RANK:5, MELD_COUNT:5,"
            + " OPEN_MELD_COUNT:5, KAN_COUNT:5, RIVER_COUNT:25, RIICHI_DECLARATION_INDEX:25,"
            + " DISCARDS_AFTER_RIICHI:25, POST_RIICHI_TSUMOGIRI_COUNT:25]");
    assertBoth(
        PlayerFloat.class,
        "[SCORE:/100000.0, SCORE_FROM_SELF:/100000.0, RIVER_PROGRESS:/24.0, OPEN_MELD_FRACTION:/4.0,"
            + " RIICHI_DECLARATION_PROGRESS:/24.0, DEALER:/1.0]");
  }

  @Test
  public void tileFieldsKeepTheirNamesOrderAndWidths() {
    assertBoth(
        TileInt.class,
        "[TILE_TYPE:35, SELF_HAND_COUNT:5, SELF_HAS_AKA:2, VISIBLE_COUNT:5, DORA_INDICATOR_MULTIPLICITY:6,"
            + " DORA_MULTIPLICITY:6, SELF_RIVER_COUNT:5, SHIMOCHA_RIVER_COUNT:5, TOIMEN_RIVER_COUNT:5,"
            + " KAMICHA_RIVER_COUNT:5, SELF_UKEIRE:2, SELF_WAIT:2, SELF_DISCARDED_WAIT:2,"
            + " SHIMOCHA_RIICHI_GENBUTSU:2, TOIMEN_RIICHI_GENBUTSU:2, KAMICHA_RIICHI_GENBUTSU:2]");
    assertBoth(
        TileFloat.class,
        "[SELF_HAND_FRACTION:/4.0, VISIBLE_FRACTION:/4.0, UNSEEN_FRACTION:/4.0, DORA_MULTIPLICITY:/1.0]");
  }

  @Test
  public void riverAndMeldFieldsKeepTheirNamesOrderAndWidths() {
    assertBoth(
        RiverInt.class,
        "[RELATIVE_PLAYER:5, INDEX:25, TILE:35, IS_AKA:2, TSUMOGIRI:2, RIICHI:2, CALLED:2, TURN:128,"
            + " GLOBAL_SEQUENCE:97, PRESENT:2]");
    assertBoth(RiverFloat.class, "[PLAYER_PROGRESS:/24.0, GLOBAL_PROGRESS:/70.0]");
    assertBoth(
        MeldInt.class,
        "[RELATIVE_PLAYER:5, INDEX:5, TYPE:6, BASE_TILE:35, CALLED_TILE:35, SOURCE:5, AKA_SOURCE:5, SIZE:5,"
            + " CALL_AFTER_RIVER:25, KAN_AFTER_RIVER:25, PRESENT:2]");
    assertBoth(MeldFloat.class, "[PLAYER_PROGRESS:/24.0, KAN_PROGRESS:/24.0]");
  }

  @Test
  public void actionFieldsKeepTheirNamesOrderAndWidthsPerArchitecture() {
    String head =
        "[ID:373, TYPE:12, GROUP:9, PRIMARY_TILE:35, TILE_SELECTION:6, CHI_BASE:35, CHI_CALLED_POSITION:4,"
            + " DISCARD_IDENTITY:75, RESULTING_MELD_AKA_SOURCE:5, RESULTING_RIICHI_STATUS:4, WIN_CONTEXT:10,"
            + " URA_ELIGIBLE:2, ";
    String tail = "STARTS_IPPATSU:2, BREAKS_IPPATSU:2, FOLLOW_UP_KIND:5, TERMINAL:2]";
    assertEquals(describe(NANO, ActionInt.class), head + "SETTLEMENT_ASSUMPTION:3, SOLE_WIN_PROJECTED_RANK:5, " + tail);
    assertEquals(describe(MAJOR, ActionInt.class), head + tail);
    assertEquals(
        describe(NANO, ActionFloat.class),
        "[RIICHI_DECLARATION_COST:/100000.0, SELF_SCORE_AFTER_DECLARATION:/100000.0,"
            + " SCORE_TO_FIRST_AFTER_DECLARATION:/100000.0, SCORE_TO_FOURTH_AFTER_DECLARATION:/100000.0,"
            + " NORMALIZED_VISIBLE_HAN_WITHOUT_URA:/13.0, NORMALIZED_VISIBLE_FU:/110.0,"
            + " NORMALIZED_VISIBLE_BASE_POINTS:/32000.0, NORMALIZED_PAYMENT_FLOOR_SELF:/100000.0,"
            + " NORMALIZED_PAYMENT_FLOOR_SHIMOCHA:/100000.0, NORMALIZED_PAYMENT_FLOOR_TOIMEN:/100000.0,"
            + " NORMALIZED_PAYMENT_FLOOR_KAMICHA:/100000.0, NORMALIZED_URA_INDICATOR_COUNT:/5.0,"
            + " NORMALIZED_SOLE_WIN_SCORE_GAP_SELF:/100000.0, NORMALIZED_SOLE_WIN_SCORE_GAP_SHIMOCHA:/100000.0,"
            + " NORMALIZED_SOLE_WIN_SCORE_GAP_TOIMEN:/100000.0, NORMALIZED_SOLE_WIN_SCORE_GAP_KAMICHA:/100000.0]");
    assertEquals(describe(MAJOR, ActionFloat.class), "[]");
    assertBoth(
        ActionRoute.class,
        "[DISCARD_IDENTITY_REPRESENTATIVE:2, DISCARD_REPRESENTATIVE_SLOT:41, DAMA_SLOT:41, RIICHI_SLOT:41]");
    assertEquals(describe(NANO, WinFact.class), "[]");
    assertEquals(
        describe(MAJOR, WinFact.class),
        "[VALID:2, HAN_WITHOUT_URA:128, FU_CODE:12, YAKUMAN_MULTIPLIER:13, REQUIRES_PAO_CORRECTION:2]");
  }

  @Test
  public void transitionFieldsKeepTheirNamesOrderAndWidths() {
    assertBoth(
        ActionTransitionInt.class,
        "[KIND:5, DISCARD_CONTEXT:5, DISCARD_ACTION_ID:373, DISCARD_TILE:35, TILE_SELECTION:6,"
            + " RESULTING_DISCARD_FURITEN:2, RESULTING_RON_FURITEN_KIND:4, SPECIAL_HANDS_AVAILABLE:2, PRESENT:2]");
    assertBoth(
        ActionTransitionFloat.class,
        "[NORMALIZED_MIN_SHANTEN:/9.0, NORMALIZED_NORMAL_SHANTEN:/9.0, NORMALIZED_CHIITOI_SHANTEN:/9.0,"
            + " NORMALIZED_KOKUSHI_SHANTEN:/9.0, NORMALIZED_UKEIRE_COUNT:/136.0, NORMALIZED_UKEIRE_KINDS:/34.0,"
            + " NORMALIZED_SHAPE_WAIT_KINDS:/34.0, CURRENT_UKEIRE_HIT_WITHIN_1_SELF_DRAW:/1.0,"
            + " CURRENT_UKEIRE_HIT_WITHIN_2_SELF_DRAWS:/1.0, CURRENT_UKEIRE_HIT_WITHIN_3_SELF_DRAWS:/1.0,"
            + " NORMALIZED_LIVE_SHAPE_WAIT_COPIES:/136.0, NORMALIZED_INTRINSIC_RON_YAKU_WAIT_KINDS:/34.0,"
            + " NORMALIZED_INTRINSIC_RON_YAKU_WAIT_COPIES:/136.0, NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_KINDS:/34.0,"
            + " NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_COPIES:/136.0,"
            + " NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_KINDS:/34.0,"
            + " NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_COPIES:/136.0,"
            + " NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_KINDS:/34.0,"
            + " NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_COPIES:/136.0,"
            + " NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_KINDS:/34.0,"
            + " NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_COPIES:/136.0, NORMALIZED_DORA_COUNT:/20.0,"
            + " NORMALIZED_AKA_TILE_COUNT:/3.0, NORMALIZED_PRIMARY_TILE_UNSEEN_COPIES:/4.0, MENZEN:/1.0,"
            + " NORMALIZED_MELD_COUNT:/4.0, NORMALIZED_CONCEALED_TILE_COUNT:/14.0]");
    assertEquals(
        describe(NANO, ActionTransitionWaitFloat.class),
        "[RON_NORMALIZED_VISIBLE_HAN:/13.0, RON_NORMALIZED_FU:/110.0, RON_NORMALIZED_BASE_POINTS:/32000.0,"
            + " TSUMO_NORMALIZED_VISIBLE_HAN:/13.0, TSUMO_NORMALIZED_FU:/110.0,"
            + " TSUMO_NORMALIZED_BASE_POINTS:/32000.0]");
    assertEquals(describe(MAJOR, ActionTransitionWaitFloat.class), "[]");
    assertBoth(
        BoundaryFloat.class,
        "[GRP_RANK:/1.0x16, KYOKU_INDEX:/8.0, HONBA:/8.0, KYOTAKU:/4.0, SCORE:/10000.0x4, PRESENT:/1.0]");
  }

  @Test
  public void tensorsKeepTheirOrderAndUnitWidths() {
    assertEquals(
        tensors(NANO),
        List.of(
            "STATE_CATEGORIES:1775",
            "ACTION_CATEGORIES:18",
            "ACTION_ROUTES:4",
            "TRANSITION_CATEGORIES:9",
            "TRANSITION_TILES:34",
            "WAIT_TILE_IDS:13",
            "WAIT_YAKUS:78",
            "STATE_NUMERICS:392",
            "BOUNDARY_CONTEXT:24",
            "ACTION_NUMERICS:16",
            "TRANSITION_NUMERICS:27",
            "WAIT_SCORES:78"));
    assertEquals(
        tensors(MAJOR),
        List.of(
            "STATE_CATEGORIES:2671",
            "ACTION_CATEGORIES:16",
            "ACTION_ROUTES:4",
            "POINT_LEDGER_100:4",
            "ACTION_WIN_FACTS:5",
            "TRANSITION_CATEGORIES:9",
            "TRANSITION_TILES:34",
            "WAIT_TILE_IDS:13",
            "WAIT_WIN_FACTS:130",
            "STATE_NUMERICS:392",
            "BOUNDARY_CONTEXT:24",
            "TRANSITION_NUMERICS:27"));
    assertEquals(MAJOR.historyStart(), 1775);
    assertEquals(NANO.historyStart(), -1);
  }

  /** 埋め込みの表での開始位置は、フィールドの幅を宣言の順に足し上げた値。同じフィールドは、トークンが違っても同じ開始位置を使う。 */
  @Test
  public void embeddingNamespacesAccumulateFieldWidths() {
    assertEquals(NANO.stateDictionary(), 2425);
    assertEquals(MAJOR.stateDictionary(), 2425);
    assertEquals(NANO.actionDictionary(), 1023);
    assertEquals(MAJOR.actionDictionary(), 1015);
    int[] state = NANO.stateOffsets();
    int playerStart = NANO.stateStart(ClassicSchema.StateToken.PLAYER, Tensor.Slab.CATEGORICAL);
    int tileStart = NANO.stateStart(ClassicSchema.StateToken.TILE, Tensor.Slab.CATEGORICAL);
    assertEquals(state[0], 0);
    assertEquals(state[1], RoundInt.PLAYER_SEAT.width());
    assertEquals(state[playerStart], 1702);
    assertEquals(state[playerStart + PlayerInt.values().length], state[playerStart]);
    assertEquals(state[tileStart], 1702 + 179);
    int[] action = MAJOR.actionOffsets();
    int[] transition = MAJOR.transitionOffsets();
    assertEquals(action[0], 0);
    assertEquals(transition[0], 1015 - 434);
    assertEquals(action.length, MAJOR.width(Tensor.ACTION_CATEGORIES));
  }

  @Test
  public void rowWidthsFollowTheBucket() {
    Bucket bucket = Buckets.required(10, 12);
    assertEquals(NANO.rowWidth(Tensor.STATE_CATEGORIES, bucket), 1775);
    assertEquals(NANO.rowWidth(Tensor.ACTION_CATEGORIES, bucket), 10 * 18);
    assertEquals(NANO.rowWidth(Tensor.TRANSITION_TILES, bucket), 10 * 12 * 34);
    assertEquals(MAJOR.rowWidth(Tensor.ACTION_NUMERICS, bucket), 0);
    ClassicInputs inputs = new ClassicInputs(MAJOR, bucket, 3);
    assertEquals(inputs.start(Tensor.STATE_CATEGORIES, 2), 2 * 2671);
    assertEquals(inputs.start(Tensor.ACTION_CATEGORIES, 0), 3 * 2671);
  }

  /**
   * フィンガープリントは、フィールド・語彙・区分・牌のコード・役のブロック・公開の履歴の記述から作る。どれかが変われば、変換した
   * チェックポイントは開けなくなる。
   */
  @Test
  public void fingerprintsAreTheReaderContractOfModelJson() {
    assertEquals(NANO.fingerprint(), NANO_FINGERPRINT);
    assertEquals(MAJOR.fingerprint(), MAJOR_FINGERPRINT);
    assertEquals(NANO.fingerprint(), NANO.describe().hex());
    assertEquals(MAJOR.fingerprint(), MAJOR.describe().hex());
  }

  private static <E extends Enum<E> & InputField> void assertBoth(Class<E> type, String expected) {
    assertEquals(describe(NANO, type), expected, type.getSimpleName());
    assertEquals(describe(MAJOR, type), expected, type.getSimpleName());
  }

  private static <E extends Enum<E> & InputField> String describe(ClassicSchema schema, Class<E> type) {
    FieldLayout<E> layout = schema.layout(type);
    assertEquals(layout.stride(), layout.fields().stream().mapToInt(InputField::count).sum());
    return layout.describe();
  }

  private static List<String> tensors(ClassicSchema schema) {
    List<String> tensors = new ArrayList<>();
    for (Tensor tensor : Tensor.values()) {
      if (tensor.in(schema.architecture())) {
        tensors.add(tensor + ":" + schema.width(tensor));
      }
    }
    return tensors;
  }
}
