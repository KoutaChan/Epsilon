package com.epsilon.pico.ai.decision.input;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionEffect;
import com.epsilon.engine.WinSettlementProjection;
import java.util.Arrays;
import java.util.function.ToIntFunction;

/**
 * カテゴリ特徴量の値域と、共有する埋め込みテーブルの位置を定義する。
 *
 * <p>各項目が使うカテゴリ数の累積和で開始位置を決める。同じ項目を持つプレイヤーや牌などは埋め込みを共有し、要素の位置によらず同じ意味の値を同じパラメータで表す。入力を書き込む際に値域を検証し、隣の項目の埋め込み領域への混入を防ぐ。
 */
public final class DecisionCategoryLayout {

  private static final int[] ROUND_OFFSETS = new int[DecisionInputSchema.ROUND_INT_COUNT];
  private static final int[] PLAYER_OFFSETS = new int[DecisionInputSchema.PLAYER_INT_STRIDE];
  private static final int[] TILE_OFFSETS = new int[DecisionInputSchema.TILE_INT_STRIDE];
  private static final int[] RIVER_OFFSETS = new int[DecisionInputSchema.RIVER_INT_STRIDE];
  private static final int[] MELD_OFFSETS = new int[DecisionInputSchema.MELD_INT_STRIDE];
  private static final int[] ACTION_OFFSETS = new int[DecisionInputSchema.ACTION_INT_STRIDE];
  private static final int[] TRANSITION_OFFSETS =
      new int[DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE];
  private static final int[] ROUND_CARDINALITIES =
      cardinalities(DecisionInputSchema.RoundInt.values(), DecisionCategoryLayout::cardinality);
  private static final int[] PLAYER_CARDINALITIES =
      cardinalities(DecisionInputSchema.PlayerInt.values(), DecisionCategoryLayout::cardinality);
  private static final int[] TILE_CARDINALITIES =
      cardinalities(DecisionInputSchema.TileInt.values(), DecisionCategoryLayout::cardinality);
  private static final int[] RIVER_CARDINALITIES =
      cardinalities(DecisionInputSchema.RiverInt.values(), DecisionCategoryLayout::cardinality);
  private static final int[] MELD_CARDINALITIES =
      cardinalities(DecisionInputSchema.MeldInt.values(), DecisionCategoryLayout::cardinality);
  private static final int[] ACTION_CARDINALITIES =
      cardinalities(DecisionInputSchema.ActionInt.values(), DecisionCategoryLayout::cardinality);
  private static final int[] TRANSITION_CARDINALITIES =
      cardinalities(
          DecisionInputSchema.ActionTransitionInt.values(), DecisionCategoryLayout::cardinality);

  /** 局・プレイヤー・牌・河・面子フィールドが共有する状態埋め込み辞書の総要素数。 */
  public static final int STATE_DICTIONARY_SIZE;

  /** 判断対象の行動・遷移フィールドが共有する候補埋め込み辞書の総要素数。 */
  public static final int ACTION_DICTIONARY_SIZE;

  static {
    int stateCursor = 0;
    stateCursor = fill(ROUND_OFFSETS, ROUND_CARDINALITIES, stateCursor);
    stateCursor = fill(PLAYER_OFFSETS, PLAYER_CARDINALITIES, stateCursor);
    stateCursor = fill(TILE_OFFSETS, TILE_CARDINALITIES, stateCursor);
    stateCursor = fill(RIVER_OFFSETS, RIVER_CARDINALITIES, stateCursor);
    stateCursor = fill(MELD_OFFSETS, MELD_CARDINALITIES, stateCursor);
    STATE_DICTIONARY_SIZE = stateCursor;

    int actionCursor = 0;
    actionCursor = fill(ACTION_OFFSETS, ACTION_CARDINALITIES, actionCursor);
    actionCursor = fill(TRANSITION_OFFSETS, TRANSITION_CARDINALITIES, actionCursor);
    ACTION_DICTIONARY_SIZE = actionCursor;
  }

  private DecisionCategoryLayout() {}

  /**
   * flattened 状態テンソルの各位置に加えるフィールド名前空間オフセットを返す。
   *
   * @return 呼び出し側から変更できる新しいオフセット配列
   */
  public static int[] stateOffsets() {
    int[] result = new int[DecisionInputSchema.STATE_INT_COUNT];
    int cursor = 0;
    cursor = appendRepeated(result, cursor, ROUND_OFFSETS, 1);
    cursor = appendRepeated(result, cursor, PLAYER_OFFSETS, GameState.NUM_PLAYERS);
    cursor = appendRepeated(result, cursor, TILE_OFFSETS, Tile.NUM_TILE_TYPES);
    cursor =
        appendRepeated(
            result,
            cursor,
            RIVER_OFFSETS,
            GameState.NUM_PLAYERS * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER);
    cursor =
        appendRepeated(
            result,
            cursor,
            MELD_OFFSETS,
            GameState.NUM_PLAYERS * DecisionInputSchema.MAX_MELDS_PER_PLAYER);
    if (cursor != result.length) {
      throw new AssertionError("state category layout mismatch");
    }
    return result;
  }

  /**
   * 判断対象の行動テンソルへ加えるフィールド名前空間オフセットを返す。
   *
   * @return 行動フィールド順の共有読み取り専用のオフセット配列
   */
  public static int[] actionOffsets() {
    return ACTION_OFFSETS;
  }

  /**
   * 行動遷移テンソルへ加えるフィールド名前空間オフセットを返す。
   *
   * @return 遷移フィールド順の共有読み取り専用のオフセット配列
   */
  public static int[] transitionOffsets() {
    return TRANSITION_OFFSETS;
  }

  static void requireValid(DecisionInputSchema.RoundInt field, int value) {
    requireRange(field, value, ROUND_CARDINALITIES[field.ordinal()]);
  }

  static void requireValid(DecisionInputSchema.PlayerInt field, int value) {
    requireRange(field, value, PLAYER_CARDINALITIES[field.ordinal()]);
  }

  static void requireValid(DecisionInputSchema.TileInt field, int value) {
    requireRange(field, value, TILE_CARDINALITIES[field.ordinal()]);
  }

  static void requireValid(DecisionInputSchema.RiverInt field, int value) {
    requireRange(field, value, RIVER_CARDINALITIES[field.ordinal()]);
  }

  static void requireValid(DecisionInputSchema.MeldInt field, int value) {
    requireRange(field, value, MELD_CARDINALITIES[field.ordinal()]);
  }

  static void requireValid(DecisionInputSchema.ActionInt field, int value) {
    requireRange(field, value, ACTION_CARDINALITIES[field.ordinal()]);
  }

  static void requireValid(DecisionInputSchema.ActionTransitionInt field, int value) {
    requireRange(field, value, TRANSITION_CARDINALITIES[field.ordinal()]);
  }

  /**
   * 互換性識別子へ含める、フィールド順とカテゴリ数の標準形式の記述を返す。
   *
   * @return スキーマ順序と各値域を安定表現した文字列
   */
  public static String descriptor() {
    return "round="
        + Arrays.toString(ROUND_CARDINALITIES)
        + ";player="
        + Arrays.toString(PLAYER_CARDINALITIES)
        + ";tile="
        + Arrays.toString(TILE_CARDINALITIES)
        + ";river="
        + Arrays.toString(RIVER_CARDINALITIES)
        + ";meld="
        + Arrays.toString(MELD_CARDINALITIES)
        + ";action="
        + Arrays.toString(ACTION_CARDINALITIES)
        + ";transition="
        + Arrays.toString(TRANSITION_CARDINALITIES);
  }

  private static int cardinality(DecisionInputSchema.RoundInt field) {
    return switch (field) {
      case PLAYER_SEAT,
          CURRENT_PLAYER_RELATIVE_SEAT,
          SOURCE_PLAYER_RELATIVE_SEAT,
          DEALER_RELATIVE_SEAT,
          EVENT_PLAYER_RELATIVE_SEAT ->
          GameState.NUM_PLAYERS + 1;
      case KYOKU_INDEX -> 256;
      case BAKAZE, JIKAZE, EVENT_TILE -> Tile.NUM_TILE_TYPES + 1;
      case HONBA, KYOTAKU -> 512;
      case WALL_REMAINING -> GameState.LIVE_WALL_SIZE + 1;
      case TURN_NUMBER -> 128;
      case TOTAL_KAN -> 5;
      case DORA_INDICATOR_COUNT -> 6;
      case ALL_LAST,
          INITIAL_DISCARD_CYCLE,
          SELF_BEFORE_FIRST_DISCARD,
          UNINTERRUPTED_FIRST_DRAW,
          FIRST_TURN_CALL_OCCURRED,
          LAST_LIVE_TILE,
          EVENT_IS_AKA,
          EVENT_RINSHAN,
          EVENT_DEFERRED_DORA ->
          2;
      case EVENT_TYPE -> DecisionInputSchema.EventType.values().length + 1;
      case EVENT_KAN_KIND -> TurnEvent.KanKind.values().length + 1;
      case EVENT_DRAW_SOURCE -> TurnEvent.DrawSource.values().length + 1;
    };
  }

  private static int cardinality(DecisionInputSchema.PlayerInt field) {
    return switch (field) {
      case RELATIVE_SEAT, ABSOLUTE_SEAT, RANK -> GameState.NUM_PLAYERS + 1;
      case JIKAZE -> Tile.NUM_TILE_TYPES + 1;
      case RIICHI_STATUS -> RiichiState.values().length + 1;
      case IPPATSU, BEFORE_FIRST_DISCARD, SELF_TEMPORARY_FURITEN, SELF_PERMANENT_FURITEN, MENZEN ->
          2;
      case MELD_COUNT, OPEN_MELD_COUNT, KAN_COUNT -> DecisionInputSchema.MAX_MELDS_PER_PLAYER + 1;
      case RIVER_COUNT,
          RIICHI_DECLARATION_INDEX,
          DISCARDS_AFTER_RIICHI,
          POST_RIICHI_TSUMOGIRI_COUNT ->
          DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + 1;
    };
  }

  private static int cardinality(DecisionInputSchema.TileInt field) {
    return switch (field) {
      case TILE_TYPE -> Tile.NUM_TILE_TYPES + 1;
      case SELF_HAND_COUNT,
          VISIBLE_COUNT,
          SELF_RIVER_COUNT,
          SHIMOCHA_RIVER_COUNT,
          TOIMEN_RIVER_COUNT,
          KAMICHA_RIVER_COUNT ->
          Tile.TILES_PER_TYPE + 1;
      case DORA_INDICATOR_MULTIPLICITY, DORA_MULTIPLICITY -> 6;
      case SELF_HAS_AKA,
          SELF_UKEIRE,
          SELF_WAIT,
          SELF_DISCARDED_WAIT,
          SHIMOCHA_RIICHI_GENBUTSU,
          TOIMEN_RIICHI_GENBUTSU,
          KAMICHA_RIICHI_GENBUTSU ->
          2;
    };
  }

  private static int cardinality(DecisionInputSchema.RiverInt field) {
    return switch (field) {
      case RELATIVE_PLAYER -> GameState.NUM_PLAYERS + 1;
      case INDEX -> DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + 1;
      case TILE -> Tile.NUM_TILE_TYPES + 1;
      case IS_AKA, TSUMOGIRI, RIICHI, CALLED, PRESENT -> 2;
      case TURN -> 128;
      case GLOBAL_SEQUENCE ->
          GameState.NUM_PLAYERS * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + 1;
    };
  }

  private static int cardinality(DecisionInputSchema.MeldInt field) {
    return switch (field) {
      case RELATIVE_PLAYER -> GameState.NUM_PLAYERS + 1;
      case INDEX -> DecisionInputSchema.MAX_MELDS_PER_PLAYER + 1;
      case TYPE -> Meld.Type.values().length + 1;
      case BASE_TILE, CALLED_TILE -> Tile.NUM_TILE_TYPES + 1;
      case SOURCE -> Meld.RelativeSource.values().length + 1;
      case AKA_SOURCE -> Meld.AkaSource.values().length + 1;
      case SIZE -> 5;
      case CALL_AFTER_RIVER, KAN_AFTER_RIVER -> DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + 1;
      case PRESENT -> 2;
    };
  }

  private static int cardinality(DecisionInputSchema.ActionInt field) {
    return switch (field) {
      case ID -> Action.ACTION_SPACE_SIZE + 1;
      case TYPE -> Action.Type.values().length + 1;
      case GROUP -> Action.Group.values().length + 1;
      case PRIMARY_TILE, CHI_BASE -> Tile.NUM_TILE_TYPES + 1;
      case TILE_SELECTION -> Action.TileSelection.values().length + 1;
      case CHI_CALLED_POSITION -> 4;
      case DISCARD_IDENTITY -> Action.DISCARD_IDENTITY_COUNT + 1;
      case RESULTING_MELD_AKA_SOURCE -> Meld.AkaSource.values().length + 1;
      case RESULTING_RIICHI_STATUS -> RiichiState.values().length + 1;
      case WIN_CONTEXT -> DecisionInputSchema.WinConditions.values().length + 1;
      case URA_ELIGIBLE, STARTS_IPPATSU, BREAKS_IPPATSU, TERMINAL -> 2;
      case SETTLEMENT_ASSUMPTION ->
          WinSettlementProjection.SettlementAssumption.values().length + 1;
      case SOLE_WIN_PROJECTED_RANK -> GameState.NUM_PLAYERS + 1;
      case FOLLOW_UP_KIND -> ActionEffect.NextStep.values().length + 1;
    };
  }

  private static int cardinality(DecisionInputSchema.ActionTransitionInt field) {
    return switch (field) {
      case KIND -> DecisionInputSchema.ActionTransitionKind.values().length + 1;
      case DISCARD_CONTEXT -> DecisionInputSchema.DiscardContext.values().length + 1;
      case DISCARD_ACTION_ID -> Action.ACTION_SPACE_SIZE + 1;
      case DISCARD_TILE -> Tile.NUM_TILE_TYPES + 1;
      case TILE_SELECTION -> Action.TileSelection.values().length + 1;
      case RESULTING_DISCARD_FURITEN, PRESENT -> 2;
      case RESULTING_RON_FURITEN_KIND -> DecisionInputSchema.RonFuritenKind.values().length + 1;
      case SPECIAL_HANDS_AVAILABLE -> 2;
    };
  }

  private static int fill(int[] offsets, int[] cardinalities, int cursor) {
    for (int field = 0; field < offsets.length; field++) {
      offsets[field] = cursor;
      cursor = Math.addExact(cursor, cardinalities[field]);
    }
    return cursor;
  }

  private static int appendRepeated(
      int[] destination, int cursor, int[] fieldOffsets, int entityCount) {
    for (int entity = 0; entity < entityCount; entity++) {
      System.arraycopy(fieldOffsets, 0, destination, cursor, fieldOffsets.length);
      cursor += fieldOffsets.length;
    }
    return cursor;
  }

  private static void requireRange(Enum<?> field, int value, int cardinality) {
    if (value < 0 || value >= cardinality) {
      throw new IllegalArgumentException(
          field.getDeclaringClass().getSimpleName()
              + '.'
              + field.name()
              + " category out of range: "
              + value
              + " / "
              + cardinality);
    }
  }

  private static <E extends Enum<E>> int[] cardinalities(E[] fields, ToIntFunction<E> cardinality) {
    int[] values = new int[fields.length];
    for (E field : fields) {
      values[field.ordinal()] = cardinality.applyAsInt(field);
    }
    return values;
  }
}
