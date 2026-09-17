package com.epsilon.pico.ai.decision.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;

/**
 * Decision 入力の1行へ、意味のある特徴量の項目ごとに書き込む。
 *
 * <p>呼び出し側に連続バッファの配置や物理オフセットを公開しない。エンコーダーで検証済みの値を受け取り、繰り返し実行する書き込み処理では再検証しない。
 */
public final class DecisionInputWriter {

  private final short[] categoricalSlab;
  private final float[] numericSlab;
  private final short[] transitionCategories;
  private final float[] transitionNumerics;
  private final int[] transitionOffsets;
  private final int transitionRowBase;
  private final int stateCategoryBase;
  private final int stateNumericBase;
  private final int boundaryContextBase;
  private final int actionCategoryBase;
  private final int actionNumericBase;
  private final int actionRouteBase;
  private final int transitionCategoryBase;
  private final int transitionNumericBase;
  private final int transitionTileBase;
  private final int waitTileBase;
  private final int waitYakuBase;
  private final int waitScoreBase;
  private final int transitionsPerAction;

  DecisionInputWriter(DecisionHostInputs inputs, int row) {
    categoricalSlab = inputs.stateCategories();
    numericSlab = inputs.stateNumerics();
    transitionCategories = inputs.transitionCategories(row);
    transitionNumerics = inputs.transitionNumerics(row);
    transitionOffsets = inputs.transitionOffsets();
    transitionRowBase = inputs.transitionRowBase(row);
    DecisionInputLayout layout = inputs.layout();
    stateCategoryBase = rowBase(layout, DecisionInputLayout.Tensor.STATE_CATEGORIES, row);
    stateNumericBase = rowBase(layout, DecisionInputLayout.Tensor.STATE_NUMERICS, row);
    boundaryContextBase = rowBase(layout, DecisionInputLayout.Tensor.BOUNDARY_CONTEXT, row);
    actionCategoryBase = rowBase(layout, DecisionInputLayout.Tensor.ACTION_CATEGORIES, row);
    actionNumericBase = rowBase(layout, DecisionInputLayout.Tensor.ACTION_NUMERICS, row);
    actionRouteBase = rowBase(layout, DecisionInputLayout.Tensor.ACTION_ROUTES, row);
    transitionCategoryBase =
        inputs.transitionRegionBase(row, DecisionInputLayout.Tensor.TRANSITION_CATEGORIES);
    transitionNumericBase =
        inputs.transitionRegionBase(row, DecisionInputLayout.Tensor.TRANSITION_NUMERICS);
    transitionTileBase =
        inputs.transitionRegionBase(row, DecisionInputLayout.Tensor.TRANSITION_TILES);
    waitTileBase = inputs.transitionRegionBase(row, DecisionInputLayout.Tensor.WAIT_TILE_IDS);
    waitYakuBase = inputs.transitionRegionBase(row, DecisionInputLayout.Tensor.WAIT_YAKUS);
    waitScoreBase = inputs.transitionRegionBase(row, DecisionInputLayout.Tensor.WAIT_SCORES);
    transitionsPerAction = inputs.bucket().actionTransitionCapacity();
  }

  /**
   * 局-単位カテゴリ値フィールドを書き込む。
   *
   * @param field 書き込むフィールド
   * @param value フィールド固有カテゴリ数内のカテゴリ ID
   */
  public void round(DecisionInputSchema.RoundInt field, int value) {
    categoricalSlab[stateCategoryBase + field.ordinal()] = (short) value;
  }

  /**
   * 局-単位数値フィールドを書き込む。
   *
   * @param field 書き込むフィールド
   * @param value 有限な数値
   */
  public void round(DecisionInputSchema.RoundFloat field, float value) {
    numericSlab[stateNumericBase + field.ordinal()] = value;
  }

  /** 価値専用の局境界コンテキストを標準形式のテンソルへ書き込む。 */
  public void boundaryContext(DecisionBoundaryContext context) {
    context.copyTo(numericSlab, boundaryContextBase);
  }

  /**
   * プレイヤートークンのカテゴリ値フィールドを書き込む。
   *
   * @param relativeSeat 自家基準の相対席
   * @param field 書き込むフィールド
   * @param value フィールド固有カテゴリ数内のカテゴリ ID
   */
  public void player(int relativeSeat, DecisionInputSchema.PlayerInt field, int value) {
    categoricalSlab[
            stateCategoryBase
                + DecisionInputSchema.ROUND_INT_COUNT
                + relativeSeat * DecisionInputSchema.PLAYER_INT_STRIDE
                + field.ordinal()] =
        (short) value;
  }

  /**
   * プレイヤートークンの数値フィールドを書き込む。
   *
   * @param relativeSeat 自家基準の相対席
   * @param field 書き込むフィールド
   * @param value 有限な数値
   */
  public void player(int relativeSeat, DecisionInputSchema.PlayerFloat field, float value) {
    numericSlab[
            stateNumericBase
                + DecisionInputSchema.ROUND_FLOAT_COUNT
                + relativeSeat * DecisionInputSchema.PLAYER_FLOAT_STRIDE
                + field.ordinal()] =
        value;
  }

  /**
   * 牌トークンのカテゴリ値フィールドを書き込む。
   *
   * @param tileType 34牌種インデックス
   * @param field 書き込むフィールド
   * @param value フィールド固有カテゴリ数内のカテゴリ ID
   */
  public void tile(int tileType, DecisionInputSchema.TileInt field, int value) {
    categoricalSlab[
            stateCategoryBase
                + DecisionInputSchema.ROUND_INT_COUNT
                + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_INT_STRIDE
                + tileType * DecisionInputSchema.TILE_INT_STRIDE
                + field.ordinal()] =
        (short) value;
  }

  /**
   * 牌トークンの数値フィールドを書き込む。
   *
   * @param tileType 34牌種インデックス
   * @param field 書き込むフィールド
   * @param value 有限な数値
   */
  public void tile(int tileType, DecisionInputSchema.TileFloat field, float value) {
    numericSlab[
            stateNumericBase
                + DecisionInputSchema.ROUND_FLOAT_COUNT
                + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_FLOAT_STRIDE
                + tileType * DecisionInputSchema.TILE_FLOAT_STRIDE
                + field.ordinal()] =
        value;
  }

  /**
   * 河イベントトークンのカテゴリ値フィールドを書き込む。
   *
   * @param relativeSeat 河所有者の相対席
   * @param riverIndex その席の河内インデックス
   * @param field 書き込むフィールド
   * @param value フィールド固有カテゴリ数内のカテゴリ ID
   */
  public void river(
      int relativeSeat, int riverIndex, DecisionInputSchema.RiverInt field, int value) {
    categoricalSlab[
            stateCategoryBase
                + DecisionInputSchema.ROUND_INT_COUNT
                + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_INT_STRIDE
                + Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_INT_STRIDE
                + (relativeSeat * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + riverIndex)
                    * DecisionInputSchema.RIVER_INT_STRIDE
                + field.ordinal()] =
        (short) value;
  }

  /**
   * 河イベントトークンの数値フィールドを書き込む。
   *
   * @param relativeSeat 河所有者の相対席
   * @param riverIndex その席の河内インデックス
   * @param field 書き込むフィールド
   * @param value 有限な数値
   */
  public void river(
      int relativeSeat, int riverIndex, DecisionInputSchema.RiverFloat field, float value) {
    numericSlab[
            stateNumericBase
                + DecisionInputSchema.ROUND_FLOAT_COUNT
                + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_FLOAT_STRIDE
                + Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_FLOAT_STRIDE
                + (relativeSeat * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + riverIndex)
                    * DecisionInputSchema.RIVER_FLOAT_STRIDE
                + field.ordinal()] =
        value;
  }

  /**
   * 面子トークンのカテゴリ値フィールドを書き込む。
   *
   * @param relativeSeat 面子所有者の相対席
   * @param meldIndex その席の面子列内インデックス
   * @param field 書き込むフィールド
   * @param value フィールド固有カテゴリ数内のカテゴリ ID
   */
  public void meld(int relativeSeat, int meldIndex, DecisionInputSchema.MeldInt field, int value) {
    categoricalSlab[
            stateCategoryBase
                + DecisionInputSchema.ROUND_INT_COUNT
                + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_INT_STRIDE
                + Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_INT_STRIDE
                + GameState.NUM_PLAYERS
                    * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER
                    * DecisionInputSchema.RIVER_INT_STRIDE
                + (relativeSeat * DecisionInputSchema.MAX_MELDS_PER_PLAYER + meldIndex)
                    * DecisionInputSchema.MELD_INT_STRIDE
                + field.ordinal()] =
        (short) value;
  }

  /**
   * 面子トークンの数値フィールドを書き込む。
   *
   * @param relativeSeat 面子所有者の相対席
   * @param meldIndex その席の面子列内インデックス
   * @param field 書き込むフィールド
   * @param value 有限な数値
   */
  public void meld(
      int relativeSeat, int meldIndex, DecisionInputSchema.MeldFloat field, float value) {
    numericSlab[
            stateNumericBase
                + DecisionInputSchema.ROUND_FLOAT_COUNT
                + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_FLOAT_STRIDE
                + Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_FLOAT_STRIDE
                + GameState.NUM_PLAYERS
                    * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER
                    * DecisionInputSchema.RIVER_FLOAT_STRIDE
                + (relativeSeat * DecisionInputSchema.MAX_MELDS_PER_PLAYER + meldIndex)
                    * DecisionInputSchema.MELD_FLOAT_STRIDE
                + field.ordinal()] =
        value;
  }

  /**
   * 判断対象の行動候補のカテゴリ値フィールドを書き込む。
   *
   * @param actionSlot 容量区分内の行動候補の位置
   * @param field 書き込むフィールド
   * @param value フィールド固有カテゴリ数内のカテゴリ ID
   */
  public void action(int actionSlot, DecisionInputSchema.ActionInt field, int value) {
    categoricalSlab[
            actionCategoryBase
                + actionSlot * DecisionInputSchema.ACTION_INT_STRIDE
                + field.ordinal()] =
        (short) value;
  }

  /**
   * 判断対象の行動候補の数値フィールドを書き込む。
   *
   * @param actionSlot 容量区分内の行動候補の位置
   * @param field 書き込むフィールド
   * @param value 有限な数値
   */
  public void action(int actionSlot, DecisionInputSchema.ActionFloat field, float value) {
    numericSlab[
            actionNumericBase
                + actionSlot * DecisionInputSchema.ACTION_FLOAT_STRIDE
                + field.ordinal()] =
        value;
  }

  /**
   * 方策グラフ用の候補の参照先メタデータを書き込む。
   *
   * @param actionSlot 経路を所有する行動候補の位置
   * @param field 経路種別
   * @param storedSlot 0は該当なし、それ以外は参照先行動候補の位置 + 1
   */
  public void actionRoute(int actionSlot, DecisionInputSchema.ActionRoute field, int storedSlot) {
    categoricalSlab[
            actionRouteBase
                + actionSlot * DecisionInputSchema.ACTION_ROUTE_STRIDE
                + field.ordinal()] =
        (short) storedSlot;
  }

  /**
   * 行動遷移のカテゴリ値フィールドを書き込む。
   *
   * @param actionSlot 親判断対象の行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param field 書き込むフィールド
   * @param value フィールド固有カテゴリ数内のカテゴリ ID
   */
  public void transition(
      int actionSlot,
      int transitionSlot,
      DecisionInputSchema.ActionTransitionInt field,
      int value) {
    transitionCategories[
            transitionCategoryBase
                + flatTransition(actionSlot, transitionSlot)
                    * DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE
                + field.ordinal()] =
        (short) value;
  }

  /**
   * 行動遷移の数値フィールドを書き込む。
   *
   * @param actionSlot 親判断対象の行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param field 書き込むフィールド
   * @param value 有限な数値
   */
  public void transition(
      int actionSlot,
      int transitionSlot,
      DecisionInputSchema.ActionTransitionFloat field,
      float value) {
    transitionNumerics[
            transitionNumericBase
                + flatTransition(actionSlot, transitionSlot)
                    * DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE
                + field.ordinal()] =
        value;
  }

  /**
   * 遷移行動直後の状態の一牌種分を詰めたカテゴリとして書き込む。
   *
   * @param actionSlot 親判断対象の行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param tileType 34牌種インデックス
   * @param storedCategory {@link DecisionFeatureCodec#transitionTile}の出力
   */
  public void transitionTile(int actionSlot, int transitionSlot, int tileType, int storedCategory) {
    transitionCategories[
            transitionTileBase
                + flatTransition(actionSlot, transitionSlot)
                    * DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT
                + tileType] =
        (short) storedCategory;
  }

  /**
   * 34牌種分の遷移カテゴリを標準形式の牌順で一括書き込みする。
   *
   * <p>入力エンコーダー内部の頻繁に実行する処理用APIであり、カテゴリは{@link DecisionFeatureCodec#transitionTile}
   * で生成済みであることを前提とする。
   */
  void transitionTiles(int actionSlot, int transitionSlot, short[] storedCategories) {
    System.arraycopy(
        storedCategories,
        0,
        transitionCategories,
        transitionTileBase
            + flatTransition(actionSlot, transitionSlot)
                * DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT,
        Tile.NUM_TILE_TYPES);
  }

  /**
   * 待ちトークン位置へ牌種ID + 1を書き込む。
   *
   * @param actionSlot 親判断対象の行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param waitSlot 待ち集合内の位置
   * @param storedTileType 0はパディング、それ以外は牌種インデックス + 1
   */
  public void waitTile(int actionSlot, int transitionSlot, int waitSlot, int storedTileType) {
    transitionCategories[
            waitTileBase
                + flatTransition(actionSlot, transitionSlot)
                    * DecisionInputSchema.MAX_WAIT_TILE_TYPES
                + waitSlot] =
        (short) storedTileType;
  }

  /**
   * 待ち牌のRON・TSUMO別役マスクまとまりを書き込む。
   *
   * @param actionSlot 親判断対象の行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param waitSlot 待ち集合内の位置
   * @param feature RON・TSUMOを通したまとまりインデックス
   * @param storedCategory パディングを分離した詰めたカテゴリ ID
   */
  public void waitYaku(
      int actionSlot, int transitionSlot, int waitSlot, int feature, int storedCategory) {
    transitionCategories[
            waitYakuBase
                + (flatTransition(actionSlot, transitionSlot)
                            * DecisionInputSchema.MAX_WAIT_TILE_TYPES
                        + waitSlot)
                    * DecisionInputSchema.ACTION_TRANSITION_WAIT_YAKU_STRIDE
                + feature] =
        (short) storedCategory;
  }

  /**
   * 待ち牌の公開得点特徴量を書き込む。
   *
   * @param actionSlot 親判断対象の行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param waitSlot 待ち集合内の位置
   * @param field RON・TSUMO別の得点フィールド
   * @param value 有限な正規化値
   */
  public void waitScore(
      int actionSlot,
      int transitionSlot,
      int waitSlot,
      DecisionInputSchema.ActionTransitionWaitFloat field,
      float value) {
    transitionNumerics[
            waitScoreBase
                + (flatTransition(actionSlot, transitionSlot)
                            * DecisionInputSchema.MAX_WAIT_TILE_TYPES
                        + waitSlot)
                    * DecisionInputSchema.ACTION_TRANSITION_WAIT_FLOAT_STRIDE
                + field.ordinal()] =
        value;
  }

  private int flatTransition(int actionSlot, int transitionSlot) {
    return transitionOffsets == null
        ? actionSlot * transitionsPerAction + transitionSlot
        : transitionOffsets[transitionRowBase + actionSlot] + transitionSlot;
  }

  private static int rowBase(
      DecisionInputLayout layout, DecisionInputLayout.Tensor tensor, int row) {
    return layout.region(tensor).rowOffset(row);
  }
}
