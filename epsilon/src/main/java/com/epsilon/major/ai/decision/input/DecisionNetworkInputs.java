package com.epsilon.major.ai.decision.input;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Decision ネットワークへ渡す各テンソルを名前で識別し、入力順序とビューの構築を管理する。
 *
 * <p>NDListの位置、ネットワーク初期化形状、連続バッファからのビュー構築をこの型だけが管理する。ネットワーク、InferenceServer、Trainerは
 * 数値インデックスで入力の意味を復元しない。
 *
 * @param stateCategories 局・プレイヤー・牌・河・面子のカテゴリ値フィールド
 * @param actionCategories 行動候補ごとのカテゴリ値フィールド
 * @param actionRoutes 方策グラフの条件付き経路 ID
 * @param pointLedger100 絶対席順の現在点を100点単位で保持するINT32台帳
 * @param actionWinFacts 即時和了候補ごとのPoint Facts
 * @param transitionCategories 行動-遷移ごとのカテゴリ値フィールド
 * @param transitionTiles 行動適用後の状態の牌種別フィールド
 * @param waitTileIds 遷移ごとの疎な wait 牌種 ID
 * @param waitWinFacts 疎なwaitと和了方法ごとのPoint Facts
 * @param stateNumerics 局・プレイヤー・牌・河・面子の数値フィールド
 * @param boundaryContext 価値専用のGRP 周辺分布と配牌前局境界特徴量
 * @param transitionNumerics 行動-遷移ごとの数値フィールド
 */
public record DecisionNetworkInputs(
    NDArray stateCategories,
    NDArray actionCategories,
    NDArray actionRoutes,
    NDArray pointLedger100,
    NDArray actionWinFacts,
    NDArray transitionCategories,
    NDArray transitionTiles,
    NDArray waitTileIds,
    NDArray waitWinFacts,
    NDArray stateNumerics,
    NDArray boundaryContext,
    NDArray transitionNumerics)
    implements DecisionPolicyInputs {

  /** 全論理的なテンソルが同じ正のバッチ行数を持つことを検証する。 */
  public DecisionNetworkInputs {
    long rows = stateCategories.getShape().get(0);
    if (rows < 1
        || actionCategories.getShape().get(0) != rows
        || actionRoutes.getShape().get(0) != rows
        || pointLedger100.getShape().get(0) != rows
        || actionWinFacts.getShape().get(0) != rows
        || transitionCategories.getShape().get(0) != rows
        || transitionTiles.getShape().get(0) != rows
        || waitTileIds.getShape().get(0) != rows
        || waitWinFacts.getShape().get(0) != rows
        || stateNumerics.getShape().get(0) != rows
        || boundaryContext.getShape().get(0) != rows
        || transitionNumerics.getShape().get(0) != rows) {
      throw new IllegalArgumentException("Decision input tensors have inconsistent row counts");
    }
  }

  /**
   * 転送済み2本の連続バッファから標準形式の論理ビューを構築する。
   *
   * <p>返すNDArrayは入力連続バッファの部分領域/reshape ビューであり、テンソルごとの追加複製を行わない。
   *
   * @param layout 連続バッファオフセットと論理的な形状の契約
   * @param categoricalSlab デバイス上の連続カテゴリ値連続バッファ
   * @param numericSlab デバイス上の連続数値連続バッファ
   * @return 名前付き論理的なテンソルビュー
   */
  public static DecisionNetworkInputs bind(
      DecisionInputLayout layout, NDArray categoricalSlab, NDArray numericSlab) {
    NDArray[] views = new NDArray[DecisionInputLayout.tensorCount()];
    for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
      DecisionInputLayout.Region region = layout.region(tensor);
      NDArray slab =
          tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL ? categoricalSlab : numericSlab;
      views[tensor.ordinal()] =
          slice(
              slab,
              region.slabOffset(),
              region.slabOffset() + region.elementCount(),
              region.shape(layout.capacity(), layout.bucket()));
    }
    return fromOrderedViews(views);
  }

  /**
   * DJL Block境界のNDListを名前付き入力へ復元する。
   *
   * @param inputs {@link DecisionInputLayout.Tensor}宣言順のNDList
   * @return 各位置へ意味名を付けた入力
   */
  public static DecisionNetworkInputs fromNDList(NDList inputs) {
    if (inputs.size() != DecisionInputLayout.tensorCount()) {
      throw new IllegalArgumentException(
          "Decision network requires "
              + DecisionInputLayout.tensorCount()
              + " input tensors, got "
              + inputs.size());
    }
    return fromOrderedViews(inputs.toArray(new NDArray[0]));
  }

  /**
   * DJL Block境界の標準形式の順NDListへ変換する。
   *
   * @return {@link DecisionInputLayout.Tensor}宣言順のNDList
   */
  public NDList toNDList() {
    return new NDList(
        stateCategories,
        actionCategories,
        actionRoutes,
        pointLedger100,
        actionWinFacts,
        transitionCategories,
        transitionTiles,
        waitTileIds,
        waitWinFacts,
        stateNumerics,
        boundaryContext,
        transitionNumerics);
  }

  /**
   * バッチ行数を返す。
   *
   * @return 全論理的なテンソルに共通する先頭軸長
   */
  public int rowCount() {
    return Math.toIntExact(stateCategories.getShape().get(0));
  }

  /**
   * テンソル形状から行動・遷移容量区分を復元する。
   *
   * @return 現在入力の物理容量区分
   */
  public DecisionBucket bucket() {
    return new DecisionBucket(
        Math.toIntExact(actionCategories.getShape().get(1)),
        Math.toIntExact(transitionCategories.getShape().get(2)));
  }

  /**
   * 局カテゴリ値テンソルから指定フィールドの {@code [batch]} ビューを返す。
   *
   * @param field 取得する局フィールド
   * @return 状態カテゴリ値連続バッファを共有する1次元ビュー
   */
  public NDArray roundCategory(DecisionInputSchema.RoundInt field) {
    return stateCategories.get(":,{}", field.ordinal());
  }

  /**
   * 行動 IDのパディングから合法候補マスクを導出する。
   *
   * @return 形状 {@code [batch, action]} のboolean マスク
   */
  public NDArray legalActionMask() {
    return actionCategories
        .get("...,{}", DecisionInputSchema.ActionInt.ID.ordinal())
        .neq(DecisionInputSchema.PAD_ID);
  }

  /**
   * ネットワーク初期化に使う全入力の動的な形状を返す。
   *
   * @return 標準形式のテンソル順の形状配列
   */
  public static Shape[] initializationShapes() {
    return DecisionInputLayout.networkInputShapes();
  }

  /**
   * 指定論理的なテンソルの動的な初期化形状を返す。
   *
   * @param tensor 対象論理的なテンソル
   * @return バッチ・行動・遷移軸を動的にした形状
   */
  public static Shape initializationShape(DecisionInputLayout.Tensor tensor) {
    return tensor.shape(-1, -1, -1);
  }

  private static DecisionNetworkInputs fromOrderedViews(NDArray[] views) {
    return new DecisionNetworkInputs(
        views[DecisionInputLayout.Tensor.STATE_CATEGORIES.ordinal()],
        views[DecisionInputLayout.Tensor.ACTION_CATEGORIES.ordinal()],
        views[DecisionInputLayout.Tensor.ACTION_ROUTES.ordinal()],
        views[DecisionInputLayout.Tensor.POINT_LEDGER_100.ordinal()].toType(DataType.INT32, false),
        views[DecisionInputLayout.Tensor.ACTION_WIN_FACTS.ordinal()],
        views[DecisionInputLayout.Tensor.TRANSITION_CATEGORIES.ordinal()],
        views[DecisionInputLayout.Tensor.TRANSITION_TILES.ordinal()],
        views[DecisionInputLayout.Tensor.WAIT_TILE_IDS.ordinal()],
        views[DecisionInputLayout.Tensor.WAIT_WIN_FACTS.ordinal()],
        views[DecisionInputLayout.Tensor.STATE_NUMERICS.ordinal()],
        views[DecisionInputLayout.Tensor.BOUNDARY_CONTEXT.ordinal()],
        views[DecisionInputLayout.Tensor.TRANSITION_NUMERICS.ordinal()]);
  }

  private static NDArray slice(NDArray slab, int start, int end, Shape shape) {
    if (start == end) {
      return slab.get(new NDIndex().addSliceDim(0, 0)).reshape(shape);
    }
    return slab.get(new NDIndex().addSliceDim(start, end)).reshape(shape);
  }
}
