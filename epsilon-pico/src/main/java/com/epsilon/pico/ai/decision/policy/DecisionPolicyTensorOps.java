package com.epsilon.pico.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.core.Action;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;

/**
 * 方策入力の有効要素の抽出、埋め込みの参照、マスク付きの遷移集約を行う。
 *
 * <p>学習パラメータは持たず、学習と推論で同じ位置の対応を使う。保存する ID は0をパディングに予約し、実際のテーブル位置には1を加えて記録する。
 */
final class DecisionPolicyTensorOps {

  private DecisionPolicyTensorOps() {}

  static NDArray compactBatchIds(
      NDArray presentIndices, long rowCount, int actionCapacity, int transitionCapacity) {
    long rowsPerBatch = Math.multiplyExact(actionCapacity, transitionCapacity);
    Math.multiplyExact(rowCount, rowsPerBatch);
    long presentCount = presentIndices.getShape().size();
    return presentIndices
        .div(rowsPerBatch)
        .toType(DataType.INT32, false)
        .reshape(1, presentCount)
        .stopGradient();
  }

  /**
   * 密な遷移クエリごとの元バッチ IDを、対応付けを使う注意機構が直接読めるINT32列として作る。
   *
   * <p>牌テーブル自体は複製せず、同じバッチの全行動・遷移へバッチ IDだけを繰り返す。返却テンソルはクエリと同じ 管理元/デバイスに置かれ、学習対象ではない。
   */
  static NDArray denseBatchIds(
      NDManager manager, long rowCount, int actionCapacity, int transitionCapacity) {
    long queriesPerBatch = Math.multiplyExact(actionCapacity, transitionCapacity);
    long queryCount = Math.multiplyExact(rowCount, queriesPerBatch);
    return manager
        .arange(0, Math.toIntExact(rowCount), 1, DataType.INT32)
        .repeat(queriesPerBatch)
        .reshape(queryCount)
        .stopGradient();
  }

  /**
   * 幅1の詰めた遷移行を行動配置へ直接戻す。
   *
   * <p>一つの行動に実在遷移が一つだけなら、条件付きsoftmaxの重みは必ず1になる。したがって実在行を指定位置へ配置し、
   * 存在しない行動を0のまま残せば、通常のマスク付き重み付け集約と同じ値になる。
   *
   * @param compactRows 実在遷移だけを詰めた行
   * @param presentIndices 密な行動配置上の実在行インデックス
   * @param rowCount バッチ行数
   * @param actionCapacity 1 行あたりの行動候補の位置数
   * @param rowWidth 各行の要素数
   * @return {@code [rowCount, actionCapacity, rowWidth]}へ戻した行
   */
  static NDArray scatterSingleCompactedTransitionRows(
      NDArray compactRows,
      NDArray presentIndices,
      long rowCount,
      int actionCapacity,
      int rowWidth) {
    long presentCount = presentIndices.getShape().size();
    long actionCount = Math.multiplyExact(rowCount, actionCapacity);
    return EpsilonMaskedRows.scatter(
            compactRows.reshape(presentCount, rowWidth), presentIndices, actionCount)
        .reshape(rowCount, actionCapacity, rowWidth);
  }

  /**
   * 幅1の遷移スコアを、対応する遷移表現と同じ演算データ型へ揃える。
   *
   * <p>通常経路ではスコアへ遷移マスクを乗算する際にこのデータ型へ昇格する。幅1の直接指定位置への配置経路でも同じ境界を明示し、
   * 後段の候補文脈Fusionが容量区分ごとに一貫したデータ型を受け取るようにする。
   *
   * @param scores 詰めた遷移スコア
   * @param transitionEmbeddings 対応する詰めた遷移表現
   * @return 遷移表現と同じデータ型のスコア
   */
  static NDArray alignSingleTransitionScoreType(NDArray scores, NDArray transitionEmbeddings) {
    return scores.toType(transitionEmbeddings.getDataType(), false);
  }

  /**
   * 詰めた遷移を条件付き方策重みで行動単位へ集約する。
   *
   * <p>GPUでは実在行だけを{@code index_add}し、CPUでは同じ値を密な指定位置へ配置して集約する。
   */
  static NDArray aggregateCompactedTransitions(
      NDArray compactTransitionEmbeddings,
      NDArray transitionWeights,
      NDArray presentIndices,
      long denseCount,
      long rowCount,
      int actionCapacity,
      int transitionCapacity,
      int hiddenSize) {
    long presentCount = presentIndices.getShape().size();
    NDArray compactRows = compactTransitionEmbeddings.reshape(presentCount, hiddenSize);
    if (compactRows instanceof PtNDArray
        && transitionWeights instanceof PtNDArray
        && presentIndices instanceof PtNDArray
        && compactRows.getDevice().isGpu()) {
      NDArray compactWeights =
          EpsilonMaskedRows.gather(transitionWeights.reshape(denseCount, 1), presentIndices);
      NDArray weightedRows = compactRows.mul(compactWeights);
      NDArray actionIndices = presentIndices.div(transitionCapacity).toType(DataType.INT64, false);
      long actionCount = Math.multiplyExact(rowCount, actionCapacity);
      NDArray zeros =
          compactRows
              .getManager()
              .zeros(new Shape(actionCount, hiddenSize), compactRows.getDataType());
      return JniUtils.indexAdd(
              (PtNDArray) zeros, (PtNDArray) actionIndices, (PtNDArray) weightedRows, 0)
          .reshape(rowCount, actionCapacity, hiddenSize);
    }

    NDArray denseEmbeddings =
        EpsilonMaskedRows.scatter(compactRows, presentIndices, denseCount)
            .reshape(rowCount, actionCapacity, transitionCapacity, hiddenSize);
    return denseEmbeddings.mul(transitionWeights.expandDims(3)).sum(new int[] {2});
  }

  static NDArray compactTransitionRows(NDArray denseRows, NDArray presentIndices, long denseCount) {
    Shape inputShape = denseRows.getShape();
    long presentCount = presentIndices.getShape().size();
    long trailingSize = inputShape.size() / denseCount;
    NDArray compactRows =
        EpsilonMaskedRows.gather(denseRows.reshape(denseCount, trailingSize), presentIndices);
    long[] compactShape = inputShape.getShape().clone();
    compactShape[0] = presentCount;
    compactShape[1] = 1;
    compactShape[2] = 1;
    return compactRows.reshape(new Shape(compactShape));
  }

  /** 応答元の牌文脈と、手番中KANの主となる-牌文脈を判断対象の行動へ割り当てる。 */
  static NDArray composeRootPlayerContexts(
      NDArray playerTileContexts,
      NDArray allPlayerTileContexts,
      NDArray actionCategories,
      DecisionPolicyCandidates.CandidateMasks candidateMasks,
      NDArray eventPlayerIds,
      NDArray eventTileIds) {
    long rowCount = actionCategories.getShape().get(0);
    int actionCapacity = Math.toIntExact(actionCategories.getShape().get(1));
    long hiddenSize = playerTileContexts.getShape().get(3);
    NDArray responseMask = candidateMasks.response().expandDims(2);
    NDArray responseContext =
        gatherPlayerTileContexts(
                playerTileContexts,
                eventPlayerIds.reshape(rowCount, 1),
                eventTileIds.reshape(rowCount, 1),
                rowCount,
                1)
            .reshape(rowCount, 1, hiddenSize)
            .broadcast(rowCount, actionCapacity, hiddenSize);
    NDArray turnKanMask = candidateMasks.group(Action.Group.KAN).expandDims(2);
    NDArray turnKanContext =
        gatherActionPrimaryTileEmbeddings(
            allPlayerTileContexts, actionCategories, rowCount, actionCapacity);
    return responseContext.mul(responseMask).add(turnKanContext.mul(turnKanMask));
  }

  static NDArray gatherTransitionDiscardContexts(
      NDArray allPlayerTileContexts,
      NDArray transitionCategories,
      long rowCount,
      int actionCapacity,
      int transitionCapacity) {
    NDArray tileIds =
        transitionCategories
            .get("...,{}", DecisionInputSchema.ActionTransitionInt.DISCARD_TILE.ordinal())
            .stopGradient();
    return gatherStoredEmbeddings(allPlayerTileContexts, tileIds)
        .reshape(
            rowCount, actionCapacity, transitionCapacity, allPlayerTileContexts.getShape().get(2));
  }

  static NDArray gatherActionPrimaryTileEmbeddings(
      NDArray tileEmbeddings, NDArray actionCategories, long rowCount, int actionCapacity) {
    NDArray tileIds =
        actionCategories
            .get("...,{}", DecisionInputSchema.ActionInt.PRIMARY_TILE.ordinal())
            .stopGradient();
    return gatherStoredEmbeddings(tileEmbeddings, tileIds)
        .reshape(rowCount, actionCapacity, tileEmbeddings.getShape().get(2));
  }

  static NDArray gatherTransitionDiscardTileEmbeddings(
      NDArray tileEmbeddings,
      NDArray transitionCategories,
      long rowCount,
      int actionCapacity,
      int transitionCapacity) {
    NDArray tileIds =
        transitionCategories
            .get("...,{}", DecisionInputSchema.ActionTransitionInt.DISCARD_TILE.ordinal())
            .stopGradient();
    return gatherStoredEmbeddings(tileEmbeddings, tileIds)
        .reshape(rowCount, actionCapacity, transitionCapacity, tileEmbeddings.getShape().get(2));
  }

  /** 詰めた遷移のバッチ IDと牌IDから、元の34牌テーブルを複製せず対象牌だけを集める。 */
  static NDArray gatherCompactedTransitionDiscardTileEmbeddings(
      NDArray tileEmbeddings, NDArray compactTransitionCategories, NDArray compactBatchIds) {
    long presentCount = compactTransitionCategories.getShape().get(0);
    NDArray tileIds =
        compactTransitionCategories
            .get("...,{}", DecisionInputSchema.ActionTransitionInt.DISCARD_TILE.ordinal())
            .stopGradient();
    return gatherStoredEmbeddingsByBatchIds(tileEmbeddings, tileIds, compactBatchIds)
        .reshape(presentCount, 1, 1, tileEmbeddings.getShape().get(2));
  }

  /** 詰めた遷移のバッチ IDと牌IDから、元の全プレイヤー牌文脈を複製せず対象牌だけを集める。 */
  static NDArray gatherCompactedTransitionDiscardContexts(
      NDArray allPlayerTileContexts, NDArray compactTransitionCategories, NDArray compactBatchIds) {
    long presentCount = compactTransitionCategories.getShape().get(0);
    NDArray tileIds =
        compactTransitionCategories
            .get("...,{}", DecisionInputSchema.ActionTransitionInt.DISCARD_TILE.ordinal())
            .stopGradient();
    return gatherStoredEmbeddingsByBatchIds(allPlayerTileContexts, tileIds, compactBatchIds)
        .reshape(presentCount, 1, 1, allPlayerTileContexts.getShape().get(2));
  }

  /** マスク外を0にし、候補ロジットの共通shiftに不変な遷移重みを返す。 */
  static NDArray maskedTransitionWeights(NDArray logits, NDArray presentMask) {
    return NDArrays.maskedSoftmax(logits, presentMask, 2).stopGradient();
  }

  private static NDArray gatherPlayerTileContexts(
      NDArray playerTileContexts,
      NDArray storedPlayerIds,
      NDArray storedTileIds,
      long rowCount,
      int flattenedCapacity) {
    NDArray playerIds = storedPlayerIds.reshape(rowCount, flattenedCapacity);
    NDArray tileIds = storedTileIds.reshape(rowCount, flattenedCapacity);
    return NDArrays.paddedBatchGather(playerTileContexts, playerIds, tileIds);
  }

  /** stored IDの0をゼロベクトル、1以降をテーブルインデックス + 1としてバッチごとに埋め込みを引く。 */
  static NDArray gatherStoredEmbeddings(NDArray embeddings, NDArray storedIds) {
    return NDArrays.paddedBatchGather(embeddings, storedIds);
  }

  /** 詰めた行ごとの元バッチ IDを使い、共有テーブルから直接stored IDを引く。 */
  private static NDArray gatherStoredEmbeddingsByBatchIds(
      NDArray embeddings, NDArray storedIds, NDArray compactBatchIds) {
    return NDArrays.paddedBatchGatherByBatchIndices(
        embeddings, compactBatchIds.reshape(storedIds.getShape()), storedIds);
  }
}
