package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.classic.input.ClassicSchema;

/**
 * 公開の履歴を読むエンコーダー（Major だけ）。4 席の要約をクエリにして河と面子の 112 トークンを読み、要約を残差で更新する。読むときは、
 * トークンの時点と、その席が立直・副露・槓をした時点との関係（公開の履歴のコード）による値を足す。
 *
 * <p>関係の表は、コードの 4 ビットのフィールドごとに 1 つの区間を持つ。表の行の先頭 4 列はヘッドごとのロジットのバイアスで、残りは
 * バリューに足す差。残差に足す 2 つの射影は、重みとバイアスを 0 で初期化する。
 */
final class PublicHistoryEncoder extends AbstractBlock {

  private final int hidden;
  private final int attentionWidth;
  private final int feedForwardWidth;
  private final LayerNorm queryNorm;
  private final Linear queryProjection;
  private final IdEmbedding relationTable;
  private final Linear contextProjection;
  private final LayerNorm feedForwardNorm;
  private final Linear feedForwardExpansion;
  private final Linear feedForwardProjection;

  PublicHistoryEncoder(Widths widths) {
    hidden = widths.hidden();
    attentionWidth = widths.context();
    feedForwardWidth = widths.historyFeedForward();
    queryNorm = addChildBlock("queryNorm", LayerNorm.builder().axis(2).build());
    queryProjection =
        addChildBlock("queryProjection", Linear.builder().setUnits(attentionWidth).optBias(false).build());
    relationTable =
        addChildBlock(
            "relationTable",
            new IdEmbedding.Builder()
                .setDictionarySize(ClassicSchema.HISTORY_TABLE_ROWS)
                .setEmbeddingSize(Widths.HEADS + attentionWidth)
                .build());
    contextProjection = addChildBlock("contextProjection", zeroResidual(hidden));
    feedForwardNorm = addChildBlock("feedForwardNorm", LayerNorm.builder().axis(2).build());
    feedForwardExpansion =
        addChildBlock("feedForwardExpansion", Linear.builder().setUnits(feedForwardWidth).build());
    feedForwardProjection = addChildBlock("feedForwardProjection", zeroResidual(hidden));
  }

  /**
   * @param summaries 4 席の要約 [行, 4, 幅]
   * @param keyValues 河と面子のトークンのキーとバリュー [行, 112, 2 × Attention の幅]
   * @param mask トークンが実在すれば 1 [行, 112]（計算の型）
   * @param codes 公開の履歴のコード [行, 4, 112, 2]（INT16）
   * @return 更新した要約 [行, 4, 幅]
   */
  NDArray encode(Pass pass, NDArray summaries, NDArray keyValues, NDArray mask, NDArray codes) {
    NDArray queries = pass.linear(queryProjection, pass.norm(queryNorm, summaries));
    NDArray context =
        NDArrays.packedRelationScaledDotProductAttention(
            queries,
            keyValues,
            mask,
            codes,
            pass.ownedTable(relationTable),
            Widths.HEADS,
            ClassicSchema.HISTORY_SEGMENT_ROWS,
            1.0 / Math.sqrt((double) attentionWidth / Widths.HEADS));
    NDArray attended = summaries.add(pass.linear(contextProjection, context));
    NDArray expanded = Pass.silu(pass.linear(feedForwardExpansion, pass.norm(feedForwardNorm, attended)));
    return attended.add(pass.linear(feedForwardProjection, expanded));
  }

  private static Linear zeroResidual(int width) {
    Linear block = Linear.builder().setUnits(width).build();
    block.getDirectParameters().get("weight").setInitializer(new ConstantInitializer(0f));
    block.getDirectParameters().get("bias").setInitializer(new ConstantInitializer(0f));
    return block;
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    Shape summaries = new Shape(-1, ClassicSchema.SEATS, hidden);
    queryNorm.initialize(manager, dataType, summaries);
    queryProjection.initialize(manager, dataType, summaries);
    relationTable.initialize(manager, dataType, new Shape(-1));
    contextProjection.initialize(manager, dataType, new Shape(-1, ClassicSchema.SEATS, attentionWidth));
    feedForwardNorm.initialize(manager, dataType, summaries);
    feedForwardExpansion.initialize(manager, dataType, summaries);
    feedForwardProjection.initialize(manager, dataType, new Shape(-1, ClassicSchema.SEATS, feedForwardWidth));
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use encode");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, ClassicSchema.SEATS, hidden)};
  }
}
