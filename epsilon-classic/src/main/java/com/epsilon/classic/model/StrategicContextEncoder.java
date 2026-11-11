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
import ai.djl.util.PairList;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 戦略の文脈を作るエンコーダー。局・自分の手・4 席の要約の 6 トークンが互いを読み合い、ほかのトークンは読まない。自分の手のトークンは、
 * 局と自分の要約をクエリにして 34 種類の牌を読んだ結果と、牌の平均から作る。6 トークンにはそれぞれ別の埋め込みを足し、段を Nano は 2 つ、
 * Major は 3 つ重ねる。
 */
final class StrategicContextEncoder extends AbstractBlock {

  /** トークンの数（局・自分の手・4 席）。 */
  static final int TOKENS = 2 + ClassicSchema.SEATS;

  private static final int HAND_HEAD_WIDTH = Widths.HAND_ATTENTION / Widths.HEADS;

  private final int hidden;
  private final Linear handQueryProjection;
  private final Linear handKeyValueProjection;
  private final Linear handContextProjection;
  private final LayerNorm handSummaryLayerNorm;
  private final IdEmbedding tokenRoleEmbedding;
  private final Stage[] stages;

  StrategicContextEncoder(Widths widths) {
    hidden = widths.hidden();
    handQueryProjection =
        addChildBlock("handQueryProjection", Linear.builder().setUnits(Widths.HAND_ATTENTION).build());
    handKeyValueProjection =
        addChildBlock(
            "handKeyValueProjection",
            Linear.builder().setUnits(2L * Widths.HAND_ATTENTION).optBias(false).build());
    handContextProjection = addChildBlock("handContextProjection", Linear.builder().setUnits(hidden).build());
    handSummaryLayerNorm = addChildBlock("handSummaryLayerNorm", LayerNorm.builder().axis(1).build());
    tokenRoleEmbedding =
        addChildBlock(
            "tokenRoleEmbedding", new IdEmbedding.Builder().setDictionarySize(TOKENS).setEmbeddingSize(hidden).build());
    stages = new Stage[widths.strategicBlocks()];
    for (int index = 0; index < stages.length; index++) {
      stages[index] = addChildBlock("block" + index, new Stage(hidden, widths.strategicFeedForward()));
    }
  }

  /**
   * @param round 正規化した局 [行, 幅]
   * @param tiles 正規化した牌 [行, 34, 幅]
   * @param summaries 4 席の要約 [行, 4, 幅]（相対席の順）
   * @return 文脈化した局と 4 席の要約
   */
  Context encode(Pass pass, NDArray round, NDArray tiles, NDArray summaries) {
    try (ArrayScope scope = new ArrayScope()) {
      long rows = round.getShape().get(0);
      NDArray hand = summarizeHand(pass, round, summaries.get(":, 0, :"), tiles);
      NDArray roles = pass.own(pass.table(tokenRoleEmbedding).reshape(1, TOKENS, hidden));
      NDArray tokens =
          NDArrays.concat(new NDList(round.expandDims(1), hand.expandDims(1), summaries), 1).add(roles);
      for (Stage stage : stages) {
        tokens = stage.encode(pass, tokens);
      }
      return new Context(
          scope.keep(tokens.get(":, 0, :")),
          scope.keep(tokens.get(":, 2:, :").reshape(rows, ClassicSchema.SEATS, hidden)));
    }
  }

  /**
   * 文脈化した局と席の要約。
   *
   * @param round [行, 幅]
   * @param summaries [行, 4, 幅]
   */
  record Context(NDArray round, NDArray summaries) {}

  /** 自分の手のトークンを作る。局と自分の要約の和を √2 で割ってクエリにし、牌を読む。読んだ結果の射影を牌の平均に足して正規化する。 */
  private NDArray summarizeHand(Pass pass, NDArray round, NDArray self, NDArray tiles) {
    try (ArrayScope scope = new ArrayScope()) {
      long rows = round.getShape().get(0);
      NDArray seed = round.add(self).mul((float) (1.0 / Math.sqrt(2.0)));
      NDArray queries = pass.linear(handQueryProjection, seed).reshape(rows, Widths.HEADS, 1, HAND_HEAD_WIDTH);
      NDList keyValues = pass.linear(handKeyValueProjection, tiles).split(2, 2);
      NDArray keys = keyValues.get(0).reshape(rows, Tiles.KINDS, Widths.HEADS, HAND_HEAD_WIDTH).swapAxes(1, 2);
      NDArray values = keyValues.get(1).reshape(rows, Tiles.KINDS, Widths.HEADS, HAND_HEAD_WIDTH).swapAxes(1, 2);
      NDArray context = Pass.attention(queries, keys, values, null).reshape(rows, Widths.HAND_ATTENTION);
      NDArray tileMean = tiles.mean(new int[] {1});
      return scope.keep(pass.norm(handSummaryLayerNorm, tileMean.add(pass.linear(handContextProjection, context))));
    }
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    handQueryProjection.initialize(manager, dataType, new Shape(-1, hidden));
    handKeyValueProjection.initialize(manager, dataType, new Shape(-1, Tiles.KINDS, hidden));
    handContextProjection.initialize(manager, dataType, new Shape(-1, Widths.HAND_ATTENTION));
    handSummaryLayerNorm.initialize(manager, dataType, new Shape(-1, hidden));
    tokenRoleEmbedding.initialize(manager, dataType, new Shape(TOKENS));
    for (Stage stage : stages) {
      stage.initialize(manager, dataType, new Shape(-1, TOKENS, hidden));
    }
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use encode");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, hidden), new Shape(-1, ClassicSchema.SEATS, hidden)};
  }

  /** 6 トークンの 1 段。Attention と FFN は入力を正規化してから計算し、結果を残差に足す。最後に出力を正規化する。 */
  private static final class Stage extends AbstractBlock {

    private static final int HEAD_WIDTH = Widths.STRATEGIC_ATTENTION / Widths.HEADS;

    private final int hidden;
    private final int feedForwardWidth;
    private final LayerNorm attentionInputLayerNorm;
    private final Linear queryKeyValueProjection;
    private final Linear attentionOutputProjection;
    private final LayerNorm feedForwardInputLayerNorm;
    private final Linear feedForwardExpansion;
    private final Linear feedForwardProjection;
    private final LayerNorm outputLayerNorm;

    Stage(int hidden, int feedForwardWidth) {
      this.hidden = hidden;
      this.feedForwardWidth = feedForwardWidth;
      attentionInputLayerNorm = addChildBlock("attentionInputLayerNorm", LayerNorm.builder().axis(2).build());
      queryKeyValueProjection =
          addChildBlock(
              "queryKeyValueProjection",
              Linear.builder().setUnits(3L * Widths.STRATEGIC_ATTENTION).optBias(false).build());
      attentionOutputProjection =
          addChildBlock("attentionOutputProjection", Linear.builder().setUnits(hidden).build());
      feedForwardInputLayerNorm = addChildBlock("feedForwardInputLayerNorm", LayerNorm.builder().axis(2).build());
      feedForwardExpansion =
          addChildBlock("feedForwardExpansion", Linear.builder().setUnits(feedForwardWidth).build());
      feedForwardProjection = addChildBlock("feedForwardProjection", Linear.builder().setUnits(hidden).build());
      outputLayerNorm = addChildBlock("outputLayerNorm", LayerNorm.builder().axis(2).build());
    }

    NDArray encode(Pass pass, NDArray tokens) {
      try (ArrayScope scope = new ArrayScope()) {
        long rows = tokens.getShape().get(0);
        NDList parts = pass.linear(queryKeyValueProjection, pass.norm(attentionInputLayerNorm, tokens)).split(3, 2);
        NDArray context =
            Pass.attention(heads(parts.get(0), rows), heads(parts.get(1), rows), heads(parts.get(2), rows), null)
                .swapAxes(1, 2)
                .reshape(rows, TOKENS, Widths.STRATEGIC_ATTENTION);
        NDArray residual = tokens.add(pass.linear(attentionOutputProjection, context));
        NDArray expanded =
            Pass.silu(pass.linear(feedForwardExpansion, pass.norm(feedForwardInputLayerNorm, residual)));
        return scope.keep(pass.norm(outputLayerNorm, residual.add(pass.linear(feedForwardProjection, expanded))));
      }
    }

    private static NDArray heads(NDArray projected, long rows) {
      return projected.reshape(rows, TOKENS, Widths.HEADS, HEAD_WIDTH).swapAxes(1, 2);
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
      Shape tokens = new Shape(-1, TOKENS, hidden);
      attentionInputLayerNorm.initialize(manager, dataType, tokens);
      queryKeyValueProjection.initialize(manager, dataType, tokens);
      attentionOutputProjection.initialize(manager, dataType, new Shape(-1, TOKENS, Widths.STRATEGIC_ATTENTION));
      feedForwardInputLayerNorm.initialize(manager, dataType, tokens);
      feedForwardExpansion.initialize(manager, dataType, tokens);
      feedForwardProjection.initialize(manager, dataType, new Shape(-1, TOKENS, feedForwardWidth));
      outputLayerNorm.initialize(manager, dataType, tokens);
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
      throw new UnsupportedOperationException("Use encode");
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
      return new Shape[] {new Shape(-1, TOKENS, hidden)};
    }
  }
}
