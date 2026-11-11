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
import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 34 種類の牌のトークンを、牌の種類どうしの決まった関係（{@link MahjongTileRelation}）を加えた Self-Attention で文脈化する。
 *
 * <p>ヘッド h で、クエリ i とキー j のロジットは (Q_i · K_j ＋ Q_i · R_key(h, r(i, j))) / √ヘッドの幅 ＋ R_bias(h, r(i, j))。r(i, j) は
 * 牌の種類 i と j の関係。関係の表は 0 で初期化するので、学習の始めは普通の Self-Attention と同じになる。段は 2 つある。どちらの段も、
 * Attention と SiLU の FFN は入力を正規化してから計算し、結果を残差に足す。最後に出力を正規化する。
 */
final class TileRelationEncoder extends AbstractBlock {

  private static final int BLOCKS = 2;
  private static final int RELATIONS = MahjongTileRelation.values().length;

  private final int hidden;
  private final Stage[] stages = new Stage[BLOCKS];
  private final LayerNorm outputLayerNorm;

  /** [クエリ, キー] → 関係の番号（INT64）。 */
  private NDArray relationIds;

  TileRelationEncoder(Widths widths) {
    hidden = widths.hidden();
    for (int index = 0; index < BLOCKS; index++) {
      stages[index] = addChildBlock("block" + index, new Stage(widths));
    }
    outputLayerNorm = addChildBlock("outputLayerNorm", LayerNorm.builder().axis(2).build());
  }

  /**
   * @param tiles 牌のトークン [行, 34, 幅]
   * @return 文脈化して正規化したトークン [行, 34, 幅]
   */
  NDArray encode(Pass pass, NDArray tiles) {
    try (ArrayScope scope = new ArrayScope()) {
      NDArray residual = tiles;
      NDArray normalized = pass.norm(stages[0].attentionInputLayerNorm, residual);
      for (int index = 0; index < BLOCKS; index++) {
        Stage stage = stages[index];
        LayerNorm following = index + 1 < BLOCKS ? stages[index + 1].attentionInputLayerNorm : outputLayerNorm;
        residual = residual.add(stage.attend(pass, normalized, relationIds));
        residual = residual.add(stage.feedForward(pass, residual));
        normalized = pass.norm(following, residual);
      }
      return scope.keep(normalized);
    }
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    relationIds = manager.create(MahjongTileRelation.relationIds(), new Shape(Tiles.KINDS, Tiles.KINDS));
    Shape tiles = new Shape(-1, Tiles.KINDS, hidden);
    for (Stage stage : stages) {
      stage.initialize(manager, dataType, tiles);
    }
    outputLayerNorm.initialize(manager, dataType, tiles);
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use encode");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, Tiles.KINDS, hidden)};
  }

  /** 1 つの段。関係のバイアスと関係のキーの表は 0 で初期化する。 */
  private static final class Stage extends AbstractBlock {

    private final int hidden;
    private final int attentionWidth;
    private final int headWidth;
    private final int feedForwardWidth;
    private final LayerNorm attentionInputLayerNorm;
    private final Linear queryKeyValueProjection;
    private final IdEmbedding relationBiasEmbedding;
    private final IdEmbedding relationKeyEmbedding;
    private final Linear attentionOutputProjection;
    private final LayerNorm feedForwardInputLayerNorm;
    private final Linear feedForwardExpansion;
    private final Linear feedForwardProjection;

    Stage(Widths widths) {
      hidden = widths.hidden();
      attentionWidth = widths.tileAttention();
      headWidth = attentionWidth / Widths.HEADS;
      feedForwardWidth = widths.tileFeedForward();
      attentionInputLayerNorm = addChildBlock("attentionInputLayerNorm", LayerNorm.builder().axis(2).build());
      queryKeyValueProjection =
          addChildBlock(
              "queryKeyValueProjection", Linear.builder().setUnits(3L * attentionWidth).optBias(false).build());
      relationBiasEmbedding = addChildBlock("relationBiasEmbedding", zeroTable(Widths.HEADS));
      relationKeyEmbedding = addChildBlock("relationKeyEmbedding", zeroTable(attentionWidth));
      attentionOutputProjection =
          addChildBlock("attentionOutputProjection", Linear.builder().setUnits(hidden).build());
      feedForwardInputLayerNorm = addChildBlock("feedForwardInputLayerNorm", LayerNorm.builder().axis(2).build());
      feedForwardExpansion =
          addChildBlock("feedForwardExpansion", Linear.builder().setUnits(feedForwardWidth).build());
      feedForwardProjection = addChildBlock("feedForwardProjection", Linear.builder().setUnits(hidden).build());
    }

    /** 正規化したトークン [行, 34, 幅] に関係つきの Attention をかけ、残差に足す更新 [行, 34, 幅] を返す。 */
    NDArray attend(Pass pass, NDArray normalized, NDArray relationIds) {
      try (ArrayScope scope = new ArrayScope()) {
        long rows = normalized.getShape().get(0);
        NDList parts = pass.linear(queryKeyValueProjection, normalized).split(3, 2);
        NDArray queries = heads(parts.get(0), rows);
        NDArray keys = heads(parts.get(1), rows);
        NDArray values = heads(parts.get(2), rows);
        // [クエリ, キー, ヘッド] → [1, ヘッド, クエリ, キー]
        NDArray bias = pass.embed(relationBiasEmbedding, relationIds).transpose(2, 0, 1).expandDims(0);
        // [関係, Attention の幅] → [1, ヘッド, ヘッドの幅, 関係]
        NDArray relationKeys =
            pass.own(pass.table(relationKeyEmbedding).reshape(RELATIONS, Widths.HEADS, headWidth))
                .transpose(1, 2, 0)
                .expandDims(0);
        NDArray context =
            NDArrays.relationBiasedScaledDotProductAttention(
                queries, keys, values, relationKeys, bias, relationIds, 1.0 / Math.sqrt(headWidth));
        return scope.keep(
            pass.linear(attentionOutputProjection, context.swapAxes(1, 2).reshape(rows, Tiles.KINDS, attentionWidth)));
      }
    }

    /** 残差 [行, 34, 幅] に FFN をかけ、残差に足す更新 [行, 34, 幅] を返す。 */
    NDArray feedForward(Pass pass, NDArray residual) {
      try (ArrayScope scope = new ArrayScope()) {
        NDArray expanded = Pass.silu(pass.linear(feedForwardExpansion, pass.norm(feedForwardInputLayerNorm, residual)));
        return scope.keep(pass.linear(feedForwardProjection, expanded));
      }
    }

    private NDArray heads(NDArray projected, long rows) {
      return projected.reshape(rows, Tiles.KINDS, Widths.HEADS, headWidth).swapAxes(1, 2);
    }

    private static IdEmbedding zeroTable(int width) {
      IdEmbedding table = new IdEmbedding.Builder().setDictionarySize(RELATIONS).setEmbeddingSize(width).build();
      table.getDirectParameters().valueAt(0).setInitializer(new ConstantInitializer(0f));
      return table;
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
      Shape tiles = new Shape(-1, Tiles.KINDS, hidden);
      attentionInputLayerNorm.initialize(manager, dataType, tiles);
      queryKeyValueProjection.initialize(manager, dataType, tiles);
      relationBiasEmbedding.initialize(manager, dataType, new Shape(Tiles.KINDS, Tiles.KINDS));
      relationKeyEmbedding.initialize(manager, dataType, new Shape(RELATIONS));
      attentionOutputProjection.initialize(manager, dataType, new Shape(-1, Tiles.KINDS, attentionWidth));
      feedForwardInputLayerNorm.initialize(manager, dataType, tiles);
      feedForwardExpansion.initialize(manager, dataType, tiles);
      feedForwardProjection.initialize(manager, dataType, new Shape(-1, Tiles.KINDS, feedForwardWidth));
    }

    @Override
    protected NDList forwardInternal(
        ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
      throw new UnsupportedOperationException("Use attend");
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
      return new Shape[] {new Shape(-1, Tiles.KINDS, hidden)};
    }
  }
}
