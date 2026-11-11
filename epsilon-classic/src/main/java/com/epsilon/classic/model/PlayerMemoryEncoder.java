package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.classic.input.ClassicSchema;

/**
 * 席ごとの記憶（要約 1・河 24・面子 4 のトークン）を、その席のトークンの中だけで文脈化する。4 席とも同じ重みを使う。河の順、立直の前か
 * 後か、手出しかツモ切りか、河の牌と副露の対応は、まとめずにトークンに持たせたまま文脈化する。
 *
 * <p>実在しないトークンは Attention のキーから外し、残差を足すたびと出力のときに 0 にする。要約のトークンはどの席にも必ずある。
 */
final class PlayerMemoryEncoder extends AbstractBlock {

  /** 1 席分の記憶のトークン数。 */
  static final int TOKENS = 1 + ClassicSchema.RIVER_TOKENS + ClassicSchema.MELD_TOKENS;

  private static final int HEAD_WIDTH = Widths.MEMORY_ATTENTION / Widths.HEADS;

  private final int hidden;
  private final LayerNorm attentionInputLayerNorm;
  private final Linear queryKeyValueProjection;
  private final Linear attentionOutputProjection;
  private final LayerNorm feedForwardInputLayerNorm;
  private final Linear feedForwardExpansion;
  private final Linear feedForwardProjection;
  private final LayerNorm outputLayerNorm;

  PlayerMemoryEncoder(Widths widths) {
    hidden = widths.hidden();
    attentionInputLayerNorm = addChildBlock("attentionInputLayerNorm", LayerNorm.builder().axis(2).build());
    queryKeyValueProjection =
        addChildBlock(
            "queryKeyValueProjection",
            Linear.builder().setUnits(3L * Widths.MEMORY_ATTENTION).optBias(false).build());
    attentionOutputProjection = addChildBlock("attentionOutputProjection", Linear.builder().setUnits(hidden).build());
    feedForwardInputLayerNorm = addChildBlock("feedForwardInputLayerNorm", LayerNorm.builder().axis(2).build());
    feedForwardExpansion =
        addChildBlock("feedForwardExpansion", Linear.builder().setUnits(Widths.MEMORY_FEED_FORWARD).build());
    feedForwardProjection = addChildBlock("feedForwardProjection", Linear.builder().setUnits(hidden).build());
    outputLayerNorm = addChildBlock("outputLayerNorm", LayerNorm.builder().axis(2).build());
  }

  /**
   * @param tokens 席ごとのトークン [行, 4, 29, 幅]（正規化の前）
   * @param mask 実在すれば 1 [行, 4, 29]（計算の型）
   * @param entityNorm 入力のトークンを正規化する層（状態のエンコーダーが局と牌のトークンに使う層と共有）
   * @return 文脈化したトークン [行, 4, 29, 幅]。実在しないトークンは 0
   */
  NDArray encode(Pass pass, NDArray tokens, NDArray mask, LayerNorm entityNorm) {
    try (ArrayScope scope = new ArrayScope()) {
      Shape shape = tokens.getShape();
      long groups = shape.get(0) * ClassicSchema.SEATS;
      NDArray present = mask.reshape(groups, TOKENS);
      NDArray kept = present.expandDims(2);
      NDArray memory = ArrayScope.of(() -> pass.norm(entityNorm, tokens.reshape(groups, TOKENS, hidden)).mul(kept));
      memory = attend(pass, memory, present, kept);
      memory = feedForward(pass, memory, kept);
      NDArray output = memory;
      return scope.keep(ArrayScope.of(() -> pass.norm(outputLayerNorm, output).mul(kept).reshape(shape)));
    }
  }

  /** Attention の残差を足し、実在しないトークンを 0 にした記憶 [席の数 × 行, 29, 幅] を返す。 */
  private NDArray attend(Pass pass, NDArray memory, NDArray present, NDArray kept) {
    try (ArrayScope scope = new ArrayScope()) {
      long groups = memory.getShape().get(0);
      NDList parts = pass.linear(queryKeyValueProjection, pass.norm(attentionInputLayerNorm, memory)).split(3, 2);
      NDArray keyMask = present.reshape(groups, 1, 1, TOKENS).log();
      NDArray context =
          Pass.attention(heads(parts.get(0), groups), heads(parts.get(1), groups), heads(parts.get(2), groups), keyMask)
              .swapAxes(1, 2)
              .reshape(groups, TOKENS, Widths.MEMORY_ATTENTION);
      return scope.keep(memory.add(pass.linear(attentionOutputProjection, context)).mul(kept));
    }
  }

  /** FFN の残差を足し、実在しないトークンを 0 にした記憶 [席の数 × 行, 29, 幅] を返す。 */
  private NDArray feedForward(Pass pass, NDArray memory, NDArray kept) {
    try (ArrayScope scope = new ArrayScope()) {
      NDArray expanded = Pass.silu(pass.linear(feedForwardExpansion, pass.norm(feedForwardInputLayerNorm, memory)));
      return scope.keep(memory.add(pass.linear(feedForwardProjection, expanded)).mul(kept));
    }
  }

  private static NDArray heads(NDArray projected, long groups) {
    return projected.reshape(groups, TOKENS, Widths.HEADS, HEAD_WIDTH).swapAxes(1, 2);
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    Shape tokens = new Shape(-1, TOKENS, hidden);
    attentionInputLayerNorm.initialize(manager, dataType, tokens);
    queryKeyValueProjection.initialize(manager, dataType, tokens);
    attentionOutputProjection.initialize(manager, dataType, new Shape(-1, TOKENS, Widths.MEMORY_ATTENTION));
    feedForwardInputLayerNorm.initialize(manager, dataType, tokens);
    feedForwardExpansion.initialize(manager, dataType, tokens);
    feedForwardProjection.initialize(manager, dataType, new Shape(-1, TOKENS, Widths.MEMORY_FEED_FORWARD));
    outputLayerNorm.initialize(manager, dataType, tokens);
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use encode");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, ClassicSchema.SEATS, TOKENS, hidden)};
  }
}
