package com.epsilon.prism.kernel;

import com.epsilon.prism.kernel.ModelSchema.Arithmetic;
import com.epsilon.prism.kernel.ModelSchema.Stage;
import java.util.ArrayList;
import java.util.List;

/** 配布モデルの候補計算に使う演算と、積和・丸めの順序を返す。 */
public final class CollectionOrder {
  private CollectionOrder() {}

  public static List<Stage> stages(boolean segments, boolean reread) {
    var stages = new ArrayList<Stage>();
    if (segments) {
      stages.add(Stage.COUNT_TILES);
      stages.add(Stage.SEGMENT_BIAS);
    }
    stages.add(Stage.CANDIDATE_ATTENTION);
    stages.add(Stage.CANDIDATE_FFN);
    if (reread) stages.add(Stage.EVENT_REREAD);
    stages.add(Stage.DISCARD_VALUE);
    stages.add(Stage.COMPOSE_VALUES);
    return List.copyOf(stages);
  }

  public static List<Arithmetic> arithmetic(boolean folded, boolean causal, boolean segments) {
    var arithmetic =
        new ArrayList<>(
            List.of(
                Arithmetic.EMBED_COLUMN_FMA,
                Arithmetic.BF16_TILE_K_ASCENDING,
                Arithmetic.PAIR_BIAS_01_23_45_6,
                Arithmetic.RESIDUAL_ROUND_BF16));
    if (folded) arithmetic.add(Arithmetic.FOLDED_QKV_UNROUNDED_RMS);
    else arithmetic.add(Arithmetic.RMS_XOR_1_2);
    if (causal)
      arithmetic.addAll(
          List.of(Arithmetic.PREFIX_FULL_SPLIT_BF16, Arithmetic.PREFIX_EDGE_FP32_FMA));
    if (segments)
      arithmetic.addAll(
          List.of(Arithmetic.COUNT_TILE_MUL_ADD, Arithmetic.SEGMENT_DECLARATION_ORDER));
    arithmetic.addAll(List.of(Arithmetic.CANDIDATE_KEY_ORDER, Arithmetic.VALUE_ACTION_ORDER));
    return List.copyOf(arithmetic);
  }
}
