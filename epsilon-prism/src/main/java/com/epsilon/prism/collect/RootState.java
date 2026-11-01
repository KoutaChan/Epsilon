package com.epsilon.prism.collect;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.engine.RoundTransition;
import java.util.Arrays;

/**
 * 局頭の公開情報と、元の GRP が読む局境界の公開系列。変更不可で、配列は内部用の読み出しでも複製しない（呼び出し側は書き換えない）。
 */
final class RootState {
  private final int kyokuIndex;
  private final float[] publicPrefix;

  private RootState(int kyokuIndex, float[] publicPrefix) {
    this.kyokuIndex = kyokuIndex;
    this.publicPrefix = publicPrefix;
  }

  static RootState initial() {
    int[] scores = {25000, 25000, 25000, 25000};
    return new RootState(0, EpsilonGrpFeature.fromProgress(0, 0, 0, scores));
  }

  /** 精算後の公開情報を足した次局の root。 */
  RootState next(RoundTransition.NextRound next, int[] finalScores) {
    float[] feature =
        EpsilonGrpFeature.fromProgress(
            next.kyokuIndex(), next.honba(), next.kyotakuCount(), finalScores);
    float[] prefix = Arrays.copyOf(publicPrefix, publicPrefix.length + feature.length);
    System.arraycopy(feature, 0, prefix, publicPrefix.length, feature.length);
    return new RootState(next.kyokuIndex(), prefix);
  }

  int kyokuIndex() {
    return kyokuIndex;
  }

  /** 内部の配列そのもの。書き換えてはならない。 */
  float[] publicPrefix() {
    return publicPrefix;
  }
}
