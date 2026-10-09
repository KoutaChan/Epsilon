package com.epsilon.cfr.collection;

import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.engine.RoundStartPosition;
import com.epsilon.engine.RoundTransition;
import java.util.Arrays;

/** 公開情報からなる開始局面。凍結したGRPが参照する公開系列全体を保持する。 */
public record RootState(
    int kyokuIndex, int dealer, int honba, int kyotakuCount, int[] scores, float[] publicPrefix) {
  public RootState {
    scores = new RoundStartPosition(kyokuIndex, dealer, honba, kyotakuCount, scores).scores();
    publicPrefix = EpsilonGrpFeature.validatedCopy(publicPrefix);
    float[] last = EpsilonGrpFeature.fromProgress(kyokuIndex, honba, kyotakuCount, scores);
    if (publicPrefix.length < last.length
        || !Arrays.equals(
            last,
            Arrays.copyOfRange(
                publicPrefix, publicPrefix.length - last.length, publicPrefix.length))) {
      throw new IllegalArgumentException("GRP prefix must end at the public root");
    }
  }

  public static RootState initial() {
    int[] scores = {25000, 25000, 25000, 25000};
    return new RootState(0, 0, 0, 0, scores, EpsilonGrpFeature.fromProgress(0, 0, 0, scores));
  }

  public RoundStartPosition engineStart() {
    return new RoundStartPosition(kyokuIndex, dealer, honba, kyotakuCount, scores);
  }

  /** 連荘を含む実際の局数。反復方策を局ごとに抽選するときの識別子として使う。 */
  public int roundOrdinal() {
    return publicPrefix.length / EpsilonGrpFeature.FEATURE_SIZE;
  }

  /** 精算後の公開情報を追加し、次局の開始状態を作る。元の局境界系列は変更しない。 */
  RootState next(RoundTransition.NextRound next, int[] finalScores) {
    float[] feature =
        EpsilonGrpFeature.fromProgress(
            next.kyokuIndex(), next.honba(), next.kyotakuCount(), finalScores);
    float[] prefix = Arrays.copyOf(publicPrefix, publicPrefix.length + feature.length);
    System.arraycopy(feature, 0, prefix, publicPrefix.length, feature.length);
    return new RootState(
        next.kyokuIndex(), next.oya(), next.honba(), next.kyotakuCount(), finalScores, prefix);
  }

  @Override
  public int[] scores() {
    return scores.clone();
  }

  @Override
  public float[] publicPrefix() {
    return publicPrefix.clone();
  }
}
