package com.epsilon.ai.decision;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Activation;

/** 二つの局境界効用による、既存の二択ゲートだけのクリップ付き方策目的。 */
public final class DecisionBranchLoss {
  private DecisionBranchLoss() {}

  /** 順伝播の値を変えず、比較対象ゲートへの通常方策損失の勾配を切る。 */
  public static NDArray detachComparedGates(NDArray scores, NDArray targets) {
    NDArray codes = targets.get(":,0");
    NDList columns = new NDList();
    for (int gate = 0; gate < 3; gate++) {
      NDArray score = scores.get(":,{}", gate);
      NDArray mask = codes.eq(gate + 1).toType(DataType.FLOAT32, false);
      columns.add(score.mul(mask.neg().add(1)).add(score.stopGradient().mul(mask)).expandDims(1));
    }
    columns.add(scores.get(":,3:"));
    return NDArrays.concat(columns, 1);
  }

  /** 完成した比較の効用差から、対象ゲートのクリップ付き方策損失を求める。 */
  public static NDArray loss(NDArray scores, NDArray targets, NDArray actorWeights, float clip) {
    NDArray codes = targets.get(":,0");
    NDArray selectedScore = scores.get(":,0").mul(codes.eq(1));
    selectedScore = selectedScore.add(scores.get(":,1").mul(codes.eq(2)));
    selectedScore = selectedScore.add(scores.get(":,2").mul(codes.eq(3)));
    NDArray probability = Activation.sigmoid(selectedScore);
    // 未完成行も同じバッチに入るため、非対象行の分母を先に有効な値にする。
    NDArray completedMask = targets.get(":,4");
    NDArray oldProbability =
        targets.get(":,1").mul(completedMask).add(completedMask.neg().add(1).mul(0.5f));
    NDArray difference = targets.get(":,2").sub(targets.get(":,3")).stopGradient();
    NDArray acceptedAdvantage = oldProbability.neg().add(1).mul(difference);
    NDArray declinedAdvantage = oldProbability.mul(difference).neg();
    NDArray acceptedRatio = probability.div(oldProbability);
    NDArray declinedRatio = probability.neg().add(1).div(oldProbability.neg().add(1));
    NDArray objective =
        clipped(acceptedRatio, acceptedAdvantage, clip)
            .mul(oldProbability)
            .add(clipped(declinedRatio, declinedAdvantage, clip).mul(oldProbability.neg().add(1)));
    NDArray weights = completedMask.mul(actorWeights).stopGradient();
    return objective.mul(weights).sum().neg().div(weights.sum().maximum(1));
  }

  private static NDArray clipped(NDArray ratio, NDArray advantage, float clip) {
    return ratio.mul(advantage).minimum(ratio.clip(1 - clip, 1 + clip).mul(advantage));
  }
}
