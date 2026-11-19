package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.types.DataType;

/** 測った手の対の二乗誤差。対の重みは教師と独立で、逆伝播では定数として扱う。 */
public record PairLoss(Weighting weighting, ClassWeight classWeight) {
  /** 学習設定で選ぶ目的関数。PRISM の類の重みだけを変えるときは同じ勾配の対の式を使う。 */
  public enum Mode {
    PRISM,
    PAIR_ROW,
    PAIR_STAR;

    /** 対の目的関数。既定の差分整合をそのまま使うときは null。 */
    public PairLoss objective(ClassWeight weight) {
      return switch (this) {
        case PRISM ->
            weight == ClassWeight.INVERSE_LAMBDA ? null : new PairLoss(Weighting.BASE, weight);
        case PAIR_ROW -> new PairLoss(Weighting.ROW, weight);
        case PAIR_STAR -> new PairLoss(Weighting.STAR, weight);
      };
    }
  }

  public enum ClassWeight {
    INVERSE_LAMBDA,
    UNIFORM
  }

  /** 基準は π⊗π と一様分布どうしの積の和。ROW は対角を除く総和で割り、STAR は基準の手以外に π と一様の下限を配る。 */
  public enum Weighting {
    BASE,
    ROW,
    STAR
  }

  private static final int[] ACTIONS = {1};

  /** 全ての順序対の二乗差の和は重み付き分散の二倍。手の数の二乗の配列を作らず、行内の平均を引いて求める。 */
  NDArray sum(NDArray scores, LossInputs inputs) {
    NDArray measured = inputs.measured();
    NDArray detached = scores.stopGradient();
    NDArray probability =
        PrismLoss.Softmax.of(inputs.manager(), detached.div(inputs.lambda()), measured).pi();
    NDArray mask = measured.toType(DataType.FLOAT32, false);
    NDArray uniform = mask.div(mask.sum(ACTIONS, true).maximum(1));
    NDArray error = NDArrays.where(measured, scores.sub(inputs.targets()), scores.zerosLike());
    NDArray row =
        switch (weighting) {
          case BASE, ROW -> {
            NDArray value =
                variance(error, probability)
                    .add(variance(error, uniform).mul(PrismLoss.WEIGHT_FLOOR));
            if (weighting == Weighting.BASE) yield value;
            NDArray mass =
                offDiagonal(probability).add(offDiagonal(uniform).mul(PrismLoss.WEIGHT_FLOOR));
            yield divide(value, mass);
          }
          case STAR -> {
            NDArray weights = probability.add(uniform.mul(PrismLoss.WEIGHT_FLOOR));
            // argMax は同点の先頭を返す。測っていない手はスコアにかかわらず選ばない。
            NDArray excluded = inputs.manager().create(Float.NEGATIVE_INFINITY);
            NDArray base = NDArrays.where(measured, detached, excluded).argMax(1).expandDims(1);
            NDArray diff = error.sub(error.gather(base, 1));
            NDArray others = weights.scatter(base, weights.gather(base, 1).zerosLike(), 1);
            yield divide(diff.square().mul(others).sum(ACTIONS, true), others.sum(ACTIONS, true));
          }
        };
    if (classWeight == ClassWeight.INVERSE_LAMBDA) {
      row = row.div(inputs.lambda());
    }
    return row.mul(inputs.weights()).sum().mul(0.5f);
  }

  private static NDArray variance(NDArray error, NDArray weights) {
    NDArray mean = error.mul(weights).sum(ACTIONS, true);
    return error.sub(mean).square().mul(weights).sum(ACTIONS, true);
  }

  private static NDArray offDiagonal(NDArray weights) {
    return weights.mul(weights.neg().add(1)).sum(ACTIONS, true);
  }

  private static NDArray divide(NDArray value, NDArray mass) {
    return value.div(NDArrays.where(mass.gt(0), mass, mass.onesLike()));
  }
}
