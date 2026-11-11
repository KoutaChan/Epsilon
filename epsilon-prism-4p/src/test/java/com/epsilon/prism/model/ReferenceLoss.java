package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Loss;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Reduction;
import java.util.List;

/**
 * テストで比べる参照の損失。教師の配列を損失の中で作る（順伝播の前に {@link LossInputs} を送らない PrismLoss.total）。
 * 教師を前もって送る実装の損失と勾配が、これとビットで同じことを確かめるのに使う。
 */
final class ReferenceLoss {

  private static final TargetSchema HEADS = TargetSchema.of(Variant.TENHOU_4P);

  /** 手の軸。 */
  private static final int[] ACTIONS = {1};

  /** where で測っていない手に置く値。exp で 0 になり、FP32 で有限。 */
  private static final float FLOOR = -1e30f;

  private static final Outcome[] OUTCOMES = Outcome.values();
  private static final int CANDIDATE_HEADS = CandidateTarget.values().length;

  private ReferenceLoss() {}

  /**
   * マイクロバッチの損失 (1/B_C)·[Σ w·ℓ_PRISM + η·Σ w·s̄_D²] + share·L_aux を返す。
   *
   * @param scope 作った配列を付けるマネージャー
   * @param weights 補助のヘッドの重み
   * @param outputs ネットワークの出力
   * @param labels 行の順に並べた教師
   * @param auxiliaryLabels 補助のヘッドの教師。補助の項を足さないなら null
   * @param gain TEMPO のゲイン K′
   * @param denominator B_C
   * @param share 補助の項に掛ける割合
   * @return 損失
   */
  static NDArray total(
      NDManager scope,
      AuxiliaryLoss weights,
      Outputs outputs,
      List<ValueLabel> labels,
      AuxiliaryLabels auxiliaryLabels,
      float gain,
      int denominator,
      float share) {
    NDArray loss =
        value(scope, outputs, labels)
            .add(rowMean(scope, outputs, labels).mul(PrismLoss.ROW_MEAN))
            .div(denominator);
    if (auxiliaryLabels != null && outputs.auxiliary() != null) {
      loss = loss.add(auxiliary(scope, weights, outputs.auxiliary(), auxiliaryLabels, gain).mul(share));
    }
    return loss;
  }

  /**
   * 差分整合の和 Σ w·ℓ_PRISM を返す。κ = 0 で、基準は C の上の一様分布（log p0_C = −log |C|）。教師はホストで [行, 容量] に並べる。
   * C の上の softmax は C の外の手を where で除き、安定させるために引く行の最大値には勾配を流さない。重みの下限の項も足す。
   */
  private static NDArray value(NDManager scope, Outputs outputs, List<ValueLabel> labels) {
    int rows = outputs.actions().length;
    int capacity = Math.toIntExact(outputs.scores().getShape().get(1));
    boolean[] measured = new boolean[rows * capacity];
    float[] targets = new float[rows * capacity];
    float[] logPrior = new float[rows * capacity];
    float[] lambdas = new float[rows];
    float[] rowWeights = new float[rows];
    for (int row = 0; row < rows; row++) {
      ValueLabel label = labels.get(row);
      float[] values = label.values();
      int base = row * capacity;
      int count = 0;
      for (int a = 0; a < values.length; a++) {
        if (!Float.isNaN(values[a])) {
          measured[base + a] = true;
          targets[base + a] = values[a];
          count++;
        }
      }
      for (int a = 0; a < values.length; a++) {
        if (measured[base + a]) {
          logPrior[base + a] = (float) (0.0 - Math.log(count));
        }
      }
      lambdas[row] = label.lambda();
      rowWeights[row] = count >= 2 ? label.weight() : 0f;
    }
    Shape shape = new Shape(rows, capacity);
    Shape column = new Shape(rows, 1);
    NDArray lambda = scope.create(lambdas, column);
    NDArray z = outputs.scores().div(lambda);
    NDArray mask = scope.create(measured, shape);
    NDArray maskF = mask.toType(DataType.FLOAT32, false);
    NDArray nonEmpty = maskF.sum(ACTIONS, true).minimum(1f);
    NDArray peak =
        NDArrays.where(mask, z, scope.full(shape, FLOOR)).max(ACTIONS, true).stopGradient().mul(nonEmpty);
    NDArray shifted = NDArrays.where(mask, z.sub(peak), scope.zeros(shape, DataType.FLOAT32));
    NDArray e = shifted.exp().mul(maskF);
    NDArray sum = e.sum(ACTIONS, true).add(nonEmpty.neg().add(1f));
    NDArray pi = e.div(sum);
    NDArray logPi = shifted.sub(sum.log()).mul(maskF);
    NDArray target = scope.create(targets, shape);
    NDArray rowLoss =
        pi.mul(target)
            .sum(ACTIONS, true)
            .neg()
            .add(pi.mul(logPi.sub(scope.create(logPrior, shape))).sum(ACTIONS, true).mul(lambda))
            .add(weightFloor(scope, outputs.scores(), target, mask, maskF).div(lambda));
    return rowLoss.mul(scope.create(rowWeights, column)).sum();
  }

  /** 重みの下限の項を λ で割る前の値 (ε/2)·(1/|C|)·Σ_C (e_a − ē_C)²。e = s − Y、ē_C は C の上の e の平均。 */
  private static NDArray weightFloor(
      NDManager scope, NDArray scores, NDArray target, NDArray mask, NDArray maskF) {
    NDArray zeros = scope.zeros(scores.getShape(), DataType.FLOAT32);
    NDArray count = maskF.sum(ACTIONS, true).maximum(1f);
    NDArray error = NDArrays.where(mask, scores.sub(target), zeros);
    NDArray centered = NDArrays.where(mask, error.sub(error.sum(ACTIONS, true).div(count)), zeros);
    return centered.square().sum(ACTIONS, true).div(count).mul(PrismLoss.WEIGHT_FLOOR / 2);
  }

  /** 行の平均のアンカーの和。行ごとに、測った打牌の s の平均を 2 乗して行の重みを掛ける。 */
  private static NDArray rowMean(NDManager scope, Outputs outputs, List<ValueLabel> labels) {
    int rows = labels.size();
    int capacity = Math.toIntExact(outputs.scores().getShape().get(1));
    boolean[] discards = outputs.discards();
    boolean[] measured = new boolean[rows * capacity];
    float[] counts = new float[rows];
    float[] weights = new float[rows];
    for (int row = 0; row < rows; row++) {
      ValueLabel label = labels.get(row);
      float[] values = label.values();
      int count = 0;
      for (int a = 0; a < values.length; a++) {
        if (!Float.isNaN(values[a]) && discards[row * capacity + a]) {
          measured[row * capacity + a] = true;
          count++;
        }
      }
      counts[row] = Math.max(count, 1);
      weights[row] = count == 0 ? 0f : label.weight();
    }
    Shape column = new Shape(rows);
    NDArray scores = outputs.scores();
    NDArray kept =
        NDArrays.where(scope.create(measured, new Shape(rows, capacity)), scores, scope.zeros(scores.getShape()));
    NDArray mean = kept.sum(new int[] {1}).div(scope.create(counts, column));
    return mean.square().mul(scope.create(weights, column)).sum();
  }

  /** 補助の項 L_aux（AuxiliaryLoss.terms の total と同じ値）。 */
  private static NDArray auxiliary(
      NDManager scope, AuxiliaryLoss weights, Outputs.Auxiliary outputs, AuxiliaryLabels labels, float gain) {
    NDArray candidateSum = scope.zeros(new Shape());
    int candidates = labels.candidateActions().length;
    NDArray candidateTargets =
        candidates == 0
            ? null
            : scope.create(labels.candidateValues(), new Shape(candidates, HEADS.candidateWidth()));
    for (CandidateTarget target : CandidateTarget.values()) {
      float[] mask = new float[candidates];
      int count = 0;
      for (int c = 0; c < candidates; c++) {
        if (target.condition().holds(OUTCOMES[labels.outcomes()[c]])) {
          mask[c] = 1f;
          count++;
        }
      }
      NDArray term = scope.zeros(new Shape());
      if (count > 0) {
        NDArray logits = section(outputs.candidates(), HEADS.offset(target), HEADS.width(target));
        NDArray truth = section(candidateTargets, HEADS.offset(target), HEADS.width(target));
        NDArray units = unitLoss(target.loss(), target.reduction(), logits, truth, target.scale());
        term = units.mul(scope.create(mask)).sum().div(count);
      }
      candidateSum = candidateSum.add(term);
    }

    NDArray readingSum = scope.zeros(new Shape());
    int rows = labels.readingRows().length;
    int units = rows * HEADS.opponents();
    NDArray readingTargets =
        rows == 0
            ? null
            : scope.create(
                labels.readingValues(), new Shape(rows, HEADS.opponents(), HEADS.readingWidth()));
    for (ReadingTarget target : ReadingTarget.values()) {
      float[] mask = readingMask(target, labels.readingValues(), units);
      float used = 0;
      for (float value : mask) {
        used += value;
      }
      NDArray term = scope.zeros(new Shape());
      if (used > 0) {
        NDArray logits = section(outputs.reading(), HEADS.offset(target), HEADS.width(target));
        NDArray truth = section(readingTargets, HEADS.offset(target), HEADS.width(target));
        if (target.loss() == Loss.SOFTMAX) {
          float[] unitMask = new float[units];
          for (int u = 0; u < units; u++) {
            unitMask[u] = mask[u * HEADS.width(target)];
          }
          NDArray entropy = crossEntropy(logits, truth);
          term =
              entropy
                  .mul(scope.create(unitMask, new Shape(rows, HEADS.opponents())))
                  .sum()
                  .div(used / HEADS.width(target));
        } else {
          NDArray columns = columnLoss(target.loss(), logits, truth, target.scale());
          float denominator = target.reduction() == Reduction.MEAN ? used : used / HEADS.width(target);
          term =
              columns
                  .mul(scope.create(mask, new Shape(rows, HEADS.opponents(), HEADS.width(target))))
                  .sum()
                  .div(denominator);
        }
      }
      readingSum = readingSum.add(term);
    }
    return readingSum
        .mul(weights.oracleWeight())
        .add(candidateSum.mul(weights.resultWeight() / CANDIDATE_HEADS))
        .mul(gain);
  }

  private static NDArray section(NDArray array, int offset, int width) {
    return array.get(new NDIndex("..., {}:{}", offset, offset + width));
  }

  private static NDArray unitLoss(
      Loss loss, Reduction reduction, NDArray logits, NDArray truth, float scale) {
    if (loss == Loss.SOFTMAX) {
      return crossEntropy(logits, truth);
    }
    NDArray columns = columnLoss(loss, logits, truth, scale);
    int last = columns.getShape().dimension() - 1;
    return reduction == Reduction.MEAN ? columns.mean(new int[] {last}) : columns.sum(new int[] {last});
  }

  private static NDArray columnLoss(Loss loss, NDArray logits, NDArray truth, float scale) {
    return switch (loss) {
      case SIGMOID ->
          logits
              .maximum(0)
              .sub(logits.mul(truth))
              .add(logits.abs().neg().exp().add(1).log());
      case SQUARED -> logits.sub(truth.div(scale)).square();
      case SOFTMAX -> throw new IllegalArgumentException("softmax has no column loss");
    };
  }

  private static NDArray crossEntropy(NDArray logits, NDArray truth) {
    int last = logits.getShape().dimension() - 1;
    return truth.mul(logits.logSoftmax(last)).sum(new int[] {last}).neg();
  }

  private static float[] readingMask(ReadingTarget target, float[] values, int units) {
    int width = HEADS.width(target);
    float[] mask = new float[units * width];
    for (int u = 0; u < units; u++) {
      int base = u * HEADS.readingWidth();
      for (int k = 0; k < width; k++) {
        boolean used =
            switch (target.condition()) {
              case ALWAYS -> true;
              case OPPONENT_TENPAI -> values[base + HEADS.offset(ReadingTarget.TENPAI)] == 1f;
              case RON_TILE -> values[base + HEADS.offset(ReadingTarget.DANGER) + k] == 1f;
            };
        mask[u * width + k] = used ? 1f : 0f;
      }
    }
    return mask;
  }
}
