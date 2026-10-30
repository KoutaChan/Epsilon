package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Loss;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Reduction;
import java.util.List;

/**
 * 試験の参照の実装: 教師の配列を損失の中で作る道（順伝播の前に送る {@link LossInputs} を使わない PrismLoss.total）。送る道の損失と
 * 勾配がこれとビットで同じことを確かめる。
 */
final class ReferenceLoss {

  /** 手の軸。 */
  private static final int[] ACTIONS = {1};

  /** where で外す枠の値（exp で 0 になる。FP32 で有限）。 */
  private static final float FLOOR = -1e30f;

  private static final Outcome[] OUTCOMES = Outcome.values();
  private static final int CANDIDATE_HEADS = CandidateTarget.values().length;

  private ReferenceLoss() {}

  /**
   * 微小バッチの損失 (1/B_C)·[Σ w·ℓ_PRISM + η·Σ w·s̄_D²] + share·L_aux。
   *
   * @param scope 作った配列を付ける管理元
   * @param weights 補助の頭の重み
   * @param outputs 網の出力
   * @param labels 行の順の教師
   * @param auxiliaryLabels 補助の頭の教師（補助の項を足さないなら null）
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
   * 差分整合の和 Σ w·ℓ_PRISM（κ = 0、基準は C の上の一様分布で log p0_C = −log |C|）。教師を [行, 容量] にホストで並べ、C の上の
   * softmax は C の外を where で外し、安定化に引く行の最大値の勾配を切る。
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
    NDArray rowLoss =
        pi.mul(scope.create(targets, shape))
            .sum(ACTIONS, true)
            .neg()
            .add(pi.mul(logPi.sub(scope.create(logPrior, shape))).sum(ACTIONS, true).mul(lambda));
    return rowLoss.mul(scope.create(rowWeights, column)).sum();
  }

  /** 行の平均の錨の和。 */
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

  /** 補助の項 L_aux（AuxiliaryLoss.terms の total）。 */
  private static NDArray auxiliary(
      NDManager scope, AuxiliaryLoss weights, Outputs.Auxiliary outputs, AuxiliaryLabels labels, float gain) {
    NDArray candidateSum = scope.zeros(new Shape());
    int candidates = labels.candidateActions().length;
    NDArray candidateTargets =
        candidates == 0
            ? null
            : scope.create(labels.candidateValues(), new Shape(candidates, CandidateTarget.WIDTH));
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
        NDArray logits = section(outputs.candidates(), target.offset(), target.width());
        NDArray truth = section(candidateTargets, target.offset(), target.width());
        NDArray units = unitLoss(target.loss(), target.reduction(), logits, truth, target.scale());
        term = units.mul(scope.create(mask)).sum().div(count);
      }
      candidateSum = candidateSum.add(term);
    }

    NDArray readingSum = scope.zeros(new Shape());
    int rows = labels.readingRows().length;
    int units = rows * AuxiliaryLabels.OPPONENTS;
    NDArray readingTargets =
        rows == 0
            ? null
            : scope.create(
                labels.readingValues(), new Shape(rows, AuxiliaryLabels.OPPONENTS, ReadingTarget.WIDTH));
    for (ReadingTarget target : ReadingTarget.values()) {
      float[] mask = readingMask(target, labels.readingValues(), units);
      float used = 0;
      for (float value : mask) {
        used += value;
      }
      NDArray term = scope.zeros(new Shape());
      if (used > 0) {
        NDArray logits = section(outputs.reading(), target.offset(), target.width());
        NDArray truth = section(readingTargets, target.offset(), target.width());
        if (target.loss() == Loss.SOFTMAX) {
          float[] unitMask = new float[units];
          for (int u = 0; u < units; u++) {
            unitMask[u] = mask[u * target.width()];
          }
          NDArray entropy = crossEntropy(logits, truth);
          term =
              entropy
                  .mul(scope.create(unitMask, new Shape(rows, AuxiliaryLabels.OPPONENTS)))
                  .sum()
                  .div(used / target.width());
        } else {
          NDArray columns = columnLoss(target.loss(), logits, truth, target.scale());
          float denominator = target.reduction() == Reduction.MEAN ? used : used / target.width();
          term =
              columns
                  .mul(scope.create(mask, new Shape(rows, AuxiliaryLabels.OPPONENTS, target.width())))
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
    int width = target.width();
    float[] mask = new float[units * width];
    for (int u = 0; u < units; u++) {
      int base = u * ReadingTarget.WIDTH;
      for (int k = 0; k < width; k++) {
        boolean used =
            switch (target.condition()) {
              case ALWAYS -> true;
              case OPPONENT_TENPAI -> values[base + ReadingTarget.TENPAI.offset()] == 1f;
              case RON_TILE -> values[base + ReadingTarget.DANGER.offset() + k] == 1f;
            };
        mask[u * width + k] = used ? 1f : 0f;
      }
    }
    return mask;
  }
}
