package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.model.ValueGraphHead.Composition;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 1 回の順伝播の損失の入力。差分整合の教師（Y・C の印・log p0_C・λ・行の重み）、行の平均の錨（D の印・|D|・行の重み）、補助の頭の教師と
 * 条件の印（{@link AuxiliaryLoss.Inputs}）を、ホストで全て並べてから型ごとに 1 本の配列でデバイスへ送る（各区画はその切り出し。
 * {@link BatchIndex} と同じ送り方）。{@link PrismLoss} は送った配列を読むだけにする。
 *
 * <p>学習では歩の作り手が前もって並べ（教師は錨を混ぜた後）、網の入力・索引と一緒に網の順伝播の前に非同期で送る（{@link StagedBatch}）。
 * 配列の値・型・形は損失がその場で作っていたものと同じ（差分整合の κ = 0・一様の基準の配列）で、損失の演算も同じ順なので、損失と勾配は
 * ビットで同じ。
 *
 * <p>行の手の数・手の容量・打牌の印（{@link Outputs#actions}・{@link Outputs#discards}）は網の入力の索引のホストの写しから {@link BatchIndex}
 * と同じ決まりで作り、損失は網の出力と照らす（{@link #check}）。
 */
public final class LossInputs {

  private static final ActionRole[] ROLES = ActionRole.values();

  private final int[] actions;
  private final boolean[] discards;
  private final int capacity;
  private final Staging.Section lambda;
  private final Staging.Section measured;
  private final Staging.Section targets;
  private final Staging.Section logPrior;
  private final Staging.Section weights;
  private final Staging.Section measuredDiscards;
  private final Staging.Section discardCounts;
  private final Staging.Section discardWeights;
  private final AuxiliaryLoss.Inputs auxiliary;

  /**
   * 教師をホストで並べて載せる（送るのは staging の持ち主）。
   *
   * <ul>
   *   <li>差分整合: [行, 容量] の C の印・Y（C の外は 0）・log p0_C（C の外は 0）と、[行, 1] の λ・行の重み（|C| &lt; 2 の行は 0）。
   *   <li>行の平均の錨: [行, 容量] の D の印（測った打牌）と、[行] の |D|（D が空の行は 1）・行の重み（D が空の行は 0）。
   * </ul>
   */
  private LossInputs(
      int[] actions,
      boolean[] discards,
      int capacity,
      List<ValueLabel> labels,
      AuxiliaryLabels auxiliaryLabels,
      AuxiliaryLoss.Denominators denominators,
      Staging staging) {
    int rows = actions.length;
    if (labels.size() != rows) {
      throw new IllegalArgumentException(labels.size() + " labels for " + rows + " rows");
    }
    if (discards.length != rows * capacity) {
      throw new IllegalArgumentException(
          discards.length + " discard flags for " + rows + " x " + capacity + " slots");
    }
    if ((auxiliaryLabels == null) != (denominators == null)) {
      throw new IllegalArgumentException("auxiliary labels and their denominators go together");
    }
    this.actions = actions;
    this.discards = discards;
    this.capacity = capacity;
    boolean[] measuredSlots = new boolean[rows * capacity];
    float[] logPriors = new float[rows * capacity];
    float[] lambdas = new float[rows];
    float[] valueWeights = new float[rows];
    boolean[] discardSlots = new boolean[rows * capacity];
    float[] counts = new float[rows];
    float[] discardRowWeights = new float[rows];
    for (int row = 0; row < rows; row++) {
      ValueLabel label = labels.get(row);
      float[] y = label.values();
      if (y.length != actions[row] || y.length > capacity) {
        throw new IllegalArgumentException(
            "row " + row + " has " + y.length + " values for " + actions[row] + " actions");
      }
      int base = row * capacity;
      int measuredCount = 0;
      int discardCount = 0;
      for (int a = 0; a < y.length; a++) {
        if (Float.isNaN(y[a])) {
          continue;
        }
        measuredSlots[base + a] = true;
        measuredCount++;
        if (discards[base + a]) {
          discardSlots[base + a] = true;
          discardCount++;
        }
      }
      if (measuredCount > 0) {
        // log p0_C = log 1 − log Σ_C 1（倍精度で引いてから丸める。|C| = 1 の行は +0）
        double logMass = Math.log(measuredCount);
        for (int a = 0; a < y.length; a++) {
          if (measuredSlots[base + a]) {
            logPriors[base + a] = (float) (0.0 - logMass);
          }
        }
      }
      lambdas[row] = label.lambda();
      valueWeights[row] = measuredCount >= 2 ? label.weight() : 0f;
      counts[row] = Math.max(discardCount, 1);
      discardRowWeights[row] = discardCount == 0 ? 0f : label.weight();
    }
    Shape slots = new Shape(rows, capacity);
    lambda = staging.add(lambdas, new Shape(rows, 1));
    measured = staging.add(measuredSlots, slots);
    targets = staging.add(targets(labels, capacity), slots);
    logPrior = staging.add(logPriors, slots);
    weights = staging.add(valueWeights, new Shape(rows, 1));
    measuredDiscards = staging.add(discardSlots, slots);
    discardCounts = staging.add(counts, new Shape(rows));
    discardWeights = staging.add(discardRowWeights, new Shape(rows));
    auxiliary =
        auxiliaryLabels == null ? null : AuxiliaryLoss.stage(auxiliaryLabels, denominators, staging);
  }

  /**
   * 塊の索引のホストの写しと教師から作って載せる（学習の歩の作り手の口。送るのは staging の持ち主）。
   *
   * @param inputs 塊の数と索引のホストの写し
   * @param labels 行の順の教師
   * @param auxiliary 補助の頭の教師（補助の頭を計算しないなら null）
   * @param denominators 補助の頭の平均の分母（教師があるときだけ。詰め方の 1 塊を分けた部分には元の塊の数）
   * @param staging 区画を載せる組
   * @return 損失の入力
   * @throws IllegalArgumentException 行の数か行の手の数が入力と合わないか、補助の教師と分母の片方だけがあるとき
   */
  static LossInputs stage(
      HostColumns inputs,
      List<ValueLabel> labels,
      AuxiliaryLabels auxiliary,
      AuxiliaryLoss.Denominators denominators,
      Staging staging) {
    int rows = inputs.rows();
    int[] actions = new int[rows];
    int capacity = 0;
    for (int row = 0; row < rows; row++) {
      actions[row] = inputs.row(row, RowColumn.ACTIONS);
      capacity = Math.max(capacity, actions[row]);
    }
    // 値を B が出す打牌の手（BatchIndex の discardValueSlots と同じ決まり）
    boolean[] discards = new boolean[rows * capacity];
    for (int row = 0; row < rows; row++) {
      int first = inputs.row(row, RowColumn.FIRST_ACTION);
      for (int k = 0; k < actions[row]; k++) {
        ActionRole role =
            ROLES[Objects.checkIndex(inputs.action(first + k, ActionColumn.ROLE), ROLES.length)];
        discards[row * capacity + k] = Composition.of(role) == Composition.DISCARD_VALUE;
      }
    }
    return new LossInputs(actions, discards, capacity, labels, auxiliary, denominators, staging);
  }

  /**
   * 網の出力の行の手の数・打牌の印と教師から作って送る（網の出力が既にあるときの口。試験と点検が使い、送り出しの置き場は管理元に付けて
   * 作る）。
   *
   * @param manager 配列と送り出しの置き場を付ける管理元
   * @param outputs 網の出力
   * @param labels 行の順の教師
   * @param auxiliary 補助の頭の教師（補助の項を足さないなら null）
   * @param denominators 補助の頭の平均の分母（教師があるときだけ）
   * @return 損失の入力
   * @throws IllegalArgumentException 行の数か行の手の数が出力と合わないか、補助の教師と分母の片方だけがあるとき
   */
  static LossInputs of(
      NDManager manager,
      Outputs outputs,
      List<ValueLabel> labels,
      AuxiliaryLabels auxiliary,
      AuxiliaryLoss.Denominators denominators) {
    Staging staging = new Staging();
    LossInputs inputs =
        new LossInputs(
            outputs.actions(),
            outputs.discards(),
            Math.toIntExact(outputs.scores().getShape().get(1)),
            labels,
            auxiliary,
            denominators,
            staging);
    staging.send(Uploads.on(manager), manager);
    return inputs;
  }

  /** Y [行 × 容量]（C の外は 0）。 */
  private static float[] targets(List<ValueLabel> labels, int capacity) {
    float[] values = new float[labels.size() * capacity];
    for (int row = 0; row < labels.size(); row++) {
      float[] y = labels.get(row).values();
      for (int a = 0; a < y.length; a++) {
        if (!Float.isNaN(y[a])) {
          values[row * capacity + a] = y[a];
        }
      }
    }
    return values;
  }

  /**
   * 網の出力が、送った教師と同じ行の手の数・打牌の印・手の容量であることを確かめる。
   *
   * @param outputs 網の出力
   * @throws IllegalArgumentException 合わないとき
   */
  void check(Outputs outputs) {
    if (!Arrays.equals(outputs.actions(), actions)
        || !Arrays.equals(outputs.discards(), discards)
        || outputs.scores().getShape().get(1) != capacity) {
      throw new IllegalArgumentException("the network outputs do not match the loss inputs");
    }
  }

  /** 送った配列の管理元（損失の配列もここに作る）。 */
  NDManager manager() {
    return lambda.array().getManager();
  }

  /** λ [行, 1]。 */
  NDArray lambda() {
    return lambda.array();
  }

  /** C の印 [行, 容量]（BOOLEAN）。 */
  NDArray measured() {
    return measured.array();
  }

  /** Y [行, 容量]（C の外は 0）。 */
  NDArray targets() {
    return targets.array();
  }

  /** log p0_C [行, 容量]（C の外は 0）。 */
  NDArray logPrior() {
    return logPrior.array();
  }

  /** 差分整合の行の重み [行, 1]（|C| &lt; 2 の行は 0）。 */
  NDArray weights() {
    return weights.array();
  }

  /** D の印 [行, 容量]（BOOLEAN）。 */
  NDArray measuredDiscards() {
    return measuredDiscards.array();
  }

  /** |D| [行]（D が空の行は 1）。 */
  NDArray discardCounts() {
    return discardCounts.array();
  }

  /** 行の平均の錨の行の重み [行]（D が空の行は 0）。 */
  NDArray discardWeights() {
    return discardWeights.array();
  }

  /** 補助の頭の教師と条件の印（補助の項を足さないなら null）。 */
  AuxiliaryLoss.Inputs auxiliary() {
    return auxiliary;
  }

}
