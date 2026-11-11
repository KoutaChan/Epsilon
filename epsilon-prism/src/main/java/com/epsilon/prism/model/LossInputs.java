package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Composition;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputFormat.Role;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 1 回の順伝播で使う損失の入力。差分整合の教師（Y・C の印・log p0_C・λ・行の重み）、行の平均のアンカー（D の印・|D|・行の重み）、
 * 補助のヘッドの教師と条件の印（{@link AuxiliaryLoss.Inputs}）を持つ。これらはホストで全て並べてから、型ごとに 1 本の配列にまとめて
 * デバイスへ送る。各部分はその配列の切り出しで、送り方は {@link BatchIndex} と同じ。{@link PrismLoss} は送った配列を読むだけ。
 *
 * <p>学習では、ステップを組み立てる側が前もって並べておき（教師はアンカーを混ぜた後の値）、ネットワークの入力・インデックスと一緒に、
 * 順伝播の前に非同期で送る（{@link StagedBatch}）。配列の値・型・形は、損失がその場で作っていたもの（差分整合の κ = 0、一様の基準の配列）
 * と同じで、損失の演算の順も同じなので、損失と勾配はビット単位で一致する。
 *
 * <p>行ごとの手の数・手の容量・打牌の印（{@link Outputs#actions}・{@link Outputs#discards}）は、ネットワークの入力のインデックスをホストに
 * 写したものから {@link BatchIndex} と同じ決まりで作る。損失はこれをネットワークの出力と照合する（{@link #check}）。
 */
public final class LossInputs {

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
   * 教師をホストで並べて {@code staging} に載せる。デバイスへ送るのは {@code staging} の所有者。
   *
   * <ul>
   *   <li>差分整合: [行, 容量] の C の印・Y（C の外は 0）・log p0_C（C の外は 0）と、[行, 1] の λ・行の重み（|C| &lt; 2 の行は 0）。
   *   <li>行の平均のアンカー: [行, 容量] の D の印（測った打牌）と、[行] の |D|（D が空の行は 1）・行の重み（D が空の行は 0）。
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
   * バッチのインデックスをホストに写したものと教師から損失の入力を作り、{@code staging} に載せる。学習でステップを組み立てる側が呼ぶ。デバイスへ
   * 送るのは {@code staging} の所有者。
   *
   * @param inputs バッチの件数とインデックスをホストに写したもの
   * @param labels 行の順に並べた教師
   * @param auxiliary 補助のヘッドの教師（補助のヘッドを計算しないなら null）
   * @param denominators 補助のヘッドの平均の分母（教師があるときだけ渡す。1 つのバッチを分けて詰めた部分には、元のバッチの件数を使う）
   * @param staging 各部分を載せる先
   * @return 損失の入力
   * @throws IllegalArgumentException 行の数か行ごとの手の数が入力と合わないとき、または補助の教師と分母の片方だけがあるとき
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
    // B が値を出す打牌の手に印を付ける（BatchIndex の discardValueSlots と同じ決まり）
    List<? extends Role> roles = inputs.format().roles();
    boolean[] discards = new boolean[rows * capacity];
    for (int row = 0; row < rows; row++) {
      int first = inputs.row(row, RowColumn.FIRST_ACTION);
      for (int k = 0; k < actions[row]; k++) {
        int roleIndex = inputs.action(first + k, ActionColumn.ROLE);
        Role role = roles.get(Objects.checkIndex(roleIndex, roles.size()));
        discards[row * capacity + k] = role.composition() == Composition.DISCARD_VALUE;
      }
    }
    return new LossInputs(actions, discards, capacity, labels, auxiliary, denominators, staging);
  }

  /**
   * ネットワークの出力にある行ごとの手の数・打牌の印と教師から損失の入力を作り、デバイスへ送る。ネットワークの出力が既にあるときに使い、
   * テストと点検が呼ぶ。送るためのバッファはマネージャーに付けて作る。
   *
   * @param manager 配列と、送るためのバッファを付けるマネージャー
   * @param outputs ネットワークの出力
   * @param labels 行の順に並べた教師
   * @param auxiliary 補助のヘッドの教師（補助の項を足さないなら null）
   * @param denominators 補助のヘッドの平均の分母（教師があるときだけ渡す）
   * @return 損失の入力
   * @throws IllegalArgumentException 行の数か行ごとの手の数が出力と合わないとき、または補助の教師と分母の片方だけがあるとき
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
    staging.send(PinnedUploads.on(manager), manager);
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
   * ネットワークの出力の行ごとの手の数・打牌の印・手の容量が、送った教師と同じであることを確かめる。
   *
   * @param outputs ネットワークの出力
   * @throws IllegalArgumentException 合わないとき
   */
  void check(Outputs outputs) {
    if (!Arrays.equals(outputs.actions(), actions)
        || !Arrays.equals(outputs.discards(), discards)
        || outputs.scores().getShape().get(1) != capacity) {
      throw new IllegalArgumentException("the network outputs do not match the loss inputs");
    }
  }

  /** 送った配列のマネージャー。損失の配列もここに作る。 */
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

  /** 行の平均のアンカーで使う行の重み [行]（D が空の行は 0）。 */
  NDArray discardWeights() {
    return discardWeights.array();
  }

  /** 補助のヘッドの教師と条件の印（補助の項を足さないなら null）。 */
  AuxiliaryLoss.Inputs auxiliary() {
    return auxiliary;
  }

}
