package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.model.ValueGraphHead.Composition;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 1 回の順伝播の損失の入力。差分整合の教師（Y・C の印・log p0_C・λ・行の重み）、行の平均の錨（D の印・|D|・行の重み）、補助の頭の教師と
 * 条件の印（{@link AuxiliaryLoss.Inputs}）を、ホストで全て並べてから型ごとに 1 本の配列でデバイスへ送る（各区画はその切り出し。
 * {@link BatchIndex} と同じ送り方）。{@link PrismLoss} は送った配列を読むだけにする。
 *
 * <p>学習のレーンは網の順伝播の前に作る（{@link #of(NetworkInputs, List, AuxiliaryLabels, AuxiliaryLoss.Denominators)}）。GPU の管理元での
 * 配列の作成はホストからの写しとストリームの同期なので、順伝播の後に作るとホストは順伝播が終わるまで待ち、その間に損失と逆伝播を積めない。
 * 配列の値・型・形は損失がその場で作っていたものと同じ（差分整合の κ = 0・一様の基準の配列）で、損失の演算も同じ順なので、損失と勾配は
 * ビットで同じ。
 *
 * <p>行の手の数・手の容量・打牌の印（{@link Outputs#actions}・{@link Outputs#discards}）は網の入力の索引のホストの写しから {@link BatchIndex}
 * と同じ決まりで作り、損失は網の出力と照らす（{@link #check}）。
 */
public final class LossInputs {

  private static final ActionRole[] ROLES = ActionRole.values();

  /** 送る区画の型。 */
  private enum Kind {
    FLOAT,
    BOOLEAN
  }

  private final NDManager manager;
  private final int[] actions;
  private final boolean[] discards;
  private final int capacity;
  private final Section lambda;
  private final Section measured;
  private final Section targets;
  private final Section logPrior;
  private final Section weights;
  private final Section measuredDiscards;
  private final Section discardCounts;
  private final Section discardWeights;
  private final AuxiliaryLoss.Inputs auxiliary;

  /**
   * 教師をホストで並べて送る。
   *
   * <ul>
   *   <li>差分整合: [行, 容量] の C の印・Y（C の外は 0）・log p0_C（C の外は 0）と、[行, 1] の λ・行の重み（|C| &lt; 2 の行は 0）。
   *   <li>行の平均の錨: [行, 容量] の D の印（測った打牌）と、[行] の |D|（D が空の行は 1）・行の重み（D が空の行は 0）。
   * </ul>
   */
  private LossInputs(
      NDManager manager,
      int[] actions,
      boolean[] discards,
      int capacity,
      List<ValueLabel> labels,
      AuxiliaryLabels auxiliaryLabels,
      AuxiliaryLoss.Denominators denominators) {
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
    this.manager = manager;
    this.actions = actions;
    this.discards = discards;
    this.capacity = capacity;
    boolean[] measuredSlots = new boolean[rows * capacity];
    float[] values = new float[rows * capacity];
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
        values[base + a] = y[a];
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
    Staging staging = new Staging();
    Shape slots = new Shape(rows, capacity);
    lambda = staging.add(lambdas, new Shape(rows, 1));
    measured = staging.add(measuredSlots, slots);
    targets = staging.add(values, slots);
    logPrior = staging.add(logPriors, slots);
    weights = staging.add(valueWeights, new Shape(rows, 1));
    measuredDiscards = staging.add(discardSlots, slots);
    discardCounts = staging.add(counts, new Shape(rows));
    discardWeights = staging.add(discardRowWeights, new Shape(rows));
    auxiliary =
        auxiliaryLabels == null ? null : AuxiliaryLoss.stage(auxiliaryLabels, denominators, staging);
    staging.send(manager);
  }

  /**
   * 網の入力（索引のホストの写し）と教師から作って送る（学習のレーンの口。網の順伝播の前に呼ぶ）。
   *
   * @param inputs 網の入力（配列は同じ管理元に作る）
   * @param labels 行の順の教師（錨を混ぜた後）
   * @param auxiliary 補助の頭の教師（補助の頭を計算しないなら null）
   * @param denominators 補助の頭の平均の分母（教師があるときだけ。詰め方の 1 塊を分けた部分には元の塊の数）
   * @return 損失の入力
   * @throws IllegalArgumentException 行の数か行の手の数が入力と合わないか、補助の教師と分母の片方だけがあるとき
   */
  public static LossInputs of(
      NetworkInputs inputs,
      List<ValueLabel> labels,
      AuxiliaryLabels auxiliary,
      AuxiliaryLoss.Denominators denominators) {
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
    return new LossInputs(
        inputs.manager(), actions, discards, capacity, labels, auxiliary, denominators);
  }

  /**
   * 網の出力の行の手の数・打牌の印と教師から作って送る（網の出力が既にあるときの口）。
   *
   * @param manager 配列を作る管理元
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
    return new LossInputs(
        manager,
        outputs.actions(),
        outputs.discards(),
        Math.toIntExact(outputs.scores().getShape().get(1)),
        labels,
        auxiliary,
        denominators);
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

  /** 配列の管理元（損失の配列もここに作る）。 */
  NDManager manager() {
    return manager;
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

  /** 送る区画。{@link Staging#send} が型ごとの 1 本の配列の切り出しを付け、それから {@link #array} で読む。 */
  static final class Section {

    private final Kind kind;
    private final Object values;
    private final int offset;
    private final int length;
    private final Shape shape;
    private NDArray array;

    private Section(Kind kind, Object values, int offset, int length, Shape shape) {
      this.kind = kind;
      this.values = values;
      this.offset = offset;
      this.length = length;
      this.shape = shape;
    }

    /**
     * 送った配列の切り出しを返す。
     *
     * @return 配列
     * @throws IllegalStateException 送る前のとき
     */
    NDArray array() {
      if (array == null) {
        throw new IllegalStateException("the section has not been sent");
      }
      return array;
    }
  }

  /** ホストの配列を型ごとに 1 本にまとめて送り、区画に切り出しを付ける。 */
  static final class Staging {

    private final List<Section> sections = new ArrayList<>();
    private final int[] sizes = new int[Kind.values().length];

    /** FP32 の区画を載せる。 */
    Section add(float[] values, Shape shape) {
      return add(Kind.FLOAT, values, values.length, shape);
    }

    /** BOOLEAN の区画を載せる。 */
    Section add(boolean[] values, Shape shape) {
      return add(Kind.BOOLEAN, values, values.length, shape);
    }

    private Section add(Kind kind, Object values, int length, Shape shape) {
      if (length == 0 || shape.size() != length) {
        throw new IllegalStateException("invalid staged section " + shape + " of " + length);
      }
      Section section = new Section(kind, values, sizes[kind.ordinal()], length, shape);
      sections.add(section);
      sizes[kind.ordinal()] += length;
      return section;
    }

    /**
     * 型ごとに 1 本の配列で送り、区画に切り出しを付ける。
     *
     * @param manager 配列を作る管理元
     */
    void send(NDManager manager) {
      float[] floats = new float[sizes[Kind.FLOAT.ordinal()]];
      boolean[] booleans = new boolean[sizes[Kind.BOOLEAN.ordinal()]];
      for (Section section : sections) {
        Object target =
            switch (section.kind) {
              case FLOAT -> floats;
              case BOOLEAN -> booleans;
            };
        System.arraycopy(section.values, 0, target, section.offset, section.length);
      }
      NDArray floatArray = floats.length == 0 ? null : manager.create(floats);
      NDArray booleanArray = booleans.length == 0 ? null : manager.create(booleans);
      for (Section section : sections) {
        NDArray all =
            switch (section.kind) {
              case FLOAT -> floatArray;
              case BOOLEAN -> booleanArray;
            };
        NDArray slice =
            section.length == all.size()
                ? all
                : all.get(new NDIndex("{}:{}", section.offset, section.offset + section.length));
        section.array = slice.reshape(section.shape);
      }
    }
  }
}
