package com.epsilon.major.ai.decision.input;

import com.epsilon.ai.decision.DecisionBranchTarget;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * 選択行動、探索前後の方策、価値教師値、アドバンテージ、学習重みを CPU 上に保持する。
 *
 * <p>入力特徴量とは独立して、選択格納位置、探索後と探索前の方策、価値効用、選択行動 アドバンテージ、方策学習/サンプル重みを所有する。カテゴリ値と数値の2本の連続バッファは変換済み
 * データセットとデバイス転送のための物理表現であり、呼び出し側は {@link #writeRow}と型付き参照メソッドを使用する。
 */
public final class DecisionTrainingTargets {

  /** スカラー効用教師値の成分数。 */
  public static final int VALUE_TARGET_SIZE = 1;

  /** スカラー選択行動アドバンテージの成分数。 */
  public static final int ADVANTAGE_SIZE = 1;

  /** 価値、アドバンテージ、学習重み、分岐比較教師値を合わせた行幅。 */
  public static final int ROW_NUMERIC_STRIDE =
      VALUE_TARGET_SIZE + ADVANTAGE_SIZE + 2 + DecisionBranchTarget.WIDTH;

  /** 行内の分岐比較教師値の開始位置。 */
  public static final int BRANCH_OFFSET = 4;

  private static final int ADVANTAGE_COMPONENT = VALUE_TARGET_SIZE;
  private static final int ACTOR_WEIGHT_COMPONENT = ADVANTAGE_COMPONENT + ADVANTAGE_SIZE;
  private static final int SAMPLE_WEIGHT_COMPONENT = ACTOR_WEIGHT_COMPONENT + 1;

  private final int capacity;
  private final int actionCapacity;
  private final int behaviorPolicyOffset;
  private final int rolloutPolicyOffset;
  private final int rowNumericOffset;
  private final int[] categoricalSlab;
  private final float[] numericSlab;

  /**
   * 指定容量と行動容量区分に対応する空の教師値連続バッファを確保する。
   *
   * @param capacity 格納できる最大行数
   * @param bucket 方策教師値幅を決めるDecision 容量区分
   */
  public DecisionTrainingTargets(int capacity, DecisionBucket bucket) {
    if (capacity < 1) {
      throw new IllegalArgumentException("capacity must be positive");
    }
    this.capacity = capacity;
    actionCapacity = bucket.legalActionCapacity();
    behaviorPolicyOffset = 0;
    rolloutPolicyOffset = Math.multiplyExact(capacity, actionCapacity);
    rowNumericOffset = Math.multiplyExact(rolloutPolicyOffset, 2);
    categoricalSlab = new int[capacity];
    numericSlab =
        new float
            [Math.addExact(rowNumericOffset, Math.multiplyExact(capacity, ROW_NUMERIC_STRIDE))];
  }

  private DecisionTrainingTargets(
      int capacity, DecisionBucket bucket, int[] categoricalSlab, float[] numericSlab) {
    if (capacity < 1) {
      throw new IllegalArgumentException("capacity must be positive");
    }
    this.capacity = capacity;
    actionCapacity = bucket.legalActionCapacity();
    behaviorPolicyOffset = 0;
    rolloutPolicyOffset = Math.multiplyExact(capacity, actionCapacity);
    rowNumericOffset = Math.multiplyExact(rolloutPolicyOffset, 2);
    requireLength(categoricalSlab.length, capacity, "target categories");
    requireLength(
        numericSlab.length,
        Math.addExact(rowNumericOffset, Math.multiplyExact(capacity, ROW_NUMERIC_STRIDE)),
        "target numerics");
    this.categoricalSlab = categoricalSlab;
    this.numericSlab = numericSlab;
  }

  /** 復号済み教師値連続バッファの所有権を受け取る。呼出元は返却後に配列を読み書きしてはならない。 */
  static DecisionTrainingTargets takeEncoded(
      int capacity, DecisionBucket bucket, int[] categoricalSlab, float[] numericSlab) {
    return new DecisionTrainingTargets(capacity, bucket, categoricalSlab, numericSlab);
  }

  /**
   * 格納できる最大行数を返す。
   *
   * @return 行容量
   */
  public int capacity() {
    return capacity;
  }

  /**
   * 一行の探索適用後の・対局生成方策幅を返す。
   *
   * @return 合法手容量区分幅
   */
  public int actionCapacity() {
    return actionCapacity;
  }

  /**
   * 永続化・一括転送用カテゴリ値連続バッファを返す。
   *
   * <p>返り値は内部実体となる配列である。通常の更新には{@link #writeRow}を使う。
   *
   * @return 行ごとの選択された格納位置配列
   */
  public int[] categoricalSlab() {
    return categoricalSlab;
  }

  /**
   * 永続化・一括転送用数値連続バッファを返す。
   *
   * <p>返り値は内部実体となる配列である。通常の更新には型付きAPIを使う。
   *
   * @return 標準形式の順の内部float配列
   */
  public float[] numericSlab() {
    return numericSlab;
  }

  /**
   * 指定行数を転送するときのカテゴリ値要素数を返す。
   *
   * @param rowCount 容量以下の正の行数
   * @return 選択された格納位置要素数
   */
  public int categoricalElementCount(int rowCount) {
    requireRowCount(rowCount);
    return rowCount;
  }

  /**
   * 指定行数を転送するときの数値要素数を返す。
   *
   * @param rowCount 容量以下の正の行数
   * @return 方策二本と行教師値を合計した要素数
   */
  public int numericElementCount(int rowCount) {
    requireRowCount(rowCount);
    return rowCount * (actionCapacity * 2 + ROW_NUMERIC_STRIDE);
  }

  /**
   * 一行の教師信号を検証して標準形式の連続バッファへ書き込む。
   *
   * @param row 容量内の行インデックス
   * @param legalActionCount パディング前の合法行動数
   * @param chosenSlot 実際に選択された合法行動候補の位置
   * @param behaviorPolicy 対局中の行動履歴生成時の最終的な行動の確率分布
   * @param rolloutPolicy 対局生成または基準時点として記録する最終的な行動の確率分布
   * @param valueTarget 期待効用のスカラー教師
   * @param advantage 選択行動のスカラー効用アドバンテージ
   * @param actorWeight 選択された方策損失へ掛ける重み
   * @param sampleWeight 方策・価値共通のサンプル重み
   */
  public void writeRow(
      int row,
      int legalActionCount,
      int chosenSlot,
      float[] behaviorPolicy,
      float[] rolloutPolicy,
      float valueTarget,
      float advantage,
      float actorWeight,
      float sampleWeight) {
    validateRow(
        row,
        legalActionCount,
        chosenSlot,
        behaviorPolicy,
        rolloutPolicy,
        valueTarget,
        advantage,
        actorWeight,
        sampleWeight);

    writeTrustedRow(
        row,
        legalActionCount,
        chosenSlot,
        behaviorPolicy,
        rolloutPolicy,
        valueTarget,
        advantage,
        actorWeight,
        sampleWeight);
  }

  /** 指定行の分岐比較教師値を書き込む。 */
  public void writeBranchTarget(int row, DecisionBranchTarget target) {
    target.writeTo(numericSlab, rowNumericOffset(row) + BRANCH_OFFSET);
  }

  /** 検証済み教師値を追加走査せず格納する内部経路。 */
  void writeTrustedRow(
      int row,
      int legalActionCount,
      int chosenSlot,
      float[] behaviorPolicy,
      float[] rolloutPolicy,
      float valueTarget,
      float advantage,
      float actorWeight,
      float sampleWeight) {
    categoricalSlab[row] = chosenSlot;
    System.arraycopy(behaviorPolicy, 0, numericSlab, behaviorPolicyOffset(row), legalActionCount);
    System.arraycopy(rolloutPolicy, 0, numericSlab, rolloutPolicyOffset(row), legalActionCount);
    int rowOffset = rowNumericOffset(row);
    numericSlab[rowOffset] = valueTarget;
    numericSlab[rowOffset + ADVANTAGE_COMPONENT] = advantage;
    numericSlab[rowOffset + ACTOR_WEIGHT_COMPONENT] = actorWeight;
    numericSlab[rowOffset + SAMPLE_WEIGHT_COMPONENT] = sampleWeight;
    DecisionBranchTarget.NONE.writeTo(numericSlab, rowOffset + BRANCH_OFFSET);
  }

  /** 状態符号化前に、一行の教師データが宣言容量区分と整合することを検証する。 */
  void validateRow(
      int row,
      int legalActionCount,
      int chosenSlot,
      float[] behaviorPolicy,
      float[] rolloutPolicy,
      float valueTarget,
      float advantage,
      float actorWeight,
      float sampleWeight) {
    requireRow(row);
    if (legalActionCount < 1 || legalActionCount > actionCapacity) {
      throw new IllegalArgumentException("legalActionCount out of range: " + legalActionCount);
    }
    if (chosenSlot < 0 || chosenSlot >= legalActionCount) {
      throw new IllegalArgumentException("chosenSlot outside legal actions: " + chosenSlot);
    }
    requireLength(behaviorPolicy.length, legalActionCount, "behaviorPolicy");
    requireLength(rolloutPolicy.length, legalActionCount, "rolloutPolicy");
    requireFinite(valueTarget, "valueTarget");
    requireFinite(advantage, "advantage");
    requirePolicy(behaviorPolicy, actorWeight > 0.0f, "behaviorPolicy");
    requirePolicy(rolloutPolicy, actorWeight > 0.0f, "rolloutPolicy");
    requireFinite(actorWeight, "actorWeight");
    requireFinite(sampleWeight, "sampleWeight");
  }

  /**
   * 指定行で実際に選択された行動候補の位置を返す。
   *
   * @param row 容量内の行インデックス
   * @return 0始まりの選択された格納位置
   */
  public int chosenSlot(int row) {
    return categoricalSlab[row];
  }

  /**
   * 指定行の方策損失重みを返す。
   *
   * @param row 容量内の行インデックス
   * @return 方策学習の重み
   */
  public float actorWeight(int row) {
    return rowNumeric(row, ACTOR_WEIGHT_COMPONENT);
  }

  /**
   * 指定行の方策損失重みを更新する。
   *
   * @param row 容量内の行インデックス
   * @param value 有限な方策学習の重み
   */
  public void setActorWeight(int row, float value) {
    setRowNumeric(row, ACTOR_WEIGHT_COMPONENT, value, "actorWeight");
  }

  /**
   * 指定行の共通サンプル重みを返す。
   *
   * @param row 容量内の行インデックス
   * @return サンプル重み
   */
  public float sampleWeight(int row) {
    return rowNumeric(row, SAMPLE_WEIGHT_COMPONENT);
  }

  /**
   * 指定行の共通サンプル重みを更新する。
   *
   * @param row 容量内の行インデックス
   * @param value 有限なサンプル重み
   */
  public void setSampleWeight(int row, float value) {
    setRowNumeric(row, SAMPLE_WEIGHT_COMPONENT, value, "sampleWeight");
  }

  /** 指定行のスカラー効用教師値を返す。 */
  public float valueTarget(int row) {
    return rowNumeric(row, 0);
  }

  /** 指定行のスカラー選択行動アドバンテージを返す。 */
  public float advantage(int row) {
    return rowNumeric(row, ADVANTAGE_COMPONENT);
  }

  void copyTrustedRowFrom(DecisionTrainingTargets source, int sourceRow, int destinationRow) {
    if (actionCapacity != source.actionCapacity) {
      throw new IllegalArgumentException("target action capacity mismatch");
    }
    categoricalSlab[destinationRow] = source.categoricalSlab[sourceRow];
    System.arraycopy(
        source.numericSlab,
        source.behaviorPolicyOffset(sourceRow),
        numericSlab,
        behaviorPolicyOffset(destinationRow),
        actionCapacity);
    System.arraycopy(
        source.numericSlab,
        source.rolloutPolicyOffset(sourceRow),
        numericSlab,
        rolloutPolicyOffset(destinationRow),
        actionCapacity);
    System.arraycopy(
        source.numericSlab,
        source.rowNumericOffset(sourceRow),
        numericSlab,
        rowNumericOffset(destinationRow),
        ROW_NUMERIC_STRIDE);
  }

  void copyCategoriesTo(int fromInclusive, int rowCount, IntBuffer destination) {
    requireRange(fromInclusive, rowCount);
    destination.put(categoricalSlab, fromInclusive, rowCount);
  }

  void copyNumericsTo(int fromInclusive, int rowCount, FloatBuffer destination) {
    requireRange(fromInclusive, rowCount);
    destination.put(numericSlab, behaviorPolicyOffset(fromInclusive), rowCount * actionCapacity);
    destination.put(numericSlab, rolloutPolicyOffset(fromInclusive), rowCount * actionCapacity);
    destination.put(numericSlab, rowNumericOffset(fromInclusive), rowCount * ROW_NUMERIC_STRIDE);
  }

  /** Decision 効用教師値と重みを行ごとのコンパクト表現へ書く。 */
  void copyValueTargetsTo(int fromInclusive, int rowCount, FloatBuffer destination) {
    requireRange(fromInclusive, rowCount);
    for (int row = fromInclusive; row < fromInclusive + rowCount; row++) {
      destination.put(numericSlab, rowNumericOffset(row), VALUE_TARGET_SIZE);
      destination.put(numericSlab[rowNumericOffset(row) + SAMPLE_WEIGHT_COMPONENT]);
    }
  }

  double sampleWeightMass(int fromInclusive, int rowCount) {
    requireRange(fromInclusive, rowCount);
    double total = 0.0;
    for (int row = fromInclusive; row < fromInclusive + rowCount; row++) {
      total += numericSlab[rowNumericOffset(row) + SAMPLE_WEIGHT_COMPONENT];
    }
    if (!(total > 0.0) || !Double.isFinite(total)) {
      throw new IllegalArgumentException("Pretrain batch sample weight mass must be positive");
    }
    return total;
  }

  private int behaviorPolicyOffset(int row) {
    return behaviorPolicyOffset + row * actionCapacity;
  }

  private int rolloutPolicyOffset(int row) {
    return rolloutPolicyOffset + row * actionCapacity;
  }

  private int rowNumericOffset(int row) {
    return rowNumericOffset + row * ROW_NUMERIC_STRIDE;
  }

  private float rowNumeric(int row, int component) {
    return numericSlab[rowNumericOffset(row) + component];
  }

  private void setRowNumeric(int row, int component, float value, String label) {
    requireFinite(value, label);
    numericSlab[rowNumericOffset(row) + component] = value;
  }

  private void requireRow(int row) {
    if (row < 0 || row >= capacity) {
      throw new IndexOutOfBoundsException("row=" + row + " capacity=" + capacity);
    }
  }

  private void requireRowCount(int rowCount) {
    if (rowCount < 1 || rowCount > capacity) {
      throw new IllegalArgumentException("rowCount out of range: " + rowCount);
    }
  }

  private void requireRange(int fromInclusive, int rowCount) {
    if (fromInclusive < 0 || rowCount < 1 || fromInclusive + rowCount > capacity) {
      throw new IndexOutOfBoundsException(
          "target row range outside capacity: from="
              + fromInclusive
              + " count="
              + rowCount
              + " capacity="
              + capacity);
    }
  }

  private static void requirePolicy(float[] policy, boolean requireFullSupport, String label) {
    double sum = 0.0;
    for (int slot = 0; slot < policy.length; slot++) {
      float probability = policy[slot];
      if (!Float.isFinite(probability) || probability < 0.0f) {
        throw new IllegalArgumentException(label + " has an invalid probability at slot " + slot);
      }
      if (requireFullSupport && probability == 0.0f) {
        throw new IllegalArgumentException(label + " lacks full support at slot " + slot);
      }
      sum += probability;
    }
    if (Math.abs(sum - 1.0) > 1.0e-4) {
      throw new IllegalArgumentException(label + " is not normalized: " + sum);
    }
  }

  private static void requireFinite(float value, String label) {
    if (!Float.isFinite(value)) {
      throw new IllegalArgumentException(label + " must be finite: " + value);
    }
  }

  private static void requireLength(int actual, int expected, String label) {
    if (actual != expected) {
      throw new IllegalArgumentException(label + " length must be " + expected + ": " + actual);
    }
  }
}
