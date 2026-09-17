package com.epsilon.nano.ai.decision.input;

/** 学習用の教師値を格納する2本の連続バッファ内の位置を定義する。 */
public final class DecisionTrainingTargetLayout {

  private final int rows;
  private final int actionCapacity;
  private final int behaviorPolicyOffset;
  private final int rolloutPolicyOffset;
  private final int rowNumericOffset;

  public DecisionTrainingTargetLayout(int rows, DecisionBucket bucket) {
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be positive: " + rows);
    }
    this.rows = rows;
    actionCapacity = bucket.legalActionCapacity();
    behaviorPolicyOffset = 0;
    rolloutPolicyOffset = Math.multiplyExact(rows, actionCapacity);
    rowNumericOffset = Math.multiplyExact(rolloutPolicyOffset, 2);
  }

  public int rows() {
    return rows;
  }

  public int actionCapacity() {
    return actionCapacity;
  }

  public int categoricalElementCount() {
    return rows;
  }

  public int numericElementCount() {
    return Math.addExact(
        rowNumericOffset, Math.multiplyExact(rows, DecisionTrainingTargets.ROW_NUMERIC_STRIDE));
  }

  public int behaviorPolicyRowOffset(int row) {
    requireRow(row);
    return behaviorPolicyOffset + row * actionCapacity;
  }

  public int rolloutPolicyRowOffset(int row) {
    requireRow(row);
    return rolloutPolicyOffset + row * actionCapacity;
  }

  public int rowNumericOffset(int row) {
    requireRow(row);
    return rowNumericOffset + row * DecisionTrainingTargets.ROW_NUMERIC_STRIDE;
  }

  public int actorWeightComponent() {
    return DecisionTrainingTargets.VALUE_TARGET_SIZE + DecisionTrainingTargets.ADVANTAGE_SIZE;
  }

  public int sampleWeightComponent() {
    return actorWeightComponent() + 1;
  }

  private void requireRow(int row) {
    if (row < 0 || row >= rows) {
      throw new IndexOutOfBoundsException("row=" + row + " rows=" + rows);
    }
  }
}
