package com.epsilon.grp.train;

import com.epsilon.grp.GrpEncoder;
import com.epsilon.grp.GrpLayout;
import com.epsilon.grp.GrpPhase;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpState;
import java.util.Arrays;
import java.util.Objects;

/**
 * GRP の学習の行を、ネットワークに入れるバッチに詰める。配列は作るときに 1 度だけ確保し、バッチごとに使い回す。行 i の値は次の
 * 位置に置く。入力は {@code inputs[i × width ...]}、局の段階で選んだ出力ヘッドの印は {@code headMask[i × headCount + ヘッド]}、
 * 目標（実際の順列の one-hot）は {@code targets[i × S! ...]}、重みは {@code weights[i]}。
 */
final class RowBatch implements TrainingLoop.Batch<GrpRows> {

  final int width;
  final int headCount;
  final int outputs;
  final float[] inputs;
  final float[] headMask;
  final float[] targets;
  final float[] weights;
  int size;

  private final int capacity;
  private final GrpShape shape;
  private final GrpEncoder encoder;
  private final float[] row;

  /** バッチの配列を確保する。 */
  RowBatch(GrpShape shape, int capacity) {
    this.shape = shape;
    GrpLayout layout = shape.layout();
    encoder = GrpEncoder.of(layout);
    this.capacity = capacity;
    width = layout.width();
    headCount = shape.heads();
    outputs = shape.outputs();
    inputs = new float[capacity * width];
    headMask = new float[capacity * headCount];
    targets = new float[capacity * outputs];
    weights = new float[capacity];
    row = new float[width];
  }

  @Override
  public int capacity() {
    return capacity;
  }

  @Override
  public void clear() {
    Arrays.fill(inputs, 0, size * width, 0f);
    Arrays.fill(headMask, 0, size * headCount, 0f);
    Arrays.fill(targets, 0, size * outputs, 0f);
    size = 0;
  }

  boolean full() {
    return size == capacity;
  }

  /** 卓の行を 1 つ足す。目標は、実際の順列（相対席で数えた順列の番号）の one-hot にする。 */
  void add(GrpState state, int permutation, float weight) {
    Objects.checkIndex(permutation, outputs);
    targets[open(state, weight) * outputs + permutation] = 1f;
  }

  @Override
  public void add(GrpRows rows, int index, float scale) {
    GrpState state = GrpState.ofKey(rows.variant(), rows.keys[index]);
    float weight = rows.weightOf(index) * scale;
    add(state, rows.permutations[index], weight);
  }

  /** 入力、選んだ出力ヘッド、重みを書いて行を 1 つ足し、その行の番号を返す。目標は呼び出し元が書く。 */
  private int open(GrpState state, float weight) {
    if (full()) {
      throw new IllegalStateException("The batch already holds " + capacity + " rows");
    }
    encoder.encode(state, row);
    System.arraycopy(row, 0, inputs, size * width, width);
    int head = shape.phaseHeads() ? GrpPhase.of(shape.variant(), state.round()).ordinal() : 0;
    headMask[size * headCount + head] = 1f;
    weights[size] = weight;
    return size++;
  }
}
