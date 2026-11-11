package com.epsilon.grp.train;

import com.epsilon.grp.GrpEncoder;
import com.epsilon.grp.GrpLayout;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.OutcomeShape;

/**
 * OutcomeNetwork の学習に使う行を、入力のバッチに詰める。配列は作るときに 1 度だけ確保し、バッチごとに使い回す。行を足すときに、
 * その行の値を全て書き直す。行 i の入力は {@code inputs[i × width ...]} に、重みは {@code weights[i]} に置く。行 i の結果 k の項の
 * 番号は {@code terms[(i × outcomes + k) × stride ...]} に置く。項の無い所には項の総数を入れる（ネットワークは値が 0 の列を 1 つ
 * 足して読む）。結果の無い k では {@code mask[i × outcomes + k]} を EMPTY に、ほかは 0 にする。
 */
final class OutcomeBatch implements TrainingLoop.Batch<OutcomeRows> {

  /** 結果の無い所の対数確率に足す値。exp すると 0 になり、float32 に収まる。 */
  static final float EMPTY = -1e30f;

  final int width;
  final int outcomes;
  final int stride;
  final int padding;
  final float[] inputs;
  final long[] terms;
  final float[] mask;
  final float[] weights;
  int size;

  private final int capacity;
  private final GrpEncoder encoder;
  private final float[] row;

  /**
   * バッチの配列を確保する。
   *
   * @param outcomes 1 行の結果の数の上限
   */
  OutcomeBatch(OutcomeShape shape, int capacity, int outcomes) {
    GrpLayout layout = shape.layout();
    encoder = GrpEncoder.of(layout);
    this.capacity = capacity;
    this.outcomes = outcomes;
    stride = shape.outcomes().maxTerms();
    width = layout.width();
    padding = shape.outcomes().terms();
    inputs = new float[capacity * width];
    terms = new long[capacity * outcomes * stride];
    mask = new float[capacity * outcomes];
    weights = new float[capacity];
    row = new float[width];
  }

  @Override
  public int capacity() {
    return capacity;
  }

  @Override
  public void clear() {
    size = 0;
  }

  boolean full() {
    return size == capacity;
  }

  /** 行を 1 つ足す。行の重みには倍率を掛ける。 */
  @Override
  public void add(OutcomeRows rows, int index, float scale) {
    if (full()) {
      throw new IllegalStateException("The batch already holds " + capacity + " rows");
    }
    int count = rows.outcomes(index);
    if (count > outcomes) {
      throw new IllegalArgumentException("Row " + index + " has " + count + " outcomes, the batch holds " + outcomes);
    }
    encoder.encode(GrpState.ofKey(rows.variant(), rows.keys[index]), row);
    System.arraycopy(row, 0, inputs, size * width, width);
    int first = rows.firstOutcome[index];
    for (int k = 0; k < outcomes; k++) {
      int at = (size * outcomes + k) * stride;
      for (int j = 0; j < stride; j++) {
        short term = k < count ? rows.terms[(first + k) * stride + j] : -1;
        terms[at + j] = term >= 0 ? term : padding;
      }
      mask[size * outcomes + k] = k < count ? 0f : EMPTY;
    }
    weights[size] = rows.weightOf(index) * scale;
    size++;
  }
}
