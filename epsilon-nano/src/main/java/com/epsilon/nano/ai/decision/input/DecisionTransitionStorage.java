package com.epsilon.nano.ai.decision.input;

import com.epsilon.engine.EngineDecisionBuffer;
import java.util.Arrays;

/** 推論の各行に実在する遷移だけを保持し、行動ごとの候補位置を連続した格納領域へ対応付ける。 */
final class DecisionTransitionStorage {

  private static final int[] REGION_STRIDES = new int[DecisionInputLayout.Tensor.values().length];
  private static final int CATEGORY_STRIDE;
  private static final int NUMERIC_STRIDE;

  static {
    int categories = 0;
    int numerics = 0;
    for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
      if (!tensor.isTransition()) {
        continue;
      }
      boolean categorical = tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL;
      REGION_STRIDES[tensor.ordinal()] = categorical ? categories : numerics;
      if (categorical) {
        categories += tensor.elementsPerTransition();
      } else {
        numerics += tensor.elementsPerTransition();
      }
    }
    CATEGORY_STRIDE = categories;
    NUMERIC_STRIDE = numerics;
  }

  private final int rowStride;
  private final int[] actionOffsets;
  private final int[] actionCounts;
  private final short[][] categories;
  private final float[][] numerics;

  DecisionTransitionStorage(int rows, int actions) {
    this(rows, actions, new Buffers());
  }

  DecisionTransitionStorage(int rows, int actions, Buffers buffers) {
    rowStride = actions + 1;
    buffers.prepare(rows, Math.multiplyExact(rows, rowStride));
    actionOffsets = buffers.actionOffsets;
    actionCounts = buffers.actionCounts;
    categories = buffers.categories;
    numerics = buffers.numerics;
  }

  /** 行ごとの遷移配列とメタデータを保持し、借用ごとに変わる行幅から独立させる。 */
  static final class Buffers {
    private int[] actionOffsets = new int[0];
    private int[] actionCounts = new int[0];
    private short[][] categories = new short[0][];
    private float[][] numerics = new float[0][];

    private void prepare(int rows, int offsets) {
      if (actionOffsets.length < offsets) actionOffsets = new int[offsets];
      if (actionCounts.length < rows) actionCounts = new int[rows];
      if (categories.length < rows) {
        categories = Arrays.copyOf(categories, rows);
        numerics = Arrays.copyOf(numerics, rows);
      }
    }

    long byteCount() {
      long bytes =
          64L
              + Integer.BYTES * ((long) actionOffsets.length + actionCounts.length)
              + 2L * Long.BYTES * categories.length;
      for (int row = 0; row < categories.length; row++) {
        if (categories[row] != null) bytes += 16L + (long) Short.BYTES * categories[row].length;
        if (numerics[row] != null) bytes += 16L + (long) Float.BYTES * numerics[row].length;
      }
      return bytes;
    }
  }

  void prepare(int row, EngineDecisionBuffer decision) {
    int base = row * rowStride;
    int count = 0;
    for (int action = 0; action < rowStride - 1; action++) {
      actionOffsets[base + action] = count;
      if (action < decision.actionCount()) {
        count += decision.transitionCount(action);
      }
    }
    actionOffsets[base + rowStride - 1] = count;
    int categoryCount = Math.multiplyExact(count, CATEGORY_STRIDE);
    int numericCount = Math.multiplyExact(count, NUMERIC_STRIDE);
    if (categories[row] == null || categories[row].length < categoryCount)
      categories[row] = new short[categoryCount];
    else Arrays.fill(categories[row], 0, categoryCount, (short) 0);
    if (numerics[row] == null || numerics[row].length < numericCount)
      numerics[row] = new float[numericCount];
    else Arrays.fill(numerics[row], 0, numericCount, 0f);
  }

  void commitActionCount(int row, int actions) {
    actionCounts[row] = actions;
  }

  int actionCount(int row) {
    return actionCounts[row];
  }

  int[] actionOffsets() {
    return actionOffsets;
  }

  int rowBase(int row) {
    return row * rowStride;
  }

  int count(int row) {
    return actionOffsets[(row + 1) * rowStride - 1];
  }

  int count(int row, int action) {
    int offset = rowBase(row) + action;
    return actionOffsets[offset + 1] - actionOffsets[offset];
  }

  int regionBase(int row, DecisionInputLayout.Tensor tensor) {
    return REGION_STRIDES[tensor.ordinal()] * count(row);
  }

  int index(int row, int action, int transition) {
    return actionOffsets[rowBase(row) + action] + transition;
  }

  short[] categories(int row) {
    return categories[row];
  }

  float[] numerics(int row) {
    return numerics[row];
  }
}
