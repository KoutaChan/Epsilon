package com.epsilon.major.ai.decision.input;

import com.epsilon.engine.EngineDecisionBuffer;

/** 各推論行に存在する遷移を保持し、行動内の候補位置を連続領域へ対応付ける。 */
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
    rowStride = actions + 1;
    actionOffsets = new int[rows * rowStride];
    actionCounts = new int[rows];
    categories = new short[rows][];
    numerics = new float[rows][];
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
    categories[row] = new short[count * CATEGORY_STRIDE];
    numerics[row] = new float[count * NUMERIC_STRIDE];
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
