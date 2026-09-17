package com.epsilon.major.ai.decision.input;

import ai.djl.ndarray.types.Shape;

/**
 * 推論用の状態・行動と、存在する遷移だけを二本の連続バッファに配置する。
 *
 * <p>遷移領域の先頭軸はホスト側の存在するインデックス順、残り二軸は1である。密な行動・遷移 容量区分は別に保持し、方策出力の指定位置への配置形状をコンパクトテンソル形状から推測しない。
 */
final class DecisionInferenceInputLayout {

  /** 推論連続バッファ内の一論理的なテンソル区間。 */
  record Region(DecisionInputLayout.Tensor tensor, int slabOffset, int elementCount, Shape shape) {}

  private final int rows;
  private final DecisionBucket bucket;
  private final int presentCount;
  private final Region[] regions = new Region[DecisionInputLayout.Tensor.values().length];
  private final int categoricalElementCount;
  private final int numericElementCount;

  DecisionInferenceInputLayout(int rows, DecisionBucket bucket, int presentCount) {
    if (rows < 1 || presentCount < 0) {
      throw new IllegalArgumentException("invalid compact inference dimensions");
    }
    this.rows = rows;
    this.bucket = bucket;
    this.presentCount = presentCount;
    int categoricalCursor = 0;
    int numericCursor = 0;
    for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
      boolean compact = tensor.isTransition();
      int tensorRows = compact ? presentCount : rows;
      int elementsPerEntry =
          compact ? tensor.elementsPerTransition() : tensor.elementsPerRow(bucket);
      int elementCount = Math.multiplyExact(tensorRows, elementsPerEntry);
      int offset =
          tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL ? categoricalCursor : numericCursor;
      Shape shape =
          compact
              ? tensor.shape(presentCount, 1, 1)
              : tensor.shape(rows, bucket.legalActionCapacity(), bucket.actionTransitionCapacity());
      regions[tensor.ordinal()] = new Region(tensor, offset, elementCount, shape);
      if (tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL) {
        categoricalCursor = Math.addExact(categoricalCursor, elementCount);
      } else {
        numericCursor = Math.addExact(numericCursor, elementCount);
      }
    }
    categoricalElementCount = categoricalCursor;
    numericElementCount = numericCursor;
  }

  int rows() {
    return rows;
  }

  DecisionBucket bucket() {
    return bucket;
  }

  int presentCount() {
    return presentCount;
  }

  Region region(DecisionInputLayout.Tensor tensor) {
    return regions[tensor.ordinal()];
  }

  int categoricalElementCount() {
    return categoricalElementCount;
  }

  int numericElementCount() {
    return numericElementCount;
  }

  long inputByteCount() {
    return (long) Short.BYTES * categoricalElementCount + (long) Float.BYTES * numericElementCount;
  }
}
