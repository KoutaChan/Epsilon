package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.List;

/** 元の候補容量を変えず、同じ種類のテンソルを区分間で連続配置する。 */
final class DecisionGroupedInferenceLayout {
  private final List<DecisionHostBatch.RowSlice> rows;
  private final DecisionInferenceInputLayout[] groups;
  private final int[][] offsets;
  private final int[] actionCounts;
  private final int[] tensorOffsets = new int[DecisionInputLayout.tensorCount()];
  private final int[] tensorCounts = new int[DecisionInputLayout.tensorCount()];
  private final int rowCount;
  private final int transitionCount;
  private final int memoryCount;
  private final int actionCount;
  private final int categories;
  private final int numerics;

  DecisionGroupedInferenceLayout(List<DecisionHostBatch.RowSlice> rows) {
    this.rows = List.copyOf(rows);
    if (rows.isEmpty()) throw new IllegalArgumentException("Inference groups must not be empty");
    groups = new DecisionInferenceInputLayout[rows.size()];
    offsets = new int[rows.size()][DecisionInputLayout.tensorCount()];
    actionCounts = new int[rows.size()];
    int totalRows = 0, transitions = 0, memories = 0, actions = 0;
    for (int index = 0; index < rows.size(); index++) {
      var row = rows.get(index);
      int count = row.transitionPresentCount();
      groups[index] = new DecisionInferenceInputLayout(row.size(), row.bucket(), count);
      totalRows = Math.addExact(totalRows, row.size());
      transitions = Math.addExact(transitions, count);
      memories = Math.addExact(memories, row.playerMemoryPresentCount());
      for (int sourceRow = row.fromInclusive();
          sourceRow < row.fromInclusive() + row.size();
          sourceRow++)
        actionCounts[index] =
            Math.addExact(actionCounts[index], row.source().legalActionCount(sourceRow));
      actions = Math.addExact(actions, actionCounts[index]);
    }
    rowCount = totalRows;
    transitionCount = transitions;
    memoryCount = memories;
    actionCount = actions;
    int categoricalOffset = 0, numericOffset = 0;
    for (var tensor : DecisionInputLayout.tensors()) {
      int start =
          tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL ? categoricalOffset : numericOffset;
      int cursor = start;
      tensorOffsets[tensor.ordinal()] = start;
      for (int group = 0; group < groups.length; group++) {
        offsets[group][tensor.ordinal()] = cursor;
        cursor = Math.addExact(cursor, groups[group].region(tensor).elementCount());
      }
      tensorCounts[tensor.ordinal()] = cursor - start;
      if (tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL) categoricalOffset = cursor;
      else numericOffset = cursor;
    }
    categories = categoricalOffset;
    numerics = numericOffset;
  }

  int categoricalCount() {
    return categories;
  }

  int numericCount() {
    return numerics;
  }

  int indexCount() {
    return Math.addExact(
        memoryCount, Math.multiplyExact(2, Math.addExact(transitionCount, actionCount)));
  }

  void copyTo(ShortBuffer categorical, FloatBuffer numeric, IntBuffer indices) {
    for (var tensor : DecisionInputLayout.tensors()) {
      for (int group = 0; group < groups.length; group++) {
        int offset = offsets[group][tensor.ordinal()];
        rows.get(group)
            .copyInferenceTensorTo(
                tensor,
                tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL
                    ? categorical.position(offset)
                    : null,
                tensor.slab() == DecisionInputLayout.Slab.NUMERIC
                    ? numeric.position(offset)
                    : null);
      }
    }
    int rowOffset = 0, memoryOffset = 0, transitionOffset = 0, actionOffset = 0;
    int actionIndicesOffset = memoryCount + 2 * transitionCount;
    for (int group = 0; group < groups.length; group++) {
      var row = rows.get(group);
      int memories = row.playerMemoryPresentCount();
      int transitions = groups[group].presentCount();
      row.copyPlayerMemoryPresentIndicesTo(rowOffset, indices.position(memoryOffset));
      row.copyTransitionPresentIndicesTo(0, indices.position(memoryCount + transitionOffset));
      int stride = row.bucket().legalActionCapacity() * row.bucket().actionTransitionCapacity();
      for (int index = 0; index < transitions; index++) {
        int local = indices.get(memoryCount + transitionOffset + index);
        indices.put(
            memoryCount + transitionCount + transitionOffset + index, rowOffset + local / stride);
      }
      for (int localRow = 0; localRow < row.size(); localRow++) {
        int count = row.source().legalActionCount(row.fromInclusive() + localRow);
        for (int action = 0; action < count; action++) {
          indices.put(
              actionIndicesOffset + actionOffset,
              localRow * row.bucket().legalActionCapacity() + action);
          indices.put(actionIndicesOffset + actionCount + actionOffset, rowOffset + localRow);
          actionOffset++;
        }
      }
      rowOffset += row.size();
      memoryOffset += memories;
      transitionOffset += transitions;
    }
  }

  DecisionGroupedInferenceBatch bind(NDArray categorical, NDArray numeric, NDArray indices) {
    var result = new ArrayList<DecisionGroupedInferenceBatch.Group>(groups.length);
    int transitionOffset = 0, actionOffset = 0;
    int actionIndicesOffset = memoryCount + 2 * transitionCount;
    for (int group = 0; group < groups.length; group++) {
      NDArray[] views = new NDArray[DecisionInputLayout.tensorCount()];
      for (var tensor : DecisionInputLayout.tensors()) {
        var region = groups[group].region(tensor);
        views[tensor.ordinal()] =
            slice(
                tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL ? categorical : numeric,
                offsets[group][tensor.ordinal()],
                region.elementCount(),
                region.shape());
      }
      int transitions = groups[group].presentCount();
      result.add(
          new DecisionGroupedInferenceBatch.Group(
              DecisionInferenceInputs.fromViews(views, groups[group].bucket()),
              slice(indices, memoryCount + transitionOffset, transitions, new Shape(transitions)),
              slice(
                  indices,
                  actionIndicesOffset + actionOffset,
                  actionCounts[group],
                  new Shape(actionCounts[group])),
              transitionOffset));
      transitionOffset += transitions;
      actionOffset += actionCounts[group];
    }
    return new DecisionGroupedInferenceBatch(
        whole(DecisionInputLayout.Tensor.STATE_CATEGORIES, categorical),
        whole(DecisionInputLayout.Tensor.STATE_NUMERICS, numeric),
        slice(indices, 0, memoryCount, new Shape(memoryCount)),
        new DecisionTransitionInputs(
            whole(DecisionInputLayout.Tensor.TRANSITION_CATEGORIES, categorical),
            whole(DecisionInputLayout.Tensor.TRANSITION_NUMERICS, numeric),
            whole(DecisionInputLayout.Tensor.TRANSITION_TILES, categorical),
            whole(DecisionInputLayout.Tensor.WAIT_TILE_IDS, categorical),
            whole(DecisionInputLayout.Tensor.WAIT_YAKUS, categorical),
            whole(DecisionInputLayout.Tensor.WAIT_SCORES, numeric),
            true),
        slice(
            indices, memoryCount + transitionCount, transitionCount, new Shape(1, transitionCount)),
        slice(indices, actionIndicesOffset + actionCount, actionCount, new Shape(actionCount)),
        result);
  }

  private NDArray whole(DecisionInputLayout.Tensor tensor, NDArray slab) {
    return slice(
        slab,
        tensorOffsets[tensor.ordinal()],
        tensorCounts[tensor.ordinal()],
        tensor.shape(tensor.isTransition() ? transitionCount : rowCount, 1, 1));
  }

  private static NDArray slice(NDArray slab, int offset, int count, Shape shape) {
    return slab.get("{}:{}", offset, offset + count).reshape(shape);
  }
}
