package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.input.InputFormat;

/** 同じ遷移に対する K のキー・値補正と、ΣC の残差を候補層で併用する。 */
final class CoreCandidate extends Module implements CandidateLayer.Features {
  private final CandidateCount count;
  private final CandidateSegments segments;

  CoreCandidate(PrismShape.V4 shape, InputFormat format) {
    count = addChildBlock("count", new CandidateCount(shape, format));
    segments = addChildBlock("segments", new CandidateSegments(shape, format));
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... ignored) {
    count.initialize(manager, dataType);
    segments.initialize(manager, dataType);
  }

  @Override
  public boolean fusedAttention(Context context) {
    return segments.fusedAttention(context);
  }

  @Override
  public NDArray attention(
      Context context,
      NetworkInputs inputs,
      BatchIndex index,
      NDArray query,
      NDArray memory,
      NDArray base) {
    return segments.attention(context, inputs, index, query, memory, base);
  }

  @Override
  public CandidateLayer.Offsets offsets(Context context, NetworkInputs inputs, NDArray base) {
    return segments.offsets(context, inputs, base);
  }

  @Override
  public NDArray add(
      Context context, NetworkInputs inputs, BatchIndex index, NDArray states, NDArray residual) {
    return count.add(context, inputs, index, states, residual);
  }
}
