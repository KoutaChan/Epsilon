package com.epsilon.prism.hip;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.hip.HipPlanOptions.SegmentLayout;

/** Candidate count and compact relation attention over a batch of transitions. */
public interface CandidateKernels {
  /** Compact inputs for V4's candidate-by-tile attention. */
  record CandidateInput(
      NDArray query,
      NDArray memory,
      NDArray base,
      NDArray weights,
      NDArray codes,
      NDArray groups,
      int tiles,
      int seats,
      SegmentLayout layout,
      float scale) {
    int[] dimensions() {
      int[] dimensions = new int[8 + SegmentLayout.INTS];
      dimensions[0] = Math.toIntExact(query.getShape().get(0));
      dimensions[1] = 1 + tiles + seats;
      dimensions[2] = tiles;
      dimensions[3] = seats;
      dimensions[4] = Math.toIntExact(base.getShape().get(0));
      dimensions[5] = Math.toIntExact(base.getShape().get(1));
      dimensions[6] = Math.toIntExact(memory.getShape().get(0));
      dimensions[7] = Math.toIntExact(query.getShape().get(1));
      layout.encode(dimensions, 8);
      return dimensions;
    }

    public Shape probabilities() {
      return new Shape(query.getShape().get(0), query.getShape().get(1), 1 + tiles + seats);
    }
  }

  /** Sum the gated BF16 tile projections in FP32, then store the result in BF16. */
  void count(NDArray projected, NDArray gates, NDArray rows, NDArray ids, NDArray output);

  /** Reduce gradients across transitions grouped by row. Write the BF16 projected and FP32 gate gradients. */
  void countGradient(
      NDArray projected,
      NDArray gates,
      NDArray rows,
      NDArray ids,
      NDArray gradient,
      NDArray gradProjected,
      NDArray gradGate);

  /** Compute attention and retain one FP32 probability per transition, head and key. */
  void candidateAttention(CandidateInput input, NDArray output, NDArray probabilities);

  /** Differentiate attention and reduce table gradients by code. Call the optional mark after scratch setup. */
  void candidateAttentionGradient(
      CandidateInput input,
      NDArray probabilities,
      NDArray upstream,
      NDArray gradQuery,
      NDArray gradMemory,
      NDArray gradBase,
      NDArray gradWeights,
      Runnable beforeLaunch);
}
