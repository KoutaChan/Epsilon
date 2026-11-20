package com.epsilon.prism.hip;

import java.util.List;

/** Candidate weights and the first-layer mask selected when an inference plan is created. */
public record HipPlanOptions(boolean causal, Count count, Segments segments) {
  public static final HipPlanOptions BASE = new HipPlanOptions(false, null, null);

  public record Count(long offset, int firstCode, int width) {}

  public record Segments(long offset, SegmentLayout layout) {}

  /** A column in the packed input and the matching span in the appended tables. */
  public record SegmentSpan(int column, int firstRow, int rows) {}

  /** Wire description shared by the inference plan and the training launches. */
  public record SegmentLayout(int codeWidth, int tableRows, List<SegmentSpan> spans) {
    // The native descriptor has slots for up to six spans.
    private static final int MAX_SPANS = HipCapabilities.limit("MaxSegmentSpans");
    static final int INTS = 3 + MAX_SPANS * 3;

    public SegmentLayout {
      spans = List.copyOf(spans);
      if (spans.isEmpty() || spans.size() > MAX_SPANS) {
        throw new IllegalArgumentException("invalid number of candidate table spans");
      }
    }

    void encode(int[] target, int at) {
      target[at++] = codeWidth;
      target[at++] = tableRows;
      target[at++] = spans.size();
      for (SegmentSpan span : spans) {
        target[at++] = span.column();
        target[at++] = span.firstRow();
        target[at++] = span.rows();
      }
    }
  }

  /** The same descriptor is used by JNI and the native benchmark. */
  public int[] encode() {
    int flags = (causal ? 1 : 0) | (count == null ? 0 : 2) | (segments == null ? 0 : 4);
    int[] descriptor = new int[5 + SegmentLayout.INTS];
    descriptor[0] = flags;
    if (count != null) {
      descriptor[1] = Math.toIntExact(count.offset());
      descriptor[2] = count.firstCode();
      descriptor[3] = count.width();
    }
    if (segments != null) {
      descriptor[4] = Math.toIntExact(segments.offset());
      segments.layout().encode(descriptor, 5);
    }
    return descriptor;
  }
}
