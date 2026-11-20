package com.epsilon.prism.backend;

import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.kernel.KernelProvider.Batch;
import com.epsilon.prism.kernel.KernelProvider.Capacity;
import com.epsilon.prism.kernel.ModelSchema.Model;
import java.nio.ByteBuffer;

/** 入力の各領域を宣言順に並べ、デバイスの計画に渡す。 */
public final class KernelInputs {
  private final Tensor[] sections;

  public KernelInputs(Model model) {
    sections =
        model.inputs().stream().map(input -> Tensor.valueOf(input.name())).toArray(Tensor[]::new);
  }

  public static Capacity capacity(InputFormat format, BatchPlan.Capacity counts) {
    return new Capacity(
        counts.rows(),
        counts.events(),
        counts.transitions(),
        counts.actions(),
        BatchPlan.maxBytes(format, counts));
  }

  public Batch batch(HostBatch input, ByteBuffer scores) {
    var layout = input.plan();
    long[] offsets = new long[sections.length];
    for (int i = 0; i < offsets.length; i++) offsets[i] = layout.offset(sections[i]);
    return new Batch(
        layout.rows(),
        layout.count(Axis.EVENT),
        layout.count(Axis.TRANSITION),
        layout.count(Axis.ACTION),
        offsets,
        input.storage().slice(0, layout.bytes()),
        scores);
  }
}
