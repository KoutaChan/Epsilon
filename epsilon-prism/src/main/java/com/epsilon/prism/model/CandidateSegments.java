package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.CandidateCodes;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.kernel.CandidateKernels;
import com.epsilon.prism.kernel.CollectionCapabilities;
import com.epsilon.prism.kernel.CollectionPlanOptions.SegmentLayout;

/**
 * 各区分の u8 コードで独立した表を引き、既存のキー・値の補正に加えてから同じ Attention 演算に渡す。
 * 各表の列は前半がキー、後半が値。
 */
final class CandidateSegments extends Module {
  private final InputFormat format;
  private final CandidateCodes layout;
  private final SegmentLayout hipLayout;
  private final Parameter[] tables;
  private final int width;

  CandidateSegments(PrismShape shape, InputFormat input) {
    format = input;
    layout = input.candidateCodes();
    hipLayout = layout.hip();
    width = 2 * shape.attention();
    tables = new Parameter[layout.slots().size()];
    for (int i = 0; i < tables.length; i++) {
      var slot = layout.slots().get(i);
      tables[i] = table(slot.parameter(), slot.rows(), width);
    }
  }

  public boolean fusedAttention(Context context) {
    var tape = context.tape();
    return CollectionCapabilities.supports(width, true, false)
        && tape != null
        && tape.fusedCandidateAttention();
  }

  public NDArray attention(
      Context context,
      NetworkInputs inputs,
      BatchIndex index,
      NDArray query,
      NDArray memory,
      NDArray base) {
    var tape = context.tape();
    if (tape == null) return null;
    var args =
        new CandidateKernels.CandidateInput(
            context.toFloat(query),
            context.toFloat(memory),
            base,
            weights(context),
            inputs.get(Tensor.TRANSITION_CODE),
            index.transitionRows.array(),
            format.tiles(),
            format.seats(),
            hipLayout,
            (float) EncoderLayer.SCALE);
    return tape.candidateAttention(context.working(), args);
  }

  private NDArray weights(Context context) {
    NDList values = new NDList();
    for (Parameter parameter : tables) {
      values.add(context.fp32(parameter));
    }
    return context.own(NDArrays.concat(values, 0));
  }

  public CandidateLayer.Offsets offsets(Context context, NetworkInputs inputs, NDArray base) {
    int transitions = inputs.transitions();
    int keys = format.fixedTokens();
    int tiles = format.tiles();
    NDArray encoded = inputs.get(Tensor.TRANSITION_CODE);
    NDArray codes = encoded.toType(DataType.INT32, false);
    NDArray original =
        NDArrays.gatherRows(base, codes.get(new NDIndex(":, :{}", keys)).reshape(-1))
            .reshape(transitions, keys, width);
    NDArray tileOffsets = original.get(new NDIndex(":, {}:", keys - tiles));
    for (int seg = 0; seg < tables.length; seg++) {
      int start = layout.slots().get(seg).column();
      NDArray ids = codes.get(new NDIndex(":, {}:{}", start, start + tiles)).reshape(-1);
      tileOffsets =
          tileOffsets.add(context.rows(tables[seg], ids).reshape(transitions, tiles, width));
    }
    NDArray fixed = original.get(new NDIndex(":, :{}", keys - tiles));
    NDArray table = fixed.concat(tileOffsets, 1).reshape(-1, width);
    NDArray ids = context.working().arange(transitions * keys).reshape(transitions, keys);
    return new CandidateLayer.Offsets(table, ids);
  }
}
