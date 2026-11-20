package com.epsilon.prism.hip;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.prism.kernel.TensorTrainingKernels;
import java.util.Objects;

/** HIP の学習演算を指定したストリームに提出する。 */
public final class HipAttention extends TensorTrainingKernels {
  private HipAttention(long stream, Device device) {
    super(stream, device);
  }

  /**
   * 演算を GPU のストリームに積む実装を作る。
   *
   * @param library 読み込み済みの libprism_hip.so
   * @param stream 演算を積むストリーム（呼び出し元の DJL の演算と同じもの）
   * @return 作った実装
   * @throws IllegalArgumentException ストリームが GPU のものでないとき
   */
  public static HipAttention on(HipLibrary library, PtStream stream) {
    Objects.requireNonNull(library, "library");
    if (!stream.getDevice().isGpu()) {
      throw new IllegalArgumentException("HIP attention needs a GPU stream");
    }
    return on(library, stream.getDevice(), stream.getId());
  }

  static HipAttention on(HipLibrary library, Device device, long stream) {
    library.requireArchitecture(device.getDeviceId());
    return new HipAttention(stream, device);
  }

  @Override
  protected long contiguousHandle(NDArray array) {
    if (!(array instanceof PtNDArray tensor) || !JniUtils.isContiguous(tensor))
      throw new IllegalArgumentException("Training requires contiguous PyTorch tensors");
    return tensor.getHandle();
  }

  @Override
  protected void relationCodes(
      int device,
      long stream,
      int rows,
      int events,
      int tokens,
      int seats,
      int tiles,
      int rowWidth,
      int rowFirstEvent,
      int rowEvents,
      int width,
      int tile,
      int owner,
      int event,
      int riichi,
      int lastDiscard,
      int link,
      int safety,
      int safeties,
      long relation,
      long rowColumn,
      long tileRelation,
      long codes) {
    PrismHip.relationCodes(
        device,
        stream,
        rows,
        events,
        tokens,
        seats,
        tiles,
        rowWidth,
        rowFirstEvent,
        rowEvents,
        width,
        tile,
        owner,
        event,
        riichi,
        lastDiscard,
        link,
        safety,
        safeties,
        relation,
        rowColumn,
        tileRelation,
        codes);
  }

  @Override
  protected void attentionForward(
      int device,
      long stream,
      int rows,
      int queries,
      int keys,
      int codeQueries,
      int heads,
      int words,
      int segments,
      int entries,
      float scale,
      int causalFixedTokens,
      long query,
      long keyValue,
      long mask,
      long codes,
      long table,
      long out,
      long outFloat,
      long logSumExp) {
    PrismHip.attentionForward(
        device,
        stream,
        rows,
        queries,
        keys,
        codeQueries,
        heads,
        words,
        segments,
        entries,
        scale,
        causalFixedTokens,
        query,
        keyValue,
        mask,
        codes,
        table,
        out,
        outFloat,
        logSumExp);
  }

  @Override
  protected void attentionBackward(
      int device,
      long stream,
      int rows,
      int queries,
      int keys,
      int codeQueries,
      int heads,
      int words,
      int segments,
      int entries,
      float scale,
      int causalFixedTokens,
      long query,
      long keyValue,
      long mask,
      long codes,
      long table,
      long outFloat,
      long logSumExp,
      long gradOut,
      long gradQuery,
      long gradKeyValue,
      long gradTable) {
    PrismHip.attentionBackward(
        device,
        stream,
        rows,
        queries,
        keys,
        codeQueries,
        heads,
        words,
        segments,
        entries,
        scale,
        causalFixedTokens,
        query,
        keyValue,
        mask,
        codes,
        table,
        outFloat,
        logSumExp,
        gradOut,
        gradQuery,
        gradKeyValue,
        gradTable);
  }

  @Override
  protected void normForward(
      int device,
      long stream,
      int rows,
      int width,
      float epsilon,
      long input,
      long gain,
      long out,
      long inverseRms) {
    PrismHip.normForward(device, stream, rows, width, epsilon, input, gain, out, inverseRms);
  }

  @Override
  protected void normBackward(
      int device,
      long stream,
      int rows,
      int width,
      float epsilon,
      long input,
      long gain,
      long inverseRms,
      long gradOut,
      long gradInput,
      long gradGain) {
    PrismHip.normBackward(
        device,
        stream,
        rows,
        width,
        epsilon,
        input,
        gain,
        inverseRms,
        gradOut,
        gradInput,
        gradGain);
  }

  @Override
  protected void countForward(
      int device,
      long stream,
      int rows,
      int transitions,
      int tiles,
      int codes,
      int width,
      long projected,
      long gates,
      long rowIndex,
      long ids,
      long output) {
    PrismHip.countForward(
        device,
        stream,
        rows,
        transitions,
        tiles,
        codes,
        width,
        projected,
        gates,
        rowIndex,
        ids,
        output);
  }

  @Override
  protected void countBackward(
      int device,
      long stream,
      int rows,
      int transitions,
      int tiles,
      int codes,
      int width,
      long projected,
      long gates,
      long rowIndex,
      long ids,
      long gradOutput,
      long gradProjected,
      long gradGate) {
    PrismHip.countBackward(
        device,
        stream,
        rows,
        transitions,
        tiles,
        codes,
        width,
        projected,
        gates,
        rowIndex,
        ids,
        gradOutput,
        gradProjected,
        gradGate);
  }

  @Override
  protected void segmentAttentionForward(
      int device,
      long stream,
      int[] dimensions,
      float scale,
      long[] inputs,
      long output,
      long probabilities) {
    PrismHip.segmentAttentionForward(
        device, stream, dimensions, scale, inputs, output, probabilities);
  }

  @Override
  protected void segmentAttentionBackward(
      int device, long stream, int[] dimensions, float scale, long[] inputs, long[] gradients) {
    PrismHip.segmentAttentionBackward(device, stream, dimensions, scale, inputs, gradients);
  }

  @Override
  protected long tensorAddress(long handle, int element, int gpu) {
    return PrismHip.tensorAddress(handle, element, gpu);
  }
}
