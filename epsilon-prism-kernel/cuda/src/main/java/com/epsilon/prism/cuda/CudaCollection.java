package com.epsilon.prism.cuda;

import com.epsilon.prism.kernel.CollectionKernels;
import com.epsilon.prism.kernel.CollectionLane;
import com.epsilon.prism.kernel.KernelProvider;
import com.epsilon.prism.kernel.ModelSchema;
import java.nio.ByteBuffer;
import java.util.Set;

/** 配布済みの pack と専用カーネルで、指定したストリームに収集バッチを提出する。 */
public final class CudaCollection extends CollectionLane.Bindings implements KernelProvider {
  @Override
  public Backend backend() {
    return Backend.CUDA;
  }

  @Override
  public int abiVersion() {
    return ModelSchema.VERSION;
  }

  @Override
  public Capabilities capabilities(Request request) {
    var model = CollectionKernels.capabilities(request, backend());
    if (model.operations().isEmpty()) return model;
    try {
      CudaLibrary.ensureLoaded();
      String reason = deviceLimitation(request.device());
      return reason == null ? model : new Capabilities(Set.of(), reason);
    } catch (LinkageError | RuntimeException failure) {
      return new Capabilities(Set.of(), "CUDA device " + request.device() + ": " + failure);
    }
  }

  @Override
  public Plan open(Request request, long stream) {
    var support = capabilities(request);
    if (!support.operations().contains(Operation.FULL_SUBMISSION))
      throw new IllegalArgumentException(support.detail());
    requireStream(request.device(), stream);
    return new CollectionLane(request, stream, this);
  }

  private static native String deviceLimitation(int device);

  private static native void requireStream(int device, long stream);

  @Override
  protected native long declaredPlan(
      int device,
      long stream,
      ByteBuffer declaration,
      int length,
      int abi,
      int countOffset,
      int segmentsOffset,
      int rows,
      int events,
      int transitions,
      int actions,
      long inputBytes,
      long weightBytes);

  @Override
  protected native void freePlan(long plan);

  @Override
  protected native void uploadWeights(
      long plan, ByteBuffer pack, int position, long bytes, long version);

  @Override
  protected native long publishedVersion(long plan);

  @Override
  protected native void configureCandidateInput(long handle, int[] descriptor);

  @Override
  protected native long[] configureEventPages(
      long handle, int remaining, int events, double reserveFraction);

  @Override
  protected native void forward(
      long plan,
      ByteBuffer input,
      int inputPosition,
      int rows,
      int events,
      int transitions,
      int actions,
      long bytes,
      long[] sections,
      ByteBuffer scores,
      int scoresPosition,
      long version);

  @Override
  protected native void forwardPaged(
      long plan,
      ByteBuffer input,
      int inputPosition,
      int rows,
      int events,
      int transitions,
      int actions,
      long bytes,
      long[] sections,
      ByteBuffer scores,
      int scoresPosition,
      long version,
      ByteBuffer eventMap,
      int misses,
      int slots,
      int transferBytes);
}
