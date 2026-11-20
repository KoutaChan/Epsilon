package com.epsilon.prism.hip;

import ai.djl.Device;
import com.epsilon.prism.kernel.CollectionKernels;
import com.epsilon.prism.kernel.CollectionLane;
import com.epsilon.prism.kernel.ModelSchema;
import com.epsilon.prism.kernel.TrainingKernels;
import com.epsilon.prism.kernel.TrainingProvider;
import java.util.Set;

/** 配布済みの pack と専用カーネルで、指定したストリームに収集バッチを提出する。 */
public final class HipCollection implements TrainingProvider {
  @Override
  public Backend backend() {
    return Backend.HIP;
  }

  @Override
  public int abiVersion() {
    return ModelSchema.VERSION;
  }

  /** 選んだ共有ライブラリが配布物に無ければ、収集でも学習でも演算を返さない。 */
  @Override
  public Capabilities capabilities(Request request) {
    if (request.use() == Use.TRAINING) return trainingCapabilities(request, HipResources.bundled());
    if (!HipResources.bundled())
      return new Capabilities(Set.of(), HipResources.selected() + " is not bundled with this jar");
    return CollectionKernels.capabilities(request, backend());
  }

  @Override
  public TrainingKernels openTraining(Request request, long stream) {
    if (request.use() != Use.TRAINING
        || !capabilities(request).operations().contains(Operation.RELATION_ATTENTION_BACKWARD))
      throw new IllegalArgumentException("Unsupported HIP training request");
    var library = HipLibrary.load();
    return HipAttention.on(library, Device.gpu(request.device()), stream);
  }

  @Override
  public Plan open(Request request, long stream) {
    if (!capabilities(request).operations().contains(Operation.FULL_SUBMISSION))
      throw new IllegalArgumentException("Unsupported HIP collection request");
    var library = HipLibrary.load();
    library.requireArchitecture(request.device());
    return new CollectionLane(request, stream, PrismHip.COLLECTION);
  }
}
