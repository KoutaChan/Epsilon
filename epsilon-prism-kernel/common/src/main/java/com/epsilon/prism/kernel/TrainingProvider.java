package com.epsilon.prism.kernel;

import java.util.EnumSet;
import java.util.Set;

/** 学習用の関係 Attention と正規化の前後伝播を、学習レーンのストリームで実行する。 */
public interface TrainingProvider extends KernelProvider {
  TrainingKernels openTraining(Request request, long stream);

  /** 指定したモデルで呼べる学習演算を返す。 */
  default Capabilities trainingCapabilities(Request request, boolean bundled) {
    return trainingCapabilities(request, backend(), bundled);
  }

  /** 指定したバックエンドとモデルで呼べる学習演算を返す。 */
  static Capabilities trainingCapabilities(Request request, Backend backend, boolean bundled) {
    var m = request.model();
    boolean segments = !m.segments().isEmpty();

    if (request.use() != Use.TRAINING
        || request.backend() != backend
        || !bundled
        || request.precision() != ModelSchema.Element.BF16
        || m.headWidth() != RelationAttention.FEATURES
        || m.heads() % 2 != 0
        || (long) m.seats() + m.tiles() + 1 + m.maxEvents() > RelationAttention.MAX_TOKENS
        || m.hidden() % 8 != 0
        || m.hidden() > RmsNormKernels.MAX_WIDTH
        || (segments && !CollectionCapabilities.supports(m.hidden(), true, false)))
      return new Capabilities(
          Set.of(), backend + " training kernels are not available for this request");
    var available =
        EnumSet.of(
            Operation.RELATION_CODES,
            Operation.RELATION_ATTENTION,
            Operation.RELATION_ATTENTION_BACKWARD,
            Operation.RMS_NORM,
            Operation.RMS_NORM_BACKWARD);
    if (segments) {
      available.add(Operation.COUNT_TILES);
      available.add(Operation.COUNT_TILES_BACKWARD);
      available.add(Operation.CANDIDATE_ATTENTION);
      available.add(Operation.SEGMENT_ATTENTION_BACKWARD);
    }
    return new Capabilities(available, backend + " attention and normalization tape");
  }
}
