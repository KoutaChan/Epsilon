package com.epsilon.prism.kernel;

import com.epsilon.prism.kernel.KernelProvider.Backend;
import com.epsilon.prism.kernel.KernelProvider.Capabilities;
import com.epsilon.prism.kernel.KernelProvider.Operation;
import com.epsilon.prism.kernel.KernelProvider.Request;
import com.epsilon.prism.kernel.KernelProvider.Use;
import java.util.EnumSet;
import java.util.Set;

/** 共通の収集カーネルが、指定された宣言の全段を実行できるかを調べる。 */
public final class CollectionKernels {
  private CollectionKernels() {}

  public static Capabilities capabilities(Request request, Backend backend) {
    var m = request.model();
    boolean segments = !m.segments().isEmpty();
    boolean reread = m.stages().contains(ModelSchema.Stage.EVENT_REREAD);
    if (request.backend() != backend
        || request.use() != Use.COLLECTION
        || request.precision() != ModelSchema.Element.BF16
        || !CollectionCapabilities.supports(m.hidden(), segments, reread)
        || m.headWidth() != CollectionArchitecture.HEAD_WIDTH
        || m.heads() * (long) m.headWidth() != m.hidden() / 2
        || m.layers().size() < CollectionArchitecture.MIN_DEPTH
        || m.layers().size() > CollectionArchitecture.MAX_DEPTH)
      return new Capabilities(
          Set.of(), "The compiled collection kernels do not support this model, use or precision");
    boolean causal = m.layers().getFirst().mask() == ModelSchema.Mask.PUBLIC_PREFIX;
    if (!m.stages().equals(CollectionOrder.stages(segments, reread))
        || !m.arithmetic()
            .equals(
                CollectionOrder.arithmetic(
                    CollectionCapabilities.foldsLayerZero(m.hidden()), causal, segments))
        || m.layers().stream().skip(1).anyMatch(layer -> layer.mask() != ModelSchema.Mask.ALL_KEYS))
      return new Capabilities(
          Set.of(), "The collection plan cannot reproduce the declared stages, mask or arithmetic");
    if (m.activation() != ModelSchema.Element.BF16
        || m.accumulation() != ModelSchema.Element.FP32
        || m.score() != ModelSchema.Element.FP32
        || m.layers().stream().anyMatch(layer -> layer.keys() != ModelSchema.Queries.ALL_TOKENS))
      return new Capabilities(
          Set.of(), "Collection requires BF16 activations, FP32 accumulation and all keys");
    var layers = m.layers();
    boolean fixed = false;
    for (var layer : layers) {
      boolean all = layer.queries() == ModelSchema.Queries.ALL_TOKENS;
      if ((all && (fixed || layer.ffn() == ModelSchema.FeedForward.BY_TOKEN))
          || layer.contextWidth() % FeedForwardShape.CHUNK != 0
          || layer.tileWidth() % FeedForwardShape.CHUNK != 0)
        return new Capabilities(Set.of(), "Unsupported backbone query order or FFN tile width");
      fixed |= !all;
    }
    long gates =
        m.roles().stream()
            .filter(
                role ->
                    switch (role.composition()) {
                      case DISCARD_VALUE, ZERO -> false;
                      default -> true;
                    })
            .count();
    if (layers.getFirst().queries() != ModelSchema.Queries.ALL_TOKENS
        || !fixed
        || !CollectionCapabilities.supportsRule(m.seats(), m.tiles())
        || (long) m.seats() + m.tiles() + 1 + m.maxEvents() > CollectionArchitecture.MAX_ROW_TOKENS
        || (long) m.keyCodes() + 1 + m.tileRelations() > CollectionArchitecture.MAX_CODES
        || m.gateWidth() % CollectionArchitecture.GATE_STEP != 0
        || m.candidateWidth() % FeedForwardShape.CHUNK != 0
        || m.roles().size() > CollectionArchitecture.MAX_ROLES
        || gates == 0
        || (causal && reread)
        || (segments
            && (m.countWidth() != CollectionCapabilities.COUNT_WIDTH
                || m.tileKeyFirstCode() >= m.keyCodes())))
      return new Capabilities(
          Set.of(), "Unsupported collection rule, candidate layout or role count");
    if (m.inputs().size() != ModelSchema.Section.values().length)
      return new Capabilities(Set.of(), "collection requires the published input sections");
    for (int i = 0; i < m.inputs().size(); i++) {
      var input = m.inputs().get(i);
      var section = ModelSchema.Section.values()[i];
      if (!section.name().equals(input.name())
          || section.axis() != input.axis()
          || section.element() != input.element())
        return new Capabilities(Set.of(), "Incompatible collection input section: " + input.name());
    }
    var ops =
        EnumSet.of(
            Operation.RELATION_CODES,
            Operation.TOKEN_EMBEDDING,
            Operation.RELATION_ATTENTION,
            Operation.FEED_FORWARD,
            Operation.CANDIDATE_ATTENTION,
            Operation.CANDIDATE_FFN,
            Operation.DISCARD_VALUE,
            Operation.COMPOSE_VALUES,
            Operation.FULL_SUBMISSION);
    if (segments) {
      ops.add(Operation.COUNT_TILES);
      ops.add(Operation.SEGMENT_BIAS);
    }
    if (reread) ops.add(Operation.EVENT_REREAD);
    if (!reread
        && m.layers().getFirst().mask() == ModelSchema.Mask.PUBLIC_PREFIX
        && m.layers().stream()
            .skip(1)
            .allMatch(layer -> layer.queries() == ModelSchema.Queries.FIXED_TOKENS))
      ops.add(Operation.EVENT_PAGES);
    return new Capabilities(ops, "BF16 collection with the published weight layout");
  }
}
