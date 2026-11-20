package com.epsilon.prism.model;

import ai.djl.nn.Parameter;
import ai.djl.util.PairList;
import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.prism.input.CandidateCodes;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.kernel.CollectionCapabilities;
import com.epsilon.prism.kernel.CollectionOrder;
import com.epsilon.prism.kernel.FeedForwardShape;
import com.epsilon.prism.kernel.ModelDeclarations;
import com.epsilon.prism.kernel.ModelSchema.Axis;
import com.epsilon.prism.kernel.ModelSchema.Composition;
import com.epsilon.prism.kernel.ModelSchema.Element;
import com.epsilon.prism.kernel.ModelSchema.FeedForward;
import com.epsilon.prism.kernel.ModelSchema.Input;
import com.epsilon.prism.kernel.ModelSchema.Layer;
import com.epsilon.prism.kernel.ModelSchema.Mask;
import com.epsilon.prism.kernel.ModelSchema.Model;
import com.epsilon.prism.kernel.ModelSchema.Normalization;
import com.epsilon.prism.kernel.ModelSchema.Order;
import com.epsilon.prism.kernel.ModelSchema.Queries;
import com.epsilon.prism.kernel.ModelSchema.Role;
import com.epsilon.prism.kernel.ModelSchema.Segment;
import com.epsilon.prism.kernel.ModelSchema.Source;
import com.epsilon.prism.kernel.ModelSchema.Weight;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** 保存形式の形と入力語彙を展開し、登録順の推論パラメーターに結び付ける。 */
final class ModelDeclaration {
  private ModelDeclaration() {}

  static Model of(PrismShape shape, InputFormat format, PairList<String, Parameter> parameters) {
    var inputs = new ArrayList<Input>();
    for (Tensor tensor : Tensor.values()) {
      var layout =
          switch (tensor) {
            case ROUND_CATEGORY -> format.round();
            case SEAT_CATEGORY -> format.seat();
            case TILE_CATEGORY -> format.tile();
            case EVENT_CATEGORY -> format.event();
            case TRANSITION_CATEGORY -> format.transition();
            case RELATION -> format.relation();
            default -> null;
          };
      inputs.add(
          new Input(
              tensor.name(),
              Axis.valueOf(tensor.axis().name()),
              Element.valueOf(tensor.element().name()),
              format.width(tensor),
              layout == null
                  ? List.of()
                  : Arrays.stream(layout.columnVocabularies()).boxed().toList()));
    }
    var layers = new ArrayList<Layer>();
    for (var spec : shape.layers()) {
      var mask =
          layers.isEmpty() && format.eventsSettleOnPublication()
              ? Mask.PUBLIC_PREFIX
              : Mask.ALL_KEYS;
      var queries = Queries.valueOf(spec.role().name());
      var source = layers.isEmpty() ? Source.EMBEDDING : Source.PREVIOUS_LAYER;
      layers.add(
          switch (spec.ffn()) {
            case FeedForwardShape.Shared shared ->
                new Layer(
                    queries,
                    Queries.ALL_TOKENS,
                    mask,
                    source,
                    Normalization.PRE_ATTENTION_AND_PRE_FFN,
                    FeedForward.SHARED,
                    shared.width(),
                    shared.width());
            case FeedForwardShape.Typed typed ->
                new Layer(
                    queries,
                    Queries.ALL_TOKENS,
                    mask,
                    source,
                    Normalization.PRE_ATTENTION_AND_PRE_FFN,
                    FeedForward.BY_TOKEN,
                    typed.context(),
                    typed.tile());
          });
    }
    var codes = format.candidateCodes();
    var segments =
        codes.slots().stream()
            .map(s -> new Segment(s.parameter(), s.column(), s.firstRow(), s.rows()))
            .toList();
    var stages = CollectionOrder.stages(shape.candidateSegments(), shape.reread());
    var roles =
        format.roles().stream()
            .map(role -> new Role(role.name(), Composition.valueOf(role.composition().name())))
            .toList();
    var weights = new ArrayList<Weight>();
    for (var parameter : parameters) {
      var axes =
          Arrays.stream(parameter.getValue().getShape().getShape())
              .mapToInt(Math::toIntExact)
              .boxed()
              .toList();
      weights.add(new Weight(parameter.getKey(), Element.FP32, axes));
    }
    var arithmetic =
        CollectionOrder.arithmetic(
            CollectionCapabilities.foldsLayerZero(shape.hidden()),
            format.eventsSettleOnPublication(),
            shape.candidateSegments());
    Model model =
        new Model(
            format.seats(),
            format.tiles(),
            format.maxEvents(),
            shape.hidden(),
            shape.heads(),
            PrismShape.HEAD_WIDTH,
            (float) shape.epsilon(),
            (float) EncoderLayer.SCALE,
            ValueGraphHead.TAU,
            shape.gate(),
            shape.candidateFfn(),
            format.keys().codes(),
            format.tileKeyCode().first(),
            MahjongTileRelation.values().length,
            Element.BF16,
            Element.FP32,
            Element.FP32,
            Order.LEGAL_ACTION,
            shape.candidateSegments() ? CandidateCodes.COUNT_WIDTH : 0,
            codes.codeWidth(),
            codes.tableRows(),
            inputs,
            layers,
            segments,
            stages,
            roles,
            arithmetic,
            weights);
    ModelDeclarations.validate(model);
    return model;
  }
}
