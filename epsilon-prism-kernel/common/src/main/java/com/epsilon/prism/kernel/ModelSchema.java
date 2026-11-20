package com.epsilon.prism.kernel;

import java.util.List;
import java.util.Objects;

/** モデル宣言のバイナリ形式。各整数はリトルエンディアンの 32 ビットで表す。 */
public final class ModelSchema {
  private ModelSchema() {}

  public static final int VERSION =
      Integer.parseInt(NativeDeclarations.resource("declaration_version.inc"));
  static final int MAGIC = 1145918032;

  public enum Element {
    U8(1),
    BF16(2),
    FP32(3),
    I32(4);
    private final int id;

    Element(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Element read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> U8;
        case 2 -> BF16;
        case 3 -> FP32;
        case 4 -> I32;
        default -> throw new IllegalArgumentException("Unknown Element ID");
      };
    }
  }

  public enum Axis {
    ROW(1),
    SEAT(2),
    TILE(3),
    EVENT(4),
    TOKEN(5),
    TRANSITION(6),
    ACTION(7);
    private final int id;

    Axis(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Axis read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> ROW;
        case 2 -> SEAT;
        case 3 -> TILE;
        case 4 -> EVENT;
        case 5 -> TOKEN;
        case 6 -> TRANSITION;
        case 7 -> ACTION;
        default -> throw new IllegalArgumentException("Unknown Axis ID");
      };
    }
  }

  public enum Queries {
    ALL_TOKENS(1),
    FIXED_TOKENS(2);
    private final int id;

    Queries(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Queries read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> ALL_TOKENS;
        case 2 -> FIXED_TOKENS;
        default -> throw new IllegalArgumentException("Unknown Queries ID");
      };
    }
  }

  public enum Mask {
    ALL_KEYS(1),
    PUBLIC_PREFIX(2);
    private final int id;

    Mask(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Mask read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> ALL_KEYS;
        case 2 -> PUBLIC_PREFIX;
        default -> throw new IllegalArgumentException("Unknown Mask ID");
      };
    }
  }

  public enum Source {
    EMBEDDING(1),
    PREVIOUS_LAYER(2);
    private final int id;

    Source(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Source read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> EMBEDDING;
        case 2 -> PREVIOUS_LAYER;
        default -> throw new IllegalArgumentException("Unknown Source ID");
      };
    }
  }

  public enum Normalization {
    PRE_ATTENTION_AND_PRE_FFN(1);
    private final int id;

    Normalization(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Normalization read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> PRE_ATTENTION_AND_PRE_FFN;
        default -> throw new IllegalArgumentException("Unknown Normalization ID");
      };
    }
  }

  public enum Order {
    LEGAL_ACTION(1);
    private final int id;

    Order(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Order read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> LEGAL_ACTION;
        default -> throw new IllegalArgumentException("Unknown Order ID");
      };
    }
  }

  public enum FeedForward {
    SHARED(1),
    BY_TOKEN(2);
    private final int id;

    FeedForward(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static FeedForward read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> SHARED;
        case 2 -> BY_TOKEN;
        default -> throw new IllegalArgumentException("Unknown FeedForward ID");
      };
    }
  }

  public enum Stage {
    COUNT_TILES(1),
    SEGMENT_BIAS(2),
    CANDIDATE_ATTENTION(3),
    CANDIDATE_FFN(4),
    EVENT_REREAD(5),
    DISCARD_VALUE(6),
    COMPOSE_VALUES(7);
    private final int id;

    Stage(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Stage read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> COUNT_TILES;
        case 2 -> SEGMENT_BIAS;
        case 3 -> CANDIDATE_ATTENTION;
        case 4 -> CANDIDATE_FFN;
        case 5 -> EVENT_REREAD;
        case 6 -> DISCARD_VALUE;
        case 7 -> COMPOSE_VALUES;
        default -> throw new IllegalArgumentException("Unknown Stage ID");
      };
    }
  }

  public enum Composition {
    DISCARD_VALUE(1),
    RIICHI_GATE(2),
    RESPONSE_GATE(3),
    TURN_GATE(4),
    ZERO(5);
    private final int id;

    Composition(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Composition read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> DISCARD_VALUE;
        case 2 -> RIICHI_GATE;
        case 3 -> RESPONSE_GATE;
        case 4 -> TURN_GATE;
        case 5 -> ZERO;
        default -> throw new IllegalArgumentException("Unknown Composition ID");
      };
    }
  }

  public enum Arithmetic {
    EMBED_COLUMN_FMA(1),
    RMS_XOR_1_2(2),
    BF16_TILE_K_ASCENDING(3),
    PAIR_BIAS_01_23_45_6(4),
    PREFIX_EDGE_FP32_FMA(5),
    PREFIX_FULL_SPLIT_BF16(6),
    RESIDUAL_ROUND_BF16(7),
    COUNT_TILE_MUL_ADD(8),
    SEGMENT_DECLARATION_ORDER(9),
    CANDIDATE_KEY_ORDER(10),
    VALUE_ACTION_ORDER(11),
    FOLDED_QKV_UNROUNDED_RMS(12);
    private final int id;

    Arithmetic(int id) {
      this.id = id;
    }

    void write(DeclarationIO.Output out) {
      out.positive(id);
    }

    static Arithmetic read(DeclarationIO.Input in) {
      return switch (in.positive()) {
        case 1 -> EMBED_COLUMN_FMA;
        case 2 -> RMS_XOR_1_2;
        case 3 -> BF16_TILE_K_ASCENDING;
        case 4 -> PAIR_BIAS_01_23_45_6;
        case 5 -> PREFIX_EDGE_FP32_FMA;
        case 6 -> PREFIX_FULL_SPLIT_BF16;
        case 7 -> RESIDUAL_ROUND_BF16;
        case 8 -> COUNT_TILE_MUL_ADD;
        case 9 -> SEGMENT_DECLARATION_ORDER;
        case 10 -> CANDIDATE_KEY_ORDER;
        case 11 -> VALUE_ACTION_ORDER;
        case 12 -> FOLDED_QKV_UNROUNDED_RMS;
        default -> throw new IllegalArgumentException("Unknown Arithmetic ID");
      };
    }
  }

  public enum Section {
    ROUND_CATEGORY(Axis.ROW, Element.U8),
    ROUND_SCALAR(Axis.ROW, Element.BF16),
    SEAT_CATEGORY(Axis.SEAT, Element.U8),
    SEAT_SCALAR(Axis.SEAT, Element.BF16),
    TILE_CATEGORY(Axis.TILE, Element.U8),
    EVENT_CATEGORY(Axis.EVENT, Element.U8),
    EVENT_SCALAR(Axis.EVENT, Element.BF16),
    RELATION(Axis.TOKEN, Element.U8),
    TRANSITION_CATEGORY(Axis.TRANSITION, Element.U8),
    TRANSITION_SCALAR(Axis.TRANSITION, Element.BF16),
    TRANSITION_CODE(Axis.TRANSITION, Element.U8),
    DISCARD_EXACT(Axis.TRANSITION, Element.FP32),
    ACTION_COLUMN(Axis.ACTION, Element.I32),
    ACTION_EXACT(Axis.ACTION, Element.FP32),
    ROW_COLUMN(Axis.ROW, Element.I32);
    private final Axis axis;
    private final Element element;

    Section(Axis axis, Element element) {
      this.axis = axis;
      this.element = element;
    }

    public Axis axis() {
      return axis;
    }

    public Element element() {
      return element;
    }
  }

  public record Input(
      String name, Axis axis, Element element, int width, List<Integer> vocabularies) {
    public Input {
      DeclarationIO.text(name);
      Objects.requireNonNull(axis);
      Objects.requireNonNull(element);
      DeclarationIO.positive(width);
      vocabularies = List.copyOf(vocabularies);
    }

    static Input read(DeclarationIO.Input in) {
      return new Input(
          in.text(), Axis.read(in), Element.read(in), in.positive(), in.list(() -> in.positive()));
    }

    void write(DeclarationIO.Output out) {
      out.text(name);
      axis.write(out);
      element.write(out);
      out.positive(width);
      out.size(vocabularies.size());
      for (var item : vocabularies) {
        out.positive(item);
      }
    }
  }

  public record Layer(
      Queries queries,
      Queries keys,
      Mask mask,
      Source source,
      Normalization normalization,
      FeedForward ffn,
      int contextWidth,
      int tileWidth) {
    public Layer {
      Objects.requireNonNull(queries);
      Objects.requireNonNull(keys);
      Objects.requireNonNull(mask);
      Objects.requireNonNull(source);
      Objects.requireNonNull(normalization);
      Objects.requireNonNull(ffn);
      DeclarationIO.positive(contextWidth);
      DeclarationIO.positive(tileWidth);
    }

    static Layer read(DeclarationIO.Input in) {
      return new Layer(
          Queries.read(in),
          Queries.read(in),
          Mask.read(in),
          Source.read(in),
          Normalization.read(in),
          FeedForward.read(in),
          in.positive(),
          in.positive());
    }

    void write(DeclarationIO.Output out) {
      queries.write(out);
      keys.write(out);
      mask.write(out);
      source.write(out);
      normalization.write(out);
      ffn.write(out);
      out.positive(contextWidth);
      out.positive(tileWidth);
    }
  }

  public record Segment(String parameter, int column, int firstRow, int rows) {
    public Segment {
      DeclarationIO.text(parameter);
      DeclarationIO.size(column);
      DeclarationIO.size(firstRow);
      DeclarationIO.positive(rows);
    }

    static Segment read(DeclarationIO.Input in) {
      return new Segment(in.text(), in.size(), in.size(), in.positive());
    }

    void write(DeclarationIO.Output out) {
      out.text(parameter);
      out.size(column);
      out.size(firstRow);
      out.positive(rows);
    }
  }

  public record Role(String name, Composition composition) {
    public Role {
      DeclarationIO.text(name);
      Objects.requireNonNull(composition);
    }

    static Role read(DeclarationIO.Input in) {
      return new Role(in.text(), Composition.read(in));
    }

    void write(DeclarationIO.Output out) {
      out.text(name);
      composition.write(out);
    }
  }

  public record Weight(String name, Element element, List<Integer> axes) {
    public Weight {
      DeclarationIO.text(name);
      Objects.requireNonNull(element);
      axes = List.copyOf(axes);
    }

    static Weight read(DeclarationIO.Input in) {
      return new Weight(in.text(), Element.read(in), in.list(() -> in.positive()));
    }

    void write(DeclarationIO.Output out) {
      out.text(name);
      element.write(out);
      out.size(axes.size());
      for (var item : axes) {
        out.positive(item);
      }
    }
  }

  public record Model(
      int seats,
      int tiles,
      int maxEvents,
      int hidden,
      int heads,
      int headWidth,
      float epsilon,
      float attentionScale,
      float temperature,
      int gateWidth,
      int candidateWidth,
      int keyCodes,
      int tileKeyFirstCode,
      int tileRelations,
      Element activation,
      Element accumulation,
      Element score,
      Order representativeOrder,
      int countWidth,
      int codeWidth,
      int tableRows,
      List<Input> inputs,
      List<Layer> layers,
      List<Segment> segments,
      List<Stage> stages,
      List<Role> roles,
      List<Arithmetic> arithmetic,
      List<Weight> weights) {
    public Model {
      DeclarationIO.positive(seats);
      DeclarationIO.positive(tiles);
      DeclarationIO.positive(maxEvents);
      DeclarationIO.positive(hidden);
      DeclarationIO.positive(heads);
      DeclarationIO.positive(headWidth);
      DeclarationIO.scale(epsilon);
      DeclarationIO.scale(attentionScale);
      DeclarationIO.scale(temperature);
      DeclarationIO.positive(gateWidth);
      DeclarationIO.positive(candidateWidth);
      DeclarationIO.positive(keyCodes);
      DeclarationIO.size(tileKeyFirstCode);
      DeclarationIO.positive(tileRelations);
      Objects.requireNonNull(activation);
      Objects.requireNonNull(accumulation);
      Objects.requireNonNull(score);
      Objects.requireNonNull(representativeOrder);
      DeclarationIO.size(countWidth);
      DeclarationIO.positive(codeWidth);
      DeclarationIO.size(tableRows);
      inputs = List.copyOf(inputs);
      layers = List.copyOf(layers);
      segments = List.copyOf(segments);
      stages = List.copyOf(stages);
      roles = List.copyOf(roles);
      arithmetic = List.copyOf(arithmetic);
      weights = List.copyOf(weights);
    }

    static Model read(DeclarationIO.Input in) {
      return new Model(
          in.positive(),
          in.positive(),
          in.positive(),
          in.positive(),
          in.positive(),
          in.positive(),
          in.scale(),
          in.scale(),
          in.scale(),
          in.positive(),
          in.positive(),
          in.positive(),
          in.size(),
          in.positive(),
          Element.read(in),
          Element.read(in),
          Element.read(in),
          Order.read(in),
          in.size(),
          in.positive(),
          in.size(),
          in.list(() -> Input.read(in)),
          in.list(() -> Layer.read(in)),
          in.list(() -> Segment.read(in)),
          in.list(() -> Stage.read(in)),
          in.list(() -> Role.read(in)),
          in.list(() -> Arithmetic.read(in)),
          in.list(() -> Weight.read(in)));
    }

    void write(DeclarationIO.Output out) {
      out.positive(seats);
      out.positive(tiles);
      out.positive(maxEvents);
      out.positive(hidden);
      out.positive(heads);
      out.positive(headWidth);
      out.scale(epsilon);
      out.scale(attentionScale);
      out.scale(temperature);
      out.positive(gateWidth);
      out.positive(candidateWidth);
      out.positive(keyCodes);
      out.size(tileKeyFirstCode);
      out.positive(tileRelations);
      activation.write(out);
      accumulation.write(out);
      score.write(out);
      representativeOrder.write(out);
      out.size(countWidth);
      out.positive(codeWidth);
      out.size(tableRows);
      out.size(inputs.size());
      for (var item : inputs) {
        item.write(out);
      }
      out.size(layers.size());
      for (var item : layers) {
        item.write(out);
      }
      out.size(segments.size());
      for (var item : segments) {
        item.write(out);
      }
      out.size(stages.size());
      for (var item : stages) {
        item.write(out);
      }
      out.size(roles.size());
      for (var item : roles) {
        item.write(out);
      }
      out.size(arithmetic.size());
      for (var item : arithmetic) {
        item.write(out);
      }
      out.size(weights.size());
      for (var item : weights) {
        item.write(out);
      }
    }
  }
}
