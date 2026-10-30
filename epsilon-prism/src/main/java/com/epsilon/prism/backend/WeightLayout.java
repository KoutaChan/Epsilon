package com.epsilon.prism.backend;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.epsilon.prism.hip.HipComposition;
import com.epsilon.prism.hip.HipDimensions;
import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipPack;
import com.epsilon.prism.hip.HipSchema;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.hip.HipShape;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.RelationAttribute;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * 網の推論の重みを、HIP のカーネルが読む並び（WeightPack）に置く表。網の形・推論のパラメーターの名前と形・入力の宣言から一度だけ作り、
 * 番地を手で並べない。
 *
 * <ul>
 *   <li>行列は BF16、Linear の偏りは BF16 に丸めた値を FP32 で、利得・関係の偏り・符号の表・頭の最後の層は FP32 のまま（eager の
 *       {@code Precision.BFLOAT16} の丸めと同じ所）。
 *   <li>埋め込みの表と型ごとの Linear は、K1 の畳んだ組の表（FP32 の W·E、1 トークン 8 引き以下）と、表に畳まない特徴の W（BF16、転置）と
 *       偏りにする。
 *   <li>全ての推論のパラメーターをちょうど 1 回ずつ置く（置き場のない名前は例外）。
 * </ul>
 *
 * <p>作った後は変わらないので、複数のスレッドから読んでよい。
 */
public final class WeightLayout {

  /** 集約の温度 τ（網の ValueGraphHead.TAU と同じ値。CPU の試験が照らす）。 */
  public static final float TAU = 0.01f;

  /** 注意のロジットの尺度 1/√16（網の EncoderLayer.SCALE と同じ値）。 */
  public static final float SCALE = (float) (1.0 / Math.sqrt(PrismShape.HEAD_WIDTH));

  /** 畳んだ組の表の行の上限の候補（小さい方から試し、表の数が {@link HipEmbedding#MAX_TABLES} 以下になる最初のもの）。 */
  private static final int[] TABLE_ROW_CAPS = {1024, 2048, 4096, 8192};

  enum Conversion {
    /** BF16 のビット（行列）。 */
    BF16,
    /** BF16 に丸めた値を FP32 で（Linear の偏り）。 */
    ROUNDED,
    /** FP32 のまま。 */
    FP32
  }

  /**
   * 1 つのパラメーターの置き場。
   *
   * @param parameter パラメーターの名前（推論の重みの束の名前）
   * @param weight 置き場
   * @param layer 層（層を持たない置き場は 0）
   * @param conversion 値の変換
   * @param offset WeightPack の中のバイトの位置
   */
  record Placement(String parameter, HipWeight weight, int layer, Conversion conversion, long offset) {}

  /**
   * 1 つの畳んだ組の表。
   *
   * @param members 埋め込む欄の番号（{@link EmbeddingSource#embedded()} の添字、列の順）
   * @param strides 欄ごとの刻み
   * @param rows 行の数
   * @param offset WeightPack の中のバイトの位置
   */
  record Group(int[] members, int[] strides, int rows, long offset) {}

  /**
   * 1 つの型の埋め込みの置き方。
   *
   * @param source 元
   * @param table 埋め込みの表のパラメーターの名前
   * @param projection Linear の重みのパラメーターの名前
   * @param bias Linear の偏りのパラメーターの名前
   * @param dimension 埋め込みの次元 d
   * @param groups 畳んだ組の表
   * @param denseOffset 表に畳まない特徴の W の位置
   * @param biasOffset 偏りの位置
   */
  record Embedding(
      EmbeddingSource source,
      String table,
      String projection,
      String bias,
      int dimension,
      List<Group> groups,
      long denseOffset,
      long biasOffset) {}

  private final PrismShape shape;
  private final HipDimensions dimensions;
  private final HipPack.Header header;
  private final List<Placement> placements;
  private final Map<EmbeddingSource, Embedding> embeddings;
  private final long bytes;

  private WeightLayout(
      PrismShape shape,
      HipDimensions dimensions,
      HipPack.Header header,
      List<Placement> placements,
      Map<EmbeddingSource, Embedding> embeddings,
      long bytes) {
    this.shape = shape;
    this.dimensions = dimensions;
    this.header = header;
    this.placements = placements;
    this.embeddings = embeddings;
    this.bytes = bytes;
  }

  /**
   * HIP のカーネルがこの形の網を回せるか（H は {@link HipShape#HIDDEN} だけ。前線の腕の遷移の列は置き場を持たない）。F・G・深さの決まりは
   * 網の形と同じ。
   *
   * @param shape 網の形
   * @return 回せるなら true
   */
  public static boolean supports(PrismShape shape) {
    return shape.hidden() == HipShape.HIDDEN && !shape.frontier();
  }

  /**
   * 網の推論の重みの束から作る。
   *
   * @param network 網
   * @return 表
   */
  public static WeightLayout of(PrismNetwork network) {
    return of(network.shape(), network.inferenceParameters());
  }

  /**
   * 形と推論の重みの束から作る。
   *
   * @param shape 網の形（H は {@link HipShape#HIDDEN} だけ）
   * @param parameters 推論の重みの束（{@link PrismNetwork#inferenceParameters()}）
   * @return 表
   * @throws IllegalArgumentException 置き場のないパラメーター、置かれないパラメーター、形の違いがあるとき
   */
  public static WeightLayout of(PrismShape shape, PairList<String, Parameter> parameters) {
    Map<String, long[]> shapes = new LinkedHashMap<>();
    for (Pair<String, Parameter> pair : parameters) {
      Shape parameterShape = pair.getValue().getShape();
      if (parameterShape == null) {
        throw new IllegalArgumentException("parameter " + pair.getKey() + " has no shape (initialize the network)");
      }
      shapes.put(pair.getKey(), parameterShape.getShape());
    }
    return new Builder(shape, shapes).build();
  }

  public PrismShape shape() {
    return shape;
  }

  /** 重みの置き場の大きさを決める数。 */
  public HipDimensions dimensions() {
    return dimensions;
  }

  /** WeightPack の頭。 */
  public HipPack.Header header() {
    return header;
  }

  /** WeightPack のバイト数。 */
  public long bytes() {
    return bytes;
  }

  /** パラメーターの置き場（WeightPack の書き手が読む）。 */
  List<Placement> placements() {
    return placements;
  }

  /** 型ごとの埋め込みの置き方（WeightPack の書き手が読む）。 */
  Map<EmbeddingSource, Embedding> embeddings() {
    return embeddings;
  }

  /**
   * この網と塊の容量の HIP の計画の形。
   *
   * @param capacity 塊の容量（入力の blob は {@link BatchPlan#maxBytes} の大きさ）
   * @return 形
   */
  public HipShape hipShape(BatchPlan.Capacity capacity) {
    return new HipShape(
        shape.hidden(),
        shape.ffn(),
        shape.gate(),
        shape.depth(),
        (float) shape.epsilon(),
        KeyBlock.KEY_COUNT,
        new HipShape.Capacity(
            capacity.rows(),
            capacity.events(),
            capacity.transitions(),
            capacity.actions(),
            BatchPlan.maxBytes(capacity)),
        bytes);
  }

  /**
   * 手の役割の作り方（網の ValueGraphHead.Composition.of と同じ写し）。default を書かないので、役割を足すと組み立てで落ちる。
   *
   * @param role 役割
   * @return 作り方
   */
  static HipComposition composition(ActionRole role) {
    return switch (role) {
      case DISCARD -> HipComposition.DISCARD_VALUE;
      case RIICHI -> HipComposition.RIICHI_GATE;
      case CALL, RON -> HipComposition.RESPONSE_GATE;
      case KAN, TSUMO, KYUSHU -> HipComposition.TURN_GATE;
      case PASS -> HipComposition.ZERO;
    };
  }

  /**
   * 門の種類の番号（網の ValueGraphHead.gateIndex と同じ: 門の役割を宣言の順に並べた番号。門でなければ −1）。
   *
   * @param role 役割
   * @return 番号
   */
  static int gateKind(ActionRole role) {
    if (!role.gate()) {
      return -1;
    }
    int kind = 0;
    for (ActionRole other : ActionRole.values()) {
      if (other == role) {
        return kind;
      }
      if (other.gate()) {
        kind++;
      }
    }
    throw new IllegalStateException("unreachable");
  }

  static int gateKinds() {
    int kinds = 0;
    for (ActionRole role : ActionRole.values()) {
      kinds += role.gate() ? 1 : 0;
    }
    return kinds;
  }

  /** パラメーターの名前を、登録の番号を除いた子の名前・層・パラメーターの名前に分ける。 */
  record Name(String child, int layer, String local) {

    private static final String ENCODER = "encoder";

    static Name parse(String key) {
      int split = key.indexOf('_');
      if (split < 3 || !Character.isDigit(key.charAt(0)) || !Character.isDigit(key.charAt(1))) {
        throw new IllegalArgumentException("unexpected parameter name " + key);
      }
      String child = key.substring(2, split);
      String local = key.substring(split + 1);
      if (child.startsWith(ENCODER)) {
        return new Name(ENCODER, Integer.parseInt(child.substring(ENCODER.length())), local);
      }
      return new Name(child, 0, local);
    }
  }

  private static final class Builder {

    private final PrismShape shape;
    private final Map<String, long[]> shapes;
    private final Map<String, String> byName = new LinkedHashMap<>();
    private final HipPack.Header header = new HipPack.Header();
    private final List<Placement> placements = new ArrayList<>();
    private final Map<EmbeddingSource, Embedding> embeddings = new EnumMap<>(EmbeddingSource.class);
    private final Set<String> used = new HashSet<>();
    private HipDimensions dimensions;
    private long cursor = HipPack.HEADER_BYTES;

    Builder(PrismShape shape, Map<String, long[]> shapes) {
      if (shape.hidden() != HipShape.HIDDEN) {
        throw new IllegalArgumentException(
            "HIP kernels are specialised for hidden " + HipShape.HIDDEN + " only, got " + shape.hidden());
      }
      this.shape = shape;
      this.shapes = shapes;
      for (String key : shapes.keySet()) {
        Name name = Name.parse(key);
        byName.put(name.child() + "/" + name.layer() + "/" + name.local(), key);
      }
    }

    WeightLayout build() {
      schema();
      for (EmbeddingSource source : EmbeddingSource.values()) {
        embedding(source);
      }
      header.weight(HipWeight.TILE_RELATION, 0, allocate(HipWeight.TILE_RELATION.bytes(dimensions)));
      for (int layer = 0; layer < shape.depth(); layer++) {
        place("encoder", layer, "attentionNorm", HipWeight.ATTENTION_NORM, Conversion.FP32);
        place("encoder", layer, "qkv", HipWeight.QKV, Conversion.BF16);
        place("encoder", layer, "qkvBias", HipWeight.QKV_BIAS, Conversion.ROUNDED);
        place("encoder", layer, "relation", HipWeight.RELATION_BIAS, Conversion.FP32);
        place("encoder", layer, "output", HipWeight.OUTPUT, Conversion.BF16);
        place("encoder", layer, "outputBias", HipWeight.OUTPUT_BIAS, Conversion.ROUNDED);
        place("encoder", layer, "ffnNorm", HipWeight.FFN_NORM, Conversion.FP32);
        place("encoder", layer, "up", HipWeight.UP, Conversion.BF16);
        place("encoder", layer, "upBias", HipWeight.UP_BIAS, Conversion.ROUNDED);
        place("encoder", layer, "down", HipWeight.DOWN, Conversion.BF16);
        place("encoder", layer, "downBias", HipWeight.DOWN_BIAS, Conversion.ROUNDED);
      }
      place("finalNorm", 0, "gain", HipWeight.FINAL_NORM, Conversion.FP32);
      place("candidate", 0, "tileEntry", HipWeight.TILE_ENTRY, Conversion.BF16);
      place("candidate", 0, "queryNorm", HipWeight.QUERY_NORM, Conversion.FP32);
      place("candidate", 0, "query", HipWeight.QUERY, Conversion.BF16);
      place("candidate", 0, "queryBias", HipWeight.QUERY_BIAS, Conversion.ROUNDED);
      place("candidate", 0, "keyValue", HipWeight.KEY_VALUE, Conversion.BF16);
      place("candidate", 0, "keyValueBias", HipWeight.KEY_VALUE_BIAS, Conversion.ROUNDED);
      place("candidate", 0, "codeKey", HipWeight.CODE_KEY, Conversion.FP32);
      place("candidate", 0, "codeValue", HipWeight.CODE_VALUE, Conversion.FP32);
      place("candidate", 0, "output", HipWeight.CANDIDATE_OUTPUT, Conversion.BF16);
      place("candidate", 0, "outputBias", HipWeight.CANDIDATE_OUTPUT_BIAS, Conversion.ROUNDED);
      place("candidate", 0, "ffnNorm", HipWeight.CANDIDATE_FFN_NORM, Conversion.FP32);
      place("candidate", 0, "up", HipWeight.CANDIDATE_UP, Conversion.BF16);
      place("candidate", 0, "upBias", HipWeight.CANDIDATE_UP_BIAS, Conversion.ROUNDED);
      place("candidate", 0, "down", HipWeight.CANDIDATE_DOWN, Conversion.BF16);
      place("candidate", 0, "downBias", HipWeight.CANDIDATE_DOWN_BIAS, Conversion.ROUNDED);
      place("discardValue", 0, "norm", HipWeight.DISCARD_NORM, Conversion.FP32);
      place("discardValue", 0, "hidden", HipWeight.DISCARD_HIDDEN, Conversion.BF16);
      place("discardValue", 0, "hiddenBias", HipWeight.DISCARD_HIDDEN_BIAS, Conversion.ROUNDED);
      place("discardValue", 0, "output", HipWeight.DISCARD_OUTPUT, Conversion.FP32);
      place("discardValue", 0, "exact", HipWeight.DISCARD_EXACT, Conversion.FP32);
      place("gates", 0, "norm", HipWeight.GATE_NORM, Conversion.FP32);
      place("gates", 0, "hidden", HipWeight.GATE_HIDDEN, Conversion.BF16);
      place("gates", 0, "hiddenBias", HipWeight.GATE_HIDDEN_BIAS, Conversion.ROUNDED);
      place("gates", 0, "kindOffset", HipWeight.GATE_KIND_OFFSET, Conversion.FP32);
      place("gates", 0, "exactProjection", HipWeight.GATE_EXACT_PROJECTION, Conversion.FP32);
      place("gates", 0, "output", HipWeight.GATE_OUTPUT, Conversion.FP32);
      place("gates", 0, "outputBias", HipWeight.GATE_OUTPUT_BIAS, Conversion.FP32);
      place("gates", 0, "exactOutput", HipWeight.GATE_EXACT_OUTPUT, Conversion.FP32);
      List<String> unused = new ArrayList<>();
      for (String key : shapes.keySet()) {
        if (!used.contains(key)) {
          unused.add(key);
        }
      }
      if (!unused.isEmpty()) {
        throw new IllegalArgumentException("inference parameters without a HIP slot: " + unused);
      }
      return new WeightLayout(
          shape,
          dimensions,
          header,
          Collections.unmodifiableList(placements),
          Collections.unmodifiableMap(embeddings),
          cursor);
    }

    /** 入力の約束・区画の幅・役割の写しを頭に置き、重みの大きさを決める数を作る。 */
    private void schema() {
      long[] relation = shape("encoder", 0, "relation");
      header
          .schema(HipSchema.SEATS, KeyBlock.SEAT.keys())
          .schema(HipSchema.TILES, KeyBlock.TILE.keys())
          .schema(HipSchema.ROW_FIRST_EVENT, RowColumn.FIRST_EVENT.ordinal())
          .schema(HipSchema.ROW_EVENTS, RowColumn.EVENTS.ordinal())
          .schema(HipSchema.ROW_FIRST_ACTION, RowColumn.FIRST_ACTION.ordinal())
          .schema(HipSchema.ROW_ACTIONS, RowColumn.ACTIONS.ordinal())
          .schema(HipSchema.ROW_FIRST_TRANSITION, RowColumn.FIRST_TRANSITION.ordinal())
          .schema(HipSchema.ROW_TRANSITIONS, RowColumn.TRANSITIONS.ordinal())
          .schema(HipSchema.ACTION_ROW, ActionColumn.ROW.ordinal())
          .schema(HipSchema.ACTION_FIRST_TRANSITION, ActionColumn.FIRST_TRANSITION.ordinal())
          .schema(HipSchema.ACTION_TRANSITIONS, ActionColumn.TRANSITIONS.ordinal())
          .schema(HipSchema.ACTION_ROLE, ActionColumn.ROLE.ordinal())
          .schema(HipSchema.ACTION_BASE, ActionColumn.BASE.ordinal())
          .schema(HipSchema.ACTION_REPRESENTATIVE, ActionColumn.REPRESENTATIVE.ordinal())
          .schema(HipSchema.RELATION_TILE, InputSchema.RELATION.column(RelationAttribute.TILE))
          .schema(HipSchema.RELATION_OWNER, InputSchema.RELATION.column(RelationAttribute.OWNER))
          .schema(HipSchema.RELATION_EVENT, InputSchema.RELATION.column(RelationAttribute.EVENT))
          .schema(HipSchema.RELATION_RIICHI, InputSchema.RELATION.column(RelationAttribute.OWNER_RIICHI_EVENT))
          .schema(
              HipSchema.RELATION_LAST_DISCARD,
              InputSchema.RELATION.column(RelationAttribute.OWNER_LAST_DISCARD_EVENT))
          .schema(HipSchema.RELATION_LINK, InputSchema.RELATION.column(RelationAttribute.LINK))
          .schema(HipSchema.RELATION_SAFETY, InputSchema.RELATION.column(RelationAttribute.SAFETY, 0))
          .schema(HipSchema.RELATION_SAFETIES, RelationAttribute.SAFETY.spec().repeats())
          .schema(HipSchema.TRANSITION_CALL_DISCARD, InputSchema.TRANSITION.column(TransitionField.CALL_DISCARD))
          .schema(HipSchema.TRANSITION_MAIN_TILE, InputSchema.TRANSITION.column(TransitionField.MAIN_TILE))
          .schema(HipSchema.CODES, KeyBlock.CODE_COUNT)
          .schema(HipSchema.RELATION_ROWS, Math.toIntExact(relation[0]))
          .schema(HipSchema.GATE_KINDS, gateKinds())
          .schema(HipSchema.ROLES, ActionRole.values().length)
          .schema(HipSchema.TAU_BITS, Float.floatToIntBits(TAU))
          .schema(HipSchema.SCALE_BITS, Float.floatToIntBits(SCALE));
      if (1 + KeyBlock.SEAT.keys() + KeyBlock.TILE.keys() != KeyBlock.KEY_COUNT
          || KeyBlock.ROUND.keys() != 1) {
        throw new IllegalStateException("the candidate keys must be the round, seat and tile tokens");
      }
      for (Tensor tensor : Tensor.values()) {
        HipSection section = HipSection.valueOf(tensor.name());
        if (section.element().bytes() != tensor.element().bytes()) {
          throw new IllegalStateException(tensor + " element differs from the HIP section " + section);
        }
        header.sectionWidth(section, tensor.width());
      }
      if (Tensor.values().length != HipSection.values().length) {
        throw new IllegalStateException("HIP sections do not match the input tensors");
      }
      if (ActionRole.values().length > HipPack.MAX_ROLES) {
        throw new IllegalStateException("too many action roles for the HIP pack");
      }
      for (ActionRole role : ActionRole.values()) {
        header.role(role.ordinal(), composition(role), gateKind(role));
      }
      dimensions =
          new HipDimensions(
              shape.hidden(),
              shape.ffn(),
              shape.gate(),
              KeyBlock.CODE_COUNT,
              Math.toIntExact(relation[0]),
              KeyBlock.TILE.keys(),
              gateKinds(),
              Tensor.DISCARD_EXACT.width(),
              Tensor.ACTION_EXACT.width());
    }

    /** 1 つの型の埋め込み: 畳んだ組の表、表に畳まない特徴の W、偏り。 */
    private void embedding(EmbeddingSource source) {
      String tableKey = key(source.child(), 0, source.table());
      String projectionKey = key(source.child(), 0, source.projection());
      String biasKey = key(source.child(), 0, source.bias());
      long[] table = shapes.get(tableKey);
      long[] projection = shapes.get(projectionKey);
      int dimension = Math.toIntExact(table[1]);
      int inputs = source.embeddedCount() * dimension + source.denseCount();
      if (projection[0] != shape.hidden() || projection[1] != inputs) {
        throw new IllegalArgumentException(
            projectionKey + " has shape " + Arrays.toString(projection) + ", expected [" + shape.hidden() + ", " + inputs + "]");
      }
      requireShape(biasKey, shape.hidden());
      List<Group> groups = new ArrayList<>();
      for (int[] members : groupColumns(source)) {
        int rows = 1;
        int[] strides = new int[members.length];
        for (int m = 0; m < members.length; m++) {
          strides[m] = rows;
          rows = Math.multiplyExact(rows, source.vocabulary(members[m]));
        }
        long offset = allocate((long) rows * shape.hidden() * Float.BYTES);
        groups.add(new Group(members, strides, rows, offset));
      }
      long denseOffset =
          source.denseCount() == 0 ? 0 : allocate((long) source.denseCount() * shape.hidden() * Short.BYTES);
      long biasOffset = allocate((long) shape.hidden() * Float.BYTES);
      used.add(tableKey);
      used.add(projectionKey);
      used.add(biasKey);
      Embedding embedding =
          new Embedding(
              source,
              tableKey,
              projectionKey,
              biasKey,
              dimension,
              Collections.unmodifiableList(groups),
              denseOffset,
              biasOffset);
      embeddings.put(source, embedding);
      List<HipEmbedding.Table> tables = new ArrayList<>();
      int[] embedded = source.embedded();
      for (Group group : groups) {
        int[] columns = new int[group.members().length];
        for (int m = 0; m < columns.length; m++) {
          columns[m] = embedded[group.members()[m]];
        }
        tables.add(new HipEmbedding.Table(Math.toIntExact(group.offset()), group.rows(), columns, group.strides()));
      }
      header.embedding(
          source.slot(),
          new HipEmbedding(
              EmbeddingSource.section(source.category()),
              EmbeddingSource.section(source.scalar()),
              EmbeddingSource.section(source.exact()),
              tables,
              List.of(source.dense()),
              Math.toIntExact(denseOffset),
              Math.toIntExact(biasOffset)));
    }

    /**
     * 埋め込む欄を畳んだ組に分ける（語彙の大きい順に、行の上限に入る最初の組へ。組の中は列の順）。行の上限は、組の数が
     * {@link HipEmbedding#MAX_TABLES} 以下になる最小の候補。
     */
    private static List<int[]> groupColumns(EmbeddingSource source) {
      Integer[] order = new Integer[source.embeddedCount()];
      for (int i = 0; i < order.length; i++) {
        order[i] = i;
      }
      Arrays.sort(
          order,
          (a, b) ->
              source.vocabulary(a) != source.vocabulary(b)
                  ? Integer.compare(source.vocabulary(b), source.vocabulary(a))
                  : Integer.compare(a, b));
      for (int cap : TABLE_ROW_CAPS) {
        List<List<Integer>> groups = new ArrayList<>();
        List<Integer> products = new ArrayList<>();
        boolean fits = true;
        for (int column : order) {
          int vocabulary = source.vocabulary(column);
          if (vocabulary > cap) {
            fits = false;
            break;
          }
          int target = -1;
          for (int g = 0; g < groups.size(); g++) {
            if ((long) products.get(g) * vocabulary <= cap && groups.get(g).size() < HipEmbedding.MAX_MEMBERS) {
              target = g;
              break;
            }
          }
          if (target < 0) {
            groups.add(new ArrayList<>());
            products.add(1);
            target = groups.size() - 1;
          }
          groups.get(target).add(column);
          products.set(target, products.get(target) * vocabulary);
        }
        if (fits && groups.size() <= HipEmbedding.MAX_TABLES) {
          List<int[]> result = new ArrayList<>();
          for (List<Integer> group : groups) {
            result.add(group.stream().mapToInt(Integer::intValue).sorted().toArray());
          }
          return result;
        }
      }
      throw new IllegalStateException(
          source + " embedding needs more than " + HipEmbedding.MAX_TABLES + " folded tables");
    }

    private void place(String child, int layer, String local, HipWeight weight, Conversion conversion) {
      String key = key(child, layer, local);
      long[] parameterShape = shapes.get(key);
      long elements = 1;
      for (long dimension : parameterShape) {
        elements *= dimension;
      }
      if (elements != weight.elements(dimensions)) {
        throw new IllegalArgumentException(
            key + " has " + elements + " elements but the HIP slot " + weight + " holds " + weight.elements(dimensions));
      }
      long bytesNeeded =
          conversion == Conversion.BF16 ? elements * Short.BYTES : elements * Float.BYTES;
      if (bytesNeeded != weight.bytes(dimensions)) {
        throw new IllegalStateException(weight + " conversion " + conversion + " does not match its element type");
      }
      long offset = allocate(bytesNeeded);
      header.weight(weight, weight.layered() ? layer : 0, offset);
      placements.add(new Placement(key, weight, layer, conversion, offset));
      used.add(key);
    }

    private void requireShape(String key, long... expected) {
      if (!Arrays.equals(shapes.get(key), expected)) {
        throw new IllegalArgumentException(
            key + " has shape " + Arrays.toString(shapes.get(key)) + ", expected " + Arrays.toString(expected));
      }
    }

    private long[] shape(String child, int layer, String local) {
      return shapes.get(key(child, layer, local));
    }

    private String key(String child, int layer, String local) {
      String key = byName.get(child + "/" + layer + "/" + local);
      if (key == null) {
        throw new IllegalArgumentException("the network has no parameter " + child + layer + "." + local);
      }
      return Objects.requireNonNull(key);
    }

    private long allocate(long size) {
      long offset = cursor;
      cursor = HipPack.align(cursor + size);
      if (cursor > Integer.MAX_VALUE) {
        throw new IllegalStateException("weight pack exceeds 2 GiB");
      }
      return offset;
    }
  }
}
