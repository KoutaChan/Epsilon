package com.epsilon.prism.backend;

import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.epsilon.prism.input.CandidateCodes;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFeature;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.InputFormat.Role;
import com.epsilon.prism.input.InputSchema.RelationAttribute;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.kernel.CandidateStages;
import com.epsilon.prism.kernel.CollectionArchitecture;
import com.epsilon.prism.kernel.CollectionCapabilities;
import com.epsilon.prism.kernel.CollectionPlanOptions;
import com.epsilon.prism.kernel.ColumnIndex;
import com.epsilon.prism.kernel.EmbeddingTable;
import com.epsilon.prism.kernel.FeedForwardShape;
import com.epsilon.prism.kernel.InputSection;
import com.epsilon.prism.kernel.PackedWeightLayout;
import com.epsilon.prism.kernel.ScoreComposition;
import com.epsilon.prism.kernel.TokenCounts;
import com.epsilon.prism.kernel.TokenGroup;
import com.epsilon.prism.kernel.WeightField;
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
 * ネットワークの推論の重みを、推論カーネルが読む並び（WeightPack）のどこに置くかを決めた表。形の記述、推論のパラメーターの名前と形、
 * 入力の宣言から一度だけ作るので、オフセットを手で並べることはない。席・牌・コード・ゲート・手の扱いの数と、層・FFN の幅は形の記述から読む。
 * 入力の列の並びは、ネットワークの入力の形式から読む。入力の形式が持つ数は、形の記述の数と同じでなければならない。比べるのは、席・牌・コード・
 * 牌の関係の種類・イベントの上限、手の扱い、ゲートの種類の数。
 *
 * <ul>
 *   <li>行列は BF16 で置く。Linear のバイアスは BF16 に丸めた値を FP32 で置く。ゲイン・関係のバイアス・コードの表・ヘッドの最後の層は
 *       FP32 のまま置く。丸める所は eager の {@code Precision.BFLOAT16} と同じ。
 *   <li>埋め込みの表と型ごとの Linear は、埋め込みで引く組の表（W·E を折り込んだ FP32 の表。1 トークンで引くのは 8 回以下）と、表に折り込まない
 *       特徴の W（BF16、転置）と、バイアスに分けて置く。局・席・牌・イベントの型には、層 0 の QKV まで折り込んだ表 T'・W'_d・b' と
 *       b_qkv（FP32）も置く。
 *   <li>推論のパラメーターは全て、ちょうど 1 回ずつ置く。置く場所の無い名前があれば例外を投げる。
 * </ul>
 *
 * <p>作った後は変わらないので、複数のスレッドから読んでよい。
 */
public final class WeightLayout {

  /** 組の表の行数の上限として試す値。小さい方から試し、表の数が {@link EmbeddingTable#MAX_TABLES} 以下になった最初の値を使う。 */
  private static final int[] TABLE_ROW_CAPS = {1024, 2048, 4096, 8192};

  enum Conversion {
    /** BF16 のビット（行列）。 */
    BF16,
    /** BF16 に丸めた値を FP32 で置く（Linear のバイアス）。 */
    ROUNDED,
    /** FP32 のまま。 */
    FP32
  }

  /**
   * 1 つのパラメーターを置く場所。
   *
   * @param parameter パラメーターの名前（推論の重みの束での名前）
   * @param weight 置く先のスロット
   * @param entry スロットの中の項目（{@link WeightField.Index} の番号）
   * @param conversion 値の変換
   * @param offset WeightPack の中のバイトオフセット
   */
  record Placement(
      String parameter, WeightField weight, int entry, Conversion conversion, long offset) {}

  /**
   * 1 つの組の表。組に入る列の値の組み合わせごとに 1 行を持つ。
   *
   * @param members 組に入る列（{@link EmbeddingSource#embedded()} の添字、列の順）
   * @param strides 列ごとのストライド
   * @param rows 行数
   * @param offset WeightPack の中のバイトオフセット
   * @param foldOffset 層 0 の QKV まで折り込んだ表 T' のオフセット（折り込まない型は −1）
   */
  record Group(int[] members, int[] strides, int rows, long offset, long foldOffset) {}

  /**
   * 1 つの型の埋め込みの置き方。
   *
   * @param source 埋め込む入力の型と列
   * @param table 埋め込みの表のパラメーターの名前
   * @param projection Linear の重みのパラメーターの名前
   * @param bias Linear のバイアスのパラメーターの名前
   * @param dimension 埋め込みの次元 d
   * @param groups 組の表
   * @param denseOffset 表に折り込まない特徴の W のオフセット
   * @param biasOffset バイアスのオフセット
   * @param foldOffset 層 0 の QKV まで折り込んだ W'_d と、それに続く b' と b_qkv のオフセット（折り込まない型は −1）
   */
  record Embedding(
      EmbeddingSource source,
      String table,
      String projection,
      String bias,
      int dimension,
      List<Group> groups,
      long denseOffset,
      long biasOffset,
      long foldOffset) {}

  /** FFN 1 つ分のパラメーターの名前（ゲイン、上りの射影とバイアス、下りの射影とバイアス）。 */
  private record FfnNames(String norm, String up, String upBias, String down, String downBias) {

    static FfnNames of(String norm, String up, String down) {
      return new FfnNames(norm, up, up + "Bias", down, down + "Bias");
    }
  }

  private static final FfnNames SHARED_FFN = FfnNames.of("ffnNorm", "up", "down");

  private static final Map<TokenGroup, FfnNames> TYPED_FFN =
      Map.of(
          TokenGroup.CONTEXT, FfnNames.of("contextNorm", "contextUp", "contextDown"),
          TokenGroup.TILE, FfnNames.of("tileNorm", "tileUp", "tileDown"));

  private static final Map<TokenGroup, String> FINAL_GAINS =
      Map.of(
          TokenGroup.ALL, "gain", TokenGroup.CONTEXT, "contextGain", TokenGroup.TILE, "tileGain");

  private final InputFormat format;
  private final CollectionArchitecture architecture;
  private final PackedWeightLayout.Header header;
  private final List<Placement> placements;
  private final Map<EmbeddingTable.Slot, Embedding> embeddings;
  private final long countOffset;
  private final long segmentsOffset;
  private final long bytes;

  private WeightLayout(
      InputFormat format,
      CollectionArchitecture architecture,
      PackedWeightLayout.Header header,
      List<Placement> placements,
      Map<EmbeddingTable.Slot, Embedding> embeddings,
      long countOffset,
      long segmentsOffset,
      long bytes) {
    this.format = format;
    this.architecture = architecture;
    this.header = header;
    this.placements = placements;
    this.embeddings = embeddings;
    this.countOffset = countOffset;
    this.segmentsOffset = segmentsOffset;
    this.bytes = bytes;
  }

  /**
   * 推論カーネルがこの形のネットワークを動かせるかを返す。既存の幅と、V4 の幅 384 に対応する。
   *
   * @param shape ネットワークの形
   * @return 動かせるなら true
   */
  public static boolean supports(PrismShape shape) {
    return CollectionCapabilities.supports(
        shape.hidden(), shape.candidateSegments(), shape.reread());
  }

  /**
   * ネットワークの推論の重みの束から表を作る。形の記述は、ネットワークの形と入力の形式から作る。
   *
   * @param network ネットワーク
   * @return 表
   * @throws IllegalArgumentException 推論カーネルが特化していない形のとき
   */
  public static WeightLayout of(PrismNetwork network) {
    InputFormat format = network.format();
    if (!supports(network.shape()))
      throw new IllegalArgumentException(
          "PRISM collection kernels do not support this shape: " + network.shape());
    return of(
        format, CollectionArchitecture.from(network.declaration()), network.inferenceParameters());
  }

  /**
   * 入力の形式、形の記述、推論の重みの束から表を作る。
   *
   * @param format 入力の形式（列の並び、手の扱い、牌の関係の表を読む）
   * @param architecture 形の記述
   * @param parameters 推論の重みの束（{@link PrismNetwork#inferenceParameters()} と同じ名前と形）
   * @return 表
   * @throws IllegalArgumentException 入力の形式と形の記述で数が合わないとき。置く場所の無いパラメーター、置かれずに残るパラメーター、
   *     形の合わないパラメーターがあるとき
   */
  public static WeightLayout of(
      InputFormat format,
      CollectionArchitecture architecture,
      PairList<String, Parameter> parameters) {
    if (!CollectionCapabilities.supports(
        architecture.hidden(),
        format.has(InputFeature.CANDIDATE_SEGMENTS),
        architecture.reread())) {
      throw new IllegalArgumentException(
          "PRISM collection kernels do not support the input format and candidate stages for this"
              + " width");
    }
    requireMatches(format, architecture);
    Map<String, long[]> shapes = new LinkedHashMap<>();
    for (Pair<String, Parameter> pair : parameters) {
      Shape parameterShape = pair.getValue().getShape();
      if (parameterShape == null) {
        throw new IllegalArgumentException(
            "parameter " + pair.getKey() + " has no shape (initialize the network)");
      }
      shapes.put(pair.getKey(), parameterShape.getShape());
    }
    return new Builder(format, architecture, shapes).build();
  }

  boolean matchesFeatures(PrismNetwork network) {
    boolean segments = network.format().has(InputFeature.CANDIDATE_SEGMENTS);
    return format == network.format()
        && (countOffset >= 0) == segments
        && (segmentsOffset >= 0) == segments;
  }

  /** 層 0 で確定したイベントを、後のすべての層が読み取り専用のキーとバリューとして使えるか。 */
  public boolean reusableEventPrefix() {
    return format.eventsSettleOnPublication()
        && architecture.allTokenLayers() == 1
        && architecture.stages() == CandidateStages.ONE;
  }

  CollectionPlanOptions planOptions() {
    var count =
        countOffset < 0
            ? null
            : new CollectionPlanOptions.Count(
                countOffset, format.tileKeyCode().first(), CandidateCodes.COUNT_WIDTH);
    var segments =
        segmentsOffset < 0
            ? null
            : new CollectionPlanOptions.Segments(segmentsOffset, format.candidateCodes().hip());
    return new CollectionPlanOptions(format.eventsSettleOnPublication(), count, segments);
  }

  /**
   * 入力の形式と形の記述で数が合うことを確かめる。
   *
   * @throws IllegalArgumentException 席・牌・コード・牌の関係の種類・イベントの上限、手の扱い、ゲートの種類の
   *     どれかの数が違うとき
   */
  private static void requireMatches(InputFormat format, CollectionArchitecture architecture) {
    TokenCounts counts = architecture.counts();
    TokenCounts declared = PrismShape.counts(format);
    if (!declared.equals(counts)) {
      throw new IllegalArgumentException(
          "the "
              + format.variant()
              + " input has the counts "
              + declared
              + ", the architecture has "
              + counts);
    }
    int roles = format.roles().size();
    int gates = (int) format.roles().stream().filter(Role::gate).count();
    if (roles != architecture.roles() || gates != architecture.gateKinds()) {
      throw new IllegalArgumentException(
          "the "
              + format.variant()
              + " input declares "
              + roles
              + " action roles with "
              + gates
              + " gate kinds, the architecture has "
              + architecture.roles()
              + " with "
              + architecture.gateKinds());
    }
  }

  /** 入力の形式。 */
  public InputFormat format() {
    return format;
  }

  /** 形の記述。 */
  public CollectionArchitecture architecture() {
    return architecture;
  }

  /** WeightPack のヘッダー。 */
  public PackedWeightLayout.Header header() {
    return header;
  }

  /** WeightPack のバイト数。 */
  public long bytes() {
    return bytes;
  }

  /** パラメーターごとの置く場所。WeightPack を書くときに読む。 */
  List<Placement> placements() {
    return placements;
  }

  /** 型ごとの埋め込みの置き方。WeightPack を書くときに読む。 */
  Map<EmbeddingTable.Slot, Embedding> embeddings() {
    return embeddings;
  }

  /**
   * 手の扱いが決める s の作り方を、推論カーネルの演算に対応させる。switch に default を書かないので、作り方を足すとここでコンパイルが通らなくなる。
   *
   * @param role 価値グラフでの手の扱い
   * @return 推論カーネルで使う演算
   */
  static ScoreComposition composition(Role role) {
    return switch (role.composition()) {
      case DISCARD_VALUE -> ScoreComposition.DISCARD_VALUE;
      case RIICHI_GATE -> ScoreComposition.RIICHI_GATE;
      case RESPONSE_GATE -> ScoreComposition.RESPONSE_GATE;
      case TURN_GATE -> ScoreComposition.TURN_GATE;
      case ZERO -> ScoreComposition.ZERO;
    };
  }

  /**
   * ゲートの種類の番号を返す。番号は、ゲートのヘッドが値を出す手の扱いを宣言の順に数えたもので、ネットワークの
   * {@code ValueGraphHead.gateIndex} と同じ。
   *
   * @param format 入力の形式
   * @param role 入力の形式が持つ手の扱い
   * @return 番号。ゲートのヘッドが値を出さない手の扱いなら −1
   * @throws IllegalArgumentException 入力の形式が持つ手の扱いでないとき
   */
  static int gateKind(InputFormat format, Role role) {
    if (!role.gate()) {
      return -1;
    }
    int kind = 0;
    for (Role other : format.roles()) {
      if (other.ordinal() == role.ordinal()) {
        return kind;
      }
      if (other.gate()) {
        kind++;
      }
    }
    throw new IllegalArgumentException(
        role.name() + " is not an action role of the " + format.variant() + " input");
  }

  /** パラメーターの名前を、子ブロックの名前（登録の番号を除く）・層・子ブロックの中での名前に分ける。 */
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
      if (child.equals("candidateFeatures")
          && local.indexOf('_') > 2
          && Character.isDigit(local.charAt(0))
          && Character.isDigit(local.charAt(1))) {
        int nested = local.indexOf('_');
        return new Name(child + "/" + local.substring(2, nested), 0, local.substring(nested + 1));
      }
      return new Name(child, 0, local);
    }
  }

  private static final class Builder {

    private final InputFormat format;
    private final CollectionArchitecture architecture;
    private final Map<String, long[]> shapes;
    private final Map<String, String> byName = new LinkedHashMap<>();
    private final PackedWeightLayout.Header header;
    private final List<Placement> placements = new ArrayList<>();
    private final Map<EmbeddingTable.Slot, Embedding> embeddings =
        new EnumMap<>(EmbeddingTable.Slot.class);
    private final Set<String> used = new HashSet<>();
    private long cursor = PackedWeightLayout.HEADER_BYTES;

    Builder(InputFormat format, CollectionArchitecture architecture, Map<String, long[]> shapes) {
      this.format = Objects.requireNonNull(format, "format");
      this.architecture = Objects.requireNonNull(architecture, "architecture");
      this.shapes = shapes;
      header = new PackedWeightLayout.Header(architecture);
      for (String key : shapes.keySet()) {
        Name name = Name.parse(key);
        byName.put(name.child() + "/" + name.layer() + "/" + name.local(), key);
      }
    }

    WeightLayout build() {
      schema();
      for (EmbeddingSource source : EmbeddingSource.of(format)) {
        embedding(source);
      }
      header.weight(
          WeightField.TILE_RELATION, 0, allocate(header.bytes(WeightField.TILE_RELATION, 0)));
      for (int layer = 0; layer < architecture.depth(); layer++) {
        place(
            "encoder", layer, "attentionNorm", WeightField.ATTENTION_NORM, layer, Conversion.FP32);
        place("encoder", layer, "qkv", WeightField.QKV, layer, Conversion.BF16);
        place("encoder", layer, "qkvBias", WeightField.QKV_BIAS, layer, Conversion.ROUNDED);
        place("encoder", layer, "relation", WeightField.RELATION_BIAS, layer, Conversion.FP32);
        place("encoder", layer, "output", WeightField.OUTPUT, layer, Conversion.BF16);
        place("encoder", layer, "outputBias", WeightField.OUTPUT_BIAS, layer, Conversion.ROUNDED);
        FeedForwardShape ffn = architecture.layer(layer).ffn();
        for (TokenGroup group : ffn.groups()) {
          FfnNames names =
              ffn instanceof FeedForwardShape.Typed ? TYPED_FFN.get(group) : SHARED_FFN;
          ffn("encoder", layer, names, WeightField.ffnEntry(layer, group));
        }
      }
      for (TokenGroup group : architecture.finalGroups()) {
        place(
            "finalNorm",
            0,
            FINAL_GAINS.get(group),
            WeightField.FINAL_NORM,
            group.ordinal(),
            Conversion.FP32);
      }
      place("candidate", 0, "tileEntry", WeightField.TILE_ENTRY, 0, Conversion.BF16);
      place("candidate", 0, "queryNorm", WeightField.QUERY_NORM, 0, Conversion.FP32);
      place("candidate", 0, "query", WeightField.QUERY, 0, Conversion.BF16);
      place("candidate", 0, "queryBias", WeightField.QUERY_BIAS, 0, Conversion.ROUNDED);
      place("candidate", 0, "keyValue", WeightField.KEY_VALUE, 0, Conversion.BF16);
      place("candidate", 0, "keyValueBias", WeightField.KEY_VALUE_BIAS, 0, Conversion.ROUNDED);
      place("candidate", 0, "codeKey", WeightField.CODE_KEY, 0, Conversion.FP32);
      place("candidate", 0, "codeValue", WeightField.CODE_VALUE, 0, Conversion.FP32);
      place("candidate", 0, "output", WeightField.CANDIDATE_OUTPUT, 0, Conversion.BF16);
      place("candidate", 0, "outputBias", WeightField.CANDIDATE_OUTPUT_BIAS, 0, Conversion.ROUNDED);
      ffn("candidate", 0, SHARED_FFN, WeightField.CANDIDATE_FFN);
      if (architecture.reread()) {
        place("reread", 0, "queryNorm", WeightField.REREAD_QUERY_NORM, 0, Conversion.FP32);
        place("reread", 0, "query", WeightField.REREAD_QUERY, 0, Conversion.BF16);
        place("reread", 0, "queryBias", WeightField.REREAD_QUERY_BIAS, 0, Conversion.ROUNDED);
        String norm = "eventNorm";
        place("reread", 0, norm, WeightField.REREAD_EVENT_NORM, 0, Conversion.FP32);
        place("reread", 0, "keyValue", WeightField.REREAD_KEY_VALUE, 0, Conversion.BF16);
        place(
            "reread", 0, "keyValueBias", WeightField.REREAD_KEY_VALUE_BIAS, 0, Conversion.ROUNDED);
        place("reread", 0, "codeKey", WeightField.REREAD_CODE_KEY, 0, Conversion.FP32);
        place("reread", 0, "codeValue", WeightField.REREAD_CODE_VALUE, 0, Conversion.FP32);
        place("reread", 0, "output", WeightField.REREAD_OUTPUT, 0, Conversion.BF16);
        place("reread", 0, "outputBias", WeightField.REREAD_OUTPUT_BIAS, 0, Conversion.ROUNDED);
      }
      place("discardValue", 0, "norm", WeightField.DISCARD_NORM, 0, Conversion.FP32);
      place("discardValue", 0, "hidden", WeightField.DISCARD_HIDDEN, 0, Conversion.BF16);
      place(
          "discardValue", 0, "hiddenBias", WeightField.DISCARD_HIDDEN_BIAS, 0, Conversion.ROUNDED);
      place("discardValue", 0, "output", WeightField.DISCARD_OUTPUT, 0, Conversion.FP32);
      place("discardValue", 0, "exact", WeightField.DISCARD_EXACT, 0, Conversion.FP32);
      place("gates", 0, "norm", WeightField.GATE_NORM, 0, Conversion.FP32);
      place("gates", 0, "hidden", WeightField.GATE_HIDDEN, 0, Conversion.BF16);
      place("gates", 0, "hiddenBias", WeightField.GATE_HIDDEN_BIAS, 0, Conversion.ROUNDED);
      place("gates", 0, "kindOffset", WeightField.GATE_KIND_OFFSET, 0, Conversion.FP32);
      place("gates", 0, "exactProjection", WeightField.GATE_EXACT_PROJECTION, 0, Conversion.FP32);
      place("gates", 0, "output", WeightField.GATE_OUTPUT, 0, Conversion.FP32);
      place("gates", 0, "outputBias", WeightField.GATE_OUTPUT_BIAS, 0, Conversion.FP32);
      place("gates", 0, "exactOutput", WeightField.GATE_EXACT_OUTPUT, 0, Conversion.FP32);
      long countOffset = -1;
      long segmentsOffset = -1;
      final boolean core = format.has(InputFeature.CANDIDATE_SEGMENTS);
      if (core) {
        String child = "candidateFeatures/count";
        int width = CandidateCodes.COUNT_WIDTH;
        int codes = format.keys().codes() - format.tileKeyCode().first();
        int hidden = architecture.hidden();
        countOffset = cursor;
        append(child, "gate", Conversion.FP32, codes, width);
        append(child, "tile", Conversion.BF16, width, hidden);
        append(child, "output", Conversion.BF16, hidden, width);
        cursor = PackedWeightLayout.align(cursor);
      }
      if (core) {
        String child = "candidateFeatures/segments";
        int width = 2 * architecture.attention();
        segmentsOffset = cursor;
        for (var slot : format.candidateCodes().slots()) {
          append(child, slot.parameter(), Conversion.FP32, slot.rows(), width);
        }
        cursor = PackedWeightLayout.align(cursor);
      }
      List<String> unused = new ArrayList<>();
      for (String key : shapes.keySet()) {
        if (!used.contains(key)) {
          unused.add(key);
        }
      }
      if (!unused.isEmpty()) {
        throw new IllegalArgumentException(
            "inference parameters without a weight pack slot: " + unused);
      }
      return new WeightLayout(
          format,
          architecture,
          header,
          Collections.unmodifiableList(placements),
          Collections.unmodifiableMap(embeddings),
          countOffset,
          segmentsOffset,
          cursor);
    }

    /** 入力の列の位置、領域ごとの幅、手の扱いごとの s の作り方とゲートの種類をヘッダーに書く。 */
    private void schema() {
      long[] relation = shape("encoder", 0, "relation");
      header
          .schema(ColumnIndex.ROW_FIRST_EVENT, RowColumn.FIRST_EVENT.ordinal())
          .schema(ColumnIndex.ROW_EVENTS, RowColumn.EVENTS.ordinal())
          .schema(ColumnIndex.ROW_FIRST_ACTION, RowColumn.FIRST_ACTION.ordinal())
          .schema(ColumnIndex.ROW_ACTIONS, RowColumn.ACTIONS.ordinal())
          .schema(ColumnIndex.ROW_FIRST_TRANSITION, RowColumn.FIRST_TRANSITION.ordinal())
          .schema(ColumnIndex.ROW_TRANSITIONS, RowColumn.TRANSITIONS.ordinal())
          .schema(ColumnIndex.ACTION_ROW, ActionColumn.ROW.ordinal())
          .schema(ColumnIndex.ACTION_FIRST_TRANSITION, ActionColumn.FIRST_TRANSITION.ordinal())
          .schema(ColumnIndex.ACTION_TRANSITIONS, ActionColumn.TRANSITIONS.ordinal())
          .schema(ColumnIndex.ACTION_ROLE, ActionColumn.ROLE.ordinal())
          .schema(ColumnIndex.ACTION_BASE, ActionColumn.BASE.ordinal())
          .schema(ColumnIndex.ACTION_REPRESENTATIVE, ActionColumn.REPRESENTATIVE.ordinal())
          .schema(ColumnIndex.RELATION_TILE, format.relation().column(RelationAttribute.TILE, 0))
          .schema(ColumnIndex.RELATION_OWNER, format.relation().column(RelationAttribute.OWNER, 0))
          .schema(ColumnIndex.RELATION_EVENT, format.relation().column(RelationAttribute.EVENT, 0))
          .schema(
              ColumnIndex.RELATION_RIICHI,
              format.relation().column(RelationAttribute.OWNER_RIICHI_EVENT, 0))
          .schema(
              ColumnIndex.RELATION_LAST_DISCARD,
              format.relation().column(RelationAttribute.OWNER_LAST_DISCARD_EVENT, 0))
          .schema(ColumnIndex.RELATION_LINK, format.relation().column(RelationAttribute.LINK, 0))
          .schema(
              ColumnIndex.RELATION_SAFETY, format.relation().column(RelationAttribute.SAFETY, 0))
          .schema(
              ColumnIndex.TRANSITION_CALL_DISCARD,
              format.transition().column(TransitionField.CALL_DISCARD))
          .schema(
              ColumnIndex.TRANSITION_MAIN_TILE,
              format.transition().column(TransitionField.MAIN_TILE))
          .schema(ColumnIndex.RELATION_ROWS, Math.toIntExact(relation[0]));
      if (Tensor.values().length != InputSection.values().length) {
        throw new IllegalStateException("collection input sections do not match the input tensors");
      }
      for (Tensor tensor : Tensor.values()) {
        InputSection section = InputSection.valueOf(tensor.name());
        if (section.element().bytes() != tensor.element().bytes()) {
          throw new IllegalStateException(
              tensor + " element differs from the collection input section " + section);
        }
        // V4 の候補区分を含め、幅は保存した形が選ぶ入力形式で決まる。
        int width = format.width(tensor);
        header.sectionWidth(section, width);
      }
      for (Role role : format.roles()) {
        header.role(format.roles().indexOf(role), composition(role), gateKind(format, role));
      }
    }

    /** 1 つの型の埋め込みを置く。組の表、表に折り込まない特徴の W、バイアスの順に置く。 */
    private void embedding(EmbeddingSource source) {
      int hidden = architecture.hidden();
      String tableKey = key(source.child(), 0, source.table());
      String projectionKey = key(source.child(), 0, source.projection());
      String biasKey = key(source.child(), 0, source.bias());
      long[] table = shapes.get(tableKey);
      long[] projection = shapes.get(projectionKey);
      int dimension = Math.toIntExact(table[1]);
      int inputs = source.embeddedCount() * dimension + source.denseCount();
      if (projection[0] != hidden || projection[1] != inputs) {
        throw new IllegalArgumentException(
            projectionKey
                + " has shape "
                + Arrays.toString(projection)
                + ", expected ["
                + hidden
                + ", "
                + inputs
                + "]");
      }
      requireShape(biasKey, hidden);
      // 局・席・牌・イベントは、層 0 の QKV まで折り込んだ表（1 行に FP32 を 3A 個）を持つ。遷移は Attention に入らないので持たない。
      boolean folding =
          CollectionCapabilities.foldsLayerZero(hidden)
              && source.slot() != EmbeddingTable.Slot.TRANSITION;
      long foldRow = 3L * architecture.attention() * Float.BYTES;
      List<Group> groups = new ArrayList<>();
      for (int[] members : groupColumns(source)) {
        int rows = 1;
        int[] strides = new int[members.length];
        for (int m = 0; m < members.length; m++) {
          strides[m] = rows;
          rows = Math.multiplyExact(rows, source.vocabulary(members[m]));
        }
        long offset = allocate((long) rows * hidden * Float.BYTES);
        long foldOffset = folding ? allocate(rows * foldRow) : -1;
        groups.add(new Group(members, strides, rows, offset, foldOffset));
      }
      long denseOffset =
          source.denseCount() == 0
              ? 0
              : allocate((long) source.denseCount() * hidden * Short.BYTES);
      long biasOffset = allocate((long) hidden * Float.BYTES);
      long foldOffset = folding ? allocate((source.denseCount() + 2L) * foldRow) : -1;
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
              biasOffset,
              foldOffset);
      embeddings.put(source.slot(), embedding);
      List<EmbeddingTable.Table> tables = new ArrayList<>();
      int[] embedded = source.embedded();
      for (Group group : groups) {
        int[] columns = new int[group.members().length];
        for (int m = 0; m < columns.length; m++) {
          columns[m] = embedded[group.members()[m]];
        }
        tables.add(
            new EmbeddingTable.Table(
                Math.toIntExact(group.offset()),
                group.rows(),
                columns,
                group.strides(),
                Math.toIntExact(group.foldOffset())));
      }
      header.embedding(
          source.slot(),
          new EmbeddingTable(
              EmbeddingSource.section(source.category()),
              EmbeddingSource.section(source.scalar()),
              EmbeddingSource.section(source.exact()),
              tables,
              List.of(source.dense()),
              Math.toIntExact(denseOffset),
              Math.toIntExact(biasOffset),
              Math.toIntExact(foldOffset)));
    }

    /**
     * 埋め込む列を組に分ける。語彙の大きい列から順に、語彙の積が行数の上限に収まり、列の数が {@link EmbeddingTable#MAX_MEMBERS} に
     * 満たない最初の組に入れる。組の中は列の順に並べる。
     * 行数の上限には、組の数が {@link EmbeddingTable#MAX_TABLES} 以下になる最も小さい候補を使う。
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
            if ((long) products.get(g) * vocabulary <= cap
                && groups.get(g).size() < EmbeddingTable.MAX_MEMBERS) {
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
        if (fits && groups.size() <= EmbeddingTable.MAX_TABLES) {
          List<int[]> result = new ArrayList<>();
          for (List<Integer> group : groups) {
            result.add(group.stream().mapToInt(Integer::intValue).sorted().toArray());
          }
          return result;
        }
      }
      throw new IllegalStateException(
          source + " embedding needs more than " + EmbeddingTable.MAX_TABLES + " folded tables");
    }

    /** FFN 1 つ分（ゲイン、上りの射影とバイアス、下りの射影とバイアス）を、FFN のスロットの 1 つの項目に置く。 */
    private void ffn(String child, int layer, FfnNames names, int entry) {
      place(child, layer, names.norm(), WeightField.FFN_NORM, entry, Conversion.FP32);
      place(child, layer, names.up(), WeightField.UP, entry, Conversion.BF16);
      place(child, layer, names.upBias(), WeightField.UP_BIAS, entry, Conversion.ROUNDED);
      place(child, layer, names.down(), WeightField.DOWN, entry, Conversion.BF16);
      place(child, layer, names.downBias(), WeightField.DOWN_BIAS, entry, Conversion.ROUNDED);
    }

    private void place(
        String child,
        int layer,
        String local,
        WeightField weight,
        int entry,
        Conversion conversion) {
      String key = key(child, layer, local);
      long[] parameterShape = shapes.get(key);
      long elements = 1;
      for (long dimension : parameterShape) {
        elements *= dimension;
      }
      if (elements != header.elements(weight, entry)) {
        throw new IllegalArgumentException(
            key
                + " has "
                + elements
                + " elements but the weight pack slot "
                + weight
                + " entry "
                + entry
                + " holds "
                + header.elements(weight, entry));
      }
      long bytesNeeded =
          conversion == Conversion.BF16 ? elements * Short.BYTES : elements * Float.BYTES;
      if (bytesNeeded != header.bytes(weight, entry)) {
        throw new IllegalStateException(
            weight + " conversion " + conversion + " does not match its element type");
      }
      long offset = allocate(bytesNeeded);
      header.weight(weight, entry, offset);
      placements.add(new Placement(key, weight, entry, conversion, offset));
      used.add(key);
    }

    private void requireShape(String key, long... expected) {
      if (!Arrays.equals(shapes.get(key), expected)) {
        throw new IllegalArgumentException(
            key
                + " has shape "
                + Arrays.toString(shapes.get(key))
                + ", expected "
                + Arrays.toString(expected));
      }
    }

    private void append(String child, String local, Conversion conversion, int rows, int columns) {
      String key = key(child, 0, local);
      requireShape(key, rows, columns);
      placements.add(new Placement(key, null, 0, conversion, cursor));
      cursor += (long) rows * columns * (conversion == Conversion.BF16 ? Short.BYTES : Float.BYTES);
      used.add(key);
    }

    private long[] shape(String child, int layer, String local) {
      return shapes.get(key(child, layer, local));
    }

    private String key(String child, int layer, String local) {
      String key = byName.get(child + "/" + layer + "/" + local);
      if (key == null) {
        throw new IllegalArgumentException(
            "the network has no parameter " + child + layer + "." + local);
      }
      return Objects.requireNonNull(key);
    }

    private long allocate(long size) {
      long offset = cursor;
      cursor = PackedWeightLayout.align(cursor + size);
      if (cursor > Integer.MAX_VALUE) {
        throw new IllegalStateException("weight pack exceeds 2 GiB");
      }
      return offset;
    }
  }
}
