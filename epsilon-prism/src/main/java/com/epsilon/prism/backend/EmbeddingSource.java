package com.epsilon.prism.backend;

import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.TokenLayout;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

/**
 * 埋め込みのカーネル（K1）に渡す、1 つの入力の形式・1 つのトークンの型の情報。フィールドの並び、入力の領域、ネットワークの
 * パラメーターの名前を持つ。並べ方はネットワークの埋め込み（TokenEmbedding と CandidateLayer の前半）と同じで、形式の宣言から作る。
 * 埋め込むフィールドを列の順に [埋め込み × d] で並べ、その後に 2 値のフィールドのフラグ（列の順）、数値、厳密な値を並べる。
 */
final class EmbeddingSource {

  /** トークンの型ごとの、形式に依らない部分。書き先、入力の領域、ネットワークのパラメーターの名前を持つ。 */
  enum Kind {
    /** 局のトークン（stateEmbedding の G）。 */
    ROUND(
        HipEmbedding.Slot.ROUND,
        InputFormat::round,
        Tensor.ROUND_CATEGORY,
        Tensor.ROUND_SCALAR,
        null,
        "stateEmbedding",
        "tableG",
        "projectG",
        "projectGBias"),
    /** 席のトークン（stateEmbedding の P）。 */
    SEAT(
        HipEmbedding.Slot.SEAT,
        InputFormat::seat,
        Tensor.SEAT_CATEGORY,
        Tensor.SEAT_SCALAR,
        null,
        "stateEmbedding",
        "tableP",
        "projectP",
        "projectPBias"),
    /** 牌のトークン（stateEmbedding の T）。 */
    TILE(
        HipEmbedding.Slot.TILE,
        InputFormat::tile,
        Tensor.TILE_CATEGORY,
        null,
        null,
        "stateEmbedding",
        "tableT",
        "projectT",
        "projectTBias"),
    /** イベントのトークン（stateEmbedding の E）。 */
    EVENT(
        HipEmbedding.Slot.EVENT,
        InputFormat::event,
        Tensor.EVENT_CATEGORY,
        Tensor.EVENT_SCALAR,
        null,
        "stateEmbedding",
        "tableE",
        "projectE",
        "projectEBias"),
    /** 遷移の静的な部分（candidate の table・project）。数値の後に、打牌の値 B のヘッドが読む厳密な値を BF16 にして並べる。 */
    TRANSITION(
        HipEmbedding.Slot.TRANSITION,
        InputFormat::transition,
        Tensor.TRANSITION_CATEGORY,
        Tensor.TRANSITION_SCALAR,
        Tensor.DISCARD_EXACT,
        "candidate",
        "table",
        "project",
        "projectBias");

    private final HipEmbedding.Slot slot;
    private final Function<InputFormat, TokenLayout<?>> layout;
    private final Tensor category;
    private final Tensor scalar;
    private final Tensor exact;
    private final String child;
    private final String table;
    private final String projection;
    private final String bias;

    Kind(
        HipEmbedding.Slot slot,
        Function<InputFormat, TokenLayout<?>> layout,
        Tensor category,
        Tensor scalar,
        Tensor exact,
        String child,
        String table,
        String projection,
        String bias) {
      this.slot = slot;
      this.layout = layout;
      this.category = category;
      this.scalar = scalar;
      this.exact = exact;
      this.child = child;
      this.table = table;
      this.projection = projection;
      this.bias = bias;
    }
  }

  private final Kind kind;
  private final int[] embedded;
  private final int[] vocabularies;
  private final int[] namespaces;
  private final int[] flags;
  private final int scalars;
  private final int exacts;

  private EmbeddingSource(Kind kind, InputFormat format) {
    this.kind = kind;
    TokenLayout<?> layout = kind.layout.apply(format);
    int[] columnVocabularies = layout.columnVocabularies();
    int[] columnNamespaces = layout.columnOffsets();
    int[] embeddedColumns = new int[layout.width()];
    int[] flagColumns = new int[layout.width()];
    int embeddedCount = 0;
    int flagCount = 0;
    for (int column = 0; column < layout.width(); column++) {
      if (layout.coding(column) == FieldSpec.Coding.FLAG) {
        flagColumns[flagCount++] = column;
      } else {
        embeddedColumns[embeddedCount++] = column;
      }
    }
    embedded = Arrays.copyOf(embeddedColumns, embeddedCount);
    flags = Arrays.copyOf(flagColumns, flagCount);
    vocabularies = new int[embeddedCount];
    namespaces = new int[embeddedCount];
    for (int i = 0; i < embeddedCount; i++) {
      vocabularies[i] = columnVocabularies[embedded[i]];
      namespaces[i] = columnNamespaces[embedded[i]];
    }
    scalars = kind.scalar == null ? 0 : format.width(kind.scalar);
    exacts = kind.exact == null ? 0 : format.width(kind.exact);
  }

  /**
   * 入力の形式から、トークンの型ごとに埋め込みの情報を作って返す。
   *
   * @param format 入力の形式
   * @return トークンの型の宣言の順に並べた情報
   */
  static List<EmbeddingSource> of(InputFormat format) {
    return Arrays.stream(Kind.values()).map(kind -> new EmbeddingSource(kind, format)).toList();
  }

  HipEmbedding.Slot slot() {
    return kind.slot;
  }

  Tensor category() {
    return kind.category;
  }

  /** 数値の領域（なければ null）。 */
  Tensor scalar() {
    return kind.scalar;
  }

  /** 厳密な値の領域（なければ null）。 */
  Tensor exact() {
    return kind.exact;
  }

  /** 数値の列の数。 */
  int scalars() {
    return scalars;
  }

  /** 厳密な値の列の数。 */
  int exacts() {
    return exacts;
  }

  /** ネットワークの子ブロックの名前（登録の番号を除く）。 */
  String child() {
    return kind.child;
  }

  /** 埋め込みの表のパラメーターの名前。 */
  String table() {
    return kind.table;
  }

  /** 型ごとの Linear の重みのパラメーターの名前。 */
  String projection() {
    return kind.projection;
  }

  /** 型ごとの Linear のバイアスのパラメーターの名前。 */
  String bias() {
    return kind.bias;
  }

  /** 埋め込むフィールドの列（列の順）。 */
  int[] embedded() {
    return embedded.clone();
  }

  /** 埋め込むフィールドの語彙の大きさ。 */
  int vocabulary(int index) {
    return vocabularies[index];
  }

  /** 埋め込むフィールドが埋め込みの表の中で使う行の先頭。 */
  int namespace(int index) {
    return namespaces[index];
  }

  int embeddedCount() {
    return embedded.length;
  }

  /** 2 値のフィールドの列（列の順）。 */
  int[] flags() {
    return flags.clone();
  }

  /** 表に折り込まない特徴の数（フラグ・数値・厳密な値）。 */
  int denseCount() {
    return flags.length + scalars + exacts;
  }

  /**
   * 表に折り込まない特徴を、Linear の入力の順に返す。
   *
   * @return 特徴（フラグ、数値、厳密な値の順）
   */
  HipEmbedding.Dense[] dense() {
    HipEmbedding.Dense[] features = new HipEmbedding.Dense[denseCount()];
    int k = 0;
    for (int flag : flags) {
      features[k++] = new HipEmbedding.Dense(HipEmbedding.Source.FLAG, flag);
    }
    for (int s = 0; s < scalars; s++) {
      features[k++] = new HipEmbedding.Dense(HipEmbedding.Source.SCALAR, s);
    }
    for (int x = 0; x < exacts; x++) {
      features[k++] = new HipEmbedding.Dense(HipEmbedding.Source.EXACT, x);
    }
    return features;
  }

  /**
   * 入力の領域と同じ名前の HIP の領域を返す。
   *
   * @param tensor 入力の領域（null でもよい）
   * @return HIP の領域。入力の領域が null なら null
   */
  static HipSection section(Tensor tensor) {
    return tensor == null ? null : HipSection.valueOf(tensor.name());
  }
}
