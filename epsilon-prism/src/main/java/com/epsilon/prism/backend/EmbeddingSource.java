package com.epsilon.prism.backend;

import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.input.InputSchema.EventField;
import com.epsilon.prism.input.InputSchema.RoundField;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.input.InputSchema.TileField;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.TokenLayout;
import java.util.Arrays;

/**
 * K1 の埋め込みの元: 型ごとの欄の宣言・区画・網のパラメーターの名前。網の埋め込み（TokenEmbedding と CandidateLayer の前半）と同じ
 * 並べ方（埋め込む欄の列の順の [埋め込み × d]、2 値の欄の列の順の旗、数値、厳密な欄）を、入力の宣言から作る。
 */
enum EmbeddingSource {
  /** 局のトークン（stateEmbedding の G）。 */
  ROUND(
      HipEmbedding.Slot.ROUND,
      InputSchema.ROUND,
      RoundField.class,
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
      InputSchema.SEAT,
      SeatField.class,
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
      InputSchema.TILE,
      TileField.class,
      Tensor.TILE_CATEGORY,
      null,
      null,
      "stateEmbedding",
      "tableT",
      "projectT",
      "projectTBias"),
  /** 出来事のトークン（stateEmbedding の E）。 */
  EVENT(
      HipEmbedding.Slot.EVENT,
      InputSchema.EVENT,
      EventField.class,
      Tensor.EVENT_CATEGORY,
      Tensor.EVENT_SCALAR,
      null,
      "stateEmbedding",
      "tableE",
      "projectE",
      "projectEBias"),
  /** 遷移の静的な部分（candidate の table・project）。数値の後に B の道を BF16 にして並べる。 */
  TRANSITION(
      HipEmbedding.Slot.TRANSITION,
      InputSchema.TRANSITION,
      TransitionField.class,
      Tensor.TRANSITION_CATEGORY,
      Tensor.TRANSITION_SCALAR,
      Tensor.DISCARD_EXACT,
      "candidate",
      "table",
      "project",
      "projectBias");

  private final HipEmbedding.Slot slot;
  private final Tensor category;
  private final Tensor scalar;
  private final Tensor exact;
  private final String child;
  private final String table;
  private final String projection;
  private final String bias;
  private final int[] embedded;
  private final int[] vocabularies;
  private final int[] namespaces;
  private final int[] flags;

  <E extends Enum<E> & InputSchema.BoundedField> EmbeddingSource(
      HipEmbedding.Slot slot,
      TokenLayout<E> layout,
      Class<E> fields,
      Tensor category,
      Tensor scalar,
      Tensor exact,
      String child,
      String table,
      String projection,
      String bias) {
    this.slot = slot;
    this.category = category;
    this.scalar = scalar;
    this.exact = exact;
    this.child = child;
    this.table = table;
    this.projection = projection;
    this.bias = bias;
    int[] columnVocabularies = layout.columnVocabularies();
    int[] columnNamespaces = layout.columnOffsets();
    int[] embeddedColumns = new int[layout.width()];
    int[] flagColumns = new int[layout.width()];
    int embeddedCount = 0;
    int flagCount = 0;
    for (E field : fields.getEnumConstants()) {
      FieldSpec spec = field.spec();
      for (int repeat = 0; repeat < spec.repeats(); repeat++) {
        int column = layout.column(field, repeat);
        if (spec.coding() == FieldSpec.Coding.FLAG) {
          flagColumns[flagCount++] = column;
        } else {
          embeddedColumns[embeddedCount++] = column;
        }
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
  }

  HipEmbedding.Slot slot() {
    return slot;
  }

  Tensor category() {
    return category;
  }

  /** 数値の区画（なければ null）。 */
  Tensor scalar() {
    return scalar;
  }

  /** 厳密な区画（なければ null）。 */
  Tensor exact() {
    return exact;
  }

  /** 数値の列の数。 */
  int scalars() {
    return scalar == null ? 0 : scalar.width();
  }

  /** 厳密な列の数。 */
  int exacts() {
    return exact == null ? 0 : exact.width();
  }

  /** 網の子の名前（登録の番号を除いたもの）。 */
  String child() {
    return child;
  }

  /** 埋め込みの表のパラメーターの名前。 */
  String table() {
    return table;
  }

  /** 型ごとの Linear の重みのパラメーターの名前。 */
  String projection() {
    return projection;
  }

  /** 型ごとの Linear の偏りのパラメーターの名前。 */
  String bias() {
    return bias;
  }

  /** 埋め込む欄の列（列の順）。 */
  int[] embedded() {
    return embedded.clone();
  }

  /** 埋め込む欄の語彙。 */
  int vocabulary(int index) {
    return vocabularies[index];
  }

  /** 埋め込む欄の表の名前空間の先頭。 */
  int namespace(int index) {
    return namespaces[index];
  }

  int embeddedCount() {
    return embedded.length;
  }

  /** 2 値の欄の列（列の順）。 */
  int[] flags() {
    return flags.clone();
  }

  /** 表に畳まない特徴の数（旗・数値・厳密な欄）。 */
  int denseCount() {
    return flags.length + scalars() + exacts();
  }

  /**
   * 表に畳まない特徴（Linear の入力の順）。
   *
   * @return 特徴
   */
  HipEmbedding.Dense[] dense() {
    HipEmbedding.Dense[] features = new HipEmbedding.Dense[denseCount()];
    int k = 0;
    for (int flag : flags) {
      features[k++] = new HipEmbedding.Dense(HipEmbedding.Source.FLAG, flag);
    }
    for (int s = 0; s < scalars(); s++) {
      features[k++] = new HipEmbedding.Dense(HipEmbedding.Source.SCALAR, s);
    }
    for (int x = 0; x < exacts(); x++) {
      features[k++] = new HipEmbedding.Dense(HipEmbedding.Source.EXACT, x);
    }
    return features;
  }

  /**
   * HIP の区画（入力の転送の区画と同じ名前）。
   *
   * @param tensor 入力の区画（null 可）
   * @return HIP の区画（null なら null）
   */
  static HipSection section(Tensor tensor) {
    return tensor == null ? null : HipSection.valueOf(tensor.name());
  }
}
