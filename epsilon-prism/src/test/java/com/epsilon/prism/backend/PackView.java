package com.epsilon.prism.backend;

import com.epsilon.prism.kernel.CollectionArchitecture;
import com.epsilon.prism.kernel.ColumnIndex;
import com.epsilon.prism.kernel.EmbeddingTable;
import com.epsilon.prism.kernel.InputSection;
import com.epsilon.prism.kernel.PackedWeightLayout;
import com.epsilon.prism.kernel.ScoreComposition;
import com.epsilon.prism.kernel.WeightField;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * 書き出した WeightPack を、native のヘッダーの並び（weight_pack_format.h の PackHeader）で読み直すテスト用のヘルパー。PackedWeightLayout.Header の
 * 書き方を、native と同じインデックスの計算で照合する。
 */
final class PackView {

  private static final int ARCHITECTURE_AT = 0;
  private static final int SCHEMA_AT = ARCHITECTURE_AT + CollectionArchitecture.INTS;
  private static final int SECTIONS_AT = SCHEMA_AT + ColumnIndex.values().length;
  private static final int ROLES_AT = SECTIONS_AT + InputSection.values().length;
  private static final int GATES_AT = ROLES_AT + CollectionArchitecture.MAX_ROLES;
  private static final int WEIGHTS_AT = GATES_AT + CollectionArchitecture.MAX_ROLES;
  private static final int EMBEDDINGS_AT = WEIGHTS_AT + WeightField.ENTRIES;

  final ByteBuffer bytes;

  PackView(ByteBuffer pack) {
    bytes = pack.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    if (bytes.limit() < PackedWeightLayout.HEADER_BYTES) {
      throw new IllegalStateException(
          "a weight pack of " + bytes.limit() + " bytes has no room for the header");
    }
  }

  int integer(int index) {
    return bytes.getInt(index * Integer.BYTES);
  }

  /** ヘッダーにある形の記述（int の列）。 */
  int[] architecture() {
    int[] ints = new int[CollectionArchitecture.INTS];
    for (int i = 0; i < ints.length; i++) {
      ints[i] = integer(ARCHITECTURE_AT + i);
    }
    return ints;
  }

  int schema(ColumnIndex field) {
    return integer(SCHEMA_AT + field.ordinal());
  }

  int width(InputSection section) {
    return integer(SECTIONS_AT + section.ordinal());
  }

  ScoreComposition composition(int role) {
    return ScoreComposition.values()[integer(ROLES_AT + role)];
  }

  int gate(int role) {
    return integer(GATES_AT + role);
  }

  /** 重みの項目の位置（項目は {@link WeightField.Index} の番号）。項目が無ければ例外を投げる。 */
  int weight(WeightField weight, int entry) {
    int offset = integer(WEIGHTS_AT + weight.firstEntry() + entry);
    if (offset < 0) {
      throw new IllegalStateException(weight + " entry " + entry + " is absent");
    }
    return offset;
  }

  /** 重みの項目の位置（項目が無ければ −1）。 */
  int weightOrAbsent(WeightField weight, int entry) {
    return integer(WEIGHTS_AT + weight.firstEntry() + entry);
  }

  float f32(WeightField weight, int entry, int index) {
    return bytes.getFloat(weight(weight, entry) + index * Float.BYTES);
  }

  float bf16(WeightField weight, int entry, int index) {
    return Float.intBitsToFloat(
        (bytes.getShort(weight(weight, entry) + index * Short.BYTES) & 0xFFFF) << 16);
  }

  int u8(WeightField weight, int entry, int index) {
    return bytes.get(weight(weight, entry) + index) & 0xFF;
  }

  /**
   * 1 つの型の埋め込み（native の EmbeddingSpec の並び）。{@code tableFold} は表ごとの T' の位置、{@code fold} は W'_d とそれに続く b'・b_qkv
   * の位置（層 0 の QKV に折り込まない型では −1）。
   */
  record Spec(
      int category,
      int scalar,
      int exact,
      int[] tableOffset,
      int[] tableRows,
      int[][] columns,
      int[][] strides,
      int[] tableFold,
      int[] source,
      int denseOffset,
      int biasOffset,
      int fold) {}

  Spec embedding(EmbeddingTable.Slot slot) {
    int at = EMBEDDINGS_AT + slot.ordinal() * EmbeddingTable.INTS;
    int tables = integer(at + 3);
    int dense = integer(at + 4);
    int[] offsets = new int[tables];
    int[] rows = new int[tables];
    int[] folds = new int[tables];
    int[][] columns = new int[tables][];
    int[][] strides = new int[tables][];
    for (int t = 0; t < tables; t++) {
      int table = at + 8 + t * EmbeddingTable.TABLE_INTS;
      offsets[t] = integer(table);
      rows[t] = integer(table + 1);
      folds[t] = integer(table + 3 + 2 * EmbeddingTable.MAX_MEMBERS);
      int members = integer(table + 2);
      columns[t] = new int[members];
      strides[t] = new int[members];
      for (int m = 0; m < members; m++) {
        columns[t][m] = integer(table + 3 + m);
        strides[t][m] = integer(table + 3 + EmbeddingTable.MAX_MEMBERS + m);
      }
    }
    int[] sources = new int[dense];
    for (int k = 0; k < dense; k++) {
      sources[k] = integer(at + 8 + EmbeddingTable.MAX_TABLES * EmbeddingTable.TABLE_INTS + k);
    }
    return new Spec(
        integer(at),
        integer(at + 1),
        integer(at + 2),
        offsets,
        rows,
        columns,
        strides,
        folds,
        sources,
        integer(at + 5),
        integer(at + 6),
        integer(at + 7));
  }
}
