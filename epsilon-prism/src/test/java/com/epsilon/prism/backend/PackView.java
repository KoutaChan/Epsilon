package com.epsilon.prism.backend;

import com.epsilon.prism.hip.HipComposition;
import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipPack;
import com.epsilon.prism.hip.HipSchema;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.hip.HipShape;
import com.epsilon.prism.hip.HipWeight;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * 書いた WeightPack を native の頭の並び（prism_hip_format.h の PackHeader）で読み直す試験の道具。HipPack.Header の書き方を、native と
 * 同じ添字の計算で照らす。
 */
final class PackView {

  private static final int SCHEMA_AT = 2;
  private static final int SECTIONS_AT = SCHEMA_AT + HipSchema.values().length;
  private static final int ROLES_AT = SECTIONS_AT + HipSection.values().length;
  private static final int GATES_AT = ROLES_AT + HipPack.MAX_ROLES;
  private static final int WEIGHTS_AT = GATES_AT + HipPack.MAX_ROLES;
  private static final int EMBEDDINGS_AT = WEIGHTS_AT + HipWeight.values().length * HipShape.MAX_DEPTH;

  final ByteBuffer bytes;

  PackView(ByteBuffer pack) {
    bytes = pack.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    if (integer(0) != HipPack.MAGIC || integer(1) != HipPack.HEADER_INTS) {
      throw new IllegalStateException("not a PRISM HIP weight pack");
    }
  }

  int integer(int index) {
    return bytes.getInt(index * Integer.BYTES);
  }

  int schema(HipSchema field) {
    return integer(SCHEMA_AT + field.ordinal());
  }

  int width(HipSection section) {
    return integer(SECTIONS_AT + section.ordinal());
  }

  HipComposition composition(int role) {
    return HipComposition.values()[integer(ROLES_AT + role)];
  }

  int gate(int role) {
    return integer(GATES_AT + role);
  }

  int weight(HipWeight weight, int layer) {
    int offset = integer(WEIGHTS_AT + weight.ordinal() * HipShape.MAX_DEPTH + (weight.layered() ? layer : 0));
    if (offset < 0) {
      throw new IllegalStateException(weight + " layer " + layer + " is absent");
    }
    return offset;
  }

  float f32(HipWeight weight, int layer, int index) {
    return bytes.getFloat(weight(weight, layer) + index * Float.BYTES);
  }

  float bf16(HipWeight weight, int layer, int index) {
    return Float.intBitsToFloat((bytes.getShort(weight(weight, layer) + index * Short.BYTES) & 0xFFFF) << 16);
  }

  int u8(HipWeight weight, int layer, int index) {
    return bytes.get(weight(weight, layer) + index) & 0xFF;
  }

  /** 1 つの型の埋め込み（native の EmbeddingSpec の並び）。 */
  record Spec(
      int category,
      int scalar,
      int exact,
      int[] tableOffset,
      int[] tableRows,
      int[][] columns,
      int[][] strides,
      int[] source,
      int denseOffset,
      int biasOffset) {}

  Spec embedding(HipEmbedding.Slot slot) {
    int at = EMBEDDINGS_AT + slot.ordinal() * HipEmbedding.INTS;
    int tables = integer(at + 3);
    int dense = integer(at + 4);
    int[] offsets = new int[tables];
    int[] rows = new int[tables];
    int[][] columns = new int[tables][];
    int[][] strides = new int[tables][];
    for (int t = 0; t < tables; t++) {
      int table = at + 8 + t * HipEmbedding.TABLE_INTS;
      offsets[t] = integer(table);
      rows[t] = integer(table + 1);
      int members = integer(table + 2);
      columns[t] = new int[members];
      strides[t] = new int[members];
      for (int m = 0; m < members; m++) {
        columns[t][m] = integer(table + 3 + m);
        strides[t][m] = integer(table + 3 + HipEmbedding.MAX_MEMBERS + m);
      }
    }
    int[] sources = new int[dense];
    for (int k = 0; k < dense; k++) {
      sources[k] = integer(at + 8 + HipEmbedding.MAX_TABLES * HipEmbedding.TABLE_INTS + k);
    }
    return new Spec(
        integer(at), integer(at + 1), integer(at + 2), offsets, rows, columns, strides, sources, integer(at + 5),
        integer(at + 6));
  }
}
