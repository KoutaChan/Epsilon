package com.epsilon.prism.hip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;

/**
 * WeightPack の形式。先頭に頭（int32、little-endian）を置き、その後に重みを {@link #ALIGNMENT} にそろえて並べる。頭は自分を説明する
 * ので、JNI は網を知らずにバイト列だけを受け取る（native は写す前に頭を読んで、全ての置き場が WeightPack の内にあることを照らす）。
 *
 * <pre>
 * [MAGIC][頭の int の数][入力の約束 HipSchema][区画の幅 HipSection][役割の作り方 MAX_ROLES][役割の門 MAX_ROLES]
 * [重みの位置 HipWeight × MAX_DEPTH（なしは −1）][埋め込み HipEmbedding.Slot × HipEmbedding.INTS]
 * </pre>
 */
public final class HipPack {

  /** 頭の印（"PHW1"）。 */
  public static final int MAGIC = 0x31574850;

  /** 置き場の位置のそろえ（バイト）。 */
  public static final int ALIGNMENT = 256;

  /** 手の役割の上限。 */
  public static final int MAX_ROLES = 8;

  public static final int HEADER_INTS =
      2
          + HipSchema.values().length
          + HipSection.values().length
          + 2 * MAX_ROLES
          + HipWeight.values().length * HipShape.MAX_DEPTH
          + HipEmbedding.Slot.values().length * HipEmbedding.INTS;

  /** 頭のバイト数（そろえた後。最初の置き場はここから）。 */
  public static final int HEADER_BYTES = (int) align((long) HEADER_INTS * Integer.BYTES);

  private static final int SCHEMA_AT = 2;
  private static final int SECTIONS_AT = SCHEMA_AT + HipSchema.values().length;
  private static final int ROLES_AT = SECTIONS_AT + HipSection.values().length;
  private static final int GATES_AT = ROLES_AT + MAX_ROLES;
  private static final int WEIGHTS_AT = GATES_AT + MAX_ROLES;
  private static final int EMBEDDINGS_AT = WEIGHTS_AT + HipWeight.values().length * HipShape.MAX_DEPTH;

  private HipPack() {}

  /** bytes を {@link #ALIGNMENT} に切り上げる。 */
  public static long align(long bytes) {
    return Math.ceilDiv(bytes, ALIGNMENT) * ALIGNMENT;
  }

  /**
   * WeightPack の頭。網の側が入力の宣言と網の形から作り、作った後は変えない。
   */
  public static final class Header {

    private final int[] schema = new int[HipSchema.values().length];
    private final int[] sectionWidths = new int[HipSection.values().length];
    private final HipComposition[] compositions = new HipComposition[MAX_ROLES];
    private final int[] gates = new int[MAX_ROLES];
    private final long[][] weights = new long[HipWeight.values().length][HipShape.MAX_DEPTH];
    private final Map<HipEmbedding.Slot, HipEmbedding> embeddings = new EnumMap<>(HipEmbedding.Slot.class);

    /** 空の頭（重みは全てなし、役割は全て {@link HipComposition#ZERO}）。 */
    public Header() {
      Arrays.fill(compositions, HipComposition.ZERO);
      Arrays.fill(gates, -1);
      for (long[] layers : weights) {
        Arrays.fill(layers, -1);
      }
    }

    public Header schema(HipSchema field, int value) {
      schema[field.ordinal()] = value;
      return this;
    }

    /** 区画の幅（1 個あたりの要素の数）を置く。 */
    public Header sectionWidth(HipSection section, int width) {
      if (width <= 0) {
        throw new IllegalArgumentException(section + " width must be positive: " + width);
      }
      sectionWidths[section.ordinal()] = width;
      return this;
    }

    /**
     * 役割の作り方と門の種類を置く。
     *
     * @param role 役割の ordinal
     * @param composition 作り方
     * @param gate 門の種類（門でなければ −1）
     */
    public Header role(int role, HipComposition composition, int gate) {
      if (role < 0 || role >= MAX_ROLES) {
        throw new IllegalArgumentException("role ordinal out of range: " + role);
      }
      compositions[role] = composition;
      gates[role] = gate;
      return this;
    }

    /** 重みの置き場の位置を置く（層を持たない置き場は層 0）。 */
    public Header weight(HipWeight weight, int layer, long offset) {
      if (layer < 0 || layer >= HipShape.MAX_DEPTH || (!weight.layered() && layer != 0)) {
        throw new IllegalArgumentException(weight + " has no layer " + layer);
      }
      if (offset < 0 || offset % ALIGNMENT != 0 || offset > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(weight + " offset " + offset + " must be aligned to " + ALIGNMENT);
      }
      weights[weight.ordinal()][layer] = offset;
      return this;
    }

    public Header embedding(HipEmbedding.Slot slot, HipEmbedding embedding) {
      embeddings.put(slot, embedding);
      return this;
    }

    public int schema(HipSchema field) {
      return schema[field.ordinal()];
    }

    public int sectionWidth(HipSection section) {
      return sectionWidths[section.ordinal()];
    }

    public HipComposition composition(int role) {
      return compositions[role];
    }

    /** 役割の門の種類（門でなければ −1）。 */
    public int gate(int role) {
      return gates[role];
    }

    /** 重みの置き場の位置（なしは −1）。 */
    public long weight(HipWeight weight, int layer) {
      return weights[weight.ordinal()][layer];
    }

    /** 型の埋め込み（なしは null）。 */
    public HipEmbedding embedding(HipEmbedding.Slot slot) {
      return embeddings.get(slot);
    }

    /** native の頭の並びの int の列。 */
    public int[] encode() {
      int[] ints = new int[HEADER_INTS];
      ints[0] = MAGIC;
      ints[1] = HEADER_INTS;
      System.arraycopy(schema, 0, ints, SCHEMA_AT, schema.length);
      System.arraycopy(sectionWidths, 0, ints, SECTIONS_AT, sectionWidths.length);
      for (int role = 0; role < MAX_ROLES; role++) {
        ints[ROLES_AT + role] = compositions[role].ordinal();
        ints[GATES_AT + role] = gates[role];
      }
      for (HipWeight weight : HipWeight.values()) {
        for (int layer = 0; layer < HipShape.MAX_DEPTH; layer++) {
          ints[WEIGHTS_AT + weight.ordinal() * HipShape.MAX_DEPTH + layer] =
              (int) weights[weight.ordinal()][layer];
        }
      }
      for (Map.Entry<HipEmbedding.Slot, HipEmbedding> entry : embeddings.entrySet()) {
        entry.getValue().encode(ints, EMBEDDINGS_AT + entry.getKey().ordinal() * HipEmbedding.INTS);
      }
      return ints;
    }

    /**
     * WeightPack の先頭に書く（位置 0 から、little-endian。pack の位置は変えない）。
     *
     * @param pack WeightPack
     */
    public void write(ByteBuffer pack) {
      ByteBuffer view = pack.duplicate().order(ByteOrder.LITTLE_ENDIAN);
      view.position(0);
      view.asIntBuffer().put(encode());
    }
  }
}
