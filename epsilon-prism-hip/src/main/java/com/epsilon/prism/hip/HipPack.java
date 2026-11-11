package com.epsilon.prism.hip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * WeightPack の形式。先頭にヘッダー（int32、little-endian）を置き、その後に重みを {@link #ALIGNMENT} にそろえて並べる。ヘッダーが
 * 配置を全て書いているので、JNI はネットワークを知らずにバイト列だけを受け取れる。native はデバイスへ写す前にヘッダーを読み、形の記述が
 * 計画のものとビット単位で同じこと、全てのバッファが WeightPack の中にあることを照合する。WeightPack は公開のたびにチェックポイントから
 * 作り、保存はしない。そのためヘッダーは、マジックナンバーも形式のバージョンも持たない（ヘッダーの int の数 {@link #HEADER_INTS} は、
 * {@link HipLibrary} が読み込むときに native と照合する）。
 *
 * <pre>
 * [形の記述 HipArchitecture][入力の列の決まり HipSchema][領域の幅 HipSection][手の扱いごとの作り方 × MAX_ROLES][手の扱いごとのゲート × MAX_ROLES]
 * [重みの位置 HipWeight の項目ごと（なしは −1）][埋め込み HipEmbedding.Slot × HipEmbedding.INTS]
 * </pre>
 */
public final class HipPack {

  /** バッファの位置をそろえる単位（バイト）。 */
  public static final int ALIGNMENT = 256;

  public static final int HEADER_INTS =
      HipArchitecture.INTS
          + HipSchema.values().length
          + HipSection.values().length
          + 2 * HipArchitecture.MAX_ROLES
          + HipWeight.ENTRIES
          + HipEmbedding.Slot.values().length * HipEmbedding.INTS;

  /** ヘッダーのバイト数（{@link #ALIGNMENT} に切り上げた後）。最初のバッファはここから始まる。 */
  public static final int HEADER_BYTES = (int) align((long) HEADER_INTS * Integer.BYTES);

  private static final int ARCHITECTURE_AT = 0;
  private static final int SCHEMA_AT = ARCHITECTURE_AT + HipArchitecture.INTS;
  private static final int SECTIONS_AT = SCHEMA_AT + HipSchema.values().length;
  private static final int ROLES_AT = SECTIONS_AT + HipSection.values().length;
  private static final int GATES_AT = ROLES_AT + HipArchitecture.MAX_ROLES;
  private static final int WEIGHTS_AT = GATES_AT + HipArchitecture.MAX_ROLES;
  private static final int EMBEDDINGS_AT = WEIGHTS_AT + HipWeight.ENTRIES;

  private HipPack() {}

  /** バイト数を {@link #ALIGNMENT} の倍数に切り上げる。 */
  public static long align(long bytes) {
    return Math.ceilDiv(bytes, ALIGNMENT) * ALIGNMENT;
  }

  /**
   * WeightPack のヘッダー。ネットワークの側が、形の記述と入力の宣言から作る。作った後は変えない。
   */
  public static final class Header {

    private final HipArchitecture architecture;
    private final int[] schema = new int[HipSchema.values().length];
    private final int[] sectionWidths = new int[HipSection.values().length];
    private final HipComposition[] compositions = new HipComposition[HipArchitecture.MAX_ROLES];
    private final int[] gates = new int[HipArchitecture.MAX_ROLES];
    private final long[] weights = new long[HipWeight.ENTRIES];
    private final Map<HipEmbedding.Slot, HipEmbedding> embeddings = new EnumMap<>(HipEmbedding.Slot.class);

    /**
     * 空のヘッダーを作る。重みは全てなし、手の扱いごとの作り方は全て {@link HipComposition#ZERO}、ゲートは全て −1 にする。
     *
     * @param architecture 形の記述
     */
    public Header(HipArchitecture architecture) {
      this.architecture = Objects.requireNonNull(architecture, "architecture");
      Arrays.fill(compositions, HipComposition.ZERO);
      Arrays.fill(gates, -1);
      Arrays.fill(weights, -1);
    }

    public HipArchitecture architecture() {
      return architecture;
    }

    public Header schema(HipSchema field, int value) {
      schema[field.ordinal()] = value;
      return this;
    }

    /** 領域の幅（1 個あたりの要素の数）を設定する。 */
    public Header sectionWidth(HipSection section, int width) {
      if (width <= 0) {
        throw new IllegalArgumentException(section + " width must be positive: " + width);
      }
      sectionWidths[section.ordinal()] = width;
      return this;
    }

    /**
     * 1 つの手の扱いについて、手の値 s の作り方とゲートの種類を設定する。
     *
     * @param role 手の扱いの ordinal
     * @param composition s の作り方
     * @param gate ゲートの種類（ゲートを使わなければ −1）
     * @throws IllegalArgumentException 手の扱いかゲートの種類が、形の記述にある数の範囲を外れるとき
     */
    public Header role(int role, HipComposition composition, int gate) {
      if (role < 0 || role >= architecture.roles()) {
        throw new IllegalArgumentException("role ordinal out of range: " + role);
      }
      if (gate < -1 || gate >= architecture.gateKinds()) {
        throw new IllegalArgumentException("gate kind out of range: " + gate);
      }
      compositions[role] = composition;
      gates[role] = gate;
      return this;
    }

    /**
     * 重みのバッファの 1 つの項目について、WeightPack の中の位置を設定する。
     *
     * @param weight 重みのバッファ
     * @param entry 項目（{@link HipWeight.Index} の番号）
     * @param offset WeightPack の中のバイト位置
     * @throws IllegalArgumentException この形にない項目のとき。また、位置が負か、{@link #ALIGNMENT} にそろっていないか、int に
     *     収まらないとき
     */
    public Header weight(HipWeight weight, int entry, long offset) {
      if (!architecture.holds(weight, entry)) {
        throw new IllegalArgumentException(weight + " has no entry " + entry + " in " + architecture);
      }
      if (offset < 0 || offset % ALIGNMENT != 0 || offset > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(weight + " offset " + offset + " must be aligned to " + ALIGNMENT);
      }
      weights[weight.firstEntry() + entry] = offset;
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

    /** 手の扱いのゲートの種類（ゲートを使わなければ −1）。 */
    public int gate(int role) {
      return gates[role];
    }

    /** 重みのバッファの項目が WeightPack のどこにあるか（なければ −1）。 */
    public long weight(HipWeight weight, int entry) {
      if (entry < 0 || entry >= weight.index().entries()) {
        throw new IllegalArgumentException(weight + " has no entry " + entry);
      }
      return weights[weight.firstEntry() + entry];
    }

    /** トークンの型の埋め込み（なければ null）。 */
    public HipEmbedding embedding(HipEmbedding.Slot slot) {
      return embeddings.get(slot);
    }

    /**
     * 重みのバッファの 1 つの項目が持つ要素の数を返す（native の weightElements と同じ式）。領域の幅と、関係のバイアスの表の行数は、
     * 先に設定した値を使う。
     *
     * @param weight 重みのバッファ
     * @param entry 項目
     * @return 要素の数
     */
    public long elements(HipWeight weight, int entry) {
      long h = architecture.hidden();
      long a = architecture.attention();
      long g = architecture.gate();
      long kinds = architecture.gateKinds();
      HipCounts counts = architecture.counts();
      return switch (weight) {
        case TILE_RELATION -> (long) counts.tiles() * counts.tiles();
        case ATTENTION_NORM,
            OUTPUT_BIAS,
            FFN_NORM,
            DOWN_BIAS,
            FINAL_NORM,
            QUERY_NORM,
            CANDIDATE_OUTPUT_BIAS,
            REREAD_QUERY_NORM,
            REREAD_EVENT_NORM,
            REREAD_OUTPUT_BIAS,
            DISCARD_NORM,
            GATE_NORM -> h;
        case QKV -> 3 * a * h;
        case QKV_BIAS -> 3 * a;
        case RELATION_BIAS -> (long) schema(HipSchema.RELATION_ROWS) * architecture.heads();
        case OUTPUT, CANDIDATE_OUTPUT, REREAD_OUTPUT -> h * a;
        case UP, DOWN -> (long) architecture.ffnWidth(entry) * h;
        case UP_BIAS -> architecture.ffnWidth(entry);
        case TILE_ENTRY -> h * h;
        case QUERY, REREAD_QUERY -> a * h;
        case QUERY_BIAS, REREAD_QUERY_BIAS -> a;
        case KEY_VALUE, REREAD_KEY_VALUE -> 2 * a * h;
        case KEY_VALUE_BIAS, REREAD_KEY_VALUE_BIAS -> 2 * a;
        case CODE_KEY, CODE_VALUE -> counts.keyCodes() * a;
        case REREAD_CODE_KEY, REREAD_CODE_VALUE -> counts.rereadCodes() * a;
        case DISCARD_HIDDEN -> g * h;
        case DISCARD_HIDDEN_BIAS, DISCARD_OUTPUT, GATE_HIDDEN_BIAS -> g;
        case DISCARD_EXACT -> sectionWidth(HipSection.DISCARD_EXACT);
        case GATE_HIDDEN -> g * 3 * h;
        case GATE_KIND_OFFSET, GATE_OUTPUT -> kinds * g;
        case GATE_EXACT_PROJECTION -> kinds * g * sectionWidth(HipSection.ACTION_EXACT);
        case GATE_OUTPUT_BIAS -> kinds;
        case GATE_EXACT_OUTPUT -> kinds * sectionWidth(HipSection.ACTION_EXACT);
      };
    }

    /** 重みのバッファの 1 つの項目が占めるバイト数。 */
    public long bytes(HipWeight weight, int entry) {
      return elements(weight, entry) * weight.element().bytes();
    }

    /** native のヘッダーと同じ並びの int の列。 */
    public int[] encode() {
      int[] ints = new int[HEADER_INTS];
      System.arraycopy(architecture.encode(), 0, ints, ARCHITECTURE_AT, HipArchitecture.INTS);
      System.arraycopy(schema, 0, ints, SCHEMA_AT, schema.length);
      System.arraycopy(sectionWidths, 0, ints, SECTIONS_AT, sectionWidths.length);
      for (int role = 0; role < HipArchitecture.MAX_ROLES; role++) {
        ints[ROLES_AT + role] = compositions[role].ordinal();
        ints[GATES_AT + role] = gates[role];
      }
      for (int entry = 0; entry < HipWeight.ENTRIES; entry++) {
        ints[WEIGHTS_AT + entry] = (int) weights[entry];
      }
      for (Map.Entry<HipEmbedding.Slot, HipEmbedding> entry : embeddings.entrySet()) {
        entry.getValue().encode(ints, EMBEDDINGS_AT + entry.getKey().ordinal() * HipEmbedding.INTS);
      }
      return ints;
    }

    /**
     * WeightPack の先頭（位置 0）から little-endian で書く。WeightPack の position は変えない。
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
