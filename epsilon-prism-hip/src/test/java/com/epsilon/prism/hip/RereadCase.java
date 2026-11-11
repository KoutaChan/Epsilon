package com.epsilon.prism.hip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * K9 のテストの入力。2 段目のある形の記述と、行ごとのイベントと遷移の数で決まる 1 つのバッチに、乱数で作った重みと入力（BF16 は
 * 丸めたビット）を置く。WeightPack は K9 が読む重みだけを持つ。blob は全てのセクションを持つが、値を入れるのは K9 が読む所だけで、
 * ほかは 0。K9 が読むのは、遷移のカテゴリ、遷移のキーのコード、B のヘッドが FP32 のまま読むフィールド、イベントの関係の属性の牌、
 * 行ごとのインデックスの列。列と幅は、{@link TestPack#header} のヘッダーの決まりに従う。
 */
final class RereadCase {

  /** K9 が読む重み（項目は全て 0 番）。 */
  static final HipWeight[] SLOTS = {
    HipWeight.TILE_RELATION,
    HipWeight.REREAD_QUERY_NORM,
    HipWeight.REREAD_QUERY,
    HipWeight.REREAD_QUERY_BIAS,
    HipWeight.REREAD_CODE_KEY,
    HipWeight.REREAD_CODE_VALUE,
    HipWeight.REREAD_OUTPUT,
    HipWeight.REREAD_OUTPUT_BIAS,
    HipWeight.DISCARD_NORM,
    HipWeight.DISCARD_HIDDEN,
    HipWeight.DISCARD_HIDDEN_BIAS,
    HipWeight.DISCARD_OUTPUT,
    HipWeight.DISCARD_EXACT
  };

  /** 牌を持たない値の割合（遷移のカテゴリの 2 列と、イベントの牌に使う）。τ かイベントの牌が無いキーは、牌の関係を含まないコードを引く。 */
  private static final double NO_TILE = 0.3;

  /**
   * バッチの 1 行。
   *
   * @param events イベントの数
   * @param transitions 遷移の数（1 以上）
   */
  record Row(int events, int transitions) {}

  /**
   * K9 の重み（BF16 の行列は丸めたビット、ほかは FP32）。
   *
   * @param tileRelation 牌の関係の種類 [牌²]
   * @param queryNorm N_q のゲイン [H]
   * @param query W_q [A, H]
   * @param queryBias b_q [A]
   * @param codeKey E_k [2 段目のコード, A]
   * @param codeValue E_v [2 段目のコード, A]
   * @param output W_O [H, A]
   * @param outputBias b_O [H]
   * @param discardNorm N_B のゲイン [H]
   * @param discardHidden W_B [G, H]
   * @param discardHiddenBias c_B [G]
   * @param discardOutput wB [G]
   * @param discardExact B のヘッドが FP32 のまま読むフィールドの係数 uBx [フィールド]
   */
  record Weights(
      byte[] tileRelation,
      float[] queryNorm,
      short[] query,
      float[] queryBias,
      float[] codeKey,
      float[] codeValue,
      short[] output,
      float[] outputBias,
      float[] discardNorm,
      short[] discardHidden,
      float[] discardHiddenBias,
      float[] discardOutput,
      float[] discardExact) {

    /** 出力の射影 W_O と b_O を 0 にした写し。 */
    Weights withoutOutput() {
      return new Weights(
          tileRelation,
          queryNorm,
          query,
          queryBias,
          codeKey,
          codeValue,
          new short[output.length],
          new float[outputBias.length],
          discardNorm,
          discardHidden,
          discardHiddenBias,
          discardOutput,
          discardExact);
    }
  }

  final HipArchitecture architecture;
  final HipPack.Header header;
  final List<Row> rows;
  final int events;
  final int transitions;
  /** 遷移ごとの、属する行の番号。 */
  final int[] transitionRow;
  /** 行ごとの、最初のイベントの番号。 */
  final int[] firstEvent;
  /** 行ごとの、最初の遷移の番号。 */
  final int[] firstTransition;
  /** z1 [遷移, H]。 */
  final short[] start;
  /** REREAD_MEMORY の領域の値。局・席・牌の [行, F, 2A] の後に、イベントの [イベント, 2A] が続く。 */
  final short[] memory;
  /** 遷移のカテゴリ [遷移, 幅]。 */
  final byte[] category;
  /** 遷移ごとの、キーのコード [遷移, F]。 */
  final byte[] codes;
  /** イベントの関係の属性の牌 [イベント]（牌の ID。0 は牌なし）。 */
  final byte[] eventTiles;
  /** B のヘッドが FP32 のまま読むフィールド [遷移, 幅]。 */
  final float[] exact;
  final Weights weights;

  private RereadCase(
      HipArchitecture architecture,
      List<Row> rows,
      short[] start,
      short[] memory,
      byte[] category,
      byte[] codes,
      byte[] eventTiles,
      float[] exact,
      Weights weights) {
    this.architecture = architecture;
    this.header = TestPack.header(architecture);
    this.rows = List.copyOf(rows);
    this.start = start;
    this.memory = memory;
    this.category = category;
    this.codes = codes;
    this.eventTiles = eventTiles;
    this.exact = exact;
    this.weights = weights;
    firstEvent = new int[rows.size()];
    firstTransition = new int[rows.size()];
    int eventCount = 0;
    int transitionCount = 0;
    for (int r = 0; r < rows.size(); r++) {
      firstEvent[r] = eventCount;
      firstTransition[r] = transitionCount;
      eventCount += rows.get(r).events();
      transitionCount += rows.get(r).transitions();
    }
    events = eventCount;
    transitions = transitionCount;
    transitionRow = new int[transitions];
    for (int r = 0; r < rows.size(); r++) {
      Arrays.fill(transitionRow, firstTransition[r], firstTransition[r] + rows.get(r).transitions(), r);
    }
  }

  /**
   * 2 段目のある形の記述を返す。バックボーンは v2 の形で、候補層の FFN の幅は 512。
   *
   * @param counts 四人打ちか三人打ちかで決まる数
   * @param gate ヘッドの隠れ層の幅 G
   * @param gateKinds ゲートの種類の数
   * @param roles 価値グラフでの手の扱いの種類の数
   * @return 形の記述
   */
  static HipArchitecture architecture(HipCounts counts, int gate, int gateKinds, int roles) {
    return new HipArchitecture(
        HipArchitecture.HIDDEN,
        1e-5f,
        0.25f,
        0.01f,
        List.of(
            new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(512)),
            new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Shared(512))),
        512,
        CandidateStages.REREAD,
        gate,
        gateKinds,
        roles,
        counts);
  }

  /**
   * 学習したネットワークに近い大きさの正規乱数で、重みと入力を作る。標準偏差は、行列が 1/√in、バイアスが 0.02、E が 0.5。ゲインは
   * 平均 1、標準偏差 0.1。W_O だけは、出力の枝が z1 と同じくらいの大きさになるよう 2/√A にする。最初の遷移は τ なし、最初のイベントは牌なしにして、牌の関係を
   * 含まないコードを必ず含める。
   *
   * @param seed 乱数の種
   * @param architecture 形の記述
   * @param rows 行ごとのイベントと遷移の数
   * @return 入力
   */
  static RereadCase random(long seed, HipArchitecture architecture, List<Row> rows) {
    Random random = new Random(seed);
    HipPack.Header header = TestPack.header(architecture);
    HipCounts counts = architecture.counts();
    int h = architecture.hidden();
    int a = architecture.attention();
    int g = architecture.gate();
    int fixed = counts.fixed();
    int tiles = counts.tiles();
    int eventCount = rows.stream().mapToInt(Row::events).sum();
    int transitionCount = rows.stream().mapToInt(Row::transitions).sum();
    int categoryWidth = header.sectionWidth(HipSection.TRANSITION_CATEGORY);
    int exactWidth = header.sectionWidth(HipSection.DISCARD_EXACT);
    int callDiscard = header.schema(HipSchema.TRANSITION_CALL_DISCARD);
    int mainTile = header.schema(HipSchema.TRANSITION_MAIN_TILE);
    byte[] category = new byte[transitionCount * categoryWidth];
    for (int t = 0; t < transitionCount; t++) {
      category[t * categoryWidth + callDiscard] = (byte) (t > 0 && random.nextDouble() >= 0.7 ? tile(random, tiles) : 0);
      category[t * categoryWidth + mainTile] = (byte) (t > 0 ? tileOrNone(random, tiles) : 0);
    }
    byte[] codes = new byte[transitionCount * fixed];
    for (int i = 0; i < codes.length; i++) {
      codes[i] = (byte) random.nextInt(counts.keyCodes());
    }
    byte[] eventTiles = new byte[eventCount];
    for (int e = 0; e < eventCount; e++) {
      eventTiles[e] = (byte) (e > 0 ? tileOrNone(random, tiles) : 0);
    }
    byte[] tileRelation = new byte[tiles * tiles];
    for (int i = 0; i < tileRelation.length; i++) {
      tileRelation[i] = (byte) random.nextInt(counts.tileRelations());
    }
    int keys = rows.size() * fixed + eventCount;
    Weights weights =
        new Weights(
            tileRelation,
            fp32(random, h, 1.0, 0.1),
            bf16(random, a * h, 1 / Math.sqrt(h)),
            fp32(random, a, 0.0, 0.02),
            fp32(random, counts.rereadCodes() * a, 0.0, 0.5),
            fp32(random, counts.rereadCodes() * a, 0.0, 0.5),
            bf16(random, h * a, 2 / Math.sqrt(a)),
            fp32(random, h, 0.0, 0.02),
            fp32(random, h, 1.0, 0.1),
            bf16(random, g * h, 1 / Math.sqrt(h)),
            fp32(random, g, 0.0, 0.02),
            fp32(random, g, 0.0, 1 / Math.sqrt(g)),
            fp32(random, exactWidth, 0.0, 0.1));
    return new RereadCase(
        architecture,
        rows,
        bf16(random, transitionCount * h, 1.0),
        bf16(random, keys * 2 * a, 1.0),
        category,
        codes,
        eventTiles,
        fp32(random, transitionCount * exactWidth, 0.0, 1.0),
        weights);
  }

  /** 出力の射影 W_O と b_O を 0 にした写し。z は z1 とビットで一致するはず。 */
  RereadCase withoutOutput() {
    return new RereadCase(
        architecture, rows, start, memory, category, codes, eventTiles, exact, weights.withoutOutput());
  }

  /**
   * {@code other} の行を前に足した写しを返す。重みはこの入力のものを使う。この入力の遷移は、バッチの中で別の位置に来る。
   *
   * @param other 前に足す入力（同じ形の記述）
   * @return 足した入力
   */
  RereadCase prepend(RereadCase other) {
    int rowKeys = architecture.counts().fixed() * 2 * architecture.attention();
    int otherFixed = other.rows.size() * rowKeys;
    int thisFixed = rows.size() * rowKeys;
    short[] keys = new short[other.memory.length + memory.length];
    System.arraycopy(other.memory, 0, keys, 0, otherFixed);
    System.arraycopy(memory, 0, keys, otherFixed, thisFixed);
    System.arraycopy(other.memory, otherFixed, keys, otherFixed + thisFixed, other.memory.length - otherFixed);
    System.arraycopy(memory, thisFixed, keys, other.memory.length + thisFixed, memory.length - thisFixed);
    List<Row> joined = new ArrayList<>(other.rows);
    joined.addAll(rows);
    return new RereadCase(
        architecture,
        joined,
        concat(other.start, start),
        keys,
        concat(other.category, category),
        concat(other.codes, codes),
        concat(other.eventTiles, eventTiles),
        concat(other.exact, exact),
        weights);
  }

  /**
   * 計画の形を返す。合法手は行ごとに 1 つとする。
   *
   * @param architecture 形の記述
   * @param rows 行の容量
   * @param events イベントの容量
   * @param transitions 遷移の容量
   * @return 計画の形
   */
  static HipShape shape(HipArchitecture architecture, int rows, int events, int transitions) {
    HipShape.Capacity capacity =
        new HipShape.Capacity(
            rows, events, transitions, rows, layout(architecture, rows, events, transitions).bytes());
    return new HipShape(architecture, capacity, packBytes(architecture));
  }

  /** このバッチの数と、blob のセクションの位置。 */
  HipBatch batch() {
    return layout(architecture, rows.size(), events, transitions);
  }

  /** 入力の blob を作る（ダイレクトの ByteBuffer、little-endian）。 */
  ByteBuffer blob() {
    HipBatch batch = batch();
    ByteBuffer blob = TestPack.direct(batch.bytes());
    int relationWidth = header.sectionWidth(HipSection.RELATION);
    int relationTile = header.schema(HipSchema.RELATION_TILE);
    int fixedTokens = architecture.counts().fixed() * rows.size();
    for (int e = 0; e < events; e++) {
      blob.put(at(batch, HipSection.RELATION, (long) (fixedTokens + e) * relationWidth + relationTile), eventTiles[e]);
    }
    for (int i = 0; i < category.length; i++) {
      blob.put(at(batch, HipSection.TRANSITION_CATEGORY, i), category[i]);
    }
    for (int i = 0; i < codes.length; i++) {
      blob.put(at(batch, HipSection.TRANSITION_CODE, i), codes[i]);
    }
    for (int i = 0; i < exact.length; i++) {
      blob.putFloat(at(batch, HipSection.DISCARD_EXACT, (long) i * Float.BYTES), exact[i]);
    }
    int rowWidth = header.sectionWidth(HipSection.ROW_COLUMN);
    for (int r = 0; r < rows.size(); r++) {
      int[] columns = new int[rowWidth];
      columns[header.schema(HipSchema.ROW_FIRST_EVENT)] = firstEvent[r];
      columns[header.schema(HipSchema.ROW_EVENTS)] = rows.get(r).events();
      columns[header.schema(HipSchema.ROW_FIRST_ACTION)] = r;
      columns[header.schema(HipSchema.ROW_ACTIONS)] = 1;
      columns[header.schema(HipSchema.ROW_FIRST_TRANSITION)] = firstTransition[r];
      columns[header.schema(HipSchema.ROW_TRANSITIONS)] = rows.get(r).transitions();
      for (int c = 0; c < rowWidth; c++) {
        blob.putInt(at(batch, HipSection.ROW_COLUMN, (long) (r * rowWidth + c) * Integer.BYTES), columns[c]);
      }
    }
    return blob;
  }

  /** 作業領域のうち、指定した領域に写す入力。TRANSITION には z1、REREAD_MEMORY には局・席・牌とイベントのキーとバリューを写す。 */
  ByteBuffer input(HipRegion region) {
    short[] values =
        switch (region) {
          case TRANSITION -> start;
          case REREAD_MEMORY -> memory;
          default -> throw new IllegalArgumentException("K9 reads only the transition and reread memory regions");
        };
    ByteBuffer buffer = TestPack.direct((long) values.length * Short.BYTES);
    buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(values);
    return buffer;
  }

  /** K9 が読み書きする領域の、このバッチでのバイト数。 */
  long bytes(HipRegion region) {
    return switch (region) {
      case INPUT -> batch().bytes();
      case TRANSITION -> (long) start.length * Short.BYTES;
      case VALUE -> (long) transitions * Float.BYTES;
      case REREAD_MEMORY -> (long) memory.length * Short.BYTES;
      default -> throw new IllegalArgumentException("K9 does not use " + region);
    };
  }

  /** WeightPack を作る（ダイレクトの ByteBuffer、little-endian、ヘッダーつき）。 */
  ByteBuffer pack() {
    HipPack.Header placed = TestPack.header(architecture);
    Map<HipWeight, Long> offsets = offsets(architecture);
    for (HipWeight weight : SLOTS) {
      placed.weight(weight, 0, offsets.get(weight));
    }
    ByteBuffer pack = TestPack.direct(packBytes(architecture));
    placed.write(pack);
    for (HipWeight weight : SLOTS) {
      ByteBuffer at = pack.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(Math.toIntExact(offsets.get(weight)));
      switch (weight) {
        case TILE_RELATION -> at.put(weights.tileRelation());
        case REREAD_QUERY_NORM -> at.asFloatBuffer().put(weights.queryNorm());
        case REREAD_QUERY -> at.asShortBuffer().put(weights.query());
        case REREAD_QUERY_BIAS -> at.asFloatBuffer().put(weights.queryBias());
        case REREAD_CODE_KEY -> at.asFloatBuffer().put(weights.codeKey());
        case REREAD_CODE_VALUE -> at.asFloatBuffer().put(weights.codeValue());
        case REREAD_OUTPUT -> at.asShortBuffer().put(weights.output());
        case REREAD_OUTPUT_BIAS -> at.asFloatBuffer().put(weights.outputBias());
        case DISCARD_NORM -> at.asFloatBuffer().put(weights.discardNorm());
        case DISCARD_HIDDEN -> at.asShortBuffer().put(weights.discardHidden());
        case DISCARD_HIDDEN_BIAS -> at.asFloatBuffer().put(weights.discardHiddenBias());
        case DISCARD_OUTPUT -> at.asFloatBuffer().put(weights.discardOutput());
        case DISCARD_EXACT -> at.asFloatBuffer().put(weights.discardExact());
        default -> throw new IllegalStateException("not a K9 slot: " + weight);
      }
    }
    return pack;
  }

  /** 重みの位置。ヘッダーの後に {@link #SLOTS} の順に並べる。 */
  private static Map<HipWeight, Long> offsets(HipArchitecture architecture) {
    HipPack.Header header = TestPack.header(architecture);
    Map<HipWeight, Long> offsets = new EnumMap<>(HipWeight.class);
    long at = HipPack.HEADER_BYTES;
    for (HipWeight weight : SLOTS) {
      offsets.put(weight, at);
      at = HipPack.align(at + header.bytes(weight, 0));
    }
    return offsets;
  }

  private static long packBytes(HipArchitecture architecture) {
    HipPack.Header header = TestPack.header(architecture);
    long at = HipPack.HEADER_BYTES;
    for (HipWeight weight : SLOTS) {
      at = HipPack.align(at + header.bytes(weight, 0));
    }
    return at;
  }

  /** blob のセクションを {@link HipSection} の順に、16 バイト境界にそろえて並べたバッチ（合法手は行ごとに 1 つ）。 */
  private static HipBatch layout(HipArchitecture architecture, int rows, int events, int transitions) {
    HipCounts counts = architecture.counts();
    HipPack.Header header = TestPack.header(architecture);
    long[] sections = new long[HipSection.values().length];
    long at = 0;
    for (HipSection section : HipSection.values()) {
      sections[section.ordinal()] = at;
      long count =
          switch (section.axis()) {
            case ROW, ACTION -> rows;
            case SEAT -> (long) counts.seats() * rows;
            case TILE -> (long) counts.tiles() * rows;
            case EVENT -> events;
            case TOKEN -> (long) counts.fixed() * rows + events;
            case TRANSITION -> transitions;
          };
      long bytes = count * header.sectionWidth(section) * section.element().bytes();
      at = Math.ceilDiv(at + bytes, HipBatch.SECTION_ALIGNMENT) * HipBatch.SECTION_ALIGNMENT;
    }
    return new HipBatch(rows, events, transitions, rows, at, sections);
  }

  private static int at(HipBatch batch, HipSection section, long offset) {
    return Math.toIntExact(batch.offset(section) + offset);
  }

  private static int tile(Random random, int tiles) {
    return 1 + random.nextInt(tiles);
  }

  private static int tileOrNone(Random random, int tiles) {
    return random.nextDouble() < NO_TILE ? 0 : tile(random, tiles);
  }

  private static short[] bf16(Random random, int count, double scale) {
    short[] values = new short[count];
    for (int i = 0; i < count; i++) {
      values[i] = MlpReference.bf16((float) (random.nextGaussian() * scale));
    }
    return values;
  }

  private static float[] fp32(Random random, int count, double mean, double scale) {
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      values[i] = (float) (mean + random.nextGaussian() * scale);
    }
    return values;
  }

  private static short[] concat(short[] first, short[] second) {
    short[] joined = Arrays.copyOf(first, first.length + second.length);
    System.arraycopy(second, 0, joined, first.length, second.length);
    return joined;
  }

  private static byte[] concat(byte[] first, byte[] second) {
    byte[] joined = Arrays.copyOf(first, first.length + second.length);
    System.arraycopy(second, 0, joined, first.length, second.length);
    return joined;
  }

  private static float[] concat(float[] first, float[] second) {
    float[] joined = Arrays.copyOf(first, first.length + second.length);
    System.arraycopy(second, 0, joined, first.length, second.length);
    return joined;
  }
}
