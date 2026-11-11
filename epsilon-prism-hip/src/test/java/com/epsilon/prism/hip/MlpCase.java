package com.epsilon.prism.hip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Consumer;

/**
 * K3・K5 のテストの入力。形の記述、回すカーネルと層、バッチの行とイベントの数、乱数で作った入力と重み（BF16 の行列は丸めたビット）を
 * 持ち、重みをヘッダーつきの WeightPack に並べる。
 *
 * <p>WeightPack には、層の O 射影と FFN の組ごとの重みを入れる。K3 では、次の層の Attention のゲインも入れる。最後の全トークンの層の K3
 * では、イベントのキーとバリューを作る重みも入れる。2 つ目以降の局・席・牌の層ごとの Attention のゲイン・QKV・バイアスと、2 段目が
 * あれば 2 段目のイベントのゲインとキー・バリューの射影がそれにあたる。K5 では、組ごとの最後のゲインと、候補層の Cross-Attention の
 * キー・バリューの射影を入れる。
 *
 * <p>トークンの軸は局 R・席 S·R・牌 T·R・イベントの順に並び、先頭の F·R 個が局・席・牌のトークン。型ごとの FFN の層では、局・席
 * [0, (1+S)·R) と牌 [(1+S)·R, F·R) を別の重みで通す。
 */
final class MlpCase {

  /** 三人打ちの数（席 3・牌 27・キーのコード 107・牌の関係の種類 23・イベントの上限 91）。 */
  static final HipCounts THREE_PLAYERS = new HipCounts(3, 27, 107, 23, 91);

  /** トークンの軸を分ける範囲（局・席・牌・イベント）の数。 */
  private static final int REGIONS = 4;

  /**
   * FFN の 1 組の重み。
   *
   * @param norm FFN の前の RMSNorm のゲイン [H]
   * @param up 上りの射影 [F, H]（BF16 のビット）
   * @param upBias 上りのバイアス [F]
   * @param down 下りの射影 [H, F]（BF16 のビット）
   * @param downBias 下りのバイアス [H]
   */
  record Ffn(float[] norm, short[] up, float[] upBias, short[] down, float[] downBias) {

    /** FFN の幅 F。 */
    int width() {
      return upBias.length;
    }

    /** 下りの射影とバイアスを 0 にした写し。 */
    Ffn withoutDown() {
      return new Ffn(norm, up, upBias, new short[down.length], new float[downBias.length]);
    }
  }

  /**
   * K5 の後半の重み。
   *
   * @param gains 組ごとの最後の正規化のゲイン [H]
   * @param weight 候補層の Cross-Attention のキーとバリューの射影 [2A, H]（BF16 のビット）
   * @param bias 射影のバイアス [2A]
   */
  record Memory(Map<TokenGroup, float[]> gains, short[] weight, float[] bias) {}

  /**
   * 最後の全トークンの層の K3 がイベントのキーとバリューを作る、1 組の重み。[K V] = ⌊⌊(x·r)·g⌉·W_kvᵀ + b_kv⌉。
   *
   * @param gain ゲイン [H]
   * @param weight キーとバリューの射影 [2A, H]（BF16 のビット）
   * @param bias 射影のバイアス [2A]
   */
  record EventKeyValue(float[] gain, short[] weight, float[] bias) {}

  /**
   * 局・席・牌の層の Attention の重み。最後の全トークンの層の K3 は、このうち k・v の行でイベントのキーとバリューを作る。
   *
   * @param gain Attention の前のゲイン [H]
   * @param qkv QKV [3A, H]（BF16 のビット）
   * @param bias QKV のバイアス [3A]
   */
  record Attention(float[] gain, short[] qkv, float[] bias) {

    /** k・v の行 [A, 3A) を取り出した、イベントのキーとバリューの重み。 */
    EventKeyValue keyValue(int attention, int hidden) {
      return new EventKeyValue(
          gain,
          Arrays.copyOfRange(qkv, attention * hidden, 3 * attention * hidden),
          Arrays.copyOfRange(bias, attention, 3 * attention));
    }
  }

  /**
   * WeightPack の重みの項目と、そこに書く値。
   *
   * @param weight 重みの種類
   * @param entry 項目の番号
   * @param length 値の数
   * @param writer 項目の先頭に値を書く
   */
  private record Placement(HipWeight weight, int entry, int length, Consumer<ByteBuffer> writer) {

    static Placement bf16(HipWeight weight, int entry, short[] values) {
      requireElement(weight, HipElement.BF16);
      return new Placement(weight, entry, values.length, at -> at.asShortBuffer().put(values));
    }

    static Placement fp32(HipWeight weight, int entry, float[] values) {
      requireElement(weight, HipElement.FP32);
      return new Placement(weight, entry, values.length, at -> at.asFloatBuffer().put(values));
    }

    private static void requireElement(HipWeight weight, HipElement element) {
      if (weight.element() != element) {
        throw new IllegalArgumentException(weight + " holds " + weight.element() + ", not " + element);
      }
    }
  }

  final HipArchitecture architecture;
  final HipKernel kernel;
  final int layer;
  final int rows;
  final int events;
  final short[] x;
  final short[] o;
  final short[] outWeight;
  final float[] outBias;
  final Map<TokenGroup, Ffn> ffns;

  /** 次の層の Attention のゲイン [H]。K3 だけが持ち、K5 では null。 */
  final float[] nextNorm;

  /** K5 の後半の重み。K3 では null。 */
  final Memory memory;

  /** 2 つ目以降の局・席・牌の層ごとの Attention の重み。最後の全トークンの層の K3 だけが持ち、ほかでは空。 */
  final List<Attention> fixedAttention;

  /** 2 段目のイベントのキーとバリューの重み。2 段目のある形の、最後の全トークンの層の K3 だけが持ち、ほかでは null。 */
  final EventKeyValue reread;

  private final long packBytes;

  private MlpCase(
      HipArchitecture architecture,
      HipKernel kernel,
      int layer,
      int rows,
      int events,
      short[] x,
      short[] o,
      short[] outWeight,
      float[] outBias,
      Map<TokenGroup, Ffn> ffns,
      float[] nextNorm,
      Memory memory,
      List<Attention> fixedAttention,
      EventKeyValue reread) {
    this.architecture = architecture;
    this.kernel = kernel;
    this.layer = layer;
    this.rows = rows;
    this.events = events;
    this.x = x;
    this.o = o;
    this.outWeight = outWeight;
    this.outBias = outBias;
    this.ffns = ffns;
    this.nextNorm = nextNorm;
    this.memory = memory;
    this.fixedAttention = fixedAttention;
    this.reread = reread;
    this.packBytes = layout(TestPack.header(architecture), new long[placements().size()]);
  }

  /**
   * V3 のバックボーンの形を返す。全トークンの層 1 つの後に、型ごとの FFN を持つ局・席・牌の層が 2 つ続き、牌の FFN の幅は 768。候補層は
   * 幅 1024 で 2 段目があり、G は 512。ゲートの種類 6 と手の扱いの種類 8 は、K3・K5 では読まない。
   *
   * @param counts 四人打ちか三人打ちかで決まる数
   * @param context 局・席の FFN の幅
   * @return 形の記述
   */
  static HipArchitecture typed(HipCounts counts, int context) {
    HipLayer fixed = new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Typed(context, 768));
    return new HipArchitecture(
        HipArchitecture.HIDDEN,
        1e-5f,
        0.25f,
        0.01f,
        List.of(new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(512)), fixed, fixed),
        1024,
        CandidateStages.REREAD,
        512,
        6,
        8,
        counts);
  }

  /**
   * 型を分けない FFN で、深さ 3 のバックボーンの形を返す。全トークンの層 2 つの後に局・席・牌の層が 1 つ続き、数は四人打ちのもの。
   * 層 0 の K3 は、全てのトークンで次の層の Attention の入力を書く。
   *
   * @param ffn FFN と候補層の FFN の幅（全ての層で同じ）
   * @return 形の記述
   */
  static HipArchitecture sharedDepth3(int ffn) {
    HipLayer all = new HipLayer(LayerRole.ALL_TOKENS, new HipFfn.Shared(ffn));
    return new HipArchitecture(
        HipArchitecture.HIDDEN,
        1e-5f,
        0.25f,
        0.01f,
        List.of(all, all, new HipLayer(LayerRole.FIXED_TOKENS, new HipFfn.Shared(ffn))),
        ffn,
        CandidateStages.ONE,
        256,
        6,
        8,
        TestPack.FOUR_PLAYERS);
  }

  /**
   * 学習したネットワークに近い大きさの正規乱数で、重みと入力を作る。標準偏差は、行列が 1/√in、バイアスが 0.02、x が 1、o が 0.5。
   * ゲインは平均 1、標準偏差 0.1。
   *
   * @param seed 乱数の種
   * @param architecture 形の記述
   * @param kernel K3（{@link HipKernel#MLP_ALL}・{@link HipKernel#MLP_FIXED}）か K5（{@link HipKernel#MLP_LAST}）
   * @param layer カーネルが回す層
   * @param rows 行の数
   * @param events イベントのトークンの数
   * @return 入力
   * @throws IllegalArgumentException MLP のカーネルでないか、カーネルがその層を回さないとき
   */
  static MlpCase random(long seed, HipArchitecture architecture, HipKernel kernel, int layer, int rows, int events) {
    if (kernel != HipKernel.MLP_ALL && kernel != HipKernel.MLP_FIXED && kernel != HipKernel.MLP_LAST) {
      throw new IllegalArgumentException("not an MLP kernel: " + kernel);
    }
    if (!kernel.acceptsLayer(layer, architecture)) {
      throw new IllegalArgumentException(kernel + " does not run layer " + layer + " of " + architecture);
    }
    Random random = new Random(seed);
    int h = architecture.hidden();
    int a = architecture.attention();
    int tokens = architecture.counts().fixed() * rows + events;
    short[] x = bf16(random, tokens * h, 1.0);
    short[] o = bf16(random, tokens * a, 0.5);
    short[] outWeight = bf16(random, h * a, 1 / Math.sqrt(a));
    float[] outBias = fp32(random, h, 0.0, 0.02);
    Map<TokenGroup, Ffn> ffns = new EnumMap<>(TokenGroup.class);
    HipFfn ffn = architecture.layer(layer).ffn();
    for (TokenGroup group : ffn.groups()) {
      int f = ffn.width(group);
      ffns.put(
          group,
          new Ffn(
              fp32(random, h, 1.0, 0.1),
              bf16(random, f * h, 1 / Math.sqrt(h)),
              fp32(random, f, 0.0, 0.02),
              bf16(random, h * f, 1 / Math.sqrt(f)),
              fp32(random, h, 0.0, 0.02)));
    }
    if (kernel != HipKernel.MLP_LAST) {
      float[] nextNorm = fp32(random, h, 1.0, 0.1);
      List<Attention> fixedAttention = new ArrayList<>();
      EventKeyValue reread = null;
      if (kernel == HipKernel.MLP_ALL && architecture.layer(layer + 1).role() == LayerRole.FIXED_TOKENS) {
        for (int l = layer + 2; l < architecture.depth(); l++) {
          fixedAttention.add(
              new Attention(
                  fp32(random, h, 1.0, 0.1), bf16(random, 3 * a * h, 1 / Math.sqrt(h)), fp32(random, 3 * a, 0.0, 0.02)));
        }
        if (architecture.reread()) {
          reread =
              new EventKeyValue(
                  fp32(random, h, 1.0, 0.1), bf16(random, 2 * a * h, 1 / Math.sqrt(h)), fp32(random, 2 * a, 0.0, 0.02));
        }
      }
      return new MlpCase(
          architecture, kernel, layer, rows, events, x, o, outWeight, outBias, ffns, nextNorm, null, fixedAttention,
          reread);
    }
    Map<TokenGroup, float[]> gains = new EnumMap<>(TokenGroup.class);
    for (TokenGroup group : architecture.finalGroups()) {
      gains.put(group, fp32(random, h, 1.0, 0.1));
    }
    Memory memory = new Memory(gains, bf16(random, 2 * a * h, 1 / Math.sqrt(h)), fp32(random, 2 * a, 0.0, 0.02));
    return new MlpCase(
        architecture, kernel, layer, rows, events, x, o, outWeight, outBias, ffns, null, memory, List.of(), null);
  }

  /** 残差の枝（O 射影と、全ての組の下りの射影）を 0 にした写し。K3 が出す x は、入力の x とビットで一致するはず。 */
  MlpCase withoutResidualBranch() {
    Map<TokenGroup, Ffn> zero = new EnumMap<>(TokenGroup.class);
    ffns.forEach((group, ffn) -> zero.put(group, ffn.withoutDown()));
    return new MlpCase(
        architecture,
        kernel,
        layer,
        rows,
        events,
        x,
        o,
        new short[outWeight.length],
        new float[outBias.length],
        zero,
        nextNorm,
        memory,
        fixedAttention,
        reread);
  }

  /**
   * {@code other} の行とイベントを前に足した写しを返す。重みはこの入力のものを使う。範囲ごとに {@code other} のトークンが先に来るので、
   * この入力のトークンは範囲ごとに違う数だけずれ、バッチの中の位置と区間の境が変わる。
   *
   * @param other 前に足す入力（同じ形の記述）
   * @return 足した入力
   */
  MlpCase prepend(MlpCase other) {
    return new MlpCase(
        architecture,
        kernel,
        layer,
        other.rows + rows,
        other.events + events,
        merge(other.x, x, architecture.hidden(), other),
        merge(other.o, o, architecture.attention(), other),
        outWeight,
        outBias,
        ffns,
        nextNorm,
        memory,
        fixedAttention,
        reread);
  }

  /**
   * {@link #prepend} で作ったバッチの値から、この入力のトークンの値を元の順に取り出す。
   *
   * @param joined 足したバッチのトークンの値 [トークン, width]
   * @param width トークンごとの値の数
   * @param count この入力の先頭から取り出すトークンの数
   * @param other 前に足した入力
   * @return [count, width]
   */
  short[] unprepend(short[] joined, int width, int count, MlpCase other) {
    short[] values = new short[count * width];
    for (int t = 0; t < count; t++) {
      System.arraycopy(joined, shifted(t, other) * width, values, t * width, width);
    }
    return values;
  }

  /**
   * {@link #prepend} で作ったバッチの Cross-Attention のキーとバリュー [行, F, 2A] から、この入力の行の分を取り出す。この入力の行は
   * {@code other} の行の後に並ぶ。
   *
   * @param joined 足したバッチのキーとバリュー
   * @param other 前に足した入力
   * @return [R, F, 2A]
   */
  short[] unprependRows(short[] joined, MlpCase other) {
    int width = architecture.counts().fixed() * 2 * architecture.attention();
    short[] values = new short[rows * width];
    System.arraycopy(joined, other.rows * width, values, 0, values.length);
    return values;
  }

  /** トークンの数（局・席・牌とイベント）。 */
  int tokens() {
    return fixedTokens() + events;
  }

  /** 局・席・牌のトークンの数 F·R。 */
  int fixedTokens() {
    return architecture.counts().fixed() * rows;
  }

  /** 局・席のトークンの数 (1 + S)·R。型ごとの FFN では、ここが局・席の区間の終わりになる。 */
  int contextTokens() {
    return (1 + architecture.counts().seats()) * rows;
  }

  /** カーネルが mlpStage を回すトークンの数。全トークンの層では全てのトークン、ほかの層では局・席・牌のトークン。 */
  int processedTokens() {
    return kernel == HipKernel.MLP_ALL ? tokens() : fixedTokens();
  }

  /** K3 が x を書き戻すトークンの数。次が全トークンの層なら全てのトークン、そうでなければ局・席・牌のトークン。 */
  int writtenTokens() {
    return kernel == HipKernel.MLP_ALL && architecture.layer(layer + 1).role() == LayerRole.ALL_TOKENS
        ? tokens()
        : fixedTokens();
  }

  /** 最後の全トークンの層の K3 がイベントのキーとバリューを作る重みの組。局・席・牌の層の順に並び、2 段目の分が最後。ほかのカーネルでは空。 */
  List<EventKeyValue> eventKeyValues() {
    List<EventKeyValue> sets = new ArrayList<>();
    for (Attention attention : fixedAttention) {
      sets.add(attention.keyValue(architecture.attention(), architecture.hidden()));
    }
    if (reread != null) {
      sets.add(reread);
    }
    return sets;
  }

  /** K3 が次の層の Attention の入力 n を書くトークンの数。全トークンの層では全てのトークン、局・席・牌の層では局・席・牌のトークン。K5 は 0。 */
  int normalizedTokens() {
    return switch (kernel) {
      case MLP_ALL -> tokens();
      case MLP_FIXED -> fixedTokens();
      default -> 0;
    };
  }

  /**
   * トークンが通る FFN の組を返す。
   *
   * @param token トークンの軸の番号（カーネルが回すトークン）
   * @return 型を分けない層は {@link TokenGroup#ALL}、型ごとの層は局・席か牌
   */
  TokenGroup group(int token) {
    if (architecture.layer(layer).ffn() instanceof HipFfn.Shared) {
      return TokenGroup.ALL;
    }
    return token < contextTokens() ? TokenGroup.CONTEXT : TokenGroup.TILE;
  }

  /**
   * 局・席・牌のトークンが、行の中で何番目かを返す（局 0、席 1..S、牌 S+1..F−1）。
   *
   * @param token トークンの軸の番号（F·R 未満）
   * @return 行の中の位置
   */
  int position(int token) {
    int seats = architecture.counts().seats();
    if (token < rows) {
      return 0;
    }
    if (token < contextTokens()) {
      return 1 + (token - rows) % seats;
    }
    return 1 + seats + (token - contextTokens()) % architecture.counts().tiles();
  }

  /**
   * 局・席・牌のトークンが属する行を返す。
   *
   * @param token トークンの軸の番号（F·R 未満）
   * @return 行の番号
   */
  int row(int token) {
    if (token < rows) {
      return token;
    }
    if (token < contextTokens()) {
      return (token - rows) / architecture.counts().seats();
    }
    return (token - contextTokens()) / architecture.counts().tiles();
  }

  /** この入力がちょうど入る計画の形。 */
  HipShape shape() {
    return shape(rows, events);
  }

  /**
   * この入力の形の記述と WeightPack の大きさで、行とイベントの容量を指定した計画の形を返す。
   *
   * @param capacityRows 行の容量
   * @param capacityEvents イベントの容量
   * @return 計画の形
   */
  HipShape shape(int capacityRows, int capacityEvents) {
    int capacityTokens = architecture.counts().fixed() * capacityRows + capacityEvents;
    HipShape.Capacity capacity =
        new HipShape.Capacity(
            capacityRows, capacityEvents, capacityRows, capacityRows, TestPack.inputBytes(capacityTokens));
    return new HipShape(architecture, capacity, packBytes);
  }

  /** この入力のバッチ。行ごとに遷移と手を 1 つずつ持つ。K3・K5 は入力の blob を読まないので、blob のセクションの位置は全て 0 にする。 */
  HipBatch batch() {
    return new HipBatch(rows, events, rows, rows, TestPack.inputBytes(tokens()), new long[HipSection.values().length]);
  }

  /** WeightPack を作る（ダイレクトの ByteBuffer、little-endian、ヘッダーつき）。 */
  ByteBuffer pack() {
    HipPack.Header header = TestPack.header(architecture);
    List<Placement> placements = placements();
    long[] offsets = new long[placements.size()];
    ByteBuffer pack = TestPack.direct(layout(header, offsets));
    header.write(pack);
    for (int i = 0; i < placements.size(); i++) {
      Placement placement = placements.get(i);
      if (header.elements(placement.weight(), placement.entry()) != placement.length()) {
        throw new IllegalStateException(placement.weight() + " entry " + placement.entry() + " has the wrong size");
      }
      placement.writer().accept(pack.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(Math.toIntExact(offsets[i])));
    }
    return pack;
  }

  /** 作業領域のうち、指定した領域に写す入力（ダイレクトの ByteBuffer）。 */
  ByteBuffer input(HipRegion region) {
    short[] values =
        switch (region) {
          case RESIDUAL -> x;
          case ATTENTION -> o;
          default -> throw new IllegalArgumentException("K3 and K5 read only the residual and attention regions");
        };
    ByteBuffer buffer = TestPack.direct((long) values.length * Short.BYTES);
    buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(values);
    return buffer;
  }

  /** 作業領域のうち、指定した領域のバイト数（このバッチの大きさで数える）。 */
  long bytes(HipRegion region) {
    return region.bytes(shape(), batch());
  }

  /**
   * WeightPack に置く重みの一覧。層の O 射影と FFN の組ごとの重みは常に置く。K3 は次の層の Attention のゲインを置く。最後の全トークンの
   * 層の K3 は、局・席・牌の層の Attention の重みと、2 段目のイベントのキーとバリューの重みも置く。K5 は組ごとの最後のゲインと、
   * キーとバリューの射影を置く。
   */
  private List<Placement> placements() {
    List<Placement> placements = new ArrayList<>();
    placements.add(Placement.bf16(HipWeight.OUTPUT, layer, outWeight));
    placements.add(Placement.fp32(HipWeight.OUTPUT_BIAS, layer, outBias));
    ffns.forEach(
        (group, ffn) -> {
          int entry = HipWeight.ffnEntry(layer, group);
          placements.add(Placement.fp32(HipWeight.FFN_NORM, entry, ffn.norm()));
          placements.add(Placement.bf16(HipWeight.UP, entry, ffn.up()));
          placements.add(Placement.fp32(HipWeight.UP_BIAS, entry, ffn.upBias()));
          placements.add(Placement.bf16(HipWeight.DOWN, entry, ffn.down()));
          placements.add(Placement.fp32(HipWeight.DOWN_BIAS, entry, ffn.downBias()));
        });
    if (nextNorm != null) {
      placements.add(Placement.fp32(HipWeight.ATTENTION_NORM, layer + 1, nextNorm));
    }
    for (int f = 0; f < fixedAttention.size(); f++) {
      Attention attention = fixedAttention.get(f);
      int at = layer + 2 + f;
      placements.add(Placement.fp32(HipWeight.ATTENTION_NORM, at, attention.gain()));
      placements.add(Placement.bf16(HipWeight.QKV, at, attention.qkv()));
      placements.add(Placement.fp32(HipWeight.QKV_BIAS, at, attention.bias()));
    }
    if (reread != null) {
      placements.add(Placement.fp32(HipWeight.REREAD_EVENT_NORM, 0, reread.gain()));
      placements.add(Placement.bf16(HipWeight.REREAD_KEY_VALUE, 0, reread.weight()));
      placements.add(Placement.fp32(HipWeight.REREAD_KEY_VALUE_BIAS, 0, reread.bias()));
    }
    if (memory != null) {
      memory.gains().forEach((group, gain) -> placements.add(Placement.fp32(HipWeight.FINAL_NORM, group.ordinal(), gain)));
      placements.add(Placement.bf16(HipWeight.KEY_VALUE, 0, memory.weight()));
      placements.add(Placement.fp32(HipWeight.KEY_VALUE_BIAS, 0, memory.bias()));
    }
    return placements;
  }

  /** 重みをヘッダーの後に順に並べ、それぞれの位置をヘッダーと {@code offsets} に書く。WeightPack のバイト数を返す。 */
  private long layout(HipPack.Header header, long[] offsets) {
    List<Placement> placements = placements();
    long at = HipPack.HEADER_BYTES;
    for (int i = 0; i < placements.size(); i++) {
      Placement placement = placements.get(i);
      header.weight(placement.weight(), placement.entry(), at);
      offsets[i] = at;
      at = HipPack.align(at + header.bytes(placement.weight(), placement.entry()));
    }
    return at;
  }

  /** この入力のトークンが、{@code other} を前に足したバッチで何番目に来るかを返す。 */
  private int shifted(int token, MlpCase other) {
    int[] mine = regions();
    int[] theirs = other.regions();
    int start = 0;
    int joined = 0;
    for (int region = 0; region < REGIONS; region++) {
      if (token < start + mine[region]) {
        return joined + theirs[region] + token - start;
      }
      start += mine[region];
      joined += mine[region] + theirs[region];
    }
    throw new IllegalArgumentException("token " + token + " is outside the " + tokens() + " tokens");
  }

  /** 範囲ごとに、{@code first}（{@code other} の値）のトークンを先に、{@code second}（この入力の値）のトークンを後に並べる。 */
  private short[] merge(short[] first, short[] second, int width, MlpCase other) {
    int[] theirs = other.regions();
    int[] mine = regions();
    short[] joined = new short[first.length + second.length];
    int at = 0;
    int from = 0;
    int to = 0;
    for (int region = 0; region < REGIONS; region++) {
      System.arraycopy(first, from * width, joined, at * width, theirs[region] * width);
      at += theirs[region];
      from += theirs[region];
      System.arraycopy(second, to * width, joined, at * width, mine[region] * width);
      at += mine[region];
      to += mine[region];
    }
    return joined;
  }

  /** トークンの軸の範囲ごとのトークンの数（局 R・席 S·R・牌 T·R・イベント）。 */
  private int[] regions() {
    HipCounts counts = architecture.counts();
    return new int[] {rows, counts.seats() * rows, counts.tiles() * rows, events};
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
}
