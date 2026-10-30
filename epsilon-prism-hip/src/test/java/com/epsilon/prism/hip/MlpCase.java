package com.epsilon.prism.hip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.Map;
import java.util.Random;

/**
 * K3 の試験の入力: 乱数の重みと入力（BF16 の行列は丸めたビット）、それを層 0 の MLP の置き場に置いた WeightPack（頭つき）。
 */
final class MlpCase {

  /** 層 0 の MLP の置き場と、K3 が次の層の注意の入力を作る層 1 の注意の利得（{@link #NEXT_NORM}）。 */
  static final HipWeight[] SLOTS = {
    HipWeight.OUTPUT,
    HipWeight.OUTPUT_BIAS,
    HipWeight.FFN_NORM,
    HipWeight.UP,
    HipWeight.UP_BIAS,
    HipWeight.DOWN,
    HipWeight.DOWN_BIAS,
    HipWeight.ATTENTION_NORM
  };

  /** 次の層の注意の利得の置き場（層 1、値は 1）。 */
  private static final HipWeight NEXT_NORM = HipWeight.ATTENTION_NORM;

  final HipShape shape;
  final int tokens;
  final short[] x;
  final short[] o;
  final short[] outWeight;
  final float[] outBias;
  final float[] normGain;
  final short[] upWeight;
  final float[] upBias;
  final short[] downWeight;
  final float[] downBias;

  private MlpCase(
      HipShape shape,
      int tokens,
      short[] x,
      short[] o,
      short[] outWeight,
      float[] outBias,
      float[] normGain,
      short[] upWeight,
      float[] upBias,
      short[] downWeight,
      float[] downBias) {
    this.shape = shape;
    this.tokens = tokens;
    this.x = x;
    this.o = o;
    this.outWeight = outWeight;
    this.outBias = outBias;
    this.normGain = normGain;
    this.upWeight = upWeight;
    this.upBias = upBias;
    this.downWeight = downWeight;
    this.downBias = downBias;
  }

  /** 計画の形（H=256、深さ 2、容量 capacity トークン）。 */
  static HipShape shape(int ffn, int capacity) {
    int h = HipShape.HIDDEN;
    HipShape.Capacity sizes =
        new HipShape.Capacity(1, capacity - TestPack.FIXED, 1, 1, TestPack.inputBytes(capacity));
    HipShape probe = new HipShape(h, ffn, 256, 2, 1e-5f * 256 / h, TestPack.FIXED, sizes, HipPack.HEADER_BYTES);
    return new HipShape(h, ffn, 256, 2, probe.epsilon(), TestPack.FIXED, sizes, packBytes(probe));
  }

  /** 学習した網に近い大きさの乱数（行列は 1/√in、偏りは 0.02、利得は 1 ± 0.1）。 */
  static MlpCase random(long seed, int ffn, int tokens, int capacity) {
    Random random = new Random(seed);
    HipShape shape = shape(ffn, capacity);
    int h = shape.hidden();
    int a = shape.attention();
    return new MlpCase(
        shape,
        tokens,
        bf16(random, tokens * h, 1.0),
        bf16(random, tokens * a, 0.5),
        bf16(random, h * a, 1 / Math.sqrt(a)),
        fp32(random, h, 0.0, 0.02),
        fp32(random, h, 1.0, 0.1),
        bf16(random, ffn * h, 1 / Math.sqrt(h)),
        fp32(random, ffn, 0.0, 0.02),
        bf16(random, h * ffn, 1 / Math.sqrt(ffn)),
        fp32(random, h, 0.0, 0.02));
  }

  /** 残差の枝（O 射影と下り）を 0 にした写し。出力は入力の x とビットで同じになるはず。 */
  MlpCase withoutResidualBranch() {
    return new MlpCase(
        shape,
        tokens,
        x,
        o,
        new short[outWeight.length],
        new float[outBias.length],
        normGain,
        upWeight,
        upBias,
        new short[downWeight.length],
        new float[downBias.length]);
  }

  /** {@code other} のトークンを前に足した写し（重みは同じ）。同じトークンが塊の中の別の位置に来る。 */
  MlpCase prepend(MlpCase other) {
    return new MlpCase(
        shape,
        other.tokens + tokens,
        concat(other.x, x),
        concat(other.o, o),
        outWeight,
        outBias,
        normGain,
        upWeight,
        upBias,
        downWeight,
        downBias);
  }

  /** 1 行の塊（K3 のトークンの数）。 */
  HipBatch batch() {
    return TestPack.batch(tokens);
  }

  /** 置き場の位置（頭の後に {@link #SLOTS} の順）。 */
  private static Map<HipWeight, Long> offsets(HipShape shape) {
    HipDimensions dimensions = dimensions(shape);
    Map<HipWeight, Long> offsets = new EnumMap<>(HipWeight.class);
    long at = HipPack.HEADER_BYTES;
    for (HipWeight weight : SLOTS) {
      offsets.put(weight, at);
      at = HipPack.align(at + weight.bytes(dimensions));
    }
    return offsets;
  }

  private static long packBytes(HipShape shape) {
    HipDimensions dimensions = dimensions(shape);
    long at = HipPack.HEADER_BYTES;
    for (HipWeight weight : SLOTS) {
      at = HipPack.align(at + weight.bytes(dimensions));
    }
    return at;
  }

  private static HipDimensions dimensions(HipShape shape) {
    return new HipDimensions(shape.hidden(), shape.ffn(), shape.gate(), 109, 112, 34, 6, 8, 10);
  }

  /** WeightPack（直の ByteBuffer、little-endian、頭つき）。 */
  ByteBuffer pack() {
    Map<HipWeight, Long> offsets = offsets(shape);
    HipPack.Header header = TestPack.header();
    for (HipWeight weight : SLOTS) {
      header.weight(weight, weight == NEXT_NORM ? 1 : 0, offsets.get(weight));
    }
    ByteBuffer pack = TestPack.direct(shape.weightBytes());
    header.write(pack);
    for (HipWeight weight : SLOTS) {
      ByteBuffer at = pack.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(Math.toIntExact(offsets.get(weight)));
      switch (weight) {
        case OUTPUT -> at.asShortBuffer().put(outWeight);
        case OUTPUT_BIAS -> at.asFloatBuffer().put(outBias);
        case FFN_NORM -> at.asFloatBuffer().put(normGain);
        case UP -> at.asShortBuffer().put(upWeight);
        case UP_BIAS -> at.asFloatBuffer().put(upBias);
        case DOWN -> at.asShortBuffer().put(downWeight);
        case DOWN_BIAS -> at.asFloatBuffer().put(downBias);
        case ATTENTION_NORM -> {
          float[] ones = new float[shape.hidden()];
          Arrays.fill(ones, 1f);
          at.asFloatBuffer().put(ones);
        }
        default -> throw new IllegalStateException("not an MLP slot: " + weight);
      }
    }
    return pack;
  }

  /** 作業域の区画に写す入力（直の ByteBuffer）。 */
  ByteBuffer input(HipRegion region) {
    short[] values =
        switch (region) {
          case RESIDUAL -> x;
          case ATTENTION -> o;
          default -> throw new IllegalArgumentException("K3 reads only the residual and attention regions");
        };
    ByteBuffer buffer = TestPack.direct((long) values.length * Short.BYTES);
    buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(values);
    return buffer;
  }

  /** 区画のバイト数（このトークンの数）。 */
  long bytes(HipRegion region) {
    return region.bytes(shape, batch());
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
    short[] joined = new short[first.length + second.length];
    System.arraycopy(first, 0, joined, 0, first.length);
    System.arraycopy(second, 0, joined, first.length, second.length);
    return joined;
  }
}
