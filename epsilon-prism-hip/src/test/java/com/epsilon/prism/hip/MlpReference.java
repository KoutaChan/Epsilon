package com.epsilon.prism.hip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;

/**
 * K3 の CPU の参照。K3 と同じ式・同じ丸めの所を、素朴な FP32 の積和（K の順に累積）で計算する。
 *
 * <pre>
 * x1 = ⌊x + ⌊o·Woᵀ + bo⌉⌉,  n = ⌊(x1·r)·g⌉,  r = 1/√(mean(x1²) + ε)
 * h = ⌊silu⌊n·W1ᵀ + b1⌉⌉,   x = ⌊x1 + ⌊h·W2ᵀ + b2⌉⌉
 * </pre>
 */
final class MlpReference {

  private MlpReference() {}

  /** float を BF16 に最近接偶数で丸めたビット（c10::BFloat16 と同じ。NaN は 0x7FC0）。 */
  static short bf16(float value) {
    int bits = Float.floatToRawIntBits(value);
    if ((bits & 0x7fffffff) > 0x7f800000) {
      return (short) 0x7fc0;
    }
    return (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16);
  }

  static float value(short bf16) {
    return Float.intBitsToFloat((bf16 & 0xffff) << 16);
  }

  /** ⌊v⌉ を float で。 */
  static float round(float value) {
    return value(bf16(value));
  }

  /** 全てのトークンの出力 x（BF16 のビット、{@code [tokens, H]}）。 */
  static short[] run(MlpCase c) {
    int h = c.shape.hidden();
    int a = c.shape.attention();
    int f = c.shape.ffn();
    float epsilon = c.shape.epsilon();
    float[] wo = values(c.outWeight);
    float[] w1 = values(c.upWeight);
    float[] w2 = values(c.downWeight);
    float[] x1 = new float[h];
    float[] n = new float[h];
    float[] hidden = new float[f];
    short[] out = new short[c.tokens * h];
    for (int t = 0; t < c.tokens; t++) {
      for (int i = 0; i < h; i++) {
        float acc = 0f;
        for (int k = 0; k < a; k++) {
          acc += value(c.o[t * a + k]) * wo[i * a + k];
        }
        x1[i] = round(value(c.x[t * h + i]) + round(acc + c.outBias[i]));
      }
      float sum = 0f;
      for (int i = 0; i < h; i++) {
        sum += x1[i] * x1[i];
      }
      float r = (float) (1.0 / Math.sqrt(sum / h + epsilon));
      for (int i = 0; i < h; i++) {
        n[i] = round((x1[i] * r) * c.normGain[i]);
      }
      for (int j = 0; j < f; j++) {
        float acc = 0f;
        for (int k = 0; k < h; k++) {
          acc += n[k] * w1[j * h + k];
        }
        float u = round(acc + c.upBias[j]);
        hidden[j] = round(u / (1f + (float) Math.exp(-u)));
      }
      for (int i = 0; i < h; i++) {
        float acc = 0f;
        for (int j = 0; j < f; j++) {
          acc += hidden[j] * w2[i * f + j];
        }
        out[t * h + i] = bf16(x1[i] + round(acc + c.downBias[i]));
      }
    }
    return out;
  }

  /** BF16 の列の比べ: 相対 RMS、1 ulp 以内の割合、最大の ulp。 */
  record Comparison(double relativeRms, double withinOneUlp, int maxUlp) {

    static Comparison of(short[] expected, short[] actual) {
      double error = 0;
      double norm = 0;
      long within = 0;
      int max = 0;
      for (int i = 0; i < expected.length; i++) {
        double e = value(expected[i]);
        double d = value(actual[i]) - e;
        error += d * d;
        norm += e * e;
        int ulp = Math.abs(ordered(expected[i]) - ordered(actual[i]));
        within += ulp <= 1 ? 1 : 0;
        max = Math.max(max, ulp);
      }
      return new Comparison(Math.sqrt(error / norm), (double) within / expected.length, max);
    }

    private static int ordered(short bf16) {
      int bits = bf16 & 0xffff;
      return (bits & 0x8000) != 0 ? -(bits & 0x7fff) : bits;
    }
  }

  /** 直の ByteBuffer（little-endian）から BF16 のビットを読む。 */
  static short[] shorts(ByteBuffer buffer, int count) {
    short[] values = new short[count];
    ShortBuffer view = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(0).asShortBuffer();
    view.get(values);
    return values;
  }

  private static float[] values(short[] bf16) {
    float[] values = new float[bf16.length];
    for (int i = 0; i < bf16.length; i++) {
      values[i] = value(bf16[i]);
    }
    return values;
  }
}
