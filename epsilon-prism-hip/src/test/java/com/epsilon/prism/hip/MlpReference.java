package com.epsilon.prism.hip;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * K3・K5 の CPU の参照。カーネルと同じ式を同じ所で丸め、積和は素朴な FP32（K の順に足していく）で計算する。各トークンは自分の組の
 * FFN を通る（O 射影は層の中で共通）。
 *
 * <pre>
 * x1 = ⌊x + ⌊o·Woᵀ + bo⌉⌉,  n = ⌊(x1·r)·g⌉,  r = 1/√(mean(x1²) + ε)
 * h = ⌊silu⌊n·W1ᵀ + b1⌉⌉,   x = ⌊x1 + ⌊h·W2ᵀ + b2⌉⌉
 * K3: 次の層の Attention の入力 ⌊(x·r)·g_next⌉。最後の全トークンの層では、イベントのトークンで組ごとに [K V] = ⌊⌊(x·r)·g⌉·W_kvᵀ + b_kv⌉
 * K5: y = ⌊(x·r)·g_final⌉（ゲインは組ごと）、[K V] = ⌊y·W_kvᵀ + b_kv⌉ を [行, 位置, 2A] に書く
 * </pre>
 */
final class MlpReference {

  private MlpReference() {}

  /**
   * 参照の出力。
   *
   * @param x 残差ストリーム [トークン, H]（K3 が書き戻さないトークンは入力のまま。K5 では全てのトークンが入力のまま）
   * @param normalized K3 が書く、次の層の Attention の入力 [n を書くトークン, H]（K5 では空）
   * @param state K5 が最後に正規化した y [F·R, H]（K3 では空）
   * @param memory K5 の Cross-Attention のキーとバリュー [R, F, 2A]（K3 では空）
   * @param events 最後の全トークンの層の K3 が作る、イベントのキーとバリュー [組][イベント, 2A]（ほかのカーネルでは組が無い）
   */
  record Output(short[] x, short[] normalized, short[] state, short[] memory, short[][] events) {}

  /** FFN の 1 組の重みの値（BF16 の行列を float にしたもの）。 */
  private record Dense(float[] norm, float[] up, float[] upBias, float[] down, float[] downBias, int width) {

    static Dense of(MlpCase.Ffn ffn) {
      return new Dense(ffn.norm(), values(ffn.up()), ffn.upBias(), values(ffn.down()), ffn.downBias(), ffn.width());
    }
  }

  /** float を最近接偶数で BF16 に丸めたビットを返す（c10::BFloat16 と同じ。NaN は 0x7FC0）。 */
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

  /** ⌊v⌉（BF16 に丸めた値）を float で返す。 */
  static float round(float value) {
    return value(bf16(value));
  }

  /**
   * カーネルの出力を計算する。
   *
   * @param c 入力
   * @return 出力
   */
  static Output run(MlpCase c) {
    int h = c.architecture.hidden();
    int a = c.architecture.attention();
    int fixed = c.architecture.counts().fixed();
    float epsilon = c.architecture.epsilon();
    float[] wo = values(c.outWeight);
    Map<TokenGroup, Dense> dense = new EnumMap<>(TokenGroup.class);
    c.ffns.forEach((group, ffn) -> dense.put(group, Dense.of(ffn)));
    boolean last = c.kernel == HipKernel.MLP_LAST;
    float[] kv = last ? values(c.memory.weight()) : null;
    List<MlpCase.EventKeyValue> sets = c.eventKeyValues();
    List<float[]> setWeights = sets.stream().map(set -> values(set.weight())).toList();
    short[] x = c.x.clone();
    short[] normalized = new short[c.normalizedTokens() * h];
    short[] state = new short[last ? c.fixedTokens() * h : 0];
    short[] memory = new short[last ? c.fixedTokens() * 2 * a : 0];
    short[][] events = new short[sets.size()][c.events * 2 * a];
    float[] stream = new float[h];
    float[] y = new float[h];
    short[] n = new short[h];
    for (int t = 0; t < c.processedTokens(); t++) {
      stage(c, t, wo, dense.get(c.group(t)), epsilon, stream);
      if (!last) {
        if (t < c.writtenTokens()) {
          for (int i = 0; i < h; i++) {
            x[t * h + i] = bf16(stream[i]);
          }
        }
        if (t < c.normalizedTokens()) {
          normalize(stream, c.nextNorm, epsilon, normalized, t * h);
        }
        int event = t - c.fixedTokens();
        for (int j = 0; j < sets.size() && event >= 0; j++) {
          normalize(stream, sets.get(j).gain(), epsilon, n, 0);
          for (int col = 0; col < 2 * a; col++) {
            float acc = 0f;
            for (int k = 0; k < h; k++) {
              acc += value(n[k]) * setWeights.get(j)[col * h + k];
            }
            events[j][event * 2 * a + col] = bf16(acc + sets.get(j).bias()[col]);
          }
        }
        continue;
      }
      normalize(stream, c.memory.gains().get(c.group(t)), epsilon, state, t * h);
      for (int i = 0; i < h; i++) {
        y[i] = value(state[t * h + i]);
      }
      int at = (c.row(t) * fixed + c.position(t)) * 2 * a;
      for (int j = 0; j < 2 * a; j++) {
        float acc = 0f;
        for (int k = 0; k < h; k++) {
          acc += y[k] * kv[j * h + k];
        }
        memory[at + j] = bf16(acc + c.memory.bias()[j]);
      }
    }
    return new Output(x, normalized, state, memory, events);
  }

  /** 1 つのトークンで mlpStage を計算し、新しい x（BF16 の値）を {@code out} に書く。 */
  private static void stage(MlpCase c, int t, float[] wo, Dense ffn, float epsilon, float[] out) {
    int h = c.architecture.hidden();
    int a = c.architecture.attention();
    float[] x1 = new float[h];
    for (int i = 0; i < h; i++) {
      float acc = 0f;
      for (int k = 0; k < a; k++) {
        acc += value(c.o[t * a + k]) * wo[i * a + k];
      }
      x1[i] = round(value(c.x[t * h + i]) + round(acc + c.outBias[i]));
    }
    float r = inverseRms(x1, epsilon);
    float[] n = new float[h];
    for (int i = 0; i < h; i++) {
      n[i] = round((x1[i] * r) * ffn.norm()[i]);
    }
    float[] hidden = new float[ffn.width()];
    for (int j = 0; j < ffn.width(); j++) {
      float acc = 0f;
      for (int k = 0; k < h; k++) {
        acc += n[k] * ffn.up()[j * h + k];
      }
      float u = round(acc + ffn.upBias()[j]);
      hidden[j] = round(u / (1f + (float) Math.exp(-u)));
    }
    for (int i = 0; i < h; i++) {
      float acc = 0f;
      for (int j = 0; j < ffn.width(); j++) {
        acc += hidden[j] * ffn.down()[i * ffn.width() + j];
      }
      out[i] = round(x1[i] + round(acc + ffn.downBias()[i]));
    }
  }

  /** ⌊(v·r)·g⌉ を計算し、{@code out} の {@code offset} から書く（v は BF16 の値）。 */
  private static void normalize(float[] v, float[] gain, float epsilon, short[] out, int offset) {
    float r = inverseRms(v, epsilon);
    for (int i = 0; i < v.length; i++) {
      out[offset + i] = bf16((v[i] * r) * gain[i]);
    }
  }

  /** 1/√(mean(v²) + ε)。 */
  private static float inverseRms(float[] v, float epsilon) {
    float sum = 0f;
    for (float value : v) {
      sum += value * value;
    }
    return (float) (1.0 / Math.sqrt(sum / v.length + epsilon));
  }

  /** BF16 の列を比べた結果。相対 RMS、1 ulp 以内の割合、ulp の最大。 */
  record Comparison(double relativeRms, double withinOneUlp, int maxUlp) {

    static Comparison of(short[] expected, short[] actual) {
      if (expected.length != actual.length) {
        throw new IllegalArgumentException("compared " + expected.length + " with " + actual.length + " values");
      }
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

  /** ダイレクトの ByteBuffer（little-endian）の先頭から、BF16 のビットを読む。 */
  static short[] shorts(ByteBuffer buffer, int count) {
    short[] values = new short[count];
    ShortBuffer view = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(0).asShortBuffer();
    view.get(values);
    return values;
  }

  /** 先頭の {@code count} 個の要素の写しを返す。 */
  static short[] head(short[] values, int count) {
    return Arrays.copyOf(values, count);
  }

  private static float[] values(short[] bf16) {
    float[] values = new float[bf16.length];
    for (int i = 0; i < bf16.length; i++) {
      values[i] = value(bf16[i]);
    }
    return values;
  }
}
