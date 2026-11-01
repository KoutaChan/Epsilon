package com.epsilon.prism.backend;

import java.util.Locale;

/**
 * 2 つの値の列の差: 相対 RMS、1 ulp 以内の割合（BF16 の値なら BF16 の ulp、FP32 なら FP32 の ulp）、最大の絶対差。
 *
 * @param relativeRms √(Σ(a − e)² / Σe²)
 * @param withinOneUlp 1 ulp 以内の割合
 * @param maxAbs 最大の |a − e|
 * @param count 要素の数
 */
record Diff(double relativeRms, double withinOneUlp, double maxAbs, int count) {

  /** BF16 の値の列（float に入れた BF16）を比べる。 */
  static Diff bf16(float[] expected, float[] actual) {
    return of(expected, actual, true);
  }

  /** FP32 の値の列を比べる。 */
  static Diff fp32(float[] expected, float[] actual) {
    return of(expected, actual, false);
  }

  private static Diff of(float[] expected, float[] actual, boolean bf16) {
    if (expected.length != actual.length) {
      throw new IllegalArgumentException("lengths differ: " + expected.length + " vs " + actual.length);
    }
    double error = 0;
    double norm = 0;
    double max = 0;
    long within = 0;
    for (int i = 0; i < expected.length; i++) {
      double d = (double) actual[i] - expected[i];
      error += d * d;
      norm += (double) expected[i] * expected[i];
      max = Math.max(max, Math.abs(d));
      within += ulps(expected[i], actual[i], bf16) <= 1 ? 1 : 0;
    }
    return new Diff(
        norm == 0 ? Math.sqrt(error) : Math.sqrt(error / norm),
        expected.length == 0 ? 1 : (double) within / expected.length,
        max,
        expected.length);
  }

  private static long ulps(float a, float b, boolean bf16) {
    long x = ordered(a, bf16);
    long y = ordered(b, bf16);
    return Math.abs(x - y);
  }

  private static long ordered(float value, boolean bf16) {
    int bits = Float.floatToRawIntBits(value);
    if (bf16) {
      bits >>>= 16;
      return (bits & 0x8000) != 0 ? -(bits & 0x7fff) : bits;
    }
    return bits < 0 ? -(long) (bits & 0x7fffffff) : bits;
  }

  /** 1 行の要約。 */
  String line(String label) {
    return String.format(
        Locale.ROOT,
        "%s: relative RMS %.3e, within 1 ulp %.5f, max |d| %.3e (n=%d)",
        label,
        relativeRms,
        withinOneUlp,
        maxAbs,
        count);
  }
}
