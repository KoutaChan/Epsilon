package com.epsilon.prism.hip;

import ai.djl.ndarray.NDArray;

/**
 * 学習で使う RMSNorm の順伝播と逆伝播。学習のとき DJL の自動微分の外で計算する。GPU の実装は {@link HipAttention}。
 *
 * <p>数式（行 i・列 j。統計・ゲイン・勾配は FP32 で計算する。⌊·⌉ は配列の型へ最近接偶数で 1 回丸めること）:
 *
 * <ul>
 *   <li>順伝播: r_i = 1/√(Σ_j x_ij² / width + ε)、y_ij = ⌊(x_ij·r_i)·g_j⌉。
 *   <li>逆伝播: x̂_ij = x_ij·r_i、m_i = Σ_j (dy_ij·g_j)·x̂_ij / width、dx_ij = ⌊r_i·(dy_ij·g_j − x̂_ij·m_i)⌉。ゲインの勾配は
 *       {@value #GAIN_ROWS} 行ずつの部分和 Σ_i dy_ij·x̂_ij で返し、呼び出し元がそれを足す（決まった順に 2 段で足す）。
 * </ul>
 *
 * 配列の形: 入力・出力とそれらの勾配は、最後の軸が width で要素の数が rows·width（連続）。ゲインは [width]（FP32）、r は [rows]（FP32）、
 * ゲインの勾配の部分和は [{@link NormShape#partials()}, width]（FP32）。
 */
public interface RmsNormKernels {

  /** 幅の上限（native の kNormMaxWidth）。 */
  int MAX_WIDTH = 1024;

  /** ゲインの勾配の部分和 1 つが受け持つ行の数（native の kNormGainRows）。 */
  int GAIN_ROWS = 128;

  /**
   * RMSNorm の次元。
   *
   * @param rows 行の数（正規化する軸以外の軸の長さの積）
   * @param width 正規化する最後の軸の長さ（8 の倍数で {@value #MAX_WIDTH} 以下）
   * @param epsilon ε（正）
   */
  record NormShape(int rows, int width, float epsilon) {

    /**
     * 次元を確かめる。
     *
     * @throws IllegalArgumentException 値が範囲の外のとき
     */
    public NormShape {
      if (rows <= 0 || width <= 0 || width % 8 != 0 || width > MAX_WIDTH || !(epsilon > 0)) {
        throw new IllegalArgumentException(
            "invalid RMSNorm dimensions: rows=" + rows + " width=" + width + " epsilon=" + epsilon);
      }
    }

    /** ゲインの勾配の部分和の数 ⌈rows / {@value #GAIN_ROWS}⌉。 */
    public int partials() {
      return (rows + GAIN_ROWS - 1) / GAIN_ROWS;
    }
  }

  /**
   * 順伝播。
   *
   * @param d 次元
   * @param input 入力
   * @param gain ゲイン（FP32）
   * @param out 出力の書き先（入力と同じ型・形）
   * @param inverseRms 逆伝播で使う r の書き先（FP32）
   */
  void forward(NormShape d, NDArray input, NDArray gain, NDArray out, NDArray inverseRms);

  /**
   * 逆伝播。
   *
   * @param d 次元
   * @param input 順伝播の入力
   * @param gain 順伝播のゲイン
   * @param inverseRms 順伝播の r
   * @param gradOut 出力の勾配（出力と同じ型）
   * @param gradInput 入力の勾配の書き先（入力と同じ型）
   * @param gradGain ゲインの勾配の、{@value #GAIN_ROWS} 行ずつの部分和の書き先（FP32）
   */
  void backward(
      NormShape d,
      NDArray input,
      NDArray gain,
      NDArray inverseRms,
      NDArray gradOut,
      NDArray gradInput,
      NDArray gradGain);
}
