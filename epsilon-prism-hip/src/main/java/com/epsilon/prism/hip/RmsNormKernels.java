package com.epsilon.prism.hip;

import ai.djl.ndarray.NDArray;

/**
 * 学習の RMSNorm の順伝播と逆伝播の演算（学習で DJL の自動微分の外に置く演算）。GPU は {@link HipAttention}。
 *
 * <p>数式（行 i・列 j。統計・利得・勾配は FP32、⌊·⌉ は配列の型への最近接偶数の 1 回の丸め）:
 *
 * <ul>
 *   <li>順伝播: r_i = 1/√(Σ_j x_ij² / width + ε)、y_ij = ⌊(x_ij·r_i)·g_j⌉。
 *   <li>逆伝播: x̂_ij = x_ij·r_i、m_i = Σ_j (dy_ij·g_j)·x̂_ij / width、dx_ij = ⌊r_i·(dy_ij·g_j − x̂_ij·m_i)⌉。利得の勾配は
 *       {@value #GAIN_ROWS} 行の塊ごとの部分和 Σ_i dy_ij·x̂_ij で、呼び手が塊で足す（決まった順の 2 段の和）。
 * </ul>
 *
 * 配列の形: 入力・出力・それらの勾配は最後の軸が width で要素の数が rows·width（連続）、利得は [width]（FP32）、r は [rows]（FP32）、
 * 利得の勾配の部分和は [{@link NormShape#partials()}, width]（FP32）。
 */
public interface RmsNormKernels {

  /** 幅の上限（native の kNormMaxWidth）。 */
  int MAX_WIDTH = 1024;

  /** 利得の勾配の部分和 1 つが受け持つ行の数（native の kNormGainRows）。 */
  int GAIN_ROWS = 128;

  /**
   * RMSNorm の数。
   *
   * @param rows 行の数（正規化する軸のほかの軸の積）
   * @param width 正規化する最後の軸の長さ（8 の倍数で {@value #MAX_WIDTH} 以下）
   * @param epsilon ε（正）
   */
  record NormShape(int rows, int width, float epsilon) {

    /**
     * 数を確かめる。
     *
     * @throws IllegalArgumentException 範囲の外のとき
     */
    public NormShape {
      if (rows <= 0 || width <= 0 || width % 8 != 0 || width > MAX_WIDTH || !(epsilon > 0)) {
        throw new IllegalArgumentException("invalid RMSNorm dimensions " + this);
      }
    }

    /** 利得の勾配の部分和の数 ⌈rows / {@value #GAIN_ROWS}⌉。 */
    public int partials() {
      return (rows + GAIN_ROWS - 1) / GAIN_ROWS;
    }
  }

  /**
   * 順伝播。
   *
   * @param d 数
   * @param input 入力
   * @param gain 利得（FP32）
   * @param out 出力（入力と同じ型・形、書く）
   * @param inverseRms 逆伝播のための r（FP32、書く）
   */
  void forward(NormShape d, NDArray input, NDArray gain, NDArray out, NDArray inverseRms);

  /**
   * 逆伝播。
   *
   * @param d 数
   * @param input 順伝播の入力
   * @param gain 順伝播の利得
   * @param inverseRms 順伝播の r
   * @param gradOut 出力の勾配（出力と同じ型）
   * @param gradInput 入力の勾配（入力と同じ型、書く）
   * @param gradGain 利得の勾配の塊ごとの部分和（FP32、書く）
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
