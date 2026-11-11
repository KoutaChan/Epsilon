package com.epsilon.grp;

/**
 * 純 Java の全結合層。GrpNetwork・OutcomeNetwork・GruSinkhorn で使う。重みは [入力][出力] の行優先で並ぶ。出力 y_j を b_j から
 * 始め、入力を i の小さい順に読んで w_ij·x_i を足していく。Java は足し算の順を入れ替えず、FMA にもまとめない。そのため結果は入力だけで
 * 決まり、ビット単位で再現する。
 *
 * <p>値が 0 の入力は飛ばす。重みは有限なので w·0 は ±0 になり、飛ばして変わりうるのは 0 の符号だけ。隠れ層では、ReLU がその符号を
 * ＋0 にそろえる。ロジットの層では、softmax と σ の値は変わらない。GRU のゲートでも σ(±0) は同じ値で、tanh(±0) の符号の違いは
 * その後の足し算で消える。
 */
final class Dense {

  private Dense() {}

  /**
   * y = ReLU(W·x ＋ b) を float で求める。
   *
   * @param weights 重みの配列
   * @param weight 重みの配列の中で W が始まる位置
   * @param bias 重みの配列の中で b が始まる位置
   * @param x 入力
   * @param inputs 入力の数
   * @param out y の書き先。先頭から {@code outputs} 個に書く
   * @param outputs 出力の数
   */
  static void relu(float[] weights, int weight, int bias, float[] x, int inputs, float[] out, int outputs) {
    affine(weights, weight, bias, x, inputs, out, outputs);
    for (int j = 0; j < outputs; j++) {
      out[j] = out[j] > 0f ? out[j] : 0f;
    }
  }

  /** y = W·x ＋ b を float で求める（活性化なし）。引数は {@link #relu} と同じ。 */
  static void affine(float[] weights, int weight, int bias, float[] x, int inputs, float[] out, int outputs) {
    System.arraycopy(weights, bias, out, 0, outputs);
    for (int i = 0; i < inputs; i++) {
      float xi = x[i];
      if (xi == 0f) {
        continue;
      }
      int row = weight + i * outputs;
      for (int j = 0; j < outputs; j++) {
        out[j] += weights[row + j] * xi;
      }
    }
  }

  /**
   * z = W·h ＋ b を double で求める。float どうしの積は、double なら丸めずに表せる。ほかの引数は {@link #relu} と同じ。
   *
   * @param z z の書き先。先頭から {@code outputs} 個に書く
   */
  static void logits(float[] weights, int weight, int bias, float[] h, int inputs, double[] z, int outputs) {
    for (int k = 0; k < outputs; k++) {
      z[k] = weights[bias + k];
    }
    for (int i = 0; i < inputs; i++) {
      float hi = h[i];
      if (hi == 0f) {
        continue;
      }
      int row = weight + i * outputs;
      for (int k = 0; k < outputs; k++) {
        z[k] += (double) weights[row + k] * hi;
      }
    }
  }
}
