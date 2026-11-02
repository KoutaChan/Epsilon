package com.epsilon.prism.luck;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;

/**
 * 運の網 Ω の制御変量（AIVAT の形）。偶然の出来事 k（状態 x、実際の結果 d）の席 r の項は
 *
 * <pre>
 * c[r] = V(x, d)[r] − Σ_t p(t)·V(x, t)[r]
 * </pre>
 *
 * で、p(t) = n_t / N は本当の残りの山（プール: 4 − 公開 − 全員の手。{@link LuckObservation#poolOf}）の上の一様な分布。結果 t は 34 種の赤でない
 * 牌と赤 5 の 3 色の 37 通り。
 *
 * <p><b>偏りが 0 になる条件</b>（Ω の出来に依らない）。結果 d は x のもとでプールの上に一様に出る（牌山は一様な並べ替えで、まだ開いていない牌は
 * 交換可能）ので、V が次の 3 つを満たせば E[c | x] = Σ_t p(t)·V(x, t) − Σ_t p(t)·V(x, t) = 0。
 *
 * <ol>
 *   <li>V は出来事の前の状態 x だけの関数である。入力は O° の出来事の前の事実だけで、結果の欄・山の並び・裏ドラ・未公開の表示牌を読まない
 *       （{@link LuckTokenizer}）。結果 d は、どの結果の値を読むかにだけ使う。
 *   <li>V の重みは、c を付ける出来事の牌山より前に決まっている（コホート t の収集で使う Ω は、コホート t − 1 までの標本で当てたもの）。
 *   <li>行をまたぐ縮約がない。V は行ごとに独立に計算する（背骨の注意・頭・4 席の中心化・この期待は全て行の中で閉じる）ので、塊の組み方に依ら
 *       ない。
 * </ol>
 *
 * <p>Ω の出来が効くのは c が Y の雑音をどれだけ消すか（R²）だけで、偏りには効かない。
 */
public final class ControlVariate {

  private ControlVariate() {}

  /**
   * 制御変量 c を返す。
   *
   * @param values 網の出力 V [行, 37, 4]（FP32）
   * @param pool 正準の結果の順のプールの枚数 [行, 37]（FP32）
   * @param chosen 実際の結果の平らな番号 [行]（INT64、行 × 37 + 正準の結果）
   * @return c [行, 4]（FP32）
   */
  public static NDArray of(NDArray values, NDArray pool, NDArray chosen) {
    return chosen(values, chosen).sub(expected(values, pool));
  }

  /**
   * 出来事の前の期待 Σ_t p(t)·V(x, t) を返す。
   *
   * @param values 網の出力 V [行, 37, 4]
   * @param pool 正準の結果の順のプールの枚数 [行, 37]
   * @return 期待 [行, 4]
   */
  public static NDArray expected(NDArray values, NDArray pool) {
    NDArray probability = pool.div(pool.sum(new int[] {1}, true));
    return values.mul(probability.expandDims(2)).sum(new int[] {1});
  }

  /**
   * 実際の結果の V(x, d) を返す。
   *
   * @param values 網の出力 V [行, 37, 4]
   * @param chosen 実際の結果の平らな番号 [行]（INT64、行 × 37 + 正準の結果）
   * @return V(x, d) [行, 4]
   */
  public static NDArray chosen(NDArray values, NDArray chosen) {
    long rows = values.getShape().get(0);
    long outcomes = values.getShape().get(1);
    long seats = values.getShape().get(2);
    return NDArrays.gatherRows(values.reshape(rows * outcomes, seats), chosen);
  }
}
