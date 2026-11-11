package com.epsilon.classic.learn;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;

/**
 * 1 つのバッチの損失（FP32）と、ステップの中で足し合わせる指標を求める。損失は、選んだ手だけで取る PPO の方策の損失、エントロピーの項、
 * 価値の損失の和。行 i で選んだ手を a、学習の前の方策を π_前、今の方策を π とする。
 *
 * <pre>
 * r_i  = π(a)/π_前(a)
 * 方策     = −Σ_i w_i·c_i·min(r_i·A_i, clip(r_i, 1 − ε, 1 + ε)·A_i) / N_方策
 * エントロピー = −β·Σ_i w_i·H(π_i) / N_方策
 * 価値     = Σ_i CE(HL-Gauss(v_i), 価値の分布_i) / N_価値
 * </pre>
 *
 * w_i は方策の損失に入れる行で 1、ほかの行で 0。c_i は選んだ手の補正（{@link ValueTargets#credit}）、A_i はアドバンテージ、v_i は価値の目標。
 * N_方策 はステップの中で方策の損失に入れる行の数、N_価値 はステップの行の数。どちらもステップの全てのバッチを合わせて数えるので、
 * バッチごとの勾配を足すとステップの平均の勾配になる。エントロピーは全ての合法手で取る。
 */
final class PpoLoss {

  /** 足し合わせる指標。{@link Terms#sums} はこの順に並ぶ。どれも行の和で、勾配は流さない。 */
  enum Sum {
    /** Σ w·c·min(r·A, clip(r)·A)。 */
    SURROGATE,
    /** Σ w·H(π)。 */
    ENTROPY,
    /** Σ w·max(0, KL(π_前 ‖ π))。 */
    KL,
    /** Σ 価値の交差エントロピー。 */
    VALUE,
    /** Σ w·[|r − 1| > ε]。 */
    CLIPPED
  }

  /**
   * 損失の係数。
   *
   * @param clip PPO のクリップの幅 ε
   * @param entropy エントロピーの係数 β
   * @param actorRows ステップの中で方策の損失に入れる行の数 N_方策（0 なら方策とエントロピーの項を入れない）
   * @param valueRows ステップの行の数 N_価値（1 以上）
   */
  record Coefficients(double clip, double entropy, double actorRows, double valueRows) {}

  /**
   * 損失と指標。
   *
   * @param objective 逆伝播する損失（スカラー）
   * @param sums 指標 [{@link Sum} の数]（FP32。勾配は流さない）
   */
  record Terms(NDArray objective, NDArray sums) {}

  private PpoLoss() {}

  /**
   * バッチの損失と指標を求める。どの配列も同じマネージャーとデバイスに置き、先頭の軸を行とする。
   *
   * @param logProbabilities 今の方策の合法手の対数確率 [行, 合法手]（FP32。合法でない位置は大きな負の値）
   * @param legal 合法手の位置で 1、ほかは 0 [行, 合法手]
   * @param chosen 選んだ手の位置で 1、ほかは 0 [行, 合法手]
   * @param rolloutLog 学習の前の方策の対数確率 [行, 合法手]（合法でない位置は 0）
   * @param advantage アドバンテージ A [行]
   * @param credit 選んだ手の補正 c [行]
   * @param actor 方策の損失に入れる行で 1、ほかは 0 [行]
   * @param valueLogits 価値のビンのロジット [行, ビン]（FP32）
   * @param valueTargets 価値の目標の分布 [行, ビン]
   * @param coefficients 損失の係数
   * @return 損失と指標
   */
  static Terms compute(
      NDArray logProbabilities,
      NDArray legal,
      NDArray chosen,
      NDArray rolloutLog,
      NDArray advantage,
      NDArray credit,
      NDArray actor,
      NDArray valueLogits,
      NDArray valueTargets,
      Coefficients coefficients) {
    int[] actions = {1};
    NDArray valueRows = valueTargets.mul(valueLogits.logSoftmax(1)).sum(actions).neg();
    NDArray value = valueRows.sum();
    NDArray objective = value.div((float) coefficients.valueRows());

    // 方策の損失に入れない行は比を 1 にする（後で 0 を掛けるが、その前に比が溢れないようにするため）
    NDArray logChosen = logProbabilities.mul(chosen).sum(actions);
    NDArray ratio = logChosen.sub(rolloutLog.mul(chosen).sum(actions)).mul(actor).exp();
    float low = (float) (1 - coefficients.clip());
    float high = (float) (1 + coefficients.clip());
    NDArray surrogate = ratio.mul(advantage).minimum(ratio.clip(low, high).mul(advantage));
    NDArray weighted = surrogate.mul(credit).mul(actor).sum();
    NDArray probabilities = logProbabilities.exp().mul(legal);
    NDArray entropy = probabilities.mul(logProbabilities.mul(legal)).sum(actions).neg().mul(actor).sum();
    if (coefficients.actorRows() > 0) {
      float policyScale = (float) (1 / coefficients.actorRows());
      objective =
          objective
              .sub(weighted.mul(policyScale))
              .sub(entropy.mul((float) (coefficients.entropy() * policyScale)));
    }

    // 丸め誤差で行の KL がわずかに負になることがあるので、0 で下から抑える
    NDArray detached = logProbabilities.stopGradient().mul(legal);
    NDArray kl =
        rolloutLog.exp().mul(legal).mul(rolloutLog.sub(detached)).sum(actions).maximum(0f).mul(actor).sum();
    float clip = (float) coefficients.clip();
    NDArray clipped =
        ratio
            .stopGradient()
            .sub(1f)
            .abs()
            .gt(clip)
            .toType(DataType.FLOAT32, false)
            .mul(actor)
            .sum();
    NDArray sums =
        NDArrays.stack(
                new NDList(weighted.stopGradient(), entropy.stopGradient(), kl, value.stopGradient(), clipped))
            .stopGradient();
    return new Terms(objective, sums);
  }
}
