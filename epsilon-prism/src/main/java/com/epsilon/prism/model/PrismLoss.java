package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.util.List;
import java.util.Objects;

/**
 * 当てはめの損失。
 *
 * <pre>
 * ℓ = (1/B_C)·[ Σ_{|C|≥2} w·ℓ_PRISM + η·Σ_{|D|≥1} w·s̄_D² ] + share·K′·[ α_o·Σ_読み L̄_h + α_r·(1/8)·Σ_候補 L̄_h ]
 * D = 行の測った手 C のうち値を B が出す打牌、s̄_D = D の上の s の平均
 * share = 1 歩の行のうち、微小バッチの元の塊（詰め方の 1 塊）の行の割合
 * </pre>
 *
 * <p>ℓ_PRISM は差分整合の κ = 0・KL の基準は C の上の一様分布で（{@link ValueLabel} は p0 を持たない）、教師は順伝播の前に送る
 * （下の 2 段）。ℓ_PRISM は s の行の定数に依らないので、それだけでは行の定数が当てはめの間に漂い、|s| が大きくなると BF16 の丸めが
 * 手の差を消す。行の定数は打牌の値 B の中にある（立直・槓・ツモ・九種は B から作る M に門 Δ を足し、応答の行は s[PASS] = 0 で定数が
 * 決まる）ので、行の平均の錨 η·s̄_D² は測った打牌 D だけで取る。D の全ての手を同じだけ動かすので手の差（ℓ_PRISM の最適）にも門 Δ
 * にも効かず、B の定数だけを 0 に止める（最適は s = Q − mean_D Q）。打つ手は行の定数に依らないので、目標を Y の平均（局 1 本の雑音を
 * 含む絶対値）にする理由はない。補助の項は {@link AuxiliaryLoss}。
 *
 * <p>損失は 2 段で取る。学習のレーンは網の順伝播の前に教師をデバイスへ送り（{@link LossInputs#of(NetworkInputs, List, AuxiliaryLabels,
 * AuxiliaryLoss.Denominators)}）、順伝播の後は送った配列を読むだけにする（{@link #total(Outputs, LossInputs, float, int, float)}）。GPU の
 * 管理元での配列の作成はホストからの写しとストリームの同期なので、順伝播の後に作るとホストは順伝播が終わるまで待ち、その間に損失と逆伝播を
 * 積めない。網の出力が既にあるときの口（{@link #total(NDManager, Outputs, List, AuxiliaryLabels, float, int, float)}・{@link #value}・
 * {@link #rowMean}）は、出力から教師を作って送り、同じ演算を通す。
 *
 * @param auxiliary 補助の頭の損失
 */
public record PrismLoss(AuxiliaryLoss auxiliary) {

  /** 行の平均の錨の重み η。 */
  static final float ROW_MEAN = 1f;

  /** 手の軸。 */
  private static final int[] ACTIONS = {1};

  /** where で外す枠の値（exp で 0 になる。FP32 で有限）。 */
  private static final float FLOOR = -1e30f;

  /**
   * 損失を確かめる。
   *
   * @throws NullPointerException 補助の損失がないとき
   */
  public PrismLoss {
    Objects.requireNonNull(auxiliary, "auxiliary");
  }

  /**
   * 行が差分整合の分母に入るなら true を返す（|C| ≥ 2）。
   *
   * @param label 行の教師
   * @return 分母に入るなら true
   */
  public static boolean contributes(ValueLabel label) {
    return label.measuredCount() >= 2;
  }

  /**
   * 差分整合の行の損失の和 Σ w·ℓ_PRISM（分母で割る前、FP32 のスカラー。網の出力が既にあるときの口）。s の行の定数に依らない。
   *
   * @param scope 作った配列を付ける管理元
   * @param outputs 網の出力
   * @param labels 行の順の教師
   * @return 和
   * @throws IllegalArgumentException 行の数か行の手の数が出力と合わないとき
   */
  public NDArray value(NDManager scope, Outputs outputs, List<ValueLabel> labels) {
    return valueSum(scope, outputs.scores(), LossInputs.of(scope, outputs, labels, null, null));
  }

  /**
   * 行の平均の錨の和 Σ_{|D|≥1} w·s̄_D²（分母で割る前、FP32 のスカラー。網の出力が既にあるときの口）。D は測った打牌
   * （{@link Outputs#discards}）で、ほかの手と詰め物は読まない。
   *
   * @param scope 作った配列を付ける管理元
   * @param outputs 網の出力
   * @param labels 行の順の教師
   * @return 和
   * @throws IllegalArgumentException 行の数か行の手の数が出力と合わないとき
   */
  public NDArray rowMean(NDManager scope, Outputs outputs, List<ValueLabel> labels) {
    return rowMeanSum(scope, outputs.scores(), LossInputs.of(scope, outputs, labels, null, null));
  }

  /**
   * 微小バッチの損失（網の出力が既にあるときの口）。教師を出力から作って送り、{@link #total(Outputs, LossInputs, float, int, float)} と
   * 同じ演算を通す。補助の頭の分母は教師から数える。
   *
   * @param scope 作った配列を付ける管理元
   * @param outputs 網の出力（補助の頭を計算した順伝播なら補助の項も足す）
   * @param labels 行の順の教師
   * @param auxiliaryLabels 補助の頭の教師（補助の項を足さないなら null）
   * @param gain ゲイン K′（replay では 1）
   * @param denominator 1 歩の全レーンで |C| ≥ 2 の行の数 B_C（なければ 1）
   * @param share 1 歩の行のうちこの微小バッチの割合（1 歩が 1 つの微小バッチなら 1）
   * @return 損失（FP32 のスカラー）
   */
  public NDArray total(
      NDManager scope,
      Outputs outputs,
      List<ValueLabel> labels,
      AuxiliaryLabels auxiliaryLabels,
      float gain,
      int denominator,
      float share) {
    LossInputs inputs =
        LossInputs.of(
            scope,
            outputs,
            labels,
            auxiliaryLabels,
            auxiliaryLabels == null ? null : AuxiliaryLoss.Denominators.of(auxiliaryLabels));
    return total(outputs, inputs, gain, denominator, share);
  }

  /**
   * 微小バッチの損失 (1/B_C)·[Σ w·ℓ_PRISM + η·Σ w·s̄_D²] + share·L_aux（教師は順伝播の前に送ったもの）。
   *
   * @param outputs 網の出力（補助の頭を計算した順伝播なら補助の項も足す）
   * @param inputs 送った教師（{@link LossInputs#of(NetworkInputs, List, AuxiliaryLabels, AuxiliaryLoss.Denominators)}。補助の教師が
   *     なければ補助の項を足さない）
   * @param gain ゲイン K′（replay では 1）
   * @param denominator 1 歩の全レーンで |C| ≥ 2 の行の数 B_C（なければ 1）
   * @param share 1 歩の行のうち、微小バッチの元の塊の行の割合（1 歩が 1 つの微小バッチなら 1）
   * @return 損失（FP32 のスカラー）
   * @throws IllegalArgumentException 分母か割合が範囲の外か、網の出力が送った教師と合わないとき
   */
  public NDArray total(Outputs outputs, LossInputs inputs, float gain, int denominator, float share) {
    if (denominator < 1 || !(share >= 0) || share > 1) {
      throw new IllegalArgumentException("invalid denominator or share: " + denominator + ", " + share);
    }
    inputs.check(outputs);
    NDManager scope = inputs.manager();
    NDArray scores = outputs.scores();
    NDArray loss =
        valueSum(scope, scores, inputs).add(rowMeanSum(scope, scores, inputs).mul(ROW_MEAN)).div(denominator);
    if (inputs.auxiliary() != null && outputs.auxiliary() != null) {
      loss = loss.add(auxiliary.terms(scope, outputs.auxiliary(), inputs.auxiliary(), gain).total().mul(share));
    }
    return loss;
  }

  /**
   * 差分整合の行の損失の和 Σ w·ℓ_PRISM。z = s/λ、π_C = softmax_C(z) として
   *
   * <pre>
   * ℓ = − Σ_{a∈C} π_C(a)·Y_a + λ·Σ_{a∈C} π_C(a)·(log π_C(a) − log p0_C(a))
   * </pre>
   */
  private static NDArray valueSum(NDManager scope, NDArray scores, LossInputs inputs) {
    NDArray lambda = inputs.lambda();
    Softmax c = Softmax.of(scope, scores.div(lambda), inputs.measured());
    NDArray rowLoss =
        c.pi()
            .mul(inputs.targets())
            .sum(ACTIONS, true)
            .neg()
            .add(c.pi().mul(c.logPi().sub(inputs.logPrior())).sum(ACTIONS, true).mul(lambda));
    return rowLoss.mul(inputs.weights()).sum();
  }

  /** 行の平均の錨の和 Σ_{|D|≥1} w·s̄_D²（D の外の手と詰め物は where で外す）。 */
  private static NDArray rowMeanSum(NDManager scope, NDArray scores, LossInputs inputs) {
    NDArray kept = NDArrays.where(inputs.measuredDiscards(), scores, scope.zeros(scores.getShape()));
    NDArray mean = kept.sum(new int[] {1}).div(inputs.discardCounts());
    return mean.square().mul(inputs.discardWeights()).sum();
  }

  /**
   * C の上の softmax（κ = 0 なので log Σ exp は作らない）。C の外の枠は掛け算でなく where で外すので、
   * そこの値が何でも値にも勾配にも混ざらない。安定化に引く行の最大値は勾配を切る（softmax は行の定数で変わらないので勾配は同じ）。
   *
   * @param pi C の上の確率（外は 0）
   * @param logPi C の上の log 確率（外は 0）
   */
  private record Softmax(NDArray pi, NDArray logPi) {

    static Softmax of(NDManager scope, NDArray z, NDArray mask) {
      Shape shape = z.getShape();
      NDArray maskF = mask.toType(DataType.FLOAT32, false);
      // C が空でない行で 1、空の行で 0。[行, 1]
      NDArray nonEmpty = maskF.sum(ACTIONS, true).minimum(1f);
      NDArray zeros = scope.zeros(shape, DataType.FLOAT32);
      NDArray peak =
          NDArrays.where(mask, z, scope.full(shape, FLOOR))
              .max(ACTIONS, true)
              .stopGradient()
              .mul(nonEmpty);
      // C の上は z − 最大値（≤ 0）、外は 0（exp は 1 だが maskF で消す）
      NDArray shifted = NDArrays.where(mask, z.sub(peak), zeros);
      NDArray e = shifted.exp().mul(maskF);
      // C が空でない行は和 ≥ 1（最大の手が exp(0) = 1）。空の行は 1 を足して log を有限にする
      NDArray sum = e.sum(ACTIONS, true).add(nonEmpty.neg().add(1f));
      NDArray logSum = sum.log();
      return new Softmax(e.div(sum), shifted.sub(logSum).mul(maskF));
    }
  }
}
