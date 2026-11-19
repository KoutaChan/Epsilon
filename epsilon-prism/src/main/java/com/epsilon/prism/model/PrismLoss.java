package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.util.List;
import java.util.Objects;

/**
 * 学習の損失。
 *
 * <pre>
 * ℓ = (1/B_C)·[ Σ_{|C|≥2} w·ℓ_PRISM + η·K′·Σ_{|D|≥1} w·s̄_D² ] + share·K′·[ α_o·Σ_読み L̄_h + α_r·(1/8)·Σ_候補 L̄_h ]
 * D = 行の測った手 C のうち、値を B が出す打牌。s̄_D = D の上の s の平均
 * share = 1 ステップの行のうち、マイクロバッチの元のバッチ（詰め方で決まる 1 バッチ）の行が占める割合
 * </pre>
 *
 * <p>ℓ_PRISM は κ = 0 の差分整合の損失で、KL の基準は C の上の一様分布にする（{@link ValueLabel} は p0 を持たない）。教師は順伝播の
 * 前に送る（2 段で取る理由は下に書く）。
 *
 * <p>ℓ_PRISM のはじめの 2 項の s_a への勾配は (π_C(a)/λ)·(e_a − ē_π) になる（e = s − Y、ē_π は π_C で重みを付けた e の平均）。π_C(a) が
 * 0 か 1 に張り付いた手には、誤差がいくら大きくても勾配が届かない。ゲート Δ のように行によらずほぼ同じ値を出す手は、全ての行で同時に
 * 張り付くと戻れない。そこで 3 番目の項で、測った全ての手に重みの下限 ε/|C| を足す。この項は Y について線形なので、ラベルの雑音の平均が
 * 0 なら偏らない。最適もはじめの 2 項と同じ s_a − s_b = Y_a − Y_b なので、学ぶ値は変えずに、張り付いた手を戻す勾配だけを足す。
 *
 * <p>ℓ_PRISM は s の行ごとの定数に依らない。そのため、これだけでは行の定数が学習の間にずれていき、|s| が大きくなると BF16 の丸めで手の差が
 * 消える。行の定数は打牌の値 B が決める（立直・槓・ツモ・九種は B から作る M にゲート Δ を足し、応答の行は s[PASS] = 0 で定数が決まる）。
 * そこで、行の平均のアンカー η·s̄_D² は測った打牌 D だけで取る。この項は D の全ての手を同じだけ動かすので、手の差（ℓ_PRISM の最適）にも
 * ゲート Δ にも効かず、B の定数だけを 0 に留める（最適は s = Q − mean_D Q）。打つ手は行の定数に依らないので、目標を Y の平均（局 1 本の
 * ノイズを含む絶対値）にする理由はない。補助の項は {@link AuxiliaryLoss} が受け持つ。
 *
 * <p>アンカーには教師のゲイン K′ を掛ける（{@link RowAnchor#GAIN}）。教師 (1−K′)·s_アンカー + K′·Ỹ では ℓ_PRISM が s を動かす力が K′ 倍になるので、アンカーだけを
 * 1 倍のままにすると、行の定数を留める力が手の差を学ぶ力に勝つ。D は行の種類で違い（根の記録では打った手と逸脱した手の 2 手だけ）、
 * 似た局面の行どうしで違う手の集合の平均を同時に 0 に留めることになるので、その釣り合いは手の差を縮める向きに働く。
 *
 * <p>損失は 2 段で取る。学習のレーンは、ネットワークの順伝播の前に教師をデバイスへ送り（{@link StagedBatch#send}）、順伝播の後は送った
 * 配列を読むだけにする（{@link #total(Outputs, LossInputs, float, int, float)}）。GPU のマネージャーで同期してコピーすると、ストリームが
 * 同期する。順伝播の後にコピーすると、ホストは順伝播が終わるまで待ち、その間に損失と逆伝播を積めない。ネットワークの出力が既にある
 * ときに使うメソッド（{@link #total(NDManager, Outputs, List, AuxiliaryLabels, float, int, float)}・{@link #value}・{@link #rowMean}）は、
 * 出力から教師を作って送り、同じ演算を通す。
 *
 * @param auxiliary 補助のヘッドの損失
 * @param rowAnchor 行の平均のアンカーの重みの決め方
 */
public record PrismLoss(AuxiliaryLoss auxiliary, RowAnchor rowAnchor) {

  /** 行の平均のアンカーの重みの決め方。 */
  public enum RowAnchor {
    /** η·K′。差分整合の教師と同じ目盛りにする。 */
    GAIN,
    /** η。教師のゲインに依らない。 */
    FIXED;

    /** ゲイン K′ のときのアンカーの重みを返す。 */
    float weight(float gain) {
      return this == GAIN ? ROW_MEAN * gain : ROW_MEAN;
    }
  }

  /** 行の平均のアンカーの重み η。 */
  static final float ROW_MEAN = 1f;

  /** 測った手の重みの下限 ε。C の手ひとつあたりの下限は ε/|C|。 */
  static final float WEIGHT_FLOOR = 0.05f;

  /** 手の軸。 */
  private static final int[] ACTIONS = {1};

  /** where で外した位置に置く値。exp で 0 になり、FP32 で有限。 */
  private static final float FLOOR = -1e30f;

  /**
   * 補助の損失とアンカーの決め方があることを確かめる。
   *
   * @throws NullPointerException 補助の損失かアンカーの決め方がないとき
   */
  public PrismLoss {
    Objects.requireNonNull(auxiliary, "auxiliary");
    Objects.requireNonNull(rowAnchor, "rowAnchor");
  }

  /** アンカーに教師のゲインを掛ける損失（{@link RowAnchor#GAIN}）。 */
  public PrismLoss(AuxiliaryLoss auxiliary) {
    this(auxiliary, RowAnchor.GAIN);
  }

  /**
   * 行が差分整合の損失の分母に数えられるか（|C| ≥ 2 か）を返す。
   *
   * @param label 行の教師
   * @return 分母に数えるなら true
   */
  public static boolean contributes(ValueLabel label) {
    return label.measuredCount() >= 2;
  }

  /** 重み 1 のラベル行の差分整合損失。教師が NaN の手は除き、測った手が 2 つ未満なら 0 を返す。 */
  public static double value(float[] scores, float[] targets, double lambda) {
    double peak = Double.NEGATIVE_INFINITY;
    double meanError = 0;
    int count = 0;
    for (int a = 0; a < targets.length; a++) {
      if (Float.isNaN(targets[a])) continue;
      peak = Math.max(peak, scores[a]);
      meanError += (double) scores[a] - targets[a];
      count++;
    }
    if (count < 2) return 0;
    meanError /= count;
    double mass = 0, weighted = 0, squares = 0;
    for (int a = 0; a < targets.length; a++) {
      if (Float.isNaN(targets[a])) continue;
      double shifted = scores[a] - peak;
      double weight = Math.exp(shifted / lambda);
      mass += weight;
      weighted += weight * (shifted - targets[a]);
      double error = (double) scores[a] - targets[a] - meanError;
      squares += error * error;
    }
    return weighted / mass
        + lambda * Math.log(count / mass)
        + WEIGHT_FLOOR * squares / (2 * lambda * count);
  }

  /**
   * 差分整合の行の損失の和 Σ w·ℓ_PRISM を返す。ネットワークの出力が既にあるときに使う。この和は s の行ごとの定数に依らない。
   *
   * @param scope 作った配列を持たせるマネージャー
   * @param outputs ネットワークの出力
   * @param labels 行の順に並べた教師
   * @return 分母で割る前の和（FP32 のスカラー）
   * @throws IllegalArgumentException 行の数か、行の手の数が出力と合わないとき
   */
  public NDArray value(NDManager scope, Outputs outputs, List<ValueLabel> labels) {
    return valueSum(scope, outputs.scores(), LossInputs.of(scope, outputs, labels, null, null));
  }

  /**
   * 行の平均のアンカーの和 Σ_{|D|≥1} w·s̄_D² を返す。ネットワークの出力が既にあるときに使う。D は測った打牌
   * （{@link Outputs#discards}）で、ほかの手とパディングは読まない。
   *
   * @param scope 作った配列を持たせるマネージャー
   * @param outputs ネットワークの出力
   * @param labels 行の順に並べた教師
   * @return 分母で割る前の和（FP32 のスカラー）
   * @throws IllegalArgumentException 行の数か、行の手の数が出力と合わないとき
   */
  public NDArray rowMean(NDManager scope, Outputs outputs, List<ValueLabel> labels) {
    return rowMeanSum(scope, outputs.scores(), LossInputs.of(scope, outputs, labels, null, null));
  }

  /**
   * マイクロバッチの損失を返す。ネットワークの出力が既にあるときに使う。教師を出力から作って送り、
   * {@link #total(Outputs, LossInputs, float, int, float)} と同じ演算を通す。補助のヘッドの分母は教師から数える。
   *
   * @param scope 作った配列を持たせるマネージャー
   * @param outputs ネットワークの出力（順伝播で補助のヘッドも計算したなら、補助の項も足す）
   * @param labels 行の順に並べた教師
   * @param auxiliaryLabels 補助のヘッドの教師（補助の項を足さないなら null）
   * @param gain ゲイン K′（replay では 1）
   * @param denominator 1 ステップの全レーンで |C| ≥ 2 の行の数 B_C（無ければ 1）
   * @param share 1 ステップの行のうち、このマイクロバッチが占める割合（1 ステップが 1 つのマイクロバッチなら 1）
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
   * マイクロバッチの損失 (1/B_C)·[Σ w·ℓ_PRISM + η·K′·Σ w·s̄_D²] + share·L_aux を返す。教師は順伝播の前に送ったものを読む。
   *
   * @param outputs ネットワークの出力（順伝播で補助のヘッドも計算したなら、補助の項も足す）
   * @param inputs 送った教師（{@link StagedBatch#send}。補助の教師が無ければ補助の項を足さない）
   * @param gain ゲイン K′（replay では 1）
   * @param denominator 1 ステップの全レーンで |C| ≥ 2 の行の数 B_C（無ければ 1）
   * @param share 1 ステップの行のうち、マイクロバッチの元のバッチの行が占める割合（1 ステップが 1 つのマイクロバッチなら 1）
   * @return 損失（FP32 のスカラー）
   * @throws IllegalArgumentException 分母か割合が範囲の外にあるか、ネットワークの出力が送った教師と合わないとき
   */
  public NDArray total(
      Outputs outputs, LossInputs inputs, float gain, int denominator, float share) {
    return total(outputs, inputs, gain, denominator, share, null, true);
  }

  /** 報告する行では順伝播の値を差分整合にそろえ、逆伝播だけに対の目的関数を使う。 */
  public NDArray total(
      Outputs outputs,
      LossInputs inputs,
      float gain,
      int denominator,
      float share,
      PairLoss pair,
      boolean report) {
    if (denominator < 1 || !(share >= 0) || share > 1) {
      throw new IllegalArgumentException(
          "invalid denominator or share: " + denominator + ", " + share);
    }
    inputs.check(outputs);
    NDManager scope = inputs.manager();
    NDArray scores = outputs.scores();
    NDArray value = pair == null ? valueSum(scope, scores, inputs) : pair.sum(scores, inputs);
    if (pair != null && report) {
      value = valueSum(scope, scores.stopGradient(), inputs).add(value.sub(value.stopGradient()));
    }
    NDArray loss =
        value.add(rowMeanSum(scope, scores, inputs).mul(rowAnchor.weight(gain))).div(denominator);
    if (inputs.auxiliary() != null && outputs.auxiliary() != null) {
      loss =
          loss.add(
              auxiliary
                  .terms(scope, outputs.auxiliary(), inputs.auxiliary(), gain)
                  .total()
                  .mul(share));
    }
    return loss;
  }

  /**
   * 差分整合の行の損失の和 Σ w·ℓ_PRISM。z = s/λ、π_C = softmax_C(z)、e = s − Y、ē_C は C の上の e の平均として、行の損失は次の式。
   *
   * <pre>
   * ℓ = − Σ_{a∈C} π_C(a)·Y_a + λ·Σ_{a∈C} π_C(a)·(log π_C(a) − log p0_C(a)) + (ε/2λ)·(1/|C|)·Σ_{a∈C} (e_a − ē_C)²
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
            .add(c.pi().mul(c.logPi().sub(inputs.logPrior())).sum(ACTIONS, true).mul(lambda))
            .add(weightFloor(scope, scores, inputs.targets(), inputs.measured()).div(lambda));
    return rowLoss.mul(inputs.weights()).sum();
  }

  /**
   * 重みの下限の項を λ で割る前の値 (ε/2)·(1/|C|)·Σ_{a∈C} (e_a − ē_C)² [行, 1]。C の外の手とパディングは where で外し、C が空の行は 0。
   *
   * @param scope 作った配列を持たせるマネージャー
   * @param scores s [行, 手の容量]
   * @param targets Y [行, 手の容量]（C の外は 0）
   * @param mask C [行, 手の容量]
   * @return 行ごとの値
   */
  private static NDArray weightFloor(
      NDManager scope, NDArray scores, NDArray targets, NDArray mask) {
    NDArray zeros = scope.zeros(scores.getShape(), DataType.FLOAT32);
    NDArray count = mask.toType(DataType.FLOAT32, false).sum(ACTIONS, true).maximum(1f);
    NDArray error = NDArrays.where(mask, scores.sub(targets), zeros);
    NDArray centered = NDArrays.where(mask, error.sub(error.sum(ACTIONS, true).div(count)), zeros);
    return centered.square().sum(ACTIONS, true).div(count).mul(WEIGHT_FLOOR / 2);
  }

  /** 行の平均のアンカーの和 Σ_{|D|≥1} w·s̄_D²。D の外の手とパディングは where で外す。 */
  private static NDArray rowMeanSum(NDManager scope, NDArray scores, LossInputs inputs) {
    NDArray kept =
        NDArrays.where(inputs.measuredDiscards(), scores, scope.zeros(scores.getShape()));
    NDArray mean = kept.sum(new int[] {1}).div(inputs.discardCounts());
    return mean.square().mul(inputs.discardWeights()).sum();
  }

  /**
   * C の上の softmax。κ = 0 なので log Σ exp は作らない。C の外の位置は掛け算でなく where で外すので、
   * そこにどんな値があっても、値にも勾配にも混ざらない。安定化のために引く行の最大値は勾配を切る（softmax は行の定数で変わらないので、
   * 勾配は同じになる）。
   *
   * @param pi C の上の確率（外は 0）
   * @param logPi C の上の log 確率（外は 0）
   */
  record Softmax(NDArray pi, NDArray logPi) {

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
