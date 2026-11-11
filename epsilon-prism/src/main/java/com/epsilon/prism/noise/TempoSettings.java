package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import java.util.Objects;

/**
 * TEMPO のセクション。TEMPO は、学習中のネットワークをアンカーにして、その反復のコホートだけで学習する。教師は、ネットワークの出力と
 * ラベルを教師のゲインで混ぜたもので、補助のヘッドの項はゲイン倍する。こうするとネットワークの出力は、全てのコホートのラベルの再帰平均
 * になる。TEMPO を使わないときはアンカーを混ぜず、教師はラベル、補助のヘッドのゲインと学習率の倍率は 1 になる。
 */
@Choice(key = "enabled", doc = "TEMPO を使うか")
public sealed interface TempoSettings {

  /**
   * 反復 t の教師のゲインを返す。
   *
   * @param t 反復
   * @param r2 前の反復で教師に掛けた補正の決定係数。ホールドアウトで測った値で、0 以下か NaN なら持ち越さない
   * @return 教師のゲイン
   */
  double gain(int t, double r2);

  /**
   * 反復 t の学習率の倍率を返す。
   *
   * @param t 反復
   * @param r2 教師のゲインを求めるのに使う決定係数
   * @return 学習率の倍率（(0, 1]）
   */
  double learningRateScale(int t, double r2);

  /** アンカーを混ぜるなら true。 */
  boolean anchored();

  /** Φ が点数だけの Φ から GRP に替わる反復で、運の精算の決定係数を持ち越さないなら true。 */
  boolean resetsOnPhiKind();

  /** アンカーなし。教師はラベル、補助のヘッドのゲインと学習率の倍率は 1。 */
  @Case("false")
  record Off() implements TempoSettings {

    @Override
    public double gain(int t, double r2) {
      return 1;
    }

    @Override
    public double learningRateScale(int t, double r2) {
      return 1;
    }

    @Override
    public boolean anchored() {
      return false;
    }

    @Override
    public boolean resetsOnPhiKind() {
      return false;
    }
  }

  /**
   * アンカーあり。K_t = gain·(1 + max(0, t − from)/tau)^(−power)。
   *
   * @param from ゲインの数え始めの反復
   * @param gain ゲインの初期値
   * @param tau ゲインの減りの時定数（反復）
   * @param power ゲインの減りの指数（0 なら一定のゲイン）
   * @param noiseGain 教師のゲインに、教師のノイズの減りをどう入れるか
   * @param resetOnPhiKind Φ が点数だけの Φ から GRP に替わる反復で、運の精算の決定係数を持ち越さないか
   * @param lrMode 学習率の倍率の決め方
   * @param lrReferenceGain {@link TempoLrMode#GAIN} の基準のゲイン
   */
  @Case("true")
  record On(
      @Key(doc = "ゲインの数え始めの反復", value = "1") @Range(min = 1) int from,
      @Key(doc = "ゲインの初期値", value = "0.9") @Range(min = 0, minInclusive = false, max = 1, maxInclusive = false)
          double gain,
      @Key(doc = "ゲインの減りの時定数（反復）", value = "4.0") @Range(min = 0, minInclusive = false) double tau,
      @Key(doc = "ゲインの減りの指数（0 なら一定のゲイン）", value = "0.6666667") @Range(min = 0) double power,
      @Key(doc = "教師のゲインに、教師のノイズの減りをどう入れるか", value = "LEDGER_R2") NoiseGain noiseGain,
      @Key(doc = "Φ が点数だけの Φ から GRP に替わる反復で、運の精算の決定係数を持ち越さないか", value = "true") boolean resetOnPhiKind,
      @Key(doc = "学習率の倍率の決め方", value = "GAIN") TempoLrMode lrMode,
      @Key(doc = "学習率の倍率（GAIN）の基準にするゲイン", value = "0.25")
          @Range(min = 0, minInclusive = false, max = 1)
          double lrReferenceGain)
      implements TempoSettings {

    public On {
      Objects.requireNonNull(noiseGain, "noiseGain");
      Objects.requireNonNull(lrMode, "lrMode");
    }

    /** 反復 t のゲイン K_t を、減り方の式だけで求めて返す（ノイズの減りは入れない）。 */
    public double scheduledGain(int t) {
      return gain * Math.pow(1 + Math.max(0, t - from) / tau, -power);
    }

    /**
     * 反復 t の教師のゲインを返す。{@link NoiseGain#LEDGER_R2} なら、減り方の式のゲインを、補正でノイズが決定係数の分だけ減ったときの
     * 最適なゲインに直した値、{@link NoiseGain#NONE} なら減り方の式のゲイン。
     *
     * @throws IllegalArgumentException 決定係数が 1 を超えるとき
     */
    @Override
    public double gain(int t, double r2) {
      double scheduled = scheduledGain(t);
      if (noiseGain != NoiseGain.LEDGER_R2 || !(r2 > 0)) {
        return scheduled;
      }
      if (!(r2 <= 1)) {
        throw new IllegalArgumentException("R² must not exceed 1: " + r2);
      }
      return scheduled / (scheduled + (1 - scheduled) * (1 - r2));
    }

    @Override
    public double learningRateScale(int t, double r2) {
      return lrMode.scale(gain(t, r2), lrReferenceGain);
    }

    @Override
    public boolean anchored() {
      return true;
    }

    @Override
    public boolean resetsOnPhiKind() {
      return resetOnPhiKind;
    }
  }
}
