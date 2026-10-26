package com.epsilon.prism.run;

/**
 * 方策の記憶: 打つ網 A を学習の網 θ の重み付きの平均にする決め方。版 v を公開するたびに A_v = (1 − η_v)·A_{v−1} + η_v·θ_v とする
 * （{@link com.epsilon.prism.model.PrismModel#mixIn}）。
 *
 * <ul>
 *   <li>{@value #NONE}: 平均しない（η = 1、A = θ）。
 *   <li>{@value #EMA}: 指数移動平均。η = 1 − 2^{−1/halfLife}（版 halfLife 個前の重みが半分になる）。
 *   <li>{@value #POWER}: 版 k（平均の始まりからの番号、始まりの版が 1）の重みを k^power とした平均。η_k = k^power / Σ_{j≤k} j^power。
 *       power = 0 は一様な平均（CFR の平均方策）、power = 1 は線形 CFR の重み。
 * </ul>
 *
 * from より前の版は平均しない（A = θ）。版 from で平均が始まり、そのときの A は θ_from そのもの（η = 1）。
 *
 * @param scheme {@value #NONE}・{@value #EMA}・{@value #POWER}
 * @param halfLife ema の半減期（版の数、正）
 * @param power power の指数（0 以上）
 * @param from 平均を始める版（1 以上）
 */
public record Averaging(String scheme, double halfLife, double power, int from) {
  public static final String NONE = "none", EMA = "ema", POWER = "power";

  /** 網の名前（collectWith・learnFrom）。 */
  public static final String AVERAGE = "average", ITERATE = "iterate";

  public Averaging {
    if (!NONE.equals(scheme) && !EMA.equals(scheme) && !POWER.equals(scheme))
      throw new IllegalArgumentException(
          "averaging must be " + NONE + ", " + EMA + " or " + POWER + ": " + scheme);
    if (!(halfLife > 0) || !(power >= 0) || !Double.isFinite(power) || from < 1)
      throw new IllegalArgumentException(
          "Invalid averaging " + scheme + " halfLife=" + halfLife + " power=" + power);
  }

  /** 平均を持つか（none でない）。 */
  public boolean enabled() {
    return !NONE.equals(scheme);
  }

  /**
   * 版 version を公開するときの混ぜる割合と、power の重みの累計。
   *
   * @param eta A_version = (1 − eta)·A_{version−1} + eta·θ_version の eta（1 なら A = θ）
   * @param totalWeight power の重みの累計 Σ_{j≤k} j^power（ほかの決め方では 0）
   */
  public record Step(double eta, double totalWeight) {}

  /**
   * 版 version の Step。
   *
   * @param totalWeight 版 version − 1 までの重みの累計（{@link Step#totalWeight}。始まりの前は 0）
   */
  public Step step(int version, double totalWeight) {
    if (!enabled() || version <= from) {
      double first = POWER.equals(scheme) && version == from ? 1 : 0;
      return new Step(1, first);
    }
    return switch (scheme) {
      case EMA -> new Step(1 - Math.pow(2, -1 / halfLife), 0);
      case POWER -> {
        double weight = Math.pow(version - from + 1, power);
        double total = totalWeight + weight;
        yield new Step(weight / total, total);
      }
      default -> throw new IllegalStateException(scheme);
    };
  }
}
