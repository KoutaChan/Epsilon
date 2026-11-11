package com.epsilon.grp.train;

/**
 * 同点のまわりの段差（{@link TieJumps#measure}）。今の順位が隣り合う 2 席について、起家に近い席（同点なら上位になる席）が
 * 最終順位でもう一方より上になる率を q とし、2 席の点差 d（起家に近い席 − もう一方）の関数として見る。曲線は d = −500〜+500 点の
 * 100 点ごとの値。予測は、2 席の間で点を動かした卓（点の和は変えない）でネットワークが出す q。実測は、d が 100 点の区間に入る
 * データの組での率（組が無い区間は NaN）。段差は q(0) − q(−100)。最終の点が同点なら起家に近い席が上になるので、0 での段差の
 * 一部は実際にある段差。
 *
 * @param rows 点差が ±500 点以内の、隣り合う 2 席の組の数（実測に使った数）
 * @param predicted 予測の段差の平均（抜き出した組での平均）
 * @param predictedSe その標準誤差（半荘をクラスタとする）
 * @param realized 実測の段差
 * @param realizedSe その標準誤差（2 つの区間の共分散を含む）
 * @param predictedCurve 予測の曲線（d = −500〜+500 点の 100 点ごと）
 * @param realizedCurve 実測の曲線（予測と同じ並びの、100 点の区間ごと。組が無い区間は NaN）
 */
public record TieJump(
    long rows,
    double predicted,
    double predictedSe,
    double realized,
    double realizedSe,
    double[] predictedCurve,
    double[] realizedCurve) {

  public double difference() {
    return predicted - realized;
  }

  public double standardError() {
    return Math.sqrt(predictedSe * predictedSe + realizedSe * realizedSe);
  }

  /** 同点のまわりの段差の基準。予測と実測の段差の差が 2 SE 以内なら true を返す。 */
  public boolean within() {
    return Math.abs(difference()) <= 2 * standardError();
  }
}
