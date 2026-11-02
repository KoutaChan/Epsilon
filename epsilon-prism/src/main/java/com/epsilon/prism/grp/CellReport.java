package com.epsilon.prism.grp;

/**
 * 1 つの Φ の 1 コホートの較正の測り（{@link CellStats#measure}）。単位は「席 × 半荘が続く局末」の残差 r = u_最終 − Φ で、標準誤差は全て
 * 半荘ごとのまとまりで取る。区分ごとの配列の添字は {@link HandGroup} の順。
 *
 * @param games 半荘の数
 * @param seatHands 席 × 半荘が続く局末の数
 * @param mse 残差の二乗の平均
 * @param dSpg 局 × 今の順位 × 隣との点差の升目 c の Σ_c (n_c/N)(m_c² − v_c)（m_c は升目の残差の平均、v_c はその半荘ごとのまとまりの分散）
 * @param dSb 局 × Φ の区間の升目の同じ量
 * @param b4 今 4 位の席の残差の平均（区分ごと）
 * @param slope 局ごとの u_最終 の Φ への傾き β（区分ごと）
 * @param slopeSe SE(β)（区分ごと。半荘が少ない区分は NaN）
 * @param slopeZ (β − 1)/SE(β)（区分ごと）
 * @param rankNll 最終順位の負の対数尤度（席局あたり）
 * @param brier 最終順位の Brier（席局あたり）
 * @param fourthEce 4 位の確率の ECE（報告だけ）
 */
public record CellReport(
    long games,
    long seatHands,
    double mse,
    double dSpg,
    double dSb,
    double[] b4,
    double[] slope,
    double[] slopeSe,
    double[] slopeZ,
    double rankNll,
    double brier,
    double fourthEce) {}
