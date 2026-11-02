package com.epsilon.prism.grp;

/**
 * 1 コホートの対局の記録の先頭: 集めたときの Φ と挙動。
 *
 * @param cohort コホート t
 * @param phiId 使った Φ の項目の id（{@link PhiEntry#id}）
 * @param phiSha256 その GRP の checkpoint の sha（点数だけの Φ は空）
 * @param epsilonB 行動方策 b の一様な床 ε_b
 * @param exploreTemperature 主経路の探索の温度 τ
 * @param mainDecisions 主経路の NN 推論判断の数
 * @param mainNonGreedy そのうち貪欲でない手を打った判断の数
 */
public record GameLogHeader(
    int cohort,
    String phiId,
    String phiSha256,
    double epsilonB,
    double exploreTemperature,
    long mainDecisions,
    long mainNonGreedy) {}
