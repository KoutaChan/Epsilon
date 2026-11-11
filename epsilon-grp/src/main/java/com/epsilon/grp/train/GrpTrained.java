package com.epsilon.grp.train;

import com.epsilon.grp.GrpNetwork;

/**
 * GRP を学習した結果。
 *
 * @param network 検証用の行の損失が最も小さかった重みのネットワーク。WARM なら始めの重みも候補に入る。STOPPED なら null
 * @param halfEpochs 検証用の行で損失を測った回数（半エポックの数）
 * @param stopNll 最良の重みでの、検証用の行の重み付きの同時 NLL（局末あたり）
 * @param clippedSteps 勾配全体のノルムが上限を超えたため、勾配を縮めたステップの数（ふだんは 0）
 * @param ending 学習の終わり方
 */
public record GrpTrained(
    GrpNetwork network, int halfEpochs, double stopNll, long clippedSteps, TrainingEnding ending) {}
