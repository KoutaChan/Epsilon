package com.epsilon.grp.train;

import com.epsilon.grp.OutcomeNetwork;

/**
 * OutcomeNetwork を学習した結果。
 *
 * @param network 学習したネットワーク。重みは、早期終了用の検証の行で重み付きの周辺 NLL が最も小さかったもの。ウォームスタート
 *     （WARM）なら始めの重みも候補に入る。STOPPED で終わったときは null
 * @param halfEpochs 検証の行で損失を測った回数（半エポックの数）
 * @param stopNll 選んだ重みでの、検証の行の NLL（局末 1 つあたり）
 * @param clippedSteps 勾配全体のノルムが上限を超えたため、勾配を縮めたステップの数（ふだんは 0）
 * @param ending 学習の終わり方
 */
public record OutcomeTrained(
    OutcomeNetwork network, int halfEpochs, double stopNll, long clippedSteps, TrainingEnding ending) {}
