package com.epsilon.grp.train;

/**
 * 学習の 1 ステップの結果。
 *
 * @param loss 更新する前の重みでの、バッチの損失
 * @param clipped 勾配全体のノルムで勾配を縮めたか
 */
public record TrainingStep(double loss, boolean clipped) {}
