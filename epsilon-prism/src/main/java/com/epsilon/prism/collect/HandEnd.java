package com.epsilon.prism.collect;

import java.util.List;

/**
 * 1局の終わり。半荘が終わった局は values に最終順位の効用を持ち、それ以外は次局 root を Φ で評価する。
 *
 * @param prefix 次局 root の公開系列（半荘終了なら null）
 * @param scores 精算後の点数
 * @param values 半荘終了時の4席の効用（それ以外は null）
 * @param targets この局の値を受ける枠（主経路はこの局で予約した行の打った手の枠、写しは根の記録の d の枠と写しの中の行の枠、分岐は候補の枠）
 */
record HandEnd(float[] prefix, int[] scores, double[] values, List<Slot> targets) {}
