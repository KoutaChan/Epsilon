package com.epsilon.prism.collect;

import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.luck.LuckBatch;
import java.util.List;

/**
 * 1局の終わり。半荘が終わった局は values に最終順位の効用を持ち、それ以外は次局 root を Φ で評価する。
 *
 * @param prefix 次局 root の公開系列（半荘終了なら null）
 * @param scores 精算後の点数
 * @param values 半荘終了時の4席の効用（それ以外は null）
 * @param targets この局の値を受ける枠（主経路はこの局で予約した行の打った手の枠、写しは根の記録の d の枠と写しの中の行の枠、分岐は候補の枠。
 *     主経路で枠がなく運の網の標本だけを持つ局は空）
 * @param facts この経路の局末の事実（行の局末の 3 語の元。行を書かない収集と、枠のない局は null）
 * @param luck この経路のこの局の運の網の標本（局末の値を Y にして書く。標本を集めない収集と、標本を引かなかった局は null）
 */
record HandEnd(
    float[] prefix,
    int[] scores,
    double[] values,
    List<Slot> targets,
    HandEndFacts facts,
    LuckBatch luck) {}
