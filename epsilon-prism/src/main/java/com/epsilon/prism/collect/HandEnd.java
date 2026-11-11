package com.epsilon.prism.collect;

import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.luck.UraValues;
import java.util.List;

/**
 * 1 局の終わり。値は、局末を作った進行がその場で裏ドラの扱いに従って求める（{@link UraValues}）。半荘が終わった局では最終順位の効用、
 * それ以外の局では次局の卓の Φ になる。
 *
 * @param values 絶対席ごとの局末の値
 * @param targets この局の値の受け取り先。主経路では、この局で予約した行の打った手。写しでは、根の記録の d と写しの中の行。分岐では候補の手
 * @param facts この経路の局末の事実（行の局末の 3 ワードの元。行を書かない収集では null）
 */
record HandEnd(double[] values, List<Slot> targets, HandEndFacts facts) {}
