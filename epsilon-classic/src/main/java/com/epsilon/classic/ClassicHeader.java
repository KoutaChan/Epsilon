package com.epsilon.classic;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * チェックポイントのヘッダー（model.json）。チェックポイントを開くときに、アーキテクチャ・ルール・入力のフィンガープリントが今のコードと
 * 合うかを照合する。
 *
 * @param utility 価値の目標に使う効用。Major は方策の中で和了したときの終局の効用を求めるのにも使うので、方策だけを使うときにも要る
 * @param inputFingerprint 入力の配置のフィンガープリント
 * @param step 学習のステップ数
 */
public record ClassicHeader(
    Architecture architecture, Variant variant, RankUtility utility, String inputFingerprint, long step) {

  public ClassicHeader {
    Objects.requireNonNull(architecture, "architecture");
    Objects.requireNonNull(variant, "variant");
    Objects.requireNonNull(utility, "utility");
    Objects.requireNonNull(inputFingerprint, "inputFingerprint");
    if (step < 0) {
      throw new IllegalArgumentException("Step must not be negative: " + step);
    }
  }
}
