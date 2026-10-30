package com.epsilon.prism.policy;

import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;

/**
 * 試験の参照: 規則の事前分布の特徴を、Nano の符号化の遷移特徴（打牌後の手で計算済み）を元の尺度に戻して求める。
 * Tokenizer が記録 O から作る p0 の特徴がこれとビットで同じことを確かめる。
 *
 * <p>向聴 = 9·NORMALIZED_MIN_SHANTEN − 1、受け入れ枚数 = 136·NORMALIZED_UKEIRE_COUNT（自分から見えない枚数で数える）。遷移の位置は 0
 * （打牌の遷移は 1 つ）。
 */
public final class R0Features {
  private R0Features() {}

  public static float shanten(DecisionHostBatch host, int row, int action) {
    return 9f
            * host.transitionNumeric(
                row, action, 0, DecisionInputSchema.ActionTransitionFloat.NORMALIZED_MIN_SHANTEN)
        - 1f;
  }

  public static float ukeire(DecisionHostBatch host, int row, int action) {
    return 136f
        * host.transitionNumeric(
            row, action, 0, DecisionInputSchema.ActionTransitionFloat.NORMALIZED_UKEIRE_COUNT);
  }
}
