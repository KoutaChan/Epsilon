package com.epsilon.core;

/**
 * 選択した行動を、方策と価値の学習にどう使うかを表す。
 *
 * <p>価値の学習はすべての役割を対象とする。方策の学習は、他家の選択を固定して自席の合法手を選び替えた場合に、応答の解決結果が変わり得る{@link #CAUSAL}だけを対象とする。
 * 選択した行動が実際に実行されたかどうかは、この分類とは別に記録する。
 */
public enum DecisionLearningRole {
  /** 複数合法手の選び替えが応答解決を変え得た。 */
  CAUSAL,
  /** 合法手が一つしかなく、実行されたが方策の学習信号にはしない。 */
  FORCED,
  /** 他家のRON・上位の鳴きにより、どの合法手を選んでも応答解決を変えられなかった。 */
  PREEMPTED;

  /** 方策損失と方策実行記録へ含める役割ならtrue。 */
  public boolean advancesActorClock() {
    return this == CAUSAL;
  }
}
