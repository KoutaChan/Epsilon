package com.epsilon.ai.decision;

/** 行動選択モデルの方策から、最終的に実行する合法手を選ぶ方式。 */
public enum DecisionSelectionMode {
  /** 最大確率の合法行動を決定的に選ぶ。 */
  POLICY_GREEDY,
  /** 方策の分岐構造の各ノードに設定済みの探索確率を加え、行動を無作為に選ぶ。 */
  FULL_SUPPORT,
  /** 探索確率を加えず、モデルの方策に従って行動を無作為に選ぶ。 */
  POLICY_SAMPLE
}
