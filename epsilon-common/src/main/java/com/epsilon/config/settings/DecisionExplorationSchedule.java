package com.epsilon.config.settings;

/** FULL_SUPPORTのノード探索用の混合確率を局面ごとに調整する方式。 */
public enum DecisionExplorationSchedule {
  /** 設定されたノード探索用の混合確率を倍率1でそのまま使う。 */
  NORMAL,
  /** 直前の学習区間で校正した方策不確実性パーセンタイルに応じてノード探索用の混合確率を拡縮する。 */
  PERCENTILE_SCALE
}
