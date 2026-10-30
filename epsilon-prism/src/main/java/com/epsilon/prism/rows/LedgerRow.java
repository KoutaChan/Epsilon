package com.epsilon.prism.rows;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.learn.LedgerFit;

/** 運の精算（{@link LedgerFit}）が読む行の欄（{@link Row}）。 */
public interface LedgerRow {
  /** 半荘の ID（交差適合の偶奇と、偏りの確かめの束）。 */
  long gameId();

  /** 判断の類（係数 β_c の類）。 */
  DecisionClass decisionClass();

  /** 半荘の組（当てはめの和を組ごとに分けて足す）。 */
  Row.Group group();

  /** 合法手ごとの値 Y（測っていない手は NaN）。 */
  float[] values();

  boolean hasLedger();

  /** 帳簿の欄の数（帳簿のない行は 0）。 */
  int ledgerFeatures();

  /** 手 action の帳簿の欄 feature（帳簿のない行は NaN）。 */
  float ledger(int action, int feature);
}
