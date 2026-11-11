package com.epsilon.prism.luck;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.rows.Row;

/**
 * 帳簿の係数の当てはめと補正が、1 行から読む値。コホートの行（{@link #of}）と、コホートを書きながら和を作るときにシャードごとに貯めた行を、
 * 同じ形で読む。
 */
interface FitRow {

  /** 半荘の ID。交差適合で行を偶奇に分けるときと、偏りを確かめるときに半荘ごとにまとめるのに使う。 */
  long gameId();

  /** 判断のクラス。係数 β_c はクラスごとに持つ。 */
  DecisionClass decisionClass();

  /** 合法手ごとの値 Y（測っていない手は NaN）。 */
  float[] values();

  /** 帳簿の列の数（帳簿の無い行は 0）。 */
  int ledgerFeatures();

  /** 手 {@code action} の、帳簿の列 {@code column} の値。 */
  float ledger(int action, int column);

  /** コホートの行を読む。 */
  static FitRow of(Row row) {
    return new Cohort(row);
  }

  /** コホートの行。 */
  record Cohort(Row row) implements FitRow {
    @Override
    public long gameId() {
      return row.gameId();
    }

    @Override
    public DecisionClass decisionClass() {
      return row.decisionClass();
    }

    @Override
    public float[] values() {
      return row.values();
    }

    @Override
    public int ledgerFeatures() {
      return row.ledgerFeatures();
    }

    @Override
    public float ledger(int action, int column) {
      return row.ledger(action, column);
    }
  }
}
