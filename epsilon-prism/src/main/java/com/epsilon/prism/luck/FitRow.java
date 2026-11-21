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

  /** 判断時の局と自席の順位の区分。区分ごとに当てはめるときだけ読む。 */
  int cell();

  /** 判断時の局面。帳簿の補正率を求めるときだけ読む。 */
  LedgerCell.Standing standing();

  /** 合法手ごとの値 Y（測っていない手は NaN）。 */
  float[] values();

  /** 帳簿の列の数（帳簿の無い行は 0）。 */
  int ledgerFeatures();

  /** 手 {@code action} の、帳簿の列 {@code column} の値。 */
  float ledger(int action, int column);

  /** {@link #values} の中で、その経路で実際に打った手の位置（打った手を持たない行では −1）。帳簿の補正率の監査だけで読む。 */
  int played();

  /** 行の種類。帳簿の補正率の監査だけで読む。 */
  Row.Source source();

  /**
   * コホートの行を読む。
   *
   * @param cells 行の区分と局面を読む区分（区分も局面も読まないときは null）
   */
  static FitRow of(Row row, LedgerCell cells) {
    return new Cohort(row, cells);
  }

  /** コホートの行。 */
  record Cohort(Row row, LedgerCell cells) implements FitRow {
    @Override
    public long gameId() {
      return row.gameId();
    }

    @Override
    public DecisionClass decisionClass() {
      return row.decisionClass();
    }

    @Override
    public int cell() {
      return cells.of(row.observation(), 0);
    }

    @Override
    public LedgerCell.Standing standing() {
      return cells.standing(row.observation(), 0);
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

    @Override
    public int played() {
      return row.played();
    }

    @Override
    public Row.Source source() {
      return row.source();
    }
  }
}
