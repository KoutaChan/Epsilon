package com.epsilon.prism.luck;

/** 帳簿の受け取り先。経路の計算器は、この席から見た向きで増分を求め、この配列に足す。 */
public interface LedgerTarget {

  /** 増分をどの席から見た向きで足すか。席を持たない受け取り先は −1 を返し、帳簿を受け取らない。 */
  int seat();

  /** 列ごとの帳簿（長さ {@link LedgerColumn#COUNT}）。帳簿を付けない収集では null。 */
  double[] ledger();
}
