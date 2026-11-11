package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerTarget;
import com.epsilon.prism.luck.UraAudit;

/**
 * 局末の値を待つ記録と、そこへ値を渡す経路の組。手を 1 つ打った経路の局末の値を、記録へ渡す。運の精算の帳簿は、この経路の進行だけが
 * 書き、局末の値と一緒にドライバへ渡す。
 *
 * @param target 局末の値を待つ記録
 * @param action この経路で打った手
 * @param ledger 運の精算の帳簿（長さ {@link LedgerColumn#COUNT}）。帳簿を付けない収集では null
 */
record Slot(Pending target, int action, double[] ledger) implements LedgerTarget, UraAudit.Row {

  /** 帳簿を足す側の席で、裏ドラの監査が値を数える席。記録した判断の席と同じ。 */
  @Override
  public int seat() {
    return target.seat();
  }

  /** 裏ドラの監査で区分を分けるクラス。記録した判断のクラスと同じ。 */
  @Override
  public DecisionClass decisionClass() {
    return target.decisionClass();
  }
}
