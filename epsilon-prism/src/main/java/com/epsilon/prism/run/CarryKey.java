package com.epsilon.prism.run;

import com.epsilon.prism.grp.PhiEntry;
import com.epsilon.prism.luck.LedgerMeaning;
import com.epsilon.prism.luck.LuckSettlement;
import com.epsilon.prism.settings.PrismSettings;

/**
 * 次の反復へ持ち越した値が使えるかを決めるキー。帳簿の列と教師の意味、Φ が点数だけで決まるかのどれかが前の反復と違えば、前の反復の帳簿の
 * R² は同じ量を測っていない。
 *
 * @param ledger 帳簿の列と教師の意味を決める設定の値
 * @param scoreBasedPhi 反復の Φ が点数だけで決まる Φ か
 */
public record CarryKey(LedgerMeaning ledger, boolean scoreBasedPhi) {

  /** 反復の設定と、その反復で使った Φ からキーを作る。 */
  public static CarryKey of(PrismSettings settings, PhiEntry phi) {
    return new CarryKey(
        LuckSettlement.of(settings.noise(), settings.run().variant()).meaning(), phi.kind().scoreBased());
  }

  /**
   * このキーを前の反復のキーと比べ、前の反復から持ち越した値を捨てる理由を返す。
   *
   * @param previous 前の反復のキー（持ち越しの無い最初の状態では null）
   * @param resetsOnPhiKind Φ が点数だけで決まる Φ から GRP に替わる反復で、持ち越しを捨てるか
   * @return 捨てる理由（英語の 1 行）。捨てなければ null
   */
  public String resetReason(CarryKey previous, boolean resetsOnPhiKind) {
    if (previous == null) {
      return null;
    }
    String ledgerChange = ledger.change(previous.ledger);
    if (ledgerChange != null) {
      return ledgerChange;
    }
    if (resetsOnPhiKind && previous.scoreBasedPhi && !scoreBasedPhi) {
      return "the Φ changed from a score-based entry to a GRP";
    }
    return null;
  }
}
