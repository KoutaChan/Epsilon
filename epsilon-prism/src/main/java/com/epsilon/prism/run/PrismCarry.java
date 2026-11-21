package com.epsilon.prism.run;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.luck.LedgerCalibrator;
import com.epsilon.prism.luck.LuckSettlement;

/**
 * 次の反復へ持ち越す値（反復の報告の carry）。キーが次の反復のキーと違えば、その反復は R² の持ち越しを捨て、学習の段の報告に理由を書く。
 * 帳簿の補正率は R² と別に、補正率の作り方が次の反復の設定と合うかで使うかを決める（{@link LuckSettlement}）。
 *
 * @param ledgerR2 教師に掛けた運の精算の補正の、ホールドアウトでの R²（補正を掛けなければ 0）。次の反復はこの値から K′ を決める
 * @param key 持ち越しのキー。始めの state では無い
 * @param calibrator 反復のコホートの行で当てはめた、次の反復の帳簿の補正率（補正率を求めなかった反復では無い）
 */
public record PrismCarry(
    double ledgerR2,
    @Key(doc = "持ち越しのキー（始めの state では無い）", presence = Presence.ABSENT) CarryKey key,
    @Key(doc = "次の反復の帳簿の補正率（補正率を求めなかった反復では無い）", presence = Presence.ABSENT)
        LedgerCalibrator calibrator) {

  /** 何も持ち越さない初期値。 */
  public static final PrismCarry NONE = new PrismCarry(0, null);

  public PrismCarry {
    if (!(ledgerR2 >= 0 && ledgerR2 <= 1)) {
      throw new IllegalArgumentException("The carried R² must be in [0, 1]: " + ledgerR2);
    }
  }

  public PrismCarry(double ledgerR2, CarryKey key) {
    this(ledgerR2, key, null);
  }
}
