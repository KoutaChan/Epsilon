package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScore;

/** 保存用の不変な和了明細。評価器の再利用バッファを参照しない。 */
public sealed interface WinClaim permits WinClaim.Tsumo, WinClaim.Ron {
  int winner();

  int from();

  HandScore score();

  WinPayment payment();

  int honba();

  PointDelta pointDelta();

  boolean hasRiichi();

  record Tsumo(
      int winner,
      HandScore score,
      WinPayment.Tsumo payment,
      int honba,
      PointDelta pointDelta,
      boolean hasRiichi)
      implements WinClaim {
    @Override
    public int from() {
      return winner;
    }
  }

  record Ron(
      int winner,
      int from,
      HandScore score,
      WinPayment.Ron payment,
      int honba,
      PointDelta pointDelta,
      boolean hasRiichi)
      implements WinClaim {}
}
