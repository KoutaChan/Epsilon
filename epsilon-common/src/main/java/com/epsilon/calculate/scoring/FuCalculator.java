package com.epsilon.calculate.scoring;

import com.epsilon.core.Tile;

/** 一つの分解・和了牌配置に対する純粋な符計算。 */
final class FuCalculator {
  private FuCalculator() {}

  static int calculate(
      WinningHandFacts hand,
      DecompositionFacts candidate,
      WinMethod method,
      WaitShape machiType,
      int agariTile,
      long yakuMask) {
    if (YakuBits.contains(yakuMask, ScoringYaku.PINFU)) {
      return method == WinMethod.TSUMO ? 20 : 30;
    }

    // rawFu = 20 + 雀頭符 + 面子符 + 待ち符 + ツモ符または門前ロン符
    int rawFu = 20 + candidate.pairFu + candidate.ankoFu + hand.declaredMeldFu + machiType.waitFu();
    if (method == WinMethod.TSUMO) rawFu += 2;
    else if (hand.menzen) rawFu += 10;

    // ロン牌で暗刻が完成した場合、その刻子だけは明刻として符を計算する。
    if (method == WinMethod.RON
        && machiType.opensTripletOnRon()
        && candidate.hasTriplet(agariTile)) {
      rawFu -= Tile.isTerminalOrHonor(agariTile) ? 4 : 2;
    }

    // roundedFu = max(30, ceil(rawFu / 10) * 10)
    return Math.max(30, (rawFu + 9) / 10 * 10);
  }
}
