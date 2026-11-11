package com.epsilon.agent.facts;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.fail;

import com.epsilon.agent.facts.WaitScores.Method;
import com.epsilon.agent.facts.WaitScores.WinningTile;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.HandScorer;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 待ちごとの点数を総当たりで確かめるテスト用のヘルパー。そのルールで使う牌の種類ごとに、その牌を 1 枚足した手を、{@link WaitScores}
 * の採点の条件を写した別の {@link WinContext} で直接採点する。待ちの種類は {@link WaitScores} の点数と照合し、待ちでない種類は
 * 和了にならないことを確かめる。純手牌・面子・抜いた北でその種類の牌を全て持っているときは採点しない。
 */
final class DirectScoring {

  private final HandScorer scorer = new HandScorer();
  private final WinContext direct;
  private final WinScore expected = new WinScore();
  private long scoredWins;
  private long redWins;
  private long exhaustedWaits;

  DirectScoring(Variant variant) {
    direct = new WinContext(variant);
  }

  /**
   * 待ちの種類と、赤の 5 でも和了する 5 の種類を {@link WaitScores#load} で読み込んだ後の点数を、直接採点した点数と照合する。
   * 採点の条件には、読み込んだ後の {@link WaitScores#context} を使う。
   */
  void check(WaitScores scores, long waits, long red) {
    WinContext conditions = scores.context();
    TileSet set = conditions.variant().tiles();
    int index = 0;
    for (long kinds = set.kinds(); kinds != 0; kinds &= kinds - 1) {
      int kind = Long.numberOfTrailingZeros(kinds);
      boolean wait = Tiles.contains(waits, kind);
      boolean possible = held(conditions, kind) < set.copies(kind);
      exhaustedWaits += wait && !possible ? 1 : 0;
      for (Method method : Method.values()) {
        for (WinningTile tile : WinningTile.values()) {
          expected.clear();
          boolean redTile = tile == WinningTile.RED;
          if (possible && (!redTile || Tiles.contains(red & ~conditions.redFives(), kind))) {
            score(conditions, kind, method, redTile);
          }
          if (!wait) {
            if (expected.base() != 0) {
              fail("Kind " + kind + " completes " + conditions + " outside its waits");
            }
            continue;
          }
          WinScore actual = scores.score(index, method, tile);
          String what = method + " " + tile + " on " + kind + " with " + conditions;
          assertEquals(actual.base(), expected.base(), what);
          assertEquals(actual.han(), expected.han(), what);
          assertEquals(actual.fu(), expected.fu(), what);
          assertEquals(actual.yaku(), expected.yaku(), what);
          assertEquals(actual.yakuman(), expected.yakuman(), what);
          assertEquals(actual.dora(), expected.dora(), what);
          assertEquals(actual.aka(), expected.aka(), what);
          assertEquals(actual.nuki(), expected.nuki(), what);
          if (actual.base() > 0) {
            scoredWins++;
            redWins += redTile ? 1 : 0;
          }
        }
      }
      if (wait) {
        assertEquals(scores.wait(index), kind);
        index++;
      }
    }
    assertEquals(scores.waits(), index);
  }

  long scoredWins() {
    return scoredWins;
  }

  long redWins() {
    return redWins;
  }

  /** その種類の牌を全て持っているので採点しなかった待ちの数。 */
  long exhaustedWaits() {
    return exhaustedWaits;
  }

  /** 条件の手に和了牌を足して採点し、期待する点数として残す。赤を指定したときは、和了牌を赤の 5 にする。 */
  private void score(WinContext conditions, int kind, Method method, boolean red) {
    direct.clear();
    direct.hand().copyFrom(conditions.hand());
    for (int index = 0; index < conditions.meldCount(); index++) {
      direct.addMeld(conditions.meld(index));
    }
    direct.setWinds(conditions.seatWind(), conditions.roundWind());
    for (int index = 0; index < conditions.indicatorCount(); index++) {
      direct.addIndicator(conditions.indicator(index));
    }
    direct.setRiichi(conditions.riichi(), conditions.ippatsu());
    direct.setNuki(conditions.nuki());
    direct.setRedFives(red ? conditions.redFives() | Tiles.bit(kind) : conditions.redFives());
    direct.setWin(kind, method == Method.RON ? WinContext.Source.RON : WinContext.Source.TSUMO);
    scorer.score(direct, expected);
  }

  /** 純手牌・面子・抜いた北にある、その種類の牌の枚数。 */
  private static int held(WinContext conditions, int kind) {
    int held = conditions.hand().count(kind) + (kind == Tiles.PEI ? conditions.nuki() : 0);
    for (int index = 0; index < conditions.meldCount(); index++) {
      Meld meld = conditions.meld(index);
      if (Tiles.contains(meld.kinds(), kind)) {
        held += meld.type() == Meld.Type.CHI ? 1 : meld.size();
      }
    }
    return held;
  }
}
