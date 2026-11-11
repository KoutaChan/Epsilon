package com.epsilon.agent.facts;

import static org.testng.Assert.assertEquals;

import com.epsilon.mahjong.tile.Tiles;
import org.testng.annotations.Test;

/** {@link SeatFacts} の説明にある危険度の表を確かめる。確かめる場面は、現物、筋、字牌の見えている枚数。 */
public class DangerTest {

  @Test
  public void genbutsuIsSafestWhateverTheShape() {
    long safe = Tiles.bit(Tiles.P5) | Tiles.bit(Tiles.CHUN);
    assertEquals(SeatFacts.danger(safe, Tiles.P5, 0), 0);
    assertEquals(SeatFacts.danger(safe, Tiles.CHUN, 0), 0);
  }

  @Test
  public void honorsFollowTheSeenCount() {
    assertEquals(SeatFacts.danger(0, Tiles.TON, 4), 1);
    assertEquals(SeatFacts.danger(0, Tiles.TON, 3), 1);
    assertEquals(SeatFacts.danger(0, Tiles.HATSU, 2), 2);
    assertEquals(SeatFacts.danger(0, Tiles.HATSU, 1), 3);
    assertEquals(SeatFacts.danger(0, Tiles.PEI, 0), 4);
  }

  @Test
  public void terminalsAreSafeOnTheirOneSideSuji() {
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.M4), Tiles.M1, 0), 1);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.S6), Tiles.S9, 0), 1);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.S4), Tiles.S9, 0), 2);
    assertEquals(SeatFacts.danger(0, Tiles.P1, 0), 2);
  }

  @Test
  public void twoThreeSevenEightNeedTheirSuji() {
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.M5), Tiles.M2, 0), 2);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.P6), Tiles.P3, 0), 2);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.S4), Tiles.S7, 0), 2);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.S5), Tiles.S8, 0), 2);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.S8), Tiles.S2, 0), 4);
    assertEquals(SeatFacts.danger(0, Tiles.M7, 0), 4);
  }

  @Test
  public void middleTilesNeedBothSuji() {
    long both = Tiles.bit(Tiles.P2) | Tiles.bit(Tiles.P8);
    assertEquals(SeatFacts.danger(both, Tiles.P5, 0), 3);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.P2), Tiles.P5, 0), 4);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.P9), Tiles.P6, 0), 4);
    assertEquals(SeatFacts.danger(0, Tiles.P4, 0), 5);
  }

  /** 筋は同じ色の中だけで数える。1p の筋は 4p で、隣の色の牌ではない。 */
  @Test
  public void sujiStaysInsideTheSuit() {
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.M7), Tiles.P1, 0), 2);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.S3), Tiles.P9, 0), 2);
    assertEquals(SeatFacts.danger(Tiles.bit(Tiles.P1) | Tiles.bit(Tiles.P7), Tiles.P4, 0), 3);
  }
}
