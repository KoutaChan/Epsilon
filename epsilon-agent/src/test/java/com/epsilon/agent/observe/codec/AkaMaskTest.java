package com.epsilon.agent.observe.codec;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.mahjong.tile.Tiles;
import org.testng.annotations.Test;

/**
 * 色ごとの赤のビットと、赤の 5 の種類の集合を相互に変換できることを確かめる。ビットは萬・筒・索の順に並び、観測のフィールドと
 * 運の記録はこの値を書く。
 */
public class AkaMaskTest {

  @Test
  public void bitsFollowManPinSou() {
    assertEquals(AkaMask.redKinds(0b001), Tiles.bit(Tiles.M5));
    assertEquals(AkaMask.redKinds(0b010), Tiles.bit(Tiles.P5));
    assertEquals(AkaMask.redKinds(0b100), Tiles.bit(Tiles.S5));
    assertEquals(AkaMask.redKinds(AkaMask.ALL), Tiles.FIVES);
    assertEquals(AkaMask.suitMask(Tiles.FIVES), AkaMask.ALL);
    assertEquals(AkaMask.suitMask(0), 0);
  }

  @Test
  public void everyMaskAndEverySetOfFivesRoundTrip() {
    for (int mask = 0; mask <= AkaMask.ALL; mask++) {
      assertEquals(AkaMask.suitMask(AkaMask.redKinds(mask)), mask);
    }
    for (long kinds = Tiles.FIVES; ; kinds = (kinds - 1) & Tiles.FIVES) {
      assertEquals(AkaMask.redKinds(AkaMask.suitMask(kinds)), kinds);
      if (kinds == 0) {
        break;
      }
    }
  }

  @Test
  public void onlyFivesHaveRedCopies() {
    int manAndSou = AkaMask.suitMask(Tiles.bit(Tiles.M5) | Tiles.bit(Tiles.S5));
    assertTrue(AkaMask.has(manAndSou, Tiles.M5));
    assertFalse(AkaMask.has(manAndSou, Tiles.P5));
    assertTrue(AkaMask.has(manAndSou, Tiles.S5));
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      assertEquals(AkaMask.has(AkaMask.ALL, kind), Tiles.contains(Tiles.FIVES, kind), "kind " + kind);
    }
    assertFalse(AkaMask.has(AkaMask.ALL, -1));
  }

  @Test
  public void valuesOutsideTheRangeAreRejected() {
    expectThrows(IllegalArgumentException.class, () -> AkaMask.suitMask(Tiles.bit(Tiles.M4)));
    expectThrows(IllegalArgumentException.class, () -> AkaMask.suitMask(Tiles.FIVES | Tiles.bit(Tiles.CHUN)));
    expectThrows(IllegalArgumentException.class, () -> AkaMask.redKinds(AkaMask.ALL + 1));
    expectThrows(IllegalArgumentException.class, () -> AkaMask.redKinds(-1));
    expectThrows(IllegalArgumentException.class, () -> AkaMask.has(AkaMask.ALL + 1, Tiles.M5));
  }
}
