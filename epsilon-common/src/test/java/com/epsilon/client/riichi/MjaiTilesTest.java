package com.epsilon.client.riichi;

import com.epsilon.core.Hand;
import com.google.gson.JsonArray;
import com.google.gson.JsonParser;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** MJAIの牌表記と、実手牌から消費する実牌の対応を検証する。 */
public class MjaiTilesTest {
  @Test
  public void threeNormalFivesAndOneRedFiveReceiveDistinctCopies() {
    int[] tiles = MjaiTiles.initialHand(array("[\"5m\",\"5mr\",\"5m\",\"5m\"]"));
    Assert.assertEquals(tiles, new int[] {17, 16, 18, 19});
    for (int index = 0; index < tiles.length; index++)
      Assert.assertEquals(MjaiTiles.format(tiles[index]), index == 1 ? "5mr" : "5m");
  }

  @Test
  public void incomingNormalTileFillsAnUnusedCopyWithoutTakingTheRedCopy() {
    Hand hand = new Hand();
    hand.addPhysicalTile(17);
    hand.addPhysicalTile(19);
    Assert.assertEquals(MjaiTiles.nextPhysicalTile(hand, "5m"), 18);
    Assert.assertEquals(MjaiTiles.nextPhysicalTile(hand, "5mr"), 16);
    hand.addPhysicalTile(18);
    Assert.expectThrows(
        IllegalArgumentException.class, () -> MjaiTiles.nextPhysicalTile(hand, "5m"));
  }

  @Test
  public void consumedTilesUseActualHeldCopiesAndPreserveMultiplicity() {
    Hand hand = new Hand();
    for (int tile : new int[] {16, 17, 19}) hand.addPhysicalTile(tile);
    Assert.assertEquals(
        MjaiTiles.consumed(hand, array("[\"5m\",\"5mr\",\"5m\"]")), new int[] {17, 16, 19});
    Assert.assertTrue(
        MjaiTiles.sameConsumed(array("[19,16,17]"), array("[\"5mr\",\"5m\",\"5m\"]")));
    Assert.assertFalse(MjaiTiles.sameConsumed(array("[17,19]"), array("[\"5mr\",\"5m\"]")));
    Assert.assertFalse(MjaiTiles.sameConsumed(array("[0,1,4]"), array("[\"1m\",\"2m\",\"2m\"]")));
    Assert.assertEquals(hand.concealedTileCount(), 3);
  }

  @DataProvider
  public Object[][] invalidTiles() {
    return new Object[][] {{"0m"}, {"4mr"}, {"5mx"}, {"10m"}, {"1x"}};
  }

  @Test(dataProvider = "invalidTiles")
  public void unsupportedTileNotationDoesNotProduceAPhysicalTile(String tile) {
    Assert.expectThrows(IllegalArgumentException.class, () -> MjaiTiles.parse(tile));
  }

  @Test
  public void hiddenDrawIsRepresentedWithoutAllowingAMaskedOwnHand() {
    Assert.assertEquals(MjaiTiles.parse("?"), -1);
    Assert.expectThrows(
        IllegalArgumentException.class, () -> MjaiTiles.initialHand(array("[\"1m\",\"?\"]")));
  }

  private static JsonArray array(String json) {
    return JsonParser.parseString(json).getAsJsonArray();
  }
}
