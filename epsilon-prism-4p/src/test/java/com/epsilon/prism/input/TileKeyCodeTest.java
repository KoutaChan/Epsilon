package com.epsilon.prism.input;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.WaitClass;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌キーの全ての桁の組が宣言順に並ぶことを確かめる。 */
public class TileKeyCodeTest {
  @Test
  public void everyTileCodeFollowsDigitOrder() {
    int next = FOUR_PLAYER.block(KeyBlock.TILE).firstCode();
    WaitClass[] classes = WaitClass.values();
    for (int count = 0; count <= Tiles.COPIES; count++) {
      for (int ukeire = 0; ukeire <= 1; ukeire++) {
        for (int wait = 0; wait <= classes.length; wait++) {
          for (int discarded = 0; discarded <= 1; discarded++) {
            Assert.assertEquals(
                FOUR_PLAYER.tileCode(
                    count, ukeire == 1, wait == 0 ? null : classes[wait - 1], discarded == 1),
                next++);
          }
        }
      }
    }
    Assert.assertEquals(
        next,
        FOUR_PLAYER.block(KeyBlock.TILE).firstCode() + FOUR_PLAYER.block(KeyBlock.TILE).codes());
  }
}
