package com.epsilon.engine;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 通常形・特殊形・副露手のテンパイ、捨て牌や同巡のフリテン、役なしのロン禁止を検証する。 */
public class HandRulesTest {
  @DataProvider(name = "shapes")
  public Object[][] shapes() {
    return new Object[][] {
      {"123456789m123p44p", 0, -1},
      {"111222333444m55m", 0, -1},
      {"789m33567789p234s", 0, -1},
      {"123456789m5523p", 0, 0},
      {"11335577m1133p55s", 0, -1},
      {"1133557799m11p3p", 0, 0},
      {"19m19p19s1234567z", 0, 0},
      {"19m19p19s11234567z", 0, -1},
      {"12345678m11p", 1, 0}
    };
  }

  @Test(dataProvider = "shapes")
  public void recognizesStandardSpecialAndOpenHandReadiness(String hand, int melds, int expected) {
    Assert.assertEquals(
        new HandShapeAnalyzer().calculateMinimum(MahjongFixtures.counts(hand), melds), expected);
  }

  @Test
  public void discardFuritenCoversEveryWinningTile() {
    GameState state = MahjongFixtures.state();
    MahjongFixtures.hand(state, 1, "123m123p123s11z55z");
    ActionGenerator generator = new ActionGenerator();
    var east = new TurnEvent.Discard(0, Tile.TON, false);
    Assert.assertTrue(generator.ronStatus(state, 1, east).canRon());
    state.commitDiscard(1, Tile.HAKU, false);
    Assert.assertFalse(generator.ronStatus(state, 1, east).canRon());
  }

  @Test(dataProvider = "riichiStates")
  public void ownDiscardReleasesTemporaryFuritenOnlyBeforeRiichi(boolean riichi) {
    GameState state = MahjongFixtures.state();
    MahjongFixtures.hand(state, 1, "123m123p123s11z55z");
    state.setRiichi(1, riichi);
    state.enterTemporaryFuriten(1);
    state.recordDraw(1, Tile.M1 * 4, TurnEvent.DrawSource.WALL, false);
    Assert.assertTrue(state.isTemporaryFuriten(1));
    state.commitDiscard(1, Tile.M1, false);
    Assert.assertEquals(state.isTemporaryFuriten(1), riichi);
  }

  @DataProvider(name = "riichiStates")
  public Object[][] riichiStates() {
    return new Object[][] {{false}, {true}};
  }

  @Test
  public void completeHandWithoutYakuCannotRon() {
    GameState state = MahjongFixtures.state();
    MahjongFixtures.hand(state, 1, "123m456m789p23s55z");
    Assert.assertEquals(
        new ActionGenerator().ronStatus(state, 1, new TurnEvent.Discard(0, Tile.S1, false)),
        WinLegality.RonStatus.NO_YAKU);
  }
}
