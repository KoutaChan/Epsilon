package com.epsilon.mahjong.action;

import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 行動を番号にして戻せることと、番号ごとの意味を確かめる。番号は系譜の記録に残っているので、変えない。 */
public class ActionTest {

  @Test
  public void everyCodeRoundTripsThroughOf() {
    for (int code = 0; code < ActionSection.SPACE; code++) {
      Action action = Action.of(code);
      Assert.assertEquals(action.code(), code);
      Assert.assertSame(action.section(), ActionSection.of(code));
      Assert.assertSame(action.type(), action.section().type());
      Assert.assertSame(action.selection(), action.section().selection());
      Assert.assertEquals(encode(action), code, action.toString());
      Assert.assertSame(Action.of(code), action);
    }
  }

  @Test
  public void discardCodesKeepTheirMeaning() {
    assertAction(0, Type.DAHAI, Tiles.M1, TileSelection.NON_AKA);
    assertAction(33, Type.DAHAI, Tiles.CHUN, TileSelection.NON_AKA);
    assertAction(34, Type.DAHAI, Tiles.M1, TileSelection.TSUMOGIRI);
    assertAction(68, Type.DAHAI, Tiles.M5, TileSelection.AKA);
    assertAction(70, Type.DAHAI, Tiles.S5, TileSelection.AKA);
    assertAction(71, Type.DAHAI, Tiles.M5, TileSelection.TSUMOGIRI_AKA);
    assertAction(294, Type.RIICHI_DAHAI, Tiles.M1, TileSelection.NON_AKA);
    assertAction(328, Type.RIICHI_DAHAI, Tiles.M1, TileSelection.TSUMOGIRI);
    assertAction(362, Type.RIICHI_DAHAI, Tiles.M5, TileSelection.AKA);
    assertAction(365, Type.RIICHI_DAHAI, Tiles.M5, TileSelection.TSUMOGIRI_AKA);
    assertAction(367, Type.RIICHI_DAHAI, Tiles.S5, TileSelection.TSUMOGIRI_AKA);
  }

  /**
   * チーの番号は 74 ＋ 色 × 21 ＋ (先頭の数 − 1) × 3 ＋ 鳴いた牌の位置（順子の中で 0 始まり）。赤 5 を含むチーは 137 ＋ 色 × 6
   * ＋ 形の番号で、形は (先頭の数, 鳴いた牌の位置) の (3,0) (3,1) (4,0) (4,2) (5,1) (5,2) の順に番号を振る。
   */
  @Test
  public void chiCodesKeepTheirMeaning() {
    assertChi(74, Tiles.M1, Tiles.M1, false);
    assertChi(81, Tiles.M3, Tiles.M4, false);
    assertChi(95, Tiles.P1, Tiles.P1, false);
    assertChi(136, Tiles.S7, Tiles.S9, false);
    assertChi(137, Tiles.M3, Tiles.M3, true);
    assertChi(138, Tiles.M3, Tiles.M4, true);
    assertChi(139, Tiles.M4, Tiles.M4, true);
    assertChi(140, Tiles.M4, Tiles.M6, true);
    assertChi(141, Tiles.M5, Tiles.M6, true);
    assertChi(142, Tiles.M5, Tiles.M7, true);
    assertChi(143, Tiles.P3, Tiles.P3, true);
    assertChi(154, Tiles.S5, Tiles.S7, true);
  }

  @Test
  public void callAndKanCodesKeepTheirMeaning() {
    assertAction(155, Type.PON, Tiles.M1, TileSelection.NON_AKA);
    assertAction(188, Type.PON, Tiles.CHUN, TileSelection.NON_AKA);
    assertAction(189, Type.PON, Tiles.M5, TileSelection.AKA);
    assertAction(191, Type.PON, Tiles.S5, TileSelection.AKA);
    assertAction(192, Type.DAIMINKAN, Tiles.M1, TileSelection.NOT_APPLICABLE);
    assertAction(226, Type.ANKAN, Tiles.M1, TileSelection.NOT_APPLICABLE);
    assertAction(260, Type.KAKAN, Tiles.M1, TileSelection.NOT_APPLICABLE);
    assertAction(293, Type.KAKAN, Tiles.CHUN, TileSelection.NOT_APPLICABLE);
  }

  @Test
  public void tilelessCodesKeepTheirMeaning() {
    Type[] types = {Type.TSUMO_AGARI, Type.RON_AGARI, Type.PASS, Type.KYUSHU_KYUHAI, Type.NUKI};
    for (int index = 0; index < types.length; index++) {
      int code = 368 + index;
      assertAction(code, types[index], -1, TileSelection.NOT_APPLICABLE);
      Assert.assertEquals(Action.code(types[index]), code);
    }
  }

  @Test
  public void encodersRejectActionsThatDoNotExist() {
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.code(Type.DAHAI, Tiles.M4, TileSelection.AKA));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Action.code(Type.DAHAI, Tiles.M1, TileSelection.NOT_APPLICABLE));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.code(Type.ANKAN, Tiles.M1, TileSelection.NON_AKA));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> Action.code(Type.PON, Tiles.KINDS, TileSelection.NON_AKA));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.code(Type.PON, -1, TileSelection.NON_AKA));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.code(Type.CHI, Tiles.M1, TileSelection.NON_AKA));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.code(Type.ANKAN));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.chi(Tiles.M8, Tiles.M8, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.chi(Tiles.TON, Tiles.TON, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.chi(Tiles.M1, Tiles.M4, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.chi(Tiles.M1, Tiles.M2, true));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.chi(Tiles.M3, Tiles.M5, true));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.of(-1));
    Assert.assertThrows(IllegalArgumentException.class, () -> Action.of(ActionSection.SPACE));
  }

  @Test
  public void discardSelectionsCombineRedAndTsumogiri() {
    Assert.assertSame(TileSelection.discard(false, false), TileSelection.NON_AKA);
    Assert.assertSame(TileSelection.discard(true, false), TileSelection.AKA);
    Assert.assertSame(TileSelection.discard(false, true), TileSelection.TSUMOGIRI);
    Assert.assertSame(TileSelection.discard(true, true), TileSelection.TSUMOGIRI_AKA);
    for (boolean red : new boolean[] {false, true}) {
      for (boolean tsumogiri : new boolean[] {false, true}) {
        TileSelection selection = TileSelection.discard(red, tsumogiri);
        Assert.assertEquals(selection.red(), red);
        Assert.assertEquals(selection.tsumogiri(), tsumogiri);
      }
    }
    Assert.assertFalse(TileSelection.NOT_APPLICABLE.red());
    Assert.assertFalse(TileSelection.NOT_APPLICABLE.tsumogiri());
  }

  private static int encode(Action action) {
    return switch (action.type()) {
      case CHI -> Action.chi(action.base(), action.tile(), action.selection().red());
      case TSUMO_AGARI, RON_AGARI, PASS, KYUSHU_KYUHAI, NUKI -> Action.code(action.type());
      default -> Action.code(action.type(), action.tile(), action.selection());
    };
  }

  private static void assertAction(int code, Type type, int tile, TileSelection selection) {
    Action action = Action.of(code);
    Assert.assertSame(action.type(), type, "code " + code);
    Assert.assertEquals(action.tile(), tile, "code " + code);
    Assert.assertSame(action.selection(), selection, "code " + code);
    if (tile >= 0) {
      Assert.assertEquals(Action.code(type, tile, selection), code);
    }
  }

  private static void assertChi(int code, int base, int called, boolean red) {
    Action action = Action.of(code);
    Assert.assertSame(action.type(), Type.CHI, "code " + code);
    Assert.assertEquals(action.base(), base, "code " + code);
    Assert.assertEquals(action.tile(), called, "code " + code);
    Assert.assertEquals(action.selection(), red ? TileSelection.AKA : TileSelection.NON_AKA, "code " + code);
    Assert.assertEquals(Action.chi(base, called, red), code);
  }
}
