package com.epsilon.client.tenhou;

import com.epsilon.core.Action;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.google.gson.JsonParser;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** サーバーからの受信時に非公開情報を漏らさず、選択した物理牌を正しく送信することを固定入力で検証する。 */
public class TenhouProtocolTest {
  @DataProvider
  public Object[][] draws() {
    return new Object[][] {
      {"<T52 t=\"64\"/>", 0, 52, true},
      {"{\"tag\":\"T52\",\"t\":64}", 0, 52, true},
      {"<U52 t=\"64\"/>", 1, -1, false},
      {"{\"tag\":\"V52\",\"t\":64}", 2, -1, false},
      {"<W/>", 3, -1, false}
    };
  }

  @Test(dataProvider = "draws")
  public void onlyOwnDrawRevealsTheTileAndGrantsADrawPrompt(
      String message, int player, int physicalTile, boolean ownDraw) {
    var draw = (TenhouEvent.Draw) TenhouMessageParser.parse(message);
    Assert.assertEquals(draw.player(), player);
    Assert.assertEquals(draw.physicalTileId(), physicalTile);
    Assert.assertEquals(draw.prompt().canDahai(), ownDraw);
    Assert.assertEquals(draw.prompt().canKyushu(), ownDraw);
    Assert.assertFalse(draw.prompt().canRon());
  }

  @Test
  public void kyushuUsesTheDrawPromptAndSendsTheAbortiveDrawCommand() {
    var response = (TenhouEvent.Dahai) TenhouMessageParser.parse("<E16 t=\"64\"/>");
    Assert.assertFalse(response.prompt().canKyushu());
    String[] encoded =
        TenhouActionEncoder.encodeAction(
            Action.kyushuKyuhai(), new Hand(), TurnEvent.None.INSTANCE);
    Assert.assertEquals(encoded.length, 1);
    var message = JsonParser.parseString(encoded[0]).getAsJsonObject();
    Assert.assertEquals(message.get("tag").getAsString(), "N");
    Assert.assertEquals(message.get("type").getAsInt(), 9);
  }

  @Test
  public void redAndNormalDiscardsKeepPhysicalIdentityAndRiichiOrder() {
    Hand hand = new Hand();
    hand.addPhysicalTile(16);
    hand.addPhysicalTile(17);
    hand.addPhysicalTile(19);
    var draw = new TurnEvent.Draw(0, 19, TurnEvent.DrawSource.WALL, false);
    String[] tsumogiri =
        TenhouActionEncoder.encodeAction(Action.dahai(Tile.M5, false, true), hand, draw);
    String[] tedashi = TenhouActionEncoder.encodeAction(Action.dahai(Tile.M5, false), hand, draw);
    String[] riichi =
        TenhouActionEncoder.encodeAction(Action.riichiDahai(Tile.M5, true), hand, draw);
    Assert.assertEquals(
        JsonParser.parseString(tsumogiri[0]).getAsJsonObject().get("p").getAsInt(), 19);
    Assert.assertEquals(
        JsonParser.parseString(tedashi[0]).getAsJsonObject().get("p").getAsInt(), 17);
    Assert.assertEquals(riichi.length, 2);
    Assert.assertEquals(
        JsonParser.parseString(riichi[0]).getAsJsonObject().get("tag").getAsString(), "REACH");
    Assert.assertEquals(
        JsonParser.parseString(riichi[1]).getAsJsonObject().get("p").getAsInt(), 16);
    Assert.assertEquals(hand.concealedTileCount(), 3, "送信encodeは確認前の手牌を変更しない");
  }

  @Test
  public void ponEncodingConsumesTheSelectedRedCombination() {
    Hand hand = new Hand();
    hand.addPhysicalTile(16);
    hand.addPhysicalTile(17);
    hand.addPhysicalTile(18);
    var red =
        JsonParser.parseString(
                TenhouActionEncoder.encodeAction(
                    Action.pon(Tile.M5, true), hand, TurnEvent.None.INSTANCE)[0])
            .getAsJsonObject();
    var normal =
        JsonParser.parseString(
                TenhouActionEncoder.encodeAction(
                    Action.pon(Tile.M5, false), hand, TurnEvent.None.INSTANCE)[0])
            .getAsJsonObject();
    Assert.assertEquals(red.get("type").getAsInt(), 1);
    Assert.assertEquals(
        new int[] {red.get("hai0").getAsInt(), red.get("hai1").getAsInt()}, new int[] {16, 17});
    Assert.assertEquals(
        new int[] {normal.get("hai0").getAsInt(), normal.get("hai1").getAsInt()},
        new int[] {17, 18});
  }

  @Test
  public void lowercaseDiscardIsTsumogiriAndMalformedMeldCannotBecomeAStateEvent() {
    var discard = (TenhouEvent.Dahai) TenhouMessageParser.parse("<e52 t=\"8\"/>");
    Assert.assertEquals(discard.player(), 1);
    Assert.assertEquals(discard.physicalTileId(), 52);
    Assert.assertTrue(discard.tsumogiri());
    Assert.assertTrue(discard.prompt().canRon());
    Assert.assertTrue(TenhouMessageParser.parse("<N who=\"1\"/>") instanceof TenhouEvent.Unknown);
  }
}
