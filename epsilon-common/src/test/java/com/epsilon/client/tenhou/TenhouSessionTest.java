package com.epsilon.client.tenhou;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.engine.Player;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 固定メッセージとテスト用プレイヤーを使い、モデルを読み込まずに、応答の送信とサーバーによる行動確定を検証する。 */
public class TenhouSessionTest {

  @DataProvider
  public Object[][] responsePrompts() {
    return new Object[][] {{"<E18/>", false}, {"<E18 t=\"1\"/>", true}};
  }

  @Test(dataProvider = "responsePrompts")
  public void playerReceivesOnlyTheActionsAllowedByTheServerPrompt(
      String discard, boolean canRespond) {
    AtomicInteger decisions = new AtomicInteger();
    Player player =
        (state, seat, actions) -> {
          decisions.incrementAndGet();
          Assert.assertEquals(seat, 0);
          Assert.assertEquals(state.hand(1).concealedTileCount(), 0, "他家の手牌は復元しない");
          Assert.assertTrue(
              actions.stream()
                  .allMatch(
                      action ->
                          action.type() == Action.Type.PON || action.type() == Action.Type.PASS));
          return actions.stream()
              .filter(action -> action.type() == Action.Type.PON)
              .findFirst()
              .orElseThrow();
        };
    var session = new TenhouSession(new TenhouGameController(player));
    session.handleMessage(init("0,4,8,16,17,36,40,44,72,76,80,108,109"));

    List<String> responses = session.handleMessage(discard);

    Assert.assertEquals(decisions.get(), canRespond ? 1 : 0);
    Assert.assertEquals(
        responses,
        canRespond ? List.of("{\"tag\":\"N\",\"type\":1,\"hai0\":16,\"hai1\":17}") : List.of());
  }

  @Test
  public void riichiIsSentBeforeDiscardAndTheHandWaitsForTheServerEcho() {
    Player player =
        (state, seat, actions) ->
            actions.stream()
                .filter(
                    action ->
                        action.type() == Action.Type.RIICHI_DAHAI && action.tileType() == Tile.CHUN)
                .findFirst()
                .orElseThrow();
    var controller = new TenhouGameController(player);
    var session = new TenhouSession(controller);
    session.handleMessage(init("0,4,8,36,40,44,72,76,80,96,100,104,108"));

    Assert.assertEquals(
        session.handleMessage("<T132 t=\"32\"/>"),
        List.of("{\"tag\":\"REACH\"}", "{\"tag\":\"D\",\"p\":132}"));
    Assert.assertEquals(controller.getState().hand(0).concealedTileCount(), 14);
    Assert.assertEquals(controller.getState().getScore(0), 25000);

    Assert.assertEquals(session.handleMessage("<REACH who=\"0\" step=\"1\"/>"), List.of());
    Assert.assertEquals(session.handleMessage("<D132/>"), List.of());
    Assert.assertEquals(controller.getState().hand(0).concealedTileCount(), 13);
    Assert.assertEquals(session.handleMessage("<REACH who=\"0\" step=\"2\"/>"), List.of());
    Assert.assertEquals(controller.getState().getScore(0), 24000);
  }

  @Test
  public void multipleWinsSendNextReadyOnceAndTheNextRoundResetsReadiness() {
    var session = new TenhouSession(new TenhouGameController(noDecision()));
    Assert.assertEquals(session.handleMessage("<GO/>"), List.of("{\"tag\":\"GOK\"}"));
    session.handleMessage(init("0,4,8,36,40,44,72,76,80,96,100,104,108"));
    Assert.assertEquals(
        session.handleMessage("<AGARI who=\"1\" fromWho=\"3\"/>"),
        List.of("{\"tag\":\"NEXTREADY\"}"));
    Assert.assertEquals(session.handleMessage("<AGARI who=\"2\" fromWho=\"3\"/>"), List.of());
    Assert.assertFalse(session.ended());

    session.handleMessage(init("0,4,8,36,40,44,72,76,80,96,100,104,108"));
    Assert.assertEquals(session.handleMessage("<RYUUKYOKU/>"), List.of("{\"tag\":\"NEXTREADY\"}"));
  }

  @DataProvider
  public Object[][] finalResults() {
    return new Object[][] {
      {"<AGARI who=\"0\" fromWho=\"1\" owari=\"250,0,250,0,250,0,250,0\"/>"},
      {"<RYUUKYOKU owari=\"250,0,250,0,250,0,250,0\"/>"},
      {"<OWARI/>"}
    };
  }

  @Test(dataProvider = "finalResults")
  public void aFinalResultEndsTheSessionWithoutRequestingAnotherRound(String result) {
    var session = new TenhouSession(new TenhouGameController(noDecision()));
    session.handleMessage(init("0,4,8,36,40,44,72,76,80,96,100,104,108"));
    Assert.assertEquals(session.handleMessage(result), List.of());
    Assert.assertTrue(session.ended());
  }

  private static Player noDecision() {
    return (state, seat, actions) -> {
      throw new AssertionError("対局通知だけでは行動選択しない");
    };
  }

  private static String init(String hand) {
    return "<INIT seed=\"0,0,0,0,0,124\" ten=\"250,250,250,250\" oya=\"0\" hai=\"" + hand + "\"/>";
  }
}
