package com.epsilon.client.tenhou;

import com.epsilon.core.Action;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.Player;
import com.google.gson.JsonObject;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 自摸時の槓は手牌から判定し、大明槓だけはサーバーの応答マスクに従う。 */
public class TenhouKanTest {

  @DataProvider
  public Object[][] selfKanPrompts() {
    return new Object[][] {{false, false}, {true, false}, {false, true}, {true, true}};
  }

  @Test(dataProvider = "selfKanPrompts")
  public void kakanUsesTheFourthPhysicalTileWithoutAServerKanBit(boolean json, boolean riichiBit) {
    // 赤5mを後から加えるポンと、赤5mを鳴いて通常5mを後から加えるポン。
    boolean addedRed = !json;
    int addedTile = addedRed ? 16 : 19;
    int ponCode = addedRed ? 6154 : 6249;
    int kakanCode = addedRed ? 6162 : 6257;
    AtomicInteger decisions = new AtomicInteger();
    Player player =
        (state, seat, actions) ->
            switch (decisions.getAndIncrement()) {
              case 0 -> select(actions, Action.Type.PON, Tile.M5);
              case 1 -> select(actions, Action.Type.DAHAI, Tile.M1);
              case 2 -> select(actions, Action.Type.KAKAN, Tile.M5);
              case 3 -> {
                Assert.assertTrue(((TurnEvent.Draw) state.getTurnEvent()).isRinshanDraw());
                yield select(actions, Action.Type.DAHAI, Tile.CHUN);
              }
              default -> throw new AssertionError("予期しない判断要求");
            };
    var controller = new TenhouGameController(player);
    var session = new TenhouSession(controller);
    session.handleMessage(
        init(json, "0,4,8," + (addedRed ? "18,19" : "17,18") + ",36,40,44,72,76,80,108,109"));
    Assert.assertEquals(
        session.handleMessage(message(json, addedRed ? "F17" : "E16", "t", "1")),
        List.of(
            addedRed
                ? "{\"tag\":\"N\",\"type\":1,\"hai0\":18,\"hai1\":19}"
                : "{\"tag\":\"N\",\"type\":1,\"hai0\":17,\"hai1\":18}"));
    Assert.assertEquals(
        session.handleMessage(message(json, "N", "who", "0", "m", Integer.toString(ponCode))),
        List.of("{\"tag\":\"D\",\"p\":0}"));
    session.handleMessage(message(json, "D0"));

    Assert.assertEquals(
        session.handleMessage(draw(json, addedTile, riichiBit)),
        List.of("{\"tag\":\"N\",\"type\":5,\"hai\":" + addedTile + "}"));
    Assert.assertEquals(controller.getState().hand(0).meld(0).type(), Meld.Type.PON);
    Assert.assertTrue(controller.getState().hand(0).containsPhysicalTile(addedTile));
    Assert.assertEquals(
        session.handleMessage(message(json, "N", "who", "0", "m", Integer.toString(kakanCode))),
        List.of());
    Assert.assertEquals(controller.getState().hand(0).meld(0).type(), Meld.Type.KAKAN);
    Assert.assertEquals(controller.getState().hand(0).meld(0).addedTileIsAka(), addedRed);
    Assert.assertFalse(controller.getState().hand(0).containsPhysicalTile(addedTile));
    Assert.assertEquals(controller.getState().hand(0).concealedTileCount(), 10);
    session.handleMessage(message(json, "DORA", "hai", "128"));
    Assert.assertEquals(
        session.handleMessage(message(json, "T132")), List.of("{\"tag\":\"D\",\"p\":132}"));
    Assert.assertEquals(decisions.get(), 4);
  }

  @Test(dataProvider = "selfKanPrompts")
  public void ankanWaitsForItsEchoThenAcceptsARinshanDraw(boolean json, boolean riichiBit) {
    AtomicInteger decisions = new AtomicInteger();
    Player player =
        (state, seat, actions) -> {
          if (decisions.getAndIncrement() == 0) {
            return select(actions, Action.Type.ANKAN, Tile.P2);
          }
          Assert.assertTrue(((TurnEvent.Draw) state.getTurnEvent()).isRinshanDraw());
          return select(actions, Action.Type.DAHAI, Tile.CHUN);
        };
    var controller = new TenhouGameController(player);
    var session = new TenhouSession(controller);
    session.handleMessage(init(json, "0,4,8,36,40,41,42,72,76,80,108,109,112"));

    Assert.assertEquals(
        session.handleMessage(draw(json, 43, riichiBit)),
        List.of("{\"tag\":\"N\",\"type\":4,\"hai\":40}"));
    Assert.assertEquals(controller.getState().hand(0).meldCount(), 0);
    Assert.assertEquals(controller.getState().hand(0).concealedTileCount(), 14);
    Assert.assertEquals(
        session.handleMessage(message(json, "N", "who", "0", "m", "10240")), List.of());
    Assert.assertEquals(controller.getState().hand(0).meld(0).type(), Meld.Type.ANKAN);
    Assert.assertEquals(controller.getState().hand(0).count(Tile.P2), 0);
    Assert.assertEquals(controller.getState().hand(0).concealedTileCount(), 10);
    session.handleMessage(message(json, "DORA", "hai", "128"));
    Assert.assertEquals(
        session.handleMessage(message(json, "T132")), List.of("{\"tag\":\"D\",\"p\":132}"));
    Assert.assertEquals(decisions.get(), 2);
  }

  @DataProvider
  public Object[][] responseKanPrompts() {
    return new Object[][] {{false, 0}, {true, 0}, {false, 1}, {true, 1}, {false, 3}, {true, 3}};
  }

  @Test(dataProvider = "responseKanPrompts")
  public void daiminkanStillRequiresTheServerKanBit(boolean json, int mask) {
    AtomicInteger decisions = new AtomicInteger();
    Player player =
        (state, seat, actions) -> {
          decisions.incrementAndGet();
          Assert.assertEquals(
              actions.stream().anyMatch(action -> action.type() == Action.Type.DAIMINKAN),
              mask == 3);
          return actions.stream()
              .filter(
                  action -> action.type() == (mask == 3 ? Action.Type.DAIMINKAN : Action.Type.PASS))
              .findFirst()
              .orElseThrow();
        };
    var session = new TenhouSession(new TenhouGameController(player));
    session.handleMessage(init(json, "0,4,8,36,40,44,72,76,80,108,124,125,127"));

    List<String> responses =
        session.handleMessage(
            mask == 0 ? message(json, "E126") : message(json, "E126", "t", Integer.toString(mask)));

    Assert.assertEquals(decisions.get(), mask == 0 ? 0 : 1);
    Assert.assertEquals(
        responses,
        switch (mask) {
          case 0 -> List.of();
          case 1 -> List.of("{\"tag\":\"N\"}");
          case 3 -> List.of("{\"tag\":\"N\",\"type\":2}");
          default -> throw new AssertionError(mask);
        });
  }

  @Test
  public void lastWallDrawStillExcludesLocallyIllegalKan() {
    Player player =
        (state, seat, actions) -> {
          Assert.assertEquals(state.remainingWallTiles(), 0);
          Assert.assertFalse(
              actions.stream()
                  .anyMatch(
                      action ->
                          action.type() == Action.Type.ANKAN
                              || action.type() == Action.Type.KAKAN));
          return actions.stream()
              .filter(action -> action.type() == Action.Type.DAHAI && action.isTsumogiri())
              .findFirst()
              .orElseThrow();
        };
    var controller = new TenhouGameController(player);
    var session = new TenhouSession(controller);
    session.handleMessage(init(false, "0,4,8,36,40,41,42,72,76,80,108,109,112"));
    while (controller.getState().remainingWallTiles() > 1) {
      controller.getState().consumeReconstructionLiveWallDraw();
    }

    Assert.assertEquals(session.handleMessage("<T43/>"), List.of("{\"tag\":\"D\",\"p\":43}"));
  }

  private static Action select(List<Action> actions, Action.Type type, int tileType) {
    return actions.stream()
        .filter(action -> action.type() == type && action.tileType() == tileType)
        .findFirst()
        .orElseThrow();
  }

  private static String draw(boolean json, int physicalTileId, boolean riichiBit) {
    return riichiBit
        ? message(json, "T" + physicalTileId, "t", "32")
        : message(json, "T" + physicalTileId);
  }

  private static String init(boolean json, String hand) {
    return message(
        json, "INIT", "seed", "0,0,0,0,0,116", "ten", "250,250,250,250", "oya", "0", "hai", hand);
  }

  private static String message(boolean json, String tag, String... fields) {
    JsonObject object = new JsonObject();
    object.addProperty("tag", tag);
    StringBuilder xml = new StringBuilder("<").append(tag);
    for (int i = 0; i < fields.length; i += 2) {
      object.addProperty(fields[i], fields[i + 1]);
      xml.append(' ').append(fields[i]).append("=\"").append(fields[i + 1]).append('"');
    }
    return json ? object.toString() : xml.append("/>").toString();
  }
}
