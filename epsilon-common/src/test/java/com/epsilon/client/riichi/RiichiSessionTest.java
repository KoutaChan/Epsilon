package com.epsilon.client.riichi;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 外部接続なしで、観測の適用と応答の寿命を検証する。 */
public class RiichiSessionTest {
  @DataProvider
  public Object[][] seats() {
    return new Object[][] {{0}, {1}, {2}, {3}};
  }

  @Test(dataProvider = "seats")
  public void repeatedRequestDoesNotApplyTheDrawOrSelectTwice(int seat) {
    AtomicInteger decisions = new AtomicInteger();
    var session =
        new RiichiSession(
            (state, self, legal) -> {
              Assert.assertEquals(self, seat);
              Assert.assertEquals(state.hand(self).concealedTileCount(), 14);
              decisions.incrementAndGet();
              return legal.getFirst();
            },
            false);
    JsonObject request = discard(7, seat, startRound(seat), draw(seat));
    JsonObject response = object(session.handleMessage(request, System.nanoTime()));
    Assert.assertEquals(response.get("request_id").getAsLong(), 7L);
    Assert.assertEquals(response.get("actor").getAsInt(), seat);
    Assert.assertTrue(response.get("tsumogiri").getAsBoolean());
    Assert.assertNull(session.handleMessage(request, System.nanoTime()));
    Assert.assertEquals(decisions.get(), 1);
    session.handleMessage(discard(8, seat), System.nanoTime());
    Assert.assertEquals(decisions.get(), 2);
  }

  @Test
  public void expiredRequestAppliesItsEventsButLeavesTheNextRequestToSelect() {
    AtomicInteger decisions = new AtomicInteger();
    var session =
        new RiichiSession(
            (state, seat, legal) -> {
              Assert.assertEquals(state.hand(seat).concealedTileCount(), 14);
              decisions.incrementAndGet();
              return legal.getFirst();
            },
            false);
    JsonObject expired = discard(1, 0, startRound(0), draw(0));
    expired.getAsJsonObject("time").addProperty("deadline_ms", 0);
    Assert.assertNull(session.handleMessage(expired, System.nanoTime()));
    Assert.assertEquals(decisions.get(), 0);
    Assert.assertNotNull(session.handleMessage(discard(2, 0), System.nanoTime()));
    Assert.assertEquals(decisions.get(), 1);
  }

  @DataProvider
  public Object[][] confirmationSources() {
    return new Object[][] {{true}, {false}};
  }

  @Test(dataProvider = "confirmationSources")
  public void riichiConfirmationKeepsTheChosenDiscardWithoutAnotherSelection(boolean receipt) {
    AtomicInteger decisions = new AtomicInteger();
    var session = riichiPlayer(decisions);
    JsonObject first = declaration();
    JsonObject response = object(session.handleMessage(first, System.nanoTime()));
    Assert.assertEquals(response.get("type").getAsString(), "reach");
    if (receipt) session.handleMessage(acknowledge(1, "accepted"), System.nanoTime());
    String[] events = receipt ? new String[0] : new String[] {"{\"type\":\"reach\",\"actor\":0}"};
    JsonObject followup = object(session.handleMessage(discard(2, 0, events), System.nanoTime()));
    Assert.assertEquals(followup.get("type").getAsString(), "dahai");
    Assert.assertEquals(followup.get("pai").getAsString(), "S");
    Assert.assertTrue(followup.get("tsumogiri").getAsBoolean());
    Assert.assertEquals(followup.get("request_id").getAsLong(), 2L);
    Assert.assertEquals(decisions.get(), 1);
  }

  @Test
  public void riichiFollowupBeforeConfirmationDoesNotSendThePendingDiscard() {
    var session = riichiPlayer(new AtomicInteger());
    session.handleMessage(declaration(), System.nanoTime());
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> session.handleMessage(discard(2, 0), System.nanoTime()));
  }

  @Test
  public void defaultedRonMakesTheNextDecisionTemporarilyFuriten() {
    AtomicInteger decisions = new AtomicInteger();
    var session =
        new RiichiSession(
            (state, seat, legal) -> {
              Assert.assertTrue(state.isTemporaryFuriten(seat));
              decisions.incrementAndGet();
              return Action.pass();
            },
            false);
    JsonObject request =
        request(
            1,
            0,
            "[{\"action_type\":\"Pass\",\"tile\":null,\"consume_tiles\":[]},"
                + "{\"action_type\":\"Ron\",\"tile\":113,\"consume_tiles\":[]}]",
            "[{\"type\":\"none\"},{\"type\":\"hora\"}]",
            startRound(0),
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"?\"}",
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"S\",\"tsumogiri\":true}");
    request.getAsJsonObject("time").addProperty("deadline_ms", 0);
    Assert.assertNull(session.handleMessage(request, System.nanoTime()));
    session.handleMessage(acknowledge(1, "defaulted"), System.nanoTime());
    session.handleMessage(
        request(
            2,
            0,
            "[{\"action_type\":\"Pass\",\"tile\":null,\"consume_tiles\":[]}]",
            "[{\"type\":\"none\"}]"),
        System.nanoTime());
    Assert.assertEquals(decisions.get(), 1);
  }

  @DataProvider
  public Object[][] rejectionStatuses() {
    return new Object[][] {{"rejected"}, {"unparseable"}};
  }

  @Test(dataProvider = "rejectionStatuses")
  public void rejectedResponseEndsTheSessionWithAnError(String status) {
    var session = new RiichiSession((state, seat, legal) -> legal.getFirst(), false);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> session.handleMessage(acknowledge(1, status), System.nanoTime()));
  }

  @Test
  public void validationWaitsForItsResultEvenAfterTheGameEnds() {
    var session = new RiichiSession((state, seat, legal) -> legal.getFirst(), true);
    session.handleMessage(
        object("{\"type\":\"end_game\",\"scores\":[25000,25000,25000,25000]}"), 0);
    Assert.assertFalse(session.ended());
    session.handleMessage(object("{\"type\":\"validation_result\",\"passed\":true}"), 0);
    Assert.assertTrue(session.ended());
    Assert.assertTrue(session.validationPassed());
  }

  private static RiichiSession riichiPlayer(AtomicInteger decisions) {
    return new RiichiSession(
        (state, seat, legal) -> {
          decisions.incrementAndGet();
          return legal.stream()
              .filter(
                  action ->
                      action.type() == Action.Type.RIICHI_DAHAI
                          && action.tileType() == Tile.NAN
                          && action.tileSelection().isTsumogiri())
              .findFirst()
              .orElseThrow();
        },
        false);
  }

  private static JsonObject declaration() {
    return request(
        1,
        0,
        "[{\"action_type\":\"Riichi\",\"tile\":null,\"consume_tiles\":[]}]",
        "[{\"type\":\"reach\"}]",
        startRound(0),
        draw(0));
  }

  private static JsonObject discard(long id, int seat, String... events) {
    return request(
        id,
        seat,
        "[{\"action_type\":\"Discard\",\"tile\":113,\"consume_tiles\":[]}]",
        "[{\"type\":\"dahai\",\"pai\":\"S\"}]",
        events);
  }

  private static JsonObject request(
      long id, int seat, String rawActions, String wireActions, String... events) {
    var observation = new JsonObject();
    observation.addProperty("player_id", seat);
    observation.addProperty("drawn_tile", 113);
    observation.add("_legal_actions", JsonParser.parseString(rawActions));
    var encodedEvents = new JsonArray();
    for (String event : events) encodedEvents.add(event);
    observation.add("events", encodedEvents);
    JsonObject request = object("{\"type\":\"request_action\",\"time\":{\"deadline_ms\":18000}}");
    request.addProperty("request_id", id);
    request.add("possible_actions", JsonParser.parseString(wireActions));
    request.addProperty(
        "observation",
        Base64.getEncoder()
            .encodeToString(observation.toString().getBytes(StandardCharsets.UTF_8)));
    return request;
  }

  private static String startRound(int seat) {
    JsonObject event =
        object(
            "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,"
                + "\"dora_marker\":\"9m\",\"scores\":[25000,25000,25000,25000]}");
    event.addProperty("oya", seat);
    var hands = new JsonArray();
    for (int player = 0; player < 4; player++)
      hands.add(
          JsonParser.parseString(
              player == seat
                  ? "[\"1m\",\"2m\",\"3m\",\"1p\",\"2p\",\"3p\",\"1s\",\"2s\",\"3s\",\"E\",\"E\",\"E\",\"S\"]"
                  : "[]"));
    event.add("tehais", hands);
    return event.toString();
  }

  private static String draw(int seat) {
    return "{\"type\":\"tsumo\",\"actor\":" + seat + ",\"pai\":\"S\"}";
  }

  private static JsonObject acknowledge(long id, String status) {
    JsonObject message = object("{\"type\":\"action_ack\"}");
    message.addProperty("request_id", id);
    message.addProperty("status", status);
    return message;
  }

  private static JsonObject object(String json) {
    return JsonParser.parseString(json).getAsJsonObject();
  }
}
