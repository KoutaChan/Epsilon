package com.epsilon.mahjong.client.riichi;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.client.Decider;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.tile.Tiles;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * RiichiLab が対局中に送るメッセージをセッションに流して、動きを確かめる。セッションは観測に載ったイベントを卓に書く。要求が
 * 来るたびに、サーバが選べるとした手だけを Decider に見せ、選んだ手をサーバのメッセージの形で返す。同じ要求には 1 回だけ答え、
 * 締め切りを過ぎた要求には答えない。
 */
public class RiichiSessionTest {

  /** 1m2m3m 1p2p3p 1s2s3s EEE S（南の単騎待ち）。 */
  private static final String WAITING =
      "\"1m\",\"2m\",\"3m\",\"1p\",\"2p\",\"3p\",\"1s\",\"2s\",\"3s\",\"E\",\"E\",\"E\",\"S\"";

  /** 引いた南（南の 4 枚のうち番号 1 の牌）。手にある南と合わせて 2 枚になる。 */
  private static final int SECOND_SOUTH = Tiles.physicalId(Tiles.NAN, 1);

  @DataProvider
  public Object[][] seats() {
    return new Object[][] {{0}, {1}, {2}, {3}};
  }

  /** request_id が同じ要求には 2 度答えず、ツモも卓に 2 度は書かない。 */
  @Test(dataProvider = "seats")
  public void aRepeatedRequestIsAnsweredOnce(int seat) {
    AtomicInteger decisions = new AtomicInteger();
    RiichiSession session =
        new RiichiSession(
            (view, slots, slot) -> {
              Assert.assertEquals(view.seat(), seat);
              Assert.assertEquals(handSize(view), 14);
              decisions.incrementAndGet();
              return 0;
            },
            false);
    String request = discard(7, seat, startRound(seat, WAITING), draw(seat, "S"));
    JsonObject reply = object(session.handle(request, System.nanoTime()));
    Assert.assertEquals(reply.get("type").getAsString(), "dahai");
    Assert.assertEquals(reply.get("request_id").getAsLong(), 7L);
    Assert.assertEquals(reply.get("actor").getAsInt(), seat);
    Assert.assertTrue(reply.get("tsumogiri").getAsBoolean());
    Assert.assertNull(session.handle(request, System.nanoTime()));
    Assert.assertEquals(decisions.get(), 1);
    Assert.assertNotNull(session.handle(discard(8, seat), System.nanoTime()));
    Assert.assertEquals(decisions.get(), 2);
  }

  /** 締め切りを過ぎた要求では、イベントを卓に書くだけで答えない。手は次の要求で選ぶ。 */
  @Test
  public void anExpiredRequestAppliesItsEventsButIsNotAnswered() {
    AtomicInteger decisions = new AtomicInteger();
    RiichiSession session =
        new RiichiSession(
            (view, slots, slot) -> {
              Assert.assertEquals(handSize(view), 14);
              decisions.incrementAndGet();
              return 0;
            },
            false);
    JsonObject expired = object(discard(1, 0, startRound(0, WAITING), draw(0, "S")));
    expired.getAsJsonObject("time").addProperty("deadline_ms", 0);
    Assert.assertNull(session.handle(expired.toString(), System.nanoTime()));
    Assert.assertEquals(decisions.get(), 0);
    Assert.assertNotNull(session.handle(discard(2, 0), System.nanoTime()));
    Assert.assertEquals(decisions.get(), 1);
  }

  @DataProvider
  public Object[][] riichiConfirmations() {
    return new Object[][] {{true}, {false}};
  }

  /**
   * 立直するときは、まず宣言を返す。宣言が受け付けられた後（受け付けの通知か reach のイベントが届いた後）の打牌の要求には、
   * 先に選んだ宣言牌を、Decider に聞かずに返す。
   */
  @Test(dataProvider = "riichiConfirmations")
  public void riichiSendsTheDeclarationThenTheChosenDiscard(boolean receipt) {
    AtomicInteger decisions = new AtomicInteger();
    RiichiSession session = riichiPlayer(decisions);
    JsonObject declaration = object(session.handle(declaration(), System.nanoTime()));
    Assert.assertEquals(declaration.get("type").getAsString(), "reach");
    Assert.assertEquals(declaration.get("request_id").getAsLong(), 1L);
    if (receipt) {
      session.handle(acknowledge(1, "accepted"), System.nanoTime());
    }
    String[] events = receipt ? new String[0] : new String[] {"{\"type\":\"reach\",\"actor\":0}"};
    JsonObject discard = object(session.handle(discard(2, 0, events), System.nanoTime()));
    Assert.assertEquals(discard.get("type").getAsString(), "dahai");
    Assert.assertEquals(discard.get("pai").getAsString(), "S");
    Assert.assertTrue(discard.get("tsumogiri").getAsBoolean());
    Assert.assertEquals(discard.get("request_id").getAsLong(), 2L);
    Assert.assertEquals(decisions.get(), 1);
  }

  @Test
  public void aRiichiDiscardRequestedBeforeTheDeclarationIsAcceptedFails() {
    RiichiSession session = riichiPlayer(new AtomicInteger());
    session.handle(declaration(), System.nanoTime());
    Assert.assertThrows(IllegalStateException.class, () -> session.handle(discard(2, 0), System.nanoTime()));
  }

  /** 締め切りでロンを見送ると同巡振聴になる。次に自分が判断するとき、席のビューは振聴を示す。 */
  @Test
  public void aRonLetPassByDefaultLeavesTheSeatFuriten() {
    List<Boolean> furiten = new ArrayList<>();
    RiichiSession session =
        new RiichiSession(
            (view, slots, slot) -> {
              furiten.add(view.furiten());
              return 0;
            },
            false);
    int nineMan = Tiles.physicalId(Tiles.M9, 0);
    Assert.assertNotNull(
        session.handle(
            request(1, 0, nineMan, discardAction(nineMan), dahai("9m"), startRound(0, WAITING), draw(0, "9m")),
            System.nanoTime()));
    String ron =
        request(
            2,
            0,
            -1,
            "[{\"action_type\":\"Pass\",\"tile\":null,\"consume_tiles\":[]},"
                + "{\"action_type\":\"Ron\",\"tile\":113,\"consume_tiles\":[]}]",
            "[{\"type\":\"none\"},{\"type\":\"hora\"}]",
            "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"9m\",\"tsumogiri\":true}",
            "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"?\"}",
            "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"S\",\"tsumogiri\":true}");
    JsonObject late = object(ron);
    late.getAsJsonObject("time").addProperty("deadline_ms", 0);
    Assert.assertNull(session.handle(late.toString(), System.nanoTime()));
    session.handle(acknowledge(2, "defaulted"), System.nanoTime());
    int threeMan = Tiles.physicalId(Tiles.M3, 1);
    Assert.assertNotNull(
        session.handle(
            request(
                3,
                0,
                threeMan,
                discardAction(threeMan),
                dahai("3m"),
                "{\"type\":\"tsumo\",\"actor\":2,\"pai\":\"?\"}",
                "{\"type\":\"dahai\",\"actor\":2,\"pai\":\"N\",\"tsumogiri\":true}",
                "{\"type\":\"tsumo\",\"actor\":3,\"pai\":\"?\"}",
                "{\"type\":\"dahai\",\"actor\":3,\"pai\":\"9p\",\"tsumogiri\":true}",
                draw(0, "3m")),
            System.nanoTime()));
    Assert.assertEquals(furiten, List.of(false, true));
  }

  /**
   * 上家が出した 2m には、エンジンはポンとチーを許す。Decider には、サーバが選べるとしたポンと見送りだけを見せる。ポンを選ぶと、
   * サーバのメッセージに target を足して返す。サーバが見送りしか認めなければ、Decider に聞かずに見送る。
   */
  @Test
  public void theServerActionsRestrictTheResponse() {
    List<List<Type>> seen = new ArrayList<>();
    RiichiSession session =
        new RiichiSession(
            (view, slots, slot) -> {
              seen.add(types(slots, slot));
              return 0;
            },
            false);
    String[] events = callScene();
    JsonObject pon =
        object(
            session.handle(
                request(
                    1,
                    1,
                    -1,
                    "[{\"action_type\":\"Pon\",\"tile\":6,\"consume_tiles\":[4,5]},"
                        + "{\"action_type\":\"Pass\",\"tile\":null,\"consume_tiles\":[]}]",
                    "[{\"type\":\"pon\",\"pai\":\"2m\",\"consumed\":[\"2m\",\"2m\"]},{\"type\":\"none\"}]",
                    events),
                System.nanoTime()));
    Assert.assertEquals(seen, List.of(List.of(Type.PON, Type.PASS)));
    Assert.assertEquals(pon.get("type").getAsString(), "pon");
    Assert.assertEquals(pon.get("actor").getAsInt(), 1);
    Assert.assertEquals(pon.get("target").getAsInt(), 0);

    RiichiSession passing =
        new RiichiSession(
            (view, slots, slot) -> {
              throw new AssertionError("Only a pass is offered");
            },
            false);
    JsonObject pass =
        object(
            passing.handle(
                request(
                    1,
                    1,
                    -1,
                    "[{\"action_type\":\"Pass\",\"tile\":null,\"consume_tiles\":[]}]",
                    "[{\"type\":\"none\"}]",
                    events),
                System.nanoTime()));
    Assert.assertEquals(pass.get("type").getAsString(), "none");
    Assert.assertEquals(pass.get("request_id").getAsLong(), 1L);
  }

  @DataProvider
  public Object[][] rejections() {
    return new Object[][] {{"rejected"}, {"unparseable"}};
  }

  @Test(dataProvider = "rejections")
  public void aRejectedActionEndsTheSession(String status) {
    RiichiSession session = new RiichiSession(first(), false);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> session.handle(acknowledge(1, status), System.nanoTime()));
  }

  @Test
  public void validationWaitsForItsResultAfterTheGameEnds() {
    RiichiSession session = new RiichiSession(first(), true);
    session.handle("{\"type\":\"start_game\",\"id\":2}", 0);
    Assert.assertTrue(session.started());
    session.handle("{\"type\":\"end_game\",\"scores\":[25000,25000,25000,25000]}", 0);
    Assert.assertTrue(session.completed());
    Assert.assertFalse(session.ended());
    session.handle("{\"type\":\"validation_result\",\"passed\":true}", 0);
    Assert.assertTrue(session.ended());
    Assert.assertTrue(session.validationPassed());
  }

  /** 席 1 の手は 2m 2m 3m 4m 1p 2p 3p 1s 2s 3s E S W。親（席 0）がツモって、2m（4 枚のうち番号 2 の牌）を手出しする。 */
  private static String[] callScene() {
    return new String[] {
      startRound(
          1, "\"2m\",\"2m\",\"3m\",\"4m\",\"1p\",\"2p\",\"3p\",\"1s\",\"2s\",\"3s\",\"E\",\"S\",\"W\"", 0),
      "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"?\"}",
      "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"2m\",\"tsumogiri\":false}"
    };
  }

  /** 引いた南をツモ切りして立直するセッション。 */
  private static RiichiSession riichiPlayer(AtomicInteger decisions) {
    return new RiichiSession(
        (view, slots, slot) -> {
          decisions.incrementAndGet();
          List<Integer> codes = codes(slots, slot);
          int riichi = codes.indexOf(Action.discard(Tiles.NAN, false, true, true));
          Assert.assertTrue(riichi >= 0, "riichi on the drawn south is offered");
          return riichi;
        },
        false);
  }

  private static String declaration() {
    return request(
        1,
        0,
        SECOND_SOUTH,
        "[{\"action_type\":\"Riichi\",\"tile\":null,\"consume_tiles\":[]}]",
        "[{\"type\":\"reach\"}]",
        startRound(0, WAITING),
        draw(0, "S"));
  }

  /**
   * 引いた南のツモ切りだけを選べる、打牌の要求。
   *
   * @param id 要求の番号（request_id）
   * @param seat 要求を受ける席
   * @param events 要求の観測に載せるイベント
   */
  private static String discard(long id, int seat, String... events) {
    return request(id, seat, SECOND_SOUTH, discardAction(SECOND_SOUTH), dahai("S"), events);
  }

  private static String discardAction(int tile) {
    return "[{\"action_type\":\"Discard\",\"tile\":" + tile + ",\"consume_tiles\":[]}]";
  }

  private static String dahai(String pai) {
    return "[{\"type\":\"dahai\",\"pai\":\"" + pai + "\"}]";
  }

  private static String request(
      long id, int seat, int drawn, String legalActions, String possibleActions, String... events) {
    JsonObject observation = new JsonObject();
    observation.addProperty("player_id", seat);
    if (drawn >= 0) {
      observation.addProperty("drawn_tile", drawn);
    }
    observation.add("_legal_actions", JsonParser.parseString(legalActions));
    JsonArray encoded = new JsonArray();
    for (String event : events) {
      encoded.add(event);
    }
    observation.add("events", encoded);
    JsonObject request = object("{\"type\":\"request_action\",\"time\":{\"deadline_ms\":18000}}");
    request.addProperty("request_id", id);
    request.add("possible_actions", JsonParser.parseString(possibleActions));
    request.addProperty(
        "observation", Base64.getEncoder().encodeToString(observation.toString().getBytes(StandardCharsets.UTF_8)));
    return request.toString();
  }

  /**
   * その席を親にして東場の局を始めるイベント。手はその席にだけ配る（席 0 なら東 1 局）。
   *
   * @param seat 手を配る席
   * @param hand 手の牌（mjai の牌の名前を並べたもの）
   */
  private static String startRound(int seat, String hand) {
    return startRound(seat, hand, seat);
  }

  /**
   * 東場の局を始めるイベント。手は指定した席にだけ配る。
   *
   * @param seat 手を配る席
   * @param hand 手の牌（mjai の牌の名前を並べたもの）
   * @param dealer 親の席
   */
  private static String startRound(int seat, String hand, int dealer) {
    JsonObject event =
        object(
            "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"honba\":0,\"kyotaku\":0,\"dora_marker\":\"9p\","
                + "\"scores\":[25000,25000,25000,25000]}");
    event.addProperty("kyoku", dealer + 1);
    event.addProperty("oya", dealer);
    JsonArray hands = new JsonArray();
    for (int each = 0; each < 4; each++) {
      hands.add(JsonParser.parseString(each == seat ? "[" + hand + "]" : "[]"));
    }
    event.add("tehais", hands);
    return event.toString();
  }

  private static String draw(int seat, String pai) {
    return "{\"type\":\"tsumo\",\"actor\":" + seat + ",\"pai\":\"" + pai + "\"}";
  }

  private static String acknowledge(long id, String status) {
    return "{\"type\":\"action_ack\",\"request_id\":" + id + ",\"status\":\"" + status + "\"}";
  }

  private static JsonObject object(String json) {
    return JsonParser.parseString(json).getAsJsonObject();
  }

  private static int handSize(SeatView view) {
    int size = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      size += view.count(kind);
    }
    return size;
  }

  private static List<Integer> codes(Decisions decisions, int slot) {
    List<Integer> codes = new ArrayList<>();
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      codes.add(decisions.legal(slot, index));
    }
    return codes;
  }

  private static List<Type> types(Decisions decisions, int slot) {
    List<Type> types = new ArrayList<>();
    for (int code : codes(decisions, slot)) {
      Type type = Action.of(code).type();
      if (!types.contains(type)) {
        types.add(type);
      }
    }
    return types;
  }

  private static Decider first() {
    return (view, slots, slot) -> 0;
  }
}
