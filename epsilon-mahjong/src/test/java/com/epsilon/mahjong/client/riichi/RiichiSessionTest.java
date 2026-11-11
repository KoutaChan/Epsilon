package com.epsilon.mahjong.client.riichi;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;
import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.client.Decider;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.Variant;
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

  /** 三人打ちの手: 1m9m 1p2p3p4p6p7p8p 1s2s3s N（北を 1 枚持つ）。 */
  private static final String NORTH_HAND =
      "\"1m\",\"9m\",\"1p\",\"2p\",\"3p\",\"4p\",\"6p\",\"7p\",\"8p\",\"1s\",\"2s\",\"3s\",\"N\"";

  /** 三人打ちの手: 1m9m 1p2p3p4p6p7p 1s2s3s WW（西の対子）。 */
  private static final String PON_HAND =
      "\"1m\",\"9m\",\"1p\",\"2p\",\"3p\",\"4p\",\"6p\",\"7p\",\"1s\",\"2s\",\"3s\",\"W\",\"W\"";

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
            TENHOU_4P,
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
            TENHOU_4P,
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
            TENHOU_4P,
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
            TENHOU_4P,
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
            TENHOU_4P,
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
    RiichiSession session = new RiichiSession(TENHOU_4P, first(), false);
    Assert.assertThrows(
        IllegalArgumentException.class, () -> session.handle(acknowledge(1, status), System.nanoTime()));
  }

  @Test
  public void validationWaitsForItsResultAfterTheGameEnds() {
    RiichiSession session = new RiichiSession(TENHOU_4P, first(), true);
    session.handle("{\"type\":\"start_game\",\"id\":2}", 0);
    Assert.assertTrue(session.started());
    session.handle("{\"type\":\"end_game\",\"scores\":[25000,25000,25000,25000]}", 0);
    Assert.assertTrue(session.completed());
    Assert.assertFalse(session.ended());
    session.handle("{\"type\":\"validation_result\",\"passed\":true}", 0);
    Assert.assertTrue(session.ended());
    Assert.assertTrue(session.validationPassed());
  }

  /**
   * 三人打ち: 親（席 0）が北を持って 9s を引くと、Decider には、サーバが選べるとしたツモ切りと北抜きだけを見せる。北抜きを選ぶと
   * kita の要素を返す。次の要求の観測で北抜きと嶺上牌のツモを卓に書くと、手の北が抜きドラになり、手は 14 枚に戻る。
   */
  @Test
  public void aThreePlayerSeatAnswersAKitaRequestAndTakesTheNorthOut() {
    List<List<Type>> seen = new ArrayList<>();
    List<Integer> nuki = new ArrayList<>();
    RiichiSession session =
        new RiichiSession(
            TENHOU_3P,
            (view, slots, slot) -> {
              seen.add(types(slots, slot));
              nuki.add(view.nuki(0));
              Assert.assertEquals(handSize(view), 14);
              return indexOf(slots, slot, Type.NUKI);
            },
            false);
    int nineSou = Tiles.physicalId(Tiles.S9, 0);
    int north = Tiles.physicalId(Tiles.PEI, 0);
    JsonObject kita =
        object(
            session.handle(
                request(
                    1,
                    0,
                    nineSou,
                    "[{\"action_type\":\"Discard\",\"tile\":" + nineSou + ",\"consume_tiles\":[]},"
                        + "{\"action_type\":\"Kita\",\"tile\":" + north + ",\"consume_tiles\":[]}]",
                    "[{\"type\":\"dahai\",\"pai\":\"9s\"},{\"type\":\"kita\",\"pai\":\"N\"}]",
                    startRound(TENHOU_3P, 0, NORTH_HAND, 0),
                    draw(0, "9s")),
                System.nanoTime()));
    Assert.assertEquals(kita.get("type").getAsString(), "kita");
    Assert.assertEquals(kita.get("pai").getAsString(), "N");
    Assert.assertEquals(kita.get("actor").getAsInt(), 0);
    Assert.assertEquals(kita.get("request_id").getAsLong(), 1L);
    Assert.assertFalse(kita.has("target"));
    Assert.assertFalse(kita.has("tsumogiri"));

    int sevenSou = Tiles.physicalId(Tiles.S7, 0);
    JsonObject discard =
        object(
            session.handle(
                request(
                    2,
                    0,
                    sevenSou,
                    discardAction(sevenSou),
                    dahai("7s"),
                    "{\"type\":\"kita\",\"actor\":0,\"pai\":\"N\"}",
                    draw(0, "7s")),
                System.nanoTime()));
    Assert.assertEquals(discard.get("type").getAsString(), "dahai");
    Assert.assertTrue(discard.get("tsumogiri").getAsBoolean());
    Assert.assertEquals(seen, List.of(List.of(Type.NUKI, Type.DAHAI), List.of(Type.DAHAI)));
    Assert.assertEquals(nuki, List.of(0, 1));
  }

  /**
   * 三人打ち: 親（席 2）が北を抜き、嶺上牌をツモって西を手出しする。下家の席 0 が西をポンする。席 0 から見て席 2 は上家なので、返す
   * ポンの target は 2 で、観測のポンのイベントも上家からの面子として卓に書ける。北抜きは席 2 の抜きドラとして席 0 の卓にも載る。
   */
  @Test
  public void aThreePlayerSeatPonsFromTheSeatAcrossTheWrap() {
    List<List<Type>> seen = new ArrayList<>();
    List<Integer> sizes = new ArrayList<>();
    RiichiSession session =
        new RiichiSession(
            TENHOU_3P,
            (view, slots, slot) -> {
              seen.add(types(slots, slot));
              sizes.add(handSize(view));
              Assert.assertEquals(view.nuki(2), 1);
              return indexOf(slots, slot, Type.PON);
            },
            false);
    JsonObject pon =
        object(
            session.handle(
                request(
                    1,
                    0,
                    -1,
                    "[{\"action_type\":\"Pon\",\"tile\":" + Tiles.physicalId(Tiles.SHA, 2) + ",\"consume_tiles\":["
                        + Tiles.physicalId(Tiles.SHA, 0) + "," + Tiles.physicalId(Tiles.SHA, 1) + "]},"
                        + "{\"action_type\":\"Pass\",\"tile\":null,\"consume_tiles\":[]}]",
                    "[{\"type\":\"pon\",\"pai\":\"W\",\"consumed\":[\"W\",\"W\"]},{\"type\":\"none\"}]",
                    startRound(TENHOU_3P, 0, PON_HAND, 2),
                    "{\"type\":\"tsumo\",\"actor\":2,\"pai\":\"?\"}",
                    "{\"type\":\"kita\",\"actor\":2,\"pai\":\"N\"}",
                    "{\"type\":\"tsumo\",\"actor\":2,\"pai\":\"?\"}",
                    "{\"type\":\"dahai\",\"actor\":2,\"pai\":\"W\",\"tsumogiri\":false}"),
                System.nanoTime()));
    Assert.assertEquals(pon.get("type").getAsString(), "pon");
    Assert.assertEquals(pon.get("actor").getAsInt(), 0);
    Assert.assertEquals(pon.get("target").getAsInt(), 2);

    int oneMan = Tiles.physicalId(Tiles.M1, 0);
    JsonObject discard =
        object(
            session.handle(
                request(
                    2,
                    0,
                    -1,
                    discardAction(oneMan),
                    dahai("1m"),
                    "{\"type\":\"pon\",\"actor\":0,\"target\":2,\"pai\":\"W\",\"consumed\":[\"W\",\"W\"]}"),
                System.nanoTime()));
    Assert.assertEquals(discard.get("type").getAsString(), "dahai");
    Assert.assertEquals(discard.get("pai").getAsString(), "1m");
    Assert.assertFalse(discard.get("tsumogiri").getAsBoolean());
    Assert.assertEquals(seen, List.of(List.of(Type.PON, Type.PASS), List.of(Type.DAHAI)));
    Assert.assertEquals(sizes, List.of(13, 11));
  }

  /**
   * 三人打ち: 席 2 が大明槓の嶺上で北を抜き、サーバはロンを聞く前に保留の槓ドラを送る（kita → dora）。北の単騎（一気通貫）で待つ
   * 席 0 には、表示牌を受けた後もロンと見送りが残り、ロンを選ぶと target が席 2 の hora を返す。
   */
  @Test
  public void aRonOnKitaSurvivesTheKanDoraSentBeforeIt() {
    List<List<Type>> seen = new ArrayList<>();
    RiichiSession session =
        new RiichiSession(
            TENHOU_3P,
            (view, slots, slot) -> {
              seen.add(types(slots, slot));
              return indexOf(slots, slot, Type.RON_AGARI);
            },
            false);
    String hand = "\"1p\",\"2p\",\"3p\",\"4p\",\"5p\",\"6p\",\"7p\",\"8p\",\"9p\",\"1s\",\"1s\",\"1s\",\"N\"";
    JsonObject ron =
        object(
            session.handle(
                request(
                    1,
                    0,
                    -1,
                    "[{\"action_type\":\"Ron\",\"tile\":" + Tiles.physicalId(Tiles.PEI, 1) + ",\"consume_tiles\":[]},"
                        + "{\"action_type\":\"Pass\",\"tile\":null,\"consume_tiles\":[]}]",
                    "[{\"type\":\"hora\"},{\"type\":\"none\"}]",
                    startRound(TENHOU_3P, 0, hand, 1),
                    "{\"type\":\"tsumo\",\"actor\":1,\"pai\":\"?\"}",
                    "{\"type\":\"dahai\",\"actor\":1,\"pai\":\"9s\",\"tsumogiri\":false}",
                    "{\"type\":\"daiminkan\",\"actor\":2,\"target\":1,\"pai\":\"9s\",\"consumed\":[\"9s\",\"9s\",\"9s\"]}",
                    "{\"type\":\"tsumo\",\"actor\":2,\"pai\":\"?\"}",
                    "{\"type\":\"kita\",\"actor\":2,\"pai\":\"N\"}",
                    "{\"type\":\"dora\",\"dora_marker\":\"1s\"}"),
                System.nanoTime()));
    Assert.assertEquals(seen, List.of(List.of(Type.RON_AGARI, Type.PASS)));
    Assert.assertEquals(ron.get("type").getAsString(), "hora");
    Assert.assertEquals(ron.get("target").getAsInt(), 2);
  }

  /** セッションは自分のルールの卓だけを打つ。席の数の違う局の始まりと、席の数を超える席の番号は、プロトコルのエラーにする。 */
  @Test
  public void aSessionPlaysOnlyItsOwnVariant() {
    RiichiSession four = new RiichiSession(TENHOU_4P, first(), false);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> four.handle(discard(1, 0, startRound(TENHOU_3P, 0, NORTH_HAND, 0), draw(0, "S")), System.nanoTime()));
    RiichiSession three = new RiichiSession(TENHOU_3P, first(), false);
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> three.handle(discard(1, 0, startRound(0, WAITING), draw(0, "S")), System.nanoTime()));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new RiichiSession(TENHOU_3P, first(), false).handle("{\"type\":\"start_game\",\"id\":3}", 0));
    RiichiSession seated = new RiichiSession(TENHOU_3P, first(), false);
    seated.handle("{\"type\":\"start_game\",\"id\":2}", 0);
    Assert.assertTrue(seated.started());
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
        TENHOU_4P,
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
   * 四人打ちの東場の局を始めるイベント。手は指定した席にだけ配る。
   *
   * @param seat 手を配る席
   * @param hand 手の牌（mjai の牌の名前を並べたもの）
   * @param dealer 親の席
   */
  private static String startRound(int seat, String hand, int dealer) {
    return startRound(TENHOU_4P, seat, hand, dealer);
  }

  /**
   * 東場の局を始めるイベント。席の数と持ち点はルールのもの。手は指定した席にだけ配る。
   *
   * @param variant ルール
   * @param seat 手を配る席
   * @param hand 手の牌（mjai の牌の名前を並べたもの）
   * @param dealer 親の席
   */
  private static String startRound(Variant variant, int seat, String hand, int dealer) {
    JsonObject event =
        object(
            "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"honba\":0,\"kyotaku\":0,\"dora_marker\":\"9p\"}");
    event.addProperty("kyoku", dealer + 1);
    event.addProperty("oya", dealer);
    JsonArray hands = new JsonArray();
    JsonArray scores = new JsonArray();
    for (int each = 0; each < variant.seats(); each++) {
      hands.add(JsonParser.parseString(each == seat ? "[" + hand + "]" : "[]"));
      scores.add(variant.startPoints());
    }
    event.add("tehais", hands);
    event.add("scores", scores);
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

  /** 合法手のうち、その種類の最初の手の番号。その種類の手が無ければ 0。 */
  private static int indexOf(Decisions decisions, int slot, Type type) {
    List<Integer> codes = codes(decisions, slot);
    for (int index = 0; index < codes.size(); index++) {
      if (Action.of(codes.get(index)).type() == type) {
        return index;
      }
    }
    return 0;
  }

  private static Decider first() {
    return (view, slots, slot) -> 0;
  }
}
