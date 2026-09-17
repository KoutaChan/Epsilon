package com.epsilon.replay.format;

import static org.testng.Assert.*;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.replay.*;
import com.epsilon.replay.ReplayEvent.*;
import com.google.gson.*;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.testng.annotations.Test;

/** 雀魂のJSONとバイナリ牌譜について、赤牌・副露・採点・終局情報の復元と不正データの拒否を検証する。 */
public class MahjongSoulReaderTest {
  @Test
  public void binaryDefaultsKeepDealerDrawAndPackedScoresWithoutGuessingCompletion() {
    byte[] initial = binaryStart(true);
    byte[] detail =
        join(
            integer(2, 210715),
            action("RecordNewRound", initial),
            action("RecordDiscardTile", join(field(2, "0m"), integer(5, 1))),
            action("RecordLiuJu", integer(1, 1)));
    ReplayRecord result = MahjongSoulReader.readRecord(wrapper("GameDetailRecords", detail));
    assertEquals(events(result, StartGame.class).size(), 1);
    assertEquals(
        events(result, StartKyoku.class).getFirst().initialHandPhysicalTileIds()[0].length, 13);
    Tsumo draw = events(result, Tsumo.class).getFirst();
    assertEquals(draw.physicalTileId(), 16);
    assertTrue(draw.inferred());
    assertFalse(events(result, Dahai.class).getFirst().inferred());
    assertEquals(result.completion(), RecordCompletion.INCOMPLETE);
    assertNull(result.metadata().finalScores());
    assertEquals(events(result, Ryukyoku.class).getFirst().reason(), "NINE_TERMINALS");
  }

  @Test
  public void decodedConcealedKanKeepsRedTileAndFinalScoreMetadata() {
    JsonObject initial = start();
    initial.add(
        "tiles0",
        new Gson()
            .toJsonTree(
                List.of(
                    "0m", "5m", "5m", "5m", "1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1z",
                    "2z")));
    replaceTile(initial, 2, "5m", "3z");
    JsonObject document =
        document(
            initial,
            record("RecordAnGangAddGang", json("{\"seat\":0,\"type\":3,\"tiles\":\"5m\"}")),
            record(
                "RecordLiuJu",
                json("{\"type\":1,\"gameend\":{\"scores\":[25000,25000,25000,25000]}}")));
    document.add("head", json("{\"accounts\":[{\"seat\":2,\"nickname\":\"\u897f\u306e\u4eba\"}]}"));
    ReplayRecord result = read(document);
    assertEquals(result.metadata().names().get(2), "\u897f\u306e\u4eba");
    assertEquals(result.metadata().finalScores(), new int[] {25000, 25000, 25000, 25000});
    assertEquals(result.completion(), RecordCompletion.COMPLETE);
    int[] consumed = events(result, Ankan.class).getFirst().consumedPhysicalTileIds();
    assertEquals(Arrays.stream(consumed).filter(Tile::isAka).count(), 1L);
    assertEquals(Arrays.stream(consumed).distinct().count(), 4L);
    assertEquals(result.metadata().format(), RecordFormat.MAHJONG_SOUL);
  }

  @Test
  public void calledRedTileAndLaterAddedTileKeepDistinctPhysicalIdentities() {
    JsonObject initial = start();
    initial.getAsJsonArray("tiles0").add("0m");
    replaceTile(initial, 0, "5m", "3z");
    replaceTile(initial, 2, "5m", "4z");
    initial.add(
        "tiles1",
        new Gson()
            .toJsonTree(
                List.of(
                    "5m", "5m", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1z", "2z")));
    ReplayRecord result =
        read(
            document(
                initial,
                record("RecordDiscardTile", json("{\"seat\":0,\"tile\":\"0m\",\"moqie\":true}")),
                record(
                    "RecordChiPengGang",
                    json(
                        "{\"seat\":1,\"type\":1,\"tiles\":[\"5m\",\"5m\",\"0m\"],\"froms\":[1,1,0]}")),
                record("RecordDealTile", json("{\"seat\":1,\"tile\":\"5m\"}")),
                record("RecordAnGangAddGang", json("{\"seat\":1,\"type\":2,\"tiles\":\"5m\"}"))));
    Pon pon = events(result, Pon.class).getFirst();
    assertEquals(pon.target(), 0);
    assertTrue(Tile.isAka(pon.calledPhysicalTileId()));
    assertEquals(Arrays.stream(pon.consumedPhysicalTileIds()).filter(Tile::isAka).count(), 0L);
    int added = events(result, Kakan.class).getFirst().addedPhysicalTileId();
    assertFalse(Tile.isAka(added));
    assertFalse(Arrays.stream(pon.consumedPhysicalTileIds()).anyMatch(id -> id == added));
  }

  @Test
  public void jsonWrapperPreservesHeaderAndMarksOnlyInferredInitialDiscard() {
    JsonObject initial = start();
    initial.getAsJsonArray("tiles0").add("0m");
    JsonObject body =
        document(
            initial,
            record("RecordDiscardTile", json("{\"seat\":0,\"tile\":\"0m\",\"moqie\":false}")),
            record("RecordLiuJu", json("{\"type\":1}")));
    JsonObject wrapper = new JsonObject();
    wrapper.addProperty("name", ".lq.GameDetailRecords");
    wrapper.add("data", body);
    JsonObject source = new JsonObject();
    source.add("data", wrapper);
    source.add(
        "head",
        json(
            "{\"result\":{\"players\":[{\"seat\":0,\"partPoint1\":30000},{\"seat\":1,\"partPoint1\":27000},{\"seat\":2,\"partPoint1\":24000},{\"seat\":3,\"partPoint1\":19000}]}}"));
    ReplayRecord result = read(source);
    Dahai discard = events(result, Dahai.class).getFirst();
    assertTrue(discard.tsumogiri());
    assertTrue(discard.inferred());
    assertEquals(result.metadata().finalScoresSource(), "majsoul.head.result.part_point_1");
    assertEquals(result.metadata().finalScores(), new int[] {30000, 27000, 24000, 19000});
    assertEquals(result.completion(), RecordCompletion.COMPLETE);
  }

  @Test
  public void missingRequiredJsonSeatFailsInsteadOfSelectingEast() {
    JsonObject initial = start();
    initial.getAsJsonArray("tiles0").add("0m");
    assertThrows(
        IllegalArgumentException.class,
        () ->
            read(
                document(
                    initial,
                    record("RecordDiscardTile", json("{\"tile\":\"0m\",\"moqie\":true}")))));
    initial.remove("chang");
    assertThrows(IllegalArgumentException.class, () -> read(document(initial)));
  }

  @Test
  public void missingHandsUnknownEventsAndTruncatedProtobufFailExplicitly() {
    JsonObject initial = start();
    initial.remove("tiles3");
    assertThrows(IllegalArgumentException.class, () -> read(document(initial)));
    assertThrows(
        IllegalArgumentException.class,
        () -> MahjongSoulReader.readRecord(new byte[] {10, 127, 1}));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            MahjongSoulReader.readRecord(
                wrapper("GameDetailRecords", field(1, wrapper("RecordBaBei", integer(1, 0))))));
  }

  @Test
  public void nestedResponseMessagesAndExcessProtobufFieldsFail() {
    byte[] nested = wrapper("ResGameRecord", new byte[0]);
    for (int i = 0; i < 100; i++) nested = wrapper("ResGameRecord", field(4, nested));
    byte[] input = nested;
    assertThrows(IllegalArgumentException.class, () -> MahjongSoulReader.readRecord(input));
    byte[] many = new byte[200002];
    for (int i = 0; i < many.length; i += 2) many[i] = 8;
    assertThrows(IllegalArgumentException.class, () -> MahjongSoulReader.readRecord(many));
  }

  @Test
  public void riichiAcceptanceAndDoraRevealPrecedeNextDraw() {
    JsonObject initial = start();
    initial.getAsJsonArray("tiles0").add("0m");
    ReplayRecord result =
        read(
            document(
                initial,
                record(
                    "RecordDiscardTile",
                    json("{\"seat\":0,\"tile\":\"0m\",\"moqie\":true,\"is_liqi\":true}")),
                record(
                    "RecordDealTile",
                    json(
                        "{\"seat\":1,\"tile\":\"0p\",\"liqi\":{\"seat\":0,\"failed\":false},\"doras\":[\"7z\",\"6z\"]}")),
                record("RecordLiuJu", json("{\"type\":4,\"gameend\":true}"))));
    List<Class<?>> tail = result.events().stream().skip(3).<Class<?>>map(Object::getClass).toList();
    assertEquals(
        tail,
        List.of(
            Reach.class,
            Dahai.class,
            ReachAccepted.class,
            Dora.class,
            Tsumo.class,
            Ryukyoku.class,
            EndKyoku.class,
            EndGame.class));
    assertEquals(result.metadata().finalScores(), new int[] {24000, 25000, 25000, 25000});
  }

  @Test
  public void decodedWinKeepsYakuRedUraAndBasePointsInsteadOfCumulativePoints() {
    JsonObject initial = start();
    initial.addProperty("chang", 1);
    initial.getAsJsonArray("tiles0").add("0m");
    Hora win =
        wins(
                initial,
                """
                {"hules":[{"seat":2,"hu_tile":"0m","zimo":false,"liqi":true,
                  "count":6,"fu":30,"yiman":false,"point_rong":12000,"point_sum":14000,
                  "li_doras":["0p","5z"],"fans":[{"id":2,"val":1},{"id":10,"val":1},
                  {"id":11,"val":1},{"id":33,"val":3},{"id":31,"val":0}]}],
                  "delta_scores":[-12000,0,12000,0],"scores":[13000,25000,37000,25000]}
                """)
            .getFirst();
    assertEquals(win.winTileType(), 4);
    assertEquals(
        win.details(),
        new WinDetails(
            6,
            30,
            12000,
            0,
            List.of(
                new WinDetails.Yaku("RIICHI", 1, 0),
                new WinDetails.Yaku("YAKUHAI_SEAT_W", 1, 0),
                new WinDetails.Yaku("YAKUHAI_ROUND_S", 1, 0),
                new WinDetails.Yaku("URADORA", 3, 0)),
            List.of("5pr", "P")));
  }

  @Test
  public void binaryWinKeepsFanFieldNumbersUraAndTsumoPayments() {
    // 固定したLiqi形式のテストデータ。フィールド番号はJSONデコーダーから独立している。
    ReplayRecord result =
        MahjongSoulReader.readRecord(
            Base64.getDecoder()
                .decode(
                    "ChUubHEuR2FtZURldGFpbFJlY29yZHMSlwMa/AEa+QEKEi5scS5SZWNvcmROZXdSb3VuZBLiASoMqMMBqMMBqMMBqMMBIgI3ejoCMW06AjJtOgIzbToCNG06AjVtOgI2bToCN206AjhtOgI5bToCMXA6AjJwOgIzcDoCNHBCAjVwQgI2cEICN3BCAjhwQgI5cEICMXNCAjJzQgIzc0ICNHNCAjVzQgI2c0ICN3NCAjhzSgI5c0oCMXpKAjJ6SgIzekoCNHpKAjV6SgI2ekoCN3pKAjFtSgIybUoCM21KAjRtSgI1bVICNm1SAjdtUgI4bVICOW1SAjFwUgIycFICM3BSAjRwUgI1cFICNnBSAjdwUgI4cFICOXAaHhocChIubHEuUmVjb3JkRGVhbFRpbGUSBggCEgIwcxp2GnQKDi5scS5SZWNvcmRIdWxlEmIKPhoCMHMgAigBOAFKAjV6SgIwcFgDYgQQARgBYgQQARgCYgQQARggYgQQABghaCh40CiAAagUiAGUCpgBgJYBGiDY6/////////8B7PX/////////AdAo7PX/////////AQ=="));
    Hora win = events(result, Hora.class).getFirst();
    assertEquals(win.target(), 2);
    assertEquals(win.winTileType(), 22);
    assertEquals(win.details().points(), Integer.valueOf(5200));
    assertEquals(win.details().han(), Integer.valueOf(3));
    assertEquals(win.details().fu(), Integer.valueOf(40));
    assertEquals(win.details().uraIndicators(), List.of("P", "5pr"));
    assertEquals(win.details().yaku().get(2).code(), "AKADORA");
  }

  @Test
  public void multipleRonKeepsPerWinnerScoringAndCombinedDeltasOnce() {
    JsonObject initial = start();
    initial.getAsJsonArray("tiles0").add("0m");
    List<Hora> wins =
        wins(
            initial,
            """
            {"hules":[
              {"seat":1,"huTile":"0m","zimo":false,"liqi":true,"count":2,"fu":30,
               "pointRong":2000,"fans":[{"id":2,"val":1},{"id":32,"val":1}]},
              {"seat":2,"huTile":"0m","zimo":false,"liqi":false,"yiman":true,"count":2,
               "pointRong":64000,"fans":[{"id":49,"val":2}]}],
              "deltaScores":[-66000,2000,64000,0]}
            """);
    assertEquals(wins.size(), 2);
    assertEquals(wins.get(0).deltas(), new int[] {-66000, 2000, 64000, 0});
    assertEquals(wins.get(1).deltas(), new int[4]);
    assertNull(wins.get(0).details().uraIndicators());
    assertEquals(
        wins.get(1).details(),
        new WinDetails(
            null, null, 64000, 2, List.of(new WinDetails.Yaku("KOKUSHI_13", 0, 2)), List.of()));
  }

  @Test
  public void dealerTsumoUsesThreeChildPaymentsAndPreservesUnknownYakuId() {
    JsonObject initial = start();
    initial.getAsJsonArray("tiles0").add("0m");
    WinDetails details =
        wins(
                initial,
                """
                {"hules":[{"seat":0,"hu_tile":"0m","zimo":true,"count":3,"fu":30,
                  "point_zimo_xian":2000,"point_sum":20000,"fans":[{"id":999,"val":3}]}],
                  "delta_scores":[6000,-2000,-2000,-2000]}
                """)
            .getFirst()
            .details();
    assertEquals(details.points(), Integer.valueOf(6000));
    assertEquals(details.yaku(), List.of(new WinDetails.Yaku("MAJSOUL_999", 3, 0)));
    assertNull(details.uraIndicators());
  }

  @Test
  public void incompleteWinDoesNotInventScoringOrHiddenUra() {
    JsonObject initial = start();
    initial.getAsJsonArray("tiles0").add("0m");
    WinDetails details =
        wins(
                initial,
                """
                {"hules":[{"seat":2,"hu_tile":"0m","zimo":false,"fans":null,"li_doras":null}],
                  "delta_scores":[-8000,0,8000,0]}
                """)
            .getFirst()
            .details();
    assertEquals(details, new WinDetails(null, null, null, null, null, null));
  }

  @Test
  public void actualOpenRoundIsIncompleteEvenWithHeaderFinalScores() {
    JsonObject source = document(start());
    source.add(
        "head",
        json(
            "{\"result\":{\"players\":[{\"seat\":0,\"part_point_1\":30000},{\"seat\":1,\"part_point_1\":27000},{\"seat\":2,\"part_point_1\":24000},{\"seat\":3,\"part_point_1\":19000}]}}"));
    ReplayRecord result = read(source);
    assertEquals(result.completion(), RecordCompletion.INCOMPLETE);
    assertTrue(events(result, EndGame.class).isEmpty());
    assertNotNull(result.metadata().finalScores());
  }

  @Test
  public void jsonBase64WrapperAndActionRecordsShareTypedDecoding() {
    byte[] detail =
        join(
            action("RecordNewRound", binaryStart(true)),
            action("RecordDiscardTile", join(field(2, "0m"), integer(5, 1))),
            action("RecordLiuJu", integer(1, 1)));
    JsonObject wrapped = new JsonObject();
    wrapped.addProperty("name", ".lq.GameDetailRecords");
    wrapped.addProperty("data", Base64.getEncoder().encodeToString(detail));
    assertEquals(events(read(wrapped), Dahai.class).getFirst().physicalTileId(), 16);
    JsonObject action = new JsonObject();
    action.add("result", record("RecordNewRound", start()));
    JsonArray actions = new JsonArray();
    actions.add(action);
    JsonObject ignoredAction = new JsonObject();
    ignoredAction.addProperty("result", "");
    actions.add(ignoredAction);
    JsonObject source = new JsonObject();
    source.add("actions", actions);
    assertEquals(events(read(source), StartKyoku.class).size(), 1);
  }

  @Test
  public void jsonApiErrorAndWrongProtobufStringTypeFailExplicitly() {
    JsonObject source = document(start());
    source.add("error", json("{\"code\":1001}"));
    assertThrows(IllegalArgumentException.class, () -> read(source));
    byte[] malformed =
        wrapper(
            "GameDetailRecords",
            action("RecordNewRound", join(packed(5, 25000, 25000, 25000, 25000), integer(4, 7))));
    assertThrows(IllegalArgumentException.class, () -> MahjongSoulReader.readRecord(malformed));
  }

  @Test
  public void emptyNativeGameEndMessageDoesNotCloseTheMatch() {
    JsonObject source =
        document(
            start(),
            record("RecordLiuJu", json("{\"type\":1,\"gameend\":{}}")),
            record("RecordNewRound", start()));
    source.add("error", new JsonObject());
    ReplayRecord decoded = read(source);
    assertEquals(events(decoded, StartKyoku.class).size(), 2);
    assertEquals(decoded.completion(), RecordCompletion.INCOMPLETE);
    byte[] binary =
        wrapper(
            "GameDetailRecords",
            join(
                action("RecordNewRound", binaryStart(false)),
                action("RecordLiuJu", join(integer(1, 1), field(2, new byte[0]))),
                action("RecordNewRound", binaryStart(false))));
    assertEquals(MahjongSoulReader.readRecord(binary).completion(), RecordCompletion.INCOMPLETE);
  }

  @Test(dataProvider = "recordEncodings")
  public void exhaustiveDrawAcceptsOmittedAndEmptyScoreChanges(boolean binary) {
    for (boolean emptyArray : new boolean[] {false, true}) {
      byte[] score = packed(2, 25000, 25000, 25000, 25000);
      JsonObject jsonScore = json("{\"old_scores\":[25000,25000,25000,25000]}");
      if (emptyArray) {
        score = join(score, packed(3));
        jsonScore.add("delta_scores", new JsonArray());
      }
      ReplayRecord result =
          exhaustiveDraw(
              binary,
              join(field(3, score), integer(4, 1)),
              json("{\"scores\":[" + jsonScore + "],\"gameend\":true}"));
      assertEquals(events(result, Ryukyoku.class).getFirst().deltas(), new int[4]);
      assertEquals(result.metadata().finalScores(), new int[] {25000, 25000, 25000, 25000});
      assertEquals(result.completion(), RecordCompletion.COMPLETE);
    }
    ReplayRecord noTransfers = exhaustiveDraw(binary, integer(4, 1), json("{\"gameend\":true}"));
    assertEquals(events(noTransfers, Ryukyoku.class).getFirst().deltas(), new int[4]);
  }

  @Test(dataProvider = "recordEncodings")
  public void exhaustiveDrawPreservesRecordedTransfers(boolean binary) {
    ReplayRecord result =
        exhaustiveDraw(
            binary,
            join(field(3, packed(3, 3000, -1000, -1000, -1000)), integer(4, 1)),
            json("{\"scores\":[{\"delta_scores\":[3000,-1000,-1000,-1000]}],\"gameend\":true}"));
    assertEquals(
        events(result, Ryukyoku.class).getFirst().deltas(), new int[] {3000, -1000, -1000, -1000});
    assertEquals(result.metadata().finalScores(), new int[] {28000, 24000, 24000, 24000});
  }

  @Test(dataProvider = "recordEncodings")
  public void exhaustiveDrawRejectsIncompleteNonemptyScoreChanges(boolean binary) {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            exhaustiveDraw(
                binary,
                field(3, packed(3, 0, 0, 0)),
                json("{\"scores\":[{\"delta_scores\":[0,0,0]}]}")));
  }

  private static ReplayRecord exhaustiveDraw(boolean binary, byte[] record, JsonObject jsonRecord) {
    return binary
        ? MahjongSoulReader.readRecord(
            wrapper(
                "GameDetailRecords",
                join(action("RecordNewRound", binaryStart(false)), action("RecordNoTile", record))))
        : read(document(start(), record("RecordNoTile", jsonRecord)));
  }

  @org.testng.annotations.DataProvider
  public Object[][] recordEncodings() {
    return new Object[][] {{false}, {true}};
  }

  @Test(dataProvider = "recordEncodings")
  public void nineTerminalsReachesTheReplayDecisionConsumer(boolean binary) {
    JsonObject initial = start();
    initial.add(
        "tiles0",
        new Gson()
            .toJsonTree(
                List.of(
                    "1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "2m", "3m", "4m", "5m",
                    "7z")));
    ReplayRecord result = readAbortiveDraw(initial, 1, binary);
    var choices = new ArrayList<Action>();
    ReplayEngine.replay(
        result,
        (point, state, player, legal) -> {
          choices.add(legal.get(point.chosenSlot()));
          assertEquals(player, 0);
          assertEquals(point.choiceKind(), DecisionPoint.ChoiceKind.RECORDED_ACTION);
          assertTrue(result.events().get(point.causeEventIndex()) instanceof Tsumo);
          assertTrue(result.events().get(point.eventIndex()) instanceof Ryukyoku);
        });
    assertEquals(choices, List.of(Action.kyushuKyuhai()));
  }

  @org.testng.annotations.DataProvider
  public Object[][] otherAbortiveDrawCodes() {
    return new Object[][] {{0}, {2}, {3}, {4}, {5}, {99}};
  }

  @Test(dataProvider = "otherAbortiveDrawCodes")
  public void otherNativeDrawCodesAreNotNineTerminals(int reason) {
    for (boolean binary : new boolean[] {false, true}) {
      ReplayRecord result = readAbortiveDraw(start(), reason, binary);
      assertEquals(events(result, Ryukyoku.class).getFirst().reason(), "MAHJONG_SOUL_" + reason);
      ReplayEngine.replay(
          result,
          (point, state, player, legal) ->
              fail("Other native draw codes must not invent a nine-terminals decision"));
    }
  }

  private static ReplayRecord readAbortiveDraw(JsonObject initial, int reason, boolean binary) {
    if (!binary)
      return read(document(initial, record("RecordLiuJu", json("{\"type\":" + reason + "}"))));
    byte[] round =
        join(packed(5, 25000, 25000, 25000, 25000), field(4, initial.get("dora").getAsString()));
    for (int seat = 0; seat < 4; seat++)
      for (JsonElement tile : initial.getAsJsonArray("tiles" + seat))
        round = join(round, field(7 + seat, tile.getAsString()));
    return MahjongSoulReader.readRecord(
        wrapper(
            "GameDetailRecords",
            join(action("RecordNewRound", round), action("RecordLiuJu", integer(1, reason)))));
  }

  private static <T extends ReplayEvent> List<T> events(ReplayRecord result, Class<T> type) {
    return result.events().stream().filter(type::isInstance).map(type::cast).toList();
  }

  private static List<Hora> wins(JsonObject initial, String text) {
    JsonObject win = json(text);
    JsonObject source = document(initial);
    JsonArray records = source.getAsJsonArray("records");
    if (!win.getAsJsonArray("hules").get(0).getAsJsonObject().get("zimo").getAsBoolean())
      records.add(record("RecordDiscardTile", json("{\"seat\":0,\"tile\":\"0m\",\"moqie\":true}")));
    records.add(record("RecordHule", win));
    return events(read(source), Hora.class);
  }

  private static ReplayRecord read(JsonElement source) {
    return MahjongSoulReader.readRecord(source.toString().getBytes(StandardCharsets.UTF_8));
  }

  private static JsonObject json(String text) {
    return JsonParser.parseString(text).getAsJsonObject();
  }

  private static JsonObject document(JsonObject initial, JsonObject... remaining) {
    JsonArray records = new JsonArray();
    records.add(record("RecordNewRound", initial));
    for (JsonObject next : remaining) records.add(next);
    JsonObject source = new JsonObject();
    source.add("records", records);
    return source;
  }

  private static void replaceTile(JsonObject initial, int seat, String from, String to) {
    JsonArray hand = initial.getAsJsonArray("tiles" + seat);
    for (int i = 0; i < hand.size(); i++)
      if (hand.get(i).getAsString().equals(from)) {
        hand.set(i, new JsonPrimitive(to));
        return;
      }
    throw new AssertionError("Fixture tile was not found: " + from);
  }

  private static byte[] binaryStart(boolean dealerDraw) {
    byte[] result = join(packed(5, 25000, 25000, 25000, 25000), field(4, "7z"));
    for (int seat = 0; seat < 4; seat++)
      for (String tile : hand(seat)) result = join(result, field(7 + seat, tile));
    return dealerDraw ? join(result, field(7, "0m")) : result;
  }

  private static JsonObject start() {
    JsonObject o =
        JsonParser.parseString(
                "{\"chang\":0,\"ju\":0,\"ben\":0,\"liqibang\":0,\"dora\":\"7z\",\"scores\":[25000,25000,25000,25000]}")
            .getAsJsonObject();
    for (int seat = 0; seat < 4; seat++) o.add("tiles" + seat, new Gson().toJsonTree(hand(seat)));
    return o;
  }

  private static List<String> hand(int seat) {
    List<String> hand = new ArrayList<>();
    for (int i = 0; i < 13; i++) {
      int type = (seat * 13 + i) % 34;
      hand.add(type < 27 ? (type % 9 + 1) + "" + "mps".charAt(type / 9) : (type - 26) + "z");
    }
    return hand;
  }

  private static JsonObject record(String name, JsonObject data) {
    JsonObject o = new JsonObject();
    o.addProperty("name", ".lq." + name);
    o.add("data", data);
    return o;
  }

  private static byte[] action(String name, byte[] data) {
    return field(3, field(3, wrapper(name, data)));
  }

  private static byte[] wrapper(String name, byte[] data) {
    return join(field(1, ".lq." + name), field(2, data));
  }

  private static byte[] field(int number, String text) {
    return field(number, text.getBytes(StandardCharsets.UTF_8));
  }

  private static byte[] field(int number, byte[] data) {
    return join(varint((number << 3) | 2), varint(data.length), data);
  }

  private static byte[] integer(int number, long value) {
    return join(varint(number << 3), varint(value));
  }

  private static byte[] packed(int number, int... values) {
    ByteArrayOutputStream bytes = new ByteArrayOutputStream();
    for (int value : values) bytes.writeBytes(varint(value));
    return field(number, bytes.toByteArray());
  }

  private static byte[] varint(long value) {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    do {
      int b = (int) value & 127;
      value >>>= 7;
      out.write(value == 0 ? b : b | 128);
    } while (value != 0);
    return out.toByteArray();
  }

  private static byte[] join(byte[]... values) {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    for (byte[] value : values) out.writeBytes(value);
    return out.toByteArray();
  }
}
