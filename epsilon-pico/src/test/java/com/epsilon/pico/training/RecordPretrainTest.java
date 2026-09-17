package com.epsilon.pico.training;

import com.epsilon.core.Tile;
import com.epsilon.replay.TileNotation;
import com.google.gson.*;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.file.*;
import java.util.*;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 3種類の牌譜形式から、同じ Decision 入力と最終順位の教師データが生成されることを検証する。 */
public final class RecordPretrainTest {
  @Test
  public void allThreeFormatsProduceTheSameDecisionInputsAndFinalRankTeachers() throws Exception {
    Path directory = Files.createTempDirectory("pretrain-formats-");
    try {
      String[] content = {tenhouRecord(), mjaiRecord(), mahjongSoulRecord()};
      String[] names = {"tenhou.xml", "mjai.jsonl", "majsoul.json"};
      var inputs = new ArrayList<List<com.epsilon.pico.ai.decision.data.EpsilonDecisionSample>>();
      for (int i = 0; i < names.length; i++) {
        Path file = Files.writeString(directory.resolve(names[i]), content[i]);
        inputs.add(EpsilonLogPretrainDataCollector.collectDecisionFile(file));
        Assert.assertFalse(EpsilonLogPretrainDataCollector.collectBeliefFile(file).isEmpty());
      }
      Assert.assertFalse(inputs.getFirst().isEmpty());
      for (int format = 1; format < inputs.size(); format++) {
        Assert.assertEquals(inputs.get(format).size(), inputs.getFirst().size());
        for (int index = 0; index < inputs.getFirst().size(); index++) {
          var expected = inputs.getFirst().get(index);
          var actual = inputs.get(format).get(index);
          Assert.assertEquals(actual.chosenActionId(), expected.chosenActionId());
          Assert.assertEquals(actual.chosenLegalSlot(), expected.chosenLegalSlot());
          Assert.assertEquals(actual.finalRank(), expected.finalRank());
          Assert.assertEquals(actual.valueTarget(), expected.valueTarget());
          Assert.assertEquals(actual.grpFinalRanksCode(), expected.grpFinalRanksCode());
          Assert.assertEquals(actual.grpFeatureSequence(), expected.grpFeatureSequence());
          var expectedRow = expected.input().sliceRows(0, 1);
          var actualRow = actual.input().sliceRows(0, 1);
          var expectedCategories = ShortBuffer.allocate(expectedRow.inputCategoricalElementCount());
          var actualCategories = ShortBuffer.allocate(actualRow.inputCategoricalElementCount());
          var expectedNumerics = FloatBuffer.allocate(expectedRow.inputNumericElementCount());
          var actualNumerics = FloatBuffer.allocate(actualRow.inputNumericElementCount());
          expectedRow.copyInputCategoriesTo(expectedCategories);
          actualRow.copyInputCategoriesTo(actualCategories);
          expectedRow.copyInputNumericsTo(expectedNumerics);
          actualRow.copyInputNumericsTo(actualNumerics);
          Assert.assertEquals(actualCategories.array(), expectedCategories.array());
          Assert.assertEquals(actualNumerics.array(), expectedNumerics.array());
        }
      }
    } finally {
      try (var files = Files.walk(directory)) {
        for (Path file : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);
      }
    }
  }

  private static String tenhouRecord() {
    var xml =
        new StringBuilder(
            "<mjloggm><INIT seed=\"0,0,0,0,0,135\" ten=\"250,250,250,250\" oya=\"0\"");
    for (int seat = 0; seat < 4; seat++) {
      var hand = new StringJoiner(",");
      for (int id = seat * 13; id < (seat + 1) * 13; id++) hand.add(Integer.toString(id));
      xml.append(" hai").append(seat).append("=\"").append(hand).append('"');
    }
    return xml.append(
            "/><T60/><D60/><RYUUKYOKU sc=\"250,0,250,0,250,0,250,0\""
                + " owari=\"250,0,250,0,250,0,250,0\"/></mjloggm>")
        .toString();
  }

  private static String mjaiRecord() {
    JsonObject start =
        JsonParser.parseString(
                "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"dora_marker\":\"C\",\"scores\":[25000,25000,25000,25000]}")
            .getAsJsonObject();
    JsonArray hands = new JsonArray();
    for (int seat = 0; seat < 4; seat++) hands.add(handTiles(seat, false));
    start.add("tehais", hands);
    return "{\"type\":\"start_game\"}\n"
        + start
        + "\n"
        + "{\"type\":\"tsumo\",\"actor\":0,\"pai\":\"7p\"}\n"
        + "{\"type\":\"dahai\",\"actor\":0,\"pai\":\"7p\",\"tsumogiri\":true}\n"
        + "{\"type\":\"ryukyoku\",\"deltas\":[0,0,0,0]}\n"
        + "{\"type\":\"end_kyoku\"}\n"
        + "{\"type\":\"end_game\",\"scores\":[25000,25000,25000,25000]}\n";
  }

  private static String mahjongSoulRecord() {
    JsonObject start =
        JsonParser.parseString(
                "{\"chang\":0,\"ju\":0,\"ben\":0,\"liqibang\":0,\"dora\":\"7z\",\"scores\":[25000,25000,25000,25000]}")
            .getAsJsonObject();
    for (int seat = 0; seat < 4; seat++) start.add("tiles" + seat, handTiles(seat, true));
    JsonArray records = new JsonArray();
    JsonObject first = new JsonObject();
    first.addProperty("name", ".lq.RecordNewRound");
    first.add("data", start);
    records.add(first);
    records.add(
        JsonParser.parseString(
            "{\"name\":\".lq.RecordDealTile\",\"data\":{\"seat\":0,\"tile\":\"7p\"}}"));
    records.add(
        JsonParser.parseString(
            "{\"name\":\".lq.RecordDiscardTile\",\"data\":{\"seat\":0,\"tile\":\"7p\",\"moqie\":true}}"));
    records.add(
        JsonParser.parseString(
            "{\"name\":\".lq.RecordNoTile\",\"data\":{\"scores\":[{\"delta_scores\":[0,0,0,0]}],\"gameend\":{\"scores\":[25000,25000,25000,25000]}}}"));
    return records.toString();
  }

  private static JsonArray handTiles(int seat, boolean soul) {
    JsonArray hand = new JsonArray();
    for (int id = seat * 13; id < (seat + 1) * 13; id++) {
      String tile = TileNotation.formatPhysicalTile(id);
      if (soul && Tile.isAka(id)) tile = "0" + tile.charAt(1);
      hand.add(tile);
    }
    return hand;
  }
}
