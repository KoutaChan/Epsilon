package com.epsilon.training;

import com.epsilon.config.settings.TenhouLogSettings;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.Tile;
import com.epsilon.engine.GameRecorder;
import com.epsilon.engine.PointDelta;
import com.epsilon.engine.RoundResult;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 記録処理の公開イベントから、点数・赤牌・局スナップショット・ビューア出力の契約を検証する。 */
public class TenhouLogWriterTest {
  @Test
  public void completedRoundKeepsScoresAndTilesWhenInputArraysAndTheNextRoundChange() {
    String[] names = {"最初の名前"};
    TenhouLogWriter writer =
        new TenhouLogWriter(names, new TenhouLogSettings("https://viewer.example/#json="));
    names[0] = "変更後";
    int[] starting = {25000, 25000, 25000, 25000};
    writer.startRound(0, 1, 2, starting);
    starting[0] = -1;
    int[] tiles = new int[Tile.NUM_TILE_TYPES];
    tiles[Tile.M5] = 2;
    writer.recordHaipai(0, tiles, AkaTileMask.includeTile(AkaTileMask.EMPTY, Tile.M5));
    Arrays.fill(tiles, 0);
    int[] dora = {16};
    writer.recordDoraIndicators(dora);
    dora[0] = 0;
    writer.recordDraw(0, Tile.M5, true);
    writer.recordDahai(0, Tile.M5, true, true, true);
    writer.recordRiichiAccepted(0);
    writer.recordRoundResult(
        new RoundResult.ExhaustiveDraw(1, 0, new PointDelta(3000, -1000, -1000, -1000)));
    int[] settled = {27000, 24000, 24000, 24000};
    writer.endRound(new GameRecorder.RoundEnd(settled, new int[] {52}));
    settled[0] = -2;

    writer.startRound(1, 0, 1, new int[] {27000, 24000, 24000, 24000});
    writer.recordRoundResult(
        new RoundResult.AbortiveDraw(RoundResult.AbortiveDrawReason.NINE_TERMINALS_AND_HONORS));
    writer.endRound(new GameRecorder.RoundEnd(new int[] {28000, 24000, 24000, 24000}, new int[0]));

    JsonObject first = JsonParser.parseString(writer.toRoundJson(0)).getAsJsonObject();
    JsonArray round = first.getAsJsonArray("log").get(0).getAsJsonArray();
    Assert.assertEquals(first.getAsJsonArray("name").get(0).getAsString(), "最初の名前");
    Assert.assertEquals(round.get(0), JsonParser.parseString("[0,1,2]"));
    Assert.assertEquals(round.get(1), JsonParser.parseString("[25000,25000,25000,25000]"));
    Assert.assertEquals(round.get(2), JsonParser.parseString("[51]"));
    Assert.assertEquals(round.get(3), JsonParser.parseString("[52]"));
    Assert.assertEquals(round.get(4), JsonParser.parseString("[15,51]"));
    Assert.assertEquals(round.get(5), JsonParser.parseString("[51]"));
    Assert.assertEquals(round.get(6), JsonParser.parseString("[\"r60\"]"));
    Assert.assertEquals(round.get(16), JsonParser.parseString("[\"流局\",[3000,-1000,-1000,-1000]]"));
    Assert.assertEquals(
        first.get("sc"), JsonParser.parseString("[27000,0,24000,0,24000,0,24000,0]"));
    JsonObject full = JsonParser.parseString(writer.toJson()).getAsJsonObject();
    Assert.assertEquals(full.getAsJsonArray("log").size(), 2);
    Assert.assertEquals(
        full.getAsJsonArray("log").get(1).getAsJsonArray().get(16),
        JsonParser.parseString("[\"九種九牌\"]"));
    Assert.assertEquals(full.getAsJsonArray("sc").get(0).getAsInt(), 28000);
  }

  @DataProvider
  public Object[][] ponSources() {
    return new Object[][] {
      {3, true, "p511515"},
      {3, false, "p155115"},
      {2, true, "15p5115"},
      {2, false, "51p1515"},
      {1, true, "1515p51"},
      {1, false, "5115p15"}
    };
  }

  @Test(dataProvider = "ponSources")
  public void ponKeepsTheCalledRedTileSeparateFromTheConsumedRedTile(
      int sourceOffset, boolean calledRed, String expectedMeld) {
    TenhouLogWriter writer = writer();
    writer.startRound(0, 0, 0, new int[] {25000, 25000, 25000, 25000});
    writer.recordPon(0, Tile.M5, sourceOffset, calledRed, !calledRed);
    finishUnchangedRound(writer);
    JsonArray round = round(writer);
    Assert.assertEquals(round.get(5).getAsJsonArray().get(0).getAsString(), expectedMeld);
  }

  @Test
  public void chiAndKanKeepRedIdentityAndTheTenhouCallMarkers() {
    TenhouLogWriter writer = writer();
    writer.startRound(0, 0, 0, new int[] {25000, 25000, 25000, 25000});
    writer.recordChi(0, new int[] {Tile.M4, Tile.M5, Tile.M6}, Tile.M4, false, true);
    writer.recordPon(1, Tile.M5, 2, false, true);
    writer.recordKakan(1, Tile.M5, false);
    writer.recordDaiminkan(2, Tile.M5, 1, true, false);
    writer.recordAnkan(3, Tile.S5, true);
    finishUnchangedRound(writer);

    JsonArray round = round(writer);
    Assert.assertEquals(round.get(5), JsonParser.parseString("[\"c145116\"]"));
    Assert.assertEquals(round.get(8), JsonParser.parseString("[\"51p1515\"]"));
    Assert.assertEquals(round.get(9), JsonParser.parseString("[\"51k151515\"]"));
    Assert.assertEquals(round.get(11), JsonParser.parseString("[\"151515m51\"]"));
    Assert.assertEquals(round.get(12), JsonParser.parseString("[0]"));
    Assert.assertEquals(round.get(15), JsonParser.parseString("[\"533535a35\"]"));
  }

  @Test
  public void customViewerWritesAllUrlsAboveTheirMatchingJsonLines() {
    String prefix = "https://custom.example/view#json=";
    TenhouLogWriter writer =
        new TenhouLogWriter(new String[] {"赤 五"}, new TenhouLogSettings(prefix));
    writer.startRound(0, 0, 0, new int[] {25000, 25000, 25000, 25000});
    finishUnchangedRound(writer);

    String[] lines = writer.toViewerUrlAndJsonLines().split("\\R", -1);
    Assert.assertEquals(lines.length, 5);
    Assert.assertEquals(lines[2], "");
    Assert.assertEquals(lines[0], writer.toViewerUrl());
    Assert.assertEquals(lines[1], writer.toRoundViewerUrl(0));
    for (int index = 0; index < 2; index++) {
      Assert.assertTrue(lines[index].startsWith(prefix));
      Assert.assertTrue(lines[index].contains("%20"));
      Assert.assertFalse(lines[index].contains("+"));
      Assert.assertEquals(
          URLDecoder.decode(lines[index].substring(prefix.length()), StandardCharsets.UTF_8),
          lines[index + 3]);
    }
    Assert.assertEquals(
        TenhouLogWriter.toViewerUrl("{\"name\":\"A B\"}", "viewer:"),
        "viewer:%7B%22name%22%3A%22A%20B%22%7D");
  }

  private static TenhouLogWriter writer() {
    return new TenhouLogWriter(
        new String[0], new TenhouLogSettings("https://viewer.example/#json="));
  }

  private static void finishUnchangedRound(TenhouLogWriter writer) {
    writer.recordRoundResult(
        new RoundResult.AbortiveDraw(RoundResult.AbortiveDrawReason.NINE_TERMINALS_AND_HONORS));
    writer.endRound(new GameRecorder.RoundEnd(new int[] {25000, 25000, 25000, 25000}, new int[0]));
  }

  private static JsonArray round(TenhouLogWriter writer) {
    return JsonParser.parseString(writer.toRoundJson(0))
        .getAsJsonObject()
        .getAsJsonArray("log")
        .get(0)
        .getAsJsonArray();
  }
}
