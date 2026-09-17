package com.epsilon.replay;

import com.epsilon.core.Tile;
import com.google.gson.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.Comparator;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌譜の形式判定と圧縮展開、終局点の根拠、不完全な対局や不正入力の扱いを検証する。 */
public final class ReplayRecordReaderTest {
  @Test
  public void gzipAndJsonEventArraysUseTheSameInputContract() throws Exception {
    byte[] plain = completeMjai().getBytes(StandardCharsets.UTF_8);
    var output = new ByteArrayOutputStream();
    try (var gzip = new GZIPOutputStream(output)) {
      gzip.write(plain);
    }
    ReplayRecord decoded = ReplayRecordReader.readRecord(output.toByteArray());
    Assert.assertEquals(decoded.metadata().format(), RecordFormat.MJAI);
    Assert.assertEquals(decoded.completion(), RecordCompletion.COMPLETE);
    Assert.assertEquals(decoded.metadata().finalScores(), new int[] {25000, 25000, 25000, 25000});
    Assert.assertEquals(
        decoded.events().stream().filter(ReplayEvent.StartGame.class::isInstance).count(), 1L);
    Assert.assertEquals(
        decoded.events().size(), ReplayRecordReader.readRecord(plain).events().size());
  }

  @Test
  public void incompleteMatchesCannotSupplyFinalRankTeachers() {
    JsonArray events = JsonParser.parseString(completeMjai()).getAsJsonArray();
    events.remove(events.size() - 1);
    ReplayRecord partial =
        ReplayRecordReader.readRecord(events.toString().getBytes(StandardCharsets.UTF_8));
    Assert.assertEquals(partial.completion(), RecordCompletion.INCOMPLETE);
    Assert.assertNull(partial.metadata().finalScores());
    Assert.expectThrows(IllegalArgumentException.class, partial::requireCompletedMatch);
  }

  @Test
  public void explicitEndWithoutScoreEvidenceCannotBecomeRankTeachers() {
    JsonArray events = JsonParser.parseString(completeMjai()).getAsJsonArray();
    events.get(events.size() - 1).getAsJsonObject().remove("scores");
    ReplayRecord record =
        ReplayRecordReader.readRecord(events.toString().getBytes(StandardCharsets.UTF_8));
    Assert.assertEquals(record.completion(), RecordCompletion.COMPLETE);
    Assert.assertFalse(record.hasKnownFinalScores());
    Assert.expectThrows(IllegalArgumentException.class, record::requireCompletedMatch);
  }

  @Test
  public void completeSettlementCanSupplyFinalScoresWithoutEndGameScores() {
    JsonArray events = JsonParser.parseString(completeMjai()).getAsJsonArray();
    events.remove(events.size() - 1);
    events.add(JsonParser.parseString("{\"type\":\"ryukyoku\",\"deltas\":[1000,-1000,0,0]}"));
    events.add(JsonParser.parseString("{\"type\":\"end_kyoku\"}"));
    events.add(JsonParser.parseString("{\"type\":\"end_game\"}"));
    ReplayRecord record =
        ReplayRecordReader.readRecord(events.toString().getBytes(StandardCharsets.UTF_8));
    Assert.assertTrue(record.hasKnownFinalScores());
    Assert.assertSame(record.requireCompletedMatch(), record);
  }

  @Test
  public void readerRejectsAnalysisResultsAndRepeatedGames() {
    Assert.expectThrows(
        IllegalArgumentException.class,
        () ->
            ReplayRecordReader.readRecord(
                "{\"formatVersion\":2,\"rounds\":[]}".getBytes(StandardCharsets.UTF_8)));
    JsonArray events = JsonParser.parseString(completeMjai()).getAsJsonArray();
    events.add(JsonParser.parseString("{\"type\":\"start_game\"}"));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> ReplayRecordReader.readRecord(events.toString().getBytes(StandardCharsets.UTF_8)));
  }

  @Test
  public void malformedJsonAndWrongFieldTypesAreReportedAsInputErrors() {
    String malformed = completeMjai().substring(0, completeMjai().length() - 1);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> ReplayRecordReader.readRecord(malformed.getBytes(StandardCharsets.UTF_8)));
    JsonArray events = JsonParser.parseString(completeMjai()).getAsJsonArray();
    events.get(1).getAsJsonObject().add("scores", new JsonObject());
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> ReplayRecordReader.readRecord(events.toString().getBytes(StandardCharsets.UTF_8)));
  }

  @Test
  public void fileDiscoveryIncludesAllThreeFormatsAndCompressedJsonLines() throws Exception {
    Path directory = Files.createTempDirectory("replay-inputs-");
    try {
      for (String name :
          new String[] {
            "a.xml.gz", "b.jsonl.gz", "c.mjai", "d.json", "e.pb", "f.majsoul", "notes.txt"
          }) Files.writeString(directory.resolve(name), "fixture");
      var files = ReplayRecordReader.listRecordFiles(directory, 0);
      Assert.assertEquals(
          files.stream().map(path -> path.getFileName().toString()).toList(),
          java.util.List.of("a.xml.gz", "b.jsonl.gz", "c.mjai", "d.json", "e.pb", "f.majsoul"));
      Assert.assertEquals(ReplayRecordReader.listRecordFiles(directory, 2).size(), 2);
      Assert.assertEquals(
          ReplayRecordReader.listRecordFiles(directory.resolve("notes.txt"), 0).size(), 1);
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> ReplayRecordReader.readRecord(directory.resolve("notes.txt")));
    } finally {
      try (var files = Files.walk(directory)) {
        for (Path file : files.sorted(Comparator.reverseOrder()).toList()) Files.delete(file);
      }
    }
  }

  private static String completeMjai() {
    JsonArray events = new JsonArray();
    events.add(
        JsonParser.parseString("{\"type\":\"start_game\",\"names\":[\"A\",\"B\",\"C\",\"D\"]}"));
    JsonObject start =
        JsonParser.parseString(
                "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"dora_marker\":\"C\",\"scores\":[25000,25000,25000,25000]}")
            .getAsJsonObject();
    JsonArray hands = new JsonArray();
    for (int seat = 0; seat < 4; seat++) {
      JsonArray hand = new JsonArray();
      for (int id = seat * 13; id < (seat + 1) * 13; id++) {
        int type = Tile.typeOf(id);
        hand.add((type % 9 + 1) + "" + "mps".charAt(type / 9) + (Tile.isAka(id) ? "r" : ""));
      }
      hands.add(hand);
    }
    start.add("tehais", hands);
    events.add(start);
    events.add(
        JsonParser.parseString("{\"type\":\"end_game\",\"scores\":[25000,25000,25000,25000]}"));
    return events.toString();
  }
}
