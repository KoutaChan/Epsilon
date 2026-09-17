package com.epsilon.replay.format;

import com.epsilon.replay.format.MahjongSoulRecord.*;
import com.google.gson.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** 雀魂 JSON のフィールドを固有レコードへ復号し、型付きの処理へ渡す。 */
final class MahjongSoulJson {
  private MahjongSoulJson() {}

  static void readSourceRecords(byte[] payload, MahjongSoulReader output) {
    String text = new String(payload, StandardCharsets.UTF_8).strip();
    if (text.startsWith("\uFEFF")) text = text.substring(1).stripLeading();
    readDocument(JsonParser.parseString(text), output, 0);
  }

  private static void readDocument(JsonElement value, MahjongSoulReader output, int depth) {
    if (depth > 8)
      throw new IllegalArgumentException("Mahjong Soul wrapper nesting limit exceeded.");
    if (value.isJsonArray()) {
      for (JsonElement record : value.getAsJsonArray()) readRecord(record, output);
      return;
    }
    JsonObject document = value.getAsJsonObject();
    if (document.has("formatVersion") || document.has("rounds"))
      throw new IllegalArgumentException("Analysis results cannot be imported as records.");
    JsonElement errorValue = document.get("error");
    if (errorValue != null && !errorValue.isJsonNull()) {
      Integer code = readOptionalInteger(errorValue.getAsJsonObject(), "code");
      if (code != null && code != 0)
        throw new IllegalArgumentException("Mahjong Soul API returned a record retrieval error.");
    }
    if (document.has("head")) readHead(document.getAsJsonObject("head"), output);
    if (document.has("records") && !document.getAsJsonArray("records").isEmpty()) {
      for (JsonElement record : document.getAsJsonArray("records")) readRecord(record, output);
      return;
    }
    if (document.has("actions")) {
      for (JsonElement action : document.getAsJsonArray("actions")) {
        JsonElement result = action.getAsJsonObject().get("result");
        if (result != null
            && !result.isJsonNull()
            && !(result.isJsonPrimitive() && result.getAsString().isEmpty()))
          readRecord(result, output);
      }
      return;
    }
    JsonElement data = document.get("data");
    if (data != null && !data.isJsonNull()) {
      if (data.isJsonPrimitive()) {
        byte[] bytes = Base64.getDecoder().decode(data.getAsString());
        if (document.has("name"))
          MahjongSoulBinary.readMessage(readRequiredString(document, "name"), bytes, output);
        else MahjongSoulBinary.readSourceRecords(bytes, output);
      } else readDocument(data, output, depth + 1);
      return;
    }
    throw new IllegalArgumentException("Mahjong Soul document contains no record data.");
  }

  private static void readHead(JsonObject head, MahjongSoulReader output) {
    for (JsonElement value : readOptionalArray(head, "accounts")) {
      JsonObject account = value.getAsJsonObject();
      output.readPlayerName(
          readRequiredInteger(account, "seat"), readRequiredString(account, "nickname"));
    }
    JsonElement result = head.get("result");
    if (result == null || result.isJsonNull()) return;
    JsonArray players = readOptionalArray(result.getAsJsonObject(), "players");
    if (players.isEmpty()) return;
    if (players.size() != 4)
      throw new IllegalArgumentException("Final scores must contain four players.");
    int[] scores = new int[4];
    boolean[] seen = new boolean[4];
    for (JsonElement value : players) {
      JsonObject player = value.getAsJsonObject();
      int seat = readRequiredInteger(player, "seat");
      if (seat < 0 || seat >= 4 || seen[seat])
        throw new IllegalArgumentException("Invalid final-score seat.");
      seen[seat] = true;
      scores[seat] = readRequiredInteger(player, "part_point_1");
    }
    output.readFinalScores(scores, "majsoul.head.result.part_point_1");
  }

  private static void readRecord(JsonElement value, MahjongSoulReader output) {
    if (value.isJsonPrimitive()) {
      MahjongSoulBinary.readRecordWrapper(Base64.getDecoder().decode(value.getAsString()), output);
      return;
    }
    JsonObject wrapper = value.getAsJsonObject();
    String name = readRequiredString(wrapper, "name");
    JsonElement data = wrapper.has("data") ? wrapper.get("data") : wrapper;
    if (data.isJsonPrimitive()) {
      MahjongSoulBinary.readRecord(name, Base64.getDecoder().decode(data.getAsString()), output);
      return;
    }
    JsonObject record = data.getAsJsonObject();
    output.acceptRecord(
        switch (name.replace(".lq.", "")) {
          case "RecordNewRound" -> readRound(record);
          case "RecordDealTile" ->
              new Draw(
                  readRequiredInteger(record, "seat"),
                  readRequiredString(record, "tile"),
                  readStrings(readOptionalArray(record, "doras")),
                  readRiichi(record));
          case "RecordDiscardTile" ->
              new Discard(
                  readRequiredInteger(record, "seat"),
                  readRequiredString(record, "tile"),
                  readOptionalBoolean(record, "is_liqi") || readOptionalBoolean(record, "is_wliqi"),
                  readRequiredField(record, "moqie").getAsBoolean(),
                  readStrings(readOptionalArray(record, "doras")));
          case "RecordChiPengGang" ->
              new Call(
                  readRequiredInteger(record, "seat"),
                  readRequiredInteger(record, "type"),
                  readStrings(readRequiredField(record, "tiles").getAsJsonArray()),
                  readIntegers(readRequiredField(record, "froms").getAsJsonArray()),
                  readRiichi(record));
          case "RecordAnGangAddGang" ->
              new Kan(
                  readRequiredInteger(record, "seat"),
                  readRequiredInteger(record, "type"),
                  readRequiredString(record, "tiles"),
                  readStrings(readOptionalArray(record, "doras")));
          case "RecordHule" -> readWin(record);
          case "RecordNoTile" -> readExhaustiveDraw(record);
          case "RecordLiuJu" ->
              new AbortiveDraw(
                  readRequiredInteger(record, "type"), readGameEnd(record), readRiichi(record));
          default -> throw new IllegalArgumentException("Unsupported Mahjong Soul event: " + name);
        });
  }

  private static Round readRound(JsonObject record) {
    List<List<String>> hands = new ArrayList<>(4);
    for (int seat = 0; seat < 4; seat++)
      hands.add(readStrings(readRequiredField(record, "tiles" + seat).getAsJsonArray()));
    List<String> indicators = readStrings(readOptionalArray(record, "doras"));
    if (indicators.isEmpty()) indicators = List.of(readRequiredString(record, "dora"));
    return new Round(
        readRequiredInteger(record, "chang"),
        readRequiredInteger(record, "ju"),
        readRequiredInteger(record, "ben"),
        readRequiredInteger(record, "liqibang"),
        readIntegers(readRequiredField(record, "scores").getAsJsonArray()),
        hands,
        indicators);
  }

  private static Riichi readRiichi(JsonObject record) {
    JsonElement value = record.get("liqi");
    if (value == null || value.isJsonNull()) return null;
    JsonObject riichi = value.getAsJsonObject();
    return new Riichi(readRequiredInteger(riichi, "seat"), readOptionalBoolean(riichi, "failed"));
  }

  private static Win readWin(JsonObject record) {
    List<Winner> winners = new ArrayList<>();
    for (JsonElement value : readRequiredField(record, "hules").getAsJsonArray()) {
      JsonObject winner = value.getAsJsonObject();
      JsonElement fansValue = winner.get("fans");
      List<Fan> fans = null;
      if (fansValue != null && !fansValue.isJsonNull()) {
        fans = new ArrayList<>();
        for (JsonElement fanValue : fansValue.getAsJsonArray()) {
          JsonObject fan = fanValue.getAsJsonObject();
          fans.add(new Fan(readRequiredInteger(fan, "id"), readRequiredInteger(fan, "val")));
        }
      }
      JsonElement ura = winner.get(fieldName(winner, "li_doras"));
      JsonElement riichi = winner.get("liqi");
      winners.add(
          new Winner(
              readRequiredInteger(winner, "seat"),
              readRequiredString(winner, "hu_tile"),
              readRequiredField(winner, "zimo").getAsBoolean(),
              riichi == null || riichi.isJsonNull() ? null : riichi.getAsBoolean(),
              readOptionalBoolean(winner, "yiman"),
              readOptionalInteger(winner, "count"),
              readOptionalInteger(winner, "fu"),
              readOptionalInteger(winner, "point_rong"),
              readOptionalInteger(winner, "point_zimo_qin"),
              readOptionalInteger(winner, "point_zimo_xian"),
              fans,
              ura == null || ura.isJsonNull() ? null : readStrings(ura.getAsJsonArray())));
    }
    JsonArray scores = readOptionalArray(record, "scores");
    return new Win(
        winners,
        readIntegers(readRequiredField(record, "delta_scores").getAsJsonArray()),
        scores.isEmpty() ? null : readIntegers(scores),
        readGameEnd(record));
  }

  private static ExhaustiveDraw readExhaustiveDraw(JsonObject record) {
    int[] deltas = new int[4];
    for (JsonElement score : readOptionalArray(record, "scores")) {
      int[] change = readIntegers(readOptionalArray(score.getAsJsonObject(), "delta_scores"));
      // 復号済み JSON でも、点数移動がなければ差分は省略または空配列になる。
      if (change.length == 0) continue;
      MahjongSoulReader.requireFourScores(change);
      for (int seat = 0; seat < 4; seat++) deltas[seat] = Math.addExact(deltas[seat], change[seat]);
    }
    return new ExhaustiveDraw(deltas, readGameEnd(record));
  }

  private static GameEnd readGameEnd(JsonObject record) {
    JsonElement value = record.get("gameend");
    if (value == null || value.isJsonNull()) return GameEnd.NONE;
    if (value.isJsonPrimitive()) return new GameEnd(value.getAsBoolean(), null);
    JsonArray scores = readOptionalArray(value.getAsJsonObject(), "scores");
    // 復号済み JSON の空 GameEnd は protobuf の既定メッセージで、終局を意味しない。
    return scores.isEmpty() ? GameEnd.NONE : new GameEnd(true, readIntegers(scores));
  }

  private static JsonElement readRequiredField(JsonObject object, String name) {
    JsonElement value = object.get(fieldName(object, name));
    if (value == null || value.isJsonNull())
      throw new IllegalArgumentException("Missing Mahjong Soul field: " + name);
    return value;
  }

  private static int readRequiredInteger(JsonObject object, String name) {
    return readRequiredField(object, name).getAsBigDecimal().intValueExact();
  }

  private static String readRequiredString(JsonObject object, String name) {
    return readRequiredField(object, name).getAsString();
  }

  private static Integer readOptionalInteger(JsonObject object, String name) {
    JsonElement value = object.get(fieldName(object, name));
    return value == null || value.isJsonNull() ? null : value.getAsBigDecimal().intValueExact();
  }

  private static boolean readOptionalBoolean(JsonObject object, String name) {
    JsonElement value = object.get(fieldName(object, name));
    return value != null && !value.isJsonNull() && value.getAsBoolean();
  }

  private static JsonArray readOptionalArray(JsonObject object, String name) {
    JsonElement value = object.get(fieldName(object, name));
    return value == null || value.isJsonNull() ? new JsonArray() : value.getAsJsonArray();
  }

  private static int[] readIntegers(JsonArray values) {
    int[] result = new int[values.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = values.get(i).getAsBigDecimal().intValueExact();
    return result;
  }

  private static List<String> readStrings(JsonArray values) {
    List<String> result = new ArrayList<>(values.size());
    for (JsonElement value : values) result.add(value.getAsString());
    return result;
  }

  private static String fieldName(JsonObject object, String name) {
    if (object.has(name) || name.indexOf('_') < 0) return name;
    StringBuilder result = new StringBuilder();
    boolean upper = false;
    for (char c : name.toCharArray()) {
      if (c == '_') upper = true;
      else {
        result.append(upper ? Character.toUpperCase(c) : c);
        upper = false;
      }
    }
    return result.toString();
  }
}
