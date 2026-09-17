package com.epsilon.replay.format;

import com.epsilon.core.Tile;
import com.epsilon.replay.*;
import com.epsilon.replay.ReplayEvent.*;
import com.google.gson.*;
import java.util.ArrayList;
import java.util.List;

/** mjai JSONL / イベント配列を一度読み、物理 ID を確定してから直接イベントを生成する。 */
public final class MjaiReader {
  private final List<ReplayEvent> events = new ArrayList<>();
  private final List<String> names = new ArrayList<>(List.of("", "", "", ""));
  private final TileIdentityAllocator tiles = new TileIdentityAllocator();
  private boolean explicitStart;
  private boolean ended;
  private int[] finalScores;
  private String ruleDescription;

  private MjaiReader() {
    events.add(new StartGame());
  }

  public static ReplayRecord readRecord(String text) {
    MjaiReader reader = new MjaiReader();
    String input = text.strip();
    if (input.startsWith("[")) {
      for (JsonElement value : JsonParser.parseString(input).getAsJsonArray())
        reader.readEvent(value.getAsJsonObject());
    } else {
      input
          .lines()
          .filter(line -> !line.isBlank())
          .forEach(line -> reader.readEvent(JsonParser.parseString(line).getAsJsonObject()));
    }
    if (reader.events.size() == 1) throw new IllegalArgumentException("Record contains no events.");
    return new ReplayRecord(
        reader.events,
        new RecordMetadata(
            RecordFormat.MJAI,
            reader.names,
            reader.finalScores,
            reader.finalScores == null ? "unavailable" : "mjai.end_game",
            reader.ruleDescription),
        null,
        reader.ended ? RecordCompletion.COMPLETE : RecordCompletion.INCOMPLETE);
  }

  private void readEvent(JsonObject event) {
    if (ended) throw new IllegalArgumentException("Events after end_game are not allowed.");
    String type = readRequiredString(event, "type");
    if (type.equals("start_game")) {
      if (explicitStart || events.size() != 1)
        throw new IllegalArgumentException("A record may contain only one start_game.");
      explicitStart = true;
      if (hasValue(event, "names")) {
        JsonArray values = event.getAsJsonArray("names");
        if (values.size() != 4)
          throw new IllegalArgumentException("Four player names are required.");
        for (int seat = 0; seat < 4; seat++)
          if (!values.get(seat).isJsonNull()) names.set(seat, values.get(seat).getAsString());
      }
      if (hasValue(event, "rule")) ruleDescription = readRequiredString(event, "rule");
      return;
    }
    ReplayEvent parsed =
        switch (type) {
          case "start_kyoku" -> startRound(event);
          case "tsumo" -> {
            int actor = readSeatIndex(event, "actor"),
                tile = readRepresentativeTileId(event, "pai");
            yield new Tsumo(
                actor, tiles.allocateDrawTile(actor, Tile.typeOf(tile), Tile.isAka(tile)));
          }
          case "dahai" -> {
            int actor = readSeatIndex(event, "actor"),
                tile = readRepresentativeTileId(event, "pai");
            boolean tsumogiri = readRequiredBoolean(event, "tsumogiri");
            yield new Dahai(
                actor,
                tiles.removeDiscardTile(actor, Tile.typeOf(tile), Tile.isAka(tile), tsumogiri),
                tsumogiri);
          }
          case "chi", "pon", "daiminkan" -> {
            int actor = readSeatIndex(event, "actor"), target = readSeatIndex(event, "target");
            int representative = readRepresentativeTileId(event, "pai");
            int called =
                tiles.takeCalledTile(
                    target, Tile.typeOf(representative), Tile.isAka(representative));
            int[] consumed =
                consumeRecordedMeldTiles(actor, event, type.equals("daiminkan") ? 3 : 2);
            yield switch (type) {
              case "chi" -> new Chi(actor, target, called, consumed);
              case "pon" -> new Pon(actor, target, called, consumed);
              default -> new Daiminkan(actor, target, called, consumed);
            };
          }
          case "ankan" -> {
            int actor = readSeatIndex(event, "actor");
            yield new Ankan(actor, consumeRecordedMeldTiles(actor, event, 4));
          }
          case "kakan" -> {
            int actor = readSeatIndex(event, "actor");
            int[] added =
                tiles.consumeMeldTiles(actor, new int[] {readRepresentativeTileId(event, "pai")});
            yield new Kakan(actor, added[0]);
          }
          case "reach" -> new Reach(readSeatIndex(event, "actor"));
          case "reach_accepted" -> new ReachAccepted(readSeatIndex(event, "actor"));
          case "dora" -> {
            int marker = readRepresentativeTileId(event, "dora_marker");
            tiles.reserveIndicator(Tile.typeOf(marker), Tile.isAka(marker));
            yield new Dora(Tile.typeOf(marker), Tile.isAka(marker));
          }
          case "hora" ->
              new Hora(
                  readSeatIndex(event, "actor"),
                  readSeatIndex(event, "target"),
                  hasValue(event, "pai")
                      ? TileNotation.parseTileType(readRequiredString(event, "pai"))
                      : -1,
                  readOptionalScores(event, "deltas"),
                  parseWinDetails(event));
          case "ryukyoku" ->
              new Ryukyoku(
                  readOptionalScores(event, "deltas"),
                  decodeDrawReason(
                      hasValue(event, "reason") ? readRequiredString(event, "reason") : null));
          case "end_kyoku" -> new EndKyoku();
          case "end_game" -> {
            finalScores = readOptionalScores(event, "scores");
            ended = true;
            yield new EndGame(finalScores);
          }
          case "none" -> new None(readSeatIndex(event, "actor"));
          default -> throw new IllegalArgumentException("Unknown mjai event type: " + type);
        };
    events.add(parsed);
  }

  private StartKyoku startRound(JsonObject event) {
    int marker = readRepresentativeTileId(event, "dora_marker");
    tiles.startRound(Tile.typeOf(marker), Tile.isAka(marker));
    JsonArray hands = readRequiredField(event, "tehais").getAsJsonArray();
    if (hands.size() != 4) throw new IllegalArgumentException("Four initial hands are required.");
    int[][] initial = new int[4][];
    for (int seat = 0; seat < 4; seat++) {
      JsonArray hand = hands.get(seat).getAsJsonArray();
      if (hand.size() != 13)
        throw new IllegalArgumentException("Each initial hand must contain 13 tiles.");
      initial[seat] = new int[13];
      for (int i = 0; i < 13; i++) {
        int tile = TileNotation.representativeTileId(hand.get(i).getAsString());
        initial[seat][i] = tiles.allocateInitialTile(seat, Tile.typeOf(tile), Tile.isAka(tile));
      }
    }
    int wind = TileNotation.parseTileType(readRequiredString(event, "bakaze"));
    if (wind < 27 || wind > 30)
      throw new IllegalArgumentException("Round wind must be E, S, W, or N.");
    return new StartKyoku(
        wind,
        readRequiredInteger(event, "kyoku"),
        readRequiredInteger(event, "honba"),
        readRequiredInteger(event, "kyotaku"),
        readSeatIndex(event, "oya"),
        Tile.typeOf(marker),
        initial,
        readFourScores(readRequiredField(event, "scores").getAsJsonArray()),
        Tile.isAka(marker));
  }

  private int[] consumeRecordedMeldTiles(int actor, JsonObject event, int count) {
    JsonArray values = readRequiredField(event, "consumed").getAsJsonArray();
    if (values.size() != count) throw new IllegalArgumentException("Invalid consumed tile count.");
    int[] consumed = new int[count];
    for (int i = 0; i < count; i++)
      consumed[i] = TileNotation.representativeTileId(values.get(i).getAsString());
    return tiles.consumeMeldTiles(actor, consumed);
  }

  private static int readRepresentativeTileId(JsonObject event, String key) {
    return TileNotation.representativeTileId(readRequiredString(event, key));
  }

  private static String decodeDrawReason(String value) {
    if (value == null) return null;
    return switch (value) {
      case "fanpai", "exhaustive_draw" -> "EXHAUSTIVE_DRAW";
      case "kyushu_kyuhai", "kyushukyuhai", "yao9" -> "NINE_TERMINALS";
      case "suchareach", "reach4" -> "FOUR_RIICHI";
      case "sanchaho", "ron3" -> "THREE_RON";
      case "sukaikan", "kan4" -> "FOUR_KANS";
      case "sufonrenta", "kaze4" -> "FOUR_WINDS";
      default -> "MJAI_" + value;
    };
  }

  private static JsonElement readRequiredField(JsonObject event, String key) {
    JsonElement value = event.get(key);
    if (value == null || value.isJsonNull())
      throw new IllegalArgumentException("Missing required mjai field: " + key);
    return value;
  }

  private static String readRequiredString(JsonObject event, String key) {
    JsonElement value = readRequiredField(event, key);
    if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isString())
      throw new IllegalArgumentException("Expected string mjai field: " + key);
    return value.getAsString();
  }

  private static boolean readRequiredBoolean(JsonObject event, String key) {
    JsonElement value = readRequiredField(event, key);
    if (!value.isJsonPrimitive() || !value.getAsJsonPrimitive().isBoolean())
      throw new IllegalArgumentException("Expected boolean mjai field: " + key);
    return value.getAsBoolean();
  }

  private static int readRequiredInteger(JsonObject event, String key) {
    return readRequiredField(event, key).getAsBigDecimal().intValueExact();
  }

  private static int readSeatIndex(JsonObject event, String key) {
    int seat = readRequiredInteger(event, key);
    if (seat < 0 || seat > 3) throw new IllegalArgumentException("Seat must be between 0 and 3.");
    return seat;
  }

  private static boolean hasValue(JsonObject event, String key) {
    return event.has(key) && !event.get(key).isJsonNull();
  }

  private static int[] readOptionalScores(JsonObject event, String key) {
    return hasValue(event, key) ? readFourScores(event.getAsJsonArray(key)) : null;
  }

  private static int[] readFourScores(JsonArray values) {
    if (values.size() != 4) throw new IllegalArgumentException("Four scores are required.");
    int[] scores = new int[4];
    for (int i = 0; i < 4; i++) scores[i] = values.get(i).getAsBigDecimal().intValueExact();
    return scores;
  }

  private static WinDetails parseWinDetails(JsonObject event) {
    if (event.has("details") && !event.get("details").isJsonNull()) {
      JsonObject details = event.getAsJsonObject("details");
      List<WinDetails.Yaku> yaku = null;
      if (details.has("yaku") && !details.get("yaku").isJsonNull()) {
        yaku = new ArrayList<>();
        for (var value : details.getAsJsonArray("yaku")) {
          JsonObject item = value.getAsJsonObject();
          int han = item.get("han").getAsInt();
          int yakuman = item.get("yakuman").getAsInt();
          if (han > 0 || yakuman > 0)
            yaku.add(new WinDetails.Yaku(item.get("code").getAsString(), han, yakuman));
        }
      }
      return new WinDetails(
          readOptionalInteger(details, "han"),
          readOptionalInteger(details, "fu"),
          readOptionalInteger(details, "points"),
          readOptionalInteger(details, "yakuman"),
          yaku,
          parseIndicatorTiles(details, "uraIndicators"));
    }
    Integer fan = readOptionalInteger(event, "fan");
    Integer fu = readOptionalInteger(event, "fu");
    Integer points = readOptionalInteger(event, "hora_points");
    List<String> uraIndicators = parseIndicatorTiles(event, "uradora_markers");
    List<WinDetails.Yaku> yaku = null;
    Integer yakuman = fan != null ? fan / 100 : null;
    Integer han = fan != null ? (fan >= 100 ? 0 : fan) : null;
    if (event.has("yakus") && !event.get("yakus").isJsonNull()) {
      yaku = new ArrayList<>();
      int recordedHan = 0;
      int recordedYakuman = 0;
      for (var value : event.getAsJsonArray("yakus")) {
        JsonArray pair = value.getAsJsonArray();
        int count = pair.get(1).getAsInt();
        if (count > 0) {
          int yakuYakuman = count / 100;
          int yakuHan = yakuYakuman > 0 ? 0 : count;
          yaku.add(
              new WinDetails.Yaku(decodeYakuCode(pair.get(0).getAsString()), yakuHan, yakuYakuman));
          recordedHan += yakuHan;
          recordedYakuman += yakuYakuman;
        }
      }
      if (han == null) han = recordedHan;
      if (yakuman == null) yakuman = recordedYakuman;
    }
    if (han == null && fu == null && points == null && yaku == null && uraIndicators == null)
      return null;
    return new WinDetails(han, fu, points, yakuman, yaku, uraIndicators);
  }

  private static Integer readOptionalInteger(JsonObject object, String key) {
    return object.has(key) && !object.get(key).isJsonNull() ? object.get(key).getAsInt() : null;
  }

  private static List<String> parseIndicatorTiles(JsonObject object, String key) {
    if (!object.has(key) || object.get(key).isJsonNull()) return null;
    List<String> tiles = new ArrayList<>();
    for (var value : object.getAsJsonArray(key)) {
      String tile = value.getAsString();
      TileNotation.parseTileType(tile);
      tiles.add(tile);
    }
    return tiles;
  }

  private static String decodeYakuCode(String name) {
    return switch (name) {
      case "reach" -> "RIICHI";
      case "double_reach" -> "DOUBLE_RIICHI";
      case "ippatsu" -> "IPPATSU";
      case "menzenchin_tsumoho" -> "MENZEN_TSUMO";
      case "pinfu" -> "PINFU";
      case "tanyaochu" -> "TANYAO";
      case "ipeko" -> "IPEIKOU";
      case "sangenpai" -> "YAKUHAI_SANGEN";
      case "bakaze" -> "YAKUHAI_ROUND";
      case "jikaze" -> "YAKUHAI_SEAT";
      case "rinshankaiho" -> "RINSHAN";
      case "chankan" -> "CHANKAN";
      case "haiteiraoyue" -> "HAITEI";
      case "hoteiraoyui" -> "HOUTEI";
      case "sanshokudojun" -> "SANSHOKU";
      case "ikkitsukan" -> "ITTSU";
      case "honchantaiyao" -> "CHANTA";
      case "chitoitsu" -> "CHIITOITSU";
      case "toitoiho" -> "TOITOI";
      case "sananko" -> "SANANKOU";
      case "honroto" -> "HONROUTOU";
      case "sanshokudoko" -> "SANSHOKU_DOUKOU";
      case "sankantsu" -> "SANKANTSU";
      case "shosangen" -> "SHOUSANGEN";
      case "honiso" -> "HONITSU";
      case "junchantaiyao" -> "JUNCHAN";
      case "ryanpeko" -> "RYANPEIKOU";
      case "chiniso" -> "CHINITSU";
      case "tenho" -> "TENHOU";
      case "chiho" -> "CHIIHOU";
      case "kokushimuso" -> "KOKUSHI";
      case "daisangen" -> "DAISANGEN";
      case "suanko" -> "SUUANKOU";
      case "tsuiso" -> "TSUUIISOU";
      case "ryuiso" -> "RYUUIISOU";
      case "chinroto" -> "CHINROUTOU";
      case "daisushi" -> "DAISUUSHII";
      case "shosushi" -> "SHOUSUUSHII";
      case "sukantsu" -> "SUUKANTSU";
      case "churenpoton" -> "CHUUREN";
      case "dora" -> "DORA";
      case "uradora" -> "URADORA";
      case "akadora" -> "AKADORA";
      default -> "MJAI_" + name;
    };
  }
}
