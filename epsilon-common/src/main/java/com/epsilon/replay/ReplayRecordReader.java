package com.epsilon.replay;

import com.epsilon.core.Tile;
import com.epsilon.replay.ReplayEvent.*;
import com.epsilon.replay.format.MahjongSoulReader;
import com.epsilon.replay.format.MjaiReader;
import com.epsilon.replay.format.TenhouXmlReader;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

/** 圧縮の展開、牌譜形式の判定、入力の整合性の検証をまとめて行う。 */
public final class ReplayRecordReader {
  /** 読み込み・再生処理の意味が変わった場合に、生成済みの教師データを再作成するための識別子。 */
  public static final String CONTRACT_REVISION = "four-player-replay-v1";

  private static final int MAX_BYTES = 16 * 1024 * 1024;
  private static final int MAX_EVENTS = 50_000;
  private static final Set<String> EXTENSIONS =
      Set.of("xml", "mjai", "mjson", "jsonl", "json", "bin", "dat", "pb", "lq", "majsoul");

  private ReplayRecordReader() {}

  public static ReplayRecord readRecord(Path file) throws IOException {
    try (InputStream input = Files.newInputStream(file)) {
      return readRecord(readBoundedBytes(input));
    } catch (IllegalArgumentException failure) {
      throw new IllegalArgumentException(
          "Could not read record " + file + ": " + failure.getMessage(), failure);
    }
  }

  public static ReplayRecord readRecord(byte[] bytes) {
    if (bytes.length == 0 || bytes.length > MAX_BYTES)
      throw new IllegalArgumentException("Record must contain between 1 byte and 16 MiB.");
    byte[] payload = decompressRecord(bytes);
    String text = new String(payload, StandardCharsets.UTF_8).strip();
    if (text.startsWith("\uFEFF")) text = text.substring(1).stripLeading();
    requireBoundedJsonNesting(text);
    RecordFormat format = detectRecordFormat(text);
    ReplayRecord record;
    try {
      record =
          switch (format) {
            case TENHOU -> TenhouXmlReader.readRecord(text);
            case MJAI -> MjaiReader.readRecord(text);
            case MAHJONG_SOUL -> MahjongSoulReader.readRecord(payload);
          };
    } catch (RuntimeException failure) {
      throw new IllegalArgumentException(
          "Could not decode " + format.sourceId() + " record: " + failure.getMessage(), failure);
    }
    requireReplayableEvents(record.events());
    return record;
  }

  /** 明示した単一ファイルは拡張子で捨てず、ディレクトリ探索だけを候補拡張子で絞る。 */
  public static List<Path> listRecordFiles(Path root, int maximumFiles) throws IOException {
    if (Files.isRegularFile(root)) return List.of(root);
    try (var paths = Files.walk(root)) {
      return paths
          .filter(Files::isRegularFile)
          .filter(ReplayRecordReader::hasRecordExtension)
          .sorted(Comparator.comparing(Path::toString))
          .limit(maximumFiles <= 0 ? Long.MAX_VALUE : maximumFiles)
          .toList();
    }
  }

  private static boolean hasRecordExtension(Path path) {
    String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
    if (name.endsWith(".gz")) name = name.substring(0, name.length() - 3);
    return EXTENSIONS.contains(name.substring(name.lastIndexOf('.') + 1));
  }

  private static void requireBoundedJsonNesting(String text) {
    if (!text.startsWith("{") && !text.startsWith("[")) return;
    int depth = 0;
    boolean quoted = false;
    boolean escaped = false;
    for (int index = 0; index < text.length(); index++) {
      char value = text.charAt(index);
      if (quoted) {
        if (escaped) escaped = false;
        else if (value == '\\') escaped = true;
        else if (value == '"') quoted = false;
      } else if (value == '"') quoted = true;
      else if (value == '{' || value == '[') {
        if (++depth > 64)
          throw new IllegalArgumentException("Record JSON nesting exceeds 64 levels.");
      } else if (value == '}' || value == ']') depth--;
    }
  }

  private static RecordFormat detectRecordFormat(String text) {
    if (text.isEmpty()) throw new IllegalArgumentException("Record contains no data.");
    if (text.charAt(0) == '<') return RecordFormat.TENHOU;
    if (text.charAt(0) != '{' && text.charAt(0) != '[') return RecordFormat.MAHJONG_SOUL;
    // 先頭オブジェクトの識別項目だけを読み、JSONL全体を木へ変換しない。
    try (JsonReader reader = new JsonReader(new StringReader(text))) {
      if (reader.peek() == JsonToken.BEGIN_ARRAY) reader.beginArray();
      if (reader.peek() != JsonToken.BEGIN_OBJECT)
        throw new IllegalArgumentException("Record must contain event objects.");
      reader.beginObject();
      while (reader.hasNext()) {
        String field = reader.nextName();
        switch (field) {
          case "formatVersion", "resultId", "rounds" ->
              throw new IllegalArgumentException("Analysis results are not source records.");
          case "type" -> {
            if (reader.peek() == JsonToken.STRING) return RecordFormat.MJAI;
            reader.skipValue();
          }
          case "records", "actions", "head", "data", "name" -> {
            return RecordFormat.MAHJONG_SOUL;
          }
          default -> reader.skipValue();
        }
      }
      throw new IllegalArgumentException("Unsupported record format.");
    } catch (IOException failure) {
      throw new IllegalArgumentException("Invalid record JSON syntax.", failure);
    }
  }

  private static byte[] decompressRecord(byte[] bytes) {
    if (bytes.length < 2 || (bytes[0] & 255) != 31 || (bytes[1] & 255) != 139) return bytes;
    try (InputStream input = new GZIPInputStream(new ByteArrayInputStream(bytes))) {
      return readBoundedBytes(input);
    } catch (IOException failure) {
      throw new IllegalArgumentException("Could not decompress record.", failure);
    }
  }

  private static byte[] readBoundedBytes(InputStream input) throws IOException {
    byte[] bytes = input.readNBytes(MAX_BYTES + 1);
    if (bytes.length > MAX_BYTES)
      throw new IllegalArgumentException("Record exceeds the 16 MiB size limit.");
    return bytes;
  }

  private static void requireReplayableEvents(List<ReplayEvent> events) {
    if (events.size() > MAX_EVENTS)
      throw new IllegalArgumentException("Record event limit exceeded.");
    boolean[] seenTiles = new boolean[136];
    boolean gameStarted = false;
    boolean gameEnded = false;
    boolean roundOpen = false;
    int rounds = 0;
    for (int index = 0; index < events.size(); index++) {
      ReplayEvent event = events.get(index);
      try {
        if (gameEnded)
          throw new IllegalArgumentException("Events after the end of a game are not allowed.");
        switch (event) {
          case StartGame ignored -> {
            if (gameStarted || rounds != 0)
              throw new IllegalArgumentException("Only one game is allowed per record.");
            gameStarted = true;
          }
          case StartKyoku start -> {
            if (!gameStarted)
              throw new IllegalArgumentException("Round starts before the game header.");
            if (roundOpen)
              throw new IllegalArgumentException(
                  "A new round starts before the previous round ended.");
            requireCompleteInitialHands(start);
            Arrays.fill(seenTiles, false);
            for (int[] hand : start.initialHandPhysicalTileIds())
              for (int tile : hand) markUniqueTile(seenTiles, tile);
            roundOpen = true;
            rounds++;
          }
          case Tsumo draw -> {
            if (!roundOpen)
              throw new IllegalArgumentException("Draw event occurs outside a round.");
            markUniqueTile(seenTiles, draw.physicalTileId());
          }
          case EndKyoku ignored -> {
            if (!roundOpen) throw new IllegalArgumentException("Round end occurs outside a round.");
            roundOpen = false;
          }
          case EndGame end -> {
            if (end.scores() != null && end.scores().length != 4)
              throw new IllegalArgumentException("Final scores must contain four players.");
            gameEnded = true;
          }
          default -> {
            if (!roundOpen)
              throw new IllegalArgumentException("Action event occurs outside a round.");
          }
        }
      } catch (IllegalArgumentException failure) {
        throw new IllegalArgumentException(
            "Record event " + index + ": " + failure.getMessage(), failure);
      }
    }
    if (rounds == 0) throw new IllegalArgumentException("Record is missing its initial round.");
  }

  private static void requireCompleteInitialHands(StartKyoku start) {
    if (start.initialHandPhysicalTileIds().length != 4
        || start.scores() == null
        || start.scores().length != 4)
      throw new IllegalArgumentException(
          "Initial hands and scores are required for all four players.");
    if (start.wind() < Tile.TON
        || start.wind() > Tile.PEI
        || start.oya() < 0
        || start.oya() > 3
        || start.kyoku() < 1
        || start.kyoku() > 4
        || start.honba() < 0
        || start.kyotaku() < 0) throw new IllegalArgumentException("Invalid round metadata.");
    for (int[] hand : start.initialHandPhysicalTileIds())
      if (hand.length != 13)
        throw new IllegalArgumentException("Every player must have 13 initial tiles.");
  }

  private static void markUniqueTile(boolean[] seen, int tile) {
    if (tile < 0 || tile >= seen.length)
      throw new IllegalArgumentException("Unknown or invalid physical tile ID.");
    if (seen[tile]) throw new IllegalArgumentException("Duplicate physical tile ID: " + tile);
    seen[tile] = true;
  }
}
