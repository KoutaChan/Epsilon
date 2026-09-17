package com.epsilon.replay.format;

import com.epsilon.replay.format.MahjongSoulRecord.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Liqi の保存形式フィールドを雀魂固有の型へ直接復号する。 */
final class MahjongSoulBinary {
  private MahjongSoulBinary() {}

  static void readSourceRecords(byte[] bytes, MahjongSoulReader output) {
    Proto root = new Proto(bytes);
    String name = root.string(1);
    if (name.startsWith(".lq.")) {
      readMessage(name, root.bytes(2), output);
      return;
    }
    if (root.has(4) && new Proto(root.bytes(4)).string(1).equals(".lq.GameDetailRecords"))
      readResponse(root, output);
    else readDetails(root, output);
  }

  static void readMessage(String name, byte[] data, MahjongSoulReader output) {
    switch (name.replace(".lq.", "")) {
      case "GameDetailRecords" -> readDetails(new Proto(data), output);
      case "ResGameRecord" -> readResponse(new Proto(data), output);
      default -> throw new IllegalArgumentException("Unsupported Mahjong Soul message: " + name);
    }
  }

  private static void readResponse(Proto response, MahjongSoulReader output) {
    if (response.has(1) && new Proto(response.bytes(1)).integer(1) != 0)
      throw new IllegalArgumentException("Mahjong Soul API returned a record retrieval error.");
    if (response.has(3)) {
      Proto head = new Proto(response.bytes(3));
      for (byte[] value : head.byteList(11)) {
        Proto account = new Proto(value);
        output.readPlayerName(account.integer(2), account.string(3));
      }
      if (head.has(12)) {
        List<byte[]> players = new Proto(head.bytes(12)).byteList(1);
        if (!players.isEmpty()) {
          if (players.size() != 4)
            throw new IllegalArgumentException("Final scores must contain four players.");
          int[] scores = new int[4];
          boolean[] seen = new boolean[4];
          for (byte[] value : players) {
            Proto player = new Proto(value);
            int seat = player.integer(1);
            if (seat < 0 || seat >= 4 || seen[seat])
              throw new IllegalArgumentException("Invalid final-score seat.");
            seen[seat] = true;
            scores[seat] = player.integer(3);
          }
          output.readFinalScores(scores, "majsoul.head.result.part_point_1");
        }
      }
    }
    Proto record = new Proto(response.bytes(4));
    if (!record.string(1).equals(".lq.GameDetailRecords"))
      throw new IllegalArgumentException("Mahjong Soul response is missing GameDetailRecords.");
    readDetails(new Proto(record.bytes(2)), output);
  }

  private static void readDetails(Proto details, MahjongSoulReader output) {
    List<byte[]> records = details.byteList(1);
    if (!records.isEmpty()) {
      for (byte[] record : records) readRecordWrapper(record, output);
      return;
    }
    for (byte[] bytes : details.byteList(3)) {
      Proto action = new Proto(bytes);
      if (action.has(3) && action.bytes(3).length > 0) readRecordWrapper(action.bytes(3), output);
    }
  }

  static void readRecordWrapper(byte[] bytes, MahjongSoulReader output) {
    Proto wrapper = new Proto(bytes);
    readRecord(wrapper.string(1), wrapper.bytes(2), output);
  }

  static void readRecord(String name, byte[] bytes, MahjongSoulReader output) {
    Proto record = new Proto(bytes);
    output.acceptRecord(
        switch (name.replace(".lq.", "")) {
          case "RecordNewRound" -> readRound(record);
          case "RecordDealTile" ->
              new Draw(record.integer(1), record.string(2), record.strings(6), readRiichi(record));
          case "RecordDiscardTile" ->
              new Discard(
                  record.integer(1),
                  record.string(2),
                  record.integer(3) != 0 || record.integer(9) != 0,
                  record.integer(5) != 0,
                  record.strings(8));
          case "RecordChiPengGang" ->
              new Call(
                  record.integer(1),
                  record.integer(2),
                  record.strings(3),
                  record.integers(4),
                  readRiichi(record));
          case "RecordAnGangAddGang" ->
              new Kan(record.integer(1), record.integer(2), record.string(3), record.strings(6));
          case "RecordHule" -> readWin(record);
          case "RecordNoTile" -> readExhaustiveDraw(record);
          case "RecordLiuJu" ->
              new AbortiveDraw(record.integer(1), readGameEnd(record, 2), readRiichi(record));
          default -> throw new IllegalArgumentException("Unsupported Mahjong Soul event: " + name);
        });
  }

  private static Round readRound(Proto record) {
    List<List<String>> hands = new ArrayList<>(4);
    for (int seat = 0; seat < 4; seat++) hands.add(record.strings(7 + seat));
    List<String> indicators = record.strings(16);
    if (indicators.isEmpty()) indicators = List.of(record.string(4));
    return new Round(
        record.integer(1),
        record.integer(2),
        record.integer(3),
        record.integer(6),
        record.integers(5),
        hands,
        indicators);
  }

  private static Riichi readRiichi(Proto record) {
    if (!record.has(5)) return null;
    Proto riichi = new Proto(record.bytes(5));
    return new Riichi(riichi.integer(1), riichi.integer(4) != 0);
  }

  private static Win readWin(Proto record) {
    List<Winner> winners = new ArrayList<>();
    for (byte[] value : record.byteList(1)) {
      Proto winner = new Proto(value);
      List<Fan> fans = null;
      if (winner.has(12)) {
        fans = new ArrayList<>();
        for (byte[] fanValue : winner.byteList(12)) {
          Proto fan = new Proto(fanValue);
          fans.add(new Fan(fan.integer(3), fan.integer(2)));
        }
      }
      winners.add(
          new Winner(
              winner.integer(4),
              winner.string(3),
              winner.integer(5) != 0,
              winner.integer(7) != 0,
              winner.integer(10) != 0,
              recordedInt(winner, 11),
              recordedInt(winner, 13),
              recordedInt(winner, 15),
              recordedInt(winner, 16),
              recordedInt(winner, 17),
              fans,
              winner.has(9) ? winner.strings(9) : null));
    }
    return new Win(
        winners,
        record.integers(3),
        record.has(5) ? record.integers(5) : null,
        readGameEnd(record, 6));
  }

  private static ExhaustiveDraw readExhaustiveDraw(Proto record) {
    int[] deltas = new int[4];
    for (byte[] value : record.byteList(3)) {
      int[] change = new Proto(value).integers(3);
      MahjongSoulReader.requireFourScores(change);
      for (int seat = 0; seat < 4; seat++) deltas[seat] = Math.addExact(deltas[seat], change[seat]);
    }
    return new ExhaustiveDraw(deltas, new GameEnd(record.integer(4) != 0, null));
  }

  private static GameEnd readGameEnd(Proto record, int field) {
    if (!record.has(field)) return GameEnd.NONE;
    int[] scores = new Proto(record.bytes(field)).integers(1);
    return scores.length == 0 ? GameEnd.NONE : new GameEnd(true, scores);
  }

  private static Integer recordedInt(Proto record, int field) {
    return record.has(field) ? record.integer(field) : null;
  }

  /** 必要なProtocol Buffersのフィールド形式を読み、長さ、可変長整数、メッセージの入れ子の深さを検証する。 */
  private static final class Proto {
    private final Map<Integer, List<Object>> fields = new HashMap<>();

    Proto(byte[] data) {
      int[] offset = {0};
      int fieldCount = 0;
      while (offset[0] < data.length) {
        if (++fieldCount > 100000)
          throw new IllegalArgumentException("Protobuf field count exceeds the limit.");
        long tag = varint(data, offset);
        long number = tag >>> 3;
        if (number < 1 || number > 536870911)
          throw new IllegalArgumentException("Invalid protobuf field number.");
        int field = (int) number, wire = (int) (tag & 7);
        Object value;
        if (wire == 0) value = varint(data, offset);
        else if (wire == 2) {
          long length = varint(data, offset);
          if (length < 0 || length > data.length - offset[0])
            throw new IllegalArgumentException("Truncated protobuf message.");
          value = Arrays.copyOfRange(data, offset[0], offset[0] + (int) length);
          offset[0] += (int) length;
        } else if (wire == 1 || wire == 5) {
          int size = wire == 1 ? 8 : 4;
          if (data.length - offset[0] < size)
            throw new IllegalArgumentException("Truncated protobuf message.");
          offset[0] += size;
          continue;
        } else throw new IllegalArgumentException("Unsupported protobuf wire type.");
        fields.computeIfAbsent(field, ignored -> new ArrayList<>()).add(value);
      }
    }

    boolean has(int field) {
      return fields.containsKey(field);
    }

    byte[] bytes(int field) {
      List<Object> list = fields.get(field);
      if (list == null) return new byte[0];
      Object value = list.getLast();
      if (!(value instanceof byte[] b))
        throw new IllegalArgumentException("Invalid protobuf field type.");
      return b;
    }

    String string(int field) {
      List<Object> list = fields.get(field);
      if (list == null) return "";
      if (!(list.getLast() instanceof byte[] bytes))
        throw new IllegalArgumentException("Invalid protobuf string field type.");
      return new String(bytes, StandardCharsets.UTF_8);
    }

    int integer(int field) {
      List<Object> list = fields.get(field);
      if (list == null) return 0;
      if (!(list.getLast() instanceof Long n))
        throw new IllegalArgumentException("Invalid protobuf integer type.");
      return n.intValue();
    }

    List<byte[]> byteList(int field) {
      List<byte[]> out = new ArrayList<>();
      for (Object value : fields.getOrDefault(field, List.of())) {
        if (!(value instanceof byte[] b))
          throw new IllegalArgumentException("Invalid protobuf array type.");
        out.add(b);
      }
      return out;
    }

    List<String> strings(int field) {
      return byteList(field).stream().map(b -> new String(b, StandardCharsets.UTF_8)).toList();
    }

    int[] integers(int field) {
      List<Integer> out = new ArrayList<>();
      for (Object value : fields.getOrDefault(field, List.of())) {
        if (value instanceof Long n) out.add(n.intValue());
        else {
          byte[] packed = (byte[]) value;
          int[] offset = {0};
          while (offset[0] < packed.length) out.add((int) varint(packed, offset));
        }
      }
      return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private static long varint(byte[] data, int[] offset) {
      long value = 0;
      for (int shift = 0; shift < 64; shift += 7) {
        if (offset[0] >= data.length)
          throw new IllegalArgumentException("Truncated protobuf varint.");
        int next = data[offset[0]++] & 255;
        if (shift == 63 && next > 1)
          throw new IllegalArgumentException("Protobuf varint is too long.");
        value |= (long) (next & 127) << shift;
        if ((next & 128) == 0) return value;
      }
      throw new IllegalArgumentException("Protobuf varint is too long.");
    }
  }
}
