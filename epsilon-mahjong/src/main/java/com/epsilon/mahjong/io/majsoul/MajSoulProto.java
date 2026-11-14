package com.epsilon.mahjong.io.majsoul;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 雀魂の牌譜の protobuf を、{@link MajSoulReader} が読む JSON の形（head と records）に直す。読むのは、牌譜を取る API の応答を
 * そのまま保存したもの（ResGameRecord を包んだ lq.Wrapper）。head には ResGameRecord の head を入れ、records には、data に包んだ
 * GameDetailRecords の records の各記録を入れる。records が空なら、actions の各要素の result にある記録を入れる。
 *
 * <p>JSON に入れるのは MajSoulReader が読むフィールドだけで、キーは snake_case。protobuf に無いフィールドは、数・真偽・文字列なら既定値
 * （0・false・空の文字列）、繰り返しなら空の配列にし、メッセージなら入れない。繰り返しの数は、詰めた形でも詰めない形でも読む。知らない
 * 名前の記録は data を空の object にして入れる（MajSoulReader が名前で断る）。壊れた protobuf と、head か data の無い応答には
 * IllegalArgumentException を投げる。
 */
final class MajSoulProto {

  /** 応答を包む lq.Wrapper の name。 */
  private static final String RESPONSE = ".lq.ResGameRecord";

  /** ResGameRecord の data を包む lq.Wrapper の name。 */
  private static final String DETAILS = ".lq.GameDetailRecords";

  /** 記録の名前の前に付く名前空間。 */
  private static final String NAMESPACE = ".lq.";

  /** protobuf の wire type。 */
  private static final int VARINT = 0;

  private static final int I64 = 1;
  private static final int LEN = 2;
  private static final int I32 = 5;

  /** protobuf のフィールド番号の上限。 */
  private static final long MAX_FIELD = (1 << 29) - 1;

  /** lq.Wrapper のフィールド番号。 */
  private static final int WRAPPER_NAME = 1;

  private static final int WRAPPER_DATA = 2;

  /** ResGameRecord のフィールド番号。 */
  private static final int RESPONSE_HEAD = 3;

  private static final int RESPONSE_DATA = 4;

  /** GameDetailRecords のフィールド番号。 */
  private static final int DETAILS_RECORDS = 1;

  private static final int DETAILS_ACTIONS = 3;

  /** GameAction の result（記録を包んだ lq.Wrapper）のフィールド番号。 */
  private static final int ACTION_RESULT = 3;

  /** 応答の先頭のバイト列（lq.Wrapper の name のタグ・長さ・文字列）。 */
  private static final byte[] RESPONSE_PREFIX = prefix(RESPONSE);

  /** フィールドの値の型。 */
  private enum Kind {
    UINT(new JsonPrimitive(0)),
    INT(new JsonPrimitive(0)),
    BOOL(new JsonPrimitive(false)),
    STRING(new JsonPrimitive("")),
    MESSAGE(null);

    /** protobuf に無いときの値。メッセージなら null（JSON に入れない）。 */
    private final JsonPrimitive absent;

    Kind(JsonPrimitive absent) {
      this.absent = absent;
    }

    /** varint で書く型か（繰り返しなら詰めた形でも書ける）。 */
    boolean varint() {
      return this == UINT || this == INT || this == BOOL;
    }

    /** varint の値を、この型の値にする。uint32 は符号なし、int32 は下位 32 ビットを符号付きで読む。 */
    JsonPrimitive of(long value) {
      return switch (this) {
        case UINT -> new JsonPrimitive(value & 0xFFFF_FFFFL);
        case INT -> new JsonPrimitive((int) value);
        case BOOL -> new JsonPrimitive(value != 0);
        case STRING, MESSAGE -> throw new IllegalStateException(this + " is not written as a varint");
      };
    }
  }

  /**
   * メッセージの 1 つのフィールド。
   *
   * @param message 値のメッセージ。型が MESSAGE でなければ null
   */
  private record Field(int number, String key, Kind kind, boolean repeated, Message message) {}

  /**
   * 読むメッセージと、そのうち読むフィールド。定数の名前を UpperCamelCase にしたものが protobuf のメッセージの名前で、記録はその名前で
   * 引く。値のメッセージは、それを使うメッセージより前に書く。
   */
  private enum Message {
    LI_QI_SUCCESS(uint(1, "seat"), int32(2, "score"), uint(3, "liqibang"), bool(4, "failed")),
    FAN_INFO(uint(2, "val"), uint(3, "id")),
    HULE_INFO(
        string(3, "hu_tile"),
        uint(4, "seat"),
        bool(5, "zimo"),
        strings(9, "li_doras"),
        bool(10, "yiman"),
        uint(11, "count"),
        messages(12, "fans", FAN_INFO),
        uint(13, "fu")),
    NO_TILE_PLAYER_INFO(bool(3, "tingpai")),
    NO_TILE_SCORE_INFO(int32s(3, "delta_scores")),
    ACCOUNT_INFO(uint(2, "seat"), string(3, "nickname")),
    PLAYER_ITEM(uint(1, "seat"), int32(3, "part_point_1")),
    GAME_END_RESULT(messages(1, "players", PLAYER_ITEM)),
    RECORD_GAME(messages(11, "accounts", ACCOUNT_INFO), message(12, "result", GAME_END_RESULT)),
    RECORD_NEW_ROUND(
        uint(1, "chang"),
        uint(2, "ju"),
        uint(3, "ben"),
        string(4, "dora"),
        int32s(5, "scores"),
        uint(6, "liqibang"),
        strings(7, "tiles0"),
        strings(8, "tiles1"),
        strings(9, "tiles2"),
        strings(10, "tiles3"),
        strings(16, "doras")),
    RECORD_DEAL_TILE(uint(1, "seat"), string(2, "tile"), message(5, "liqi", LI_QI_SUCCESS), strings(6, "doras")),
    RECORD_DISCARD_TILE(
        uint(1, "seat"),
        string(2, "tile"),
        bool(3, "is_liqi"),
        bool(5, "moqie"),
        strings(8, "doras"),
        bool(9, "is_wliqi")),
    RECORD_CHI_PENG_GANG(
        uint(1, "seat"), uint(2, "type"), strings(3, "tiles"), uints(4, "froms"), message(5, "liqi", LI_QI_SUCCESS)),
    RECORD_AN_GANG_ADD_GANG(uint(1, "seat"), uint(2, "type"), string(3, "tiles"), strings(6, "doras")),
    RECORD_HULE(messages(1, "hules", HULE_INFO), int32s(3, "delta_scores")),
    RECORD_NO_TILE(
        bool(1, "liujumanguan"),
        messages(2, "players", NO_TILE_PLAYER_INFO),
        messages(3, "scores", NO_TILE_SCORE_INFO)),
    RECORD_LIU_JU(uint(1, "type"), message(5, "liqi", LI_QI_SUCCESS));

    private final List<Field> fields;

    Message(Field... fields) {
      this.fields = List.of(fields);
    }

    /** その番号の、読むフィールド。読まないフィールドなら null。 */
    Field field(int number) {
      for (Field field : fields) {
        if (field.number() == number) {
          return field;
        }
      }
      return null;
    }

    /** 名前空間を除いた名前のメッセージ。知らない名前なら null。 */
    static Message named(String name) {
      for (Message message : values()) {
        if (MajSoulReader.camel(message.name(), true).equals(name)) {
          return message;
        }
      }
      return null;
    }
  }

  private MajSoulProto() {}

  /** バイト列が、牌譜を取る API の応答（ResGameRecord を包んだ lq.Wrapper）か。先頭の name で見分ける。 */
  static boolean isResponse(byte[] bytes) {
    return Arrays.equals(
        bytes, 0, Math.min(bytes.length, RESPONSE_PREFIX.length), RESPONSE_PREFIX, 0, RESPONSE_PREFIX.length);
  }

  /** 応答を、MajSoulReader が読む JSON に直す。 */
  static JsonObject document(byte[] response) {
    JsonObject head = null;
    byte[] data = null;
    for (Wire wire = new Wire(unwrap(response, RESPONSE)); wire.next(); ) {
      switch (wire.number()) {
        case RESPONSE_HEAD -> head = decode(Message.RECORD_GAME, wire.bytes());
        case RESPONSE_DATA -> data = wire.bytes();
        default -> wire.skip();
      }
    }
    if (head == null || data == null) {
      throw new IllegalArgumentException("The Mahjong Soul record has no " + (head == null ? "head" : "data"));
    }
    JsonObject document = new JsonObject();
    document.add("head", head);
    document.add("records", records(unwrap(data, DETAILS)));
    return document;
  }

  /** GameDetailRecords の記録を、name と data の組の配列にする。records が空なら、actions の result から取る。 */
  private static JsonArray records(byte[] details) {
    List<byte[]> records = new ArrayList<>();
    List<byte[]> results = new ArrayList<>();
    for (Wire wire = new Wire(details); wire.next(); ) {
      switch (wire.number()) {
        case DETAILS_RECORDS -> records.add(wire.bytes());
        case DETAILS_ACTIONS -> {
          for (Wire action = new Wire(wire.bytes()); action.next(); ) {
            if (action.number() != ACTION_RESULT) {
              action.skip();
              continue;
            }
            byte[] result = action.bytes();
            if (result.length > 0) {
              results.add(result);
            }
          }
        }
        default -> wire.skip();
      }
    }
    JsonArray array = new JsonArray();
    for (byte[] record : records.isEmpty() ? results : records) {
      String name = null;
      byte[] data = new byte[0];
      for (Wire wire = new Wire(record); wire.next(); ) {
        switch (wire.number()) {
          case WRAPPER_NAME -> name = wire.string();
          case WRAPPER_DATA -> data = wire.bytes();
          default -> wire.skip();
        }
      }
      if (name == null) {
        throw new IllegalArgumentException("A Mahjong Soul record has no name");
      }
      Message message = name.startsWith(NAMESPACE) ? Message.named(name.substring(NAMESPACE.length())) : null;
      JsonObject entry = new JsonObject();
      entry.addProperty("name", name);
      entry.add("data", message == null ? new JsonObject() : decode(message, data));
      array.add(entry);
    }
    return array;
  }

  /** lq.Wrapper の name が expected であることを確かめて、data を返す。 */
  private static byte[] unwrap(byte[] wrapper, String expected) {
    String name = null;
    byte[] data = null;
    for (Wire wire = new Wire(wrapper); wire.next(); ) {
      switch (wire.number()) {
        case WRAPPER_NAME -> name = wire.string();
        case WRAPPER_DATA -> data = wire.bytes();
        default -> wire.skip();
      }
    }
    if (!expected.equals(name) || data == null) {
      throw new IllegalArgumentException("Expected a Mahjong Soul " + expected + ", got " + name);
    }
    return data;
  }

  /** メッセージを、読むフィールドだけを持つ JSON の object にする。 */
  private static JsonObject decode(Message message, byte[] bytes) {
    JsonObject object = new JsonObject();
    for (Field field : message.fields) {
      JsonElement absent = field.repeated() ? new JsonArray() : field.kind().absent;
      if (absent != null) {
        object.add(field.key(), absent);
      }
    }
    for (Wire wire = new Wire(bytes); wire.next(); ) {
      Field field = message.field(wire.number());
      if (field == null) {
        wire.skip();
      } else if (!field.repeated()) {
        object.add(field.key(), value(field, wire));
      } else if (field.kind().varint() && wire.type() == LEN) {
        JsonArray values = object.getAsJsonArray(field.key());
        for (Wire packed = new Wire(wire.bytes()); packed.more(); ) {
          values.add(field.kind().of(packed.varint()));
        }
      } else {
        object.getAsJsonArray(field.key()).add(value(field, wire));
      }
    }
    return object;
  }

  /** フィールドの値を 1 つ読む。 */
  private static JsonElement value(Field field, Wire wire) {
    return switch (field.kind()) {
      case STRING -> new JsonPrimitive(wire.string());
      case MESSAGE -> decode(field.message(), wire.bytes());
      case UINT, INT, BOOL -> field.kind().of(wire.scalar());
    };
  }

  private static byte[] prefix(String name) {
    byte[] text = name.getBytes(StandardCharsets.US_ASCII);
    byte[] prefix = new byte[text.length + 2];
    prefix[0] = WRAPPER_NAME << 3 | LEN;
    prefix[1] = (byte) text.length;
    System.arraycopy(text, 0, prefix, 2, text.length);
    return prefix;
  }

  private static Field uint(int number, String key) {
    return new Field(number, key, Kind.UINT, false, null);
  }

  private static Field uints(int number, String key) {
    return new Field(number, key, Kind.UINT, true, null);
  }

  private static Field int32(int number, String key) {
    return new Field(number, key, Kind.INT, false, null);
  }

  private static Field int32s(int number, String key) {
    return new Field(number, key, Kind.INT, true, null);
  }

  private static Field bool(int number, String key) {
    return new Field(number, key, Kind.BOOL, false, null);
  }

  private static Field string(int number, String key) {
    return new Field(number, key, Kind.STRING, false, null);
  }

  private static Field strings(int number, String key) {
    return new Field(number, key, Kind.STRING, true, null);
  }

  private static Field message(int number, String key, Message message) {
    return new Field(number, key, Kind.MESSAGE, false, message);
  }

  private static Field messages(int number, String key, Message message) {
    return new Field(number, key, Kind.MESSAGE, true, message);
  }

  /** protobuf のバイト列を先頭から読む。範囲を外れる長さと、壊れたタグ・varint には IllegalArgumentException を投げる。 */
  private static final class Wire {

    /** varint のバイト数の上限。 */
    private static final int VARINT_BYTES = 10;

    private final byte[] bytes;
    private int at;
    private int number;
    private int type;

    Wire(byte[] bytes) {
      this.bytes = bytes;
    }

    /** 次のフィールドのタグを読む。バイト列が尽きていれば false を返す。 */
    boolean next() {
      if (!more()) {
        return false;
      }
      long tag = varint();
      long field = tag >>> 3;
      if (field < 1 || field > MAX_FIELD) {
        throw new IllegalArgumentException("A Mahjong Soul protobuf has field number " + field);
      }
      number = (int) field;
      type = (int) (tag & 7);
      return true;
    }

    /** まだ読んでいないバイトがあるか。 */
    boolean more() {
      return at < bytes.length;
    }

    int number() {
      return number;
    }

    int type() {
      return type;
    }

    /** VARINT のフィールドの値。 */
    long scalar() {
      require(VARINT);
      return varint();
    }

    /** LEN のフィールドの中身。 */
    byte[] bytes() {
      require(LEN);
      int length = length();
      byte[] value = Arrays.copyOfRange(bytes, at, at + length);
      at += length;
      return value;
    }

    /** LEN のフィールドの中身を UTF-8 として読んだ文字列。 */
    String string() {
      return new String(bytes(), StandardCharsets.UTF_8);
    }

    /** 読まないフィールドを読み飛ばす。 */
    void skip() {
      switch (type) {
        case VARINT -> varint();
        case LEN -> advance(length());
        case I64 -> advance(Long.BYTES);
        case I32 -> advance(Integer.BYTES);
        default -> throw new IllegalArgumentException("A Mahjong Soul protobuf has wire type " + type);
      }
    }

    long varint() {
      long value = 0;
      for (int index = 0; index < VARINT_BYTES && more(); index++) {
        int next = bytes[at++] & 0xFF;
        value |= (long) (next & 0x7F) << 7 * index;
        if ((next & 0x80) == 0) {
          return value;
        }
      }
      throw new IllegalArgumentException("A Mahjong Soul protobuf has a broken varint");
    }

    private int length() {
      long length = varint();
      if (length > bytes.length - at) {
        throw new IllegalArgumentException("A Mahjong Soul protobuf field runs past its message");
      }
      return (int) length;
    }

    private void advance(int count) {
      if (count > bytes.length - at) {
        throw new IllegalArgumentException("A Mahjong Soul protobuf field runs past its message");
      }
      at += count;
    }

    private void require(int expected) {
      if (type != expected) {
        throw new IllegalArgumentException(
            "Mahjong Soul protobuf field " + number + " has wire type " + type + ", not " + expected);
      }
    }
  }
}
