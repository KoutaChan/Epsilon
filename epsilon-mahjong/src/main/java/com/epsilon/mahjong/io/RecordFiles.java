package com.epsilon.mahjong.io;

import com.epsilon.mahjong.io.majsoul.MajSoulReader;
import com.epsilon.mahjong.io.mjai.MjaiCodec;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.Set;
import java.util.zip.GZIPInputStream;

/**
 * 牌譜のファイルを読む入口。gzip なら展開する。展開した中身が雀魂の牌譜を取る API の応答の protobuf（ResGameRecord を包んだ
 * lq.Wrapper）なら、雀魂のリーダーに渡す。ほかは UTF-8 として読んで、前後の空白と先頭の BOM を除き、形式を見分けて、その形式の
 * リーダーに渡す。
 *
 * <p>文字の形式は先頭の文字で見分ける。'&lt;' なら天鳳の XML。'{' か '[' なら JSON で、最初の object に文字列の type があれば mjai とする
 * （1 行に 1 つのイベントか、イベントの配列）。JSON が配列でなく、先頭の object に head か records があれば、復号した雀魂の JSON とする。
 *
 * <p>解析結果の文書（先頭の object に formatVersion・resultId・rounds のどれかがあるもの）は読まない。base64 や、雀魂の records だけの
 * 配列、GameDetailRecords だけの protobuf など、ほかの形も読まない。
 */
public final class RecordFiles {

  /** ファイルの中身のバイト数の上限（16 MiB）。gzip を展開した中身にも同じ上限を使う。 */
  public static final int MAX_BYTES = 16 * 1024 * 1024;

  /** JSON の配列と object の入れ子の深さの上限。 */
  public static final int MAX_NESTING = 64;

  /** 半荘の全ての局を合わせたイベントの数の上限。 */
  public static final int MAX_EVENTS = 50_000;

  /** gzip の先頭の 2 バイト。 */
  private static final int GZIP_FIRST = 0x1f;

  private static final int GZIP_SECOND = 0x8b;

  private static final char BOM = '\uFEFF';

  /** 解析結果の文書だけが、先頭の object に持つキー。 */
  private static final Set<String> RESULT_FIELDS = Set.of("formatVersion", "resultId", "rounds");

  /** 復号した雀魂の JSON が、先頭の object に持つキー。 */
  private static final Set<String> MAJSOUL_FIELDS = Set.of("head", "records");

  private RecordFiles() {}

  /**
   * 牌譜のバイト列を、形式を見分けて読む。
   *
   * @param bytes 牌譜のファイルの中身（gzip でもよい）
   * @param id 牌譜の id
   * @return 読んだ半荘。形式は meta の source に入る
   * @throws IllegalArgumentException 空のとき、バイト数かイベントの数が上限を超えるとき、形式が分からないとき、解析結果の文書のとき、
   *     その形式として読めないとき
   */
  public static RecordedGame read(byte[] bytes, String id) {
    Objects.requireNonNull(id, "id");
    if (bytes.length == 0 || bytes.length > MAX_BYTES) {
      throw new IllegalArgumentException("A record holds between 1 byte and 16 MiB, got " + bytes.length + " bytes");
    }
    byte[] content = unpack(bytes);
    boolean response = MajSoulReader.isResponse(content);
    String text = response ? null : text(content);
    RecordedGame.Source source = response ? RecordedGame.Source.MAJSOUL : source(text);
    RecordedGame game;
    try {
      game =
          switch (source) {
            case TENHOU -> TenhouXml.read(text, id);
            case MJAI -> MjaiCodec.read(text, id);
            case MAJSOUL -> response ? MajSoulReader.read(content, id) : MajSoulReader.read(text, id);
          };
    } catch (IllegalArgumentException failure) {
      throw failure;
    } catch (RuntimeException failure) {
      throw new IllegalArgumentException("Could not read " + id + " as " + source + ": " + failure, failure);
    }
    int events = 0;
    for (RecordedRound round : game.rounds()) {
      events += round.events().size();
    }
    if (events > MAX_EVENTS) {
      throw new IllegalArgumentException(id + " has " + events + " events, more than " + MAX_EVENTS);
    }
    return game;
  }

  /**
   * gzip なら展開した中身を、そうでなければそのまま返す。展開した中身が上限を超えるか、gzip として壊れていれば
   * IllegalArgumentException を投げる。
   */
  private static byte[] unpack(byte[] bytes) {
    if (bytes.length < 2 || (bytes[0] & 0xff) != GZIP_FIRST || (bytes[1] & 0xff) != GZIP_SECOND) {
      return bytes;
    }
    try (InputStream in = new GZIPInputStream(new ByteArrayInputStream(bytes))) {
      byte[] unpacked = in.readNBytes(MAX_BYTES + 1);
      if (unpacked.length > MAX_BYTES) {
        throw new IllegalArgumentException("The record unpacks to more than 16 MiB");
      }
      return unpacked;
    } catch (IOException failure) {
      throw new IllegalArgumentException("The record is not a valid gzip stream", failure);
    }
  }

  /** UTF-8 として読み、前後の空白と先頭の BOM を除いた文字列を返す。 */
  private static String text(byte[] bytes) {
    String text = new String(bytes, StandardCharsets.UTF_8).strip();
    return !text.isEmpty() && text.charAt(0) == BOM ? text.substring(1).strip() : text;
  }

  /** 先頭の文字と、JSON なら最初の object のキーで形式を見分ける。見分けられないときと、解析結果の文書のときは IllegalArgumentException を投げる。 */
  private static RecordedGame.Source source(String text) {
    if (text.isEmpty()) {
      throw new IllegalArgumentException("The record is empty");
    }
    char first = text.charAt(0);
    if (first == '<') {
      return RecordedGame.Source.TENHOU;
    }
    if (first != '{' && first != '[') {
      throw new IllegalArgumentException("The record is neither a Tenhou log nor JSON");
    }
    requireNesting(text);
    try (JsonReader reader = new JsonReader(new StringReader(text))) {
      boolean array = reader.peek() == JsonToken.BEGIN_ARRAY;
      if (array) {
        reader.beginArray();
        if (reader.peek() != JsonToken.BEGIN_OBJECT) {
          throw new IllegalArgumentException("A JSON record array holds event objects");
        }
      }
      reader.beginObject();
      while (reader.hasNext()) {
        String name = reader.nextName();
        if (RESULT_FIELDS.contains(name)) {
          throw new IllegalArgumentException("An analysis result is not a record");
        }
        if (name.equals("type") && reader.peek() == JsonToken.STRING) {
          return RecordedGame.Source.MJAI;
        }
        if (!array && MAJSOUL_FIELDS.contains(name)) {
          return RecordedGame.Source.MAJSOUL;
        }
        reader.skipValue();
      }
      throw new IllegalArgumentException("The JSON record is neither mjai nor a decoded Mahjong Soul record");
    } catch (IOException | IllegalStateException failure) {
      throw new IllegalArgumentException("The record is not valid JSON", failure);
    }
  }

  /** JSON の配列と object の入れ子が上限より深ければ IllegalArgumentException を投げる。文字列の中の括弧は数えない。 */
  private static void requireNesting(String text) {
    int depth = 0;
    boolean quoted = false;
    boolean escaped = false;
    for (int index = 0; index < text.length(); index++) {
      char c = text.charAt(index);
      if (quoted) {
        if (escaped) {
          escaped = false;
        } else if (c == '\\') {
          escaped = true;
        } else if (c == '"') {
          quoted = false;
        }
      } else if (c == '"') {
        quoted = true;
      } else if (c == '{' || c == '[') {
        if (++depth > MAX_NESTING) {
          throw new IllegalArgumentException("The record nests JSON deeper than " + MAX_NESTING + " levels");
        }
      } else if (c == '}' || c == ']') {
        depth--;
      }
    }
  }
}
