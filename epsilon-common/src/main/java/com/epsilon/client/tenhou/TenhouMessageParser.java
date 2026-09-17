package com.epsilon.client.tenhou;

import com.epsilon.core.Tile;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 天鳳のWebSocketプロトコルのメッセージを解析する。
 *
 * <p>JSON と XML を同じイベント変換経路で処理し、受信メッセージごとに一つのイベントを返す。
 */
public final class TenhouMessageParser {

  private static final Logger log = LoggerFactory.getLogger(TenhouMessageParser.class);
  private static final Pattern ATTR_PATTERN = Pattern.compile("(\\w+)=\"([^\"]*)\"");
  private static final String DRAW_TAGS = "TUVW";
  private static final String DAHAI_TAGS = "DEFG";
  private static final String TSUMOGIRI_DAHAI_TAGS = "defg";

  private TenhouMessageParser() {}

  /**
   * 一つの受信メッセージを一つのイベントへ変換する。
   *
   * <p>既知タグの必須値が欠けている場合も、壊れた局状態を作らず {@link TenhouEvent.Unknown} を返す。
   *
   * @param message tenhou.net から受信した JSON または XML 文字列
   * @return 対応するプロトコルイベント。空または不正な入力では {@link TenhouEvent.Unknown}
   */
  public static TenhouEvent parse(String message) {
    if (message == null || message.isBlank()) {
      return new TenhouEvent.Unknown(message);
    }

    String trimmed = message.trim();
    try {
      MessageFields fields;
      if (trimmed.startsWith("{")) {
        fields = jsonFields(trimmed);
      } else if (trimmed.startsWith("<")) {
        fields = xmlFields(trimmed);
      } else {
        throw malformed("unsupported format");
      }
      return parse(fields);
    } catch (MalformedMessage e) {
      log.debug("Malformed tenhou message ({}): {}", e.getMessage(), trimmed);
      return new TenhouEvent.Unknown(message);
    }
  }

  private static TenhouEvent parse(MessageFields fields) {
    String tag = fields.tag();
    TenhouEvent tileEvent = parseTileEvent(tag, fields);
    if (tileEvent != null) {
      return tileEvent;
    }

    return switch (tag) {
      case "HELO" -> {
        String auth = fields.value("auth");
        yield new TenhouEvent.Helo(auth == null ? "" : auth);
      }
      case "GO" ->
          new TenhouEvent.Go(fields.intOrDefault("type", 0), fields.intOrDefault("lobby", 0));
      case "UN" ->
          new TenhouEvent.Un(
              new String[] {
                fields.orEmpty("n0"),
                fields.orEmpty("n1"),
                fields.orEmpty("n2"),
                fields.orEmpty("n3")
              });
      case "INIT" -> parseInit(fields);
      case "N" -> parseNaki(fields);
      case "REACH" -> parseReach(fields);
      case "DORA" -> new TenhouEvent.NewDora(Tile.typeOf(requiredPhysicalTileId(fields, "hai")));
      case "AGARI" -> parseAgari(fields);
      case "RYUUKYOKU" ->
          new TenhouEvent.Ryukyoku(optionalSettledScores(fields), fields.has("owari"));
      case "OWARI" -> new TenhouEvent.EndGame();
      case "TAIKYOKU", "SAIKAI" -> new TenhouEvent.GameStart();
      case "LN" -> new TenhouEvent.KeepAlive();
      case "SHUFFLE", "CHAT", "DATE", "KANSEN", "PROF", "BYE" -> new TenhouEvent.Ignored(tag);
      default -> {
        log.debug("Unknown tenhou tag: {}", tag);
        yield new TenhouEvent.Unknown(fields.raw());
      }
    };
  }

  private static TenhouEvent parseTileEvent(String tag, MessageFields fields) {
    char first = tag.charAt(0);
    String suffix = tag.substring(1);
    int drawPlayer = DRAW_TAGS.indexOf(first);
    if (drawPlayer >= 0 && (suffix.isEmpty() || isDigits(suffix))) {
      if (drawPlayer == 0) {
        int physicalTileId = parsePhysicalTileId(suffix, "draw tag");
        Integer optionalPromptMask = fields.optionalInt("t");
        int promptMask = optionalPromptMask == null ? 0 : optionalPromptMask;
        requirePromptMask(promptMask);
        return new TenhouEvent.Draw(
            drawPlayer, physicalTileId, TenhouEvent.ActionPrompt.draw(promptMask));
      }
      if (!suffix.isEmpty()) {
        parsePhysicalTileId(suffix, "draw tag");
      }
      return new TenhouEvent.Draw(drawPlayer, -1, TenhouEvent.ActionPrompt.none());
    }

    int dahaiPlayer = DAHAI_TAGS.indexOf(first);
    boolean tsumogiri = false;
    if (dahaiPlayer < 0) {
      dahaiPlayer = TSUMOGIRI_DAHAI_TAGS.indexOf(first);
      tsumogiri = dahaiPlayer >= 0;
    }
    if (dahaiPlayer < 0 || !isDigits(suffix)) {
      return null;
    }

    int physicalTileId = parsePhysicalTileId(suffix, "dahai tag");
    return new TenhouEvent.Dahai(
        dahaiPlayer, physicalTileId, tsumogiri, responsePrompt(fields.optionalInt("t")));
  }

  private static TenhouEvent.Init parseInit(MessageFields fields) {
    int[] roundSeed = requiredLength(fields, "seed", 6);
    int[] initialScores = pointScores(requiredLength(fields, "ten", 4));
    int dealer = fields.requiredSeat("oya");
    int[] initialHandPhysicalTileIds =
        fields.has("hai") ? requiredLength(fields, "hai", 13) : requiredLength(fields, "hai0", 13);
    for (int physicalTileId : initialHandPhysicalTileIds) {
      requirePhysicalTileId(physicalTileId, "hai");
    }
    requirePhysicalTileId(roundSeed[5], "seed dora indicator");
    return new TenhouEvent.Init(
        roundSeed[0],
        roundSeed[1],
        roundSeed[2],
        dealer,
        Tile.typeOf(roundSeed[5]),
        initialScores,
        initialHandPhysicalTileIds);
  }

  private static TenhouEvent.Naki parseNaki(MessageFields fields) {
    int player = fields.requiredSeat("who");
    try {
      TenhouMeldDecoder.DecodedMeld decodedMeld = TenhouMeldDecoder.decode(fields.requiredInt("m"));
      return new TenhouEvent.Naki(player, decodedMeld, responsePrompt(fields.optionalInt("t")));
    } catch (IllegalArgumentException e) {
      throw malformed("invalid meld");
    }
  }

  private static TenhouEvent.GameEvent parseReach(MessageFields fields) {
    int player = fields.requiredSeat("who");
    int step = fields.requiredInt("step");
    if (step == 1) {
      return new TenhouEvent.ReachDeclared(player);
    }
    if (step == 2) {
      int[] updatedScores =
          fields.has("ten") ? pointScores(requiredLength(fields, "ten", 4)) : null;
      return new TenhouEvent.ReachAccepted(player, updatedScores);
    }
    throw malformed("REACH step must be 1 or 2");
  }

  private static TenhouEvent.Agari parseAgari(MessageFields fields) {
    int winner = fields.requiredSeat("who");
    Integer fromWho = fields.optionalSeat("fromWho");
    return new TenhouEvent.Agari(
        winner,
        fromWho == null ? winner : fromWho,
        optionalSettledScores(fields),
        fields.has("owari"));
  }

  private static TenhouEvent.ActionPrompt responsePrompt(Integer promptMask) {
    if (promptMask == null) {
      return TenhouEvent.ActionPrompt.none();
    }
    requirePromptMask(promptMask);
    return TenhouEvent.ActionPrompt.response(promptMask);
  }

  private static int requiredPhysicalTileId(MessageFields fields, String name) {
    return requirePhysicalTileId(fields.requiredInt(name), name);
  }

  private static int parsePhysicalTileId(String value, String name) {
    return requirePhysicalTileId(parseInt(value, name), name);
  }

  private static int requirePhysicalTileId(int physicalTileId, String name) {
    if (physicalTileId < 0 || physicalTileId >= 136) {
      throw malformed(name + " is not a physical tile id");
    }
    return physicalTileId;
  }

  private static int[] requiredLength(MessageFields fields, String name, int length) {
    int[] values = fields.requiredIntArray(name);
    if (values.length != length) {
      throw malformed(name + " must contain " + length + " integers");
    }
    return values;
  }

  private static int[] optionalLength(MessageFields fields, String name, int length) {
    return fields.has(name) ? requiredLength(fields, name, length) : null;
  }

  private static int[] pointScores(int[] scoresInHundreds) {
    int[] scores = new int[scoresInHundreds.length];
    for (int player = 0; player < scoresInHundreds.length; player++) {
      scores[player] = scoresInHundreds[player] * 100;
    }
    return scores;
  }

  private static int[] optionalSettledScores(MessageFields fields) {
    int[] scoresAndDeltas = optionalLength(fields, "sc", 8);
    if (scoresAndDeltas == null) {
      return null;
    }
    int[] scores = new int[4];
    for (int player = 0; player < scores.length; player++) {
      int offset = player * 2;
      scores[player] = (scoresAndDeltas[offset] + scoresAndDeltas[offset + 1]) * 100;
    }
    return scores;
  }

  private static void requirePromptMask(int mask) {
    if (mask < 0) {
      throw malformed("t must be non-negative");
    }
  }

  private static MessageFields jsonFields(String json) {
    try {
      JsonElement root = JsonParser.parseString(json);
      if (!root.isJsonObject()) {
        throw malformed("JSON root is not an object");
      }
      JsonObject object = root.getAsJsonObject();
      JsonElement tag = object.get("tag");
      if (tag == null || !tag.isJsonPrimitive() || tag.getAsString().isEmpty()) {
        throw malformed("missing tag");
      }
      return new JsonFields(json, tag.getAsString(), object);
    } catch (MalformedMessage e) {
      throw e;
    } catch (RuntimeException e) {
      throw malformed("invalid JSON");
    }
  }

  private static MessageFields xmlFields(String xml) {
    String tag = extractTagName(xml);
    if (tag == null) {
      throw malformed("missing tag");
    }
    Map<String, String> attributes = new HashMap<>();
    Matcher matcher = ATTR_PATTERN.matcher(xml);
    while (matcher.find()) {
      attributes.put(matcher.group(1), matcher.group(2));
    }
    return new XmlFields(xml, tag, attributes);
  }

  private sealed interface MessageFields permits JsonFields, XmlFields {
    String raw();

    String tag();

    String value(String name);

    default boolean has(String name) {
      return value(name) != null;
    }

    default String required(String name) {
      String value = value(name);
      if (value == null || value.isEmpty()) {
        throw malformed("missing " + name);
      }
      return value;
    }

    default int requiredInt(String name) {
      return parseInt(required(name), name);
    }

    default int intOrDefault(String name, int defaultValue) {
      Integer value = optionalInt(name);
      return value == null ? defaultValue : value;
    }

    default Integer optionalInt(String name) {
      String value = value(name);
      return value == null ? null : parseInt(value, name);
    }

    default String orEmpty(String name) {
      String value = value(name);
      return value == null ? "" : value;
    }

    default int requiredSeat(String name) {
      int seat = requiredInt(name);
      if (seat < 0 || seat >= 4) {
        throw malformed(name + " is not a seat");
      }
      return seat;
    }

    default Integer optionalSeat(String name) {
      Integer seat = optionalInt(name);
      if (seat != null && (seat < 0 || seat >= 4)) {
        throw malformed(name + " is not a seat");
      }
      return seat;
    }

    default int[] requiredIntArray(String name) {
      String value = required(name);
      String[] parts = value.split(",", -1);
      int[] result = new int[parts.length];
      for (int i = 0; i < parts.length; i++) {
        result[i] = parseInt(parts[i], name);
      }
      return result;
    }
  }

  private record JsonFields(String raw, String tag, JsonObject object) implements MessageFields {
    @Override
    public String value(String name) {
      JsonElement element = object.get(name);
      if (element == null || element.isJsonNull()) {
        return null;
      }
      if (!element.isJsonPrimitive()) {
        throw malformed(name + " is not a primitive");
      }
      return element.getAsString();
    }
  }

  private record XmlFields(String raw, String tag, Map<String, String> attributes)
      implements MessageFields {
    @Override
    public String value(String name) {
      return attributes.get(name);
    }
  }

  /** XML タグ名を抽出。{@code <TAG .../>} → {@code TAG}。 */
  static String extractTagName(String xml) {
    int start = xml.indexOf('<');
    if (start < 0) {
      return null;
    }
    start++;
    if (start < xml.length() && xml.charAt(start) == '/') {
      start++;
    }
    int end = start;
    while (end < xml.length()) {
      char c = xml.charAt(end);
      if (Character.isWhitespace(c) || c == '/' || c == '>') {
        break;
      }
      end++;
    }
    return end == start ? null : xml.substring(start, end);
  }

  private static int parseInt(String value, String name) {
    try {
      return Integer.parseInt(value.trim());
    } catch (NumberFormatException e) {
      throw malformed(name + " is not an integer");
    }
  }

  private static boolean isDigits(String value) {
    if (value.isEmpty()) {
      return false;
    }
    for (int i = 0; i < value.length(); i++) {
      if (!Character.isDigit(value.charAt(i))) {
        return false;
      }
    }
    return true;
  }

  private static MalformedMessage malformed(String message) {
    return new MalformedMessage(message);
  }

  private static final class MalformedMessage extends RuntimeException {
    private MalformedMessage(String message) {
      super(message);
    }
  }
}
