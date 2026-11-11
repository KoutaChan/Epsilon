package com.epsilon.mahjong.client.tenhou;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.io.tenhou.TenhouTag;
import com.epsilon.mahjong.io.tenhou.TenhouTiles;
import com.epsilon.mahjong.rules.Variant;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 天鳳のライブの対局のメッセージを読み書きする。サーバのメッセージは JSON（{@code {"tag":"T52","t":"64"}}）か
 * XML（{@code <T52 t="64"/>}）で、どちらも TenhouTag に読む。返すメッセージは JSON。
 *
 * <p>席は自分を 0 とする相対席。ツモのタグは T・U・V・W の後に牌の番号が続く（他家のツモは番号なし）。打牌のタグは D・E・F・G の後に
 * 牌の番号が続き、ツモ切りなら小文字になる。判断を求めるメッセージは、出せる手のビット（Offer）を属性 t に持つ。三人打ちの卓の
 * タグは席 0〜2 の文字だけを使う。
 */
final class TenhouCodec {

  /** JSON のメッセージで、タグの名前を持つキー。 */
  private static final String TAG = "tag";

  /**
   * 判断を求めるメッセージの属性 t のビット。列挙した順に bit 0 から並ぶ。各ビットは、手番で出す手か、打牌・加槓への応答で
   * 出す手のどちらかに当たる。
   */
  enum Offer {
    PON(Type.PON, false),
    DAIMINKAN(Type.DAIMINKAN, false),
    CHI(Type.CHI, false),
    RON(Type.RON_AGARI, false),
    TSUMO(Type.TSUMO_AGARI, true),
    RIICHI(Type.RIICHI_DAHAI, true),
    KYUSHU(Type.KYUSHU_KYUHAI, true);

    /** [Action.Type の ordinal] → その手を出すビット。ビットを持たない手は null。 */
    private static final Offer[] BY_TYPE = byType();

    private final Type type;
    private final boolean turn;

    Offer(Type type, boolean turn) {
      this.type = type;
      this.turn = turn;
    }

    int bit() {
      return 1 << ordinal();
    }

    /** 行動の番号から、その手を出すビットを返す。ビットを持たない手なら null。 */
    static Offer of(int code) {
      return BY_TYPE[Action.of(code).type().ordinal()];
    }

    /**
     * 手番の判断か応答の判断のうち、指定した方で立ちうるビットを全て合わせた値を返す。
     *
     * @param turn 手番の判断なら true、打牌・加槓への応答なら false
     */
    static int bits(boolean turn) {
      int bits = 0;
      for (Offer offer : values()) {
        bits |= offer.turn == turn ? offer.bit() : 0;
      }
      return bits;
    }

    private static Offer[] byType() {
      Offer[] offers = new Offer[Type.values().length];
      for (Offer offer : values()) {
        offers[offer.type.ordinal()] = offer;
      }
      return offers;
    }
  }

  /** 返す N のメッセージの {@code type} に書く、天鳳の鳴きの番号。 */
  private enum Claim {
    PON(1),
    DAIMINKAN(2),
    CHI(3),
    ANKAN(4),
    KAKAN(5),
    RON(6),
    TSUMO(7),
    KYUSHU(9),
    /** 三人打ちの北抜き。 */
    NUKI(10);

    private final int type;

    Claim(int type) {
      this.type = type;
    }
  }

  /**
   * 読んだツモか打牌のタグ。
   *
   * @param kind ツモか打牌か
   * @param seat 相対席
   * @param tile 牌の番号。他家のツモなら {@code TurnEvent.NONE}
   * @param tsumogiri ツモ切りなら true
   */
  record Move(TenhouTiles.Move kind, int seat, int tile, boolean tsumogiri) {}

  private TenhouCodec() {}

  /** サーバのメッセージを読む。JSON でも XML のタグ 1 つでもなければ IllegalArgumentException を投げる。 */
  static TenhouTag parse(String message) {
    String text = message.strip();
    if (text.startsWith("{")) {
      return json(text);
    }
    List<TenhouTag> tags = text.startsWith("<") ? TenhouTag.parse(text) : List.of();
    if (tags.size() != 1) {
      throw new IllegalArgumentException("Not a Tenhou message: " + message);
    }
    return tags.get(0);
  }

  /** タグの名前がツモか打牌なら読み、違えば null を返す。牌の番号がその卓に無い牌なら IllegalArgumentException を投げる。 */
  static Move move(String name, Variant variant) {
    if (name.isEmpty() || !name.chars().skip(1).allMatch(Character::isDigit)) {
      return null;
    }
    char letter = name.charAt(0);
    boolean lower = Character.isLowerCase(letter);
    String number = name.substring(1);
    for (TenhouTiles.Move kind : TenhouTiles.Move.values()) {
      int seat = kind.seat(Character.toUpperCase(letter), variant);
      boolean draw = kind == TenhouTiles.Move.DRAW;
      if (seat < 0 || lower && draw || number.isEmpty() && !draw) {
        continue;
      }
      int tile = number.isEmpty() ? TurnEvent.NONE : TenhouTiles.tile(Integer.parseInt(number), variant.tiles());
      return new Move(kind, seat, tile, lower);
    }
    return null;
  }

  /** メッセージの属性 t（出せる手のビット）を返す。無ければ 0。負の値なら IllegalArgumentException を投げる。 */
  static int offers(TenhouTag tag) {
    if (tag.get("t") == null) {
      return 0;
    }
    int bits = tag.integer("t");
    if (bits < 0) {
      throw new IllegalArgumentException("Tenhou " + tag.name() + " offers negative bits " + bits);
    }
    return bits;
  }

  /**
   * その行動を出せるなら true。ビットを持たない手（打牌・暗槓・加槓・北抜き・見送り）はいつも出せる。
   *
   * @param code 行動の番号
   * @param bits メッセージの属性 t
   */
  static boolean offered(int code, int bits) {
    Offer offer = Offer.of(code);
    return offer == null || (bits & offer.bit()) != 0;
  }

  /**
   * 合法手に当たるビットを全て合わせた値を返す。手番の判断か応答の判断のうち、指定した方のビットだけを残す。
   *
   * @param decisions 席ごとの判断
   * @param slot 判断する席が、席ごとの判断の何番目か
   * @param turn 手番の判断なら true、応答なら false
   */
  static int bits(Decisions decisions, int slot, boolean turn) {
    int bits = 0;
    for (int index = 0; index < decisions.legalCount(slot); index++) {
      Offer offer = Offer.of(decisions.legal(slot, index));
      bits |= offer == null ? 0 : offer.bit();
    }
    return bits & Offer.bits(turn);
  }

  /** ログインのメッセージ。名前は URL エンコードし、sx（声の性別）は M。 */
  static String helo(String name) {
    JsonObject message = message("HELO");
    message.addProperty("name", URLEncoder.encode(name, StandardCharsets.UTF_8));
    message.addProperty("sx", "M");
    return message.toString();
  }

  static String lobby(String room) {
    JsonObject message = message("LOBBY");
    message.addProperty("id", room);
    return message.toString();
  }

  /**
   * 卓を申し込むメッセージ。
   *
   * @param target 「ロビー,種別[,追加の印]」の形の申し込み先
   */
  static String join(String target) {
    JsonObject message = message("JOIN");
    message.addProperty("t", target);
    return message.toString();
  }

  static String gok() {
    return message("GOK").toString();
  }

  static String nextReady() {
    return message("NEXTREADY").toString();
  }

  /** キープアライブのメッセージ。返すメッセージのうち、これだけは JSON でなく XML の空要素。 */
  static String keepAlive() {
    return "<Z/>";
  }

  /** 鳴き・ロンを見送るメッセージ。 */
  static String pass() {
    return message("N").toString();
  }

  /**
   * 選んだ行動を送るメッセージ。立直は宣言と打牌の 2 つになる。
   *
   * @param action 選んだ行動
   * @param tiles 手から出す物理牌（{@code SeatTable.tiles} の出力）
   */
  static List<String> reply(Action action, int[] tiles) {
    return switch (action.type()) {
      case DAHAI -> List.of(discard(tiles[0]));
      case RIICHI_DAHAI -> List.of(message("REACH").toString(), discard(tiles[0]));
      case CHI -> List.of(pair(Claim.CHI, tiles));
      case PON -> List.of(pair(Claim.PON, tiles));
      case DAIMINKAN -> List.of(claim(Claim.DAIMINKAN).toString());
      case ANKAN -> List.of(single(Claim.ANKAN, tiles[0]));
      case KAKAN -> List.of(single(Claim.KAKAN, tiles[0]));
      case RON_AGARI -> List.of(claim(Claim.RON).toString());
      case TSUMO_AGARI -> List.of(claim(Claim.TSUMO).toString());
      case KYUSHU_KYUHAI -> List.of(claim(Claim.KYUSHU).toString());
      case PASS -> List.of(pass());
      case NUKI -> List.of(claim(Claim.NUKI).toString());
    };
  }

  private static String discard(int tile) {
    JsonObject message = message("D");
    message.addProperty("p", tile);
    return message.toString();
  }

  /** 手の 2 枚を出すチー・ポン。 */
  private static String pair(Claim claim, int[] tiles) {
    JsonObject message = claim(claim);
    message.addProperty("hai0", tiles[0]);
    message.addProperty("hai1", tiles[1]);
    return message.toString();
  }

  /** 牌 1 枚で種類を示す暗槓・加槓。 */
  private static String single(Claim claim, int tile) {
    JsonObject message = claim(claim);
    message.addProperty("hai", tile);
    return message.toString();
  }

  private static JsonObject claim(Claim claim) {
    JsonObject message = message("N");
    message.addProperty("type", claim.type);
    return message;
  }

  private static JsonObject message(String tag) {
    JsonObject message = new JsonObject();
    message.addProperty(TAG, tag);
    return message;
  }

  /**
   * JSON のメッセージを読む。キー {@code tag} の値がタグの名前になり、ほかのキーの文字列か数の値が属性になる。配列と
   * オブジェクトの値は、どのタグも使わないので読まない。
   */
  private static TenhouTag json(String text) {
    JsonObject object;
    try {
      object = JsonParser.parseString(text).getAsJsonObject();
    } catch (JsonParseException | IllegalStateException failure) {
      throw new IllegalArgumentException("Not a Tenhou JSON message: " + text, failure);
    }
    String name = null;
    Map<String, String> attributes = new HashMap<>();
    for (Map.Entry<String, JsonElement> field : object.entrySet()) {
      JsonElement value = field.getValue();
      if (!value.isJsonPrimitive()) {
        continue;
      }
      if (field.getKey().equals(TAG)) {
        name = value.getAsString();
      } else {
        attributes.put(field.getKey(), value.getAsString());
      }
    }
    if (name == null || name.isEmpty()) {
      throw new IllegalArgumentException("Tenhou JSON message has no tag: " + text);
    }
    return new TenhouTag(name, attributes);
  }
}
