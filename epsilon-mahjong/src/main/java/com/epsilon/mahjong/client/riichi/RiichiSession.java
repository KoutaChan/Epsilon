package com.epsilon.mahjong.client.riichi;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.client.Decider;
import com.epsilon.mahjong.client.LiveTable;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Face;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.function.IntPredicate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * RiichiLab の 1 つの接続での対局（四人打ち）。サーバの JSON のメッセージを受け、観測に入った mjai のイベントを LiveTable に書く。
 * 行動を求められたら、Decider が選んだ手を返す。席は起家を 0 とする絶対席で、mjai と同じ。
 *
 * <p>合法手は、エンジンの合法手のうち、要求の {@code _legal_actions} にある手に絞る。返すメッセージは、選んだ手に当たる
 * {@code possible_actions} の要素に、{@code request_id} と {@code actor} を足したもの。打牌には {@code tsumogiri} を、鳴き・
 * ロン・ツモには {@code target} も足す。立直では、まず宣言（{@code reach}）を返す。続く打牌の要求には、先に選んでおいた宣言牌を
 * Decider に聞かずに返す。前と同じか古い {@code request_id} の要求と、締め切りを過ぎた要求には答えない。
 *
 * <p>mjai の牌は表記だけなので、物理 id はこちらで割り当てる。自分の手に入る牌は、同じ表記のうちまだ持っていない最初の 1 枚にする
 * （赤の 5 は、同じ種類の 4 枚のうち番号 0 の牌）。ほかの牌は、同じ表記の最初の 1 枚にする。
 *
 * <p>RiichiLab の和了のイベントは、点の移動と裏ドラしか持たない（役・翻・符が無い）。局の結果を RoundResult にできないので、
 * こちらの精算との照らし合わせはしない。局は、いつもサーバの卓から始める。同時に使えるスレッドは 1 つだけ。
 */
public final class RiichiSession {

  private static final Logger log = LoggerFactory.getLogger(RiichiSession.class);

  private static final Variant VARIANT = Variant.TENHOU_4P;

  /**
   * 行動の種類。{@code _legal_actions} の {@code action_type} と、それに当たる mjai の {@code type}
   * （{@code possible_actions} で使う）を持つ。
   */
  private enum Legal {
    DISCARD("Discard", "dahai"),
    CHI("Chi", "chi"),
    PON("Pon", "pon"),
    DAIMINKAN("Daiminkan", "daiminkan"),
    ANKAN("Ankan", "ankan"),
    KAKAN("Kakan", "kakan"),
    RIICHI("Riichi", "reach"),
    TSUMO("Tsumo", "hora"),
    RON("Ron", "hora"),
    PASS("Pass", "none"),
    KYUSHU_KYUHAI("KyushuKyuhai", "ryukyoku");

    private final String actionType;
    private final String mjai;

    Legal(String actionType, String mjai) {
      this.actionType = actionType;
      this.mjai = mjai;
    }

    static Legal parse(String actionType) {
      for (Legal legal : values()) {
        if (legal.actionType.equals(actionType)) {
          return legal;
        }
      }
      throw new IllegalArgumentException("Unknown RiichiLab action " + actionType);
    }
  }

  /** {@code action_ack} の {@code status}。サーバが送る値は、名前を小文字にしたもの。 */
  private enum Receipt {
    ACCEPTED,
    DEFAULTED,
    STALE,
    REJECTED,
    UNPARSEABLE;

    static Receipt parse(String text) {
      return valueOf(text.toUpperCase(Locale.ROOT));
    }
  }

  private final Decider decider;
  private final boolean validation;
  private final TileSet set = VARIANT.tiles();

  /** 行動の番号 → 返す {@code possible_actions} の要素。要求のたびに作り直す。 */
  private final Map<Integer, JsonObject> wires = new LinkedHashMap<>();

  private LiveTable live;
  private int seat = TurnEvent.NONE;
  private long requestId = -1;
  private boolean completed;
  private Boolean validationPassed;

  /** いまの要求で立直の宣言を出せるなら、その {@code possible_actions} の要素。出せなければ null。 */
  private JsonObject reachWire;

  /** 最後に送った手。受け付けられたかは {@code action_ack} で分かる。 */
  private int submitted = LiveTable.NO_DECISION;

  /** 立直の宣言を送ったときに選んだ宣言牌（立直の打牌の行動の番号）。続く打牌の要求で、これを送る。 */
  private int riichiDiscard = LiveTable.NO_DECISION;

  /** 立直の宣言が受け付けられたか。{@code action_ack} か、観測の {@code reach} のイベントで分かる。 */
  private boolean riichiAccepted;

  /**
   * @param decider 手を選ぶもの
   * @param validation 検証の接続なら true。検証の接続は {@code validation_result} が届くまで続く
   */
  public RiichiSession(Decider decider, boolean validation) {
    this.decider = Objects.requireNonNull(decider, "decider");
    this.validation = validation;
  }

  /** 対局が始まって席が決まったなら true。 */
  public boolean started() {
    return seat != TurnEvent.NONE;
  }

  /** 対局が最後まで終わったなら true。 */
  public boolean completed() {
    return completed;
  }

  /** この接続でやることが終わったなら true。検証の接続なら結果が届いたとき、ほかは対局が終わったとき。 */
  public boolean ended() {
    return validation ? validationPassed != null : completed;
  }

  public boolean validationPassed() {
    return Boolean.TRUE.equals(validationPassed);
  }

  /**
   * サーバのメッセージを 1 つ受ける。
   *
   * @param message サーバのメッセージ（JSON）
   * @param receivedNanos 受け取り始めた時刻（System.nanoTime の値）
   * @return 返すメッセージ。返すものが無ければ null
   * @throws IllegalArgumentException メッセージが読めないとき、またはサーバが手を拒んだとき
   * @throws IllegalStateException 観測が卓の進みと合わないとき
   */
  public String handle(String message, long receivedNanos) {
    JsonObject object;
    try {
      object = JsonParser.parseString(message).getAsJsonObject();
    } catch (JsonParseException | IllegalStateException failure) {
      throw new IllegalArgumentException("Not a RiichiLab message: " + message, failure);
    }
    String type = object.has("type") ? object.get("type").getAsString() : "";
    switch (type) {
      case "start_game" -> assignSeat(object.get("id").getAsInt());
      case "request_action" -> {
        return respond(object, receivedNanos);
      }
      case "action_ack" -> acknowledge(object);
      case "end_game" -> {
        completed = true;
        riichiDiscard = LiveTable.NO_DECISION;
        log.info("RiichiLab game ends: scores={}", object.get("scores"));
      }
      case "validation_result" -> {
        validationPassed = object.get("passed").getAsBoolean();
        log.info("RiichiLab validation passed={} reason={}", validationPassed, object.get("reason"));
      }
      default -> log.debug("RiichiLab message {} is not read", type);
    }
    return null;
  }

  private void assignSeat(int id) {
    if (id < 0 || id >= VARIANT.seats() || seat != TurnEvent.NONE && seat != id) {
      throw new IllegalArgumentException("RiichiLab assigns seat " + id + " to the player of seat " + seat);
    }
    if (seat == TurnEvent.NONE) {
      seat = id;
      log.info("RiichiLab {} game starts at seat {}", validation ? "validation" : "ranked", seat);
    }
  }

  private String respond(JsonObject message, long receivedNanos) {
    long id = message.get("request_id").getAsLong();
    if (id <= requestId) {
      return null;
    }
    requestId = id;
    submitted = LiveTable.NO_DECISION;
    byte[] encoded = Base64.getDecoder().decode(message.get("observation").getAsString());
    JsonObject observation = JsonParser.parseString(new String(encoded, StandardCharsets.UTF_8)).getAsJsonObject();
    assignSeat(observation.get("player_id").getAsInt());
    for (JsonElement event : observation.getAsJsonArray("events")) {
      apply(JsonParser.parseString(event.getAsString()).getAsJsonObject());
    }
    if (live == null) {
      throw new IllegalStateException("RiichiLab requests an action before start_kyoku");
    }
    long deadline = message.getAsJsonObject("time").get("deadline_ms").getAsLong();
    if (expired(receivedNanos, deadline)) {
      return null;
    }
    offers(observation, message.getAsJsonArray("possible_actions"));
    boolean followUp = riichiDiscard != LiveTable.NO_DECISION;
    int code = followUp ? riichiDiscard : choose();
    JsonObject wire = wireOf(code, followUp, id);
    if (expired(receivedNanos, deadline)) {
      return null;
    }
    String reply = encode(code, wire, id, followUp);
    submitted = code;
    boolean declares = !followUp && code != LiveTable.NO_DECISION && Action.of(code).type() == Type.RIICHI_DAHAI;
    riichiDiscard = declares ? code : LiveTable.NO_DECISION;
    riichiAccepted &= !declares;
    return reply;
  }

  /**
   * エンジンの合法手のうち、サーバが出した手から Decider に選ばせる。サーバが出した手のうち、エンジンの合法手に無いものが
   * あれば警告を出す。
   */
  private int choose() {
    for (int offered : wires.keySet()) {
      if (!allows(offered)) {
        log.warn("RiichiLab offers {} to seat {}, which the engine does not allow", Action.of(offered), seat);
      }
    }
    return live.decide(this::offered);
  }

  /**
   * サーバが出した手なら true。{@code possible_actions} に要素がある手と、立直の宣言を出せるときの立直の打牌を、サーバが
   * 出した手とみなす。
   */
  private boolean offered(int code) {
    return wires.containsKey(code) || reachWire != null && Action.of(code).type() == Type.RIICHI_DAHAI;
  }

  /**
   * 選んだ手に当たる {@code possible_actions} の要素を返す。立直の打牌なら、宣言のときは {@code reach} の要素を返し、続く
   * 打牌の要求では宣言牌の打牌の要素を返す。当たる要素が無いときと、宣言が受け付けられる前に宣言牌を求められたときは、
   * IllegalStateException を投げる。
   *
   * @param code 選んだ手の行動の番号。選ぶ手が無ければ {@code NO_DECISION} で、そのときは見送りの要素を返す
   * @param followUp 立直の宣言の後の打牌の要求なら true
   * @param id 要求の番号
   */
  private JsonObject wireOf(int code, boolean followUp, long id) {
    JsonObject wire;
    if (code == LiveTable.NO_DECISION) {
      wire = wires.get(Action.code(Type.PASS));
    } else if (Action.of(code).type() != Type.RIICHI_DAHAI) {
      wire = wires.get(code);
    } else if (!followUp) {
      wire = reachWire;
    } else if (!riichiAccepted) {
      throw new IllegalStateException("RiichiLab asks for the riichi discard before accepting the declaration");
    } else {
      Action riichi = Action.of(code);
      wire = wires.get(Action.discard(riichi.tile(), riichi.selection().red(), riichi.selection().tsumogiri(), false));
    }
    if (wire == null) {
      throw new IllegalStateException(
          "RiichiLab request " + id + " offers seat " + seat + " no action for "
              + (code == LiveTable.NO_DECISION ? "a pass" : Action.of(code)));
    }
    return wire;
  }

  /**
   * 返すメッセージを作る。{@code possible_actions} の要素をコピーして、{@code request_id} と {@code actor} を足す。打牌
   * （立直の宣言の後に送る宣言牌を含む）なら {@code tsumogiri} を、鳴き・ロン・ツモなら {@code target} も足す。
   *
   * @param code 選んだ手の行動の番号。選ぶ手が無ければ {@code NO_DECISION}
   * @param wire 選んだ手に当たる {@code possible_actions} の要素
   * @param id 要求の番号
   * @param followUp 立直の宣言の後の打牌の要求なら true
   */
  private String encode(int code, JsonObject wire, long id, boolean followUp) {
    JsonObject reply = wire.deepCopy();
    reply.addProperty("request_id", id);
    reply.addProperty("actor", seat);
    if (code == LiveTable.NO_DECISION) {
      return reply.toString();
    }
    Action action = Action.of(code);
    switch (action.type()) {
      case DAHAI -> reply.addProperty("tsumogiri", action.selection().tsumogiri());
      case RIICHI_DAHAI -> {
        if (followUp) {
          reply.addProperty("tsumogiri", action.selection().tsumogiri());
        }
      }
      case CHI, PON, DAIMINKAN, RON_AGARI -> reply.addProperty("target", live.view().last().seat());
      case TSUMO_AGARI -> reply.addProperty("target", seat);
      default -> {}
    }
    return reply.toString();
  }

  /**
   * 要求の {@code _legal_actions} を行動の番号に直し、それぞれに当たる {@code possible_actions} の要素を覚える。立直の宣言は
   * 別に覚える。
   */
  private void offers(JsonObject observation, JsonArray possible) {
    wires.clear();
    reachWire = null;
    JsonElement drawnTile = observation.get("drawn_tile");
    int drawn = drawnTile == null || drawnTile.isJsonNull() ? TurnEvent.NONE : drawnTile.getAsInt();
    for (JsonElement value : observation.getAsJsonArray("_legal_actions")) {
      JsonObject raw = value.getAsJsonObject();
      Legal legal = Legal.parse(raw.get("action_type").getAsString());
      JsonObject wire = match(legal, raw, possible, drawn);
      if (legal == Legal.RIICHI) {
        reachWire = wire;
      } else {
        wires.putIfAbsent(code(legal, raw, drawn), wire);
      }
    }
  }

  /** 自分の合法手にその手があれば true。 */
  private boolean allows(int code) {
    Decisions decisions = live.decisions();
    for (int index = 0; decisions.size() > 0 && index < decisions.legalCount(0); index++) {
      if (decisions.legal(0, index) == code) {
        return true;
      }
    }
    return false;
  }

  /** {@code _legal_actions} の要素を行動の番号に直す。要素の牌は 0〜135 の番号で、天鳳と同じ物理 id。 */
  private int code(Legal legal, JsonObject raw, int drawn) {
    int tile = raw.get("tile").isJsonNull() ? TurnEvent.NONE : raw.get("tile").getAsInt();
    int[] consumed = integers(raw.getAsJsonArray("consume_tiles"));
    return switch (legal) {
      case DISCARD -> Action.discard(Tiles.kind(tile), set.red(tile), tile == drawn, false);
      case CHI -> {
        int base = Tiles.kind(tile);
        for (int each : consumed) {
          base = Math.min(base, Tiles.kind(each));
        }
        yield Action.chi(base, Tiles.kind(tile), anyRed(consumed));
      }
      case PON ->
          Action.code(Type.PON, Tiles.kind(tile), anyRed(consumed) ? TileSelection.AKA : TileSelection.NON_AKA);
      case DAIMINKAN -> Action.code(Type.DAIMINKAN, Tiles.kind(tile), TileSelection.NOT_APPLICABLE);
      case ANKAN -> Action.code(Type.ANKAN, Tiles.kind(consumed[0]), TileSelection.NOT_APPLICABLE);
      case KAKAN -> Action.code(Type.KAKAN, Tiles.kind(tile), TileSelection.NOT_APPLICABLE);
      case TSUMO -> Action.code(Type.TSUMO_AGARI);
      case RON -> Action.code(Type.RON_AGARI);
      case PASS -> Action.code(Type.PASS);
      case KYUSHU_KYUHAI -> Action.code(Type.KYUSHU_KYUHAI);
      case RIICHI -> throw new IllegalArgumentException("A riichi declaration names no discard");
    };
  }

  /**
   * {@code _legal_actions} の要素に当たる {@code possible_actions} の要素を返す。当たるのは、mjai の {@code type} が同じで、
   * {@code pai}・{@code consumed}・{@code tsumogiri} があればそれも一致する要素。見つからなければ IllegalArgumentException を
   * 投げる。
   */
  private JsonObject match(Legal legal, JsonObject raw, JsonArray possible, int drawn) {
    for (JsonElement value : possible) {
      JsonObject wire = value.getAsJsonObject();
      if (!legal.mjai.equals(wire.get("type").getAsString())) {
        continue;
      }
      if (wire.has("pai") && !raw.get("tile").isJsonNull()
          && face(raw.get("tile").getAsInt()) != Face.parse(wire.get("pai").getAsString())) {
        continue;
      }
      if (wire.has("consumed")
          && !sameFaces(integers(raw.getAsJsonArray("consume_tiles")), wire.getAsJsonArray("consumed"))) {
        continue;
      }
      if (legal == Legal.DISCARD
          && wire.has("tsumogiri")
          && wire.get("tsumogiri").getAsBoolean() != (raw.get("tile").getAsInt() == drawn)) {
        continue;
      }
      return wire;
    }
    throw new IllegalArgumentException("RiichiLab offers " + raw + " without a matching possible action");
  }

  /** 観測に入った mjai のイベントを 1 つ卓に書く。 */
  private void apply(JsonObject event) {
    String type = event.get("type").getAsString();
    switch (type) {
      case "start_kyoku" -> startRound(event);
      case "tsumo" -> {
        int actor = event.get("actor").getAsInt();
        table().draw(actor, actor == seat ? free(Face.parse(event.get("pai").getAsString())) : TurnEvent.NONE);
      }
      case "dahai" ->
          table().discard(
              event.get("actor").getAsInt(),
              representative(Face.parse(event.get("pai").getAsString())),
              event.get("tsumogiri").getAsBoolean());
      case "chi", "pon", "daiminkan" -> call(event, Meld.Type.valueOf(type.toUpperCase(Locale.ROOT)));
      case "ankan" ->
          table().closedKan(event.get("actor").getAsInt(), faces(event.getAsJsonArray("consumed")).get(0).kind());
      case "kakan" ->
          table().addedKan(event.get("actor").getAsInt(), Face.parse(event.get("pai").getAsString()).kind());
      case "dora" -> table().dora(representative(Face.parse(event.get("dora_marker").getAsString())));
      case "reach" -> {
        int actor = event.get("actor").getAsInt();
        table().declareRiichi(actor);
        if (actor == seat) {
          riichiAccepted = true;
        }
      }
      case "reach_accepted" -> table().acceptRiichi(event.get("actor").getAsInt());
      case "hora", "ryukyoku" -> log.info("RiichiLab round result: {}", event);
      default -> log.debug("RiichiLab event {} is not read", type);
    }
  }

  private void startRound(JsonObject event) {
    JsonArray hands = event.getAsJsonArray("tehais");
    int[] scores = integers(event.getAsJsonArray("scores"));
    if (hands.size() != VARIANT.seats() || scores.length != VARIANT.seats()) {
      throw new IllegalArgumentException("RiichiLab plays " + hands.size() + " seats, not " + VARIANT.seats());
    }
    int wind = Face.parse(event.get("bakaze").getAsString()).kind() - Tiles.TON;
    int round = wind * VARIANT.seats() + event.get("kyoku").getAsInt() - 1;
    RoundStart start =
        new RoundStart(VARIANT, round, event.get("honba").getAsInt(), event.get("kyotaku").getAsInt(), scores);
    if (event.get("oya").getAsInt() != VARIANT.dealer(round)) {
      throw new IllegalArgumentException("RiichiLab names dealer " + event.get("oya") + " in round " + round);
    }
    if (live == null) {
      live = new LiveTable(VARIANT, seat, decider);
    }
    List<Face> faces = faces(hands.get(seat).getAsJsonArray());
    int[] hand = new int[faces.size()];
    boolean[] dealt = new boolean[Tiles.PHYSICAL_IDS];
    for (int index = 0; index < hand.length; index++) {
      hand[index] = copy(faces.get(index), tile -> dealt[tile]);
      dealt[hand[index]] = true;
    }
    live.startRound(start, hand, representative(Face.parse(event.get("dora_marker").getAsString())));
    riichiDiscard = LiveTable.NO_DECISION;
    riichiAccepted = false;
    log.info("RiichiLab {} starts with this player at seat {}", start, seat);
  }

  private void call(JsonObject event, Meld.Type type) {
    int actor = event.get("actor").getAsInt();
    Meld meld =
        Meld.call(
            type,
            Meld.RelativeSource.of(actor, event.get("target").getAsInt(), VARIANT.seats()),
            Face.parse(event.get("pai").getAsString()),
            faces(event.getAsJsonArray("consumed")));
    table().call(actor, meld.id());
  }

  private void acknowledge(JsonObject message) {
    long id = message.get("request_id").getAsLong();
    Receipt receipt = Receipt.parse(message.get("status").getAsString());
    log.info("RiichiLab receipt: {}", message);
    switch (receipt) {
      case REJECTED, UNPARSEABLE -> {
        riichiDiscard = LiveTable.NO_DECISION;
        throw new IllegalArgumentException(
            "RiichiLab " + receipt + " request " + id + ": " + message.get("reason"));
      }
      case ACCEPTED -> {
        if (id == requestId && submitted != LiveTable.NO_DECISION && riichiDiscard == submitted) {
          riichiAccepted = true;
        }
      }
      case DEFAULTED -> {
        riichiDiscard = LiveTable.NO_DECISION;
        log.warn("RiichiLab answers request {} by default", id);
      }
      case STALE -> riichiDiscard = LiveTable.NO_DECISION;
    }
  }

  private LiveTable table() {
    if (live == null) {
      throw new IllegalStateException("RiichiLab sends a round event before start_kyoku");
    }
    return live;
  }

  /** 自分の手に入る牌の物理 id。同じ表記のうち、まだ持っていない最初の 1 枚。 */
  private int free(Face face) {
    return copy(face, live.view()::holds);
  }

  /** 自分の手に加わらない牌（打牌・表示牌）の物理 id。同じ表記の最初の 1 枚。 */
  private int representative(Face face) {
    return copy(face, tile -> false);
  }

  /**
   * 同じ表記の牌のうち、使われていない最初の 1 枚の物理 id を返す。無ければ IllegalArgumentException を投げる。
   *
   * @param taken 物理 id を受け、使われている牌なら true を返す条件
   */
  private int copy(Face face, IntPredicate taken) {
    for (int copy = 0; copy < Tiles.COPIES; copy++) {
      int tile = Tiles.physicalId(face.kind(), copy);
      if (set.contains(tile) && set.red(tile) == face.red() && !taken.test(tile)) {
        return tile;
      }
    }
    throw new IllegalArgumentException("No copy of " + face.mjai() + " is left for seat " + seat);
  }

  private Face face(int tile) {
    return Face.of(Tiles.kind(tile), set.red(tile));
  }

  private boolean anyRed(int[] tiles) {
    for (int tile : tiles) {
      if (set.red(tile)) {
        return true;
      }
    }
    return false;
  }

  /** 物理 id の列と表記の列が、順番を問わず同じ表記を同じ枚数ずつ持つなら true。 */
  private boolean sameFaces(int[] tiles, JsonArray names) {
    if (tiles.length != names.size()) {
      return false;
    }
    List<Face> left = new ArrayList<>(faces(names));
    for (int tile : tiles) {
      if (!left.remove(face(tile))) {
        return false;
      }
    }
    return true;
  }

  private static List<Face> faces(JsonArray names) {
    List<Face> faces = new ArrayList<>();
    for (JsonElement name : names) {
      faces.add(Face.parse(name.getAsString()));
    }
    return faces;
  }

  private static int[] integers(JsonArray values) {
    int[] integers = new int[values.size()];
    for (int index = 0; index < integers.length; index++) {
      integers[index] = values.get(index).getAsInt();
    }
    return integers;
  }

  private static boolean expired(long receivedNanos, long deadlineMillis) {
    return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - receivedNanos) >= deadlineMillis;
  }
}
