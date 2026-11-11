package com.epsilon.mahjong.client.tenhou;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.io.tenhou.TenhouMeldCode;
import com.epsilon.mahjong.io.tenhou.TenhouTag;
import com.epsilon.mahjong.io.tenhou.TenhouTiles;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 天鳳の牌譜を、1 席がライブの対局で受け取るメッセージの列に直す。タグの席はその席から数えた相対席に直し、他家のツモ牌は伏せ、
 * ツモ切りの打牌は小文字にする。その席に判断を求めるメッセージには t の bit を付ける。
 * 席の判断と返すはずのメッセージは、同じ牌譜を再生するエンジン（ReplayDriver）がその席に見せる合法手と、牌譜の手から作る。
 * 牌譜から分からない応答は見送りにする。返すはずのメッセージに書く物理牌は、その手でエンジンの手牌から出た牌。
 * 三人打ちの牌譜では、席ごとに並ぶ値の 4 つ目（空き席）は回さずに残す。N タグの m にある鳴き元は、鳴いた席から天鳳の 4 席の上で数えた
 * 値なので、相対席の上で数え直して書く（ライブのサーバも、自分を席 0 として数えるものとみなす）。北抜きの N タグには、自分の北抜きなら返事をせず、
 * 他家の北抜きなら抜いた北への応答を求める。
 */
final class LiveTranscript {

  static final String GOK = "{\"tag\":\"GOK\"}";
  static final String NEXT_READY = "{\"tag\":\"NEXTREADY\"}";
  static final String PASS = "{\"tag\":\"N\"}";

  private static final int PASS_CODE = Action.code(Action.Type.PASS);

  /** サーバのメッセージと、それに返すはずのメッセージ。 */
  record Line(String message, List<String> replies) {}

  /** その席の判断。判断の種類、エンジンの合法手、選ぶ手を持つ。 */
  record Decision(EngineDecisionKind kind, List<Integer> legal, int code) {}

  /** 判断を求めるメッセージの t と、返すはずのメッセージ。 */
  private record Prompt(int bits, List<String> replies) {}

  /** 1 局で判断を求めるメッセージ。手番の判断と鳴いた後の打牌はメッセージの順に使う。他家の応答は、打牌・加槓の公開イベントの番号で引く。 */
  private static final class RoundPrompts {
    final ArrayDeque<Prompt> turns = new ArrayDeque<>();
    final ArrayDeque<Prompt> postCalls = new ArrayDeque<>();
    final Map<Integer, Prompt> responses = new HashMap<>();
  }

  private final Variant variant;
  private final int seats;
  private final int seat;
  private final boolean json;
  private final List<Line> lines = new ArrayList<>();
  private final List<Decision> decisions = new ArrayList<>();
  private final List<RoundPrompts> rounds = new ArrayList<>();

  private LiveTranscript(Variant variant, int seat, boolean json) {
    this.variant = variant;
    seats = variant.seats();
    this.seat = seat;
    this.json = json;
  }

  /**
   * 牌譜の 1 席が、ライブの対局で受け取るメッセージを作る。
   *
   * @param variant 牌譜のルール
   * @param xml 牌譜の XML
   * @param id 牌譜 id
   * @param seat 牌譜の上の席（相対席ではない）
   * @param json true ならメッセージを JSON で、false なら XML で書く
   * @return その席のメッセージの列
   */
  static LiveTranscript of(Variant variant, String xml, String id, int seat, boolean json) {
    LiveTranscript transcript = new LiveTranscript(variant, seat, json);
    transcript.collect(TenhouXml.read(xml, id));
    transcript.convert(xml);
    for (RoundPrompts round : transcript.rounds) {
      if (!round.turns.isEmpty() || !round.postCalls.isEmpty() || !round.responses.isEmpty()) {
        throw new AssertionError(id + " seat " + seat + " leaves prompts without a message");
      }
    }
    return transcript;
  }

  List<Line> lines() {
    return lines;
  }

  /** その席の判断。エンジンが求めた順に並ぶ。 */
  List<Decision> decisions() {
    return decisions;
  }

  /** エンジンで牌譜を再生して、その席の判断を集める。局ごとに、判断を求めるメッセージの t と返すはずのメッセージも集める。 */
  private void collect(RecordedGame game) {
    ReplayDriver driver = new ReplayDriver(variant);
    RuleWorkspace ws = new RuleWorkspace(variant);
    for (RecordedRound round : game.rounds()) {
      RoundPrompts prompts = new RoundPrompts();
      rounds.add(prompts);
      Boundary boundary = driver.start(round, ws);
      while (boundary == Boundary.DECISIONS) {
        Decisions engine = driver.engine().decisions();
        int slot = engine.slotOf(seat);
        if (slot < 0) {
          boundary = driver.step(ws);
          continue;
        }
        EngineDecisionKind kind = engine.kind(slot);
        List<Integer> legal = new ArrayList<>();
        for (int index = 0; index < engine.legalCount(slot); index++) {
          legal.add(engine.legal(slot, index));
        }
        int choice = driver.choice(slot);
        int code = choice == ReplayDriver.UNOBSERVED ? PASS_CODE : legal.get(choice);
        SeatView view = driver.engine().view(seat);
        int event = view.events().size() - 1;
        boolean[] held = new boolean[Tiles.PHYSICAL_IDS];
        for (int tile = 0; tile < Tiles.PHYSICAL_IDS; tile++) {
          held[tile] = view.holds(tile);
        }
        decisions.add(new Decision(kind, List.copyOf(legal), code));
        boundary = driver.step(ws);
        List<Integer> removed = new ArrayList<>();
        for (int tile = 0; tile < Tiles.PHYSICAL_IDS; tile++) {
          if (held[tile] && !driver.engine().view(seat).holds(tile)) {
            removed.add(tile);
          }
        }
        Prompt prompt = new Prompt(bits(legal), replies(code, removed));
        switch (kind) {
          case TURN -> prompts.turns.add(prompt);
          case POST_CALL_DAHAI -> prompts.postCalls.add(prompt);
          case RESPONSE, CHANKAN, NUKI_RESPONSE -> prompts.responses.put(event, prompt);
        }
      }
    }
  }

  /** 牌譜のタグを、その席がライブで受け取るメッセージに直す。 */
  private void convert(String xml) {
    RoundPrompts prompts = null;
    int round = -1;
    int events = 0;
    boolean ended = false;
    int[] drawn = new int[seats];
    for (TenhouTag tag : TenhouTag.parse(xml)) {
      switch (tag.name()) {
        case "mjloggm", "SHUFFLE", "BYE", "PROF" -> {}
        case "GO" -> line(tag.name(), tag.attributes(), List.of(GOK));
        case "UN" -> line(tag.name(), tag.attributes(), List.of());
        case "TAIKYOKU" -> line(tag.name(), tag.attributes(), List.of(NEXT_READY));
        case "INIT" -> {
          prompts = rounds.get(++round);
          events = 0;
          ended = false;
          Arrays.fill(drawn, TurnEvent.NONE);
          line("INIT", init(tag), List.of());
        }
        case "DORA" -> line(tag.name(), tag.attributes(), List.of());
        case "REACH" -> line(tag.name(), reach(tag), List.of());
        case "N" -> {
          int who = tag.integer("who");
          int code = tag.integer("m");
          boolean nuki = TenhouMeldCode.nuki(code);
          TenhouMeldCode meld = nuki ? null : TenhouMeldCode.decode(code, variant, who);
          Meld.Type type = nuki ? null : meld.meld().type();
          int relativeWho = Math.floorMod(who - seat, seats);
          String m = nuki ? tag.get("m") : Integer.toString(meld.encode(variant, relativeWho));
          Map<String, String> attributes = attributes("who", relativeWho, "m", m);
          List<String> replies = List.of();
          if (who == seat && (type == Meld.Type.CHI || type == Meld.Type.PON)) {
            replies = prompts.postCalls.removeFirst().replies();
          } else if (who != seat && prompts.responses.containsKey(events)) {
            Prompt prompt = prompts.responses.remove(events);
            attributes.put("t", Integer.toString(prompt.bits()));
            replies = prompt.replies();
          }
          line("N", attributes, replies);
          events++;
          drawn[who] = TurnEvent.NONE;
        }
        case "AGARI", "RYUUKYOKU" -> {
          boolean last = tag.get("owari") != null;
          line(tag.name(), result(tag), last || ended ? List.of() : List.of(NEXT_READY));
          ended = true;
        }
        default -> {
          TenhouTiles.TileTag tile = TenhouTiles.TileTag.parse(tag.name(), variant);
          int letter = Math.floorMod(tile.seat() - seat, seats);
          switch (tile.move()) {
            case DRAW -> {
              drawn[tile.seat()] = tile.tile();
              if (tile.seat() == seat) {
                Prompt prompt = prompts.turns.removeFirst();
                Map<String, String> offers = prompt.bits() == 0 ? attributes() : attributes("t", prompt.bits());
                line("T" + tile.tile(), offers, prompt.replies());
              } else {
                line(String.valueOf((char) ('T' + letter)), attributes(), List.of());
              }
            }
            case DISCARD -> {
              boolean tsumogiri = tile.tile() == drawn[tile.seat()];
              drawn[tile.seat()] = TurnEvent.NONE;
              String name = (char) ((tsumogiri ? 'd' : 'D') + letter) + Integer.toString(tile.tile());
              Prompt prompt = tile.seat() == seat ? null : prompts.responses.remove(events);
              if (prompt == null) {
                line(name, attributes(), List.of());
              } else {
                line(name, attributes("t", prompt.bits()), prompt.replies());
              }
              events++;
            }
          }
        }
      }
    }
  }

  /** INIT のタグ。ten と oya をその席から数え直し、hai にはその席の配牌だけを置く。 */
  private Map<String, String> init(TenhouTag tag) {
    return attributes(
        "seed", tag.get("seed"),
        "ten", rotate(tag.get("ten"), 1),
        "oya", relative(tag.integer("oya")),
        "hai", tag.get("hai" + seat));
  }

  private Map<String, String> reach(TenhouTag tag) {
    Map<String, String> attributes = attributes("who", relative(tag.integer("who")), "step", tag.get("step"));
    if (tag.get("ten") != null) {
      attributes.put("ten", rotate(tag.get("ten"), 1));
    }
    return attributes;
  }

  /** AGARI・RYUUKYOKU のタグ。席を表す属性（who・fromWho・paoWho・sc・owari、聴牌の hai0〜3）を相対席に直す。 */
  private Map<String, String> result(TenhouTag tag) {
    Map<String, String> attributes = new LinkedHashMap<>(tag.attributes());
    for (String key : List.of("who", "fromWho", "paoWho")) {
      if (tag.get(key) != null) {
        attributes.put(key, relative(tag.integer(key)));
      }
    }
    for (String key : List.of("sc", "owari")) {
      if (tag.get(key) != null) {
        attributes.put(key, rotate(tag.get(key), 2));
      }
    }
    for (int absolute = 0; absolute < seats; absolute++) {
      attributes.remove("hai" + absolute);
    }
    for (int absolute = 0; absolute < seats; absolute++) {
      String hand = tag.get("hai" + absolute);
      if (hand != null) {
        attributes.put("hai" + relative(absolute), hand);
      }
    }
    return attributes;
  }

  /**
   * 席ごとに同じ数ずつ並ぶ値の列を、この席が先頭になるように回す。ルールの席数より後ろの値（三人打ちの空き席）はそのまま残す。
   *
   * @param values コンマで区切った値の列
   * @param width 1 席の値の数
   */
  private String rotate(String values, int width) {
    String[] split = values.split(",", -1);
    String[] rotated = split.clone();
    for (int relative = 0; relative < seats; relative++) {
      int absolute = (relative + seat) % seats;
      System.arraycopy(split, absolute * width, rotated, relative * width, width);
    }
    return String.join(",", rotated);
  }

  private String relative(int absolute) {
    return Integer.toString(Math.floorMod(absolute - seat, seats));
  }

  private void line(String name, Map<String, String> attributes, List<String> replies) {
    StringBuilder text = new StringBuilder();
    if (json) {
      text.append("{\"tag\":\"").append(name).append('"');
      attributes.forEach((key, value) -> text.append(",\"").append(key).append("\":\"").append(value).append('"'));
      text.append('}');
    } else {
      text.append('<').append(name);
      attributes.forEach((key, value) -> text.append(' ').append(key).append("=\"").append(value).append('"'));
      text.append("/>");
    }
    lines.add(new Line(text.toString(), replies));
  }

  /** キーと値を交互に並べた引数から、並び順を保つ属性を作る。 */
  private static Map<String, String> attributes(Object... pairs) {
    Map<String, String> attributes = new LinkedHashMap<>();
    for (int index = 0; index < pairs.length; index += 2) {
      attributes.put(pairs[index].toString(), pairs[index + 1].toString());
    }
    return attributes;
  }

  /** 判断を求めるメッセージの t。合法手にある手の bit を重ねる（ポン 1、大明槓 2、チー 4、ロン 8、ツモ 16、立直 32、九種九牌 64）。 */
  private static int bits(List<Integer> legal) {
    int bits = 0;
    for (int code : legal) {
      bits |=
          switch (Action.of(code).type()) {
            case PON -> 1;
            case DAIMINKAN -> 2;
            case CHI -> 4;
            case RON_AGARI -> 8;
            case TSUMO_AGARI -> 16;
            case RIICHI_DAHAI -> 32;
            case KYUSHU_KYUHAI -> 64;
            default -> 0;
          };
    }
    return bits;
  }

  /**
   * 選んだ手に対して返すはずのメッセージ。
   *
   * @param code 選ぶ手
   * @param removed その手でエンジンの手牌から出た物理牌（昇順）
   */
  private static List<String> replies(int code, List<Integer> removed) {
    return switch (Action.of(code).type()) {
      case DAHAI -> List.of(discard(removed.get(0)));
      case RIICHI_DAHAI -> List.of("{\"tag\":\"REACH\"}", discard(removed.get(0)));
      case CHI -> List.of(claim(3, "\"hai0\":" + removed.get(0) + ",\"hai1\":" + removed.get(1)));
      case PON -> List.of(claim(1, "\"hai0\":" + removed.get(0) + ",\"hai1\":" + removed.get(1)));
      case DAIMINKAN -> List.of(claim(2, null));
      case ANKAN -> List.of(claim(4, "\"hai\":" + removed.get(0)));
      case KAKAN -> List.of(claim(5, "\"hai\":" + removed.get(0)));
      case RON_AGARI -> List.of(claim(6, null));
      case TSUMO_AGARI -> List.of(claim(7, null));
      case KYUSHU_KYUHAI -> List.of(claim(9, null));
      case PASS -> List.of(PASS);
      case NUKI -> List.of(claim(10, null));
    };
  }

  private static String discard(int tile) {
    return "{\"tag\":\"D\",\"p\":" + tile + "}";
  }

  private static String claim(int type, String tiles) {
    return "{\"tag\":\"N\",\"type\":" + type + (tiles == null ? "" : "," + tiles) + "}";
  }

  @Override
  public String toString() {
    return lines.stream().map(Line::message).collect(Collectors.joining("\n"));
  }
}
