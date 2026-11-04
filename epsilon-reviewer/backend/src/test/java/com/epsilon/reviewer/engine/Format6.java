package com.epsilon.reviewer.engine;

import com.epsilon.core.Action;
import com.epsilon.reviewer.engine.format.ModelInfo;
import com.epsilon.reviewer.engine.format.ResultWriter;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.review.Analysis;
import com.epsilon.reviewer.engine.review.JobControl;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.view.Meld;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.json.Json;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/** 解析結果の文書を戻す試験だけの読み手（書き手の部品を使わない）。各手順の局面を start と delta から作り直す。 */
public final class Format6 {
  public static final ModelInfo MODEL = new ModelInfo("fake", "fixed-revision", "Fake", "1");

  private static final JobControl FREE =
      new JobControl() {
        @Override
        public boolean cancelled() {
          return false;
        }

        @Override
        public void progress(int percent) {}
      };

  private Format6() {}

  /** 戻した文書。 */
  public record Document(JsonObject json, List<Round> rounds) {
    /** 全ての局の判断の手順（局の位置つき）。 */
    public List<Located> decisions() {
      List<Located> decisions = new ArrayList<>();
      for (int round = 0; round < rounds.size(); round++)
        for (Step step : rounds.get(round).steps())
          if (step.decision()) decisions.add(new Located(round, step));
      return decisions;
    }
  }

  /** 局の位置と判断の手順。 */
  public record Located(int round, Step step) {}

  public record Round(int roundIndex, int honba, int dealer, int firstEvent, List<Step> steps) {}

  /**
   * 1 手順。
   *
   * @param event 出来事の名前。判断は null
   * @param actor 出来事の席（無ければ -1）か判断の席
   * @param eventIndex 出来事の位置か、判断の resolve
   * @param cause 判断の cause。出来事は -1
   * @param table この手順の局面
   */
  public record Step(
      String event, int actor, int eventIndex, int cause, JsonObject json, TableState table) {
    public boolean decision() {
      return event == null;
    }

    /** candidates・chosen・probabilities の欄だけの写し。 */
    public JsonObject candidateSet() {
      JsonObject set = new JsonObject();
      for (String key : List.of("candidates", "chosen", "probabilities"))
        if (json.has(key)) set.add(key, json.get(key));
      return set;
    }

    public JsonArray candidates() {
      return json.getAsJsonArray("candidates");
    }

    /** 選んだ候補の位置。無ければ -1。 */
    public int chosen() {
      return json.has("chosen") ? json.get("chosen").getAsInt() : -1;
    }

    public int actionId(int candidate) {
      return candidates().get(candidate).getAsJsonArray().get(0).getAsInt();
    }

    public Action.Type type(int candidate) {
      return Action.fromIndex(actionId(candidate)).type();
    }

    /** 牌譜で選んだ手の種類。記録に無い判断は response。 */
    public String chosenType() {
      return chosen() < 0 ? "response" : type(chosen()).name();
    }

    public boolean offers(Action.Type type) {
      for (int i = 0; i < candidates().size(); i++) if (type(i) == type) return true;
      return false;
    }
  }

  /** 牌譜を解析して文書を記憶に書き、戻す。 */
  public static Document analyze(LoadedRecord record, ReviewModel model) {
    return decode(JsonParser.parseString(text(record, model)).getAsJsonObject());
  }

  /** 牌譜を解析した文書の JSON。 */
  public static String text(LoadedRecord record, ReviewModel model) {
    var out = new ByteArrayOutputStream();
    try (var writer =
        new ResultWriter(
            out,
            "00000000-0000-4000-8000-000000000000",
            "2026-11-04T00:00:00Z",
            MODEL,
            record.summary())) {
      Analysis.run(record, model, FREE, writer);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    return out.toString(StandardCharsets.UTF_8);
  }

  public static Document decode(JsonObject json) {
    List<Round> rounds = new ArrayList<>();
    for (JsonElement element : json.getAsJsonArray("rounds")) {
      JsonObject round = element.getAsJsonObject();
      int firstEvent = round.get("firstEvent").getAsInt();
      TableState table = table(round.getAsJsonObject("start"));
      List<Step> steps = new ArrayList<>();
      int events = 0;
      int lastEvent = -1;
      for (JsonElement item : round.getAsJsonArray("steps")) {
        JsonObject step = item.getAsJsonObject();
        if (step.has("delta")) table = apply(table, step.getAsJsonObject("delta"));
        if (step.has("event")) {
          lastEvent = firstEvent + events++;
          int actor = step.has("actor") ? step.get("actor").getAsInt() : -1;
          steps.add(new Step(step.get("event").getAsString(), actor, lastEvent, -1, step, table));
        } else {
          int cause = step.has("cause") ? step.get("cause").getAsInt() : lastEvent;
          int resolve = step.has("resolve") ? step.get("resolve").getAsInt() : cause + 1;
          steps.add(new Step(null, step.get("seat").getAsInt(), resolve, cause, step, table));
        }
      }
      rounds.add(
          new Round(
              round.get("roundIndex").getAsInt(),
              round.get("honba").getAsInt(),
              round.get("dealer").getAsInt(),
              firstEvent,
              steps));
    }
    return new Document(json, rounds);
  }

  /**
   * もしもの答えの candidates・chosen・probabilities の欄だけ。数を文書と同じ十進の字で比べるため、書いた JSON を読み直す（float の木は
   * double に広げた値で比べられる）。
   */
  public static JsonObject candidateSet(ScenarioAnswer answer) {
    JsonObject json =
        JsonParser.parseString(new String(Json.write(answer), StandardCharsets.UTF_8))
            .getAsJsonObject();
    json.remove("table");
    return json;
  }

  static TableState table(JsonObject json) {
    List<PlayerState> players = new ArrayList<>();
    for (JsonElement player : json.getAsJsonArray("players"))
      players.add(player(player.getAsJsonObject()));
    return new TableState(
        players,
        strings(json.getAsJsonArray("dora")),
        json.get("honba").getAsInt(),
        json.get("kyotaku").getAsInt(),
        json.get("remaining").getAsInt(),
        json.get("activeSeat").getAsInt());
  }

  private static PlayerState player(JsonObject json) {
    List<RiverTile> river = new ArrayList<>();
    for (JsonElement tile : json.getAsJsonArray("river")) river.add(riverTile(tile));
    List<Meld> melds = new ArrayList<>();
    for (JsonElement meld : json.getAsJsonArray("melds")) melds.add(meld(meld));
    return new PlayerState(
        json.get("score").getAsInt(),
        json.get("riichi").getAsBoolean(),
        strings(json.getAsJsonArray("hand")),
        json.get("draw").isJsonNull() ? null : json.get("draw").getAsString(),
        river,
        melds);
  }

  private static RiverTile riverTile(JsonElement element) {
    JsonObject json = element.getAsJsonObject();
    return new RiverTile(
        json.get("tile").getAsString(),
        flag(json, "tsumogiri"),
        flag(json, "riichi"),
        flag(json, "called"));
  }

  private static boolean flag(JsonObject json, String name) {
    if (!json.has(name)) return false;
    if (!json.get(name).getAsBoolean()) throw new AssertionError(name + " is written as false.");
    return true;
  }

  private static Meld meld(JsonElement element) {
    JsonObject json = element.getAsJsonObject();
    return new Meld(
        json.get("type").getAsString(),
        strings(json.getAsJsonArray("tiles")),
        json.get("from").getAsInt(),
        json.get("calledIndex").getAsInt(),
        json.get("addedIndex").getAsInt());
  }

  private static TableState apply(TableState before, JsonObject delta) {
    List<PlayerState> players = new ArrayList<>(before.players());
    if (delta.has("players"))
      for (JsonElement item : delta.getAsJsonArray("players")) {
        JsonObject update = item.getAsJsonObject();
        int seat = update.get("seat").getAsInt();
        PlayerState old = players.get(seat);
        String draw = old.draw();
        if (update.has("draw"))
          draw = update.get("draw").isJsonNull() ? null : update.get("draw").getAsString();
        players.set(
            seat,
            new PlayerState(
                update.has("score") ? update.get("score").getAsInt() : old.score(),
                update.has("riichi") ? update.get("riichi").getAsBoolean() : old.riichi(),
                splice(old.hand(), update, "hand", JsonElement::getAsString),
                draw,
                splice(old.river(), update, "river", Format6::riverTile),
                splice(old.melds(), update, "melds", Format6::meld)));
      }
    return new TableState(
        players,
        splice(before.dora(), delta, "dora", JsonElement::getAsString),
        delta.has("honba") ? delta.get("honba").getAsInt() : before.honba(),
        delta.has("kyotaku") ? delta.get("kyotaku").getAsInt() : before.kyotaku(),
        delta.has("remaining") ? delta.get("remaining").getAsInt() : before.remaining(),
        delta.has("activeSeat") ? delta.get("activeSeat").getAsInt() : before.activeSeat());
  }

  /** [位置, 消す数, ...値] を当てる。 */
  private static <T> List<T> splice(
      List<T> before, JsonObject delta, String name, Function<JsonElement, T> value) {
    if (!delta.has(name)) return before;
    JsonArray splice = delta.getAsJsonArray(name);
    int index = splice.get(0).getAsInt();
    int removed = splice.get(1).getAsInt();
    List<T> after = new ArrayList<>(before);
    after.subList(index, index + removed).clear();
    List<T> values = new ArrayList<>();
    for (int i = 2; i < splice.size(); i++) values.add(value.apply(splice.get(i)));
    after.addAll(index, values);
    return after;
  }

  private static List<String> strings(JsonArray array) {
    List<String> values = new ArrayList<>(array.size());
    for (JsonElement value : array) values.add(value.getAsString());
    return values;
  }
}
