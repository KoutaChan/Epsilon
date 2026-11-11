package com.epsilon.reviewer.engine;

import com.epsilon.mahjong.action.Action;
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

/** テストだけで使う、解析結果の文書のリーダー（ライターの部品は使わない）。各手順の局面は、局の start に各手順の delta を順に反映して作り直す。 */
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

  /** 読み戻した文書。 */
  public record Document(JsonObject json, List<Round> rounds) {
    /** 全ての局から判断の手順を集め、どの局のものかを添えて返す。 */
    public List<Located> decisions() {
      List<Located> decisions = new ArrayList<>();
      for (int round = 0; round < rounds.size(); round++)
        for (Step step : rounds.get(round).steps())
          if (step.decision()) decisions.add(new Located(round, step));
      return decisions;
    }
  }

  /** 判断の手順と、それがある局（文書の中で何番目の局か。0 から）。 */
  public record Located(int round, Step step) {}

  public record Round(int roundIndex, int honba, int dealer, int firstEvent, List<Step> steps) {}

  /**
   * 文書の 1 つの手順。イベントか判断のどちらか。
   *
   * @param event イベントの名前。判断なら null
   * @param actor イベントなら動いた席（無ければ -1）、判断なら判断する席
   * @param eventIndex イベントならイベントの位置、判断なら判断が決まる位置（resolve）
   * @param cause 判断ならきっかけのイベントの位置（cause）。イベントなら -1
   * @param table この手順の局面
   */
  public record Step(
      String event, int actor, int eventIndex, int cause, JsonObject json, TableState table) {
    public boolean decision() {
      return event == null;
    }

    /** candidates と chosen のフィールドだけをコピーした JSON。 */
    public JsonObject candidateSet() {
      JsonObject set = new JsonObject();
      for (String key : List.of("candidates", "chosen"))
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
      return Action.of(actionId(candidate)).type();
    }

    /** 牌譜で選んだ手の種類の名前。牌譜に記録されていない判断なら response。 */
    public String chosenType() {
      return chosen() < 0 ? "response" : type(chosen()).name();
    }

    public boolean offers(Action.Type type) {
      for (int i = 0; i < candidates().size(); i++) if (type(i) == type) return true;
      return false;
    }
  }

  /** 牌譜を解析して文書をメモリに書き、読み戻す。 */
  public static Document analyze(LoadedRecord record, ReviewModel model) {
    return decode(JsonParser.parseString(text(record, model)).getAsJsonObject());
  }

  /** 牌譜を解析し、文書の JSON を文字列で返す。 */
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
   * もしもの応答から table を除き、残りのフィールド（candidates と chosen）を返す。数を文書と同じ十進の文字列で比べるため、
   * 一度 JSON の文字列に書いてから読み直す。JSON の木に直接変えると、float を double に広げた値になってしまう。
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

  /** [位置, 消す数, ...値] の形の差分をリストに反映し、新しいリストを返す。差分が無ければ元のリストを返す。 */
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
