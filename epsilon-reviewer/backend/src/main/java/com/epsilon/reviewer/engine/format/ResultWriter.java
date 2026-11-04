package com.epsilon.reviewer.engine.format;

import com.epsilon.replay.WinDetails;
import com.epsilon.reviewer.engine.load.RecordSummary;
import com.epsilon.reviewer.engine.review.EvaluatedDecision;
import com.epsilon.reviewer.engine.view.EventKind;
import com.epsilon.reviewer.engine.view.RoundOutcome;
import com.epsilon.reviewer.engine.view.TableState;
import com.google.gson.stream.JsonWriter;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;

/**
 * 解析結果の文書（protocol/result.ts）を流して書く。文書の全体を記憶に作らない。
 *
 * <p>局の手順は牌譜の順で、判断はそれが解く出来事の前に呼ぶ。出来事の手順の位置は局の firstEvent から 1 つずつ続く。判断の cause は直前の
 * 出来事の位置、resolve は cause + 1 のとき書かない。局面は前の手順と同じなら渡さない（null）。
 */
public final class ResultWriter implements Closeable {
  public static final int FORMAT = 6;

  private final Writer text;
  private final JsonWriter json;
  private boolean inRound;
  private int firstEvent;
  private int events;
  private int lastEvent;
  private TableState previous;

  /** 書いた判断が損失か。判断がまだ無ければ null。 */
  private Boolean valued;

  /** @param createdAt 解析した時刻（ISO-8601、UTC） */
  public ResultWriter(
      OutputStream out, String resultId, String createdAt, ModelInfo model, RecordSummary record)
      throws IOException {
    text = new OutputStreamWriter(out, StandardCharsets.UTF_8);
    json = new JsonWriter(text);
    json.beginObject().name("format").value(FORMAT);
    json.name("resultId").value(resultId);
    json.name("createdAt").value(createdAt);
    json.name("model").beginObject();
    json.name("modelId").value(model.modelId());
    json.name("revision").value(model.revision());
    json.name("displayName").value(model.displayName());
    json.name("version").value(model.version());
    json.endObject().name("record").beginObject();
    json.name("source").value(record.source());
    json.name("names");
    TableJson.strings(json, record.names());
    if (record.finalScores() != null) {
      json.name("finalScores").beginArray();
      for (int score : record.finalScores()) json.value(score);
      json.endArray();
    }
    json.name("roundCount").value(record.roundCount());
    json.endObject().name("rounds").beginArray();
  }

  /** @param start 局の最初の手順（start_kyoku）の局面 */
  public void startRound(int roundIndex, int honba, int dealer, int firstEvent, TableState start)
      throws IOException {
    if (inRound) json.endArray().endObject();
    json.beginObject().name("roundIndex").value(roundIndex);
    json.name("honba").value(honba);
    json.name("dealer").value(dealer);
    json.name("firstEvent").value(firstEvent);
    json.name("start");
    TableJson.write(json, start);
    json.name("steps").beginArray();
    inRound = true;
    this.firstEvent = firstEvent;
    events = 0;
    previous = start;
  }

  /**
   * @param actor 席を持たない種類では読まない
   * @param table 変わった局面。前の手順と同じなら null
   * @param outcome hora・ryukyoku の結果。他は null
   * @throws IllegalStateException 出来事の位置が続かない
   */
  public void event(
      int eventIndex, EventKind kind, int actor, TableState table, RoundOutcome outcome)
      throws IOException {
    requireRound();
    if (eventIndex != firstEvent + events)
      throw new IllegalStateException(
          "Event step "
              + eventIndex
              + " does not follow event "
              + (firstEvent + events - 1)
              + " of the round.");
    json.beginObject().name("event").value(kind.wireName());
    if (kind.hasActor()) json.name("actor").value(actor);
    delta(table);
    if (outcome != null) outcome(outcome);
    json.endObject();
    lastEvent = eventIndex;
    events++;
  }

  /**
   * @param cause 判断を起こした出来事の位置
   * @param resolve 判断が解かれた出来事の位置（牌譜の終わりで開いた窓は出来事の数）
   * @param table 変わった局面。前の手順と同じなら null
   * @throws IllegalStateException 価値のある評価と無い評価が混ざる
   */
  public void decision(
      int seat, int cause, int resolve, EvaluatedDecision decision, TableState table)
      throws IOException {
    requireRound();
    if (events == 0) throw new IllegalStateException("A round must start with an event step.");
    if (valued == null) valued = decision.valued();
    else if (valued != decision.valued())
      throw new IllegalStateException("Valued and policy-only evaluations are mixed.");
    json.beginObject().name("seat").value(seat);
    if (cause != lastEvent) json.name("cause").value(cause);
    if (resolve != cause + 1) json.name("resolve").value(resolve);
    CandidateJson.write(json, decision);
    delta(table);
    json.endObject();
  }

  /**
   * 文書を閉じる。
   *
   * @param temperature 価値のある評価の表示の温度。価値の無いモデルは null
   * @throws IllegalStateException 温度の有無が評価と合わない
   */
  public void finish(Double temperature) throws IOException {
    if (valued != null && valued != (temperature != null))
      throw new IllegalStateException("The temperature does not match the evaluations.");
    if (inRound) json.endArray().endObject();
    inRound = false;
    json.endArray();
    if (temperature != null) json.name("temperature").value(temperature.doubleValue());
    json.endObject().flush();
  }

  /** 下の流れを閉じる。finish の前に閉じた文書は途中で切れる（捨てる保存で使う）。 */
  @Override
  public void close() throws IOException {
    text.close();
  }

  private void requireRound() {
    if (!inRound) throw new IllegalStateException("No round has started.");
  }

  private void delta(TableState table) throws IOException {
    if (table == null) return;
    TableDiff.write(json, previous, table);
    previous = table;
  }

  private void outcome(RoundOutcome outcome) throws IOException {
    json.name("outcome").beginObject();
    switch (outcome) {
      case RoundOutcome.Hora hora -> {
        json.name("type").value("hora");
        json.name("actor").value(hora.actor());
        json.name("target").value(hora.target());
        json.name("winningTile").value(hora.winningTile());
        if (hora.details() != null) details(hora.details());
      }
      case RoundOutcome.Ryukyoku ignored -> json.name("type").value("ryukyoku");
    }
    json.endObject();
  }

  /** 和了の内訳。分からない欄は書かない（uraIndicators の空の配列は「見た記録があり 0 枚」）。 */
  private void details(WinDetails details) throws IOException {
    json.name("details").beginObject();
    optional("han", details.han());
    optional("fu", details.fu());
    optional("points", details.points());
    optional("yakuman", details.yakuman());
    if (details.yaku() != null) {
      json.name("yaku").beginArray();
      for (WinDetails.Yaku yaku : details.yaku())
        json.beginObject()
            .name("code")
            .value(yaku.code())
            .name("han")
            .value(yaku.han())
            .name("yakuman")
            .value(yaku.yakuman())
            .endObject();
      json.endArray();
    }
    if (details.uraIndicators() != null) {
      json.name("uraIndicators");
      TableJson.strings(json, details.uraIndicators());
    }
    json.endObject();
  }

  private void optional(String name, Integer value) throws IOException {
    if (value != null) json.name(name).value(value.intValue());
  }
}
