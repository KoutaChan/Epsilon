package com.epsilon.reviewer.engine.format;

import com.epsilon.reviewer.engine.load.RecordSummary;
import com.epsilon.reviewer.engine.review.EvaluatedDecision;
import com.epsilon.reviewer.engine.view.EventKind;
import com.epsilon.reviewer.engine.view.RoundOutcome;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.engine.view.WinDetails;
import com.google.gson.stream.JsonWriter;
import java.io.Closeable;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;

/**
 * 解析結果の文書（protocol/result.ts）をストリームで書く。文書の全体をメモリに作らない。
 *
 * <p>局の手順は牌譜の順に書く。判断は、その判断が決まるイベントより前に書く。イベントの位置は、局の {@code firstEvent} から 1 つずつ増える。
 * 判断の {@code cause} は直前のイベントの位置なら省き、{@code resolve} は {@code cause + 1} なら省く。局面が前の手順と変わらなければ null を渡す。
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

  /** 書いた判断の値が損失（価値を返すモデルの評価）なら true、確率なら false。判断をまだ書いていなければ null。 */
  private Boolean valued;

  /**
   * @param createdAt 解析した時刻（ISO-8601、UTC）
   * @param record 解析する牌譜の概要。読み込んだ牌譜のものなので、最終の持ち点を持つ
   */
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
    json.name("finalScores").beginArray();
    for (int score : record.finalScores()) json.value(score);
    json.endArray();
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
   * @param actor 動いた席。席を持たない種類のイベントでは読まない
   * @param table 変わった局面。前の手順と同じなら null
   * @param outcome 和了（hora）か流局（ryukyoku）のイベントなら、局の結果。ほかのイベントでは null
   * @throws IllegalStateException イベントの位置が、前のイベントの次になっていない
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
   * @param cause 判断のきっかけになったイベントの位置
   * @param resolve 判断が決まったイベントの位置。牌譜の終わりまで決まらなかった判断では、イベントの数
   * @throws IllegalStateException 価値を返すモデルの評価と、返さないモデルの評価が混ざった
   */
  public void decision(int seat, int cause, int resolve, EvaluatedDecision decision)
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
    json.endObject();
  }

  /**
   * 文書を最後まで書く。
   *
   * @param temperature 損失から表示の確率を作るときの温度。価値を返さないモデルでは null
   * @throws IllegalStateException 温度の有無が、書いた評価の種類と合わない
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

  /** 書き込み先のストリームを閉じる。finish の前に閉じると、文書は途中で切れる。保存せずに捨てるときは、finish を呼ばずに閉じる。 */
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
        details(hora.details());
      }
      case RoundOutcome.Ryukyoku ignored -> json.name("type").value("ryukyoku");
    }
    json.endObject();
  }

  /** 和了の内訳。役満では符を書かず、裏ドラの表示牌は立直の和了のときだけ書く。 */
  private void details(WinDetails details) throws IOException {
    json.name("details").beginObject();
    json.name("han").value(details.han());
    if (details.yakuman() == 0) json.name("fu").value(details.fu());
    json.name("points").value(details.points());
    json.name("yakuman").value(details.yakuman());
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
    if (!details.uraIndicators().isEmpty()) {
      json.name("uraIndicators");
      TableJson.strings(json, details.uraIndicators());
    }
    json.endObject();
  }
}
