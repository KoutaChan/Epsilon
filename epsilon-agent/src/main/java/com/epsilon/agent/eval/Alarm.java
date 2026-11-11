package com.epsilon.agent.eval;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import java.util.Objects;

/**
 * リグレッションのアラートの 1 行（run の reports/alarms.jsonl）。同じ名前の行は 1 度だけ書く。
 *
 * @param alarm 行の名前（ジョブの名前と印を : でつないだもの）
 * @param job ジョブの名前
 * @param version 測ったバージョン
 * @param signal アラートの種類
 * @param estimate 候補が良い向きの値と区間。王者との決闘とラダーの卓では、その卓の平均と区間。低下のアラートでは、牌山ごとの対の差
 *     （今のバージョン − 王者）の平均と、平均 ± しきい値 × 標準誤差 の区間
 * @param handling 扱い
 * @param champion 低下のアラートで、牌山ごとの値を対にした王者のバージョン（ほかのアラートには無い）
 */
public record Alarm(
    String alarm,
    String job,
    int version,
    Signal signal,
    Estimate estimate,
    Handling handling,
    @Key(doc = "低下のアラートで、牌山ごとの値を対にした王者のバージョン", presence = Presence.ABSENT) Integer champion) {

  /** reports の中のファイルの名前。 */
  public static final String FILE = "alarms.jsonl";

  /** アラートの扱い。悪いが続くの印は設定（{@link OnWorseSettings}）の形で決まり、ほかの印はいつも REPORT。 */
  public enum Handling {
    /** 記録に書いて、そのまま続ける。 */
    REPORT,
    /** run の一時停止を要求する。 */
    PAUSE,
    /** 次の反復を王者のネットワークから始める。 */
    ROLLBACK
  }

  public Alarm {
    Objects.requireNonNull(alarm, "alarm");
    Objects.requireNonNull(job, "job");
    Objects.requireNonNull(signal, "signal");
    Objects.requireNonNull(estimate, "estimate");
    Objects.requireNonNull(handling, "handling");
  }

  /**
   * ジョブの印の行。
   *
   * @param champion 低下の印で対にした王者のバージョン。ほかの印は null
   */
  static Alarm of(String job, int version, Signal signal, Estimate estimate, Handling handling, Integer champion) {
    return new Alarm(job + ":" + signal, job, version, signal, estimate, handling, champion);
  }
}
