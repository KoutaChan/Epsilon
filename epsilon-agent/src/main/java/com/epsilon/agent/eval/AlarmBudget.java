package com.epsilon.agent.eval;

import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;
import java.util.Objects;

/**
 * アラートの予算。全ての真の差が 0 のときに、偶然で出るアラートの数の期待値。1 回ごとの確率は、王者との決闘とラダーの卓では
 * 両側の有意水準の半分、低下のアラートでは標準正規分布の値がしきい値を超える確率（低下の幅を掛ける前の上限）。「悪い」が続いたときのアラートは、続く回数ぶんの積になる。
 *
 * @param promotions 誤った昇格の数の期待値
 * @param alarms アラートの種類ごとの数の期待値
 */
public record AlarmBudget(double promotions, Map<Signal, Double> alarms) {

  public AlarmBudget {
    Objects.requireNonNull(alarms, "alarms");
  }

  /**
   * @param watch 監視の設定（「悪い」が続いたときの扱いは、続いた回数だけを使う）
   * @param exposure 測る回数と有意水準
   */
  public static AlarmBudget of(Watch watch, Exposure exposure) {
    double duelSide = exposure.championAlpha() / 2;
    double alarmTail = Normal.upper(watch.alarmZ());
    Map<Signal, Double> alarms = new EnumMap<>(Signal.class);
    alarms.put(Signal.CHAMPION_WORSE, exposure.championDuels() * duelSide);
    int streak = watch.onWorse().streak();
    alarms.put(Signal.WORSE_STREAK, Math.max(0, exposure.championDuels() - streak + 1) * Math.pow(duelSide, streak));
    alarms.put(Signal.LADDER_WORSE, exposure.ladders() * exposure.ladderAlpha() / 2);
    alarms.put(Signal.BASELINE_DROP, exposure.baselines() * alarmTail);
    alarms.put(Signal.ANCHOR_DROP, exposure.anchors() * alarmTail);
    return new AlarmBudget(exposure.championDuels() * duelSide, Collections.unmodifiableMap(alarms));
  }

  /**
   * アラートを出しうる卓を打つ回数と、その有意水準。
   *
   * @param championDuels 王者との決闘の回数
   * @param championAlpha 王者との決闘の両側の有意水準
   * @param ladders ラダーの卓の回数（悪くなったかを見る卓だけ）
   * @param ladderAlpha ラダーの卓の両側の有意水準
   * @param baselines 基準の卓を王者と比べる回数
   * @param anchors ルールベースの打ち手との卓を王者と比べる回数
   */
  public record Exposure(int championDuels, double championAlpha, int ladders, double ladderAlpha, int baselines, int anchors) {
    public Exposure {
      if (championDuels < 0 || ladders < 0 || baselines < 0 || anchors < 0) {
        throw new IllegalArgumentException(
            "Counts must be non-negative: " + championDuels + ", " + ladders + ", " + baselines + ", " + anchors);
      }
      if (!(championAlpha > 0 && championAlpha < 1) || !(ladderAlpha > 0 && ladderAlpha < 1)) {
        throw new IllegalArgumentException("Alpha must be in (0, 1): " + championAlpha + ", " + ladderAlpha);
      }
    }
  }
}
