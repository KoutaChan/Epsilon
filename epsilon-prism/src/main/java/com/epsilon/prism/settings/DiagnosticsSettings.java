package com.epsilon.prism.settings;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.hip.HipLibrary;

/**
 * 診断のセクション。どれも結果の値を変えず、記録を足すか、検査つきのランタイムを選ぶだけ。JVM を起動したときに {@link #apply} で
 * 反映する。
 *
 * @param fitTiming 学習の計測（無ければ測らない）
 * @param hipChecked HIP のランタイムを検査つきのビルドにするか
 */
public record DiagnosticsSettings(
    @Key(doc = "学習の計測（無ければ測らない）", presence = Presence.ABSENT) FitTiming fitTiming,
    @Key(doc = "HIP のランタイムを検査つきのビルドにするか", value = "false") boolean hipChecked) {

  /**
   * 学習の計測。
   *
   * @param every 計るステップの間隔（every ステップに 1 ステップを同期して計る）
   * @param warmup 計らない始めのステップ数（ウォームアップ）
   */
  public record FitTiming(
      @Key(doc = "計るステップの間隔", presence = Presence.REQUIRED) @Range(min = 1, max = 999_999) int every,
      @Key(doc = "計らない始めのステップ数", presence = Presence.REQUIRED) @Range(min = 0, max = 999_999) int warmup) {}

  /** この JVM で追加する診断の出力を、このセクションの値に合わせる。ネットワークを開く前と、学習を始める前に呼ぶ。 */
  public void apply() {
    DiagnosticProperty.FIT_TIMING.set(fitTiming == null ? null : "sync:" + fitTiming.every() + ":" + fitTiming.warmup());
    System.setProperty(HipLibrary.CHECKED_PROPERTY, Boolean.toString(hipChecked));
  }
}
