package com.epsilon.prism.run;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.luck.LedgerReport;
import java.util.Objects;

/**
 * 学習の段の報告。
 *
 * @param start 生徒の学習を始めたバージョン（アンカーも同じバージョン）
 * @param rows 生徒の行の数
 * @param holdout コホートのうち、ホールドアウトの半荘の行の数
 * @param carryReset 前の反復からの持ち越しを捨てた理由（捨てなければ無い）
 * @param tempo 教師のゲインと学習率の倍率
 * @param ledger 帳簿の当てはめの要約（当てはめなければ無い）
 * @param appliedR2 教師に掛けた補正の、ホールドアウトでの R²（補正を掛けなければ 0）。次の反復はこの値から K′ を決める
 * @param fit 生徒の学習の結果
 */
public record LearnReport(
    int start,
    int rows,
    long holdout,
    @Key(doc = "持ち越しを捨てた理由", presence = Presence.ABSENT) String carryReset,
    Tempo tempo,
    @Key(doc = "帳簿の当てはめの要約", presence = Presence.ABSENT) LedgerReport.Summary ledger,
    double appliedR2,
    Fit fit) {

  /** 学習の段の出力に入る、クラスごとの帳簿の当てはめのファイル（{@link LedgerReport.Classes}）。 */
  public static final String CLASSES = "classes.json";

  /** 学習の段の出力に入る、帳簿の当てはめの報告を全て書いたファイル（{@link LedgerReport.Full}）。 */
  public static final String LEDGER = "ledger.json";

  /** reports に置く、帳簿を当てはめた反復ごとの報告。1 行が 1 反復で、{@value #LEDGER} の中身に反復の番号を付けたもの。 */
  public static final String LEDGER_LINES = "ledger.jsonl";

  public LearnReport {
    Objects.requireNonNull(tempo, "tempo");
    Objects.requireNonNull(fit, "fit");
  }

  /**
   * 教師のゲインと学習率の倍率。
   *
   * @param gain 教師のゲイン K′_t
   * @param anchorWeight アンカーの重み 1 − K′_t（アンカーを混ぜなければ 0）
   * @param lrScale 学習率の倍率
   * @param scheduledGain スケジュールで決まるゲイン K_t
   * @param r2 K′_t を決めるのに使った R²（持ち越しを捨てたら 0）
   */
  public record Tempo(double gain, double anchorWeight, double lrScale, double scheduledGain, double r2) {}

  /**
   * 生徒の学習の結果。有限でない損失は書かない（ステップが無ければ損失も無い）。
   *
   * @param rows 使える行の数
   * @param steps ステップ数
   * @param clippedSteps 勾配のノルムを上限で切ったステップ数の見積もり
   * @param loss 回収したステップの損失の平均
   * @param firstLoss 最初に回収したステップの損失
   * @param lastLoss 最後に回収したステップの損失
   * @param seconds 学習にかかった時間（秒）
   */
  public record Fit(
      int rows,
      int steps,
      int clippedSteps,
      @Key(doc = "回収したステップの損失の平均", presence = Presence.ABSENT) Double loss,
      @Key(doc = "最初に回収したステップの損失", presence = Presence.ABSENT) Double firstLoss,
      @Key(doc = "最後に回収したステップの損失", presence = Presence.ABSENT) Double lastLoss,
      double seconds) {

    /** 学習の記録から作る。 */
    public static Fit of(FitResult result) {
      return new Fit(
          result.rows(),
          result.steps(),
          result.clippedSteps(),
          finite(result.loss()),
          finite(result.firstLoss()),
          finite(result.lastLoss()),
          result.seconds());
    }
  }

  private static Double finite(double value) {
    return Double.isFinite(value) ? value : null;
  }
}
