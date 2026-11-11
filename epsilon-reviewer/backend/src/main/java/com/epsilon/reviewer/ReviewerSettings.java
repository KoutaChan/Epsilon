package com.epsilon.reviewer;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import com.epsilon.reviewer.analysis.AnalysisSettings;

/**
 * Reviewer 全体の設定。HTTP サーバとデスクトップのエンジンが読む。コマンドの引数は、コマンドと同じ名前のセクション（{@code [serve]} と
 * {@code [engine]}）にも書ける。
 *
 * @param server モデルと HTTP サーバの上限
 * @param analysis 解析ジョブの設定
 */
public record ReviewerSettings(
    @Key(doc = "モデルと HTTP サーバの上限", change = Change.MACHINE) Server server,
    @Key(doc = "解析ジョブの設定", change = Change.MACHINE) AnalysisSettings analysis) {

  /**
   * モデルと HTTP サーバの上限。
   *
   * @param maxOpenModels 同時に開いておけるモデルの数
   * @param requestSeconds 要求を読み終えるまでの制限時間（秒）。JDK の HTTP サーバが最初に作られるときに読む
   * @param responseSeconds 応答を書き終えるまでの制限時間（秒）。JDK の HTTP サーバが最初に作られるときに読む
   * @param maxConnections 同時接続数の上限。JDK の HTTP サーバが最初に作られるときに読む
   */
  public record Server(
      @Key(doc = "同時に開いておけるモデルの数", value = "2") @Range(min = 1) int maxOpenModels,
      @Key(doc = "要求を読み終えるまでの制限時間（秒）", value = "60") @Range(min = 1) int requestSeconds,
      @Key(doc = "応答を書き終えるまでの制限時間（秒）", value = "120") @Range(min = 1) int responseSeconds,
      @Key(doc = "同時接続数の上限", value = "64") @Range(min = 1) int maxConnections) {}
}
