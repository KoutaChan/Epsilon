package com.epsilon.agent.app;

import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import java.io.PrintStream;
import java.util.List;

/**
 * コマンドが引数のほかに読むもの。
 *
 * @param <R> アプリ全体の設定の record
 */
public interface Context<R extends Record> {

  Application<R> application();

  /**
   * アプリ全体の設定。同梱の既定値、ルールごとの既定の層、プリセット、設定ファイル、{@code --set} を重ねて、初めて呼んだときに
   * 1 回だけ作る。
   *
   * @throws SettingsException 設定をバインドできない
   * @throws IllegalStateException run を動かすコマンドが呼んだ。そのコマンドは run の記録から設定を読む
   */
  Settings<R> settings();

  /**
   * 全体の設定に重ねる層。プリセット、設定ファイル、{@code --set} の順に並ぶ。ルールごとの既定の層とコマンドのセクションは含まない。
   */
  List<Layer> layers();

  /** 設定ファイルの先頭に書いたプリセットと、{@code --preset} で指定したプリセットの名前。この順に並ぶ。 */
  List<String> presets();

  /** コマンドの結果（TOML や使い方の表示）を書く先。ログは SLF4J に書く。 */
  PrintStream out();

  /**
   * SIGTERM と SIGINT を、次のセーフポイントで止める要求として受ける。これを呼んだコマンドは、要求があればセーフポイントで止まって
   * {@link Exit#STOPPED} を返す。呼ばないコマンドは、割り込みですぐに終わる。
   */
  Signals signals();

  /**
   * run の記録から設定を読むコマンドが、重ねる層を受けていないことを確かめる。
   *
   * @throws SettingsException {@code --settings}・{@code --preset}・{@code --set} のどれかを受けた
   */
  default void requireNoLayers() {
    if (!layers().isEmpty()) {
      throw new SettingsException(
          List.of("This command reads the run's settings: drop --settings, --preset and --set"));
    }
  }
}
