package com.epsilon.agent.app;

import com.epsilon.agent.settings.SettingsException;

/**
 * コマンド。record のコンポーネントが引数になる。引数は、コンポーネントの名前を小文字とハイフンにした
 * {@code --名前} で渡す（{@code stopAfter} なら {@code --stop-after}）。コンポーネントには、設定と同じキーの注釈を付ける。
 * 設定ファイルにも、コマンドの名前をパスにしたセクションとして書ける（{@code settings change} なら {@code [settings.change]}）。
 *
 * @param <R> アプリ全体の設定の record
 */
public interface Command<R extends Record> {

  /**
   * コマンドを実行する。設定と引数の誤りは {@link SettingsException} で知らせる。入口は、それを使い方の誤りとして終える。
   *
   * @return 終了の種類と、終わりのログの行に足す要約
   */
  Outcome run(Context<R> context) throws Exception;
}
