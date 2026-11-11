package com.epsilon.agent.app;

/** コマンドが読む設定。 */
public enum Reads {
  /** 同梱の既定値、ルールごとの既定の層、プリセット、設定ファイル、{@code --set} を重ねた、アプリ全体の設定を読む。 */
  LAYERS,
  /**
   * run を動かし、run の記録にある設定だけを読む。{@code --settings}・{@code --preset}・{@code --set} を受けたら、
   * {@code settings change} を使うよう知らせて、使い方の誤りで止まる。
   */
  RUN
}
