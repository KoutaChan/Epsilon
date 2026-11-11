package com.epsilon.agent.run;

import java.nio.file.Path;

/**
 * 1 つの反復。反復の番号、作業ディレクトリ、固定した設定の行と jar を持つ。反復 t はバージョン t で集め、バージョン t + 1 を
 * 公開する。
 */
public interface Iteration {

  int number();

  /** 段の途中の出力と完了マーカーを置くディレクトリ（{@link RunDirectory#work(int)}）。 */
  Path work();

  /** 固定した設定の行。 */
  int settingsRow();

  /** 固定した jar の ID（段を回す JVM の jar と同じ）。 */
  String jar();

  /** 段を回す JVM を起動したときの設定の行（段の報告に書く）。 */
  int launchRow();
}
