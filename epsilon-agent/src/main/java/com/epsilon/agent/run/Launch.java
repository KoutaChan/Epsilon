package com.epsilon.agent.run;

import java.util.Objects;

/**
 * JVM を起動したときに決まる値。リソースはこの値で開き、JVM が動いている間は開き直さない。
 *
 * @param run run のディレクトリ
 * @param control セーフポイントの確認（段の中で待つときと、バックグラウンドの処理を一時停止するときに使う）
 * @param row 起動したときの最新の設定の行（速さに関わるキーは、この行の値を使う）
 * @param settings その行の設定
 * @param jar この JVM の jar の ID
 * @param <R> run 全体の設定
 */
public record Launch<R extends Record>(RunDirectory run, RunControl control, int row, R settings, String jar) {

  public Launch {
    Objects.requireNonNull(run, "run");
    Objects.requireNonNull(control, "control");
    Objects.requireNonNull(settings, "settings");
    Objects.requireNonNull(jar, "jar");
  }
}
