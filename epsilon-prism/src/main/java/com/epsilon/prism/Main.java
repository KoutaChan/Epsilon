package com.epsilon.prism;

import com.epsilon.agent.app.App;
import com.epsilon.prism.app.PrismApp;

/** epsilon-prism を起動する。 */
public final class Main {

  private Main() {}

  /** コマンドを 1 つ実行して終了コードを返す。シグナルのハンドラーは登録しない。 */
  public static int run(String[] arguments) {
    return App.run(PrismApp.application(), arguments);
  }

  /** コマンドを 1 つ実行して、その終了コードで JVM を終える。 */
  public static void main(String[] arguments) {
    App.main(PrismApp.application(), arguments);
  }
}
