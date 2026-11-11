package com.epsilon.reviewer;

import com.epsilon.agent.app.App;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Toml;
import com.epsilon.reviewer.error.ErrorBody;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.json.Json;
import com.epsilon.reviewer.stdio.FrameWriter;
import com.google.gson.JsonObject;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

/**
 * デスクトップアプリの子プロセスの入口（protocol/stdio.ts）。標準出力にはフレームだけを書く。そのため、最初に標準出力をフレーム用に取り出し、
 * {@code System.out} を標準エラーへ向けてから、ほかのクラスを使う。静的な Logger を持たないのも同じ理由で、ログは reviewer-stdio-logback.xml で
 * 決めたファイルへ書く。
 *
 * <p>日本語を含みうるパスは、引数でなく環境変数で受け取る。受け取った値は {@code engine} のコマンドのキーへの {@code --set} に直し、Reviewer の
 * アプリで実行する。環境変数が無いときや、引数か設定が誤っていて engine を始められないときは、{"event":"failed", error} を書いて終了コード 1 で終わる。
 */
public final class StdioMain {

  /** 環境変数と、その値を渡す engine のコマンドのキー。ログのフォルダーはログの設定だけが読むので、キーにはしない。 */
  private enum Variable {
    DATA("EPSILON_REVIEWER_DATA", "dataDir"),
    MODELS("EPSILON_REVIEWER_MODELS", "models"),
    LOGS("EPSILON_REVIEWER_LOGS", null);

    private final String environment;
    private final String key;

    Variable(String environment, String key) {
      this.environment = environment;
      this.key = key;
    }
  }

  private StdioMain() {}

  public static void main(String[] args) {
    PrintStream frames = new PrintStream(new FileOutputStream(FileDescriptor.out), false);
    System.setOut(System.err);
    List<String> arguments = new ArrayList<>();
    arguments.add("engine");
    for (Variable variable : Variable.values()) {
      String value = System.getenv(variable.environment);
      if (value == null || value.isBlank()) {
        // ログのフォルダーが分からないので、ログは書かない。
        fail(frames, ErrorBody.of(ErrorCode.INVALID_REQUEST, variable.environment + " is not set."));
        return;
      }
      if (variable.key != null) {
        arguments.add("--set");
        arguments.add("engine." + variable.key + "=" + Toml.inline(new Node.Text(value, variable.environment)));
      }
    }
    arguments.addAll(List.of(args));
    int code = App.run(Main.application(), arguments.toArray(String[]::new), frames);
    if (code == Exit.USAGE.code()) {
      fail(frames, ErrorBody.of(ErrorCode.INVALID_REQUEST, "The engine settings are invalid; see the engine log."));
      return;
    }
    System.exit(code);
  }

  private static void fail(PrintStream frames, ErrorBody error) {
    JsonObject failed = new JsonObject();
    failed.addProperty("event", "failed");
    failed.add("error", Json.GSON.toJsonTree(error));
    try {
      new FrameWriter(frames).write(failed, null);
    } catch (IOException ignored) {
      // 読む側がいなければ、終了コード 1 だけを残す。
    }
    System.exit(1);
  }
}
