package com.epsilon.nano.ai.decision.data;

import com.epsilon.io.AtomicFiles;
import com.google.gson.Gson;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/** 学習状態・対局レポートのJSONを、同じディレクトリの一時ファイルから置換する。 */
public final class DecisionJsonFiles {
  private DecisionJsonFiles() {}

  public static void write(Path file, Object value, Gson gson) throws IOException {
    AtomicFiles.replace(file, gson.toJson(value).getBytes(StandardCharsets.UTF_8));
  }
}
