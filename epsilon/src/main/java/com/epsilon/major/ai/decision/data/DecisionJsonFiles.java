package com.epsilon.major.ai.decision.data;

import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** 学習状態・対局レポートのJSONを、同じディレクトリの一時ファイルから置換する。 */
public final class DecisionJsonFiles {
  private DecisionJsonFiles() {}

  public static void write(Path file, Object value, Gson gson) throws IOException {
    Path target = file.toAbsolutePath().normalize();
    Files.createDirectories(target.getParent());
    Path temporary = Files.createTempFile(target.getParent(), ".decision-", ".json");
    try {
      Files.writeString(temporary, gson.toJson(value));
      try {
        Files.move(
            temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
      } catch (AtomicMoveNotSupportedException unsupported) {
        Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
      }
    } finally {
      Files.deleteIfExists(temporary);
    }
  }
}
