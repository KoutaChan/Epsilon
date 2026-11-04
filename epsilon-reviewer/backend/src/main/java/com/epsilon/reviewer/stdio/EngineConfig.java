package com.epsilon.reviewer.stdio;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;

/**
 * デスクトップの子プロセスの設定。パスは日本語を含み得るので引数でなく環境変数で受ける（protocol/stdio.ts の ENGINE_ENV）。
 *
 * @param data 解析結果の保存先（EPSILON_REVIEWER_DATA）
 * @param models 同梱の models.json（EPSILON_REVIEWER_MODELS）
 * @param logs 記録のフォルダー（EPSILON_REVIEWER_LOGS。reviewer-stdio-logback.xml が読む）
 */
public record EngineConfig(Path data, Path models, Path logs) {
  /** @throws ReviewerException invalid_request（無い・空・パスでない） */
  public static EngineConfig fromEnvironment() {
    return new EngineConfig(
        path("EPSILON_REVIEWER_DATA"),
        path("EPSILON_REVIEWER_MODELS"),
        path("EPSILON_REVIEWER_LOGS"));
  }

  private static Path path(String name) {
    String value = System.getenv(name);
    if (value == null || value.isBlank())
      throw new ReviewerException(ErrorCode.INVALID_REQUEST, name + " is not set.");
    try {
      return Path.of(value);
    } catch (InvalidPathException failure) {
      throw new ReviewerException(ErrorCode.INVALID_REQUEST, name + " is not a valid path.");
    }
  }
}
