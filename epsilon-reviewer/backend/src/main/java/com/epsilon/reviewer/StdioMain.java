package com.epsilon.reviewer;

import com.epsilon.reviewer.error.ErrorBody;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.json.Json;
import com.epsilon.reviewer.stdio.EngineConfig;
import com.epsilon.reviewer.stdio.FrameReader;
import com.epsilon.reviewer.stdio.FrameWriter;
import com.epsilon.reviewer.stdio.StdioHost;
import com.google.gson.JsonObject;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * デスクトップの子プロセスの入口（protocol/stdio.ts）。stdout は枠だけなので、最初に stdout を枠の書き手に取り、System.out を stderr へ
 * 向けてから他のクラスを使う（静的な Logger を持たない。記録は reviewer-stdio-logback.xml のファイルへ）。
 *
 * <p>保存先とカタログを開いたら {"event":"ready"}（モデルは開かない）。起動に失敗すれば {"event":"failed", error} を書いて状態 1 で終わる。
 * stdin の終わりで閉じて状態 0、壊れた枠は状態 1 で終わる。
 */
public final class StdioMain {
  private static final int MAX_OPEN_MODELS = 2;

  private StdioMain() {}

  public static void main(String[] args) {
    var frames = new FrameWriter(new FileOutputStream(FileDescriptor.out));
    System.setOut(System.err);
    EngineConfig config;
    try {
      config = EngineConfig.fromEnvironment();
    } catch (ReviewerException failure) {
      // 記録のフォルダーが分からないので記録しない。
      fail(frames, ErrorBody.of(failure));
      return;
    }
    Logger log = LoggerFactory.getLogger(StdioMain.class);
    Reviewer reviewer;
    try {
      reviewer = Reviewer.open(new ReviewerConfig(config.data(), config.models(), MAX_OPEN_MODELS));
    } catch (ReviewerException failure) {
      log.error("Could not start the reviewer engine: {}", failure.getMessage());
      fail(frames, ErrorBody.of(failure));
      return;
    } catch (IOException | RuntimeException failure) {
      log.error("Could not start the reviewer engine.", failure);
      fail(frames, ErrorBody.of(ErrorCode.SERVER_ERROR));
      return;
    }
    try {
      frames.write(event("ready"), null);
      new StdioHost(reviewer, new FrameReader(System.in), frames).run();
    } catch (IOException failure) {
      log.error("The reviewer engine stopped on a broken frame.", failure);
      System.exit(1);
    }
    System.exit(0);
  }

  private static void fail(FrameWriter frames, ErrorBody error) {
    JsonObject failed = event("failed");
    failed.add("error", Json.GSON.toJsonTree(error));
    try {
      frames.write(failed, null);
    } catch (IOException ignored) {
      // 読む側が居なければ状態 1 だけを残す。
    }
    System.exit(1);
  }

  private static JsonObject event(String name) {
    JsonObject event = new JsonObject();
    event.addProperty("event", name);
    return event;
  }
}
