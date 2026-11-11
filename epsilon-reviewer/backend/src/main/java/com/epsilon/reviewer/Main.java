package com.epsilon.reviewer;

import com.epsilon.agent.app.App;
import com.epsilon.agent.app.Application;
import com.epsilon.agent.app.Command;
import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Signals;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.reviewer.error.ErrorBody;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.http.HttpReviewer;
import com.epsilon.reviewer.http.ServerConfig;
import com.epsilon.reviewer.json.Json;
import com.epsilon.reviewer.model.ModelFamilies;
import com.epsilon.reviewer.stdio.FrameReader;
import com.epsilon.reviewer.stdio.FrameWriter;
import com.epsilon.reviewer.stdio.StdioHost;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Reviewer の入口。HTTP サーバとして動くコマンドと、デスクトップアプリの子プロセスとして動くコマンドを持つ。 */
public final class Main {
  private static final Logger LOG = LoggerFactory.getLogger(Main.class);

  /** 停止の要求が来たかを確かめる間隔（ミリ秒）。 */
  private static final long STOP_CHECK_MILLIS = 200;

  /** 全てのアドレスで待ち受けるときのホストの書き方。これらで待ち受けるときは、公開の origin を必ず指定する。 */
  private static final Set<String> WILDCARD_HOSTS = Set.of("0.0.0.0", "::", "[::]");

  private Main() {}

  /** Reviewer のコマンド。使い方の表示はこの順に並ぶ。 */
  public sealed interface ReviewerCommand extends Command<ReviewerSettings> permits Serve, Engine {}

  /**
   * HTTP サーバとして動く。SIGTERM か SIGINT で停止を求められるまで、解析を受け付ける。止めるときは、処理中の要求が終わるのを待ってから閉じる。
   *
   * @param host バインドするアドレス
   * @param port バインドするポート
   * @param origin 公開の origin。形は http(s)://host[:port] で、ユーザー情報・クエリ・フラグメントは付けず、パスは空か / にする。Host ヘッダーの
   *     検査と共有リンクに使う。無ければ http://host:port
   * @param models models.json。無ければ空のカタログ
   * @param dataDir 解析結果の保存先
   * @param webDir 画面の web/dist。無ければ画面のパスは 404 を返す
   */
  @Named(value = "serve", doc = "Start the HTTP server that takes analyses (stops on SIGTERM or SIGINT)")
  public record Serve(
      @Key(doc = "Address to bind", value = "127.0.0.1") String host,
      @Key(doc = "Port to bind", value = "8080") @Range(min = 1, max = 65535) int port,
      @Key(doc = "Public origin (http(s)://host[:port]; http://host:port if absent)", presence = Presence.ABSENT) String origin,
      @Key(doc = "models.json (an empty catalog if absent)", presence = Presence.ABSENT) Path models,
      @Key(doc = "Where analysis results are stored", value = "reviewer-data") Path dataDir,
      @Key(doc = "web/dist of the UI (UI paths return 404 if absent)", presence = Presence.ABSENT) Path webDir)
      implements ReviewerCommand {

    public Serve {
      if (host.isBlank()) {
        throw new InvalidSetting("host", "must not be blank");
      }
      if (origin != null) {
        try {
          ServerConfig.origin(origin);
        } catch (IllegalArgumentException invalid) {
          throw new InvalidSetting("origin", "expected http(s)://host[:port]");
        }
      } else if (WILDCARD_HOSTS.contains(host)) {
        throw new InvalidSetting(
            "origin", "is required when host is the wildcard address " + host + " (give the URL that browsers open)");
      }
    }

    /** 公開の origin を、{@code scheme://rawAuthority} の形にそろえて返す。 */
    public URI publicOrigin() {
      if (origin != null) {
        return ServerConfig.origin(origin);
      }
      String authority = host.contains(":") && !host.startsWith("[") ? "[" + host + "]" : host;
      return ServerConfig.origin("http://" + authority + ":" + port);
    }

    @Override
    public Outcome run(Context<ReviewerSettings> context) throws IOException, InterruptedException {
      Signals signals = context.signals();
      ReviewerSettings settings = context.settings().value();
      ReviewerSettings.Server server = settings.server();
      // JDK の HTTP サーバは、最初に作られるときにこれらの値を読む。
      System.setProperty("sun.net.httpserver.maxReqTime", Integer.toString(server.requestSeconds()));
      System.setProperty("sun.net.httpserver.maxRspTime", Integer.toString(server.responseSeconds()));
      System.setProperty("sun.net.httpserver.maxConnections", Integer.toString(server.maxConnections()));
      URI publicOrigin = publicOrigin();
      Reviewer reviewer = Reviewer.open(new ReviewerConfig(dataDir, models, settings, ModelFamilies.load()));
      HttpReviewer http;
      try {
        http = HttpReviewer.start(new ServerConfig(host, port, publicOrigin, webDir), reviewer);
      } catch (IOException | RuntimeException failure) {
        reviewer.close();
        throw failure;
      }
      LOG.info("Epsilon AI Reviewer started at {} (port {})", publicOrigin, http.port());
      try {
        while (!signals.requested()) {
          Thread.sleep(STOP_CHECK_MILLIS);
        }
      } finally {
        http.close();
        reviewer.close();
      }
      return Outcome.ok("served at " + publicOrigin + " until " + signals.reason());
    }
  }

  /**
   * デスクトップアプリの子プロセスとして動く（protocol/stdio.ts）。標準入出力でフレームをやり取りする。保存先とカタログを開けたら
   * {"event":"ready"} を書く。モデルはこの時点ではまだ開かない。開けなければ {"event":"failed", error} を書いて、失敗の終了コードで終わる。
   * 標準入力が閉じられたら、後片付けをして正常に終わる。
   *
   * @param dataDir 解析結果の保存先
   * @param models 同梱の models.json
   */
  @Named(value = "engine", doc = "Take operations as framed stdin/stdout messages, as a child process of the desktop app")
  public record Engine(
      @Key(doc = "Where analysis results are stored", presence = Presence.REQUIRED) Path dataDir,
      @Key(doc = "Bundled models.json", presence = Presence.REQUIRED) Path models)
      implements ReviewerCommand {

    @Override
    public Outcome run(Context<ReviewerSettings> context) throws IOException {
      FrameWriter frames = new FrameWriter(context.out());
      Reviewer reviewer;
      try {
        reviewer =
            Reviewer.open(new ReviewerConfig(dataDir, models, context.settings().value(), ModelFamilies.load()));
      } catch (ReviewerException failure) {
        LOG.error("Could not start the reviewer engine: {}", failure.getMessage());
        return failed(frames, ErrorBody.of(failure));
      } catch (SettingsException invalid) {
        invalid.problems().forEach(LOG::error);
        return failed(frames, ErrorBody.of(ErrorCode.INVALID_REQUEST, String.join("; ", invalid.problems())));
      } catch (IOException | RuntimeException failure) {
        LOG.error("Could not start the reviewer engine", failure);
        return failed(frames, ErrorBody.of(ErrorCode.SERVER_ERROR));
      }
      frames.write(event("ready"), null);
      new StdioHost(reviewer, new FrameReader(System.in), frames).run();
      return Outcome.ok("standard input closed");
    }

    private static Outcome failed(FrameWriter frames, ErrorBody error) throws IOException {
      JsonObject failed = event("failed");
      failed.add("error", Json.GSON.toJsonTree(error));
      frames.write(failed, null);
      return new Outcome(Exit.FAILED, "the engine could not start: " + error.code());
    }

    private static JsonObject event(String name) {
      JsonObject event = new JsonObject();
      event.addProperty("event", name);
      return event;
    }
  }

  /** Reviewer のアプリの定義。 */
  public static Application<ReviewerSettings> application() {
    return Application.of("reviewer", ReviewerSettings.class, List.of(ReviewerCommand.class));
  }

  /** コマンドを 1 つ実行し、その終了コードで JVM を終える。 */
  public static void main(String[] arguments) {
    App.main(application(), arguments);
  }
}
