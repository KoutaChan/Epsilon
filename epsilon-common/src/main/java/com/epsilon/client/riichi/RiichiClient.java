package com.epsilon.client.riichi;

import com.epsilon.client.WebSocketTransport;
import com.epsilon.engine.Player;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.net.http.WebSocketHandshakeException;
import java.util.Map;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 一つのトークンで検証、ランク戦、対局間の再接続を順番に進める。 */
public final class RiichiClient {
  private static final Logger log = LoggerFactory.getLogger(RiichiClient.class);
  static final String RANKED_URI = "wss://game.riichi.dev/ws/ranked";
  static final String VALIDATION_URI = "wss://game.riichi.dev/ws/validate";
  static final int MAX_CONNECTION_FAILURES = 12;
  private static final String TOKEN_VERIFICATION_ERROR_PREFIX = "Token verification failed:";
  private final ConnectionFactory connections;
  private final Sleeper sleeper;
  private volatile WebSocketTransport activeConnection;

  /** 完走したランク戦と、開始後に通信が切れたランク戦の数。検証対局は含まない。 */
  public record Result(int completedGames, int interruptedGames) {}

  /** 接続先とヘッダーを受け取り、WebSocketの通信処理を生成する。テストなどで通信実装を差し替えるために使う。 */
  @FunctionalInterface
  interface ConnectionFactory {
    WebSocketTransport connect(String uri, Map<String, String> headers)
        throws IOException, InterruptedException;
  }

  /** 再接続までの待機。テストでは実時間を消費しない実装を渡す。 */
  @FunctionalInterface
  interface Sleeper {
    void sleep(long millis) throws InterruptedException;
  }

  private RiichiClient(ConnectionFactory connections, Sleeper sleeper) {
    this.connections = connections;
    this.sleeper = sleeper;
  }

  /**
   * 接続ごとにプレイヤーを生成して連続対局する。maxGames が 0 なら停止操作まで続ける。
   *
   * <p>プレイヤーが借用するモデルと推論資源は呼び出し元が所有し、この呼び出しの終了後に閉じる。
   */
  public static Result play(String token, int maxGames, Supplier<? extends Player> players)
      throws IOException, InterruptedException {
    var client = new RiichiClient(WebSocketTransport::connect, Thread::sleep);
    Thread owner = Thread.currentThread();
    Thread shutdown =
        new Thread(
            () -> {
              owner.interrupt();
              WebSocketTransport connection = client.activeConnection;
              if (connection != null) {
                connection.close();
              }
            },
            "riichi-shutdown");
    Runtime.getRuntime().addShutdownHook(shutdown);
    try {
      return client.run(token, maxGames, players);
    } finally {
      try {
        Runtime.getRuntime().removeShutdownHook(shutdown);
      } catch (IllegalStateException ignored) {
        // JVM の終了中は登録済み終了処理自体が接続を解放する。
      }
    }
  }

  /** テスト用の通信処理と待機処理を使って、本番と同じ接続ライフサイクルを実行する。 */
  static Result play(
      String token,
      int maxGames,
      Supplier<? extends Player> players,
      ConnectionFactory connections,
      Sleeper sleeper)
      throws IOException, InterruptedException {
    return new RiichiClient(connections, sleeper).run(token, maxGames, players);
  }

  private Result run(String token, int maxGames, Supplier<? extends Player> players)
      throws IOException, InterruptedException {
    Map<String, String> headers = Map.of("Authorization", "Bearer " + token);
    int completed = 0;
    int interrupted = 0;
    int failures = 0;
    boolean validationAttempted = false;
    while (maxGames == 0 || completed < maxGames) {
      var session = new RiichiSession(players.get(), false);
      try {
        playSession(RANKED_URI, headers, token, session);
        completed++;
        failures = 0;
        log.info("RiichiLab ranked games completed: {}", completed);
      } catch (AuthenticationFailure error) {
        if (validationAttempted) {
          throw error;
        }
        validationAttempted = true;
        validate(headers, token, players, error);
        failures = 0;
      } catch (TerminalFailure error) {
        throw error;
      } catch (IOException e) {
        if (session.started() && !session.completed()) {
          interrupted++;
          log.warn("RiichiLab game disconnected; the interrupted game cannot be resumed");
        }
        failures++;
        if (failures >= MAX_CONNECTION_FAILURES) {
          throw new IOException("RiichiLab connection failure limit reached", e);
        }
        long delay = Math.min(30_000L, 1_000L << Math.min(failures - 1, 5));
        log.warn("RiichiLab reconnecting in {} ms: consecutiveFailures={}", delay, failures);
        sleeper.sleep(delay);
      }
    }
    return new Result(completed, interrupted);
  }

  /** 検証は一接続だけ行い、失敗した場合はランク戦の拒否理由と合わせて終了する。 */
  private void validate(
      Map<String, String> headers,
      String token,
      Supplier<? extends Player> players,
      AuthenticationFailure rankedRejection)
      throws IOException, InterruptedException {
    log.info("RiichiLab ranked authentication rejected; attempting validation once");
    var session = new RiichiSession(players.get(), true);
    try {
      playSession(VALIDATION_URI, headers, token, session);
      if (!session.validationPassed()) {
        throw new TerminalFailure("RiichiLab validation failed; check the game log");
      }
    } catch (IOException error) {
      String reason =
          error instanceof TerminalFailure
              ? error.getMessage()
              : "Connection lost during validation (" + redact(error.toString(), token) + ")";
      throw new TerminalFailure(
          "RiichiLab ranked authentication and validation failed. Ranked: "
              + rankedRejection.getMessage()
              + "; Validation: "
              + reason);
    }
    log.info("RiichiLab validation passed; entering ranked play");
  }

  /** 一接続を完了・解放し、JSON と HTTP の認証拒否を呼び出し元へ同じ形で返す。 */
  private void playSession(
      String uri, Map<String, String> headers, String token, RiichiSession session)
      throws IOException, InterruptedException {
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException("RiichiLab play interrupted");
    }
    log.info("RiichiLab connecting: {}", uri);
    try (WebSocketTransport connection = connections.connect(uri, headers)) {
      activeConnection = connection;
      while (!session.ended()) {
        WebSocketTransport.Message frame = connection.readMessage();
        JsonObject message;
        try {
          message = JsonParser.parseString(frame.text()).getAsJsonObject();
        } catch (RuntimeException error) {
          throw new TerminalFailure("RiichiLab received a frame that is not a JSON object", error);
        }
        if (message.has("error")) {
          String error = message.get("error").getAsString();
          if (!session.started() && error.startsWith(TOKEN_VERIFICATION_ERROR_PREFIX)) {
            throw new AuthenticationFailure(redact(error, token));
          }
          throw new TerminalFailure(redact(error, token));
        }
        try {
          String response = session.handleMessage(message, frame.receivedNanos());
          if (response != null) {
            connection.send(response);
            log.info("Riichi action sent: {}", response);
          }
        } catch (IllegalArgumentException | IllegalStateException error) {
          throw new TerminalFailure("RiichiLab game protocol error", error);
        }
      }
    } catch (IOException error) {
      for (Throwable cause = error; cause != null; cause = cause.getCause()) {
        if (cause instanceof WebSocketHandshakeException handshake) {
          int status = handshake.getResponse().statusCode();
          if (!session.started() && (status == 401 || status == 403)) {
            throw new AuthenticationFailure("HTTP " + status);
          }
          if (status >= 400 && status < 500 && status != 408 && status != 409 && status != 429) {
            throw new TerminalFailure("RiichiLab connection rejected: HTTP " + status);
          }
          break;
        }
      }
      throw error;
    } finally {
      activeConnection = null;
    }
  }

  private static String redact(String message, String token) {
    return token.isEmpty() ? message : message.replace(token, "<redacted>");
  }

  /** 再接続で解消しない認証・プロトコルエラー。 */
  private static class TerminalFailure extends IOException {
    TerminalFailure(String message) {
      super(message);
    }

    TerminalFailure(String message, Throwable cause) {
      super(message, cause);
    }
  }

  /** 対局開始前の認証拒否。初回だけ検証接続先への接続を試せる。 */
  private static final class AuthenticationFailure extends TerminalFailure {
    AuthenticationFailure(String message) {
      super(message);
    }
  }
}
