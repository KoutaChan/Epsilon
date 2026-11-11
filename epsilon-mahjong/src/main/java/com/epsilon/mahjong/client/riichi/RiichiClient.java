package com.epsilon.mahjong.client.riichi;

import com.epsilon.mahjong.client.Decider;
import com.epsilon.mahjong.client.Transport;
import com.epsilon.mahjong.client.WebSocketTransport;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.net.URI;
import java.net.http.WebSocketHandshakeException;
import java.time.Duration;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * RiichiLab で対局を続けて打つ。トークンでランク戦に接続し、接続ごとに新しい Decider で対局を進める。対局の前に認証を拒まれたら、
 * 検証の対局を 1 回だけ打ってからランク戦に戻る。
 *
 * <p>接続が切れたら、待ってからつなぎ直す。待ち時間は 1 秒から倍にしていき、上限は 30 秒。続けて 12 回切れたら止める。つなぎ直しても
 * 直らない拒否や、プロトコルのエラーでも止める。エラーメッセージに出たトークンは伏せる。サーバのメッセージが 30 秒届かないたびに
 * Ping を送り、次の 30 秒のうちに何も届かなければ、接続が切れたものとして扱う。
 */
public final class RiichiClient {

  static final URI RANKED = URI.create("wss://game.riichi.dev/ws/ranked");
  static final URI VALIDATION = URI.create("wss://game.riichi.dev/ws/validate");

  /** 接続がこの回数だけ続けて切れたら止める。 */
  static final int MAX_FAILURES = 12;

  private static final Logger log = LoggerFactory.getLogger(RiichiClient.class);

  private static final long FIRST_DELAY_MILLIS = 1_000;
  private static final long MAX_DELAY_MILLIS = 30_000;

  /** サーバのメッセージがこの時間届かないたびに、Ping を送って接続が生きているかを確かめる。 */
  private static final Duration PING_INTERVAL = Duration.ofSeconds(30);

  /** 待ち時間を倍にする回数の上限（1 秒 × 2^5 = 32 秒で 30 秒を超える）。 */
  private static final int MAX_DOUBLINGS = 5;

  /** 認証の拒否を表すエラーメッセージの書き出し。 */
  private static final String TOKEN_REJECTED = "Token verification failed:";

  /** WebSocket のハンドシェイクで、認証の拒否を表す HTTP のステータスコード。 */
  private static final Set<Integer> UNAUTHORIZED = Set.of(401, 403);

  /** ハンドシェイクの 4xx の拒否のうち、つなぎ直せば通ることがあるもの（タイムアウト・競合・リクエスト過多）。 */
  private static final Set<Integer> RETRYABLE = Set.of(408, 409, 429);

  /** クライアントのエラーを表す HTTP のステータスコード（4xx）の下限。 */
  private static final int CLIENT_ERRORS = 400;

  private static final int SERVER_ERRORS = 500;

  /** 完走したランク戦の数と、始まった後に接続が切れたランク戦の数（検証の対局は含まない）。 */
  public record Result(int completedGames, int interruptedGames) {}

  /** 接続を作る。テストでは偽の接続を返す。 */
  @FunctionalInterface
  interface Connector {
    Transport connect(URI uri, Map<String, String> headers) throws IOException, InterruptedException;
  }

  /** つなぎ直す前に待つ。テストでは実際には待たない。 */
  @FunctionalInterface
  interface Sleeper {
    void sleep(long millis) throws InterruptedException;
  }

  /** つなぎ直しても直らない拒否と、プロトコルのエラー。 */
  private static class TerminalFailure extends IOException {

    TerminalFailure(String message) {
      super(message);
    }

    TerminalFailure(String message, Throwable cause) {
      super(message, cause);
    }
  }

  /** 対局の前の認証の拒否。初めての拒否なら、検証の対局を試す。 */
  private static final class Rejection extends TerminalFailure {

    Rejection(String message) {
      super(message);
    }
  }

  private final String token;
  private final Map<String, String> headers;
  private final Supplier<? extends Decider> deciders;
  private final Connector connector;
  private final Sleeper sleeper;

  private RiichiClient(String token, Supplier<? extends Decider> deciders, Connector connector, Sleeper sleeper) {
    this.token = token;
    headers = Map.of("Authorization", "Bearer " + token);
    this.deciders = deciders;
    this.connector = connector;
    this.sleeper = sleeper;
  }

  /**
   * ランク戦を続けて打つ。
   *
   * @param token RiichiLab のトークン
   * @param games 打つ対局の数。0 なら止められるまで打つ
   * @param deciders 接続ごとに新しい Decider を作るもの
   * @return 完走した対局の数と、途中で接続が切れた対局の数
   */
  public static Result play(String token, int games, Supplier<? extends Decider> deciders)
      throws IOException, InterruptedException {
    return play(
        token,
        games,
        deciders,
        (uri, headers) -> WebSocketTransport.connect(uri, headers, PING_INTERVAL),
        Thread::sleep);
  }

  static Result play(
      String token, int games, Supplier<? extends Decider> deciders, Connector connector, Sleeper sleeper)
      throws IOException, InterruptedException {
    return new RiichiClient(token, deciders, connector, sleeper).run(games);
  }

  private Result run(int games) throws IOException, InterruptedException {
    int completed = 0;
    int interrupted = 0;
    int failures = 0;
    boolean validated = false;
    while (games == 0 || completed < games) {
      RiichiSession session = new RiichiSession(deciders.get(), false);
      try {
        connect(RANKED, session);
        completed++;
        failures = 0;
        log.info("RiichiLab ranked games completed: {}", completed);
      } catch (Rejection rejection) {
        if (validated) {
          throw rejection;
        }
        validated = true;
        validate(rejection);
        failures = 0;
      } catch (TerminalFailure failure) {
        throw failure;
      } catch (IOException lost) {
        if (session.started() && !session.completed()) {
          interrupted++;
          log.warn("RiichiLab game disconnected; the game cannot be resumed");
        }
        failures++;
        if (failures >= MAX_FAILURES) {
          throw new IOException("RiichiLab connection failure limit reached", lost);
        }
        long delay = Math.min(MAX_DELAY_MILLIS, FIRST_DELAY_MILLIS << Math.min(failures - 1, MAX_DOUBLINGS));
        log.warn("RiichiLab reconnects in {} ms after {} failures", delay, failures);
        sleeper.sleep(delay);
      }
    }
    return new Result(completed, interrupted);
  }

  /** 検証の対局を 1 回打つ。通らないか切れたら、ランク戦で拒まれた理由も添えて TerminalFailure を投げる。 */
  private void validate(Rejection rejection) throws IOException, InterruptedException {
    log.info("RiichiLab ranked authentication rejected; trying the validation game once");
    RiichiSession session = new RiichiSession(deciders.get(), true);
    try {
      connect(VALIDATION, session);
      if (!session.validationPassed()) {
        throw new TerminalFailure("RiichiLab validation failed; check the game log");
      }
    } catch (IOException failure) {
      String reason =
          failure instanceof TerminalFailure
              ? failure.getMessage()
              : "Connection lost during validation (" + redact(failure.toString()) + ")";
      throw new TerminalFailure(
          "RiichiLab ranked authentication and validation failed. Ranked: " + rejection.getMessage()
              + "; Validation: " + reason);
    }
    log.info("RiichiLab validation passed; entering ranked play");
  }

  /** 接続して、対局が終わるまでサーバのメッセージを RiichiSession に渡し、その返事を送る。接続は必ず閉じる。 */
  private void connect(URI uri, RiichiSession session) throws IOException, InterruptedException {
    if (Thread.currentThread().isInterrupted()) {
      throw new InterruptedException("RiichiLab play interrupted");
    }
    log.info("RiichiLab connects to {}", uri);
    try (Transport transport = connector.connect(uri, headers)) {
      while (!session.ended()) {
        Transport.Frame frame = transport.read();
        rejectError(frame.text(), session);
        String reply;
        try {
          reply = session.handle(frame.text(), frame.receivedNanos());
        } catch (IllegalArgumentException | IllegalStateException failure) {
          throw new TerminalFailure("RiichiLab game protocol error", failure);
        }
        if (reply != null) {
          transport.send(reply);
          log.info("RiichiLab action sent: {}", reply);
        }
      }
    } catch (IOException failure) {
      throw classified(failure, session);
    }
  }

  /** サーバのメッセージがエラーなら止める。対局の前の認証の拒否なら Rejection を、ほかは TerminalFailure を投げる。 */
  private void rejectError(String text, RiichiSession session) throws TerminalFailure {
    JsonElement error;
    try {
      JsonObject message = JsonParser.parseString(text).getAsJsonObject();
      error = message.get("error");
    } catch (JsonParseException | IllegalStateException failure) {
      throw new TerminalFailure("RiichiLab received a frame that is not a JSON object", failure);
    }
    if (error == null) {
      return;
    }
    String reason = redact(error.getAsString());
    if (!session.started() && reason.startsWith(TOKEN_REJECTED)) {
      throw new Rejection(reason);
    }
    throw new TerminalFailure(reason);
  }

  /** WebSocket のハンドシェイクの拒否（HTTP 4xx）を、認証の拒否か、つなぎ直しても直らない拒否に分ける。ほかの失敗はそのまま返す。 */
  private static IOException classified(IOException failure, RiichiSession session) {
    for (Throwable cause = failure; cause != null; cause = cause.getCause()) {
      if (cause instanceof WebSocketHandshakeException handshake) {
        int status = handshake.getResponse().statusCode();
        if (!session.started() && UNAUTHORIZED.contains(status)) {
          return new Rejection("HTTP " + status);
        }
        if (status >= CLIENT_ERRORS && status < SERVER_ERRORS && !RETRYABLE.contains(status)) {
          return new TerminalFailure("RiichiLab connection rejected: HTTP " + status);
        }
        return failure;
      }
    }
    return failure;
  }

  private String redact(String message) {
    return token.isEmpty() ? message : message.replace(token, "<redacted>");
  }
}
