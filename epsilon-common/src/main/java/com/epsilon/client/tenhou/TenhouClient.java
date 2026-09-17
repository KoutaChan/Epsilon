package com.epsilon.client.tenhou;

import com.epsilon.engine.Player;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 系列に依存しない天鳳個室への接続、ログイン、対局メッセージの送受信。 */
public final class TenhouClient {

  private static final Logger log = LoggerFactory.getLogger(TenhouClient.class);

  private TenhouClient() {}

  /** 接続中だけプレイヤーを借用する。接続と生存確認はここで閉じ、モデル資源は呼び出し側が閉じる。 */
  public static void run(Player player, Options options) throws IOException {
    log.info("=== Epsilon Tenhou Client ===");
    log.info("Checkpoint: {}", options.checkpointDir());
    log.info("Room: {}", options.room());
    log.info("Name: {}", options.name());
    log.info("Server: {}", options.uri());
    log.info("Policy mode: {}", options.samplePolicy() ? "FULL_SUPPORT" : "POLICY_GREEDY");

    try (TenhouConnection connection = new TenhouConnection(options.uri())) {
      connection.connect();
      connection.startKeepalive();
      login(connection, options);
      TenhouSession session = new TenhouSession(new TenhouGameController(player));
      while (!session.ended()) {
        for (String response : session.handleMessage(connection.readMessage())) {
          connection.send(response);
        }
      }
    }
    log.info("Game complete");
  }

  private static void login(TenhouConnection connection, Options options) throws IOException {
    String encodedName = URLEncoder.encode(options.name(), StandardCharsets.UTF_8);
    connection.send("{\"tag\":\"HELO\",\"name\":\"" + encodedName + "\",\"sx\":\"M\"}");
    String response = connection.readMessage();
    if (!(TenhouMessageParser.parse(response) instanceof TenhouEvent.Helo)) {
      throw new IOException("Unexpected login response: " + response);
    }
    log.info("HELO response received");
    connection.send("{\"tag\":\"LOBBY\",\"id\":\"" + options.room() + "\"}");
    // 南赤入の個室へ参加する。
    connection.send("{\"tag\":\"JOIN\",\"t\":\"" + options.room() + ",9\"}");
    log.info("Join request sent for room {}", options.room());
  }

  /** 起動引数。チェックポイントの解釈とサンプルの選択方式への変換は各系列が行う。 */
  public record Options(
      String checkpointDir, String room, String name, String uri, boolean samplePolicy) {

    /** 既存の長短オプションを読み取る。help または個室未指定では null を返す。 */
    public static Options parse(String[] args, String defaultCheckpointDir) {
      String checkpointDir = defaultCheckpointDir;
      String room = "";
      String name = "EpsilonAI";
      String uri = TenhouConnection.DEFAULT_URI;
      boolean samplePolicy = false;
      for (int i = 0; i < args.length; i++) {
        switch (args[i]) {
          case "--checkpoint", "-c" -> {
            if (i + 1 < args.length) {
              checkpointDir = args[++i];
            }
          }
          case "--room", "-r" -> {
            if (i + 1 < args.length) {
              room = args[++i];
            }
          }
          case "--name", "-n" -> {
            if (i + 1 < args.length) {
              name = args[++i];
            }
          }
          case "--uri" -> {
            if (i + 1 < args.length) {
              uri = args[++i];
            }
          }
          case "--sample" -> samplePolicy = true;
          case "--help", "-h" -> {
            return null;
          }
          default -> log.warn("Unknown argument: {}", args[i]);
        }
      }
      if (room.isEmpty()) {
        log.error("--room parameter is required");
        return null;
      }
      return new Options(checkpointDir, room, name, uri, samplePolicy);
    }
  }

  /** 系列の既定チェックポイントを使って共通の CLI 説明を表示する。 */
  public static void printUsage(String defaultCheckpointDir) {
    log.info(
        """
        Usage: TenhouMain --room <room> [options]

        Options:
          --checkpoint, -c <dir>   Checkpoint directory (default: {})
          --room, -r <room>        Private room ID (e.g., C12345678)
          --name, -n <name>        Player name (default: EpsilonAI)
          --uri <uri>              WebSocket URI (default: wss://b-ww.mjv.jp)
          --sample                 Enable full-support policy mode
          --help, -h               Show help\
        """,
        defaultCheckpointDir);
  }
}
