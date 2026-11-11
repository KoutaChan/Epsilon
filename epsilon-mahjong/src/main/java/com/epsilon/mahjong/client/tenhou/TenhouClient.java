package com.epsilon.mahjong.client.tenhou;

import com.epsilon.mahjong.client.Decider;
import com.epsilon.mahjong.client.Transport;
import com.epsilon.mahjong.client.WebSocketTransport;
import com.epsilon.mahjong.io.tenhou.TenhouRule;
import com.epsilon.mahjong.io.tenhou.TenhouTag;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.net.URI;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 天鳳の個室で 1 半荘を打つ。ログインして卓を申し込み、サーバのメッセージを TenhouSession に渡して、その返事を送る。接続している
 * 間は、10 秒ごとにキープアライブを送る。
 */
public final class TenhouClient {

  /** 天鳳の WebSocket の接続先。 */
  public static final URI SERVER = URI.create("wss://b-ww.mjv.jp");

  private static final Logger log = LoggerFactory.getLogger(TenhouClient.class);

  /** 接続の HTTP ヘッダー。天鳳はブラウザの Origin からの接続だけを受ける。 */
  private static final Map<String, String> HEADERS =
      Map.of(
          "Origin",
          "https://tenhou.net",
          "User-Agent",
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
              + " Chrome/146.0.0.0 Safari/537.36");

  /** キープアライブを送る間隔（秒）。 */
  private static final long KEEPALIVE_SECONDS = 10;

  private TenhouClient() {}

  /**
   * 天鳳に接続し、個室に入って 1 半荘を打つ。
   *
   * @param server 接続先
   * @param variant 申し込む卓のルール
   * @param room 入る個室
   * @param name 打ち手の名前
   * @param decider 手を選ぶもの
   * @return 局の結果がこちらの精算と合わなかった局の数（結果が読めなかった局を含む）
   */
  public static int play(URI server, Variant variant, String room, String name, Decider decider)
      throws IOException, InterruptedException {
    try (WebSocketTransport transport = WebSocketTransport.connect(server, HEADERS)) {
      return play(transport, variant, room, name, decider);
    }
  }

  /** 開いている接続で個室に入り、1 半荘を打つ。接続は呼び出し元が閉じる。 */
  static int play(Transport transport, Variant variant, String room, String name, Decider decider)
      throws IOException, InterruptedException {
    ScheduledExecutorService keepalive =
        Executors.newSingleThreadScheduledExecutor(
            task -> {
              Thread thread = new Thread(task, "tenhou-keepalive");
              thread.setDaemon(true);
              return thread;
            });
    try {
      transport.send(TenhouCodec.helo(name));
      TenhouTag helo = TenhouCodec.parse(transport.read().text());
      if (!helo.name().equals("HELO")) {
        throw new IOException("Tenhou answers the login with " + helo.name());
      }
      keepalive.scheduleAtFixedRate(() -> keepAlive(transport), KEEPALIVE_SECONDS, KEEPALIVE_SECONDS, TimeUnit.SECONDS);
      transport.send(TenhouCodec.lobby(room));
      transport.send(TenhouCodec.join(room + "," + TenhouRule.hanchan(variant)));
      log.info("Tenhou room {} is joined as {} for {}", room, name, variant);
      TenhouSession session = new TenhouSession(variant, decider);
      while (!session.ended()) {
        String message = transport.read().text();
        if (message.isBlank()) {
          continue;
        }
        for (String reply : session.handle(message)) {
          transport.send(reply);
        }
      }
      return session.mismatches();
    } finally {
      keepalive.shutdownNow();
    }
  }

  private static void keepAlive(Transport transport) {
    try {
      transport.send(TenhouCodec.keepAlive());
    } catch (IOException failure) {
      log.warn("Tenhou keepalive failed", failure);
    }
  }
}
