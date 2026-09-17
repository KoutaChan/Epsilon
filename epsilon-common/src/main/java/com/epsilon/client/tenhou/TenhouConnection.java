package com.epsilon.client.tenhou;

import com.epsilon.client.WebSocketTransport;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** tenhou.net の接続設定と、10 秒ごとに {@code <Z/>} を送る生存確認を管理する。 */
public final class TenhouConnection implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(TenhouConnection.class);
  public static final String DEFAULT_URI = "wss://b-ww.mjv.jp";
  private static final int KEEPALIVE_INTERVAL_SEC = 10;
  private static final Map<String, String> HEADERS =
      Map.of(
          "User-Agent",
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
              + " Chrome/146.0.0.0 Safari/537.36");
  private final String uri;
  private WebSocketTransport transport;
  private ScheduledExecutorService keepaliveExecutor;

  /** 既定の tenhou.net 接続先を使う未接続クライアントを生成する。 */
  public TenhouConnection() {
    this(DEFAULT_URI);
  }

  /** 指定接続先を使う未接続クライアントを生成する。 */
  public TenhouConnection(String uri) {
    this.uri = uri;
  }

  /** 構築時に指定したサーバーへ接続する。 */
  public void connect() throws IOException {
    log.info("Connecting to tenhou server: {}", uri);
    try {
      transport = WebSocketTransport.connect(uri, HEADERS);
      log.info("Connected to tenhou server");
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while connecting", e);
    }
  }

  /** メッセージを一つ読み込む。読み込みに失敗すると{@link IOException}を送出する。 */
  public String readMessage() throws IOException {
    if (transport == null) {
      throw new IOException("Not connected");
    }
    try {
      String text;
      do {
        text = transport.readMessage().text();
      } while (text.isEmpty());
      log.debug("recv: {}", text);
      return text;
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while reading", e);
    }
  }

  /** JSON または XML のプロトコルメッセージをそのまま送信する。 */
  public void send(String text) throws IOException {
    if (transport == null) {
      throw new IOException("Not connected");
    }
    log.debug("send: {}", text);
    transport.send(text);
  }

  /** 定期生存確認タイマーを開始する。 */
  public void startKeepalive() {
    if (keepaliveExecutor != null) {
      return;
    }
    keepaliveExecutor =
        Executors.newSingleThreadScheduledExecutor(
            r -> {
              Thread thread = new Thread(r, "tenhou-keepalive");
              thread.setDaemon(true);
              return thread;
            });
    keepaliveExecutor.scheduleAtFixedRate(
        () -> {
          try {
            send("<Z/>");
          } catch (IOException e) {
            log.warn("Keepalive send failed", e);
          }
        },
        KEEPALIVE_INTERVAL_SEC,
        KEEPALIVE_INTERVAL_SEC,
        TimeUnit.SECONDS);
  }

  /** WebSocket が接続済みかを返す。 */
  public boolean isConnected() {
    return transport != null && transport.isConnected();
  }

  /** 生存確認を止め、共有通信処理を閉じる。 */
  @Override
  public void close() {
    if (keepaliveExecutor != null) {
      keepaliveExecutor.shutdownNow();
      keepaliveExecutor = null;
    }
    if (transport != null) {
      transport.close();
    }
    log.info("Tenhou connection closed");
  }
}
