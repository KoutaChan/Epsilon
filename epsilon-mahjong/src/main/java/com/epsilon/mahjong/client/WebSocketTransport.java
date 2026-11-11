package com.epsilon.mahjong.client;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * JDK の WebSocket を使う接続。テキストのメッセージは受けた順に読む。接続が切れたら、残りのメッセージを読み終えた後で
 * IOException を投げる。Ping を送る間隔を指定した接続では、読むメッセージがその間隔だけ無いたびに read が Ping を送り、次の
 * 間隔のうちに相手から何も届かなければ、接続が切れたものとして IOException を投げる。
 */
public final class WebSocketTransport implements Transport, WebSocket.Listener {

  /** 接続・送信・切断の応答を待つ時間。 */
  private static final Duration TIMEOUT = Duration.ofSeconds(30);

  private static final Duration CLOSE_TIMEOUT = Duration.ofSeconds(2);

  /** Ping を送らない接続の、Ping を送るまでの間隔（来ることのない長さ）。 */
  private static final Duration NO_PING = Duration.ofNanos(Long.MAX_VALUE);

  /** 接続が切れたことを表す目印のフレーム。 */
  private static final Frame CLOSED = new Frame(null, 0);

  private final HttpClient client = HttpClient.newBuilder().connectTimeout(TIMEOUT).build();
  private final BlockingQueue<Frame> incoming = new LinkedBlockingQueue<>();
  private final CountDownLatch closed = new CountDownLatch(1);
  private final Duration pingInterval;

  /** 組み立て中のメッセージ。WebSocket は 1 つのメッセージを分けて届けることがある。受信のスレッドだけが触る。 */
  private final StringBuilder buffer = new StringBuilder();

  private WebSocket socket;
  private long receivedNanos;

  /** 相手から最後にフレーム（メッセージの断片・Ping・Pong）が届いた時刻（System.nanoTime の値）。 */
  private volatile long heardNanos;
  private volatile boolean closing;
  private volatile boolean connected;
  private volatile IOException failure = new IOException("WebSocket connection closed");

  private WebSocketTransport(Duration pingInterval) {
    this.pingInterval = pingInterval;
  }

  /** 指定のヘッダーを付けて接続する。Ping は送らない。接続に失敗したか時間切れなら IOException を投げる。 */
  public static WebSocketTransport connect(URI uri, Map<String, String> headers)
      throws IOException, InterruptedException {
    return connect(uri, headers, NO_PING);
  }

  /**
   * 指定のヘッダーを付けて接続する。接続に失敗したか時間切れなら IOException を投げる。
   *
   * @param pingInterval read が Ping を送るまでの、読むメッセージの無い時間。Ping の後、同じ時間のうちに相手から何も届かなければ、
   *     read は IOException を投げる
   */
  public static WebSocketTransport connect(URI uri, Map<String, String> headers, Duration pingInterval)
      throws IOException, InterruptedException {
    WebSocketTransport transport = new WebSocketTransport(pingInterval);
    CompletableFuture<WebSocket> opening = null;
    try {
      WebSocket.Builder builder = transport.client.newWebSocketBuilder().connectTimeout(TIMEOUT);
      headers.forEach(builder::header);
      opening = builder.buildAsync(uri, transport);
      transport.socket = opening.get(TIMEOUT.toSeconds(), TimeUnit.SECONDS);
      return transport;
    } catch (ExecutionException failure) {
      transport.close();
      throw new IOException("WebSocket connection to " + uri + " failed", failure.getCause());
    } catch (TimeoutException failure) {
      opening.cancel(true);
      transport.close();
      throw new IOException("WebSocket connection to " + uri + " timed out", failure);
    } catch (InterruptedException | RuntimeException failure) {
      if (opening != null) {
        opening.cancel(true);
      }
      transport.close();
      throw failure;
    }
  }

  @Override
  public Frame read() throws IOException, InterruptedException {
    if (!connected && incoming.isEmpty()) {
      throw failure;
    }
    Frame frame = incoming.poll(pingInterval.toNanos(), TimeUnit.NANOSECONDS);
    while (frame == null) {
      long pinged = System.nanoTime();
      ping();
      frame = incoming.poll(pingInterval.toNanos(), TimeUnit.NANOSECONDS);
      if (frame == null && heardNanos - pinged < 0) {
        throw new IOException("WebSocket peer sent nothing within " + pingInterval + " of a ping");
      }
    }
    if (frame == CLOSED) {
      throw failure;
    }
    return frame;
  }

  /** Ping を送る。相手は Pong で応える。 */
  private synchronized void ping() throws IOException, InterruptedException {
    if (!connected) {
      throw failure;
    }
    try {
      socket.sendPing(ByteBuffer.allocate(0)).get(TIMEOUT.toSeconds(), TimeUnit.SECONDS);
    } catch (ExecutionException | TimeoutException | RuntimeException failed) {
      throw new IOException("WebSocket ping failed", failed);
    }
  }

  @Override
  public synchronized void send(String text) throws IOException {
    if (!connected) {
      throw failure;
    }
    try {
      socket.sendText(text, true).get(TIMEOUT.toSeconds(), TimeUnit.SECONDS);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while sending", interrupted);
    } catch (ExecutionException | TimeoutException | RuntimeException failed) {
      throw new IOException("WebSocket send failed", failed);
    }
  }

  /** 切断を送って相手の応答を少し待ち、ソケットと HTTP のリソースを解放する。 */
  @Override
  public synchronized void close() {
    closing = true;
    connected = false;
    if (socket != null) {
      try {
        socket.sendClose(WebSocket.NORMAL_CLOSURE, "bye").get(CLOSE_TIMEOUT.toSeconds(), TimeUnit.SECONDS);
        closed.await(CLOSE_TIMEOUT.toSeconds(), TimeUnit.SECONDS);
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
      } catch (ExecutionException | TimeoutException | RuntimeException unanswered) {
        // 相手が切断に応えなくても、finally の abort でリソースを解放する。
      } finally {
        socket.abort();
        socket = null;
      }
    }
    client.shutdownNow();
    incoming.offer(CLOSED);
  }

  @Override
  public synchronized void onOpen(WebSocket webSocket) {
    if (closing) {
      webSocket.abort();
      return;
    }
    socket = webSocket;
    heardNanos = System.nanoTime();
    connected = true;
    webSocket.request(1);
  }

  @Override
  public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
    heardNanos = System.nanoTime();
    if (buffer.isEmpty()) {
      receivedNanos = System.nanoTime();
    }
    buffer.append(data);
    if (last) {
      incoming.offer(new Frame(buffer.toString(), receivedNanos));
      buffer.setLength(0);
    }
    webSocket.request(1);
    return null;
  }

  @Override
  public CompletionStage<?> onPing(WebSocket webSocket, ByteBuffer message) {
    heardNanos = System.nanoTime();
    webSocket.request(1);
    return null;
  }

  @Override
  public CompletionStage<?> onPong(WebSocket webSocket, ByteBuffer message) {
    heardNanos = System.nanoTime();
    webSocket.request(1);
    return null;
  }

  @Override
  public CompletionStage<?> onClose(WebSocket webSocket, int statusCode, String reason) {
    connected = false;
    closed.countDown();
    incoming.offer(CLOSED);
    return null;
  }

  @Override
  public void onError(WebSocket webSocket, Throwable error) {
    failure = new IOException("WebSocket receive failed", error);
    connected = false;
    closed.countDown();
    incoming.offer(CLOSED);
  }
}
