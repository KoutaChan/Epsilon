package com.epsilon.client;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
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

/** 対局プロトコルに依存せず、WebSocketのテキストフレームの送受信と接続の終了を行う。 */
public interface WebSocketTransport extends AutoCloseable {
  /** 完成したテキストフレームと、その受信が始まった単調時刻。 */
  record Message(String text, long receivedNanos) {}

  /** 指定されたヘッダーだけを付けて接続する。認証情報はこの層では出力しない。 */
  static WebSocketTransport connect(String uri, Map<String, String> headers)
      throws IOException, InterruptedException {
    return JdkWebSocketTransport.connect(uri, headers);
  }

  /** 一つの完成フレームを読む。通信終了と割り込みは呼び出し元へ伝える。 */
  Message readMessage() throws IOException, InterruptedException;

  /** 一つのテキストフレームを送る。複数の送信元がある場合も順番を保持する。 */
  void send(String text) throws IOException;

  /** 接続が開いているかを返す。 */
  boolean isConnected();

  /** 切断時の応答を短時間待ち、残っているソケットと HTTP クライアントを解放する。 */
  @Override
  void close();
}

/** JDK WebSocket による共有通信処理の実装。 */
final class JdkWebSocketTransport implements WebSocketTransport, WebSocket.Listener {
  private static final Message CLOSED = new Message(null, 0);
  private final HttpClient client =
      HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
  private final BlockingQueue<Message> incoming = new LinkedBlockingQueue<>();
  private final CountDownLatch closed = new CountDownLatch(1);
  private final StringBuilder buffer = new StringBuilder();
  private WebSocket socket;
  private long receivedNanos;
  private boolean receiving;
  private volatile boolean closing;
  private volatile boolean connected;
  private volatile IOException failure = new IOException("WebSocket connection closed");

  static WebSocketTransport connect(String uri, Map<String, String> headers)
      throws IOException, InterruptedException {
    var transport = new JdkWebSocketTransport();
    CompletableFuture<WebSocket> opening = null;
    try {
      var builder = transport.client.newWebSocketBuilder().connectTimeout(Duration.ofSeconds(30));
      headers.forEach(builder::header);
      opening = builder.buildAsync(URI.create(uri), transport);
      transport.socket = opening.get(30, TimeUnit.SECONDS);
      return transport;
    } catch (ExecutionException e) {
      transport.close();
      throw new IOException("WebSocket connection failed", e.getCause());
    } catch (TimeoutException e) {
      opening.cancel(true);
      transport.close();
      throw new IOException("WebSocket connection timed out", e);
    } catch (InterruptedException | RuntimeException e) {
      if (opening != null) {
        opening.cancel(true);
      }
      transport.close();
      throw e;
    }
  }

  @Override
  public Message readMessage() throws IOException, InterruptedException {
    if (!connected && incoming.isEmpty()) {
      throw failure;
    }
    Message message = incoming.take();
    if (message == CLOSED) {
      throw failure;
    }
    return message;
  }

  @Override
  public synchronized void send(String text) throws IOException {
    if (!connected) {
      throw failure;
    }
    try {
      socket.sendText(text, true).get(30, TimeUnit.SECONDS);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while sending", e);
    } catch (ExecutionException | TimeoutException | RuntimeException e) {
      throw new IOException("WebSocket send failed", e);
    }
  }

  @Override
  public boolean isConnected() {
    return connected;
  }

  @Override
  public synchronized void close() {
    closing = true;
    connected = false;
    if (socket != null) {
      try {
        socket.sendClose(WebSocket.NORMAL_CLOSURE, "bye").get(2, TimeUnit.SECONDS);
        closed.await(2, TimeUnit.SECONDS);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      } catch (ExecutionException | TimeoutException | RuntimeException ignored) {
        // 相手が close に応答しなくても、最後に abort して資源を解放する。
      } finally {
        socket.abort();
        socket = null;
      }
    }
    client.shutdownNow();
    incoming.offer(CLOSED);
  }

  @Override
  public synchronized void onOpen(WebSocket ws) {
    if (closing) {
      ws.abort();
      return;
    }
    socket = ws;
    connected = true;
    ws.request(1);
  }

  @Override
  public CompletionStage<?> onText(WebSocket ws, CharSequence data, boolean last) {
    if (!receiving) {
      receivedNanos = System.nanoTime();
      receiving = true;
    }
    buffer.append(data);
    if (last) {
      incoming.offer(new Message(buffer.toString(), receivedNanos));
      buffer.setLength(0);
      receiving = false;
    }
    ws.request(1);
    return CompletableFuture.completedFuture(null);
  }

  @Override
  public CompletionStage<?> onClose(WebSocket ws, int statusCode, String reason) {
    connected = false;
    closed.countDown();
    incoming.offer(CLOSED);
    return CompletableFuture.completedFuture(null);
  }

  @Override
  public void onError(WebSocket ws, Throwable error) {
    failure = new IOException("WebSocket receive failed", error);
    connected = false;
    closed.countDown();
    incoming.offer(CLOSED);
  }
}
