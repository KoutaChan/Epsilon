package com.epsilon.mahjong.client;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.Base64;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.testng.Assert;
import org.testng.annotations.Test;

/** JDK の WebSocket を使う接続のテスト。テストの中で待ち受ける相手につなぎ、相手が黙ったときの read を確かめる。 */
public class WebSocketTransportTest {

  private static final Duration PING_INTERVAL = Duration.ofMillis(200);

  /** 相手が Ping に応えなければ、read は待ち続けずに IOException を投げる。 */
  @Test(timeOut = 10_000)
  public void aPeerThatAnswersNoPingEndsTheRead() throws Exception {
    try (Peer peer = new Peer(false);
        WebSocketTransport transport = WebSocketTransport.connect(peer.uri(), Map.of(), PING_INTERVAL)) {
      Assert.assertEquals(transport.read().text(), Peer.GREETING);
      IOException lost = Assert.expectThrows(IOException.class, transport::read);
      Assert.assertTrue(lost.getMessage().contains("of a ping"), lost.getMessage());
    }
  }

  /** 相手が Ping に応えている間は、メッセージが届かなくても read は待ち、後から届いたメッセージを返す。 */
  @Test(timeOut = 10_000)
  public void aPeerThatAnswersPingsIsWaitedFor() throws Exception {
    try (Peer peer = new Peer(true);
        WebSocketTransport transport = WebSocketTransport.connect(peer.uri(), Map.of(), PING_INTERVAL)) {
      Assert.assertEquals(transport.read().text(), Peer.GREETING);
      Assert.assertEquals(transport.read().text(), Peer.LATE);
    }
  }

  /**
   * このテストの中で待ち受ける WebSocket の相手。接続を 1 つ受け、ハンドシェイクに応えて挨拶のメッセージを送る。その後は、Ping に
   * 応えるなら Pong を返し、3 つ目の Ping の後にもう 1 つメッセージを送る。応えないなら何も送らない。切断にはどちらも応える。
   */
  private static final class Peer implements AutoCloseable {

    static final String GREETING = "hello";
    static final String LATE = "late";

    private static final int PINGS_BEFORE_LATE = 3;

    /** ハンドシェイクの応答の Sec-WebSocket-Accept を作るときに、鍵の後ろへ付ける決まった文字列（RFC 6455）。 */
    private static final String ACCEPT_SUFFIX = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

    private static final Pattern KEY = Pattern.compile("(?im)^Sec-WebSocket-Key:\\s*(\\S+)");
    private static final String HEADERS_END = "\r\n\r\n";

    /** フレームの 1 バイト目の、最後の断片を表すビット。 */
    private static final int FIN = 0x80;

    private static final int OPCODE_MASK = 0x0f;

    /** フレームの 2 バイト目の、中身の長さのビット（このテストのフレームはどれも 125 バイト以下）。 */
    private static final int LENGTH_MASK = 0x7f;

    private static final int MASK_BYTES = 4;
    private static final int TEXT = 0x1;
    private static final int CLOSE = 0x8;
    private static final int PING = 0x9;
    private static final int PONG = 0xa;

    private final ServerSocket server = new ServerSocket(0, 1, InetAddress.getLoopbackAddress());
    private final boolean answersPings;

    Peer(boolean answersPings) throws IOException {
      this.answersPings = answersPings;
      Thread thread = new Thread(this::serve, "websocket-peer");
      thread.setDaemon(true);
      thread.start();
    }

    URI uri() {
      return URI.create("ws://" + server.getInetAddress().getHostAddress() + ":" + server.getLocalPort() + "/");
    }

    @Override
    public void close() throws IOException {
      server.close();
    }

    private void serve() {
      try (Socket socket = server.accept()) {
        DataInputStream in = new DataInputStream(socket.getInputStream());
        OutputStream out = socket.getOutputStream();
        out.write(handshake(in));
        send(out, TEXT, GREETING.getBytes(StandardCharsets.UTF_8));
        int pings = 0;
        while (true) {
          int opcode = in.readUnsignedByte() & OPCODE_MASK;
          byte[] payload = new byte[in.readUnsignedByte() & LENGTH_MASK];
          byte[] mask = in.readNBytes(MASK_BYTES);
          in.readFully(payload);
          for (int index = 0; index < payload.length; index++) {
            payload[index] ^= mask[index % MASK_BYTES];
          }
          if (opcode == CLOSE) {
            send(out, CLOSE, payload);
            return;
          }
          if (opcode == PING && answersPings) {
            send(out, PONG, payload);
            if (++pings == PINGS_BEFORE_LATE) {
              send(out, TEXT, LATE.getBytes(StandardCharsets.UTF_8));
            }
          }
        }
      } catch (IOException closed) {
        // テストが接続か待ち受けを閉じた。
      }
    }

    /** ハンドシェイクの要求をヘッダーの終わりまで読み、応答を返す。 */
    private static byte[] handshake(DataInputStream in) throws IOException {
      StringBuilder request = new StringBuilder();
      while (request.indexOf(HEADERS_END) < 0) {
        request.append((char) in.readUnsignedByte());
      }
      Matcher key = KEY.matcher(request);
      if (!key.find()) {
        throw new IOException("The handshake has no Sec-WebSocket-Key");
      }
      byte[] digest;
      try {
        digest =
            MessageDigest.getInstance("SHA-1")
                .digest((key.group(1) + ACCEPT_SUFFIX).getBytes(StandardCharsets.US_ASCII));
      } catch (NoSuchAlgorithmException missing) {
        throw new IllegalStateException(missing);
      }
      String response =
          "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: "
              + Base64.getEncoder().encodeToString(digest)
              + HEADERS_END;
      return response.getBytes(StandardCharsets.US_ASCII);
    }

    /** マスクなしの、断片に分けないフレームを 1 つ送る。 */
    private static void send(OutputStream out, int opcode, byte[] payload) throws IOException {
      out.write(FIN | opcode);
      out.write(payload.length);
      out.write(payload);
      out.flush();
    }
  }
}
