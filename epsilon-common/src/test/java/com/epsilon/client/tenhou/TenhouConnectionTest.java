package com.epsilon.client.tenhou;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Base64;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.Test;

/** ローカルの WebSocket サーバーで天鳳接続時の実際の HTTP ヘッダーを検証する。 */
public class TenhouConnectionTest {
  @Test(timeOut = 15000)
  public void openingHandshakeIncludesTenhouOrigin() throws Exception {
    try (var executor = Executors.newSingleThreadExecutor();
        var server = new ServerSocket(0, 1, InetAddress.getByName("127.0.0.1"))) {
      server.setSoTimeout(5000);
      var openingRequest = executor.submit(() -> acceptWebSocket(server));

      try (var connection = new TenhouConnection("ws://127.0.0.1:" + server.getLocalPort() + "/")) {
        connection.connect();
        Assert.assertTrue(connection.isConnected());
      }

      Map<String, String> headers = openingRequest.get(5, TimeUnit.SECONDS);
      Assert.assertEquals(headers.get("Origin"), "https://tenhou.net");
    }
  }

  private static Map<String, String> acceptWebSocket(ServerSocket server) throws Exception {
    try (var socket = server.accept()) {
      socket.setSoTimeout(5000);
      var input = socket.getInputStream();
      var reader = new BufferedReader(new InputStreamReader(input, StandardCharsets.US_ASCII));
      Assert.assertEquals(reader.readLine(), "GET / HTTP/1.1");
      Map<String, String> headers = new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
      String line;
      while ((line = reader.readLine()) != null && !line.isEmpty()) {
        int separator = line.indexOf(':');
        Assert.assertTrue(separator > 0, "HTTP ヘッダーの形式: " + line);
        headers.put(line.substring(0, separator), line.substring(separator + 1).trim());
      }
      Assert.assertNotNull(line, "HTTP ヘッダー終端まで受信する");
      String key = headers.get("Sec-WebSocket-Key");
      Assert.assertNotNull(key);
      String accept =
          Base64.getEncoder()
              .encodeToString(
                  MessageDigest.getInstance("SHA-1")
                      .digest(
                          (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11")
                              .getBytes(StandardCharsets.US_ASCII)));
      var output = socket.getOutputStream();
      output.write(
          ("HTTP/1.1 101 Switching Protocols\r\n"
                  + "Upgrade: websocket\r\n"
                  + "Connection: Upgrade\r\n"
                  + "Sec-WebSocket-Accept: "
                  + accept
                  + "\r\n\r\n")
              .getBytes(StandardCharsets.US_ASCII));
      output.flush();

      // クライアントの close に応答し、切断待ちのタイムアウトを発生させず終了する。
      var frames = new DataInputStream(input);
      Assert.assertEquals(frames.readUnsignedByte(), 0x88);
      int maskedLength = frames.readUnsignedByte();
      Assert.assertTrue((maskedLength & 0x80) != 0);
      int length = maskedLength & 0x7f;
      Assert.assertTrue(length <= 125);
      byte[] mask = new byte[4];
      frames.readFully(mask);
      byte[] payload = new byte[length];
      frames.readFully(payload);
      for (int i = 0; i < payload.length; i++) {
        payload[i] ^= mask[i % mask.length];
      }
      output.write(0x88);
      output.write(length);
      output.write(payload);
      output.flush();
      return headers;
    }
  }
}
