package com.epsilon.client.riichi;

import com.epsilon.client.WebSocketTransport;
import com.epsilon.engine.Player;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 偽の通信路を使い、認証・再接続・終了時の資源解放を検証する。 */
public class RiichiClientTest {
  private static final String START = "{\"type\":\"start_game\",\"id\":0}";
  private static final String END = "{\"type\":\"end_game\",\"scores\":[25000,25000,25000,25000]}";
  private static final String PASSED = "{\"type\":\"validation_result\",\"passed\":true}";
  private static final String INACTIVE =
      "{\"error\":\"Token verification failed: Bot is inactive\"}";

  @Test
  public void validationAndRankedConnectionsEachReceiveAFreshPlayer() throws Exception {
    var script =
        new Script(
            new FakeTransport(INACTIVE),
            new FakeTransport(START, END, PASSED),
            new FakeTransport(START, END),
            new FakeTransport(START, END));
    Assert.assertEquals(script.play(2), new RiichiClient.Result(2, 0));
    Assert.assertEquals(script.players.get(), 4);
    Assert.assertEquals(
        script.uris,
        List.of(
            RiichiClient.RANKED_URI,
            RiichiClient.VALIDATION_URI,
            RiichiClient.RANKED_URI,
            RiichiClient.RANKED_URI));
    Assert.assertTrue(script.transports.stream().allMatch(transport -> transport.closed));
    Assert.assertTrue(script.delays.isEmpty());
  }

  @Test
  public void aDisconnectedGameIsCountedAndTheNextGameGetsANewPlayer() throws Exception {
    var script = new Script(new FakeTransport(START), new FakeTransport(START, END));
    Assert.assertEquals(script.play(1), new RiichiClient.Result(1, 1));
    Assert.assertEquals(script.players.get(), 2);
    Assert.assertEquals(script.delays, List.of(1_000L));
    Assert.assertTrue(script.transports.stream().allMatch(transport -> transport.closed));
  }

  @Test
  public void authenticationRejectionAfterValidationDoesNotRunValidationAgain() {
    var script =
        new Script(
            new FakeTransport(INACTIVE), new FakeTransport(PASSED), new FakeTransport(INACTIVE));
    Assert.expectThrows(IOException.class, () -> script.play(1));
    Assert.assertEquals(
        script.uris,
        List.of(RiichiClient.RANKED_URI, RiichiClient.VALIDATION_URI, RiichiClient.RANKED_URI));
    Assert.assertTrue(script.transports.stream().allMatch(transport -> transport.closed));
    Assert.assertTrue(script.delays.isEmpty());
  }

  @Test
  public void validationFailureReportsBothReasonsAndRedactsTheToken() {
    var script =
        new Script(
            new FakeTransport("{\"error\":\"Token verification failed: ranked test-token\"}"),
            new FakeTransport("{\"error\":\"Token verification failed: validation test-token\"}"));
    IOException error = Assert.expectThrows(IOException.class, () -> script.play(1));
    Assert.assertTrue(
        error.getMessage().contains("Ranked: Token verification failed: ranked <redacted>"));
    Assert.assertTrue(
        error
            .getMessage()
            .contains("Validation: Token verification failed: validation <redacted>"));
    Assert.assertFalse(error.toString().contains("test-token"));
    Assert.assertTrue(script.transports.stream().allMatch(transport -> transport.closed));
  }

  @Test
  public void validationDisconnectCannotBeMistakenForPassingValidation() {
    var script = new Script(new FakeTransport(INACTIVE), new FakeTransport(START, END));
    IOException error = Assert.expectThrows(IOException.class, () -> script.play(1));
    Assert.assertTrue(error.getMessage().contains("Connection lost during validation"));
    Assert.assertEquals(script.players.get(), 2);
    Assert.assertTrue(script.delays.isEmpty());
    Assert.assertTrue(script.transports.stream().allMatch(transport -> transport.closed));
  }

  @Test
  public void repeatedDisconnectsStopAfterBoundedBackoff() {
    var transports = new FakeTransport[12];
    Arrays.setAll(transports, ignored -> new FakeTransport());
    var script = new Script(transports);
    Assert.expectThrows(IOException.class, () -> script.play(1));
    Assert.assertEquals(script.players.get(), 12);
    Assert.assertEquals(
        script.delays,
        List.of(
            1_000L, 2_000L, 4_000L, 8_000L, 16_000L, 30_000L, 30_000L, 30_000L, 30_000L, 30_000L,
            30_000L));
    Assert.assertTrue(script.transports.stream().allMatch(transport -> transport.closed));
  }

  @Test
  public void interruptionDuringBackoffStopsUnlimitedPlayAndClosesTheConnection() {
    var transport = new FakeTransport(START);
    Assert.expectThrows(
        InterruptedException.class,
        () ->
            RiichiClient.play(
                "test-token",
                0,
                () -> unusedPlayer(0),
                (uri, headers) -> transport,
                delay -> {
                  throw new InterruptedException("stop");
                }));
    Assert.assertTrue(transport.closed);
  }

  @Test
  public void malformedProtocolEndsTheConnectionWithoutRetry() {
    var script = new Script(new FakeTransport("not-json"));
    IOException error = Assert.expectThrows(IOException.class, () -> script.play(1));
    Assert.assertTrue(error.getMessage().contains("not a JSON object"));
    Assert.assertEquals(script.players.get(), 1);
    Assert.assertTrue(script.delays.isEmpty());
    Assert.assertTrue(script.transports.getFirst().closed);
  }

  private static Player unusedPlayer(int id) {
    return (state, seat, legal) -> {
      throw new AssertionError("接続イベントだけでは player " + id + " の推論を開始しない");
    };
  }

  private static final class Script {
    private final List<FakeTransport> transports;
    private final List<String> uris = new ArrayList<>();
    private final List<Long> delays = new ArrayList<>();
    private final AtomicInteger players = new AtomicInteger();

    Script(FakeTransport... transports) {
      this.transports = List.of(transports);
    }

    RiichiClient.Result play(int maxGames) throws IOException, InterruptedException {
      return RiichiClient.play(
          "test-token",
          maxGames,
          () -> unusedPlayer(players.incrementAndGet()),
          (uri, headers) -> {
            Assert.assertEquals(headers.get("Authorization"), "Bearer test-token");
            int index = uris.size();
            if (index > 0) Assert.assertTrue(transports.get(index - 1).closed);
            uris.add(uri);
            return transports.get(index);
          },
          delays::add);
    }
  }

  private static final class FakeTransport implements WebSocketTransport {
    private final ArrayDeque<String> frames;
    private boolean closed;

    FakeTransport(String... messages) {
      frames = new ArrayDeque<>(List.of(messages));
    }

    @Override
    public Message readMessage() throws IOException {
      if (frames.isEmpty()) throw new IOException("connection closed");
      return new Message(frames.removeFirst(), System.nanoTime());
    }

    @Override
    public void send(String text) {
      throw new AssertionError("この接続シナリオでは行動を送信しない: " + text);
    }

    @Override
    public boolean isConnected() {
      return !closed;
    }

    @Override
    public void close() {
      closed = true;
    }
  }
}
