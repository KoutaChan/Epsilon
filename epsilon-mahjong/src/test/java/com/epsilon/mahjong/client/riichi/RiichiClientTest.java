package com.epsilon.mahjong.client.riichi;

import com.epsilon.mahjong.client.Decider;
import com.epsilon.mahjong.client.FakeTransport;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

/** RiichiLab のクライアントを偽の接続で動かし、認証、検証、再接続、止まり方と、どの接続も閉じることを確かめる。 */
public class RiichiClientTest {

  private static final String TOKEN = "test-token";
  private static final String START = "{\"type\":\"start_game\",\"id\":0}";
  private static final String END = "{\"type\":\"end_game\",\"scores\":[25000,25000,25000,25000]}";
  private static final String PASSED = "{\"type\":\"validation_result\",\"passed\":true}";
  private static final String INACTIVE = "{\"error\":\"Token verification failed: Bot is inactive\"}";

  @Test
  public void validationAndRankedConnectionsEachGetAFreshDecider() throws Exception {
    Script script =
        new Script(transport(INACTIVE), transport(START, END, PASSED), transport(START, END), transport(START, END));
    Assert.assertEquals(script.play(2), new RiichiClient.Result(2, 0));
    Assert.assertEquals(script.deciders.get(), 4);
    Assert.assertEquals(
        script.uris, List.of(RiichiClient.RANKED, RiichiClient.VALIDATION, RiichiClient.RANKED, RiichiClient.RANKED));
    script.assertAllClosed();
    Assert.assertTrue(script.delays.isEmpty());
  }

  @Test
  public void aDisconnectedGameIsCountedAndTheNextGameGetsANewDecider() throws Exception {
    Script script = new Script(transport(START), transport(START, END));
    Assert.assertEquals(script.play(1), new RiichiClient.Result(1, 1));
    Assert.assertEquals(script.deciders.get(), 2);
    Assert.assertEquals(script.delays, List.of(1_000L));
    script.assertAllClosed();
  }

  @Test
  public void aRejectionAfterValidationDoesNotValidateAgain() {
    Script script = new Script(transport(INACTIVE), transport(PASSED), transport(INACTIVE));
    Assert.assertThrows(IOException.class, () -> script.play(1));
    Assert.assertEquals(script.uris, List.of(RiichiClient.RANKED, RiichiClient.VALIDATION, RiichiClient.RANKED));
    script.assertAllClosed();
    Assert.assertTrue(script.delays.isEmpty());
  }

  @Test
  public void aFailedValidationReportsBothReasonsWithoutTheToken() {
    Script script =
        new Script(
            transport("{\"error\":\"Token verification failed: ranked " + TOKEN + "\"}"),
            transport("{\"error\":\"Token verification failed: validation " + TOKEN + "\"}"));
    IOException error = Assert.expectThrows(IOException.class, () -> script.play(1));
    Assert.assertTrue(error.getMessage().contains("Ranked: Token verification failed: ranked <redacted>"));
    Assert.assertTrue(error.getMessage().contains("Validation: Token verification failed: validation <redacted>"));
    Assert.assertFalse(error.toString().contains(TOKEN));
    script.assertAllClosed();
  }

  @Test
  public void aValidationDisconnectIsNotAPass() {
    Script script = new Script(transport(INACTIVE), transport(START, END));
    IOException error = Assert.expectThrows(IOException.class, () -> script.play(1));
    Assert.assertTrue(error.getMessage().contains("Connection lost during validation"));
    Assert.assertEquals(script.deciders.get(), 2);
    Assert.assertTrue(script.delays.isEmpty());
    script.assertAllClosed();
  }

  @Test
  public void repeatedDisconnectsStopAfterABoundedBackoff() {
    FakeTransport[] transports = new FakeTransport[RiichiClient.MAX_FAILURES];
    Arrays.setAll(transports, ignored -> transport());
    Script script = new Script(transports);
    Assert.assertThrows(IOException.class, () -> script.play(1));
    Assert.assertEquals(script.deciders.get(), RiichiClient.MAX_FAILURES);
    Assert.assertEquals(
        script.delays,
        List.of(1_000L, 2_000L, 4_000L, 8_000L, 16_000L, 30_000L, 30_000L, 30_000L, 30_000L, 30_000L, 30_000L));
    script.assertAllClosed();
  }

  @Test
  public void anInterruptionDuringTheBackoffStopsAndClosesTheConnection() {
    FakeTransport transport = transport(START);
    Assert.expectThrows(
        InterruptedException.class,
        () ->
            RiichiClient.play(
                TOKEN,
                Variant.TENHOU_4P,
                0,
                RiichiClientTest::unused,
                (uri, headers) -> transport,
                delay -> {
                  throw new InterruptedException("stop");
                }));
    Assert.assertTrue(transport.closed());
  }

  /** 三人打ちのランク戦は四人打ちと同じ接続先で打ち、席 2 の対局を完走する。三人打ちに無い席 3 は、つなぎ直さずに止める。 */
  @Test
  public void threePlayerGamesUseTheSameEndpointAndThreeSeats() throws Exception {
    Script script =
        new Script(
            Variant.TENHOU_3P,
            transport("{\"type\":\"start_game\",\"id\":2}", "{\"type\":\"end_game\",\"scores\":[35000,35000,35000]}"));
    Assert.assertEquals(script.play(1), new RiichiClient.Result(1, 0));
    Assert.assertEquals(script.uris, List.of(RiichiClient.RANKED));
    script.assertAllClosed();

    Script fourth = new Script(Variant.TENHOU_3P, transport("{\"type\":\"start_game\",\"id\":3}"));
    IOException error = Assert.expectThrows(IOException.class, () -> fourth.play(1));
    Assert.assertTrue(error.getMessage().contains("protocol error"), error.getMessage());
    Assert.assertTrue(fourth.delays.isEmpty());
    fourth.assertAllClosed();
  }

  @Test
  public void aFrameThatIsNotJsonEndsWithoutARetry() {
    Script script = new Script(transport("not-json"));
    IOException error = Assert.expectThrows(IOException.class, () -> script.play(1));
    Assert.assertTrue(error.getMessage().contains("not a JSON object"));
    Assert.assertEquals(script.deciders.get(), 1);
    Assert.assertTrue(script.delays.isEmpty());
    script.assertAllClosed();
  }

  private static FakeTransport transport(String... frames) {
    return new FakeTransport(List.of(frames));
  }

  private static Decider unused() {
    return (view, decisions, slot) -> {
      throw new AssertionError("No action is requested on these connections");
    };
  }

  /** 用意した接続を順に渡し、接続先、再接続までの待ち時間、作った Decider の数を記録する。 */
  private static final class Script {

    private final Variant variant;
    private final List<FakeTransport> transports;
    private final List<URI> uris = new ArrayList<>();
    private final List<Long> delays = new ArrayList<>();
    private final AtomicInteger deciders = new AtomicInteger();

    Script(FakeTransport... transports) {
      this(Variant.TENHOU_4P, transports);
    }

    Script(Variant variant, FakeTransport... transports) {
      this.variant = variant;
      this.transports = List.of(transports);
    }

    RiichiClient.Result play(int games) throws IOException, InterruptedException {
      return RiichiClient.play(
          TOKEN,
          variant,
          games,
          () -> {
            deciders.incrementAndGet();
            return unused();
          },
          (uri, headers) -> {
            Assert.assertEquals(headers.get("Authorization"), "Bearer " + TOKEN);
            int index = uris.size();
            if (index > 0) {
              Assert.assertTrue(transports.get(index - 1).closed());
            }
            uris.add(uri);
            return transports.get(index);
          },
          delays::add);
    }

    void assertAllClosed() {
      for (int index = 0; index < uris.size(); index++) {
        Assert.assertTrue(transports.get(index).closed(), "connection " + index);
      }
    }
  }
}
