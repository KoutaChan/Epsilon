package com.epsilon.mahjong.client.tenhou;

import com.epsilon.mahjong.client.FakeTransport;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 天鳳のクライアントを、決まったメッセージを返す偽の接続で動かし、ログイン、卓の申し込み、対局中の返事を確かめる。 */
public class TenhouClientTest {

  private static final String HELO = "{\"tag\":\"HELO\",\"uname\":\"Epsilon\",\"auth\":\"20261107-0000\"}";

  @DataProvider
  public Object[][] tables() {
    return new Object[][] {{Variant.TENHOU_4P, 1, "9"}, {Variant.TENHOU_3P, 2, "25"}};
  }

  /** ログインして、卓のルールに合う種別で個室に申し込む。自己対局の半荘の全てのメッセージに返すはずのメッセージを送り、終局で戻る。 */
  @Test(dataProvider = "tables")
  public void aRecordedGameIsPlayedAfterTheLogin(Variant variant, int seed, String type)
      throws IOException, InterruptedException {
    String id = "self-" + seed;
    String xml = TenhouXml.write(RecordedGames.play(variant, seed, id));
    LiveTranscript transcript = LiveTranscript.of(variant, xml, id, 2, true);
    List<String> frames = new ArrayList<>();
    frames.add(HELO);
    List<String> expected = new ArrayList<>();
    expected.add("{\"tag\":\"HELO\",\"name\":\"Epsilon+%E3%83%86%E3%82%B9%E3%83%88\",\"sx\":\"M\"}");
    expected.add("{\"tag\":\"LOBBY\",\"id\":\"C12345678\"}");
    expected.add("{\"tag\":\"JOIN\",\"t\":\"C12345678," + type + "\"}");
    for (LiveTranscript.Line line : transcript.lines()) {
      frames.add(line.message());
      expected.addAll(line.replies());
    }
    ArrayDeque<LiveTranscript.Decision> decisions = new ArrayDeque<>(transcript.decisions());
    FakeTransport transport = new FakeTransport(frames);
    int mismatches =
        TenhouClient.play(
            transport,
            variant,
            "C12345678",
            "Epsilon テスト",
            (view, slots, slot) -> {
              LiveTranscript.Decision decision = decisions.removeFirst();
              return decision.legal().indexOf(decision.code());
            });
    Assert.assertEquals(mismatches, 0);
    Assert.assertEquals(transport.sent(), expected);
    Assert.assertTrue(decisions.isEmpty());
  }

  @Test
  public void aLoginAnsweredWithAnotherTagFails() {
    FakeTransport transport = new FakeTransport(List.of("<LN/>"));
    Assert.assertThrows(
        IOException.class,
        () ->
            TenhouClient.play(
                transport,
                Variant.TENHOU_4P,
                "C1",
                "Epsilon",
                (view, slots, slot) -> {
                  throw new AssertionError("No decision before the login");
                }));
  }

  /** 終局の前に接続が切れたら IOException を投げる。 */
  @Test
  public void aConnectionLostMidGameFails() {
    FakeTransport transport = new FakeTransport(List.of(HELO, "<GO type=\"9\" lobby=\"0\"/>"));
    Assert.assertThrows(
        IOException.class,
        () ->
            TenhouClient.play(
                transport,
                Variant.TENHOU_4P,
                "C1",
                "Epsilon",
                (view, slots, slot) -> {
                  throw new AssertionError("No decision without a round");
                }));
    Assert.assertEquals(transport.sent().get(transport.sent().size() - 1), LiveTranscript.GOK);
  }
}
