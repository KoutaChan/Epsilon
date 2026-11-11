package com.epsilon.reviewer.scenario;

import com.epsilon.reviewer.StoredResults;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 文書から判断のインデックスを読むテスト。cause と resolve を省いた判断は、ライターと同じ決まりで読む（cause は直前のイベント、
 * resolve はその次のイベント）。何番目の判断かは、局の中で席ごとに数える。候補は各行の最初の値を読む。
 */
public class DocumentDecisionsTest {
  /** 2 局の文書。イベントの番号は、1 局目は 10 から、2 局目は 3 から始まる。 */
  private static final String DOCUMENT =
      """
      {"format":6,"resultId":"x","rounds":[
       {"roundIndex":0,"honba":0,"dealer":0,"firstEvent":10,"start":{"players":[]},"steps":[
        {"event":"start_kyoku"},
        {"event":"tsumo","actor":0},
        {"seat":0,"candidates":[[29,0.0],[33,0.5]],"chosen":0},
        {"event":"dahai","actor":0},
        {"seat":1,"candidates":[[40,0.0,["1m","2m"]],[71,0.25]],"probabilities":[0.5,0.5]},
        {"seat":2,"cause":11,"resolve":14,"candidates":[[71,0.0]]},
        {"event":"tsumo","actor":1},
        {"event":"dahai","actor":1},
        {"seat":0,"cause":12,"candidates":[[5,0.0],[71,0.1]]}]},
       {"roundIndex":1,"honba":0,"dealer":1,"firstEvent":3,"steps":[
        {"event":"start_kyoku"},
        {"event":"tsumo","actor":1},
        {"seat":1,"resolve":9,"candidates":[[7,0.0]]}]}],
      "temperature":0.002}
      """;

  @Test
  public void defaultCauseIsTheLastEventAndDefaultResolveTheNext() throws IOException {
    DocumentDecisions decisions = read();
    assertStored(decisions.find(0, 0, 12), 0, new int[] {29, 33});
    assertStored(decisions.find(0, 1, 13), 0, new int[] {40, 71});
    Assert.assertNull(decisions.find(0, 0, 11), "the cause is not the resolve");
  }

  @Test
  public void explicitCauseAndResolveAreRead() throws IOException {
    DocumentDecisions decisions = read();
    assertStored(decisions.find(0, 2, 14), 0, new int[] {71});
    assertStored(decisions.find(0, 0, 13), 1, new int[] {5, 71});
    assertStored(decisions.find(1, 1, 9), 0, new int[] {7});
    Assert.assertNull(decisions.find(1, 1, 5), "an explicit resolve replaces cause + 1");
  }

  @Test
  public void ordinalsCountEachSeatWithinItsRound() throws IOException {
    DocumentDecisions decisions = read();
    Assert.assertEquals(decisions.find(0, 0, 13).ordinal(), 1);
    Assert.assertEquals(decisions.find(1, 1, 9).ordinal(), 0, "a new round counts again");
    Assert.assertNull(decisions.find(2, 0, 3));
    Assert.assertNull(decisions.find(0, 4, 12));
    Assert.assertNull(decisions.find(0, 3, 12));
  }

  @Test
  public void aDecisionWithoutCandidatesIsRefused() {
    String broken = DOCUMENT.replace("{\"seat\":2,\"cause\":11,\"resolve\":14,\"candidates\":[[71,0.0]]}", "{\"seat\":2}");
    Assert.assertNotEquals(broken, DOCUMENT);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () ->
            DocumentDecisions.read(
                StoredResults.gzip(broken.getBytes(StandardCharsets.UTF_8)), 4));
  }

  private static DocumentDecisions read() throws IOException {
    return DocumentDecisions.read(StoredResults.gzip(DOCUMENT.getBytes(StandardCharsets.UTF_8)), 4);
  }

  private static void assertStored(DocumentDecisions.Stored stored, int ordinal, int[] candidates) {
    Assert.assertNotNull(stored);
    Assert.assertEquals(stored.ordinal(), ordinal);
    Assert.assertEquals(stored.candidates(), candidates);
  }
}
