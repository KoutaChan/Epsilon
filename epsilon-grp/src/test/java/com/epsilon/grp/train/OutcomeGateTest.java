package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeSection;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 局の結果のネットワーク q の関門を確かめる。エンジンで打った半荘の数のファイルから局末を作り、数え上げる局末のうち、記録した結果を
 * 表せる局末だけを測る。結果を表せない局末は数えるだけにする。差は、記録した結果の −log q の差（候補 − 比べる q）を局末あたりに
 * 平均した値。同じ q なら差も標準誤差も 0 で、関門を通る。数えて作った q は一様な q より良く、初めての q として通る。ずれた q は、
 * 使用中の q と比べても一様な q と比べても落ちる。一様な q そのものは差が 0 なので、初めての q としては通らない。測る局末が無ければ
 * 差は NaN で、どちらの関門も通らない。
 */
public class OutcomeGateTest {

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  @Test(dataProvider = "variants")
  public void gateComparesTheMarginalNllOnCoveredHandEnds(Variant variant) {
    EngineGames.Played played = EngineGames.play(variant, 200, 1, 41 + variant.seats());
    HandEnds ends = HandEnds.of(variant, RankUtility.TENHOU, played.games());
    CountedHands counted = EngineGames.counted(variant, played.hands());
    NextHands next = NextHands.of(ends, counted);
    OutcomeShape shape = new OutcomeShape(variant, GrpFixtures.plainFields(), 8);
    OutcomeNetwork uniform = OutcomeFixtures.uniform(shape);
    OutcomeRows rows = OutcomeRows.builder(variant, new RecencyWindow(4, 1, 64), GateCohort.NEXT, state -> true)
        .add(4, counted)
        .build()
        .train();
    OutcomeNetwork fitted = OutcomeFixtures.counted(shape, rows);
    OutcomeNetwork wrong = abortive(shape, 5);

    Map<Long, List<EngineGames.Hand>> byGame = new HashMap<>();
    for (EngineGames.Hand hand : played.hands()) {
      byGame.computeIfAbsent(hand.gameId(), id -> new ArrayList<>()).add(hand);
    }
    OutcomeCodes codes = new OutcomeCodes(variant);
    short[] terms = new short[shape.outcomes().maxTerms()];
    OutcomeWorkspace ofFitted = OutcomeWorkspace.of(shape);
    OutcomeWorkspace ofUniform = OutcomeWorkspace.of(shape);
    long measured = 0;
    long skipped = 0;
    double difference = 0;
    for (int h = 0; h < ends.handEnds(); h++) {
      if (!FinalRound.covers(ends.state(h))) {
        continue;
      }
      EngineGames.Hand hand = byGame.get(ends.gameId(ends.game(h))).get(h - ends.firstHand(ends.game(h)) + 1);
      int count = codes.terms(HandResult.raw(hand.outcome(), hand.start().dealer()), terms, 0);
      if (count == 0) {
        skipped++;
        continue;
      }
      measured++;
      fitted.forward(ends.state(h), ofFitted);
      uniform.forward(ends.state(h), ofUniform);
      for (int j = 0; j < count; j++) {
        difference += ofUniform.term(terms[j]) - ofFitted.term(terms[j]);
      }
    }
    Assert.assertTrue(measured > 100, variant + " " + measured);

    OutcomeGate.Report same = OutcomeGate.refit(next, fitted, fitted);
    Assert.assertEquals(same.kind(), OutcomeGate.Kind.REFIT);
    Assert.assertTrue(same.passed(), same.toString());
    Assert.assertEquals(same.versus().mean(), 0, 0, same.toString());
    Assert.assertEquals(same.versus().standardError(), 0, 0, same.toString());
    Assert.assertEquals(same.handEnds(), measured, same.toString());
    Assert.assertEquals(same.skipped(), skipped, same.toString());

    OutcomeGate.Report better = OutcomeGate.refit(next, uniform, fitted);
    double mean = difference / measured;
    Assert.assertTrue(better.passed(), better.toString());
    Assert.assertEquals(better.versus().mean(), mean, 1e-12 * Math.abs(mean), better.toString());
    Assert.assertTrue(better.versus().z() < -2, better.toString());
    Assert.assertEquals(better.handEnds(), measured);
    Assert.assertEquals(better.skipped(), skipped);
    OutcomeGate.Report first = OutcomeGate.first(next, fitted);
    Assert.assertEquals(first.kind(), OutcomeGate.Kind.FIRST);
    Assert.assertTrue(first.passed(), first.toString());
    Assert.assertEquals(first.versus(), better.versus());

    OutcomeGate.Report worse = OutcomeGate.refit(next, fitted, wrong);
    Assert.assertFalse(worse.passed(), worse.toString());
    Assert.assertTrue(worse.versus().mean() > OutcomeGate.STANDARD_ERRORS * worse.versus().standardError(),
        worse.toString());
    OutcomeGate.Report firstWrong = OutcomeGate.first(next, wrong);
    Assert.assertFalse(firstWrong.passed(), firstWrong.toString());
    Assert.assertTrue(firstWrong.versus().mean() > 0, firstWrong.toString());
    OutcomeGate.Report firstUniform = OutcomeGate.first(next, uniform);
    Assert.assertFalse(firstUniform.passed(), firstUniform.toString());
    Assert.assertEquals(firstUniform.versus().mean(), 0, 0, firstUniform.toString());

    HandEnds covered = ends.select(h -> FinalRound.covers(ends.state(h)));
    Assert.assertEquals(OutcomeGate.refit(NextHands.of(covered, counted), uniform, fitted), better);

    HandEnds early = ends.select(h -> !FinalRound.covers(ends.state(h)));
    NextHands none = NextHands.of(early, counted);
    for (OutcomeGate.Report empty : List.of(OutcomeGate.refit(none, fitted, fitted), OutcomeGate.first(none, fitted))) {
      Assert.assertFalse(empty.passed(), empty.toString());
      Assert.assertEquals(empty.handEnds(), 0, empty.toString());
      Assert.assertEquals(empty.skipped(), 0, empty.toString());
      Assert.assertTrue(Double.isNaN(empty.versus().mean()), empty.toString());
    }

    Variant other = variant == Variant.TENHOU_4P ? Variant.TENHOU_3P : Variant.TENHOU_4P;
    OutcomeNetwork foreign = OutcomeFixtures.uniform(new OutcomeShape(other, GrpFixtures.plainFields(), 8));
    Assert.assertThrows(IllegalArgumentException.class, () -> OutcomeGate.refit(next, fitted, foreign));
    Assert.assertThrows(IllegalArgumentException.class, () -> OutcomeGate.refit(next, foreign, fitted));
    Assert.assertThrows(IllegalArgumentException.class, () -> OutcomeGate.first(next, foreign));
  }

  /** 卓に依らず、途中流局のロジットだけを持ち上げた q を作る。ほかの終わり方の確率は一様より低くなる。 */
  private static OutcomeNetwork abortive(OutcomeShape shape, double logit) {
    HandOutcomes hands = shape.outcomes();
    double[] logits = new double[hands.logits()];
    for (int main = 0; main < hands.mainCount(); main++) {
      if (hands.section(main) == OutcomeSection.ABORTIVE) {
        logits[hands.mainTerm(main)] = logit;
      }
    }
    return GrpFixtures.biased(shape, logits);
  }
}
