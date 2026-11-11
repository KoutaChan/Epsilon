package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeSection;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.function.IntPredicate;
import java.util.function.Supplier;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 局の結果のネットワークについて、較正の測り方を確かめる。温度 1 の NLL は OutcomeRows の NLL と同じになる。温度 2 の
 * ネットワークでは、終わり方の対数確率が ℓ/2 − log Σ exp(ℓ/2) になる（ℓ は元の対数確率）。行の結果を数えて作った、卓に依らない
 * ネットワークの温度は 1 に近い。そのロジットを 2 倍にしたネットワークの温度は 2 倍になる。信頼性の表は、行と終わり方の全ての組を
 * 1 度ずつ数える。予測の率の和も実際の率の和も行の数になり、区分ごとの予測の率の和も実際の率の和も 1 になる。数のファイルの局末で
 * 測る較正（{@code check}）は、局の結果を q から引いた半荘では温度が 1 に近い。q がずれていれば、ずれたセルを外れとして拾う。
 */
public class OutcomeCalibrationTest {

  @Test
  public void temperaturesAndReliabilityOfOutcomeNetworks() {
    OutcomeRows rows = OutcomeFixtures.rows(RecencyWindow.uniform(4, 2), 150).train();
    OutcomeShape shape = new OutcomeShape(OutcomeFixtures.FOUR, GrpFixtures.plainFields(), 8);
    HandOutcomes hands = shape.outcomes();
    Supplier<OutcomeWorkspace> workspaces = () -> OutcomeWorkspace.of(shape);
    OutcomeNetwork random = GrpFixtures.outcomes(shape, 11, 0.5);
    List<OutcomeCalibration.Part> randomRows = List.of(new OutcomeCalibration.Part(rows, random));
    double nll = OutcomeCalibration.nll(randomRows, 1, workspaces);
    Assert.assertEquals(nll, rows.nll(random, OutcomeWorkspace.of(shape)), 1e-12 * nll);
    Assert.assertEquals(OutcomeCalibration.tempered(random, 1).weights(), random.weights());

    OutcomeNetwork softened = OutcomeCalibration.tempered(random, 2);
    OutcomeWorkspace original = OutcomeWorkspace.of(shape);
    OutcomeWorkspace tempered = OutcomeWorkspace.of(shape);
    for (int row = 0; row < 20; row++) {
      GrpState state = GrpState.ofKey(OutcomeFixtures.FOUR, rows.key(row));
      random.forward(state, original);
      softened.forward(state, tempered);
      double sum = 0;
      for (int main = 0; main < hands.mainCount(); main++) {
        sum += Math.exp(original.term(hands.mainTerm(main)) / 2);
      }
      for (int main = 0; main < hands.mainCount(); main++) {
        int term = hands.mainTerm(main);
        Assert.assertEquals(tempered.term(term), original.term(term) / 2 - Math.log(sum), 1e-9, "row " + row);
      }
    }

    OutcomeNetwork counted = OutcomeFixtures.counted(shape, rows);
    List<OutcomeCalibration.Part> countedRows = List.of(new OutcomeCalibration.Part(rows, counted));
    double fitted = OutcomeCalibration.temperature(countedRows, workspaces);
    Assert.assertTrue(fitted > 0.8 && fitted < 1.25, "temperature " + fitted);
    double best = OutcomeCalibration.nll(countedRows, fitted, workspaces);
    Assert.assertTrue(best <= OutcomeCalibration.nll(countedRows, 1, workspaces) + 1e-12);
    Assert.assertTrue(best <= OutcomeCalibration.nll(countedRows, fitted * 1.05, workspaces));
    Assert.assertTrue(best <= OutcomeCalibration.nll(countedRows, fitted / 1.05, workspaces));
    OutcomeNetwork sharpened = OutcomeCalibration.tempered(counted, 0.5);
    double doubled =
        OutcomeCalibration.temperature(List.of(new OutcomeCalibration.Part(rows, sharpened)), workspaces);
    Assert.assertEquals(doubled, 2 * fitted, 0.01 * fitted);

    OutcomeCalibration.Reliability reliability = OutcomeCalibration.reliability(countedRows, workspaces);
    Assert.assertEquals(reliability.rows(), rows.size());
    // 途中流局と全員聴牌の荒牌は、点数の動きも局の進み方も同じなので、どちらの終わり方でも再現できる。
    Assert.assertTrue(reliability.ambiguousRows() > 0 && reliability.ambiguousRows() < rows.size(),
        reliability.ambiguousRows() + " ambiguous rows");
    long pairs = 0;
    double predicted = 0;
    double realized = 0;
    for (OutcomeCalibration.Bin bin : reliability.bins()) {
      if (bin.pairs() > 0) {
        pairs += bin.pairs();
        predicted += bin.pairs() * bin.predicted();
        realized += bin.pairs() * bin.realized();
      }
    }
    Assert.assertEquals(pairs, (long) rows.size() * hands.mainCount());
    Assert.assertEquals(predicted, rows.size(), 1e-9 * rows.size());
    Assert.assertEquals(realized, rows.size(), 1e-9 * rows.size());
    double sectionPredicted = 0;
    double sectionRealized = 0;
    for (OutcomeCalibration.Bin section : reliability.sections()) {
      Assert.assertEquals(section.pairs(), rows.size(), section.name());
      sectionPredicted += section.predicted();
      sectionRealized += section.realized();
    }
    Assert.assertEquals(sectionPredicted, 1, 1e-9);
    Assert.assertEquals(sectionRealized, 1, 1e-9);
  }

  /**
   * 局の結果を q から引いて打った半荘の数のファイル（局末ごとに記録した結果を 1 つ持つ）で、較正を測る。NLL は、記録した結果の
   * −log q を局末あたりに平均した値。温度は 1 に近く、q のロジットを温度 0.5 で鋭くすると、測った温度は 2 倍になる。選んだ局末は
   * 全て測られ、どれも表せる。外れは、区分ごとのセルの分類のうち |z| が 3 を越えるものだけで、|z| の大きい順に並ぶ。終わり方を
   * 荒牌に寄せた q では、荒牌の分類が外れる。記録した結果だけで求めた NLL は、遷移を再現する結果を全て持つ半荘の記録から求めた NLL
   * より小さくならない。局末を 1 つも選ばなければ、数は 0 で、NLL と温度は NaN になる。
   */
  @Test(dataProvider = "variants")
  public void exactRowsCheckTheDrawingNetwork(Variant variant) {
    OutcomeShape shape = new OutcomeShape(variant, GrpFixtures.plainFields(), 8);
    HandOutcomes hands = shape.outcomes();
    OutcomeNetwork q = GrpFixtures.outcomes(shape, 19, 0.5);
    OutcomeFixtures.Drawn drawn = OutcomeFixtures.drawn(q, 500, 1, 7 + variant.seats());
    HandEnds ends = HandEnds.of(variant, RankUtility.TENHOU, drawn.games());
    CountedHands counted = drawn.counted();
    NextHands next = NextHands.of(ends, counted);

    OutcomeCodes codes = new OutcomeCodes(variant);
    OutcomeWorkspace workspace = OutcomeWorkspace.of(shape);
    short[] terms = new short[hands.maxTerms()];
    double[] nll = new double[ends.handEnds()];
    for (int h = 0; h < ends.handEnds(); h++) {
      int hand = counted.firstHand(ends.game(h)) + h - ends.firstHand(ends.game(h)) + 1;
      q.forward(ends.state(h), workspace);
      int count = codes.terms(counted.code(hand), terms, 0);
      Assert.assertTrue(count > 0, ends.state(h).toString());
      for (int j = 0; j < count; j++) {
        nll[h] -= workspace.term(terms[j]);
      }
    }
    IntPredicate covered = h -> FinalRound.covers(ends.state(h));
    OutcomeCalibration.Check all = OutcomeCalibration.check(next, h -> true, q);
    OutcomeCalibration.Check late = OutcomeCalibration.check(next, covered, q);
    for (OutcomeCalibration.Check check : List.of(all, late)) {
      IntPredicate selected = check == all ? h -> true : covered;
      double sum = 0;
      long count = 0;
      for (int h = 0; h < ends.handEnds(); h++) {
        if (selected.test(h)) {
          sum += nll[h];
          count++;
        }
      }
      Assert.assertTrue(count > 50, variant + " " + count);
      Assert.assertEquals(check.handEnds(), count, check.toString());
      Assert.assertEquals(check.skipped(), 0, check.toString());
      Assert.assertEquals(check.nll(), sum / count, 1e-12 * sum / count, check.toString());
    }
    Assert.assertTrue(Math.abs(all.temperature() - 1) < 0.1, all.toString());
    OutcomeCalibration.Check sharpened = OutcomeCalibration.check(next, h -> true, OutcomeCalibration.tempered(q, 0.5));
    Assert.assertEquals(sharpened.temperature(), 2 * all.temperature(), 0.02 * all.temperature(), sharpened.toString());

    assertOutliers(all.outliers());
    double[] logits = new double[hands.logits()];
    for (int main = 0; main < hands.mainCount(); main++) {
      if (hands.section(main) == OutcomeSection.DRAW) {
        logits[hands.mainTerm(main)] = 3;
      }
    }
    OutcomeCalibration.Check drawing = OutcomeCalibration.check(next, h -> true, GrpFixtures.biased(shape, logits));
    assertOutliers(drawing.outliers());
    Assert.assertTrue(
        drawing.outliers().stream().anyMatch(outlier -> outlier.rate().name().equals("DRAW")
            && outlier.rate().realized() > 0 && outlier.rate().predicted() > outlier.rate().realized() + 0.3),
        drawing.outliers().toString());

    NextHands searched = NextHands.of(ends, drawn.games());
    double recorded = OutcomeCalibration.nll(searched, h -> true, q, 1, () -> OutcomeWorkspace.of(shape));
    Assert.assertTrue(all.nll() >= recorded - 1e-12, all.nll() + " vs " + recorded);

    OutcomeCalibration.Check none = OutcomeCalibration.check(next, h -> false, q);
    Assert.assertEquals(none.handEnds(), 0);
    Assert.assertTrue(Double.isNaN(none.nll()) && Double.isNaN(none.temperature()), none.toString());
    Assert.assertTrue(none.outliers().isEmpty());
  }

  /**
   * 局の結果を引いたネットワークそのもので測ると、外れ（区分ごとのセルの分類のうち |z| が 3 を越えるもの）は少ない。区分ごとの
   * セルの分類の数の 30 分の 1（少なくとも 3 つ）までに収まる。セルの中で一度も起きなかった分類も、起きないことが珍しくなければ
   * 外れにしない。
   */
  @Test(dataProvider = "variants")
  public void theDrawingNetworkHasFewOutliers(Variant variant) {
    OutcomeShape shape = new OutcomeShape(variant, GrpFixtures.plainFields(), 8);
    OutcomeNetwork q = GrpFixtures.outcomes(shape, 19, 0.5);
    OutcomeFixtures.Drawn drawn = OutcomeFixtures.drawn(q, 500, 1, 7 + variant.seats());
    NextHands next = NextHands.of(HandEnds.of(variant, RankUtility.TENHOU, drawn.games()), drawn.counted());
    OutcomeClasses.Result classes = OutcomeClasses.measure(next, h -> q, h -> true, () -> OutcomeWorkspace.of(shape));
    long rates = 0;
    for (OutcomeClasses.SeatCell cell : classes.seats()) {
      rates += cell.group() < 0 ? 0 : cell.rates().size();
    }
    for (OutcomeClasses.DealerCell cell : classes.dealers()) {
      rates += cell.group() < 0 ? 0 : cell.rates().size();
    }
    List<OutcomeClasses.Outlier> outliers = OutcomeCalibration.check(next, h -> true, q).outliers();
    long neverSeen = outliers.stream().filter(outlier -> outlier.rate().realized() == 0).count();
    Assert.assertTrue(
        outliers.size() <= Math.max(3, rates / 30),
        variant + ": " + outliers.size() + " outliers among " + rates + " rates, " + neverSeen
            + " of them never seen in their cell; the largest: " + outliers.subList(0, Math.min(3, outliers.size())));
  }

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /** 外れが区分ごとのセルだけで、|z| が 3 を越え、|z| の大きい順に並ぶことを確かめる。親から見た分類は点差の区間を持たない。 */
  private static void assertOutliers(List<OutcomeClasses.Outlier> outliers) {
    double previous = Double.POSITIVE_INFINITY;
    for (OutcomeClasses.Outlier outlier : outliers) {
      double z = Math.abs(outlier.rate().z());
      Assert.assertTrue(z > OutcomeCalibration.OUTLIER_Z && z <= previous, outlier.toString());
      Assert.assertTrue(outlier.group() >= 0, outlier.toString());
      if (outlier.view() == OutcomeClasses.View.DEALER) {
        Assert.assertEquals(outlier.lowGap(), 0, outlier.toString());
        Assert.assertEquals(outlier.highGap(), Integer.MAX_VALUE, outlier.toString());
      }
      previous = z;
    }
  }
}
