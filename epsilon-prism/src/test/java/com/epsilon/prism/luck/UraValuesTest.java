package com.epsilon.prism.luck;

import com.epsilon.agent.settings.RecordFile;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.noise.UraSettings;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 収集で使う局末の値（{@link UraValues}）を確かめる。実際の値で学ぶときは、監査の割合に依らず、どの局末でも期待値を使わない式（続く局は
 * 次局の卓の Φ、終局は Φ を呼ばずに最終順位の効用）とビットまで同じ値を書く。Φ を呼ぶのは、続く局の局末に 1 回ずつだけ。監査に選ぶかは、
 * コホートの種・半荘・局の序数から引く乱数だけで決まり、局末の中身（裏ドラ・和了の形）には依らない。報告の要約は有限でない値を書かず、
 * 値が全て同じ区分には z も分散の減りも書かない。
 */
public class UraValuesTest {

  private static final long SEED = 0x5eed_0a0aL;

  private static final int HAND_ENDS = 3_000;

  /** 自己対局で数える局末。 */
  private enum Count {
    TERMINAL,
    CONTINUING,
    RIICHI_WINS,
    TERMINAL_RIICHI_WINS,
    HALF_AVERAGED
  }

  @Test
  public void fourPlayerValuesKeepTheFormerFormula() {
    play(Variant.TENHOU_4P, 61L);
  }

  @Test
  public void threePlayerValuesKeepTheFormerFormula() {
    play(Variant.TENHOU_3P, 67L);
  }

  /**
   * 自己対局の局末ごとに、監査の割合が 0・0.05・0.5・1 の場合と期待値で学ぶ場合について、書く値と監査に選ぶかを、期待値を使わない式と
   * 乱数から求めた値と比べる。全ての局末を同じ半荘・局の序数として渡すと、割合が乱数のすぐ上なら全ての立直の和了を選び、割合が乱数
   * ちょうどなら 1 つも選ばない（局末の中身に依らない）。
   */
  private static void play(Variant variant, long seed) {
    Phi reference = UraExpectationPlayTest.phi(variant);
    Phi phiNone = UraExpectationPlayTest.phi(variant);
    Phi phiAll = UraExpectationPlayTest.phi(variant);
    Phi phiExpected = UraExpectationPlayTest.phi(variant);
    UraValues none = new UraValues(new UraSettings.Realized(0), phiNone, SEED);
    UraValues sampled = new UraValues(new UraSettings.Realized(0.05), UraExpectationPlayTest.phi(variant), SEED);
    UraValues half = new UraValues(new UraSettings.Realized(0.5), UraExpectationPlayTest.phi(variant), SEED);
    UraValues all = new UraValues(new UraSettings.Realized(1), phiAll, SEED);
    UraValues expected = new UraValues(new UraSettings.Expected(), phiExpected, SEED);
    long fixedGame = 9;
    int fixedHand = 3;
    double coin = UraValues.coin(SEED, fixedGame, fixedHand);
    UraValues above =
        new UraValues(new UraSettings.Realized(Math.nextUp(coin)), UraExpectationPlayTest.phi(variant), SEED);
    UraValues at = new UraValues(new UraSettings.Realized(coin), UraExpectationPlayTest.phi(variant), SEED);
    int seats = variant.seats();
    double[] former = new double[seats], out = new double[seats];
    long[] game = {0};
    int[] hand = {0};
    Map<Count, Long> counts = new EnumMap<>(Count.class);
    for (Count count : Count.values()) counts.put(count, 0L);
    new UraGames(variant)
        .play(
            seed,
            HAND_ENDS,
            (engine, kan) -> {
              FullView full = engine.full();
              RoundEnd end = engine.roundEnd();
              TableState start = engine.handStart();
              TableState next = end.next();
              boolean over = next.gameOver();
              formerValues(reference, next, former);
              counts.merge(over ? Count.TERMINAL : Count.CONTINUING, 1L, Long::sum);
              String where = variant + " game " + game[0] + " hand " + hand[0] + " " + end.outcome();
              UraExpectation values = none.handEnd(full, end, start, null, game[0], hand[0], out);
              boolean riichiWin = values.riichiWin();
              Assert.assertFalse(values.averaged(), where);
              assertBits(out, former, "sample 0 " + where);
              sampled.handEnd(full, end, start, null, game[0], hand[0], out);
              assertBits(out, former, "sample 0.05 " + where);
              UraExpectation halved = half.handEnd(full, end, start, null, game[0], hand[0], out);
              assertBits(out, former, "sample 0.5 " + where);
              boolean chosen = UraValues.coin(SEED, game[0], hand[0]) < 0.5;
              Assert.assertEquals(halved.averaged(), riichiWin && chosen, where);
              if (halved.averaged()) counts.merge(Count.HALF_AVERAGED, 1L, Long::sum);
              UraExpectation averaged = all.handEnd(full, end, start, null, game[0], hand[0], out);
              assertBits(out, former, "sample 1 " + where);
              Assert.assertEquals(averaged.averaged(), riichiWin, where);
              UraExpectation learned = expected.handEnd(full, end, start, null, game[0], hand[0], out);
              Assert.assertEquals(learned.averaged(), riichiWin, where);
              for (int seat = 0; seat < seats; seat++) {
                double value = riichiWin ? learned.expected(seat) : former[seat];
                Assert.assertEquals(
                    Double.doubleToRawLongBits(out[seat]), Double.doubleToRawLongBits(value), where + " seat " + seat);
              }
              Assert.assertEquals(
                  above.handEnd(full, end, start, null, fixedGame, fixedHand, out).averaged(), riichiWin, where);
              Assert.assertFalse(at.handEnd(full, end, start, null, fixedGame, fixedHand, out).averaged(), where);
              if (riichiWin) {
                counts.merge(Count.RIICHI_WINS, 1L, Long::sum);
                if (over) counts.merge(Count.TERMINAL_RIICHI_WINS, 1L, Long::sum);
              }
              if (over) {
                game[0]++;
                hand[0] = 0;
              } else {
                hand[0]++;
              }
            });
    String tally = variant + " " + counts;
    long riichiWins = counts.get(Count.RIICHI_WINS), halfAveraged = counts.get(Count.HALF_AVERAGED);
    Assert.assertTrue(counts.get(Count.TERMINAL) > 10 && counts.get(Count.TERMINAL_RIICHI_WINS) > 0, tally);
    Assert.assertTrue(riichiWins > 100 && halfAveraged > 0 && halfAveraged < riichiWins, tally);
    // 実際の値は、続く局の局末に Φ を 1 回ずつ呼ぶだけ（終局の卓には呼ばない）
    Assert.assertEquals(phiNone.stats().requests(), (long) counts.get(Count.CONTINUING), tally);
    Assert.assertTrue(phiAll.stats().requests() > phiNone.stats().requests(), tally);
    Assert.assertEquals(phiExpected.stats().requests(), phiAll.stats().requests(), tally);
    UraAudit.Report off = none.report(), on = all.report();
    Assert.assertEquals(off.handEnds(), HAND_ENDS);
    Assert.assertEquals(off.riichiWins(), riichiWins);
    Assert.assertEquals(off.averaged(), 0);
    Assert.assertEquals(on.handEnds(), HAND_ENDS);
    Assert.assertEquals(on.riichiWins(), riichiWins);
    Assert.assertEquals(on.averaged(), riichiWins);
    Assert.assertEquals(half.report().averaged(), halfAveraged);
    Assert.assertEquals(above.report().averaged(), riichiWins);
    Assert.assertEquals(at.report().averaged(), 0);
  }

  /** 期待値を使わない局末の値。終局は Φ を呼ばずに最終順位の効用、続く局は次局の卓の Φ。 */
  private static void formerValues(Phi phi, TableState next, double[] out) {
    if (next.gameOver()) {
      for (int seat = 0; seat < out.length; seat++) out[seat] = phi.utility().value(phi.variant(), next.rank(seat));
    } else {
      phi.values(null, next, out);
    }
  }

  private static void assertBits(double[] actual, double[] expected, String where) {
    for (int seat = 0; seat < expected.length; seat++)
      Assert.assertEquals(
          Double.doubleToRawLongBits(actual[seat]), Double.doubleToRawLongBits(expected[seat]), where + " seat " + seat);
  }

  /**
   * 監査の乱数は unit(derive(コホートの種, URA_AUDIT, 半荘, 局の序数)) で、同じ座標でほかのドメインから作った値と重ならず、一様に散る
   * （割合 0.05 で選ぶ局は 5% 前後）。
   */
  @Test
  public void theAuditCoinReadsOnlyTheSeedTheGameAndTheHand() {
    SplittableRandom random = new SplittableRandom(71);
    for (int trial = 0; trial < 1_000; trial++) {
      long seed = random.nextLong(), game = random.nextLong(1 << 20);
      int hand = random.nextInt(40);
      long bits = CrnSeeds.derive(seed, CrnSeeds.Domain.URA_AUDIT, game, hand);
      Assert.assertEquals(UraValues.coin(seed, game, hand), CrnSeeds.unit(bits));
      for (CrnSeeds.Domain domain : CrnSeeds.Domain.values())
        if (domain != CrnSeeds.Domain.URA_AUDIT)
          Assert.assertNotEquals(CrnSeeds.derive(seed, domain, game, hand), bits, domain.toString());
    }
    UraSettings sample = new UraSettings.Realized(0.05);
    int games = 20_000, hands = 10;
    long chosen = 0;
    for (long game = 0; game < games; game++)
      for (int hand = 0; hand < hands; hand++) if (sample.averages(UraValues.coin(SEED, game, hand))) chosen++;
    double n = (double) games * hands, share = chosen / n;
    Assert.assertEquals(share, 0.05, 5 * Math.sqrt(0.05 * 0.95 / n), "share " + share);
    Assert.assertNotEquals(UraValues.coin(SEED, 1, 2), UraValues.coin(SEED, 2, 1));
    Assert.assertNotEquals(UraValues.coin(SEED, 1, 2), UraValues.coin(SEED + 1, 1, 2));
  }

  /**
   * 収集の段の報告に載せる要約を確かめる。要約は、局末の数・立直の和了の数・期待値を求めた数・基本点の組の数を 8 つのビンで数えた度数・
   * 期待値 1 回あたりの µs と、行のある区分ごとの行の数・z・分散の減りを持つ。z と分散の減りは、有限なら書き、有限でなければ書かない。JSON から
   * 読み戻すと同じ要約になる。何も数えていない監査の要約は、区分が空で数は全て 0。
   */
  @Test
  public void theSummaryLeavesOutValuesThatAreNotFinite() {
    UraAudit.Summary nothing = new UraAudit().report().summary();
    Assert.assertEquals(nothing.handEnds(), 0);
    Assert.assertEquals(nothing.averaged(), 0);
    Assert.assertEquals(nothing.groups().size(), UraAudit.GROUP_BINS);
    Assert.assertTrue(nothing.groups().stream().allMatch(bin -> bin == 0));
    Assert.assertEquals(nothing.microsPerAveraged(), 0.0);
    Assert.assertTrue(nothing.cells().isEmpty());
    Assert.assertEquals(RecordFile.read(UraAudit.Summary.class, RecordFile.json(nothing), "test").value(), nothing);

    // 行が 1 つの区分（z も分散の減りも有限でない）と、30 の束の 60 行の区分（どちらも有限）
    UraAudit.Cell single = UraAudit.Cell.of(DecisionClass.END, false, 1, 1, 0.25, 0, 0.0625, 0.01, 0);
    UraAudit.Cell sixty = UraAudit.Cell.of(DecisionClass.KAN, true, 60, 30, 0.5, 0.1, 0.75, 0.2, 1);
    long[] groups = {0, 3, 1, 0, 0, 0, 0, 0};
    UraAudit.Report report = new UraAudit.Report(10, 5, 4, groups, 6_000, List.of(single, sixty));
    UraAudit.Summary summary = report.summary();
    Assert.assertEquals(summary.handEnds(), 10);
    Assert.assertEquals(summary.riichiWins(), 5);
    Assert.assertEquals(summary.averaged(), 4);
    Assert.assertEquals(summary.groups(), List.of(0L, 3L, 1L, 0L, 0L, 0L, 0L, 0L));
    Assert.assertEquals(summary.microsPerAveraged(), 1.5);
    Assert.assertEquals(
        summary.cells(),
        List.of(
            new UraAudit.CellSummary(DecisionClass.END, false, 1, null, null),
            new UraAudit.CellSummary(DecisionClass.KAN, true, 60, sixty.z(), sixty.reduction())));
    JsonObject json = JsonParser.parseString(RecordFile.json(summary)).getAsJsonObject();
    Assert.assertEquals(json.getAsJsonArray("groups").size(), UraAudit.GROUP_BINS);
    JsonArray cells = json.getAsJsonArray("cells");
    JsonObject first = cells.get(0).getAsJsonObject(), second = cells.get(1).getAsJsonObject();
    Assert.assertFalse(first.has("z") || first.has("reduction"), first.toString());
    Assert.assertEquals(second.get("z").getAsDouble(), sixty.z());
    Assert.assertEquals(second.get("reduction").getAsDouble(), sixty.reduction());
    Assert.assertEquals(RecordFile.read(UraAudit.Summary.class, RecordFile.json(summary), "test").value(), summary);
  }

  /**
   * 値が全て同じ区分（行が 1 つ、同じ局の同じ席の行だけ、または同じ値の局がいくつか）では、d も実際の値も散らないので、z と分散の
   * 減りは有限でなく、要約に書かない。行に足す値を 2^−32 の固定小数点に丸めても、丸めで残る誤差を分散と取り違えてはならない。局で束ねた
   * 分散も、束が 2 つ以上あってちょうど 0 になる。
   */
  @Test
  public void equalValuesHaveNoSpread() {
    UraScenes.Played played = UraScenes.plainRon();
    Engine engine = played.engine();
    UraExpectation values = new UraExpectation(UraExpectationPlayTest.phi(UraScenes.VARIANT));
    Assert.assertTrue(values.evaluate(engine.full(), engine.roundEnd(), played.start(), null, true));
    int winner = 2;
    Assert.assertNotEquals(values.actual(winner), values.expected(winner));
    Assert.assertTrue(values.variance(winner) > 0);
    List<UraAudit.Row> rows = UraAuditTest.rows(DecisionClass.END, winner);
    rows.addAll(UraAuditTest.rows(DecisionClass.DISCARD_MID, winner, winner, winner));
    UraAudit once = new UraAudit();
    once.handEnd(values);
    once.rows(values, 0, 0, rows);
    UraAudit repeated = new UraAudit();
    for (long game : UraAuditTest.games(3)) repeated.rows(values, game, 0, rows);
    List<String> spread = new ArrayList<>();
    for (UraAudit audit : List.of(once, repeated)) {
      for (UraAudit.Cell cell : audit.report().cells()) {
        String where = cell.decisionClass() + " rows " + cell.rows() + " clusters " + cell.clusters();
        if (!Double.isNaN(cell.z())) spread.add(where + " z " + cell.z());
        if (!Double.isNaN(cell.reduction()))
          spread.add(
              where + " reduction " + cell.reduction() + " (label variance " + cell.labelVariance()
                  + ", standard error " + cell.standardError() + ")");
      }
    }
    Assert.assertEquals(spread, List.of(), String.join("; ", spread));
    for (UraAudit.Cell cell : repeated.report().cells()) {
      Assert.assertEquals(cell.clusters(), 3, cell.toString());
      Assert.assertEquals(cell.standardError(), 0.0, cell.toString());
    }
  }

  /**
   * 同じ局の主経路・写し・分岐の局末は、同じ裏ドラで d が決まるので、1 つの束に数える。束が互いに違う 4 つの局のそれぞれで、3 本の経路の
   * 局末が同じ席の行を 1 つずつ足すと、束は 4 つで、標準誤差は局ごとの d の和から求めた √(G/(G−1)·Σ_g (D_g − m_g·d̄)² / N²) になる。
   * 12 の行を独立とみなした値より大きい。
   */
  @Test
  public void pathsOfOneHandShareACluster() {
    UraScenes.Played played = UraScenes.plainRon();
    Engine engine = played.engine();
    UraValues values = new UraValues(new UraSettings.Expected(), UraExpectationPlayTest.phi(UraScenes.VARIANT), SEED);
    int hands = UraScenes.VARIANT.seats(), paths = 3, n = hands * paths;
    double[] out = new double[hands], difference = new double[hands];
    long[] games = UraAuditTest.games(hands);
    for (int hand = 0; hand < hands; hand++) {
      for (int path = 0; path < paths; path++) {
        UraExpectation ura =
            values.handEnd(engine.full(), engine.roundEnd(), played.start(), null, games[hand], 0, out);
        Assert.assertTrue(ura.averaged());
        // 局ごとに違う席の行を足すので、局ごとに d が違う
        values.rows(ura, games[hand], 0, UraAuditTest.rows(DecisionClass.DISCARD_MID, hand));
        difference[hand] = ura.actual(hand) - ura.expected(hand);
      }
    }
    double mean = 0, clustered = 0, independent = 0;
    for (double d : difference) mean += d / hands;
    for (double d : difference) {
      clustered += paths * (d - mean) * paths * (d - mean);
      independent += paths * (d - mean) * (d - mean);
    }
    double error = Math.sqrt(hands / (hands - 1.0) * clustered / ((double) n * n));
    double rowError = Math.sqrt(independent / ((double) n * (n - 1)));
    List<UraAudit.Cell> cells = values.report().cells();
    Assert.assertEquals(cells.size(), 1);
    UraAudit.Cell cell = cells.get(0);
    String where = cell + " row error " + rowError;
    Assert.assertEquals(cell.rows(), n, where);
    Assert.assertEquals(cell.clusters(), hands, where);
    Assert.assertEquals(cell.meanDifference(), mean, 1e-9, where);
    Assert.assertEquals(cell.standardError(), error, 1e-9, where);
    Assert.assertTrue(error > 1.5 * rowError, where);
  }
}
