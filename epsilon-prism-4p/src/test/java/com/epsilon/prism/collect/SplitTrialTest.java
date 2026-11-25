package com.epsilon.prism.collect;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerCorrection;
import com.epsilon.prism.luck.LedgerRecording;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.noise.PointUnit;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.Row;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.function.ToIntFunction;
import org.testng.Assert;
import org.testng.annotations.Test;

public class SplitTrialTest {
  private static final ToIntFunction<PolicyRequest> FIRST = o -> 0;
  private static final ToIntFunction<PolicyRequest> LAST = o -> o.legalCount() - 1;

  private record Choice(ToIntFunction<PolicyRequest> select) implements PolicyService {
    @Override
    public int version() {
      return 1;
    }

    @Override
    public boolean inline() {
      return true;
    }

    @Override
    public void evaluateInline(PolicyRequest query) {
      query.sigma = new float[query.legalCount()];
      query.sigma[select.applyAsInt(query)] = 1;
    }
  }

  private static final class BatchedChoice implements PolicyService {
    private final Choice choice;
    int largest;

    BatchedChoice(Choice choice) {
      this.choice = choice;
    }

    @Override
    public int version() {
      return choice.version();
    }

    @Override
    public boolean inline() {
      return false;
    }

    @Override
    public int admission(Observation observation) {
      return 0;
    }

    @Override
    public CompletableFuture<Void> submit(
        List<PolicyRequest> batch, DecisionInferenceIngress ingress) {
      ingress.handoff(this).move();
      largest = Math.max(largest, batch.size());
      batch.forEach(choice::evaluateInline);
      return CompletableFuture.completedFuture(null);
    }
  }

  private static List<SplitTrial.Result> play(PolicyService a, PolicyService b, int workers) {
    var variant = Variant.TENHOU_4P;
    var cohort =
        new Collector.Cohort(
            variant,
            17,
            31,
            List.of(4L, 5L, 6L),
            DrawOrder.SEAT_STREAMS,
            Sampling.OFF,
            null,
            new UraSettings.Realized(0),
            null,
            TwinDiscard.TSUMOGIRI,
            null);
    try (var table = new TableLoop(new TableSettings(3, workers, 100_000, 1))) {
      return SplitTrial.play(
          table,
          cohort,
          new CollectorTest.HashedScorePolicy(),
          Phi.scoreSmooth(variant, RankUtility.TENHOU),
          0.07,
          a,
          b,
          null);
    }
  }

  @Test
  public void forcedDecisionsEnterTheSampledDenominator() {
    var variant = Variant.TENHOU_4P;
    var cohort =
        new Collector.Cohort(
            variant,
            17,
            31,
            List.of(4L, 5L),
            DrawOrder.SEAT_STREAMS,
            Sampling.OFF,
            null,
            new UraSettings.Realized(0),
            null,
            TwinDiscard.TSUMOGIRI,
            null);
    try (var table = new TableLoop(new TableSettings(2, 1, 100, 1))) {
      var phi = Phi.scoreSmooth(variant, RankUtility.TENHOU);
      var mu = new CollectorTest.HashedScorePolicy();
      var counts = Collector.play(table, cohort, mu, phi, null, null).counts();
      Assert.assertTrue(counts.forcedDecisions() > 0);
      var all =
          SplitTrial.play(table, cohort, mu, phi, 1, new Choice(FIRST), new Choice(FIRST), null)
              .getLast()
              .all();
      Assert.assertEquals(all.positions(), counts.modelDecisions() + counts.forcedDecisions());
    }
  }

  @Test
  public void terminalForksSubtractEachActionsOwnLedger() throws Exception {
    var variant = Variant.TENHOU_4P;
    var recording = new LedgerRecording(Set.of(LedgerGroup.IMPROVE), 32, 8, PointUnit.THOUSANDS);
    var sampling = TestSampling.grouped(0, .12, .12, .12, .12, 1, 2, false, 0, 0, 0, 0, 0, 1);
    var walls = List.of(13L);
    var cohort =
        new Collector.Cohort(
            variant,
            17,
            31,
            walls,
            DrawOrder.SEAT_STREAMS,
            sampling,
            recording,
            new UraSettings.Realized(0),
            null,
            TwinDiscard.TSUMOGIRI,
            null);
    var duel =
        new Collector.Cohort(
            variant,
            17,
            31,
            walls,
            DrawOrder.SEAT_STREAMS,
            Sampling.OFF,
            recording,
            cohort.ura(),
            null,
            cohort.twinDiscard(),
            null);
    var phi = Phi.scoreSmooth(variant, RankUtility.TENHOU);
    var mu = new CollectorTest.HashedScorePolicy();
    try (var table = new TableLoop(new TableSettings(2, 2, 100, 1));
        var saved = new CollectorTest.TemporaryCohort(recording)) {
      Collector.collect(table, cohort, mu, phi, saved.sink);
      var rows = saved.sealAndRead();
      rows.sort(Comparator.comparingLong(Row::gameId).thenComparingLong(Row::decisionId));
      var example = terminalPair(rows, variant);
      var row = example.row();
      double[][][] weights = new double[2][DecisionClass.COUNT][LedgerColumn.COUNT];
      double coefficient = .375;
      weights[(int) (row.gameId() & 1)][row.decisionClass().ordinal()][example.column()] =
          coefficient;
      var results =
          SplitTrial.play(
              table,
              duel,
              mu,
              phi,
              1,
              new Choice(r -> example.matches(r) ? row.played() : 0),
              new Choice(r -> example.matches(r) ? example.alternative() : 0),
              new LedgerCorrection.ByClass(weights));
      var diff = results.getLast().disagreements();
      Assert.assertEquals(diff.positions(), 1L);
      double termA = example.term(row.played()), termB = example.term(example.alternative());
      double expected =
          example.utility(row.played())
              - example.utility(example.alternative())
              - coefficient * (termA - termB);
      // 行に保存した帳簿は確率的に FP16 へ丸める。各項の誤差は隣り合う表現値の間隔以内。
      double tolerance = coefficient * (halfUlp(termA) + halfUlp(termB));
      Assert.assertEquals(diff.mean().doubleValue(), expected, tolerance);
    }
  }

  private static double halfUlp(double value) {
    return Math.scalb(1., Math.max(-14, Math.getExponent(Math.abs(value))) - 10);
  }

  private record TerminalPair(Row row, int alternative, int column, int seat, Variant variant) {
    boolean matches(PolicyRequest request) {
      if (request.wall() != row.gameId()) return false;
      var records = request.records();
      int start = records.start(request.record());
      return Arrays.equals(
          records.words(),
          start,
          start + records.longs(request.record()),
          row.observation(),
          0,
          row.observation().length);
    }

    double term(int action) {
      return row.ledger(action, column);
    }

    double utility(int action) {
      int[] scores = new int[variant.seats()];
      var codec = OutcomeCodec.of(variant);
      for (int relative = 0; relative < scores.length; relative++) {
        scores[(seat + relative) % scores.length] =
            codec.finalScore(row.outcomes(), action * OutcomeCodec.WORDS, relative);
      }
      return RankUtility.TENHOU.value(variant, Placement.rank(scores, seat));
    }
  }

  private static TerminalPair terminalPair(List<Row> rows, Variant variant) {
    var codec = OutcomeCodec.of(variant);
    var reader = new Observation(variant);
    for (Row row : rows) {
      int a = row.played();
      if (!terminal(codec, row, a)) continue;
      int seat = reader.bind(row.observation(), 0).observer();
      for (int b = 0; b < row.actions(); b++) {
        if (a == b || !terminal(codec, row, b)) continue;
        for (var column : LedgerColumn.values()) {
          var pair = new TerminalPair(row, b, column.ordinal(), seat, variant);
          double left = pair.term(a), right = pair.term(b);
          double margin = 8 * (halfUlp(left) + halfUlp(right));
          if (Math.min(Math.abs(left), Math.abs(right)) > margin
              && Math.abs(left - right) > margin
              && pair.utility(a) != pair.utility(b)) return pair;
        }
      }
    }
    throw new AssertionError("No two terminal branches with distinct nonzero ledger terms");
  }

  private static boolean terminal(OutcomeCodec codec, Row row, int action) {
    int at = action * OutcomeCodec.WORDS;
    return row.measured(action)
        && codec.outcome(row.outcomes(), at) != null
        && codec.nextRound(row.outcomes(), at) < 0;
  }

  @Test
  public void identicalActionsCountAsExactZeros() {
    var first = new Choice(FIRST);
    var overall = play(first, first, 2).getLast();
    Assert.assertTrue(overall.all().positions() > 0);
    Assert.assertEquals(overall.disagreements().positions(), 0L);
    Assert.assertEquals(overall.all().mean(), 0.0);
    Assert.assertEquals(overall.all().se(), 0.0);
  }

  @Test
  public void forksRepeatAcrossWorkerCountsAndReverseSign() {
    var first = new Choice(FIRST);
    var last = new Choice(LAST);
    var result = play(first, last, 1);
    var a = new BatchedChoice(first);
    var b = new BatchedChoice(last);
    Assert.assertEquals(play(a, b, 3), result);
    Assert.assertTrue(a.largest > 1);
    Assert.assertTrue(b.largest > 1);
    var overall = result.getLast();
    Assert.assertTrue(overall.disagreements().positions() > 0);
    Assert.assertTrue(overall.disagreements().positions() <= overall.all().positions());
    Assert.assertEquals(
        overall.all().mean(),
        overall.disagreements().mean()
            * overall.disagreements().positions()
            / overall.all().positions(),
        1e-14);
    Assert.assertEquals(play(last, first, 2).getLast().all().mean(), -overall.all().mean(), 1e-14);
  }
}
