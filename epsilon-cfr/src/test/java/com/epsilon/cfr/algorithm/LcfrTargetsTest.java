package com.epsilon.cfr.algorithm;

import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.ObservationPayload;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class LcfrTargetsTest {
  private static final int[] SEATS = {0, 1, 2, 3, 0, 1};

  @Test
  public void fullEnumerationMatchesCounterfactualValuesForEverySeatAndZeroPolicyAction() {
    Map<String, double[]> estimated = new HashMap<>();
    for (int terminal = 0; terminal < (1 << SEATS.length); terminal++) {
      List<CfrDecision> decisions = new ArrayList<>();
      double samplingReach = 1;
      int prefix = 0;
      for (int depth = 0; depth < SEATS.length; depth++) {
        double[] sigma = policy(depth, prefix, 1);
        double[] mu = RegretMatching.behavior(sigma, .2);
        int action = terminal >>> (SEATS.length - 1 - depth) & 1;
        decisions.add(
            new CfrDecision(
                depth,
                SEATS[depth],
                new int[] {0, 1},
                action,
                sigma,
                mu[action],
                ObservationPayload.copyOf(new byte[] {(byte) depth, (byte) prefix})));
        samplingReach *= mu[action];
        prefix = prefix * 2 + action;
      }
      double[] utility = new double[4];
      for (int seat = 0; seat < 4; seat++) utility[seat] = utility(terminal, seat);
      List<CfrTrainingRow> rows =
          LcfrTargets.generate(new CfrEpisode(terminal, decisions, utility), 1, 1);
      prefix = 0;
      for (int depth = 0; depth < SEATS.length; depth++) {
        CfrTrainingRow row = rows.get(depth);
        double[] total = estimated.computeIfAbsent(depth + ":" + prefix, key -> new double[2]);
        double[] target = row.regretTargets();
        double dot = 0;
        for (int action = 0; action < 2; action++) {
          total[action] += samplingReach * Math.exp(row.logRegretWeight()) * target[action];
          dot += row.targetPolicy()[action] * target[action];
        }
        Assert.assertEquals(dot, 0, 1e-13);
        prefix = prefix * 2 + row.selectedSlot();
      }
    }
    for (int depth = 0; depth < SEATS.length; depth++) {
      for (int prefix = 0; prefix < (1 << depth); prefix++) {
        int seat = SEATS[depth];
        double reachOthers = 1;
        int partial = 0;
        for (int prior = 0; prior < depth; prior++) {
          int action = prefix >>> (depth - 1 - prior) & 1;
          if (SEATS[prior] != seat) reachOthers *= policy(prior, partial, 1)[action];
          partial = partial * 2 + action;
        }
        double first = value(depth + 1, prefix * 2, seat, 1);
        double second = value(depth + 1, prefix * 2 + 1, seat, 1);
        double expected =
            policy(depth, prefix, 1)[0] * first + policy(depth, prefix, 1)[1] * second;
        double[] actual = estimated.get(depth + ":" + prefix);
        Assert.assertEquals(actual[0], reachOthers * (first - expected), 2e-12);
        Assert.assertEquals(actual[1], reachOthers * (second - expected), 2e-12);
      }
    }
  }

  @Test
  public void correctedAverageMatchesOwnReachWeightedLinearAverageAcrossIterations() {
    Map<String, double[]> observed = new HashMap<>();
    Map<String, double[]> expected = new HashMap<>();
    for (int iteration = 1; iteration <= 3; iteration++) {
      for (int terminal = 0; terminal < (1 << SEATS.length); terminal++) {
        var decisions = new ArrayList<CfrDecision>();
        int prefix = 0;
        double q = 1;
        for (int depth = 0; depth < SEATS.length; depth++) {
          double[] sigma = policy(depth, prefix, iteration);
          double[] mu = RegretMatching.behavior(sigma, .3);
          int action = terminal >>> (SEATS.length - 1 - depth) & 1;
          decisions.add(
              new CfrDecision(
                  depth,
                  SEATS[depth],
                  new int[] {0, 1},
                  action,
                  sigma,
                  mu[action],
                  ObservationPayload.copyOf(new byte[0])));
          q *= mu[action];
          prefix = prefix * 2 + action;
        }
        var rows =
            LcfrTargets.generate(new CfrEpisode(terminal, decisions, new double[4]), iteration, 1);
        prefix = 0;
        for (int depth = 0; depth < SEATS.length; depth++) {
          CfrTrainingRow row = rows.get(depth);
          double[] sum = observed.computeIfAbsent(depth + ":" + prefix, key -> new double[2]);
          for (int action = 0; action < 2; action++)
            sum[action] += q * Math.exp(row.logAverageWeight()) * row.targetPolicy()[action];
          prefix = prefix * 2 + row.selectedSlot();
        }
      }
      for (int depth = 0; depth < SEATS.length; depth++) {
        for (int prefix = 0; prefix < 1 << depth; prefix++) {
          double ownReach = 1;
          int partial = 0;
          for (int prior = 0; prior < depth; prior++) {
            int action = prefix >>> (depth - 1 - prior) & 1;
            if (SEATS[prior] == SEATS[depth]) ownReach *= policy(prior, partial, iteration)[action];
            partial = partial * 2 + action;
          }
          double[] sum = expected.computeIfAbsent(depth + ":" + prefix, key -> new double[2]);
          for (int action = 0; action < 2; action++)
            sum[action] += iteration * ownReach * policy(depth, prefix, iteration)[action];
        }
      }
    }
    expected.forEach(
        (key, target) -> {
          double[] actual = observed.get(key);
          double total = target[0] + target[1];
          if (total == 0) Assert.assertEquals(actual[0] + actual[1], 0, 0);
          else Assert.assertEquals(actual[0] / (actual[0] + actual[1]), target[0] / total, 1e-12);
        });
  }

  @Test
  public void explorationOfZeroTargetActionKeepsItsOwnRegretButZerosEarlierSuffix() {
    var episode =
        new CfrEpisode(
            7,
            List.of(
                new CfrDecision(
                    0,
                    0,
                    new int[] {1, 2},
                    0,
                    new double[] {.5, .5},
                    .5,
                    ObservationPayload.copyOf(new byte[0])),
                new CfrDecision(
                    1,
                    1,
                    new int[] {1, 2},
                    1,
                    new double[] {1, 0},
                    .1,
                    ObservationPayload.copyOf(new byte[0]))),
            new double[] {1, 2, 3, 4});
    var rows = LcfrTargets.generate(episode, 6, 3);
    Assert.assertEquals(rows.getFirst().logRegretWeight(), Double.NEGATIVE_INFINITY);
    Assert.assertTrue(Double.isFinite(rows.getLast().logRegretWeight()));
    Assert.assertEquals(rows.getLast().regretTargets(), new double[] {0, 2});
  }

  @Test
  public void regretMatchingKeepsSignedValuesAndSurvivesLargeFiniteMagnitudes() {
    Assert.assertEquals(
        RegretMatching.policy(new double[] {-3, 0, -1}), new double[] {1.0 / 3, 1.0 / 3, 1.0 / 3});
    Assert.assertEquals(
        RegretMatching.policy(new double[] {Double.MAX_VALUE, Double.MAX_VALUE, -1}),
        new double[] {.5, .5, 0});
    Assert.expectThrows(
        IllegalArgumentException.class, () -> RegretMatching.behavior(new double[] {1, 0}, 0));
  }

  private static double[] policy(int depth, int prefix, int iteration) {
    double first = ((depth * 3 + prefix * 5 + iteration * 2) % 7) / 6.0;
    return new double[] {first, 1 - first};
  }

  private static double value(int depth, int prefix, int seat, int iteration) {
    if (depth == SEATS.length) return utility(prefix, seat);
    double[] sigma = policy(depth, prefix, iteration);
    return sigma[0] * value(depth + 1, prefix * 2, seat, iteration)
        + sigma[1] * value(depth + 1, prefix * 2 + 1, seat, iteration);
  }

  private static double utility(int terminal, int seat) {
    return ((terminal * (seat * 2 + 3) + seat * 5) % 17 - 8) / 8.0;
  }
}
