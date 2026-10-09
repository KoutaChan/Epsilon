package com.epsilon.cfr.algorithm;

import com.epsilon.cfr.data.CfrBranchOutcome;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.ObservationPayload;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class LcfrBranchTargetsTest {
  private static final double[] POLICY = {.15, 0, .45, .4};
  private static final double[] BEHAVIOR = {.15, .2, .35, .3};
  private static final double[] PREFIX_BEHAVIOR = {.55, .45};
  private static final double[] SUFFIX_BEHAVIOR = {.6, .4};

  @Test
  public void enumeratedRonDeclineAndKyushuPreserveRegretNumeratorAndRegressionMass() {
    for (boolean kyushu : new boolean[] {false, true}) {
      verifyEnumeration(kyushu, new double[] {.3, .7}, new double[] {.25, .75});
      verifyEnumeration(kyushu, new double[] {0, 1}, new double[] {0, 1});
    }
  }

  private static void verifyEnumeration(
      boolean kyushu, double[] prefixPolicy, double[] suffixPolicy) {
    double mass = 0;
    double[] numerator = new double[POLICY.length];
    for (int prefix = 0; prefix < 2; prefix++) {
      for (int selected = 0; selected < POLICY.length; selected++) {
        for (Alternative alternative : alternatives(kyushu, selected)) {
          int alternatives = alternative.slot() < 0 ? 1 : 2;
          for (int suffix = 0; suffix < 2; suffix++) {
            for (int alternateSuffix = 0; alternateSuffix < alternatives; alternateSuffix++) {
              double probability =
                  PREFIX_BEHAVIOR[prefix]
                      * BEHAVIOR[selected]
                      * SUFFIX_BEHAVIOR[suffix]
                      * alternative.probability()
                      * (alternative.slot() < 0 ? 1 : SUFFIX_BEHAVIOR[alternateSuffix]);
              var decisions =
                  List.of(
                      new CfrDecision(
                          0,
                          1,
                          new int[] {0, 1},
                          prefix,
                          prefixPolicy,
                          PREFIX_BEHAVIOR[prefix],
                          ObservationPayload.copyOf(new byte[0])),
                      new CfrDecision(
                          1,
                          0,
                          new int[] {0, 1, 2, 3},
                          selected,
                          POLICY,
                          BEHAVIOR[selected],
                          ObservationPayload.copyOf(new byte[0])),
                      new CfrDecision(
                          2,
                          2,
                          new int[] {0, 1},
                          suffix,
                          suffixPolicy,
                          SUFFIX_BEHAVIOR[suffix],
                          ObservationPayload.copyOf(new byte[0])));
              double[] utility = {utility(prefix, selected, suffix), .1, -.2, .3};
              CfrBranchOutcome branch = null;
              if (!(kyushu && selected == 3)) {
                double gateMass = kyushu ? .7 : 1;
                double inclusion =
                    selected == 0 ? gateMass : BEHAVIOR[selected] * (kyushu ? gateMass / .55 : 1);
                branch =
                    alternative.slot() < 0
                        ? new CfrBranchOutcome(1, 1, -1, 0, 0, Double.NEGATIVE_INFINITY)
                        : new CfrBranchOutcome(
                            1,
                            inclusion,
                            alternative.slot(),
                            alternative.slot() == 0
                                ? gateMass
                                : BEHAVIOR[alternative.slot()] * gateMass / .55,
                            utility(prefix, alternative.slot(), alternateSuffix),
                            Math.log(suffixPolicy[alternateSuffix])
                                - Math.log(SUFFIX_BEHAVIOR[alternateSuffix]));
              }
              var ordinary = LcfrTargets.generate(new CfrEpisode(0, decisions, utility), 6, 3);
              var branched =
                  LcfrTargets.generate(
                      new CfrEpisode(
                          0, decisions, utility, branch == null ? List.of() : List.of(branch)),
                      6,
                      3);
              for (int index = 0; index < ordinary.size(); index++) {
                Assert.assertEquals(
                    branched.get(index).logAverageWeight(), ordinary.get(index).logAverageWeight());
                Assert.assertEquals(
                    branched.get(index).targetPolicy(), ordinary.get(index).targetPolicy());
                if (index != 1)
                  Assert.assertTrue(branched.get(index).contentEquals(ordinary.get(index)));
              }
              CfrTrainingRow row = branched.get(1);
              double weight = probability * Math.exp(row.logRegretWeight());
              mass += weight;
              double[] target = row.regretTargets();
              double dot = 0;
              for (int action = 0; action < POLICY.length; action++) {
                numerator[action] += weight * target[action];
                dot += POLICY[action] * target[action];
              }
              Assert.assertEquals(dot, 0, 1e-12);
            }
          }
        }
      }
    }
    double[] actionValue = new double[POLICY.length];
    double value = 0;
    for (int action = 0; action < POLICY.length; action++) {
      for (int prefix = 0; prefix < 2; prefix++) {
        for (int suffix = 0; suffix < 2; suffix++) {
          actionValue[action] +=
              prefixPolicy[prefix] * suffixPolicy[suffix] * utility(prefix, action, suffix);
        }
      }
      value += POLICY[action] * actionValue[action];
    }
    Assert.assertEquals(mass, 2d * POLICY.length, 1e-12);
    for (int action = 0; action < POLICY.length; action++) {
      Assert.assertEquals(numerator[action], 2 * (actionValue[action] - value), 1e-12);
    }
  }

  @Test
  public void logSpaceCollapseKeepsFiniteTargetsForExtremeSuffixWeights() {
    var decision =
        new CfrDecision(
            0,
            0,
            new int[] {0, 1},
            0,
            new double[] {.25, .75},
            .3,
            ObservationPayload.copyOf(new byte[0]));
    var episode =
        new CfrEpisode(
            0,
            List.of(decision),
            new double[] {1, 0, 0, 0},
            List.of(new CfrBranchOutcome(0, 1, 1, .7, -1, 10_000)));
    CfrTrainingRow row = LcfrTargets.generate(episode, 1, 1).getFirst();
    Assert.assertEquals(row.logRegretWeight(), 10_000 - Math.log(.7), 1e-12);
    Assert.assertEquals(row.regretTargets(), new double[] {.75, -.25});
  }

  private static List<Alternative> alternatives(boolean kyushu, int selected) {
    if (selected == 0) {
      return kyushu
          ? List.of(new Alternative(1, .2 / .55), new Alternative(2, .35 / .55))
          : List.of(new Alternative(-1, 1));
    }
    return List.of(new Alternative(kyushu && selected == 3 ? -1 : 0, 1));
  }

  private static double utility(int prefix, int action, int suffix) {
    return ((prefix * 11 + action * 3 + suffix * 7) % 13 - 6) / 8d;
  }

  private record Alternative(int slot, double probability) {}
}
