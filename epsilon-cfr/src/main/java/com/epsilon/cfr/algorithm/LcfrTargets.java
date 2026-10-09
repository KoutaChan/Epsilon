package com.epsilon.cfr.algorithm;

import com.epsilon.cfr.data.CfrBranchOutcome;
import com.epsilon.cfr.data.CfrDecision;
import com.epsilon.cfr.data.CfrEpisode;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import java.util.HashMap;
import java.util.List;

/** 行動価値Qを使わず、全4席を対象とするアウトカムサンプリングLCFRの教師値を求める。牌山などの偶然事象は対局エンジンに従って抽選され、その確率は重要度比で相殺される。 */
public final class LcfrTargets {
  private LcfrTargets() {}

  public static List<CfrTrainingRow> generate(
      CfrEpisode episode, long iteration, long trajectoriesInIteration) {
    if (iteration <= 0 || trajectoriesInIteration <= 0)
      throw new IllegalArgumentException("Invalid LCFR iteration");
    int count = episode.decisions().size();
    double[] prefixRegret = new double[count];
    double[] prefixAverage = new double[count];
    double[] logSigma = new double[4];
    double[] logUniform = new double[4];
    int[] sigmaZeros = new int[4];
    double logMu = 0;
    for (int k = 0; k < count; k++) {
      CfrDecision decision = episode.decisions().get(k);
      int seat = decision.seat();
      double otherSigma = 0;
      double otherUniform = 0;
      boolean otherZero = false;
      for (int j = 0; j < 4; j++) {
        if (j == seat) continue;
        otherSigma += logSigma[j];
        otherUniform += logUniform[j];
        otherZero |= sigmaZeros[j] != 0;
      }
      prefixRegret[k] = otherZero ? Double.NEGATIVE_INFINITY : otherSigma - logMu;
      prefixAverage[k] =
          sigmaZeros[seat] != 0 ? Double.NEGATIVE_INFINITY : logSigma[seat] + otherUniform - logMu;
      double probability = decision.selectedTargetProbability();
      if (probability == 0) sigmaZeros[seat]++;
      else logSigma[seat] += Math.log(probability);
      logUniform[seat] -= Math.log(decision.actionCount());
      logMu += Math.log(decision.behaviorProbability());
    }
    double iterationWeight = Math.log(iteration) - Math.log(trajectoriesInIteration);
    double suffix = 0;
    boolean suffixZero = false;
    var branches = new HashMap<Long, CfrBranchOutcome>();
    for (CfrBranchOutcome branch : episode.branchOutcomes())
      branches.put(branch.decisionId(), branch);
    CfrTrainingRow[] rows = new CfrTrainingRow[count];
    for (int k = count - 1; k >= 0; k--) {
      CfrDecision decision = episode.decisions().get(k);
      double logRegret =
          suffixZero
              ? Double.NEGATIVE_INFINITY
              : iterationWeight
                  + prefixRegret[k]
                  + suffix
                  - Math.log(decision.behaviorProbability());
      double logAverage = iterationWeight + prefixAverage[k];
      double[] regretTargets = null;
      CfrBranchOutcome branch = branches.get(decision.decisionId());
      if (branch != null) {
        double commonWeight = iterationWeight + prefixRegret[k];
        double mainWeight =
            suffixZero
                ? Double.NEGATIVE_INFINITY
                : commonWeight + suffix - Math.log(branch.mainInclusionProbability());
        double alternativeWeight =
            branch.alternativeSlot() < 0
                ? Double.NEGATIVE_INFINITY
                : commonWeight
                    + branch.alternativeLogSuffixWeight()
                    - Math.log(branch.alternativeInclusionProbability());
        logRegret = logAdd(mainWeight, alternativeWeight);
        regretTargets = new double[decision.actionCount()];
        if (logRegret != Double.NEGATIVE_INFINITY) {
          addTarget(
              regretTargets,
              decision.selectedSlot(),
              decision.targetPolicy(),
              episode.utility(decision.seat()),
              Math.exp(mainWeight - logRegret));
          if (branch.alternativeSlot() >= 0)
            addTarget(
                regretTargets,
                branch.alternativeSlot(),
                decision.targetPolicy(),
                branch.alternativeUtility(),
                Math.exp(alternativeWeight - logRegret));
        }
      }
      rows[k] =
          new CfrTrainingRow(
              new CfrSampleId(
                  iteration, episode.episodeId(), decision.decisionId(), decision.seat()),
              decision.legalActionIds(),
              decision.selectedSlot(),
              decision.targetPolicy(),
              episode.utility(decision.seat()),
              logRegret,
              logAverage,
              decision.observation(),
              regretTargets);
      double probability = decision.selectedTargetProbability();
      if (probability == 0) suffixZero = true;
      else suffix += Math.log(probability) - Math.log(decision.behaviorProbability());
    }
    return List.of(rows);
  }

  /** 一部の行動を分岐評価した結果を、重み付き回帰への寄与の期待値を変えずに合算する。 */
  private static void addTarget(
      double[] targets, int selected, double[] policy, double utility, double fraction) {
    double contribution = fraction * utility;
    double expected = contribution * policy[selected];
    for (int action = 0; action < targets.length; action++) targets[action] -= expected;
    targets[selected] += contribution;
  }

  private static double logAdd(double first, double second) {
    if (first == Double.NEGATIVE_INFINITY) return second;
    if (second == Double.NEGATIVE_INFINITY) return first;
    double maximum = Math.max(first, second);
    return maximum + Math.log1p(Math.exp(Math.min(first, second) - maximum));
  }
}
