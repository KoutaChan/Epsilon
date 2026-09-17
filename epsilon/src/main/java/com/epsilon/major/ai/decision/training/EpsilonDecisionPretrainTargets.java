package com.epsilon.major.ai.decision.training;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.major.ai.decision.EpsilonUtilityTargets;
import com.epsilon.major.ai.decision.data.EpsilonDecisionDataException;
import com.epsilon.major.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.major.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.grp.EpsilonGrpTrainingSession;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * 牌譜の各判断に、スカラー効用を予測する価値関数の教師値を付与する。
 *
 * <p>同一ゲームの局境界を一意化して重みを固定したGRPの4x4 事前予測を一度だけ推論し、方策から分離された価値コンテキストへ書く。非終端局の Decision
 * 価値には直後の局境界のGRP予測の期待効用を使い、最終境界またはGRP無効時だけ終局効用へ戻す。
 */
public final class EpsilonDecisionPretrainTargets {

  private EpsilonDecisionPretrainTargets() {}

  public static List<EpsilonDecisionSample> prepare(
      List<EpsilonDecisionSample> source, EpsilonGrpTrainingSession grpSession) {
    return prepare(source, grpSession, EpsilonUtilityTargets.configuredTrainingProfile());
  }

  public static List<EpsilonDecisionSample> prepare(
      List<EpsilonDecisionSample> source,
      EpsilonGrpTrainingSession grpSession,
      EpsilonUtilityProfile utilityProfile) {
    MarginalPredictor predictor =
        grpSession != null && grpSession.enabled()
            ? grpSession.inference()::predictMarginalProbabilities
            : null;
    return prepare(source, predictor, utilityProfile);
  }

  static List<EpsilonDecisionSample> prepare(
      List<EpsilonDecisionSample> source, MarginalPredictor predictor) {
    return prepare(source, predictor, EpsilonUtilityTargets.configuredTrainingProfile());
  }

  static List<EpsilonDecisionSample> prepare(
      List<EpsilonDecisionSample> source,
      MarginalPredictor predictor,
      EpsilonUtilityProfile utilityProfile) {
    if (source.isEmpty()) {
      return List.of();
    }

    LinkedHashMap<Long, GameRows> games = new LinkedHashMap<>();
    for (int sourceIndex = 0; sourceIndex < source.size(); sourceIndex++) {
      EpsilonDecisionSample sample = source.get(sourceIndex);
      games.computeIfAbsent(sample.gameId(), GameRows::new).add(sourceIndex, sample);
    }

    ArrayList<PredictionRef> predictionRefs = new ArrayList<>();
    ArrayList<float[]> sequences = new ArrayList<>();
    for (GameRows game : games.values()) {
      game.finishBoundaries();
      for (Map.Entry<Integer, float[]> entry : game.predictionSequences().entrySet()) {
        predictionRefs.add(new PredictionRef(game, entry.getKey()));
        sequences.add(entry.getValue());
      }
    }

    List<float[]> priors = predictGrpPriors(sequences, predictor);
    for (int index = 0; index < predictionRefs.size(); index++) {
      PredictionRef ref = predictionRefs.get(index);
      ref.game().setPrior(ref.steps(), priors.get(index));
    }

    EpsilonDecisionSample[] prepared = new EpsilonDecisionSample[source.size()];
    for (GameRows game : games.values()) {
      game.materialize(prepared, predictor != null, utilityProfile);
    }
    return List.of(prepared);
  }

  private static List<float[]> predictGrpPriors(
      List<float[]> sequences, MarginalPredictor predictor) {
    if (sequences.isEmpty()) {
      return List.of();
    }
    if (predictor != null) {
      List<float[]> predictions = predictor.predict(sequences);
      if (predictions == null || predictions.size() != sequences.size()) {
        throw new EpsilonDecisionDataException(
            "GRP pretrain prediction count mismatch: expected="
                + sequences.size()
                + " actual="
                + (predictions == null ? -1 : predictions.size()));
      }
      return predictions;
    }
    ArrayList<float[]> uniform = new ArrayList<>(sequences.size());
    for (int index = 0; index < sequences.size(); index++) {
      float[] prior = new float[EpsilonGrpRanks.MATRIX_SIZE];
      Arrays.fill(prior, 1.0f / EpsilonGrpRanks.RANK_COUNT);
      uniform.add(prior);
    }
    return List.copyOf(uniform);
  }

  private static final class GameRows {
    private final long gameId;
    private final ArrayList<IndexedSample> rows = new ArrayList<>();
    private final TreeMap<Integer, float[]> sequenceByStep = new TreeMap<>();
    private final TreeMap<Integer, float[]> predictionSequenceByStep = new TreeMap<>();
    private final HashMap<Integer, float[]> priorByStep = new HashMap<>();
    private List<float[]> boundarySequences = List.of();
    private final HashMap<Integer, Integer> boundaryIndexByStep = new HashMap<>();
    private DecisionBoundaryContext[] boundaryContexts;
    private int finalRanksCode = -1;

    private GameRows(long gameId) {
      this.gameId = gameId;
    }

    private void add(int sourceIndex, EpsilonDecisionSample sample) {
      float[] grpFeatureSequence = sample.grpFeatureSequenceView();
      int steps = EpsilonGrpFeature.steps(grpFeatureSequence);
      if (steps <= 0) {
        throw new EpsilonDecisionDataException(
            "Decision pretrain sample has no GRP boundary: gameId=" + gameId);
      }
      float[] existing = sequenceByStep.putIfAbsent(steps, grpFeatureSequence);
      if (existing != null && !rawEquals(existing, grpFeatureSequence)) {
        throw new EpsilonDecisionDataException(
            "Decision pretrain game has conflicting GRP prefixes at step "
                + steps
                + ": gameId="
                + gameId);
      }
      EpsilonGrpRanks.requireCode(sample.grpFinalRanksCode());
      if (finalRanksCode != -1 && finalRanksCode != sample.grpFinalRanksCode()) {
        throw new EpsilonDecisionDataException(
            "Decision pretrain game has conflicting final-ranks codes: gameId=" + gameId);
      }
      finalRanksCode = sample.grpFinalRanksCode();
      rows.add(new IndexedSample(sourceIndex, steps, sample));
    }

    private void finishBoundaries() {
      if (finalRanksCode == -1) {
        throw new EpsilonDecisionDataException(
            "Decision pretrain game has no final-ranks code: gameId=" + gameId);
      }
      ArrayList<float[]> ordered = new ArrayList<>(sequenceByStep.size());
      ArrayList<Map.Entry<Integer, float[]>> observed = new ArrayList<>(sequenceByStep.entrySet());
      int boundaryIndex = 0;
      for (Map.Entry<Integer, float[]> entry : observed) {
        boundaryIndexByStep.put(entry.getKey(), boundaryIndex++);
        ordered.add(entry.getValue());
        predictionSequenceByStep.put(entry.getKey(), entry.getValue());
      }
      for (int index = 0; index + 1 < observed.size(); index++) {
        Map.Entry<Integer, float[]> current = observed.get(index);
        Map.Entry<Integer, float[]> later = observed.get(index + 1);
        requireNested(current, later);
        int nextSteps = current.getKey() + 1;
        if (later.getKey() > nextSteps) {
          predictionSequenceByStep.put(
              nextSteps,
              Arrays.copyOf(later.getValue(), nextSteps * EpsilonGrpFeature.FEATURE_SIZE));
        }
      }
      boundarySequences = List.copyOf(ordered);
      boundaryContexts = new DecisionBoundaryContext[boundarySequences.size()];
    }

    private TreeMap<Integer, float[]> predictionSequences() {
      return predictionSequenceByStep;
    }

    private void setPrior(int steps, float[] prior) {
      priorByStep.put(steps, prior);
      Integer boundaryIndex = boundaryIndexByStep.get(steps);
      if (boundaryIndex != null) {
        boundaryContexts[boundaryIndex] =
            DecisionBoundaryContext.fromGrpBoundary(boundarySequences.get(boundaryIndex), prior);
      }
    }

    private void materialize(
        EpsilonDecisionSample[] destination,
        boolean useBoundaryTargets,
        EpsilonUtilityProfile utilityProfile) {
      int[] seatOrdinals = new int[EpsilonGrpRanks.SEAT_COUNT];
      for (IndexedSample row : rows) {
        EpsilonDecisionSample source = row.sample();
        int boundaryIndex = boundaryIndexByStep.get(row.grpSteps());
        float[] nextBoundaryPrior = useBoundaryTargets ? priorByStep.get(row.grpSteps() + 1) : null;
        float valueTarget =
            nextBoundaryPrior == null
                ? utilityProfile.utilityForRank(source.finalRank())
                : EpsilonUtilityTargets.expectedRankUtilityTrusted(
                    utilityProfile,
                    EpsilonGrpRanks.seatMarginal(nextBoundaryPrior, source.playerSeat()));
        DecisionHostBatch input = source.input().copyRow(0);
        input.inputs().writer(0).boundaryContext(boundaryContexts[boundaryIndex]);
        destination[row.sourceIndex()] =
            new EpsilonDecisionSample(
                input,
                source.chosenLegalSlot(),
                source.chosenActionId(),
                source.behaviorLogProb(),
                valueTarget,
                0.0f,
                source.finalRank(),
                source.behaviorPolicy(),
                source.rolloutPolicy(),
                source.actorSnapshotId(),
                utilityProfile.ordinal(),
                source.gameId(),
                boundaryIndex,
                seatOrdinals[source.playerSeat()]++,
                boundarySequences.get(boundaryIndex),
                finalRanksCode,
                source.learningRole());
      }
    }

    private static void requireNested(
        Map.Entry<Integer, float[]> current, Map.Entry<Integer, float[]> later) {
      if (later.getKey() <= current.getKey()
          || later.getValue().length <= current.getValue().length) {
        throw new EpsilonDecisionDataException("GRP boundary prefixes must grow by whole steps");
      }
      for (int index = 0; index < current.getValue().length; index++) {
        if (Float.floatToRawIntBits(current.getValue()[index])
            != Float.floatToRawIntBits(later.getValue()[index])) {
          throw new EpsilonDecisionDataException(
              "GRP boundary prefixes are not nested: currentSteps="
                  + current.getKey()
                  + " laterSteps="
                  + later.getKey());
        }
      }
    }
  }

  private static boolean rawEquals(float[] first, float[] second) {
    if (first.length != second.length) {
      return false;
    }
    for (int index = 0; index < first.length; index++) {
      if (Float.floatToRawIntBits(first[index]) != Float.floatToRawIntBits(second[index])) {
        return false;
      }
    }
    return true;
  }

  private record IndexedSample(int sourceIndex, int grpSteps, EpsilonDecisionSample sample) {}

  private record PredictionRef(GameRows game, int steps) {}

  @FunctionalInterface
  interface MarginalPredictor {
    List<float[]> predict(List<float[]> sequences);
  }
}
