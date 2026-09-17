package com.epsilon.pico.training.replay;

import com.epsilon.ai.belief.EpsilonBeliefSample;
import com.epsilon.ai.belief.EpsilonBeliefTargetBuilder;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.ai.grp.EpsilonGrpSequence;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.ScoreRanking;
import com.epsilon.pico.ai.decision.EpsilonUtilityTargets;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.replay.DecisionPoint;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayObserver;
import com.epsilon.replay.ReplayRecord;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/** 牌譜の再生中に各判断時点の観測を保存し、対局終了後に教師値を付与する。 */
public final class EpsilonReplaySampleCollector implements ReplayObserver {
  private static final int NUM_PLAYERS = GameState.NUM_PLAYERS;
  private static final AtomicLong GAME_ID_SEQUENCE = new AtomicLong(1L);

  private final boolean collectBelief;
  private final EpsilonBeliefTargetBuilder beliefTargets;
  private final long gameId = GAME_ID_SEQUENCE.getAndIncrement();
  private final List<EpsilonBeliefSample<DecisionHostBatch>> beliefSamples = new ArrayList<>();
  private final EpsilonGrpSequence grpSequence = new EpsilonGrpSequence();

  @SuppressWarnings("unchecked")
  private final List<DecisionPretrainRecord>[] decisionRecordsBySeat =
      (List<DecisionPretrainRecord>[]) new List<?>[NUM_PLAYERS];

  private int decisionCount;

  private EpsilonReplaySampleCollector(boolean collectBelief) {
    this.collectBelief = collectBelief;
    beliefTargets = collectBelief ? new EpsilonBeliefTargetBuilder() : null;
    if (!collectBelief) {
      for (int seat = 0; seat < NUM_PLAYERS; seat++) {
        decisionRecordsBySeat[seat] = new ArrayList<>();
      }
    }
  }

  /** 対局終了が確定した牌譜から、最終順位を含むDecision教師を生成する。 */
  public static List<EpsilonDecisionSample> collectDecisionSamples(ReplayRecord record) {
    record.requireCompletedMatch();
    var collector = new EpsilonReplaySampleCollector(false);
    int[] scores = ReplayEngine.replay(record, collector);
    return collector.finalizeDecisionSamples(scores);
  }

  /** Belief教師は最終順位を使わず、記録内で確定した判断だけを利用する。 */
  public static List<EpsilonBeliefSample<DecisionHostBatch>> collectBeliefSamples(
      ReplayRecord record) {
    var collector = new EpsilonReplaySampleCollector(true);
    ReplayEngine.replay(record, collector);
    return collector.beliefSamples;
  }

  /** 状態と合法手はこの呼出し中だけ借用し、系列固有の入力・教師へ取り込む。 */
  @Override
  public void onDecision(
      DecisionPoint point, GameState state, int player, List<Action> legalActions) {
    if (point.choiceKind() == DecisionPoint.ChoiceKind.UNOBSERVED) return;
    int chosenSlot = point.chosenSlot();
    if (collectBelief) {
      beliefSamples.add(
          new EpsilonBeliefSample<>(
              DecisionBatchBuilder.stateOnlyBatch(state, player),
              beliefTargets.build(state, player)));
      return;
    }
    int actionId = legalActions.get(chosenSlot).toIndex();
    DecisionBatchBuilder builder =
        DecisionBatchBuilder.inference(1, DecisionBatchBuilder.selectInferenceBucket(legalActions));
    builder.addDetachedInferenceRow(
        state, player, legalActions, state.publicState(), DecisionBoundaryContext.uniform());
    decisionRecordsBySeat[player].add(
        new DecisionPretrainRecord(
            builder.build(), chosenSlot, actionId, grpSequence.include(state)));
    decisionCount++;
  }

  private List<EpsilonDecisionSample> finalizeDecisionSamples(int[] finalScores) {
    if (finalScores == null || decisionCount == 0) {
      return List.of();
    }

    int[] finalRanks = ScoreRanking.byScoreThenSeat(finalScores);
    int finalRanksCode = EpsilonGrpRanks.encode(finalRanks);
    int trainingProfile = EpsilonUtilityTargets.configuredTrainingProfile().ordinal();
    ArrayList<EpsilonDecisionSample> samples = new ArrayList<>(decisionCount);
    for (int seat = 0; seat < NUM_PLAYERS; seat++) {
      int finalRank = finalRanks[seat];
      for (DecisionPretrainRecord record : decisionRecordsBySeat[seat]) {
        samples.add(
            EpsilonDecisionSample.fromPretrainingLog(
                record.input(),
                record.chosenLegalSlot(),
                record.chosenActionId(),
                finalRank,
                trainingProfile,
                gameId,
                record.grpFeatureSequence(),
                finalRanksCode));
      }
    }
    return samples;
  }

  private record DecisionPretrainRecord(
      DecisionHostBatch input,
      int chosenLegalSlot,
      int chosenActionId,
      float[] grpFeatureSequence) {}
}
