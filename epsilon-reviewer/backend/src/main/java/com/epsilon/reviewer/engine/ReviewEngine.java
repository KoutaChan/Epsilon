package com.epsilon.reviewer.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.replay.DecisionPoint;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.*;
import com.epsilon.replay.ReplayObserver;
import com.epsilon.reviewer.dto.*;
import com.epsilon.reviewer.model.ModelDefinition;
import com.epsilon.spi.ReviewPolicy;
import com.epsilon.spi.ReviewPolicyProvider;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import java.util.function.IntConsumer;

/** 実牌譜の復元と系列所有の方策を接続する。推論失敗を疑似確率で隠さない。 */
public final class ReviewEngine {
  public PreparedRecord prepareRecord(byte[] bytes, String fileName) {
    return RecordPreparation.readPreparedRecord(bytes, fileName);
  }

  public ReviewResult analyze(
      PreparedRecord record,
      ModelDefinition model,
      BooleanSupplier cancelled,
      IntConsumer progress) {
    ReviewPolicyProvider provider =
        ServiceLoader.load(ReviewPolicyProvider.class).stream()
            .map(ServiceLoader.Provider::get)
            .filter(candidate -> candidate.seriesId().equals(model.series()))
            .findFirst()
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "Review policy provider is unavailable for series: " + model.series()));
    throwIfAnalysisCancelled(cancelled);
    try (ReviewPolicy policy = provider.openReview(model.checkpoint(), model.options())) {
      return analyzeWithPolicy(record, model, policy, cancelled, progress);
    }
  }

  private ReviewResult analyzeWithPolicy(
      PreparedRecord record,
      ModelDefinition model,
      ReviewPolicy policy,
      BooleanSupplier cancelled,
      IntConsumer progress) {
    List<ReplayRound> rounds = new ArrayList<>();
    var observer =
        new ReplayObserver() {
          private List<ReplayStep> steps;
          private ReviewStateSequence states;

          @Override
          public void onDecision(
              DecisionPoint point, GameState state, int player, List<Action> legal) {
            int eventIndex = point.eventIndex();
            int chosen = point.chosenSlot();
            throwIfAnalysisCancelled(cancelled);
            // 全情報の牌譜再生状態はここから推論へ渡さず、当該席が知る公開観測だけを借用する。
            float[] probabilities = policy.probabilities(state.publicObservation(player), legal);
            requireNormalizedLegalActionProbabilities(probabilities, legal.size());
            List<ActionCandidate> candidates = new ArrayList<>(legal.size());
            for (int i = 0; i < legal.size(); i++)
              candidates.add(
                  ReviewSnapshots.candidate(
                      legal.get(i), probabilities[i], i == chosen, state, player));
            candidates.sort(Comparator.comparingDouble(ActionCandidate::probability).reversed());
            states.captureState(point.stateId(), state);
            steps.add(
                new ReplayStep(
                    steps.size(),
                    eventIndex,
                    point.stateId(),
                    chosen < 0 ? "response" : legal.get(chosen).type().name(),
                    player,
                    true,
                    point.causeEventIndex(),
                    candidates,
                    null));
          }

          @Override
          public void onEventApplied(
              int eventIndex, int stateId, ReplayEvent event, GameState state) {
            throwIfAnalysisCancelled(cancelled);
            if (event instanceof StartGame) return;
            if (event instanceof StartKyoku start) {
              steps = new ArrayList<>();
              states = new ReviewStateSequence();
              rounds.add(
                  new ReplayRound(
                      "round-" + rounds.size(),
                      state.roundIndex(),
                      state.honba(),
                      state.dealer(),
                      states.frames(),
                      steps));
            }
            if (steps != null) {
              states.captureState(stateId, state);
              steps.add(
                  new ReplayStep(
                      steps.size(),
                      eventIndex,
                      stateId,
                      eventType(event),
                      actor(event),
                      false,
                      null,
                      List.of(),
                      ReviewSnapshots.snapshotRoundOutcome(event, state)));
            }
            progress.accept(Math.min(99, (eventIndex + 1) * 100 / record.replay().events().size()));
          }
        };
    ReplayEngine.replay(record.replay(), observer);
    throwIfAnalysisCancelled(cancelled);
    progress.accept(100);
    return new ReviewResult(
        ReviewResult.CURRENT_FORMAT_VERSION,
        UUID.randomUUID().toString(),
        Instant.now().toString(),
        record.metadata(),
        model.info(),
        rounds);
  }

  private static void requireNormalizedLegalActionProbabilities(float[] probabilities, int count) {
    if (probabilities.length != count)
      throw new IllegalStateException("Policy output count does not match legal action count.");
    double sum = 0;
    for (float value : probabilities) {
      if (!Float.isFinite(value) || value < 0 || value > 1)
        throw new IllegalStateException("Policy returned an invalid probability.");
      sum += value;
    }
    if (Math.abs(sum - 1) > 0.0001)
      throw new IllegalStateException("Policy probabilities do not sum to one: " + sum);
  }

  private static void throwIfAnalysisCancelled(BooleanSupplier cancelled) {
    if (cancelled.getAsBoolean() || Thread.currentThread().isInterrupted())
      throw new CancellationException("Analysis was cancelled.");
  }

  private static String eventType(ReplayEvent event) {
    return switch (event) {
      case StartGame ignored -> "start_game";
      case StartKyoku ignored -> "start_kyoku";
      case Tsumo ignored -> "tsumo";
      case Dahai ignored -> "dahai";
      case Chi ignored -> "chi";
      case Pon ignored -> "pon";
      case Daiminkan ignored -> "daiminkan";
      case Ankan ignored -> "ankan";
      case Kakan ignored -> "kakan";
      case Reach ignored -> "reach";
      case ReachAccepted ignored -> "reach_accepted";
      case Dora ignored -> "dora";
      case Hora ignored -> "hora";
      case Ryukyoku ignored -> "ryukyoku";
      case EndKyoku ignored -> "end_kyoku";
      case EndGame ignored -> "end_game";
      case None ignored -> "none";
    };
  }

  private static int actor(ReplayEvent event) {
    return switch (event) {
      case Tsumo e -> e.actor();
      case Dahai e -> e.actor();
      case Chi e -> e.actor();
      case Pon e -> e.actor();
      case Daiminkan e -> e.actor();
      case Ankan e -> e.actor();
      case Kakan e -> e.actor();
      case Reach e -> e.actor();
      case ReachAccepted e -> e.actor();
      case Hora e -> e.actor();
      case None e -> e.actor();
      default -> -1;
    };
  }
}
