package com.epsilon.reviewer.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.replay.DecisionPoint;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Ankan;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.EndGame;
import com.epsilon.replay.ReplayEvent.EndKyoku;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.Reach;
import com.epsilon.replay.ReplayEvent.ReachAccepted;
import com.epsilon.replay.ReplayEvent.Ryukyoku;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.replay.ReplayObserver;
import com.epsilon.reviewer.dto.ActionCandidate;
import com.epsilon.reviewer.dto.PreparedRecord;
import com.epsilon.reviewer.dto.ReplayRound;
import com.epsilon.reviewer.dto.ReplayStep;
import com.epsilon.reviewer.dto.ReviewResult;
import com.epsilon.reviewer.model.ModelDefinition;
import com.epsilon.spi.ReviewEvaluation;
import com.epsilon.spi.ReviewPolicy;
import com.epsilon.spi.ReviewPolicyProvider;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import java.util.function.IntConsumer;

/** 実牌譜の復元と系列所有の方策を接続する。推論失敗を疑似確率で隠さない。 */
public final class ReviewEngine {
  /** 確率の高い順（安定ソートで同率は合法手の順）。 */
  private static final Comparator<ActionCandidate> BY_PROBABILITY =
      Comparator.comparingDouble(ActionCandidate::probability).reversed();

  /** 価値の損失の小さい順、同じなら確率の高い順。 */
  private static final Comparator<ActionCandidate> BY_VALUE =
      Comparator.comparingDouble((ActionCandidate candidate) -> candidate.valueLoss())
          .thenComparing(BY_PROBABILITY);

  /** 系列 ID → 方策の提供元（クラスパスの ServiceLoader から起動時に 1 回だけ集める）。 */
  private final Map<String, ReviewPolicyProvider> providers;

  public ReviewEngine() {
    var found = new LinkedHashMap<String, ReviewPolicyProvider>();
    for (ReviewPolicyProvider provider : ServiceLoader.load(ReviewPolicyProvider.class))
      if (found.putIfAbsent(provider.seriesId(), provider) != null)
        throw new IllegalStateException("Duplicate review policy series: " + provider.seriesId());
    providers = Map.copyOf(found);
  }

  /** 解析できるモデル系列の ID。 */
  public Set<String> series() {
    return providers.keySet();
  }

  public PreparedRecord prepareRecord(byte[] bytes, String fileName) {
    return RecordPreparation.readPreparedRecord(bytes, fileName);
  }

  public ReviewResult analyze(
      PreparedRecord record,
      ModelDefinition model,
      BooleanSupplier cancelled,
      IntConsumer progress) {
    ReviewPolicyProvider provider = providers.get(model.series());
    if (provider == null)
      throw new IllegalArgumentException(
          "Review policy provider is unavailable for series: " + model.series());
    throwIfAnalysisCancelled(cancelled);
    try (ReviewPolicy policy = provider.openReview(model.checkpoint(), model.options())) {
      return analyzeWithPolicy(record, model, policy, cancelled, progress);
    }
  }

  /** 開いた方策で解析する。方策は閉じない。 */
  ReviewResult analyzeWithPolicy(
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
            ReviewEvaluation evaluation = policy.evaluate(state.publicObservation(player), legal);
            if (evaluation.actions() != legal.size())
              throw new IllegalStateException(
                  "Policy evaluated "
                      + evaluation.actions()
                      + " actions for "
                      + legal.size()
                      + " legal actions.");
            float[] losses = evaluation.valued() ? evaluation.losses() : null;
            List<ActionCandidate> candidates = new ArrayList<>(legal.size());
            for (int i = 0; i < legal.size(); i++)
              candidates.add(
                  ReviewSnapshots.candidate(
                      legal.get(i),
                      evaluation.probabilities()[i],
                      losses == null ? null : losses[i],
                      i == chosen,
                      state,
                      player));
            candidates.sort(evaluation.valued() ? BY_VALUE : BY_PROBABILITY);
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
