package com.epsilon.nano.ai.decision.training;

import ai.djl.training.tracker.Tracker;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings.ActorKlControlSettings;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings.TargetKlDecaySettings;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** 方策更新の KL ダイバージェンスを使って次回の学習率を調整し、目標値の減衰計画と受理済みの更新回数を管理する。 */
final class DecisionActorLearningControl implements Tracker {

  private static final float MINIMUM_CONTROL_KL = 1.0e-12f;

  private final ActorKlControlSettings settings;
  private volatile State state;

  DecisionActorLearningControl(ActorKlControlSettings settings) {
    this(
        settings,
        new State(
            0,
            0L,
            Objects.requireNonNull(settings, "settings").initialLearningRate(),
            List.of(),
            false,
            0.0f,
            0.0f,
            List.of(),
            settings.targetDecay(),
            null));
  }

  DecisionActorLearningControl(ActorKlControlSettings settings, State restoredState) {
    validateRestored(settings, restoredState);
    this.settings = settings;
    state = restoredState;
  }

  float currentLearningRate() {
    return state.currentActorLearningRate();
  }

  /** AdamWのupdateCountで受理された更新回数を進めず、受理学習の反復で確定した現在LRだけを返す。 */
  @Override
  public float getNewValue(int updateCount) {
    return state.currentActorLearningRate();
  }

  State state() {
    return state;
  }

  /** 受理した学習の反復を履歴へ追加し、次の学習の反復へ適用する学習率を決める。 */
  Adjustment accept(float observedMeanKl, int optimizerSteps) {
    requireObservedKl(observedMeanKl);
    if (optimizerSteps <= 0) {
      throw new IllegalArgumentException("accepted Actor optimizer steps must be positive");
    }
    long acceptedUpdates = Math.addExact(state.acceptedActorOptimizerSteps(), optimizerSteps);
    float appliedLearningRate = state.currentActorLearningRate();
    int acceptedMacros = Math.addExact(state.acceptedMacros(), 1);

    List<Float> warmup = state.warmupSamePathUpdateKls();
    ArrayList<Float> recent = new ArrayList<>(state.recentSamePathUpdateKls());
    appendBounded(recent, observedMeanKl, settings.medianWindowMacros());

    boolean targetResolved = state.targetResolved();
    float baseline = state.baselineSamePathUpdateKl();
    float target = state.targetSamePathUpdateKl();
    TargetSchedule schedule = state.targetSchedule();
    if (!targetResolved) {
      ArrayList<Float> collectingWarmup = new ArrayList<>(warmup);
      collectingWarmup.add(observedMeanKl);
      warmup = collectingWarmup;
      if (warmup.size() == settings.warmupMacros()) {
        baseline = Math.max(MINIMUM_CONTROL_KL, median(warmup));
        float initialTarget =
            finiteFloat((double) baseline * settings.targetMultiplier(), "initial target KL");
        schedule =
            TargetSchedule.initial(acceptedUpdates, initialTarget, state.initialTargetPlan());
        target = schedule.targetAt(acceptedUpdates);
        targetResolved = true;
      }
      state =
          new State(
              acceptedMacros,
              acceptedUpdates,
              appliedLearningRate,
              warmup,
              targetResolved,
              baseline,
              target,
              recent,
              state.initialTargetPlan(),
              schedule);
      return adjustment(
          Action.WARMUP, appliedLearningRate, observedMeanKl, median(recent), appliedLearningRate);
    }

    target = schedule.targetAt(acceptedUpdates);
    float windowMedian = median(recent);
    float lower = target / settings.deadbandFactor();
    float upper = target * settings.deadbandFactor();
    Action action = Action.HOLD;
    float rawNext = appliedLearningRate;
    if (windowMedian < lower) {
      // 減衰目標が数値上の下限より小さくても、低KLに対するLR増加を逆転させない。
      double observedFloor = Math.min(MINIMUM_CONTROL_KL, lower);
      rawNext =
          finiteFloat(
              (double) appliedLearningRate
                  * Math.sqrt((double) target / Math.max(windowMedian, observedFloor)),
              "raw next Actor learning rate");
      action = Action.INCREASE;
    } else if (windowMedian > upper) {
      rawNext =
          finiteFloat(
              (double) appliedLearningRate * Math.sqrt((double) target / windowMedian),
              "raw next Actor learning rate");
      action = Action.DECREASE;
    }

    float next = rawNext;
    if (action != Action.HOLD) {
      float maximumFactor = settings.maximumAdjustmentFactorPerMacro();
      next =
          Math.max(
              appliedLearningRate / maximumFactor,
              Math.min(appliedLearningRate * maximumFactor, rawNext));
      if (next > settings.hardMaximumLearningRate()) {
        next = settings.hardMaximumLearningRate();
        action = Action.HARD_MAX_CLAMPED;
      } else if (next < settings.hardMinimumLearningRate()) {
        next = settings.hardMinimumLearningRate();
        action = Action.HARD_MIN_CLAMPED;
      } else if (action == Action.INCREASE
          && appliedLearningRate >= settings.hardMaximumLearningRate()) {
        next = settings.hardMaximumLearningRate();
        action = Action.HARD_MAX_CLAMPED;
      } else if (action == Action.DECREASE
          && appliedLearningRate <= settings.hardMinimumLearningRate()) {
        next = settings.hardMinimumLearningRate();
        action = Action.HARD_MIN_CLAMPED;
      }
    }

    state =
        new State(
            acceptedMacros,
            acceptedUpdates,
            next,
            warmup,
            true,
            baseline,
            target,
            recent,
            state.initialTargetPlan(),
            schedule);
    return adjustment(action, appliedLearningRate, observedMeanKl, windowMedian, rawNext);
  }

  /** 棄却した学習の反復を履歴へ加えず、LRも変更しない診断結果を作る。 */
  Adjustment reject(float observedMeanKl) {
    requireObservedKl(observedMeanKl);
    float current = state.currentActorLearningRate();
    float windowMedian =
        state.recentSamePathUpdateKls().isEmpty() ? 0.0f : median(state.recentSamePathUpdateKls());
    return adjustment(Action.REJECTED, current, observedMeanKl, windowMedian, current);
  }

  private Adjustment adjustment(
      Action action,
      float appliedLearningRate,
      float observedMeanKl,
      float windowMedian,
      float rawNextLearningRate) {
    State current = state;
    return new Adjustment(
        action,
        current.acceptedMacros(),
        current.acceptedActorOptimizerSteps(),
        appliedLearningRate,
        observedMeanKl,
        windowMedian,
        current.baselineSamePathUpdateKl(),
        current.targetSamePathUpdateKl(),
        rawNextLearningRate,
        current.currentActorLearningRate(),
        current.initialTargetPlan(),
        current.targetSchedule());
  }

  /** チェックポイント読込時に、制御オブジェクトを作らず保存状態と設定の整合を検証する。 */
  static void validateRestored(ActorKlControlSettings settings, State restored) {
    Objects.requireNonNull(settings, "settings");
    Objects.requireNonNull(restored, "restoredState");
    float current = restored.currentActorLearningRate();
    if (current < settings.hardMinimumLearningRate()
        || current > settings.hardMaximumLearningRate()) {
      throw new IllegalArgumentException(
          "restored Actor learning rate is outside configured hard limits: " + current);
    }
    int expectedWarmup = Math.min(restored.acceptedMacros(), settings.warmupMacros());
    if (restored.warmupSamePathUpdateKls().size() != expectedWarmup) {
      throw new IllegalArgumentException("restored Actor KL warmup history length is inconsistent");
    }
    boolean expectedResolved = restored.acceptedMacros() >= settings.warmupMacros();
    if (restored.targetResolved() != expectedResolved) {
      throw new IllegalArgumentException("restored Actor KL target resolution is inconsistent");
    }
    int expectedRecent = Math.min(restored.acceptedMacros(), settings.medianWindowMacros());
    if (restored.recentSamePathUpdateKls().size() != expectedRecent) {
      throw new IllegalArgumentException("restored Actor KL recent history length is inconsistent");
    }
    if (restored.targetResolved()) {
      float expectedBaseline =
          Math.max(MINIMUM_CONTROL_KL, median(restored.warmupSamePathUpdateKls()));
      if (Float.compare(expectedBaseline, restored.baselineSamePathUpdateKl()) != 0) {
        throw new IllegalArgumentException("restored Actor KL baseline or target is inconsistent");
      }
    }
  }

  /** 現在の更新回数と目標値から、目標値を維持する期間を追加せずにコサイン減衰計画を組み直す。オプティマイザーと履歴は維持する。 */
  void replan(long endOptimizerStep, float endTargetKl) {
    requireResolvedNonIncreasingTarget(endTargetKl);
    long currentUpdate = state.acceptedActorOptimizerSteps();
    replaceSchedule(
        TargetSchedule.replan(
            currentUpdate, state.targetSamePathUpdateKl(), endOptimizerStep, endTargetKl));
  }

  /** 明示的な固定目標実験。現在目標以下だけを許し、学習率、AdamW、更新回数は変更しない。 */
  void holdTarget(float targetKl) {
    requireResolvedNonIncreasingTarget(targetKl);
    long currentUpdate = state.acceptedActorOptimizerSteps();
    replaceSchedule(TargetSchedule.hold(currentUpdate, targetKl));
  }

  private void requireResolvedNonIncreasingTarget(float target) {
    if (!state.targetResolved()) {
      throw new IllegalStateException(
          "KL calibration must complete before changing the target plan");
    }
    if (!Float.isFinite(target) || target <= 0.0f || target > state.targetSamePathUpdateKl()) {
      throw new IllegalArgumentException(
          "explicit target must be positive, finite, and no greater than the current target");
    }
  }

  private void replaceSchedule(TargetSchedule schedule) {
    state =
        new State(
            state.acceptedMacros(),
            state.acceptedActorOptimizerSteps(),
            state.currentActorLearningRate(),
            state.warmupSamePathUpdateKls(),
            true,
            state.baselineSamePathUpdateKl(),
            schedule.targetAt(state.acceptedActorOptimizerSteps()),
            state.recentSamePathUpdateKls(),
            state.initialTargetPlan(),
            schedule);
  }

  private static void appendBounded(ArrayList<Float> values, float value, int maximumSize) {
    values.add(value);
    while (values.size() > maximumSize) {
      values.removeFirst();
    }
  }

  private static float median(List<Float> values) {
    if (values.isEmpty()) {
      return 0.0f;
    }
    ArrayList<Float> sorted = new ArrayList<>(values);
    Collections.sort(sorted);
    int middle = sorted.size() / 2;
    return sorted.size() % 2 == 1
        ? sorted.get(middle)
        : (float) (((double) sorted.get(middle - 1) + sorted.get(middle)) / 2.0);
  }

  private static float finiteFloat(double value, String label) {
    float result = (float) value;
    if (!Float.isFinite(result) || result <= 0.0f) {
      throw new IllegalStateException(label + " must be positive and finite: " + value);
    }
    return result;
  }

  private static void requireObservedKl(float value) {
    if (!Float.isFinite(value) || value < 0.0f) {
      throw new IllegalArgumentException("observed mean KL must be finite and non-negative");
    }
  }

  /** 受理更新数と同じ実行経路で測った KLを保存する。計画は設定から独立してチェックポイントが所有する。 */
  record State(
      int acceptedMacros,
      long acceptedActorOptimizerSteps,
      float currentActorLearningRate,
      List<Float> warmupSamePathUpdateKls,
      boolean targetResolved,
      float baselineSamePathUpdateKl,
      float targetSamePathUpdateKl,
      List<Float> recentSamePathUpdateKls,
      TargetKlDecaySettings initialTargetPlan,
      TargetSchedule targetSchedule) {

    State {
      if (acceptedMacros < 0 || acceptedActorOptimizerSteps < acceptedMacros) {
        throw new IllegalArgumentException("accepted macro/update clocks are inconsistent");
      }
      requirePositiveFinite(currentActorLearningRate, "currentActorLearningRate");
      warmupSamePathUpdateKls = validatedKls(warmupSamePathUpdateKls, "warmup KL");
      recentSamePathUpdateKls = validatedKls(recentSamePathUpdateKls, "recent KL");
      Objects.requireNonNull(initialTargetPlan, "initialTargetPlan");
      if (initialTargetPlan.plannedOptimizerSteps() < 0
          || !Float.isFinite(initialTargetPlan.finalScale())
          || initialTargetPlan.finalScale() <= 0.0f
          || initialTargetPlan.finalScale() > 1.0f) {
        throw new IllegalArgumentException("invalid initial KL target plan");
      }
      if (targetResolved) {
        requirePositiveFinite(baselineSamePathUpdateKl, "baselineSamePathUpdateKl");
        requirePositiveFinite(targetSamePathUpdateKl, "targetSamePathUpdateKl");
        Objects.requireNonNull(targetSchedule, "targetSchedule");
        if (targetSchedule.anchorUpdate() > acceptedActorOptimizerSteps
            || Float.compare(
                    targetSamePathUpdateKl, targetSchedule.targetAt(acceptedActorOptimizerSteps))
                != 0) {
          throw new IllegalArgumentException("KL target does not match the saved update schedule");
        }
      } else if (baselineSamePathUpdateKl != 0.0f
          || targetSamePathUpdateKl != 0.0f
          || targetSchedule != null) {
        throw new IllegalArgumentException("unresolved KL target must have no schedule or target");
      }
    }
  }

  /** 初回目標値を維持する期間とコサイン減衰区間。明示計画変更ではdecayStartUpdateをanchorUpdateへ置く。 */
  record TargetSchedule(
      long anchorUpdate,
      float anchorTarget,
      long decayStartUpdate,
      long endUpdate,
      float endTarget) {
    TargetSchedule {
      requirePositiveFinite(anchorTarget, "anchorTarget");
      requirePositiveFinite(endTarget, "endTarget");
      if (anchorUpdate < 0
          || decayStartUpdate < anchorUpdate
          || endUpdate < decayStartUpdate
          || endTarget > anchorTarget) {
        throw new IllegalArgumentException(
            "KL schedule requires ordered clocks and non-increasing targets");
      }
    }

    static TargetSchedule initial(
        long calibrationUpdate, float target, TargetKlDecaySettings plan) {
      long planned = plan.plannedOptimizerSteps();
      if (planned == 0L) {
        return hold(calibrationUpdate, target);
      }
      if (planned <= calibrationUpdate) {
        throw new IllegalStateException("Actor KL target plan ended before calibration completed");
      }
      long holdUntil = Math.max(calibrationUpdate, planned / 10 + (planned % 10 == 0 ? 0 : 1));
      return new TargetSchedule(
          calibrationUpdate,
          target,
          holdUntil,
          planned,
          finiteFloat((double) target * plan.finalScale(), "end target KL"));
    }

    static TargetSchedule replan(long update, float target, long endUpdate, float endTarget) {
      if (endUpdate <= update) {
        throw new IllegalArgumentException(
            "replanned end update must be after the current accepted update");
      }
      return new TargetSchedule(update, target, update, endUpdate, endTarget);
    }

    static TargetSchedule hold(long update, float target) {
      return new TargetSchedule(update, target, update, update, target);
    }

    float targetAt(long update) {
      if (update >= endUpdate) {
        return endTarget;
      }
      if (update <= decayStartUpdate) {
        return anchorTarget;
      }
      double progress = (double) (update - decayStartUpdate) / (endUpdate - decayStartUpdate);
      return (float)
          (endTarget
              + ((double) anchorTarget - endTarget) * 0.5 * (1.0 + Math.cos(Math.PI * progress)));
    }
  }

  private static List<Float> validatedKls(List<Float> values, String label) {
    List<Float> copy = List.copyOf(values);
    for (float value : copy) {
      if (!Float.isFinite(value) || value < 0.0f) {
        throw new IllegalArgumentException(label + " values must be finite and non-negative");
      }
    }
    return copy;
  }

  private static void requirePositiveFinite(float value, String label) {
    if (!Float.isFinite(value) || value <= 0.0f) {
      throw new IllegalArgumentException(label + " must be positive and finite");
    }
  }

  enum Action {
    WARMUP,
    HOLD,
    INCREASE,
    DECREASE,
    HARD_MIN_CLAMPED,
    HARD_MAX_CLAMPED,
    REJECTED
  }

  /** 一つの学習の反復観測、次の学習の反復 LRと保存中の目標計画。targetMeanKlは受理した方策モデル更新数に対応する。 */
  record Adjustment(
      Action action,
      int acceptedMacros,
      long acceptedActorOptimizerSteps,
      float appliedLearningRate,
      float observedMeanKl,
      float windowMedianMeanKl,
      float baselineMeanKl,
      float targetMeanKl,
      float rawNextLearningRate,
      float nextLearningRate,
      TargetKlDecaySettings initialTargetPlan,
      TargetSchedule targetSchedule) {}
}
