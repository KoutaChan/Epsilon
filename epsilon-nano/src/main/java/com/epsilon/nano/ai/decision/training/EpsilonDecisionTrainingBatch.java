package com.epsilon.nano.ai.decision.training;

import com.epsilon.nano.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import java.util.List;

/** 1つの小バッチに含まれる入力と教師値を、型付きの CPU バッファとして保持する。 */
public final class EpsilonDecisionTrainingBatch {

  private final DecisionHostBatch host;

  private EpsilonDecisionTrainingBatch(DecisionHostBatch host) {
    this.host = host;
  }

  /**
   * マイクロバッチの行数を返す。
   *
   * @return 型付きホスト側バッチの行数
   */
  public int size() {
    return host.size();
  }

  /**
   * デバイス転送前の入力と教師値を持つホスト側バッチを返す。
   *
   * @return 学習教師値を含むホスト側バッチ
   */
  public DecisionHostBatch host() {
    return host;
  }

  boolean hasPositiveActorWeight() {
    for (int row = 0; row < size(); row++) {
      if (host.actorWeight(row) > 0.0f && host.sampleWeight(row) > 0.0f) {
        return true;
      }
    }
    return false;
  }

  public float actorWeight(int row) {
    return host.actorWeight(row);
  }

  void setActorWeight(int row, float value) {
    host.setActorWeight(row, value);
  }

  void multiplyActorWeight(int row, float multiplier) {
    host.setActorWeight(row, host.actorWeight(row) * multiplier);
  }

  public float sampleWeight(int row) {
    return host.sampleWeight(row);
  }

  void setSampleWeight(int row, float value) {
    host.setSampleWeight(row, value);
  }

  float advantage(int row) {
    return host.advantage(row);
  }

  public static final class Materializer {

    private final float riichiDecisionWeight;
    private final float reactionDecisionWeight;

    public Materializer() {
      this(1.0f, 1.0f);
    }

    Materializer(float riichiDecisionWeight, float reactionDecisionWeight) {
      requireNonNegativeFinite(riichiDecisionWeight, "riichiDecisionWeight");
      requireNonNegativeFinite(reactionDecisionWeight, "reactionDecisionWeight");
      this.riichiDecisionWeight = riichiDecisionWeight;
      this.reactionDecisionWeight = reactionDecisionWeight;
    }

    public EpsilonDecisionTrainingBatch materialize(
        List<EpsilonDecisionSample> samples, int start, int count) {
      if (start < 0 || count < 1 || start + count > samples.size()) {
        throw new IndexOutOfBoundsException(
            "training batch slice outside samples: start=" + start + " count=" + count);
      }
      DecisionBucket bucket = samples.get(start).input().bucket();
      for (int index = start + 1; index < start + count; index++) {
        DecisionBucket sampleBucket = samples.get(index).input().bucket();
        if (!sampleBucket.equals(bucket)) {
          throw new IllegalArgumentException(
              "training microbatch mixes decision buckets: expected="
                  + bucket
                  + " actual="
                  + sampleBucket);
        }
      }
      DecisionBatchBuilder builder = DecisionBatchBuilder.training(count, bucket);
      for (int index = start; index < start + count; index++) {
        EpsilonDecisionSample sample = samples.get(index);
        int legalCount = sample.input().legalActionCount(0);
        boolean selfPlay = sample.actorSnapshotId() > 0L;
        float sampleWeight = sampleWeight(sample, riichiDecisionWeight, reactionDecisionWeight);
        float actorWeight =
            selfPlay && sample.learningRole().advancesActorClock() && legalCount > 1 ? 1.0f : 0.0f;
        builder.addTrustedEncodedTrainingRow(
            sample.input(),
            0,
            sample.chosenLegalSlot(),
            sample.behaviorPolicy(),
            sample.rolloutPolicy(),
            sample.valueTarget(),
            sample.advantage(),
            actorWeight,
            sampleWeight);
        builder.writeBranchTarget(index - start, sample.branchTarget());
      }
      return new EpsilonDecisionTrainingBatch(builder.build());
    }

    static float sampleWeight(
        EpsilonDecisionSample sample, float riichiDecisionWeight, float reactionDecisionWeight) {
      if (sample.actorSnapshotId() > 0L) {
        return 1.0f;
      }
      return switch (EpsilonDecisionPointKind.of(sample.input(), 0)) {
        case DAHAI -> 1.0f;
        case RIICHI -> riichiDecisionWeight;
        case REACTION -> reactionDecisionWeight;
      };
    }

    private static void requireNonNegativeFinite(float value, String label) {
      if (!Float.isFinite(value) || value < 0.0f) {
        throw new IllegalArgumentException(label + " must be finite and non-negative");
      }
    }
  }
}
