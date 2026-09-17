package com.epsilon.major.ai.decision.training;

import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.major.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import com.epsilon.major.ai.decision.data.EpsilonDecisionTrainingSampleDescriptorReader;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.major.ai.model.EpsilonDecisionLoss;
import com.epsilon.major.ai.model.EpsilonDecisionNetwork;
import com.epsilon.major.ai.model.EpsilonDecisionOutput;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 収集・更新単位ごとに固定した方策学習行を使い、更新前後の KL ダイバージェンスを測る。
 *
 * <p>選抜済みサンプルから一様に抽出し、データ本体への参照と更新前の合法手確率だけを保持する。収集時の方策やモデルの複製は使わず、勾配も計算しない。二回の順伝播は同じ学習ワーカー、容量区分、行順、学習モード、自動混合精度の設定で実行する。
 */
final class DecisionActorUpdateKlProbe {
  static final int MAXIMUM_ROWS = 2048;
  private static final long SAMPLE_SEED = 0x4143_544f_524b_4cL;

  private final List<DecisionTrainingMicroBatch> batches;
  private final DecisionPolicySignalMultipliers multipliers;
  private final DecisionComputePrecision precision;
  private final float[][] before;

  private DecisionActorUpdateKlProbe(
      List<DecisionTrainingMicroBatch> batches,
      DecisionPolicySignalMultipliers multipliers,
      DecisionComputePrecision precision) {
    this.batches = batches;
    this.multipliers = multipliers;
    this.precision = precision;
    before = new float[batches.size()][];
  }

  static DecisionActorUpdateKlProbe select(
      List<Path> paths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int microBatchRows,
      int maximumDeviceTransitionCells,
      int laneCount,
      DecisionPolicySignalMultipliers multipliers,
      DecisionComputePrecision precision)
      throws IOException {
    ArrayList<EpsilonDecisionTrainingSampleDescriptor> reservoir = new ArrayList<>(MAXIMUM_ROWS);
    Random random = new Random(SAMPLE_SEED);
    long eligible = 0;
    for (Path path : paths) {
      for (EpsilonDecisionTrainingSampleDescriptor sample : reader.read(path)) {
        if (!(actorWeight(sample, multipliers) > 0.0f)) {
          continue;
        }
        long selected = random.nextLong(++eligible);
        if (reservoir.size() < MAXIMUM_ROWS) {
          reservoir.add(sample);
        } else if (selected < MAXIMUM_ROWS) {
          reservoir.set((int) selected, sample);
        }
      }
    }
    EpsilonDecisionTrainingMicroBatchQueue queue =
        new EpsilonDecisionTrainingMicroBatchQueue(
            Math.min(microBatchRows, MAXIMUM_ROWS * laneCount),
            laneCount,
            maximumDeviceTransitionCells);
    ArrayList<DecisionTrainingMicroBatch> batches = new ArrayList<>();
    for (EpsilonDecisionTrainingSampleDescriptor sample : reservoir) {
      var complete = queue.add(sample);
      if (complete.isPresent()) {
        batches.add(
            DecisionTrainingMicroBatch.take(batches.size(), complete.orElseThrow(), multipliers));
      }
    }
    for (ArrayList<EpsilonDecisionTrainingSampleDescriptor> tail : queue.drain()) {
      batches.add(DecisionTrainingMicroBatch.take(batches.size(), tail, multipliers));
    }
    return new DecisionActorUpdateKlProbe(batches, multipliers, precision);
  }

  void captureBefore(EpsilonDecisionDataParallel dataParallel) throws IOException {
    evaluate(dataParallel, true);
  }

  float measureAfter(EpsilonDecisionDataParallel dataParallel) throws IOException {
    return evaluate(dataParallel, false);
  }

  int rows() {
    return batches.stream().mapToInt(batch -> batch.samples().size()).sum();
  }

  private float evaluate(EpsilonDecisionDataParallel dataParallel, boolean capture)
      throws IOException {
    if (batches.isEmpty()) {
      return 0.0f;
    }
    return dataParallel
        .<Float>invokeAssigned(new int[] {0}, List.of(lane -> evaluate(lane, capture)))
        .getFirst();
  }

  private float evaluate(EpsilonDecisionDataParallel.Lane lane, boolean capture)
      throws IOException {
    double weightedKl = 0.0;
    double weight = 0.0;
    try (EpsilonDecisionDataParallel.TrainingInputSequence inputs =
        lane.openTrainingInputs(batches, multipliers)) {
      EpsilonDecisionDataParallel.PreparedTrainingInput prepared;
      while ((prepared = inputs.next()) != null) {
        DecisionTrainingMicroBatch batch = prepared.microBatch();
        try (DecisionTrainingReadyBatch ready = prepared.ready();
            DecisionTrainingDeviceBatchLease input = lane.trainingInput().acquire(ready);
            // 自動微分の設定も実行するカーネルに影響するため、学習器と同じ設定にする。
            // 逆伝播とパラメーター更新は実行しない。
            GradientCollector collector = lane.newGradientCollector()) {
          NDManager manager = input.manager();
          EpsilonDecisionOutput output;
          try (Autocast autocast = EpsilonDecisionAutocast.open(lane.manager(), precision)) {
            output =
                ((EpsilonDecisionNetwork) lane.model().getBlock())
                    .forwardDecision(
                        new ParameterStore(manager, true),
                        input.batch(),
                        true,
                        lane.runtimeParameters());
            manager.attachAll(output.toNDList());
          }
          NDArray policy =
              EpsilonDecisionLoss.composePolicyDistributionForLoss(
                      output.policyScores(), input.batch())
                  .probabilities();
          float[] probabilities = policy.toFloatArray();
          if (capture) {
            before[batch.sourceOrdinal()] = probabilities;
            continue;
          }
          float[] previous = before[batch.sourceOrdinal()];
          int stride = batch.samples().getFirst().bucket().legalActionCapacity();
          for (int row = 0; row < batch.samples().size(); row++) {
            EpsilonDecisionTrainingSampleDescriptor sample = batch.samples().get(row);
            double actorWeight = actorWeight(sample, multipliers);
            weightedKl +=
                actorWeight
                    * policyKl(previous, probabilities, row * stride, sample.legalActionCount());
            weight += actorWeight;
          }
        }
      }
    }
    return capture ? 0.0f : (float) (weightedKl / weight);
  }

  /** FLOAT32の丸めで総和が1からずれる分だけ倍精度で正規化し、合法位置のKLを測る。 */
  static double policyKl(float[] before, float[] after, int offset, int count) {
    double beforeMass = 0.0;
    double afterMass = 0.0;
    for (int slot = offset; slot < offset + count; slot++) {
      beforeMass += before[slot];
      afterMass += after[slot];
    }
    double kl = 0.0;
    for (int slot = offset; slot < offset + count; slot++) {
      double previous = before[slot] / beforeMass;
      if (previous > 0.0) {
        double current = after[slot] / afterMass;
        kl += previous * Math.log(previous / current);
      }
    }
    return Math.max(0.0, kl);
  }

  private static float actorWeight(
      EpsilonDecisionTrainingSampleDescriptor sample, DecisionPolicySignalMultipliers multipliers) {
    if (sample.actorSnapshotId() <= 0L
        || !sample.learningRole().advancesActorClock()
        || sample.legalActionCount() <= 1) {
      return 0.0f;
    }
    return multipliers.forKind(
        EpsilonDecisionPointKind.ofLegalActions(
            sample.legalActionIdsView(), sample.legalActionCount()));
  }
}
