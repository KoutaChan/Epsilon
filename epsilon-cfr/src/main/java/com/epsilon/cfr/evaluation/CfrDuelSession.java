package com.epsilon.cfr.evaluation;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.ai.decision.duel.DuelEvaluationSource;
import com.epsilon.ai.decision.duel.EpsilonDecisionDuelArena;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.runtime.CfrPolicyFleet;
import com.epsilon.core.GameState;
import com.epsilon.network.NetworkDeviceResolver;
import com.epsilon.spi.PolicyExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Consumer;

/** 二つの平均方策を一度だけ読み込み、Nanoと共通の仕組みで同一牌山による対戦評価を行う。 */
public final class CfrDuelSession implements DuelEvaluationSource, AutoCloseable {
  private final Path candidateCheckpoint;
  private final Path opponentCheckpoint;
  private final CfrConfig.Evaluation settings;
  private final EpsilonUtilityProfile utilityProfile;
  private final PolicyExecutionContext execution;
  private final CfrPolicyFleet candidate;
  private final CfrPolicyFleet opponent;
  private final CfrDuelPolicy candidatePolicy;
  private final CfrDuelPolicy opponentPolicy;

  private CfrDuelSession(
      Path candidateCheckpoint,
      Path opponentCheckpoint,
      CfrConfig config,
      PolicyExecutionContext execution,
      CfrPolicyFleet candidate,
      CfrPolicyFleet opponent) {
    this.candidateCheckpoint = candidateCheckpoint.toAbsolutePath().normalize();
    this.opponentCheckpoint = opponentCheckpoint.toAbsolutePath().normalize();
    settings = config.evaluation();
    utilityProfile = config.utilityProfile();
    this.execution = execution;
    this.candidate = candidate;
    this.opponent = opponent;
    int batch = settings.evalVs().maximumInferenceBatch();
    candidatePolicy = new CfrDuelPolicy(candidate, batch);
    opponentPolicy = new CfrDuelPolicy(opponent, batch);
  }

  public static CfrDuelSession open(
      Path candidateCheckpoint, Path opponentCheckpoint, CfrConfig config) throws IOException {
    var devices = NetworkDeviceResolver.inference(config.network().devices());
    PolicyExecutionContext execution =
        new PolicyExecutionContext(
            config.evaluation().arena().advanceWorkers(),
            config.collection().slotsPerDevice(),
            config.collection().readyBatchesPerDevice());
    CfrPolicyFleet candidate = null;
    CfrPolicyFleet opponent = null;
    try (CfrModel candidateModel = CfrModel.loadModel(candidateCheckpoint, Device.cpu());
        CfrModel opponentModel = CfrModel.loadModel(opponentCheckpoint, Device.cpu())) {
      if (candidateModel.role() != CfrModel.Role.AVERAGE
          || opponentModel.role() != CfrModel.Role.AVERAGE)
        throw new IOException("CFR duel requires average-policy checkpoints");
      if (candidateModel.config().utilityProfile() != config.utilityProfile()
          || opponentModel.config().utilityProfile() != config.utilityProfile())
        throw new IOException("CFR duel/checkpoint utility profile mismatch");
      candidate = new CfrPolicyFleet(candidateModel, devices, execution);
      opponent = new CfrPolicyFleet(opponentModel, devices, execution);
      return new CfrDuelSession(
          candidateCheckpoint, opponentCheckpoint, config, execution, candidate, opponent);
    } catch (IOException | RuntimeException | Error failure) {
      if (opponent != null) closeAfterFailure(opponent, failure);
      if (candidate != null) closeAfterFailure(candidate, failure);
      closeAfterFailure(execution, failure);
      throw failure;
    }
  }

  @Override
  public DuelEvaluation evaluateWalls(
      int wallSeeds,
      long seedBase,
      long firstWallFamilyId,
      int gamesInFlight,
      Consumer<DuelEvaluation.WallOutcome> sink) {
    return EpsilonDecisionDuelArena.evaluateDuelStreaming(
        candidateCheckpoint,
        opponentCheckpoint,
        candidatePolicy,
        opponentPolicy,
        Math.multiplyExact(wallSeeds, GameState.NUM_PLAYERS),
        seedBase,
        firstWallFamilyId,
        gamesInFlight,
        sink,
        utilityProfile,
        settings.arena(),
        settings.batching());
  }

  @Override
  public Path candidateCheckpoint() {
    return candidateCheckpoint;
  }

  @Override
  public Path opponentCheckpoint() {
    return opponentCheckpoint;
  }

  @Override
  public void close() {
    try {
      opponent.close();
    } finally {
      try {
        candidate.close();
      } finally {
        execution.close();
      }
    }
  }

  private static void closeAfterFailure(AutoCloseable resource, Throwable failure) {
    try {
      resource.close();
    } catch (Exception | Error closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }
}
