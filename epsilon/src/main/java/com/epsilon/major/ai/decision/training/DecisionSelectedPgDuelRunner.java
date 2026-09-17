package com.epsilon.major.ai.decision.training;

import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator;
import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.duel.EpsilonDecisionDuelSession;
import com.epsilon.major.config.settings.DecisionSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 候補モデルを固定牌山で対戦評価し、採用・継続・終了の状態を確定する。 */
final class DecisionSelectedPgDuelRunner {

  private static final Logger log = LoggerFactory.getLogger(DecisionSelectedPgDuelRunner.class);

  private DecisionSelectedPgDuelRunner() {}

  /** 検証条件と対戦評価から候補の判定を確定し、採用時に対局モデルの参照を更新する。 */
  static Resolution resolve(
      Request request, DecisionExecutionContext context, SettingsLoader config) throws IOException {
    Request actual = Objects.requireNonNull(request, "request");
    Path journal = actual.reportDirectory().resolve("duel-resolution.json");
    SavedResolution saved;
    if (Files.isRegularFile(journal)) {
      saved = DecisionSelectedPgCampaignState.readJson(journal, SavedResolution.class);
      if (!saved.candidate().equals(actual.candidate())) {
        throw new IOException("Duel resolution belongs to another candidate: " + journal);
      }
    } else {
      ChampionDuel duel =
          actual.guardFailure().isPresent() ? null : evaluate(actual, context, config);
      DecisionSelectedPgDuelResult.Status status =
          duel == null
              ? DecisionSelectedPgDuelResult.Status.GUARD_FAILED
              : DecisionSelectedPgDuelResult.resolveStatus(
                  duel.result().decision(), actual.campaignMacros(), actual.maximumMacros());
      saved =
          new SavedResolution(
              actual.candidate(),
              status,
              duel == null ? actual.guardFailure().orElseThrow() : duel.summary(),
              duel);
      // 昇格参照等の副作用より先に評価を確定し、再開時に別の評価を選び直さない。
      DecisionSelectedPgCampaignState.writeJson(journal, saved);
    }
    Optional<ChampionDuel> championDuel = Optional.ofNullable(saved.championDuel());
    DecisionSelectedPgDuelResult.Status status = saved.status();
    String summary = saved.summary();

    if (status == DecisionSelectedPgDuelResult.Status.PROMOTED) {
      EpsilonDecisionChampionStore.promoteArenaChampion(
          actual.checkpointRoot(), actual.candidate(), status + ": " + summary);
    }
    return new Resolution(status, summary, championDuel);
  }

  private static ChampionDuel evaluate(
      Request request, DecisionExecutionContext context, SettingsLoader config) throws IOException {
    Path resolvedChampion =
        requireImmutableCheckpoint(request.immutableChampion(), "immutable champion");
    DecisionChampionDuelSettings settings = request.duelSettings();
    long duelSequence =
        Math.addExact(
            Math.addExact(
                Math.multiplyExact((long) request.candidateIteration(), 1_000_000L),
                Math.multiplyExact((long) request.lineage(), 1_000L)),
            request.duelRound());
    var plan =
        EpsilonDecisionWallDuelEvaluator.EvaluationPlan.twoLook(
            settings.interimWallSeeds(),
            settings.maximumWallSeeds(),
            settings.alpha(),
            settings.interimAlphaFraction());
    EpsilonDecisionWallDuelEvaluator.Result result;
    try (var session =
        EpsilonDecisionDuelSession.open(request.candidate(), resolvedChampion, context, config)) {
      result =
          EpsilonDecisionWallDuelEvaluator.evaluate(
              session,
              plan,
              request.duelSeedBase() + 9_000_000L,
              duelSequence,
              0.0,
              0.0,
              config.bind(DecisionSettings.class).utilityProfile(),
              config.bind(DecisionEvalVsSettings.class));
    }
    ChampionDuel duel =
        new ChampionDuel(
            request.lineage(),
            request.candidateIteration(),
            request.duelRound(),
            request.duelSeedBase() + 9_000_000L,
            result);
    DecisionSelectedPgReportWriter.writeChampionDuel(request.reportDirectory(), duel);
    log.info(
        "Decision selected-PG champion duel complete: lineage={} candidateIteration={} "
            + "duelRound={} decision={} mode={} looks={} games={} wallSeeds={}/{} sequence={} "
            + "alpha={} decisionAlpha={} utilityProfile={} pairedUtilityDeltaMean={} "
            + "pairedUtilityDeltaLower={} pairedUtilityDeltaUpper={} pairedRankDeltaMean={} "
            + "pairedRankDeltaLower={} pairedRankDeltaUpper={} candidate={} immutableChampion={}",
        request.lineage(),
        request.candidateIteration(),
        request.duelRound(),
        result.decision(),
        result.mode(),
        result.completedLooks(),
        result.games(),
        result.wallSeeds(),
        result.plannedWallSeeds(),
        result.duelSequence(),
        result.alpha(),
        result.decisionAlpha(),
        result.utilityProfile(),
        result.pairedUtilityDeltaMean(),
        result.pairedUtilityDeltaLower(),
        result.pairedUtilityDeltaUpper(),
        result.pairedRankDeltaMean(),
        result.pairedRankDeltaLower(),
        result.pairedRankDeltaUpper(),
        request.candidate(),
        resolvedChampion);
    return duel;
  }

  private static Path requireImmutableCheckpoint(Path checkpoint, String label) throws IOException {
    Path expected = checkpoint.toAbsolutePath().normalize();
    Path resolved = EpsilonDecisionCheckpointManager.resolveExistingStrict(expected);
    if (resolved == null || !expected.equals(resolved.toAbsolutePath().normalize())) {
      throw new IOException(label + " does not resolve to the exact checkpoint: " + checkpoint);
    }
    return resolved;
  }

  /** 一つの候補判定に必要な全状態。 */
  record Request(
      Path checkpointRoot,
      Path reportDirectory,
      Path candidate,
      Path immutableChampion,
      int lineage,
      int candidateIteration,
      int duelRound,
      long duelSeedBase,
      int campaignMacros,
      int maximumMacros,
      Optional<String> guardFailure,
      DecisionChampionDuelSettings duelSettings) {

    Request {
      checkpointRoot = Objects.requireNonNull(checkpointRoot, "checkpointRoot");
      reportDirectory = Objects.requireNonNull(reportDirectory, "reportDirectory");
      candidate = Objects.requireNonNull(candidate, "candidate");
      immutableChampion = Objects.requireNonNull(immutableChampion, "immutableChampion");
      guardFailure = Objects.requireNonNull(guardFailure, "guardFailure");
      if (lineage <= 0
          || candidateIteration <= 0
          || duelRound <= 0
          || campaignMacros < 0
          || maximumMacros <= 0) {
        throw new IllegalArgumentException("selected PG duel request counters are invalid");
      }
    }
  }

  private record SavedResolution(
      Path candidate,
      DecisionSelectedPgDuelResult.Status status,
      String summary,
      ChampionDuel championDuel) {}

  /** 候補の判定と必要に応じた採用モデルとの対戦評価。 */
  record Resolution(
      DecisionSelectedPgDuelResult.Status status,
      String summary,
      Optional<ChampionDuel> championDuel) {

    Resolution {
      status = Objects.requireNonNull(status, "status");
      summary = Objects.requireNonNull(summary, "summary");
      championDuel = Objects.requireNonNull(championDuel, "championDuel");
    }
  }

  /** 固定牌山対戦評価の識別情報と全指標。 */
  record ChampionDuel(
      int lineage,
      int iteration,
      int duelRound,
      long seedBase,
      EpsilonDecisionWallDuelEvaluator.Result result) {

    ChampionDuel {
      result = Objects.requireNonNull(result, "result");
    }

    String summary() {
      return "lineage="
          + lineage
          + ",iteration="
          + iteration
          + ",duelRound="
          + duelRound
          + ",seedBase="
          + seedBase
          + ",decision="
          + result.decision()
          + ",mode="
          + result.mode()
          + ",completedLooks="
          + result.completedLooks()
          + ",games="
          + result.games()
          + ",wallSeeds="
          + result.wallSeeds()
          + "/"
          + result.plannedWallSeeds()
          + ",progressUpdates="
          + result.progressUpdates()
          + ",duelSequence="
          + result.duelSequence()
          + ",alpha="
          + result.alpha()
          + ",decisionAlpha="
          + result.decisionAlpha()
          + ",pairedRankDeltaMean="
          + result.pairedRankDeltaMean()
          + ",pairedRankDeltaLower="
          + result.pairedRankDeltaLower()
          + ",pairedRankDeltaUpper="
          + result.pairedRankDeltaUpper();
    }
  }
}
