package com.epsilon.major.ai.decision.training;

import com.epsilon.util.FormatUtils;
import com.epsilon.util.SeedMixer;
import java.nio.file.Path;
import java.util.Objects;
import java.util.UUID;

/** 選択行動の方策勾配学習実行の識別情報、乱数シード階層、保存パスを一か所で導出する。 */
record DecisionSelectedPgRunContext(
    Path checkpointRoot, String runId, long seedBase, SeedMode seedMode) {

  /** 実行乱数シードの決定方法。 */
  enum SeedMode {
    UUID_DERIVED,
    FIXED
  }

  private static final long RUN_SEED_SALT = 0x5255_4e5f_5345_4544L;
  private static final long LINEAGE_SEED_SALT = 0x4252_414e_4348_5344L;
  private static final long DUEL_ROUND_SEED_SALT = 0x4455_454c_524f_554eL;
  private static final long ACTOR_SNAPSHOT_SALT = 0x4143_544f_525f_4944L;

  DecisionSelectedPgRunContext {
    checkpointRoot =
        Objects.requireNonNull(checkpointRoot, "checkpointRoot").toAbsolutePath().normalize();
    if (runId == null || runId.isBlank()) {
      throw new IllegalArgumentException("selected PG runId must not be blank");
    }
    Objects.requireNonNull(seedMode, "seedMode");
  }

  static DecisionSelectedPgRunContext create(
      Path checkpointRoot, int initialIteration, long configuredSeedBase) {
    return create(checkpointRoot, initialIteration, configuredSeedBase, UUID.randomUUID());
  }

  static DecisionSelectedPgRunContext create(
      Path checkpointRoot, int initialIteration, long configuredSeedBase, UUID runUuid) {
    UUID actualRunUuid = Objects.requireNonNull(runUuid, "runUuid");
    long resolvedSeed = resolveSeedBase(actualRunUuid, configuredSeedBase);
    return new DecisionSelectedPgRunContext(
        checkpointRoot,
        "selected-pg-" + actualRunUuid + "-from-" + initialIteration,
        resolvedSeed,
        configuredSeedBase == 0L ? SeedMode.UUID_DERIVED : SeedMode.FIXED);
  }

  static long resolveSeedBase(UUID runUuid, long configuredSeedBase) {
    UUID actualRunUuid = Objects.requireNonNull(runUuid, "runUuid");
    return configuredSeedBase != 0L
        ? configuredSeedBase
        : SeedMixer.indexed(
            actualRunUuid.getMostSignificantBits(),
            RUN_SEED_SALT,
            actualRunUuid.getLeastSignificantBits());
  }

  long lineageSeed(int lineage) {
    requirePositive(lineage, "lineage");
    return SeedMixer.indexed(seedBase, LINEAGE_SEED_SALT, lineage);
  }

  long duelSeed(long lineageSeed, int duelRound) {
    requirePositive(duelRound, "duelRound");
    return duelRound == 1
        ? lineageSeed
        : SeedMixer.indexed(lineageSeed, DUEL_ROUND_SEED_SALT, duelRound);
  }

  long actorSnapshotId(long lineageSeed, int lineageMacro) {
    requirePositive(lineageMacro, "lineageMacro");
    long mixed = SeedMixer.indexed(lineageSeed, ACTOR_SNAPSHOT_SALT, lineageMacro);
    return 1L + (mixed & (Long.MAX_VALUE - 1L));
  }

  Path lineageDirectory(int lineage, int candidateIteration) {
    requirePositive(lineage, "lineage");
    requirePositive(candidateIteration, "candidateIteration");
    return checkpointRoot
        .resolve("selected-pg")
        .resolve("runs")
        .resolve(runId)
        .resolve(lineageName(lineage, candidateIteration));
  }

  Path duelDirectory(Path lineageDirectory, int duelRound) {
    requirePositive(duelRound, "duelRound");
    return lineageDirectory.resolve("duel_round_" + FormatUtils.zeroPad(duelRound, 2));
  }

  Path spoolDirectory(
      int lineage,
      int candidateIteration,
      int duelRound,
      int campaignMacro,
      String spoolDirectory) {
    requirePositive(duelRound, "duelRound");
    requirePositive(campaignMacro, "campaignMacro");
    Path configured = Path.of(spoolDirectory);
    Path root = configured.isAbsolute() ? configured : checkpointRoot.resolve(configured);
    return root.resolve("selected-pg")
        .resolve(runId)
        .resolve(lineageName(lineage, candidateIteration))
        .resolve("duel_round_" + FormatUtils.zeroPad(duelRound, 2))
        .resolve("macro_" + FormatUtils.zeroPad(campaignMacro, 3));
  }

  String candidateRunId(int lineage, int candidateIteration, int duelRound) {
    requirePositive(lineage, "lineage");
    requirePositive(candidateIteration, "candidateIteration");
    requirePositive(duelRound, "duelRound");
    return runId
        + "-lineage"
        + FormatUtils.zeroPad(lineage, 3)
        + "-iteration"
        + FormatUtils.zeroPad(candidateIteration, 5)
        + "-duel"
        + FormatUtils.zeroPad(duelRound, 2);
  }

  private static String lineageName(int lineage, int candidateIteration) {
    return "lineage_"
        + FormatUtils.zeroPad(lineage, 3)
        + "_iteration_"
        + FormatUtils.zeroPad(candidateIteration, 5);
  }

  private static void requirePositive(int value, String label) {
    if (value <= 0) {
      throw new IllegalArgumentException(label + " must be positive");
    }
  }
}
