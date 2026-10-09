package com.epsilon.cfr.training;

/** 一度の原子的な更新で公開する実行状態。参照先はすべて、世代ごとに保存した変更不可のデータとする。 */
public record CfrRunState(
    int version,
    String solveId,
    int solveIndex,
    long iteration,
    Phase phase,
    int regretUpdate,
    int averageUpdate,
    String regretCheckpoint,
    String averageCheckpoint,
    String frozenCheckpoint,
    long replayGeneration,
    String boundaryRoot,
    String boundaryHash,
    String configHash) {
  public enum Phase {
    READY,
    COLLECTING,
    SEALED,
    REPLAY_COMMITTED,
    TRAINING_REGRET,
    TRAINING_AVERAGE,
    TRAINED,
    COMPLETE
  }

  public CfrRunState {
    if (version != 2
        || solveId == null
        || solveId.isBlank()
        || solveIndex < 0
        || iteration < 1
        || phase == null
        || regretUpdate < 0
        || averageUpdate < 0
        || replayGeneration < 0
        || regretCheckpoint == null
        || averageCheckpoint == null
        || configHash == null
        || boundaryHash == null
        || (phase != Phase.READY && phase != Phase.COMPLETE && frozenCheckpoint == null))
      throw new IllegalArgumentException("Invalid LCFR run state");
  }

  public CfrRunState progress(
      Phase next,
      int regret,
      int average,
      String regretPath,
      String averagePath,
      String frozenPath,
      long replay) {
    return new CfrRunState(
        version,
        solveId,
        solveIndex,
        iteration,
        next,
        regret,
        average,
        regretPath,
        averagePath,
        frozenPath,
        replay,
        boundaryRoot,
        boundaryHash,
        configHash);
  }

  public CfrRunState nextIteration() {
    return new CfrRunState(
        version,
        solveId,
        solveIndex,
        iteration + 1,
        Phase.READY,
        0,
        0,
        regretCheckpoint,
        averageCheckpoint,
        null,
        replayGeneration,
        boundaryRoot,
        boundaryHash,
        configHash);
  }
}
