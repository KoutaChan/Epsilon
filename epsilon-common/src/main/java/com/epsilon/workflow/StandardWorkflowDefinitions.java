package com.epsilon.workflow;

/** 全モデル系列に共通するCLIのコマンド名・引数の数・使用方法を定義する。コマンドの登録順と実行処理は各系列で指定する。 */
public final class StandardWorkflowDefinitions {

  public static final WorkflowDefinition INIT_DECISION_CHECKPOINT =
      WorkflowDefinition.range(
          "init-decision-checkpoint",
          WorkflowKind.CHECKPOINT,
          "init-decision-checkpoint [checkpointDir]",
          0,
          1);

  public static final WorkflowDefinition PRETRAIN_GRP_LOGS =
      WorkflowDefinition.range(
          "pretrain-grp-logs",
          WorkflowKind.TRAINING,
          "pretrain-grp-logs [checkpointDir] [logDir] [epochs] [maxFiles=0]",
          0,
          4);

  public static final WorkflowDefinition PRETRAIN_BELIEF_LOGS =
      WorkflowDefinition.range(
          "pretrain-belief-logs",
          WorkflowKind.TRAINING,
          "pretrain-belief-logs [checkpointDir] [logDir] [epochs] [maxFiles=0]",
          0,
          4);

  public static final WorkflowDefinition CALIBRATE_BELIEF_LOGS =
      WorkflowDefinition.range(
          "calibrate-belief-logs",
          WorkflowKind.VALIDATION,
          "calibrate-belief-logs [checkpointDir] [logDir] [maxFiles=0]" + " [maxSamples=20000]",
          0,
          4);

  public static final WorkflowDefinition PRETRAIN_DECISION_LOGS =
      WorkflowDefinition.range(
          "pretrain-decision-logs",
          WorkflowKind.TRAINING,
          "pretrain-decision-logs [checkpointDir] [logDir] [epochs] [maxFiles=0]",
          0,
          4);

  public static final WorkflowDefinition VALIDATE_DECISION_PRETRAIN =
      WorkflowDefinition.range(
          "validate-decision-pretrain",
          WorkflowKind.VALIDATION,
          "validate-decision-pretrain <grpCheckpointRoot> <checkpoint> <logDir>" + " [maxFiles=0]",
          3,
          4);

  public static final WorkflowDefinition BENCHMARK_DECISION_PRETRAIN =
      WorkflowDefinition.range(
          "benchmark-decision-pretrain",
          WorkflowKind.BENCHMARK,
          "benchmark-decision-pretrain [optimizerBatchRows=1024]"
              + " [maxDeviceBatchRows=512] [warmupSteps=3] [measuredSteps=10]"
              + " [legalActionBucket=16] [actionTransitionBucket=8]"
              + " [tensorTransfer=DIRECT_BUFFER]",
          0,
          7);

  public static final WorkflowDefinition BENCHMARK_DECISION_SELFPLAY =
      WorkflowDefinition.range(
          "benchmark-decision-selfplay",
          WorkflowKind.BENCHMARK,
          "benchmark-decision-selfplay [checkpointDir] [games=2048] [seedBase=79000000]"
              + " [scratchDir=checkpointDir/benchmark-inflight] [warmupGames=2048]",
          0,
          5);

  public static final WorkflowDefinition BENCHMARK_DECISION_DUEL =
      WorkflowDefinition.range(
          "benchmark-decision-duel",
          WorkflowKind.BENCHMARK,
          "benchmark-decision-duel <candidateCheckpoint> <parentCheckpoint>"
              + " [warmupWallSeeds=64] [measuredWallSeeds=512] [seedBase=98200000]"
              + " [gamesInFlight=settings]",
          2,
          6);

  public static final WorkflowDefinition BENCHMARK_DECISION_INFERENCE_REPLAY =
      WorkflowDefinition.range(
          "benchmark-decision-inference-replay",
          WorkflowKind.BENCHMARK,
          "benchmark-decision-inference-replay <checkpoint> <report.json> <corpus.bin>"
              + " <CREATE|LOAD> [seedBase=202608310001] [repetitions=3]"
              + " [targetRowsPerSecond=181000]"
              + " [batchRows=512,1024,1536,2048,3072,4096]",
          4,
          8);

  public static final WorkflowDefinition AUDIT_DECISION_VALUE_GRP =
      WorkflowDefinition.atLeast(
          "audit-decision-value-grp",
          WorkflowKind.AUDIT,
          "audit-decision-value-grp <outputFile> <inputFileManifest> <grpCheckpointRoot>"
              + " <validationFraction> <maxValidationFiles> <baselineLabel=checkpoint>"
              + " <candidateLabel=checkpoint>...",
          7);

  public static final WorkflowDefinition AUDIT_DECISION_PAIRED_TRAJECTORY =
      WorkflowDefinition.exact(
          "audit-decision-paired-trajectory",
          WorkflowKind.AUDIT,
          "audit-decision-paired-trajectory <reportFile> <traceFile> <scratchDir>"
              + " <grpCheckpointRoot> <parentCheckpoint> <candidateCheckpoint>"
              + " <gamesPerSource> <seedBase>",
          8);

  public static final WorkflowDefinition AUDIT_DECISION_COUNTERFACTUAL_CALLS =
      WorkflowDefinition.exact(
          "audit-decision-counterfactual-calls",
          WorkflowKind.AUDIT,
          "audit-decision-counterfactual-calls <reportFile> <traceFile>"
              + " <grpCheckpointRoot> <parentCheckpoint> <candidateCheckpoint> <baseGames>"
              + " <seedBase> <gamesInFlight>",
          8);

  public static final WorkflowDefinition PROBE_DECISION_COUNTERFACTUAL_CALLS =
      WorkflowDefinition.exact(
          "probe-decision-counterfactual-calls",
          WorkflowKind.PROBE,
          "probe-decision-counterfactual-calls <reportFile> <predictionTraceFile>"
              + " <grpCheckpointRoot> <parentCheckpoint> <candidateCheckpoint> <baseGames>"
              + " <seedBase> <gamesInFlight>",
          8);

  public static final WorkflowDefinition TRAIN_DECISION =
      WorkflowDefinition.range(
          "train-decision", WorkflowKind.TRAINING, "train-decision [checkpointDir]", 0, 1);

  public static final WorkflowDefinition PLAN_KL_DECISION =
      WorkflowDefinition.exact(
          "plan-kl-decision",
          WorkflowKind.TRAINING,
          "plan-kl-decision <checkpointRoot> <endActorOptimizerStep> <endTargetKl>",
          3);

  public static final WorkflowDefinition TRIAL_KL_DECISION =
      WorkflowDefinition.exact(
          "trial-kl-decision",
          WorkflowKind.TRAINING,
          "trial-kl-decision <sourceRoot> <trialRoot> <totalMacrosPerArm> <trainSeed> <evalSeed>",
          5);

  public static final WorkflowDefinition EVAL_DECISION =
      WorkflowDefinition.range(
          "eval-decision",
          WorkflowKind.EVALUATION,
          "eval-decision [checkpointDir] [games]"
              + " [tenhouLogDir=checkpointDir/eval-tenhou-logs]",
          0,
          3);

  public static final WorkflowDefinition EVAL_VS =
      WorkflowDefinition.range(
          "eval-vs",
          WorkflowKind.EVALUATION,
          "eval-vs [candidateCheckpointDir] [opponentCheckpointDir] [games]"
              + " [outFile=candidate/eval-vs.txt]",
          0,
          4);

  public static final WorkflowDefinition EVAL_VS_WALL =
      WorkflowDefinition.exact(
          "eval-vs-wall",
          WorkflowKind.EVALUATION,
          "eval-vs-wall <reportFile> <candidateCheckpoint> <parentCheckpoint>"
              + " <wallSeeds> <seedBase> <duelSequence> <alpha> <promotionMargin>"
              + " <harmfulMargin>",
          9);

  public static final WorkflowDefinition PROMOTE_DECISION_EXTERNAL_GATE =
      WorkflowDefinition.exact(
          "promote-decision-external-gate",
          WorkflowKind.PROMOTION,
          "promote-decision-external-gate <resultFile> <checkpointRoot>"
              + " <candidateCheckpoint> <fixedWallReport>",
          4);

  public static final WorkflowDefinition PLAY_RIICHI =
      WorkflowDefinition.exact("play-riichi", WorkflowKind.PLAY, "play-riichi", 0);

  public static final WorkflowDefinition BENCHMARK =
      WorkflowDefinition.range("benchmark", WorkflowKind.BENCHMARK, "benchmark [games]", 0, 1);

  private StandardWorkflowDefinitions() {}
}
