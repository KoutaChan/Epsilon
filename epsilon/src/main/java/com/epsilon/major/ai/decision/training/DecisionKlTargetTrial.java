package com.epsilon.major.ai.decision.training;

import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.config.settings.DecisionSnapshotPoolSettings;
import com.epsilon.config.settings.DecisionTrainArenaSettings;
import com.epsilon.config.settings.DecisionTrainSettings;
import com.epsilon.config.settings.DecisionTrainSpoolSettings;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.GrpInferenceSettings;
import com.epsilon.config.settings.GrpSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.arena.DecisionAdaptiveExploration;
import com.epsilon.major.ai.decision.duel.DecisionKlTargetComparison;
import com.epsilon.major.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.major.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.major.ai.network.NetworkFactory;
import com.epsilon.major.config.settings.DecisionInferenceSettings;
import com.epsilon.major.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.util.SeedMixer;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 同じ学習器からKとK/2を比較する、昇格を伴わない再開可能な対照実験。 */
public final class DecisionKlTargetTrial {
  private static final Logger log = LoggerFactory.getLogger(DecisionKlTargetTrial.class);
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final String TRIAL_STATE = "trial-state.json";
  private static final long MACRO_SALT = 0x4b4c_5452_4941_4c4dL;
  private static final long SNAPSHOT_SALT = 0x4b4c_5452_4941_4c53L;
  private static final int OBSERVATION_MACROS = 10;

  private final SettingsLoader config;

  private DecisionKlTargetTrial(SettingsLoader config) {
    this.config = config;
  }

  public static String run(
      Path sourceRoot,
      Path trialRoot,
      int macrosPerArm,
      long trainSeed,
      long evalSeed,
      SettingsLoader config)
      throws Exception {
    return new DecisionKlTargetTrial(config)
        .runConfigured(
            sourceRoot,
            trialRoot,
            macrosPerArm,
            trainSeed,
            evalSeed,
            config.bind(DecisionSelectedPgCampaignSettings.class));
  }

  public static String run(
      Path sourceRoot, Path trialRoot, int macrosPerArm, long trainSeed, long evalSeed)
      throws Exception {
    return run(
        sourceRoot,
        trialRoot,
        macrosPerArm,
        trainSeed,
        evalSeed,
        DecisionSelectedPgCampaignSettings.defaults());
  }

  static String run(
      Path sourceRoot,
      Path trialRoot,
      int macrosPerArm,
      long trainSeed,
      long evalSeed,
      DecisionSelectedPgCampaignSettings settings)
      throws Exception {
    return new DecisionKlTargetTrial(EpsilonSettings.defaults())
        .runConfigured(sourceRoot, trialRoot, macrosPerArm, trainSeed, evalSeed, settings);
  }

  private String runConfigured(
      Path sourceRoot,
      Path trialRoot,
      int macrosPerArm,
      long trainSeed,
      long evalSeed,
      DecisionSelectedPgCampaignSettings settings)
      throws Exception {
    if (macrosPerArm <= 0 || trainSeed == evalSeed) {
      throw new IllegalArgumentException(
          "positive macros and separate train/evaluation seeds required");
    }
    Path root = trialRoot.toAbsolutePath().normalize();
    if (root.equals(sourceRoot.toAbsolutePath().normalize())) {
      throw new IllegalArgumentException("trial root must differ from the training root");
    }
    Files.createDirectories(root);
    var resolved = DecisionSelectedPgLearnerContract.resolve(settings, config);
    String contract = resolved.learnerContractId();
    Manifest manifest =
        prepare(
            sourceRoot.toAbsolutePath().normalize(), root, trainSeed, evalSeed, settings, contract);
    if (!manifest.schema().equals("kl-target-trial-v2")
        || !manifest.learnerContract().equals(contract)
        || !manifest.sourceRoot().equals(sourceRoot.toAbsolutePath().normalize().toString())
        || manifest.trainSeed() != trainSeed
        || manifest.evalSeed() != evalSeed) {
      throw new IllegalArgumentException("trial settings/seeds differ from its saved plan");
    }
    manifest.conditions().requireSame(TrialConditions.current(settings, config));
    manifest.teacher().verify(root);
    Path evaluationPlan = root.resolve("evaluation-plan.json");
    EvaluationPlan requested =
        new EvaluationPlan(macrosPerArm, config.bind(DecisionChampionDuelSettings.class));
    if (Files.exists(evaluationPlan)
        && !GSON.fromJson(Files.readString(evaluationPlan), EvaluationPlan.class)
            .equals(requested)) {
      throw new IllegalArgumentException(
          "an evaluated trial cannot change its training or wall budget; use a fresh trial");
    }
    try (var context = new DecisionExecutionContext()) {
      ArmState a = trainArm(root, "a", 1.0f, macrosPerArm, manifest, settings, resolved, context);
      ArmState b = trainArm(root, "b", 0.5f, macrosPerArm, manifest, settings, resolved, context);
      double aKl = median(a.recentUpdateKls());
      double bKl = median(b.recentUpdateKls());
      Separation separation =
          separation(
              a.acceptedMacros(),
              b.acceptedMacros(),
              aKl,
              bKl,
              settings.optimizer().actorKlControl().deadbandFactor());
      DecisionSelectedPgCampaignState.writeJson(
          root.resolve("learning-comparison.json"),
          new LearningComparison(
              a.acceptedMacros(),
              b.acceptedMacros(),
              aKl,
              bKl,
              a.lastActorLearningRate(),
              b.lastActorLearningRate(),
              separation));
      if (separation != Separation.SEPARATED) {
        return separation
            + " aMedianUpdateKl="
            + aKl
            + " bMedianUpdateKl="
            + bKl
            + "; extend the same trial before evaluation";
      }
      // この時点で学習予算と二つの評価時点を固定する。中断時は同じ検査を再実行する。
      DecisionSelectedPgCampaignState.writeJson(evaluationPlan, requested);
      Path report = root.resolve("comparison.json");
      DecisionKlTargetComparison.Result result;
      if (Files.exists(report)) {
        result = GSON.fromJson(Files.readString(report), DecisionKlTargetComparison.Result.class);
      } else {
        result =
            DecisionKlTargetComparison.evaluate(
                root.resolve("a/final"),
                root.resolve("b/final"),
                root.resolve("opponent"),
                root.resolve("evaluation"),
                evalSeed,
                requested.duel(),
                context,
                config);
        DecisionSelectedPgCampaignState.writeJson(report, result);
      }
      return "decision="
          + result.decision()
          + " difference="
          + result.meanDifference()
          + " CI=["
          + result.lower()
          + ","
          + result.upper()
          + "] report="
          + report;
    }
  }

  private Manifest prepare(
      Path sourceRoot,
      Path root,
      long trainSeed,
      long evalSeed,
      DecisionSelectedPgCampaignSettings settings,
      String contract)
      throws IOException {
    Path file = root.resolve("trial.json");
    if (Files.exists(file)) return GSON.fromJson(Files.readString(file), Manifest.class);
    Path source = EpsilonDecisionCheckpointManager.working(sourceRoot);
    EpsilonDecisionCheckpointManager.recoverWorkingCheckpoint(sourceRoot);
    var generation = EpsilonDecisionCheckpointManager.requireValidCheckpoint(source);
    var state =
        DecisionLearnerCheckpoint.requireIfPresent(
            source,
            generation,
            settings.optimizer().valueLearningRate(),
            contract,
            settings.optimizer().actorKlControl(),
            config.bind(com.epsilon.major.config.settings.DecisionSettings.class));
    if (state == null || !state.targetResolved())
      throw new IllegalStateException("complete KL calibration before a target trial");
    // 学習前状態を一度だけディスクへ固定する。各比較条件はこの同じAdamW状態と更新回数から始める。
    EpsilonDecisionCheckpointManager.copyLearnerCheckpoint(source, root.resolve("base"));
    EpsilonDecisionCheckpointManager.copyAtIteration(
        source, root.resolve("opponent"), generation.iteration);
    var pool =
        EpsilonDecisionSnapshotPool.load(
            sourceRoot, config.bind(DecisionSnapshotPoolSettings.class).max());
    pool.registerArenaChampion(
        EpsilonDecisionCheckpointManager.resolveArenaChampionStrict(sourceRoot));
    pool.save(root);
    var campaign = DecisionSelectedPgCampaignState.read(source);
    Manifest manifest =
        new Manifest(
            "kl-target-trial-v2",
            sourceRoot.toString(),
            trainSeed,
            evalSeed,
            contract,
            state.targetSamePathUpdateKl(),
            campaign == null ? null : campaign.exploration(),
            TrialConditions.current(settings, config),
            snapshotTeacher(sourceRoot, root, config.bind(GrpSettings.class).enabled()));
    DecisionSelectedPgCampaignState.writeJson(file, manifest);
    return manifest;
  }

  private ArmState trainArm(
      Path root,
      String name,
      float scale,
      int totalMacros,
      Manifest manifest,
      DecisionSelectedPgCampaignSettings settings,
      DecisionSelectedPgLearnerContract.Resolved resolved,
      DecisionExecutionContext context)
      throws Exception {
    Path arm = root.resolve(name);
    EpsilonDecisionCheckpointManager.recoverWorkingCheckpoint(arm);
    Path working = EpsilonDecisionCheckpointManager.working(arm);
    boolean resumed = Files.exists(working.resolve(TRIAL_STATE));
    Path initial = resumed ? working : root.resolve("base");
    var generation = EpsilonDecisionCheckpointManager.loadManifest(initial);
    try (var learner =
            DecisionLearner.openWorking(
                initial,
                NetworkFactory.getLearnerDevice(
                    config.bind(com.epsilon.config.settings.DeviceSettings.class)),
                settings.optimizer(),
                manifest.learnerContract(),
                context,
                config);
        var grp =
            manifest.teacher().enabled()
                ? EpsilonGrpTrainingSession.openInferenceOnly(
                    root,
                    NetworkFactory.getGrpDevices(
                            config.bind(com.epsilon.config.settings.DeviceSettings.class))
                        .primary(),
                    config)
                : EpsilonGrpTrainingSession.disabled(root.resolve("grp"))) {
      ArmState state =
          resumed
              ? GSON.fromJson(Files.readString(working.resolve(TRIAL_STATE)), ArmState.class)
              : new ArmState(
                  0,
                  generation.globalStep,
                  generation.selfPlayGames,
                  List.of(),
                  learner.actorLearningRate(),
                  manifest.exploration());
      if (totalMacros < state.acceptedMacros()) {
        throw new IllegalArgumentException("trial budget cannot be below completed macros");
      }
      if (!resumed) learner.holdTargetKl(manifest.targetKl() * scale);
      var pool =
          EpsilonDecisionSnapshotPool.load(
              root, config.bind(DecisionSnapshotPoolSettings.class).max());
      var runner =
          new DecisionSelectedPgMacroRunner(
              learner,
              grp,
              pool,
              settings,
              resolved.actorRollout(),
              resolved.opponentRollout(),
              config.bind(DecisionTrainSettings.class).sampleLimit(),
              config.bind(DecisionTrainSpoolSettings.class).keepFragments(),
              context);
      if (state.exploration() != null) runner.restoreExplorationState(state.exploration());
      saveArm(learner, arm, generation.iteration, state);
      while (state.acceptedMacros() < totalMacros) {
        int macro = state.acceptedMacros() + 1;
        long seed = SeedMixer.indexed(manifest.trainSeed(), MACRO_SALT, macro);
        long actorId =
            1L
                + (SeedMixer.indexed(manifest.trainSeed(), SNAPSHOT_SALT, macro)
                    & (Long.MAX_VALUE - 1L));
        long[][] opponents =
            pool.sampleOpponentIdsForSeats(
                Long.MAX_VALUE,
                seed,
                config
                    .bind(DecisionTrainArenaSettings.class)
                    .maximumOpponentSnapshotsPerInterval());
        var execution =
            runner.run(
                new DecisionSelectedPgMacroRunner.Request(
                    arm.resolve("spool/macro-" + macro),
                    working,
                    actorId,
                    opponents,
                    settings.gamesPerMacro(),
                    seed));
        var training = execution.training();
        if (!training.passed()) {
          // 最後の受諾一組のデータを残す。失敗した比較条件を成績比較に混ぜない。
          throw new IllegalStateException(
              "trial arm=" + name + " macro=" + macro + " " + training.reason());
        }
        var adjustment =
            learner.acceptMacroKl(training.meanActorUpdateKl(), training.optimizerSteps());
        ArrayList<Float> recent = new ArrayList<>(state.recentUpdateKls());
        recent.add(training.meanActorUpdateKl());
        if (recent.size() > OBSERVATION_MACROS) recent.removeFirst();
        state =
            new ArmState(
                macro,
                Math.addExact(state.globalStep(), training.optimizerSteps()),
                Math.addExact(state.selfPlayGames(), execution.games()),
                List.copyOf(recent),
                adjustment.nextLearningRate(),
                runner.explorationState());
        saveArm(learner, arm, generation.iteration, state);
        log.info(
            "KL target trial: arm={} macro={}/{} updateKl={} targetKl={} nextActorLr={}",
            name,
            macro,
            totalMacros,
            training.meanActorUpdateKl(),
            adjustment.targetMeanKl(),
            adjustment.nextLearningRate());
      }
      EpsilonDecisionCheckpointManager.copyAtIteration(
          working, arm.resolve("final"), generation.iteration);
      return state;
    }
  }

  private static void saveArm(DecisionLearner learner, Path arm, int iteration, ArmState state)
      throws IOException {
    EpsilonDecisionCheckpointManager.saveWorking(
        learner,
        arm,
        state.globalStep(),
        iteration,
        state.selfPlayGames(),
        dir -> Files.writeString(dir.resolve(TRIAL_STATE), GSON.toJson(state)));
  }

  /** 二つのKL目標が評価へ進めるだけ分離したか。 */
  enum Separation {
    INSUFFICIENT_ADAPTATION,
    SEPARATED,
    INSUFFICIENT_KL_SEPARATION
  }

  static Separation separation(int aMacros, int bMacros, double aKl, double bKl, double deadband) {
    if (Math.min(aMacros, bMacros) < 30) return Separation.INSUFFICIENT_ADAPTATION;
    return aKl > 1.0e-12 && bKl < aKl / deadband
        ? Separation.SEPARATED
        : Separation.INSUFFICIENT_KL_SEPARATION;
  }

  private static double median(List<Float> values) {
    if (values.isEmpty()) return 0.0;
    var sorted = values.stream().sorted().toList();
    int middle = sorted.size() / 2;
    return sorted.size() % 2 == 0
        ? ((double) sorted.get(middle - 1) + sorted.get(middle)) / 2.0
        : sorted.get(middle);
  }

  private record Manifest(
      String schema,
      String sourceRoot,
      long trainSeed,
      long evalSeed,
      String learnerContract,
      float targetKl,
      DecisionAdaptiveExploration.State exploration,
      TrialConditions conditions,
      Teacher teacher) {}

  /** KL以外の収集・バッチ・計算条件は固定し、呼び出しの追加収集・更新単位予算だけを延長する。 */
  record TrialConditions(
      int gamesPerMacro,
      int microBatchSize,
      int maximumDeviceTransitionCells,
      DecisionTrainSettings train,
      DecisionTrainArenaSettings arena,
      DecisionInferenceSettings inference,
      DecisionInferenceFusionSettings inferenceFusion,
      DecisionSnapshotPoolSettings snapshotPool,
      DeviceSettings devices,
      boolean grpEnabled,
      GrpInferenceSettings grpInference) {
    static TrialConditions current(
        DecisionSelectedPgCampaignSettings settings, SettingsLoader config) {
      return new TrialConditions(
          settings.gamesPerMacro(),
          settings.microBatchSize(),
          settings.maximumDeviceTransitionCells(),
          config.bind(DecisionTrainSettings.class),
          config.bind(DecisionTrainArenaSettings.class),
          config.bind(DecisionInferenceSettings.class),
          config.bind(DecisionInferenceFusionSettings.class),
          config.bind(DecisionSnapshotPoolSettings.class),
          config.bind(DeviceSettings.class),
          config.bind(GrpSettings.class).enabled(),
          config.bind(GrpInferenceSettings.class));
    }

    void requireSame(TrialConditions requested) {
      if (!equals(requested)) {
        throw new IllegalArgumentException(
            "trial collection/batch/precision/GRP settings differ from its saved conditions");
      }
    }
  }

  /** 固定教師モデルの内容を記録する。入力元のlatest更新には追従しない。 */
  record Teacher(boolean enabled, String sha256) {
    void verify(Path root) throws IOException {
      if (enabled
          && !sha256.equals(EpsilonGrpCheckpointManager.checkpointSha256(root.resolve("grp")))) {
        throw new IOException("fixed trial GRP checkpoint differs from its saved SHA-256");
      }
    }
  }

  static Teacher snapshotTeacher(Path sourceRoot, Path trialRoot, boolean enabled)
      throws IOException {
    if (!enabled) return new Teacher(false, null);
    Path target = trialRoot.resolve("grp");
    // 初回準備中の中断から再開するときも、先に固定した教師モデルを引き継ぐ。
    if (Files.exists(target)) {
      return new Teacher(true, EpsilonGrpCheckpointManager.checkpointSha256(target));
    }
    Path source = EpsilonGrpCheckpointManager.resolveExisting(sourceRoot.resolve("grp"));
    if (source == null)
      throw new IOException("GRP checkpoint not found in " + sourceRoot.resolve("grp"));
    String expectedSha = EpsilonGrpCheckpointManager.checkpointSha256(source);
    Path temporary = Files.createTempDirectory(trialRoot, ".grp-");
    try {
      // {@code optimizer.state}は推論では使わないがチェックポイントマニフェストの必須保存物なので含める。
      try (var files = Files.list(source)) {
        for (Path file : files.filter(Files::isRegularFile).toList()) {
          Files.copy(file, temporary.resolve(file.getFileName()));
        }
      }
      if (!expectedSha.equals(EpsilonGrpCheckpointManager.checkpointSha256(temporary))) {
        throw new IOException("GRP checkpoint changed while preparing the fixed trial teacher");
      }
      try {
        Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
      } catch (AtomicMoveNotSupportedException unsupported) {
        Files.move(temporary, target);
      }
      return new Teacher(true, expectedSha);
    } finally {
      if (Files.exists(temporary)) {
        try (var files = Files.list(temporary)) {
          for (Path file : files.toList()) Files.delete(file);
        }
        Files.delete(temporary);
      }
    }
  }

  private record ArmState(
      int acceptedMacros,
      int globalStep,
      int selfPlayGames,
      List<Float> recentUpdateKls,
      float lastActorLearningRate,
      DecisionAdaptiveExploration.State exploration) {}

  private record EvaluationPlan(int macrosPerArm, DecisionChampionDuelSettings duel) {}

  private record LearningComparison(
      int aMacros,
      int bMacros,
      double aMedianUpdateKl,
      double bMedianUpdateKl,
      float aActorLearningRate,
      float bActorLearningRate,
      Separation separation) {}
}
