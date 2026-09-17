package com.epsilon.nano.ai.decision.probe;

import ai.djl.Device;
import com.epsilon.ai.grp.EpsilonGrpCheckpointBundle;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.decision.EpsilonDecisionConstants;
import com.epsilon.nano.ai.decision.audit.EpsilonDecisionCallCounterfactualAudit;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointBundle;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.ai.network.NetworkDevices;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.DecisionSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 見送りとチー・ポンを比較するデータを対局ごとに分割し、固定したモデル表現から分岐結果を予測できるか調べる。
 *
 * <p>比較元・候補・GRP
 * のチェックポイントは読み取り専用とし、一時的なリッジ回帰モデルだけを学習する。外側は5分割の交差検証とし、各回の学習データの一部を検証用に分け、テストデータを使わずに正則化の強さを選ぶ。候補モデルが持つ行動後の状態表現の差を加えることで、判断時点の状態だけを使う予測器より平均二乗誤差が改善するかを主に調べる。
 */
public final class EpsilonDecisionCallCounterfactualOfflineProbe {

  private static final Logger log =
      LoggerFactory.getLogger(EpsilonDecisionCallCounterfactualOfflineProbe.class);
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final Gson LINE_GSON = new Gson();
  private static final String SCHEMA = "epsilon-decision-call-counterfactual-offline-probe-v3";
  private static final int FOLDS = 5;
  private static final int PROJECTION_PER_TOWER = 48;
  private static final long FOLD_SALT = 0x243F6A8885A308D3L;
  private static final long POLICY_PROJECTION_SALT = 0x13198A2E03707344L;
  private static final long VALUE_PROJECTION_SALT = 0xA4093822299F31D0L;
  private static final long ACTION_INTERACTION_SALT = 0x082EFA98EC4E6C89L;
  private static final double[] RIDGE_LAMBDAS = {1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0};
  private static final double Z_95 = 1.96;
  private static final double MIN_SCALE = 1.0e-8;

  private static final int TYPE_FEATURES = Action.Type.values().length;
  private static final int TILE_SELECTION_FEATURES = Action.TileSelection.values().length;
  private static final int TILE_FEATURES = EpsilonDecisionConstants.TILE_TYPES;
  private static final int CHI_POSITION_FEATURES = 3;
  private static final int CHI_SEQUENCE_FEATURES = 3 * 7;
  private static final int ACTION_DESCRIPTOR_SIZE =
      TYPE_FEATURES
          + TILE_SELECTION_FEATURES
          + TILE_FEATURES
          + CHI_POSITION_FEATURES
          + CHI_SEQUENCE_FEATURES;

  private EpsilonDecisionCallCounterfactualOfflineProbe() {}

  /**
   * 対応をそろえた行動を変えた場合の比較データセットを生成し、固定した表現上のリッジ回帰検証用の予測器を交差検証評価する。
   *
   * @param reportFile 集約JSON レポートの新規出力先
   * @param predictionTraceFile 学習に使わなかった分割群の予測トレースの新規出力先
   * @param grpDecisionCheckpointDir 重みを固定したGRPとDecision チェックポイントを含むルートディレクトリ
   * @param parentCheckpointDir 比較元 Decision チェックポイント
   * @param candidateCheckpointDir 固定した特徴量を評価する候補チェックポイント
   * @param baseGames 行動の不一致比較局面を探索する新しく生成した半荘数
   * @param seedBase 基準対局乱数シードの先頭
   * @param gamesInFlight 同時進行させる基準対局数
   * @return データセット、交差検証指標、事前基準判定をまとめたレポート
   * @throws Exception チェックポイント読込、対局生成、検証用の予測器計算、またはファイル出力に失敗した場合
   */
  public static ProbeReport run(
      Path reportFile,
      Path predictionTraceFile,
      Path grpDecisionCheckpointDir,
      Path parentCheckpointDir,
      Path candidateCheckpointDir,
      int baseGames,
      long seedBase,
      int gamesInFlight)
      throws Exception {
    return run(
        reportFile,
        predictionTraceFile,
        grpDecisionCheckpointDir,
        parentCheckpointDir,
        candidateCheckpointDir,
        baseGames,
        seedBase,
        gamesInFlight,
        EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static ProbeReport run(
      Path reportFile,
      Path predictionTraceFile,
      Path grpDecisionCheckpointDir,
      Path parentCheckpointDir,
      Path candidateCheckpointDir,
      int baseGames,
      long seedBase,
      int gamesInFlight,
      SettingsLoader snapshot)
      throws Exception {
    Objects.requireNonNull(reportFile, "reportFile");
    Objects.requireNonNull(predictionTraceFile, "predictionTraceFile");
    Objects.requireNonNull(grpDecisionCheckpointDir, "grpDecisionCheckpointDir");
    Objects.requireNonNull(parentCheckpointDir, "parentCheckpointDir");
    Objects.requireNonNull(candidateCheckpointDir, "candidateCheckpointDir");
    if (baseGames <= 0) {
      throw new IllegalArgumentException("baseGames must be positive");
    }
    if (gamesInFlight <= 0) {
      throw new IllegalArgumentException("gamesInFlight must be positive");
    }

    Path report =
        EpsilonDecisionCallCounterfactualAudit.newOutputFile(reportFile, "probe reportFile");
    Path predictionTrace =
        EpsilonDecisionCallCounterfactualAudit.newOutputFile(
            predictionTraceFile, "probe predictionTraceFile");
    if (report.equals(predictionTrace)) {
      throw new IOException("probe reportFile and predictionTraceFile must differ");
    }
    Path parent =
        EpsilonDecisionCallCounterfactualAudit.requireCheckpoint(parentCheckpointDir, "parent");
    Path candidate =
        EpsilonDecisionCallCounterfactualAudit.requireCheckpoint(
            candidateCheckpointDir, "candidate");
    if (parent.equals(candidate)) {
      throw new IOException("parent and candidate checkpoints must differ");
    }
    EpsilonDecisionCallCounterfactualAudit.requireOutputOutsideCheckpoint(
        report, parent, candidate);
    EpsilonDecisionCallCounterfactualAudit.requireOutputOutsideCheckpoint(
        predictionTrace, parent, candidate);

    Path grpRoot = grpDecisionCheckpointDir.toRealPath();
    Path grpCheckpoint = EpsilonGrpCheckpointManager.resolveExisting(grpRoot.resolve("grp"));
    if (grpCheckpoint == null) {
      throw new IOException("Frozen GRP checkpoint not found below " + grpRoot.resolve("grp"));
    }
    grpCheckpoint = grpCheckpoint.toRealPath();

    EpsilonDecisionCheckpointBundle parentManifest =
        EpsilonDecisionCheckpointManager.loadManifest(parent);
    EpsilonDecisionCheckpointBundle candidateManifest =
        EpsilonDecisionCheckpointManager.loadManifest(candidate);
    EpsilonGrpCheckpointBundle grpManifest =
        EpsilonGrpCheckpointManager.loadManifest(grpCheckpoint);

    NetworkDevices inferenceDevices =
        NetworkFactory.getInferenceDevices(snapshot.bind(DeviceSettings.class));
    Device parentDevice = inferenceDevices.primary();
    Device candidateDevice = inferenceDevices.get(Math.min(1, inferenceDevices.size() - 1));

    EpsilonDecisionCallCounterfactualAudit.ProbeDataset dataset;
    FrozenFeatures frozen;
    String parentDevices;
    String candidateDevices;
    try (var context = new DecisionExecutionContext();
        EpsilonDecisionEvaluatorFactory.Handle parentHandle =
            EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                parent, NetworkDevices.of(parentDevice), context, snapshot);
        EpsilonDecisionEvaluatorFactory.Handle candidateHandle =
            EpsilonDecisionEvaluatorFactory.openCheckpointEvaluator(
                candidate, NetworkDevices.of(candidateDevice), context, snapshot)) {
      parentDevices = parentHandle.devices();
      candidateDevices = candidateHandle.devices();
      dataset =
          EpsilonDecisionCallCounterfactualAudit.evaluateProbeDataset(
              parentHandle.evaluator(),
              candidateHandle.evaluator(),
              baseGames,
              seedBase,
              gamesInFlight,
              snapshot);
      frozen =
          extractFrozenFeatures(candidateHandle.singleServerForDiagnostics(), dataset.samples());
    }

    requireUniqueGames(dataset.samples());
    FeatureDataset features = buildFeatureDataset(dataset.samples(), frozen);
    int[] folds = folds(features.wallSeeds());
    LinkedHashMap<String, CrossFitResult> fitted = new LinkedHashMap<>();
    fitted.put(
        "root-state", crossFit("root-state", features.rootState(), features.target(), folds));
    fitted.put(
        "root-state+action",
        crossFit("root-state+action", features.rootStateAction(), features.target(), folds));
    fitted.put(
        "afterstate-delta",
        crossFit("afterstate-delta", features.afterStateDelta(), features.target(), folds));
    fitted.put("combined", crossFit("combined", features.combined(), features.target(), folds));

    double[] crossFitMeanPrediction = crossFitMeans(features.target(), folds);
    Baselines baselines = baselines(features, frozen, crossFitMeanPrediction);
    LinkedHashMap<String, ModelMetrics> modelMetrics = new LinkedHashMap<>();
    for (Map.Entry<String, CrossFitResult> entry : fitted.entrySet()) {
      modelMetrics.put(
          entry.getKey(),
          modelMetrics(
              entry.getKey(), entry.getValue(), features.target(), crossFitMeanPrediction));
    }
    LinkedHashMap<String, Comparison> comparisons = new LinkedHashMap<>();
    double[] rootStatePrediction = fitted.get("root-state").predictions();
    comparisons.put(
        "root-state-vs-crossfit-mean",
        lossComparison(
            "root-state-vs-crossfit-mean",
            crossFitMeanPrediction,
            rootStatePrediction,
            features.target()));
    comparisons.put(
        "root-state+action-vs-root-state",
        lossComparison(
            "root-state+action-vs-root-state",
            rootStatePrediction,
            fitted.get("root-state+action").predictions(),
            features.target()));
    comparisons.put(
        "afterstate-delta-vs-root-state",
        lossComparison(
            "afterstate-delta-vs-root-state",
            rootStatePrediction,
            fitted.get("afterstate-delta").predictions(),
            features.target()));
    comparisons.put(
        "combined-vs-root-state",
        lossComparison(
            "combined-vs-root-state",
            rootStatePrediction,
            fitted.get("combined").predictions(),
            features.target()));

    Verdict verdict = verdict(modelMetrics, comparisons);
    String predictionLines =
        predictionTraceLines(dataset.samples(), features, frozen, folds, fitted);
    Files.writeString(
        predictionTrace,
        predictionLines,
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE_NEW,
        StandardOpenOption.WRITE);
    ProbeReport result =
        new ProbeReport(
            SCHEMA,
            Instant.now().toString(),
            true,
            true,
            false,
            new Protocol(
                "one reservoir-sampled parent-PASS/candidate-CHI-or-PON root per fresh"
                    + " parent-greedy hanchan",
                "forced PASS and forced selected CHI/PON with parent-greedy continuation",
                "candidate policy/value towers frozen; only ephemeral ridge heads are fitted",
                "five outer folds grouped by wall seed; validation=(outer+1)%5; remaining three"
                    + " folds train",
                "ridge lambda selected on validation only; every root receives exactly one"
                    + " out-of-fold prediction",
                "primary: afterstate-delta versus root-state paired squared-error improvement"),
            baseGames,
            seedBase,
            gamesInFlight,
            snapshot.bind(DecisionSettings.class).utilityProfile().name(),
            EpsilonDecisionCallCounterfactualAudit.checkpointIdentity(
                parent, parentManifest, parentDevices),
            EpsilonDecisionCallCounterfactualAudit.checkpointIdentity(
                candidate, candidateManifest, candidateDevices),
            new GrpIdentity(
                grpRoot.toString(),
                grpCheckpoint.toString(),
                grpManifest.iteration,
                "metadata only; no GRP mutation or inference"),
            dataset.search(),
            dataset.branchMetrics(),
            new DatasetSummary(
                dataset.samples().size(),
                foldCounts(folds),
                frozen.hiddenSize(),
                features.rootState()[0].length,
                features.rootStateAction()[0].length,
                features.afterStateDelta()[0].length,
                features.combined()[0].length,
                nonZeroCount(features.target())),
            new FeatureConfiguration(
                "candidate frozen policyState + valueState",
                "deterministic signed CountSketch",
                PROJECTION_PER_TOWER,
                ACTION_DESCRIPTOR_SIZE,
                "root projected state + semantic call descriptor + deterministic hashed"
                    + " root-by-action interaction + frozen policy/value scalars",
                "projected candidate (CALL afterstate - PASS afterstate) + rank-value delta",
                FOLDS,
                RIDGE_LAMBDAS.clone()),
            baselines,
            Map.copyOf(modelMetrics),
            Map.copyOf(comparisons),
            verdict,
            predictionTrace.toString());
    Files.writeString(
        report,
        GSON.toJson(result) + System.lineSeparator(),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE_NEW,
        StandardOpenOption.WRITE);
    log.info(
        "Call counterfactual offline probe complete: roots={} primary={} policy={} report={}"
            + " trace={}",
        dataset.samples().size(),
        verdict.afterStateRepresentationSignal(),
        verdict.afterStatePolicySignal(),
        report,
        predictionTrace);
    return result;
  }

  private static FrozenFeatures extractFrozenFeatures(
      EpsilonDecisionInferenceServer server,
      List<EpsilonDecisionCallCounterfactualAudit.ProbeSample> samples) {
    ArrayList<DecisionHostBatch> roots = new ArrayList<>(samples.size());
    ArrayList<DecisionHostBatch> passAfterStates = new ArrayList<>(samples.size());
    ArrayList<DecisionHostBatch> callAfterStates = new ArrayList<>(samples.size());
    for (EpsilonDecisionCallCounterfactualAudit.ProbeSample sample : samples) {
      roots.add(sample.rootInput());
      passAfterStates.add(sample.passAfterStateInput());
      callAfterStates.add(sample.callAfterStateInput());
    }
    FrozenBatch root = buildFrozenBatch(server, roots);
    FrozenBatch pass = buildFrozenBatch(server, passAfterStates);
    FrozenBatch call = buildFrozenBatch(server, callAfterStates);
    if (root.hiddenSize() != pass.hiddenSize() || root.hiddenSize() != call.hiddenSize()) {
      throw new IllegalStateException("Frozen embedding hidden sizes differ across probe inputs");
    }
    for (int row = 0; row < samples.size(); row++) {
      double recorded = samples.get(row).outcome().candidateExpectedUtility();
      if (Math.abs(recorded - root.expectedUtility()[row]) > 1.0e-4) {
        throw new IllegalStateException(
            "Candidate root value replay mismatch at game="
                + samples.get(row).outcome().gameIndex()
                + " recorded="
                + recorded
                + " replay="
                + root.expectedUtility()[row]);
      }
    }
    return new FrozenFeatures(root, pass, call, root.hiddenSize());
  }

  private static FrozenBatch buildFrozenBatch(
      EpsilonDecisionInferenceServer server, List<DecisionHostBatch> inputRows) {
    int rowCount = inputRows.size();
    ArrayList<float[]> policyStateEmbeddingParts = new ArrayList<>();
    ArrayList<float[]> valueStateEmbeddingParts = new ArrayList<>();
    double[] expectedUtilities = new double[rowCount];
    int predictionOffset = 0;
    for (int rowStart = 0; rowStart < rowCount; ) {
      int rowEnd = rowStart + 1;
      while (rowEnd < rowCount
          && rowEnd - rowStart < server.preferredBatchSize()
          && inputRows.get(rowEnd).bucket().equals(inputRows.get(rowStart).bucket())) {
        rowEnd++;
      }
      EpsilonDecisionInferenceServer.DiagnosticBatchEvaluation evaluation =
          server.evaluateDiagnosticBatch(
              DecisionHostBatch.concatenate(inputRows.subList(rowStart, rowEnd)));
      policyStateEmbeddingParts.add(evaluation.policyStateEmbeddings());
      valueStateEmbeddingParts.add(evaluation.valueStateEmbeddings());
      for (EpsilonDecisionInferenceServer.Prediction prediction : evaluation.predictions()) {
        expectedUtilities[predictionOffset++] = prediction.valueUtility();
      }
      rowStart = rowEnd;
    }
    float[] flatPolicyStateEmbeddings = concatenate(policyStateEmbeddingParts);
    float[] flatValueStateEmbeddings = concatenate(valueStateEmbeddingParts);
    if (rowCount == 0
        || flatPolicyStateEmbeddings.length % rowCount != 0
        || flatValueStateEmbeddings.length != flatPolicyStateEmbeddings.length
        || predictionOffset != rowCount) {
      throw new IllegalStateException("Invalid frozen embedding batch shape");
    }
    int hiddenSize = flatPolicyStateEmbeddings.length / rowCount;
    float[][] policyStateEmbeddings = splitRows(flatPolicyStateEmbeddings, rowCount, hiddenSize);
    float[][] valueStateEmbeddings = splitRows(flatValueStateEmbeddings, rowCount, hiddenSize);
    return new FrozenBatch(
        policyStateEmbeddings, valueStateEmbeddings, expectedUtilities, hiddenSize);
  }

  private static float[] concatenate(List<float[]> parts) {
    int length = parts.stream().mapToInt(part -> part.length).sum();
    float[] result = new float[length];
    int offset = 0;
    for (float[] part : parts) {
      System.arraycopy(part, 0, result, offset, part.length);
      offset += part.length;
    }
    return result;
  }

  private static float[][] splitRows(float[] flat, int rows, int columns) {
    float[][] out = new float[rows][columns];
    for (int row = 0; row < rows; row++) {
      System.arraycopy(flat, row * columns, out[row], 0, columns);
    }
    return out;
  }

  private static FeatureDataset buildFeatureDataset(
      List<EpsilonDecisionCallCounterfactualAudit.ProbeSample> samples, FrozenFeatures frozen) {
    int rows = samples.size();
    float[][] rootState = new float[rows][];
    float[][] rootStateAction = new float[rows][];
    float[][] afterStateDelta = new float[rows][];
    float[][] combined = new float[rows][];
    double[] target = new double[rows];
    long[] wallSeeds = new long[rows];
    for (int row = 0; row < rows; row++) {
      EpsilonDecisionCallCounterfactualAudit.RootOutcome outcome = samples.get(row).outcome();
      float[] rootProjected =
          projectTowers(frozen.root().policyState()[row], frozen.root().valueState()[row]);
      float[] callProjected =
          projectTowers(frozen.call().policyState()[row], frozen.call().valueState()[row]);
      float[] passProjected =
          projectTowers(frozen.pass().policyState()[row], frozen.pass().valueState()[row]);
      float[] afterDifference = subtract(callProjected, passProjected);
      float[] descriptor = actionDescriptor(outcome.callActionId());
      float[] actionInteraction =
          hashedInteraction(rootProjected, descriptor, rootProjected.length);
      float[] policyValueScalars =
          new float[] {
            (float) outcome.parentPassProbability(),
            (float) outcome.parentCallProbability(),
            (float) outcome.candidatePassProbability(),
            (float) outcome.candidateCallProbability(),
            (float) (outcome.candidateCallProbability() - outcome.candidatePassProbability()),
            (float) (outcome.parentPassProbability() - outcome.parentCallProbability()),
            (float) outcome.parentExpectedUtility(),
            (float) outcome.candidateExpectedUtility(),
            (float) (outcome.candidateExpectedUtility() - outcome.parentExpectedUtility())
          };
      float[] afterScalars =
          new float[] {
            (float) frozen.pass().expectedUtility()[row],
            (float) frozen.call().expectedUtility()[row],
            (float)
                (callUtilityImprovement(
                    frozen.pass().expectedUtility()[row], frozen.call().expectedUtility()[row]))
          };
      rootState[row] = rootProjected;
      rootStateAction[row] =
          concatenate(rootProjected, descriptor, actionInteraction, policyValueScalars);
      afterStateDelta[row] = concatenate(afterDifference, afterScalars);
      combined[row] = concatenate(rootStateAction[row], afterStateDelta[row]);
      target[row] = outcome.callMinusPassConfiguredUtility();
      wallSeeds[row] = outcome.wallSeed();
    }
    return new FeatureDataset(
        rootState, rootStateAction, afterStateDelta, combined, target, wallSeeds);
  }

  private static float[] projectTowers(float[] policy, float[] value) {
    return concatenate(
        countSketch(policy, PROJECTION_PER_TOWER, POLICY_PROJECTION_SALT),
        countSketch(value, PROJECTION_PER_TOWER, VALUE_PROJECTION_SALT));
  }

  private static float[] countSketch(float[] input, int outputSize, long salt) {
    float[] out = new float[outputSize];
    for (int index = 0; index < input.length; index++) {
      long hash = mix64(salt ^ (0x9E3779B97F4A7C15L * (index + 1L)));
      int bucket = (int) Long.remainderUnsigned(hash, outputSize);
      float sign = (hash & Long.MIN_VALUE) == 0L ? 1.0f : -1.0f;
      out[bucket] += sign * input[index];
    }
    return out;
  }

  private static float[] actionDescriptor(int actionId) {
    Action action = Action.fromIndex(actionId);
    float[] descriptor = new float[ACTION_DESCRIPTOR_SIZE];
    descriptor[action.type().ordinal()] = 1.0f;
    int tileSelectionStart = TYPE_FEATURES;
    descriptor[tileSelectionStart + action.tileSelection().ordinal()] = 1.0f;
    int tileStart = tileSelectionStart + TILE_SELECTION_FEATURES;
    if (Tile.isValidType(action.tileType())) {
      descriptor[tileStart + action.tileType()] = 1.0f;
    }
    if (action.type() == Action.Type.CHI) {
      int[] chiTiles = action.chiTileTypes();
      int calledPosition = action.tileType() - chiTiles[0];
      int positionStart = tileStart + TILE_FEATURES;
      descriptor[positionStart + calledPosition] = 1.0f;
      int sequenceStart = positionStart + CHI_POSITION_FEATURES;
      int suit = chiTiles[0] / 9;
      int rank = chiTiles[0] % 9;
      descriptor[sequenceStart + suit * 7 + rank] = 1.0f;
    }
    return descriptor;
  }

  private static float[] subtract(float[] left, float[] right) {
    if (left.length != right.length) {
      throw new IllegalArgumentException("Feature lengths differ");
    }
    float[] out = new float[left.length];
    for (int i = 0; i < out.length; i++) {
      out[i] = left[i] - right[i];
    }
    return out;
  }

  private static float[] hashedInteraction(float[] state, float[] action, int outputSize) {
    float[] out = new float[outputSize];
    for (int stateIndex = 0; stateIndex < state.length; stateIndex++) {
      for (int actionIndex = 0; actionIndex < action.length; actionIndex++) {
        if (action[actionIndex] == 0.0f) {
          continue;
        }
        long hash =
            mix64(
                ACTION_INTERACTION_SALT
                    ^ (0x9E3779B97F4A7C15L * (stateIndex + 1L))
                    ^ (0xD1B54A32D192ED03L * (actionIndex + 1L)));
        int bucket = (int) Long.remainderUnsigned(hash, outputSize);
        float sign = (hash & Long.MIN_VALUE) == 0L ? 1.0f : -1.0f;
        out[bucket] += sign * state[stateIndex] * action[actionIndex];
      }
    }
    return out;
  }

  private static float[] concatenate(float[]... parts) {
    int size = 0;
    for (float[] part : parts) {
      size = Math.addExact(size, part.length);
    }
    float[] out = new float[size];
    int offset = 0;
    for (float[] part : parts) {
      System.arraycopy(part, 0, out, offset, part.length);
      offset += part.length;
    }
    return out;
  }

  static CrossFitResult crossFit(String name, float[][] features, double[] target, int[] folds) {
    requireFeatureMatrix(features, target, folds);
    double[] predictions = new double[target.length];
    double[] selectedLambdas = new double[FOLDS];
    Arrays.fill(predictions, Double.NaN);
    for (int outer = 0; outer < FOLDS; outer++) {
      int outerFold = outer;
      int validationFold = (outer + 1) % FOLDS;
      int[] train = indices(folds, fold -> fold != outerFold && fold != validationFold);
      int[] validation = indices(folds, fold -> fold == validationFold);
      int[] test = indices(folds, fold -> fold == outerFold);
      PreparedRegression prepared = prepare(features, target, train);
      LinearModel best = null;
      double bestMse = Double.POSITIVE_INFINITY;
      double bestLambda = Double.NaN;
      for (double lambda : RIDGE_LAMBDAS) {
        LinearModel model = prepared.solve(lambda);
        double mse = mse(model, features, target, validation);
        if (mse < bestMse) {
          best = model;
          bestMse = mse;
          bestLambda = lambda;
        }
      }
      if (best == null) {
        throw new IllegalStateException("No finite ridge fit for " + name + " outer=" + outer);
      }
      selectedLambdas[outer] = bestLambda;
      for (int row : test) {
        predictions[row] = best.predict(features[row]);
      }
    }
    for (double prediction : predictions) {
      if (!Double.isFinite(prediction)) {
        throw new IllegalStateException("Cross-fit prediction is missing for " + name);
      }
    }
    return new CrossFitResult(name, features[0].length, predictions, selectedLambdas);
  }

  private static PreparedRegression prepare(float[][] features, double[] target, int[] rows) {
    int dimension = features[0].length;
    double[] mean = new double[dimension];
    double targetMean = 0.0;
    for (int row : rows) {
      targetMean += target[row];
      for (int column = 0; column < dimension; column++) {
        mean[column] += features[row][column];
      }
    }
    targetMean /= rows.length;
    for (int column = 0; column < dimension; column++) {
      mean[column] /= rows.length;
    }
    double[] scale = new double[dimension];
    for (int row : rows) {
      for (int column = 0; column < dimension; column++) {
        double centered = features[row][column] - mean[column];
        scale[column] += centered * centered;
      }
    }
    for (int column = 0; column < dimension; column++) {
      scale[column] = Math.sqrt(scale[column] / rows.length);
      if (!Double.isFinite(scale[column]) || scale[column] < MIN_SCALE) {
        scale[column] = 1.0;
      }
    }

    double[][] gram = new double[dimension][dimension];
    double[] rhs = new double[dimension];
    double[] standardized = new double[dimension];
    for (int row : rows) {
      for (int column = 0; column < dimension; column++) {
        standardized[column] = (features[row][column] - mean[column]) / scale[column];
      }
      double centeredTarget = target[row] - targetMean;
      for (int left = 0; left < dimension; left++) {
        double value = standardized[left];
        rhs[left] += value * centeredTarget;
        for (int right = 0; right <= left; right++) {
          gram[left][right] += value * standardized[right];
        }
      }
    }
    double inverseRows = 1.0 / rows.length;
    for (int left = 0; left < dimension; left++) {
      rhs[left] *= inverseRows;
      for (int right = 0; right <= left; right++) {
        double value = gram[left][right] * inverseRows;
        gram[left][right] = value;
        gram[right][left] = value;
      }
    }
    return new PreparedRegression(mean, scale, targetMean, gram, rhs);
  }

  private static double mse(LinearModel model, float[][] features, double[] target, int[] rows) {
    double sum = 0.0;
    for (int row : rows) {
      double error = model.predict(features[row]) - target[row];
      sum += error * error;
    }
    return sum / rows.length;
  }

  private static double[] choleskySolve(double[][] source, double[] rhs, double lambda) {
    int dimension = rhs.length;
    double[][] lower = new double[dimension][dimension];
    for (int row = 0; row < dimension; row++) {
      for (int column = 0; column <= row; column++) {
        double value = source[row][column] + (row == column ? lambda : 0.0);
        for (int inner = 0; inner < column; inner++) {
          value -= lower[row][inner] * lower[column][inner];
        }
        if (row == column) {
          if (!(value > 0.0) || !Double.isFinite(value)) {
            throw new IllegalStateException(
                "Ridge normal matrix is not positive definite at " + row + " value=" + value);
          }
          lower[row][column] = Math.sqrt(value);
        } else {
          lower[row][column] = value / lower[column][column];
        }
      }
    }
    double[] intermediate = new double[dimension];
    for (int row = 0; row < dimension; row++) {
      double value = rhs[row];
      for (int column = 0; column < row; column++) {
        value -= lower[row][column] * intermediate[column];
      }
      intermediate[row] = value / lower[row][row];
    }
    double[] solution = new double[dimension];
    for (int row = dimension - 1; row >= 0; row--) {
      double value = intermediate[row];
      for (int column = row + 1; column < dimension; column++) {
        value -= lower[column][row] * solution[column];
      }
      solution[row] = value / lower[row][row];
    }
    return solution;
  }

  private static ModelMetrics modelMetrics(
      String name, CrossFitResult fit, double[] target, double[] crossFitMeanPrediction) {
    double[] prediction = fit.predictions();
    double mse = meanSquaredError(prediction, target);
    double baselineMse = meanSquaredError(crossFitMeanPrediction, target);
    double[] selectedUtility = new double[target.length];
    double[] gainVsAlwaysCall = new double[target.length];
    int calls = 0;
    for (int row = 0; row < target.length; row++) {
      if (prediction[row] > 0.0) {
        selectedUtility[row] = target[row];
        calls++;
      }
      gainVsAlwaysCall[row] = selectedUtility[row] - target[row];
    }
    return new ModelMetrics(
        name,
        fit.dimension(),
        fit.selectedLambdas().clone(),
        mse,
        baselineMse,
        baselineMse > 0.0 ? 1.0 - mse / baselineMse : 0.0,
        pearson(prediction, target),
        auc(prediction, target),
        nonTieSignAccuracy(prediction, target),
        calls / (double) target.length,
        interval(selectedUtility),
        interval(gainVsAlwaysCall));
  }

  private static Baselines baselines(
      FeatureDataset features, FrozenFeatures frozen, double[] crossFitMeanPrediction) {
    double[] target = features.target();
    double[] oracle = new double[target.length];
    double[] frozenAfterStatePrediction = new double[target.length];
    for (int row = 0; row < target.length; row++) {
      oracle[row] = Math.max(0.0, target[row]);
      frozenAfterStatePrediction[row] =
          callUtilityImprovement(
              frozen.pass().expectedUtility()[row], frozen.call().expectedUtility()[row]);
    }
    return new Baselines(
        interval(new double[target.length]),
        interval(target),
        interval(oracle),
        meanSquaredError(crossFitMeanPrediction, target),
        new FrozenValueMetrics(
            meanSquaredError(frozenAfterStatePrediction, target),
            pearson(frozenAfterStatePrediction, target),
            auc(frozenAfterStatePrediction, target),
            nonTieSignAccuracy(frozenAfterStatePrediction, target)));
  }

  private static Comparison lossComparison(
      String name, double[] baselinePrediction, double[] candidatePrediction, double[] target) {
    double[] improvements = new double[target.length];
    for (int row = 0; row < target.length; row++) {
      double baselineError = baselinePrediction[row] - target[row];
      double candidateError = candidatePrediction[row] - target[row];
      improvements[row] = baselineError * baselineError - candidateError * candidateError;
    }
    return new Comparison(name, interval(improvements));
  }

  private static Verdict verdict(
      Map<String, ModelMetrics> models, Map<String, Comparison> comparisons) {
    Comparison afterState = comparisons.get("afterstate-delta-vs-root-state");
    Comparison rootAction = comparisons.get("root-state+action-vs-root-state");
    ModelMetrics afterStateModel = models.get("afterstate-delta");
    boolean afterStateRepresentation = afterState.squaredErrorImprovement().lower95() > 0.0;
    boolean rootActionRepresentation = rootAction.squaredErrorImprovement().lower95() > 0.0;
    boolean afterStatePolicy =
        afterStateModel.selectedUtility().lower95() > 0.0
            && afterStateModel.gainVsAlwaysCall().lower95() > 0.0;
    RepresentationSignal representation =
        afterStateRepresentation
            ? RepresentationSignal.SUPPORTED
            : (rootActionRepresentation
                ? RepresentationSignal.ROOT_ACTION_ONLY
                : RepresentationSignal.NOT_DEMONSTRATED);
    PolicySignal policy = afterStatePolicy ? PolicySignal.SUPPORTED : PolicySignal.NOT_DEMONSTRATED;
    return new Verdict(
        representation,
        policy,
        afterStateRepresentation,
        rootActionRepresentation,
        afterStatePolicy,
        afterStateRepresentation
            ? "Frozen AfterState differences improve held-out prediction beyond root state."
            : "The primary AfterState probe does not beat root-state-only with a positive 95%"
                + " paired loss-improvement lower bound.",
        afterStatePolicy
            ? "The AfterState switch policy beats both always-PASS and always-CALL at 95%."
            : "Predictive evidence is not sufficient to claim a deployable PASS/CALL switch. No"
                + " actor or checkpoint update is authorized by this probe alone.");
  }

  private static String predictionTraceLines(
      List<EpsilonDecisionCallCounterfactualAudit.ProbeSample> samples,
      FeatureDataset features,
      FrozenFeatures frozen,
      int[] folds,
      Map<String, CrossFitResult> fitted) {
    StringBuilder out = new StringBuilder();
    for (int row = 0; row < samples.size(); row++) {
      EpsilonDecisionCallCounterfactualAudit.RootOutcome outcome = samples.get(row).outcome();
      LinkedHashMap<String, Double> predictions = new LinkedHashMap<>();
      for (Map.Entry<String, CrossFitResult> entry : fitted.entrySet()) {
        predictions.put(entry.getKey(), entry.getValue().predictions()[row]);
      }
      PredictionTrace trace =
          new PredictionTrace(
              outcome.gameIndex(),
              outcome.wallSeed(),
              folds[row],
              outcome.callType(),
              outcome.callActionId(),
              features.target()[row],
              outcome.callMinusPassRank(),
              outcome.callMinusPassScore(),
              frozen.pass().expectedUtility()[row],
              frozen.call().expectedUtility()[row],
              Map.copyOf(predictions));
      out.append(LINE_GSON.toJson(trace)).append(System.lineSeparator());
    }
    return out.toString();
  }

  private static double[] crossFitMeans(double[] target, int[] folds) {
    double[] prediction = new double[target.length];
    for (int outer = 0; outer < FOLDS; outer++) {
      int validationFold = (outer + 1) % FOLDS;
      double sum = 0.0;
      int count = 0;
      for (int row = 0; row < target.length; row++) {
        if (folds[row] != outer && folds[row] != validationFold) {
          sum += target[row];
          count++;
        }
      }
      double mean = sum / count;
      for (int row = 0; row < target.length; row++) {
        if (folds[row] == outer) {
          prediction[row] = mean;
        }
      }
    }
    return prediction;
  }

  private static int[] folds(long[] wallSeeds) {
    int[] folds = new int[wallSeeds.length];
    for (int row = 0; row < wallSeeds.length; row++) {
      long hash = mix64(wallSeeds[row] ^ FOLD_SALT);
      folds[row] = (int) Long.remainderUnsigned(hash, FOLDS);
    }
    int[] counts = foldCounts(folds);
    for (int fold = 0; fold < counts.length; fold++) {
      if (counts[fold] == 0) {
        throw new IllegalStateException("Empty probe fold " + fold);
      }
    }
    return folds;
  }

  private static int[] foldCounts(int[] folds) {
    int[] counts = new int[FOLDS];
    for (int fold : folds) {
      counts[fold]++;
    }
    return counts;
  }

  private static int[] indices(int[] folds, FoldPredicate predicate) {
    int count = 0;
    for (int fold : folds) {
      if (predicate.test(fold)) {
        count++;
      }
    }
    int[] out = new int[count];
    int offset = 0;
    for (int row = 0; row < folds.length; row++) {
      if (predicate.test(folds[row])) {
        out[offset++] = row;
      }
    }
    return out;
  }

  private static void requireFeatureMatrix(float[][] features, double[] target, int[] folds) {
    if (features.length == 0 || features.length != target.length || target.length != folds.length) {
      throw new IllegalArgumentException("Probe feature/target/fold row counts differ");
    }
    int dimension = features[0].length;
    if (dimension == 0) {
      throw new IllegalArgumentException("Probe features must be non-empty");
    }
    for (int row = 0; row < features.length; row++) {
      if (features[row].length != dimension || !Double.isFinite(target[row])) {
        throw new IllegalArgumentException("Invalid probe feature row " + row);
      }
      for (float value : features[row]) {
        if (!Float.isFinite(value)) {
          throw new IllegalArgumentException("Non-finite probe feature at row " + row);
        }
      }
    }
  }

  private static void requireUniqueGames(
      List<EpsilonDecisionCallCounterfactualAudit.ProbeSample> samples) {
    Set<Integer> games = new HashSet<>();
    for (EpsilonDecisionCallCounterfactualAudit.ProbeSample sample : samples) {
      if (!games.add(sample.outcome().gameIndex())) {
        throw new IllegalStateException(
            "More than one probe root for base game " + sample.outcome().gameIndex());
      }
    }
  }

  private static double meanSquaredError(double[] prediction, double[] target) {
    double sum = 0.0;
    for (int row = 0; row < target.length; row++) {
      double error = prediction[row] - target[row];
      sum += error * error;
    }
    return sum / target.length;
  }

  private static Double pearson(double[] left, double[] right) {
    double leftMean = mean(left);
    double rightMean = mean(right);
    double covariance = 0.0;
    double leftVariance = 0.0;
    double rightVariance = 0.0;
    for (int i = 0; i < left.length; i++) {
      double l = left[i] - leftMean;
      double r = right[i] - rightMean;
      covariance += l * r;
      leftVariance += l * l;
      rightVariance += r * r;
    }
    double denominator = Math.sqrt(leftVariance * rightVariance);
    return denominator > 0.0 ? covariance / denominator : null;
  }

  private static Double auc(double[] prediction, double[] target) {
    ArrayList<Integer> rows = new ArrayList<>();
    int positives = 0;
    int negatives = 0;
    for (int row = 0; row < target.length; row++) {
      if (target[row] > 0.0) {
        positives++;
        rows.add(row);
      } else if (target[row] < 0.0) {
        negatives++;
        rows.add(row);
      }
    }
    if (positives == 0 || negatives == 0) {
      return null;
    }
    rows.sort((a, b) -> Double.compare(prediction[a], prediction[b]));
    double positiveRankSum = 0.0;
    int index = 0;
    while (index < rows.size()) {
      int end = index + 1;
      while (end < rows.size()
          && Double.compare(prediction[rows.get(index)], prediction[rows.get(end)]) == 0) {
        end++;
      }
      double averageRank = ((index + 1) + end) * 0.5;
      for (int at = index; at < end; at++) {
        if (target[rows.get(at)] > 0.0) {
          positiveRankSum += averageRank;
        }
      }
      index = end;
    }
    return (positiveRankSum - positives * (positives + 1.0) * 0.5)
        / (positives * (double) negatives);
  }

  private static Double nonTieSignAccuracy(double[] prediction, double[] target) {
    int correct = 0;
    int count = 0;
    for (int row = 0; row < target.length; row++) {
      if (target[row] == 0.0) {
        continue;
      }
      count++;
      if ((prediction[row] > 0.0) == (target[row] > 0.0)) {
        correct++;
      }
    }
    return count > 0 ? correct / (double) count : null;
  }

  private static MeanInterval interval(double[] values) {
    double mean = mean(values);
    if (values.length < 2) {
      return new MeanInterval(mean, 0.0, mean, mean);
    }
    double sumSquares = 0.0;
    for (double value : values) {
      double centered = value - mean;
      sumSquares += centered * centered;
    }
    double standardError = Math.sqrt(sumSquares / (values.length - 1)) / Math.sqrt(values.length);
    return new MeanInterval(
        mean, standardError, mean - Z_95 * standardError, mean + Z_95 * standardError);
  }

  private static double mean(double[] values) {
    double sum = 0.0;
    for (double value : values) {
      sum += value;
    }
    return sum / values.length;
  }

  private static int nonZeroCount(double[] values) {
    int count = 0;
    for (double value : values) {
      if (value != 0.0) {
        count++;
      }
    }
    return count;
  }

  /** Utilityは大きい方が良いので、CALLの改善量はCALL-PASSで測る。 */
  static double callUtilityImprovement(double passUtility, double callUtility) {
    return callUtility - passUtility;
  }

  private static long mix64(long value) {
    value ^= value >>> 30;
    value *= 0xBF58476D1CE4E5B9L;
    value ^= value >>> 27;
    value *= 0x94D049BB133111EBL;
    return value ^ (value >>> 31);
  }

  /**
   * 固定した Decision 表現から CALL−PASS 効用を予測できるか調べる牌譜などの固定データによる検証用の予測器レポート。
   *
   * @param schema レポートスキーマ ID
   * @param generatedAt 生成時刻
   * @param baseNetworksFrozen 基準ネットワークパラメーターを固定したか
   * @param probeFitting 検証用の予測器モデルだけを学習したか
   * @param checkpointMutation チェックポイントを変更したか
   * @param protocol データセット・分割・モデル選択規則
   * @param baseGames 比較局面探索に使った基底対局数
   * @param seedBase 基底牌山乱数シード
   * @param gamesInFlight 同時進行対局数
   * @param configuredUtilityProfile 教師値効用の定義
   * @param parent 比較元 Decision 識別情報
   * @param candidate 候補 Decision 識別情報
   * @param grp 固定 GRP 教師モデル識別情報
   * @param rootSearch 比較局面探索指標
   * @param branchMetrics 分岐実行指標
   * @param dataset 検証用の予測器データセットの形状と件数
   * @param features 検証用の予測器特徴量構成
   * @param baselines 定数予測および正解を知る場合の比較基準
   * @param models 特徴量組ごとの交差検証指標
   * @param comparisons モデル間比較
   * @param verdict 表現更新量と方策更新量の判定
   * @param predictionTraceFile 交差検証予測トレース
   */
  public record ProbeReport(
      String schema,
      String generatedAt,
      boolean baseNetworksFrozen,
      boolean probeFitting,
      boolean checkpointMutation,
      Protocol protocol,
      int baseGames,
      long seedBase,
      int gamesInFlight,
      String configuredUtilityProfile,
      EpsilonDecisionCallCounterfactualAudit.CheckpointIdentity parent,
      EpsilonDecisionCallCounterfactualAudit.CheckpointIdentity candidate,
      GrpIdentity grp,
      EpsilonDecisionCallCounterfactualAudit.RootSearch rootSearch,
      EpsilonDecisionCallCounterfactualAudit.BranchMetrics branchMetrics,
      DatasetSummary dataset,
      FeatureConfiguration features,
      Baselines baselines,
      Map<String, ModelMetrics> models,
      Map<String, Comparison> comparisons,
      Verdict verdict,
      String predictionTraceFile) {}

  /**
   * 牌譜などの固定データによる検証用の予測器の 情報漏洩 を防ぐ実験規則。
   *
   * @param dataset 行動を変えた場合の比較データセットの作り方
   * @param intervention PASS／CALL 教師値の定義
   * @param frozenRepresentation 読み出す固定したネットワーク表現
   * @param split 外側の分割群の分割単位
   * @param modelSelection リッジ回帰係数の選択規則
   * @param primaryComparison 主要な特徴量比較
   */
  public record Protocol(
      String dataset,
      String intervention,
      String frozenRepresentation,
      String split,
      String modelSelection,
      String primaryComparison) {}

  /**
   * 検証用の予測器教師値に使った GRP 教師モデル識別情報。
   *
   * @param root チェックポイントのルートディレクトリ
   * @param checkpoint チェックポイントパス
   * @param iteration チェックポイント反復回数
   * @param use 検証用の予測器内での用途
   */
  public record GrpIdentity(String root, String checkpoint, int iteration, String use) {}

  /**
   * 交差検証データセットの件数と各特徴量ブロックの次元。
   *
   * @param roots 対応をそろえた比較局面数
   * @param foldCounts 外側の分割群ごとの比較局面数
   * @param frozenHiddenSize 固定したネットワークの隠れ層幅
   * @param rootStateDimension 比較局面の状態特徴量次元
   * @param rootStateActionDimension 比較局面の状態＋行動特徴量次元
   * @param afterStateDeltaDimension CALL−PASS 行動後の状態差特徴量次元
   * @param combinedDimension 全特徴量結合次元
   * @param nonTieTargets 効用教師値がゼロでない比較局面数
   */
  public record DatasetSummary(
      int roots,
      int[] foldCounts,
      int frozenHiddenSize,
      int rootStateDimension,
      int rootStateActionDimension,
      int afterStateDeltaDimension,
      int combinedDimension,
      int nonTieTargets) {
    /** 分割群件数配列を防御複製し、要約を変更不可にする。 */
    public DatasetSummary {
      foldCounts = foldCounts.clone();
    }
  }

  /**
   * 固定した特徴量抽出とリッジ回帰検証用の予測器の設定。
   *
   * @param frozenSource 固定したチェックポイント入力元
   * @param projection 隠れ層表現の固定射影方法
   * @param projectedFeaturesPerTower ネットワークごとの射影次元
   * @param actionDescriptorFeatures 行動記述情報次元
   * @param rootStateAction 比較局面の状態-行動特徴量の定義
   * @param afterStateDelta 行動後の状態差特徴量の定義
   * @param outerFolds 外側の交差検証分割群数
   * @param ridgeLambdas 内側の選択で比較したリッジ回帰係数
   */
  public record FeatureConfiguration(
      String frozenSource,
      String projection,
      int projectedFeaturesPerTower,
      int actionDescriptorFeatures,
      String rootStateAction,
      String afterStateDelta,
      int outerFolds,
      double[] ridgeLambdas) {
    /** リッジ回帰係数候補配列を防御複製し、設定を変更不可にする。 */
    public FeatureConfiguration {
      ridgeLambdas = ridgeLambdas.clone();
    }
  }

  /**
   * 学習済み検証用の予測器と比較する効用／価値比較基準。
   *
   * @param alwaysPassUtility 常に PASS を選ぶ効用
   * @param alwaysCallUtility 常に CALL を選ぶ効用
   * @param oracleSwitchUtility 教師値の符号に基づいて最善の行動を選んだ場合の効用
   * @param crossFitMeanMse 学習分割群平均だけを予測する MSE
   * @param frozenAfterStateValue 固定した価値自体の予測指標
   */
  public record Baselines(
      MeanInterval alwaysPassUtility,
      MeanInterval alwaysCallUtility,
      MeanInterval oracleSwitchUtility,
      double crossFitMeanMse,
      FrozenValueMetrics frozenAfterStateValue) {}

  /**
   * 固定した価値の CALL−PASS 教師値予測指標。
   *
   * @param mse 平均二乗誤差
   * @param pearson 教師値との ピアソン相関係数。未定義なら {@code null}
   * @param nonTieAuc 同順位を除く教師値の符号 AUC。未定義なら {@code null}
   * @param nonTieSignAccuracy 同順位を除く教師値の符号正解率。未定義なら {@code null}
   */
  public record FrozenValueMetrics(
      double mse, Double pearson, Double nonTieAuc, Double nonTieSignAccuracy) {}

  /**
   * 一つの特徴量構成を外側の交差検証した検証用の予測器指標。
   *
   * @param name 特徴量構成名
   * @param dimension 入力次元
   * @param selectedLambdaByFold 各外側の分割群で選んだリッジ回帰係数
   * @param mse 全学習に使わなかった分割群の予測の MSE
   * @param crossFitMeanMse 分割群平均比較基準の MSE
   * @param rSquaredVsCrossFitMean 分割群平均比較基準に対する決定係数
   * @param pearson 教師値との ピアソン相関係数
   * @param nonTieAuc 同順位を除く教師値の符号 AUC
   * @param nonTieSignAccuracy 同順位を除く教師値の符号正解率
   * @param predictedCallRate 予測効用が正の比較局面割合
   * @param selectedUtility 予測符号で PASS／CALL を選んだ効用
   * @param gainVsAlwaysCall 常時 CALL 比較基準からの効用改善
   */
  public record ModelMetrics(
      String name,
      int dimension,
      double[] selectedLambdaByFold,
      double mse,
      double crossFitMeanMse,
      double rSquaredVsCrossFitMean,
      Double pearson,
      Double nonTieAuc,
      Double nonTieSignAccuracy,
      double predictedCallRate,
      MeanInterval selectedUtility,
      MeanInterval gainVsAlwaysCall) {
    /** 分割群別選択係数配列を防御複製し、指標を変更不可にする。 */
    public ModelMetrics {
      selectedLambdaByFold = selectedLambdaByFold.clone();
    }
  }

  /**
   * 二つの検証用の予測器特徴量構成の対応をそろえた誤差改善。
   *
   * @param name 比較名
   * @param squaredErrorImprovement サンプルごとの二乗誤差の改善量の区間推定
   */
  public record Comparison(String name, MeanInterval squaredErrorImprovement) {}

  /** 評価データで改善を確認できた表現。 */
  public enum RepresentationSignal {
    SUPPORTED,
    ROOT_ACTION_ONLY,
    NOT_DEMONSTRATED
  }

  /** PASS／CALL の選択改善に対する実証の有無。 */
  public enum PolicySignal {
    SUPPORTED,
    NOT_DEMONSTRATED
  }

  /**
   * 行動後の状態／比較局面の行動表現に実証更新量があるかを事前基準で判定した結果。
   *
   * @param representationSignal 表現更新量の総合判定
   * @param policySignal 方策選択改善更新量の総合判定
   * @param afterStateRepresentationSignal 行動後の状態特徴量の誤差改善が基準を満たしたか
   * @param rootActionRepresentationSignal 比較局面の行動特徴量の誤差改善が基準を満たしたか
   * @param afterStatePolicySignal 行動後の状態特徴量の選択効用改善が基準を満たしたか
   * @param representationReason 表現判定の根拠
   * @param policyReason 方策判定の根拠
   */
  public record Verdict(
      RepresentationSignal representationSignal,
      PolicySignal policySignal,
      boolean afterStateRepresentationSignal,
      boolean rootActionRepresentationSignal,
      boolean afterStatePolicySignal,
      String representationReason,
      String policyReason) {}

  /**
   * スカラー標本平均と正規近似95%区間。
   *
   * @param mean 標本平均
   * @param standardError 平均の標準誤差
   * @param lower95 95%下限
   * @param upper95 95%上限
   */
  public record MeanInterval(double mean, double standardError, double lower95, double upper95) {}

  private record FrozenFeatures(
      FrozenBatch root, FrozenBatch pass, FrozenBatch call, int hiddenSize) {}

  private record FrozenBatch(
      float[][] policyState, float[][] valueState, double[] expectedUtility, int hiddenSize) {}

  private record FeatureDataset(
      float[][] rootState,
      float[][] rootStateAction,
      float[][] afterStateDelta,
      float[][] combined,
      double[] target,
      long[] wallSeeds) {}

  record CrossFitResult(
      String name, int dimension, double[] predictions, double[] selectedLambdas) {
    CrossFitResult {
      predictions = predictions.clone();
      selectedLambdas = selectedLambdas.clone();
    }
  }

  private record PreparedRegression(
      double[] mean, double[] scale, double targetMean, double[][] gram, double[] rhs) {

    private LinearModel solve(double lambda) {
      return new LinearModel(mean, scale, targetMean, choleskySolve(gram, rhs, lambda));
    }
  }

  private record LinearModel(
      double[] mean, double[] scale, double targetMean, double[] coefficients) {

    private double predict(float[] features) {
      double prediction = targetMean;
      for (int column = 0; column < coefficients.length; column++) {
        prediction += coefficients[column] * (features[column] - mean[column]) / scale[column];
      }
      return prediction;
    }
  }

  private record PredictionTrace(
      int gameIndex,
      long wallSeed,
      int fold,
      Action.Type callType,
      int callActionId,
      double targetUtilityDelta,
      int rankDelta,
      int scoreDelta,
      double passAfterExpectedUtility,
      double callAfterExpectedUtility,
      Map<String, Double> predictions) {}

  @FunctionalInterface
  private interface FoldPredicate {
    boolean test(int fold);
  }
}
