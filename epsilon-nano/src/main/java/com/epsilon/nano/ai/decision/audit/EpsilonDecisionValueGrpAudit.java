package com.epsilon.nano.ai.decision.audit;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpTeacherIdentity;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.nano.ai.decision.training.DecisionSampleBatcher;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointBundle;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionLogPretrainRunner;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionPretrainTargets;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.DecisionInferenceSettings;
import com.epsilon.nano.config.settings.DecisionSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.nano.training.EpsilonLogPretrainDataCollector;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 既存の Decision チェックポイントについて、価値予測と重みを固定した GRP の教師値との整合性を調べる。 */
public final class EpsilonDecisionValueGrpAudit {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionValueGrpAudit.class);
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final String SCHEMA = "epsilon-decision-value-grp-audit-v3";

  private EpsilonDecisionValueGrpAudit() {}

  /**
   * 事前学習と同じファイル単位で分離した検証データから安定ハッシュで部分集合を選び、全チェックポイントを同じ順序で評価する。
   *
   * <p>チェックポイントと GRP は一切保存せず、書き込み先は新規 {@code outputFile} だけである。
   *
   * @param outputFile 監査 JSON の未作成出力先
   * @param inputFileManifest 事前学習入力元ファイルを列挙したマニフェスト
   * @param grpDecisionCheckpointDir 重みを固定したGRP チェックポイントを含むルートディレクトリ
   * @param validationFraction ファイル単位で分離した検証データを再現する割合
   * @param maxValidationFiles 評価する検証用ファイル数の上限
   * @param checkpointSpecs 同じサンプル順で比較する Decision チェックポイント
   * @return GRP 教師モデルと各 Decision 価値の一致・校正レポート
   * @throws Exception 入力元選択、チェックポイント推論、識別情報検証、または出力に失敗した場合
   */
  public static AuditReport run(
      Path outputFile,
      Path inputFileManifest,
      Path grpDecisionCheckpointDir,
      float validationFraction,
      int maxValidationFiles,
      List<CheckpointSpec> checkpointSpecs)
      throws Exception {
    return run(
        outputFile,
        inputFileManifest,
        grpDecisionCheckpointDir,
        validationFraction,
        maxValidationFiles,
        checkpointSpecs,
        EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static AuditReport run(
      Path outputFile,
      Path inputFileManifest,
      Path grpDecisionCheckpointDir,
      float validationFraction,
      int maxValidationFiles,
      List<CheckpointSpec> checkpointSpecs,
      SettingsLoader snapshot)
      throws Exception {
    Objects.requireNonNull(outputFile, "outputFile");
    Objects.requireNonNull(inputFileManifest, "inputFileManifest");
    Objects.requireNonNull(grpDecisionCheckpointDir, "grpDecisionCheckpointDir");
    Objects.requireNonNull(checkpointSpecs, "checkpointSpecs");
    if (!(validationFraction > 0.0f && validationFraction < 1.0f)) {
      throw new IllegalArgumentException(
          "validationFraction must be between 0 and 1: " + validationFraction);
    }
    if (maxValidationFiles <= 0) {
      throw new IllegalArgumentException("maxValidationFiles must be positive");
    }
    if (checkpointSpecs.size() < 2) {
      throw new IllegalArgumentException("At least baseline and one candidate checkpoint required");
    }

    Path output = outputFile.toAbsolutePath().normalize();
    if (Files.exists(output)) {
      throw new IOException("Refusing to overwrite existing audit output: " + output);
    }
    Path outputParent = output.getParent();
    if (outputParent != null) {
      Files.createDirectories(outputParent);
    }

    Selection selection =
        selectValidationFiles(inputFileManifest, validationFraction, maxValidationFiles);
    validateCheckpointSpecs(checkpointSpecs);

    Path grpRoot = grpDecisionCheckpointDir.toRealPath();
    Path grpCheckpoint = EpsilonGrpCheckpointManager.resolveExisting(grpRoot.resolve("grp"));
    if (grpCheckpoint == null) {
      throw new IOException("Frozen GRP checkpoint not found below " + grpRoot.resolve("grp"));
    }
    grpCheckpoint = grpCheckpoint.toRealPath();

    Device device =
        NetworkFactory.getInferenceDevices(snapshot.bind(DeviceSettings.class)).primary();
    ArrayList<CheckpointAudit> checkpoints = new ArrayList<>(checkpointSpecs.size());
    long expectedSamples = -1L;
    long expectedGrpTeacherSamples = -1L;
    EpsilonGrpTeacherIdentity grpIdentity;
    try (EpsilonGrpTrainingSession grpSession =
        EpsilonGrpTrainingSession.openInferenceOnly(
            grpRoot,
            NetworkFactory.getGrpDevices(snapshot.bind(DeviceSettings.class)).primary(),
            snapshot)) {
      grpIdentity = grpSession.identity();
      for (int index = 0; index < checkpointSpecs.size(); index++) {
        CheckpointSpec spec = checkpointSpecs.get(index).normalized();
        Path checkpoint = spec.path().toRealPath();
        EpsilonDecisionCheckpointBundle manifest =
            EpsilonDecisionCheckpointManager.loadManifest(checkpoint);
        log.info(
            "Value-GRP audit checkpoint start: label={} checkpoint={} files={}",
            spec.label(),
            checkpoint,
            selection.selectedFiles().size());
        ValueGrpMetrics metrics;
        try (Model model =
            EpsilonDecisionCheckpointManager.loadForInference(
                checkpoint, device, snapshot.bind(DecisionInferenceSettings.class))) {
          metrics = evaluate(model, selection.selectedFiles(), grpSession, snapshot);
        }
        if (expectedSamples < 0L) {
          expectedSamples = metrics.samples();
          expectedGrpTeacherSamples = metrics.grpTeacherSamples();
        } else if (metrics.samples() != expectedSamples
            || metrics.grpTeacherSamples() != expectedGrpTeacherSamples) {
          throw new IOException(
              "Audit sample population changed between checkpoints: expectedSamples="
                  + expectedSamples
                  + " actualSamples="
                  + metrics.samples()
                  + " expectedGrpTeacherSamples="
                  + expectedGrpTeacherSamples
                  + " actualGrpTeacherSamples="
                  + metrics.grpTeacherSamples());
        }
        checkpoints.add(
            new CheckpointAudit(
                spec.label(),
                checkpoint.toString(),
                manifest.globalStep,
                manifest.iteration,
                manifest.selfPlayGames,
                metrics,
                null));
        log.info(
            "Value-GRP audit checkpoint complete: label={} samples={}"
                + " grpTeacherSamples={} teacherUtilityMse={} teacherUtilityPearson={}"
                + " actualUtilityMse={}",
            spec.label(),
            metrics.samples(),
            metrics.grpTeacherSamples(),
            metrics.teacherUtilityMse(),
            metrics.teacherUtilityPearson(),
            metrics.actualUtilityMse());
      }
    }

    ValueGrpMetrics baseline = checkpoints.getFirst().metrics();
    ArrayList<CheckpointAudit> withDeltas = new ArrayList<>(checkpoints.size());
    for (CheckpointAudit checkpoint : checkpoints) {
      withDeltas.add(
          new CheckpointAudit(
              checkpoint.label(),
              checkpoint.path(),
              checkpoint.globalStep(),
              checkpoint.iteration(),
              checkpoint.selfPlayGames(),
              checkpoint.metrics(),
              ValueGrpDelta.between(checkpoint.metrics(), baseline)));
    }

    AuditReport report =
        new AuditReport(
            SCHEMA,
            Instant.now().toString(),
            false,
            false,
            device.toString(),
            inputFileManifest.toRealPath().toString(),
            validationFraction,
            selection.eligibleValidationFiles(),
            maxValidationFiles,
            selection.selectedFiles().size(),
            selection.selectedFiles().stream().map(Path::toString).toList(),
            grpRoot.toString(),
            grpCheckpoint.toString(),
            grpIdentity.iteration(),
            checkpoints.getFirst().label(),
            expectedSamples,
            expectedGrpTeacherSamples,
            List.copyOf(withDeltas));
    Files.writeString(
        output,
        GSON.toJson(report) + System.lineSeparator(),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE_NEW,
        StandardOpenOption.WRITE);
    log.info("Value-GRP audit report written: {}", output);
    return report;
  }

  static Selection selectValidationFiles(
      Path inputFileManifest, float validationFraction, int maxValidationFiles) throws IOException {
    if (!Files.isRegularFile(inputFileManifest)) {
      throw new IOException("Input file manifest not found: " + inputFileManifest);
    }
    LinkedHashSet<Path> eligiblePaths = new LinkedHashSet<>();
    for (String rawLine : Files.readAllLines(inputFileManifest, StandardCharsets.UTF_8)) {
      String line = rawLine.trim();
      if (line.isEmpty()) {
        continue;
      }
      Path file = Path.of(line).toAbsolutePath().normalize();
      if (!EpsilonDecisionLogPretrainRunner.isValidationFile(file, validationFraction)) {
        continue;
      }
      if (!eligiblePaths.add(file)) {
        throw new IOException("Duplicate validation input path: " + file);
      }
    }
    if (eligiblePaths.isEmpty()) {
      throw new IOException("No validation files selected from " + inputFileManifest);
    }
    ArrayList<Path> ranked = new ArrayList<>(eligiblePaths);
    ranked.sort(Comparator.comparing(Path::toString));
    int selectedCount = Math.min(maxValidationFiles, ranked.size());
    List<Path> selected = List.copyOf(ranked.subList(0, selectedCount));
    for (Path file : selected) {
      if (!Files.isRegularFile(file)) {
        throw new IOException("Selected validation log is missing: " + file);
      }
    }
    return new Selection(ranked.size(), selected);
  }

  private static ValueGrpMetrics evaluate(
      Model model, List<Path> files, EpsilonGrpTrainingSession grpSession, SettingsLoader snapshot)
      throws Exception {
    Totals totals = new Totals();
    try (EpsilonDecisionInferenceServer server =
        EpsilonDecisionInferenceServer.forFrozenModel(
            model,
            snapshot.bind(DecisionInferenceSettings.class).maxBatch(),
            snapshot.bind(DecisionInferenceSettings.class),
            snapshot.bind(DecisionInferenceFusionSettings.class))) {
      for (Path file : files) {
        List<EpsilonDecisionSample> rawSamples;
        try {
          rawSamples = EpsilonLogPretrainDataCollector.collectDecisionFile(file);
        } catch (Exception e) {
          throw new IOException("Failed to load fixed audit log: " + file, e);
        }
        if (rawSamples.isEmpty()) {
          throw new IOException("Fixed audit log produced no Decision samples: " + file);
        }
        boolean[] grpBacked = grpBackedSamples(rawSamples);
        List<EpsilonDecisionSample> samples =
            EpsilonDecisionPretrainTargets.prepare(
                rawSamples, grpSession, snapshot.bind(DecisionSettings.class).utilityProfile());
        List<EpsilonDecisionInferenceServer.Prediction> predictions =
            DecisionSampleBatcher.evaluateInBucketedBatches(server, samples);
        if (predictions.size() != samples.size()) {
          throw new IOException(
              "Decision prediction count mismatch: file="
                  + file
                  + " expected="
                  + samples.size()
                  + " actual="
                  + predictions.size());
        }
        for (int index = 0; index < samples.size(); index++) {
          EpsilonDecisionSample sample = samples.get(index);
          totals.add(
              predictions.get(index).valueUtility(),
              sample.valueTarget(),
              grpBacked[index],
              sample.finalRank(),
              EpsilonUtilityProfile.values()[sample.ruleProfile()]);
        }
        totals.files++;
      }
    }
    return totals.toMetrics();
  }

  /** 最終局先頭部分のワンホット代替処理を除き、実際に次局境界GRPを持つサンプルだけを識別する。 */
  static boolean[] grpBackedSamples(List<EpsilonDecisionSample> samples) {
    HashMap<Long, Integer> maxStepsByGame = new HashMap<>();
    for (EpsilonDecisionSample sample : samples) {
      int steps = EpsilonGrpFeature.steps(sample.grpFeatureSequenceView());
      maxStepsByGame.merge(sample.gameId(), steps, Math::max);
    }
    boolean[] backed = new boolean[samples.size()];
    for (int index = 0; index < samples.size(); index++) {
      EpsilonDecisionSample sample = samples.get(index);
      int steps = EpsilonGrpFeature.steps(sample.grpFeatureSequenceView());
      backed[index] = steps < maxStepsByGame.get(sample.gameId());
    }
    return backed;
  }

  private static void validateCheckpointSpecs(List<CheckpointSpec> specs) throws IOException {
    Set<String> labels = new HashSet<>();
    Set<Path> paths = new HashSet<>();
    for (CheckpointSpec raw : specs) {
      CheckpointSpec spec = raw.normalized();
      if (!labels.add(spec.label())) {
        throw new IOException("Duplicate audit checkpoint label: " + spec.label());
      }
      Path real = spec.path().toRealPath();
      if (!paths.add(real)) {
        throw new IOException("Duplicate audit checkpoint path: " + real);
      }
      EpsilonDecisionCheckpointManager.requireValidCheckpoint(real);
    }
  }

  /**
   * 監査レポートで使うチェックポイントの表示名とパス。
   *
   * @param label レポートマップで一意な安全な表示名
   * @param path 評価対象チェックポイントパス
   */
  public record CheckpointSpec(String label, Path path) {

    /**
     * 表示名とチェックポイントパスを検証して監査対象を作る。
     *
     * @param label レポート内で一意になる英数字主体の表示名
     * @param path 読み取る Decision チェックポイント
     */
    public CheckpointSpec {
      if (label == null || !label.matches("[A-Za-z0-9._-]+")) {
        throw new IllegalArgumentException("Invalid checkpoint label: " + label);
      }
      Objects.requireNonNull(path, "path");
    }

    /**
     * {@code label=path} 形式の CLI 引数を監査対象へ変換する。
     *
     * @param value チェックポイントの表示名とパス
     * @return 検証済みの監査対象
     * @throws IllegalArgumentException 区切りまたは値が不正な場合
     */
    public static CheckpointSpec parse(String value) {
      int equals = value == null ? -1 : value.indexOf('=');
      if (equals <= 0 || equals == value.length() - 1) {
        throw new IllegalArgumentException("Checkpoint must be label=path: " + value);
      }
      return new CheckpointSpec(value.substring(0, equals), Path.of(value.substring(equals + 1)));
    }

    CheckpointSpec normalized() {
      return new CheckpointSpec(label, path.toAbsolutePath().normalize());
    }
  }

  record Selection(int eligibleValidationFiles, List<Path> selectedFiles) {}

  /**
   * 同一検証ファイル集合で複数 Decision 価値を GRP／終局教師へ照合したレポート。
   *
   * @param schema レポートスキーマ ID
   * @param generatedAt 生成時刻
   * @param training 学習を行ったか
   * @param promotionMutation 採用状態を変更したか
   * @param device 推論デバイス
   * @param inputFileManifest 入力ファイルマニフェストパス
   * @param validationFraction 検証データの分割比率
   * @param eligibleValidationFiles 条件を満たした検証ファイル数
   * @param requestedValidationFiles 要求したファイル数
   * @param selectedValidationFiles 実際に選択したファイル数
   * @param selectedFiles 選択した相対ファイル名
   * @param grpDecisionCheckpointDir GRP 教師モデル比較局面
   * @param grpCheckpoint GRP チェックポイントパス
   * @param grpIteration GRP チェックポイント反復回数
   * @param baselineLabel 差の基準チェックポイントラベル
   * @param samples 全 Decision サンプル数
   * @param grpTeacherSamples GRP 教師モデルを利用できたサンプル数
   * @param checkpoints チェックポイントごとの評価結果
   */
  public record AuditReport(
      String schema,
      String generatedAt,
      boolean training,
      boolean promotionMutation,
      String device,
      String inputFileManifest,
      float validationFraction,
      int eligibleValidationFiles,
      int requestedValidationFiles,
      int selectedValidationFiles,
      List<String> selectedFiles,
      String grpDecisionCheckpointDir,
      String grpCheckpoint,
      int grpIteration,
      String baselineLabel,
      long samples,
      long grpTeacherSamples,
      List<CheckpointAudit> checkpoints) {}

  /**
   * 一つの Decision チェックポイントの識別情報と価値指標。
   *
   * @param label レポート内の表示名
   * @param path チェックポイントパス
   * @param globalStep チェックポイント累積更新回数
   * @param iteration チェックポイント反復回数
   * @param selfPlayGames チェックポイントまでの自己対局対局数
   * @param metrics 絶対価値／GRP 指標
   * @param deltaFromBaseline 比較基準からの差
   */
  public record CheckpointAudit(
      String label,
      String path,
      int globalStep,
      int iteration,
      int selfPlayGames,
      ValueGrpMetrics metrics,
      ValueGrpDelta deltaFromBaseline) {}

  /** Scalar 価値の重みを固定したGRP期待効用と実順位効用に対する誤差。 */
  public record ValueGrpMetrics(
      int files,
      long samples,
      long grpTeacherSamples,
      long finalBoundaryFallbackSamples,
      double teacherUtilityBias,
      double teacherUtilityMse,
      double teacherUtilityRmse,
      Double teacherUtilityPearson,
      Double teacherUtilityCalibrationSlope,
      Double teacherUtilityCalibrationIntercept,
      double predictedUtilityMean,
      double predictedUtilityStd,
      double teacherUtilityMean,
      double teacherUtilityStd,
      double actualUtilityBias,
      double actualUtilityMse,
      double actualUtilityRmse,
      double valueUtilityCenteredRms) {}

  /** 現在のチェックポイントの効用指標から比較基準指標を引いた差。 */
  public record ValueGrpDelta(
      double teacherUtilityBias,
      double teacherUtilityMse,
      Double teacherUtilityPearson,
      Double teacherUtilityCalibrationSlope,
      double actualUtilityBias,
      double actualUtilityMse) {
    static ValueGrpDelta between(ValueGrpMetrics current, ValueGrpMetrics baseline) {
      return new ValueGrpDelta(
          current.teacherUtilityBias() - baseline.teacherUtilityBias(),
          current.teacherUtilityMse() - baseline.teacherUtilityMse(),
          subtract(current.teacherUtilityPearson(), baseline.teacherUtilityPearson()),
          subtract(
              current.teacherUtilityCalibrationSlope(), baseline.teacherUtilityCalibrationSlope()),
          current.actualUtilityBias() - baseline.actualUtilityBias(),
          current.actualUtilityMse() - baseline.actualUtilityMse());
    }

    private static Double subtract(Double current, Double baseline) {
      return current == null || baseline == null ? null : current - baseline;
    }
  }

  static final class Totals {
    int files;
    long samples;
    long grpTeacherSamples;
    double teacherUtilityErrorSum;
    double teacherUtilitySquaredErrorSum;
    final BivariateMoments teacherUtilityMoments = new BivariateMoments();
    double actualUtilityErrorSum;
    double actualUtilitySquaredErrorSum;
    final UnivariateMoments utilityMoments = new UnivariateMoments();

    void add(
        float prediction,
        float teacher,
        boolean grpBacked,
        int finalRank,
        EpsilonUtilityProfile profile) {
      if (!Float.isFinite(prediction) || !Float.isFinite(teacher)) {
        throw new IllegalArgumentException("audit utility values must be finite");
      }
      double actualUtility = profile.utilityForRank(finalRank);
      samples++;
      utilityMoments.add(prediction);
      double actualUtilityError = prediction - actualUtility;
      actualUtilityErrorSum += actualUtilityError;
      actualUtilitySquaredErrorSum += actualUtilityError * actualUtilityError;
      if (grpBacked) {
        grpTeacherSamples++;
        double teacherUtilityError = prediction - (double) teacher;
        teacherUtilityErrorSum += teacherUtilityError;
        teacherUtilitySquaredErrorSum += teacherUtilityError * teacherUtilityError;
        teacherUtilityMoments.add(prediction, teacher);
      }
    }

    ValueGrpMetrics toMetrics() {
      if (samples <= 0L || grpTeacherSamples <= 0L) {
        throw new IllegalStateException(
            "Value-GRP audit requires samples with next-boundary GRP teachers");
      }
      double actualMse = actualUtilitySquaredErrorSum / samples;
      double teacherMse = teacherUtilitySquaredErrorSum / grpTeacherSamples;
      return new ValueGrpMetrics(
          files,
          samples,
          grpTeacherSamples,
          samples - grpTeacherSamples,
          teacherUtilityErrorSum / grpTeacherSamples,
          teacherMse,
          Math.sqrt(teacherMse),
          teacherUtilityMoments.pearson(),
          teacherUtilityMoments.calibrationSlope(),
          teacherUtilityMoments.calibrationIntercept(),
          teacherUtilityMoments.meanX(),
          teacherUtilityMoments.stdX(),
          teacherUtilityMoments.meanY(),
          teacherUtilityMoments.stdY(),
          actualUtilityErrorSum / samples,
          actualMse,
          Math.sqrt(actualMse),
          Math.sqrt(utilityMoments.variance()));
    }
  }

  static final class BivariateMoments {
    long count;
    double sumX;
    double sumY;
    double sumXX;
    double sumYY;
    double sumXY;

    void add(double x, double y) {
      count++;
      sumX += x;
      sumY += y;
      sumXX += x * x;
      sumYY += y * y;
      sumXY += x * y;
    }

    Double meanX() {
      return count <= 0L ? null : sumX / count;
    }

    Double meanY() {
      return count <= 0L ? null : sumY / count;
    }

    Double stdX() {
      return standardDeviation(sumX, sumXX);
    }

    Double stdY() {
      return standardDeviation(sumY, sumYY);
    }

    Double pearson() {
      if (count <= 1L) {
        return null;
      }
      double centeredXX = centered(sumX, sumXX);
      double centeredYY = centered(sumY, sumYY);
      if (!(centeredXX > 0.0 && centeredYY > 0.0)) {
        return null;
      }
      double centeredXY = sumXY - sumX * sumY / count;
      return centeredXY / Math.sqrt(centeredXX * centeredYY);
    }

    /** 教師値 = 切片 + 傾き × 予測値という式に当てはめる校正回帰。 */
    Double calibrationSlope() {
      if (count <= 1L) {
        return null;
      }
      double centeredXX = centered(sumX, sumXX);
      if (!(centeredXX > 0.0)) {
        return null;
      }
      return (sumXY - sumX * sumY / count) / centeredXX;
    }

    Double calibrationIntercept() {
      Double slope = calibrationSlope();
      return slope == null ? null : meanY() - slope * meanX();
    }

    private Double standardDeviation(double sum, double sumSquares) {
      if (count <= 0L) {
        return null;
      }
      return Math.sqrt(Math.max(0.0, centered(sum, sumSquares) / count));
    }

    private double centered(double sum, double sumSquares) {
      return sumSquares - sum * sum / count;
    }
  }

  private static final class UnivariateMoments {
    long count;
    double sum;
    double sumSquares;

    void add(double value) {
      count++;
      sum += value;
      sumSquares += value * value;
    }

    double variance() {
      if (count <= 0L) {
        return 0.0;
      }
      double mean = sum / count;
      return Math.max(0.0, sumSquares / count - mean * mean);
    }
  }
}
