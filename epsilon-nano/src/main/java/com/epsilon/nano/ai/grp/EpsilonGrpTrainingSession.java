package com.epsilon.nano.ai.grp;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.grp.EpsilonGrpCheckpointBundle;
import com.epsilon.ai.grp.EpsilonGrpExample;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.ai.grp.EpsilonGrpTeacherIdentity;
import com.epsilon.ai.grp.EpsilonGrpTrainer;
import com.epsilon.config.settings.GrpInferenceSettings;
import com.epsilon.config.settings.GrpSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.util.FormatUtils;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 事前学習と自己対局で使う単体 GRP モデルのライフサイクルを管理する。 */
public final class EpsilonGrpTrainingSession implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(EpsilonGrpTrainingSession.class);

  private final Path checkpointDir;
  private final Model model;
  private final GrpSettings trainingSettings;
  private EpsilonGrpTrainer trainer;
  private final EpsilonGrpInference inference;
  private Path identityCheckpoint;
  private String identityCheckpointSha256;
  private int globalStep;
  private int iteration;

  private EpsilonGrpTrainingSession(Path checkpointDir) {
    this.checkpointDir = checkpointDir;
    this.model = null;
    this.trainingSettings = null;
    this.trainer = null;
    this.inference = null;
    this.identityCheckpoint = null;
    this.identityCheckpointSha256 = "";
    this.globalStep = 0;
    this.iteration = 0;
  }

  private EpsilonGrpTrainingSession(
      Path checkpointDir,
      Model model,
      int globalStep,
      int iteration,
      boolean trainable,
      Path identityCheckpoint,
      String identityCheckpointSha256,
      SettingsLoader settings) {
    this.trainingSettings = settings.bind(GrpSettings.class);
    this.checkpointDir = checkpointDir;
    this.model = model;
    this.trainer = trainable ? newTrainer(model, trainingSettings) : null;
    this.inference =
        new EpsilonGrpInference(model, settings.bind(GrpInferenceSettings.class).maxBatch());
    this.identityCheckpoint = identityCheckpoint;
    this.identityCheckpointSha256 = identityCheckpointSha256;
    this.globalStep = globalStep;
    this.iteration = iteration;
  }

  /**
   * 学習用セッションを開く。チェックポイントがなければ新規モデルを作成する。
   *
   * @param decisionCheckpointDir GRP のサブディレクトリを持つ Decision チェックポイントのルートディレクトリ
   * @param device モデルを配置するデバイス
   * @return オプティマイザー状態を持つ学習可能セッション
   * @throws IOException 既存チェックポイントの検証または復元に失敗した場合
   */
  public static EpsilonGrpTrainingSession openForTraining(Path decisionCheckpointDir, Device device)
      throws IOException {
    return openForTraining(decisionCheckpointDir, device, EpsilonSettings.defaults());
  }

  public static EpsilonGrpTrainingSession openForTraining(
      Path decisionCheckpointDir, Device device, SettingsLoader settings) throws IOException {
    Path grpDir = decisionCheckpointDir.resolve("grp");
    Path existing = EpsilonGrpCheckpointManager.resolveExisting(grpDir);
    if (existing == null) {
      var config = settings.bind(GrpSettings.class);
      Model model = NetworkFactory.createGrpModel(device, config.hidden(), config.layers());
      log.info("GRP session created new Epsilon model: checkpointDir={}", grpDir);
      return new EpsilonGrpTrainingSession(grpDir, model, 0, 0, true, null, "", settings);
    }
    return loadExisting(grpDir, existing, device, true, settings);
  }

  /**
   * 方策モデル用の推論専用セッションを開く。オプティマイザーと学習処理状態は確保しない。
   *
   * @param decisionCheckpointDir GRP のサブディレクトリを持つ Decision チェックポイントのルートディレクトリ
   * @param device モデルを配置するデバイス
   * @return 固定したチェックポイントを読む推論専用セッション
   * @throws IOException 有効な GRP チェックポイントが存在しないか復元に失敗した場合
   */
  public static EpsilonGrpTrainingSession openInferenceOnly(
      Path decisionCheckpointDir, Device device) throws IOException {
    return openInferenceOnly(decisionCheckpointDir, device, EpsilonSettings.defaults());
  }

  public static EpsilonGrpTrainingSession openInferenceOnly(
      Path decisionCheckpointDir, Device device, SettingsLoader settings) throws IOException {
    Path grpDir = decisionCheckpointDir.resolve("grp");
    Path existing = EpsilonGrpCheckpointManager.resolveExisting(grpDir);
    if (existing == null) {
      throw new IOException(
          "GRP checkpoint does not exist at " + grpDir + ". Run pretrain-grp-logs first.");
    }
    return loadExisting(grpDir, existing, device, false, settings);
  }

  private static EpsilonGrpTrainingSession loadExisting(
      Path grpDir, Path existing, Device device, boolean trainable, SettingsLoader settings)
      throws IOException {
    Path exact = existing.toAbsolutePath().normalize();
    String checkpointSha256 = EpsilonGrpCheckpointManager.checkpointSha256(exact);
    EpsilonGrpCheckpointBundle manifest = EpsilonGrpCheckpointManager.loadManifest(exact);
    Model model = EpsilonGrpCheckpointManager.load(exact, device);
    EpsilonGrpTrainingSession session =
        new EpsilonGrpTrainingSession(
            grpDir,
            model,
            manifest.globalStep,
            manifest.iteration,
            trainable,
            exact,
            checkpointSha256,
            settings);
    try {
      if (trainable) {
        EpsilonGrpCheckpointManager.loadOptimizerState(session.trainer, exact);
      }
    } catch (IOException | RuntimeException e) {
      session.close();
      throw e;
    }
    log.info(
        "GRP session loaded checkpoint: checkpoint={} step={} iteration={}",
        exact,
        manifest.globalStep,
        manifest.iteration);
    return session;
  }

  /**
   * モデルを持たず、学習・評価を何も行わないとするセッションを作る。
   *
   * @param grpDir 将来の GRP チェックポイントディレクトリ
   * @return 無効なセッション
   */
  public static EpsilonGrpTrainingSession disabled(Path grpDir) {
    return new EpsilonGrpTrainingSession(grpDir);
  }

  /**
   * セッションが GRP モデルを保持しているかを返す。
   *
   * @return 推論可能なら {@code true}
   */
  public boolean enabled() {
    return model != null;
  }

  /**
   * セッションが所有する同期推論 API を返す。
   *
   * @return 推論。無効なセッションでは {@code null}
   */
  public EpsilonGrpInference inference() {
    return inference;
  }

  /**
   * 学習例を学習し、実行したオプティマイザーバッチ数だけ累積更新回数を進める。
   *
   * @param examples 一意化済み GRP 学習例
   * @param epochs 全学習例を反復する回数
   * @return 学習指標。無効なセッションでは空
   */
  public EpsilonGrpTrainer.TrainMetrics trainExamples(
      List<EpsilonGrpExample> examples, int epochs) {
    if (!enabled()) {
      return EpsilonGrpTrainer.TrainMetrics.empty();
    }
    requireTrainable();
    EpsilonGrpTrainer.TrainMetrics metrics = trainer.trainExamples(examples, epochs);
    globalStep += metrics.batches();
    return metrics;
  }

  /**
   * 現在のパラメーターで学習例を評価する。
   *
   * @param examples 評価する GRP 学習例
   * @return 検証指標。無効なセッションでは空
   */
  public EpsilonGrpInference.EvalMetrics evaluateExamples(List<EpsilonGrpExample> examples) {
    if (!enabled()) {
      return EpsilonGrpInference.EvalMetrics.empty();
    }
    return inference.evaluateExamples(examples);
  }

  /**
   * 現在のパラメーターを新しい変更不可世代と latest へ保存する。
   *
   * @param prefix 世代ディレクトリ名の接頭辞
   * @param nextIteration 保存する世代番号
   * @throws IOException 変更不可世代が既存か保存に失敗した場合
   */
  public void saveAt(String prefix, int nextIteration) throws IOException {
    if (!enabled()) {
      return;
    }
    requireTrainable();
    Path iterationDir = checkpointDir.resolve(prefix + "_" + FormatUtils.zeroPad(nextIteration, 5));
    EpsilonGrpCheckpointManager.saveImmutable(
        model, trainer, iterationDir, globalStep, nextIteration);
    iteration = nextIteration;
    EpsilonGrpCheckpointManager.save(
        model, trainer, EpsilonGrpCheckpointManager.latest(checkpointDir), globalStep, iteration);
    identityCheckpoint = iterationDir.toAbsolutePath().normalize();
    identityCheckpointSha256 = EpsilonGrpCheckpointManager.checkpointSha256(identityCheckpoint);
  }

  /**
   * 現在の反復回数の次世代として保存する。
   *
   * @param prefix 世代ディレクトリ名の接頭辞
   * @throws IOException チェックポイント保存に失敗した場合
   */
  public void saveNext(String prefix) throws IOException {
    saveAt(prefix, iteration + 1);
  }

  /**
   * 検証で棄却した候補を最新の昇格済みチェックポイントへ戻す。
   *
   * @throws IOException 復元先が存在しないか保存物の読み込みに失敗した場合
   */
  public void restoreLatest() throws IOException {
    if (!enabled()) {
      return;
    }
    requireTrainable();
    Path existing = EpsilonGrpCheckpointManager.resolveExisting(checkpointDir);
    if (existing == null) {
      throw new IOException("Promoted GRP checkpoint not found: " + checkpointDir);
    }
    EpsilonGrpCheckpointBundle manifest = EpsilonGrpCheckpointManager.loadManifest(existing);
    trainer.close();
    trainer = null;
    EpsilonGrpCheckpointManager.loadInto(model, existing);
    trainer = newTrainer(model, trainingSettings);
    EpsilonGrpCheckpointManager.loadOptimizerState(trainer, existing);
    globalStep = manifest.globalStep;
    iteration = manifest.iteration;
    identityCheckpoint = existing.toAbsolutePath().normalize();
    identityCheckpointSha256 = EpsilonGrpCheckpointManager.checkpointSha256(identityCheckpoint);
    log.info(
        "GRP candidate rejected and restored: checkpoint={} step={} iteration={}",
        existing,
        globalStep,
        iteration);
  }

  /**
   * 現在のオプティマイザー更新累計を返す。
   *
   * @return 累積更新回数
   */
  public int globalStep() {
    return globalStep;
  }

  /**
   * 現在のチェックポイント世代番号を返す。
   *
   * @return 反復回数
   */
  public int iteration() {
    return iteration;
  }

  /**
   * 学習データファイルヘッダーに保存する、現在開いている固定した教師モデルの内容識別情報を返す。
   *
   * @return チェックポイント世代と推論保存物 SHA-256
   * @throws IOException 有効なセッションが保存済みチェックポイント識別情報を持たない場合
   */
  public EpsilonGrpTeacherIdentity identity() throws IOException {
    if (!enabled()) {
      return EpsilonGrpTeacherIdentity.disabled();
    }
    if (identityCheckpoint == null || identityCheckpointSha256.isBlank()) {
      throw new IOException("GRP checkpoint identity is unavailable: " + checkpointDir);
    }
    return EpsilonGrpTeacherIdentity.enabled(iteration, identityCheckpointSha256);
  }

  private void requireTrainable() {
    if (trainer == null) {
      throw new IllegalStateException("GRP session is inference-only");
    }
  }

  @Override
  public void close() {
    if (trainer != null) {
      trainer.close();
    }
    if (model != null) {
      model.close();
    }
  }

  private static EpsilonGrpTrainer newTrainer(Model model, GrpSettings settings) {
    return new EpsilonGrpTrainer(
        model,
        settings.batchSize(),
        settings.learningRate(),
        settings.weightDecay(),
        settings.gradClip());
  }
}
