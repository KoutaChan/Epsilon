package com.epsilon.prism.run;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.collect.GrpPhi;
import com.epsilon.prism.collect.HandValue;
import com.epsilon.prism.collect.PhiCalibration;
import com.epsilon.prism.grp.ActivePhi;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiEntry;
import com.epsilon.prism.grp.PhiEntryKind;
import com.epsilon.prism.grp.PhiResolver;
import com.epsilon.prism.grp.PhiSource;
import com.epsilon.prism.grp.ScorePhi;
import com.epsilon.spi.PolicyExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * train の系譜（{@link Lineage}）が反復をまたいで 1 つずつ持つ収集と評価の資源: 符号化と GPU の実行の資源、収集のドライバ、Φ（{@link
 * PhiSource}: FIXED_GRP は固定の GRP、SCORE_RANK・SCORE_SMOOTH は GRP を開かずに局末の点数の順位かなめらかな関数とみなす代用で、どれも全ての
 * 反復で同じ。SELF_GRP は反復ごとに phi.jsonl の項目を解決する {@link PhiResolver}）。閉じる順は開いた順の逆。
 */
public final class TrainingResources implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(TrainingResources.class);

  /** Φ のメモの上限。 */
  private static final int PHI_MEMO = 1 << 16;

  private final PolicyExecutionContext execution;
  private final CollectionDriver driver;

  /** 固定の GRP（FIXED_GRP だけ）。 */
  private final EpsilonGrpTrainingSession grp;

  /** 全ての反復で同じ Φ（FIXED_GRP・SCORE_RANK・SCORE_SMOOTH。SELF_GRP は null）。 */
  private final ActivePhi fixed;

  /** 反復ごとの Φ（SELF_GRP だけ）。 */
  private final PhiResolver resolver;

  private TrainingResources(
      PolicyExecutionContext execution,
      CollectionDriver driver,
      EpsilonGrpTrainingSession grp,
      ActivePhi fixed,
      PhiResolver resolver) {
    this.execution = execution;
    this.driver = driver;
    this.grp = grp;
    this.fixed = fixed;
    this.resolver = resolver;
  }

  /**
   * run の設定から資源を開く（途中で失敗したら開いた分を閉じる）。
   *
   * @param profile 順位効用（Φ と半荘末の効用）
   * @param root run のディレクトリ（SELF_GRP の phi.jsonl）
   * @throws IOException GRP か Φ の較正の表か phi.jsonl が読めないとき
   */
  public static TrainingResources open(PrismConfig config, EpsilonUtilityProfile profile, Path root)
      throws IOException {
    var execution =
        new PolicyExecutionContext(config.encoders(), config.slots(), config.readyBatches());
    CollectionDriver driver = null;
    EpsilonGrpTrainingSession grp = null;
    HandValue value = null;
    PhiResolver resolver = null;
    try {
      driver =
          new CollectionDriver(
              new CollectionDriver.Settings(
                  config.maxActiveGames(),
                  config.advanceWorkers(),
                  new InferenceBatchingSettings(config.batchWaitMicros())));
      ActivePhi fixed = null;
      switch (config.phiSource()) {
        case FIXED_GRP -> {
          Path checkpoint = Path.of(config.grpCheckpoint());
          grp =
              EpsilonGrpTrainingSession.openInferenceOnly(
                  checkpoint, PrismConfig.device(config.grpDevice()));
          value =
              new GrpPhi(
                  grp.inference(),
                  profile,
                  driver.executor(),
                  config.grpBatch(),
                  PHI_MEMO,
                  config.phiTemperatureTable(),
                  config.hasPhiCalibration()
                      ? PhiCalibration.load(Path.of(config.phiCalibration()))
                      : null);
          // openInferenceOnly が読んだのと同じ子（一番新しい有効な checkpoint）と、その時に計算した sha
          Path exact = EpsilonGrpCheckpointManager.resolveExisting(checkpoint.resolve("grp"));
          fixed =
              new ActivePhi(
                  PhiEntry.fixed(config, exact.toString(), grp.identity().checkpointSha256()),
                  value,
                  false);
        }
        case SCORE_RANK -> {
          value = HandValue.scoreRank(profile);
          fixed = new ActivePhi(PhiEntry.scoreRank(), value, false);
          LOG.warn(
              "phiSource = SCORE_RANK: hand ends are valued by the final score ranking, not a GRP");
        }
        case SCORE_SMOOTH -> {
          value = ScorePhi.handValue(PhiEntryKind.SCORE_SMOOTH, profile);
          fixed = new ActivePhi(PhiEntry.scoreSmooth(), value, false);
          LOG.warn(
              "phiSource = SCORE_SMOOTH: hand ends are valued by a smooth function of the scores, not a GRP");
        }
        case SELF_GRP ->
            resolver = PhiResolver.open(config, profile, driver.executor(), new GrpFiles(root));
      }
      return new TrainingResources(execution, driver, grp, fixed, resolver);
    } catch (IOException | RuntimeException | Error failure) {
      try (execution;
          CollectionDriver openedDriver = driver;
          EpsilonGrpTrainingSession openedGrp = grp;
          HandValue openedValue = value;
          PhiResolver openedResolver = resolver) {
        // 開いた分を逆の順に閉じる
      } catch (RuntimeException | Error closing) {
        failure.addSuppressed(closing);
      }
      throw failure;
    }
  }

  /** 符号化と GPU の実行の資源。 */
  public PolicyExecutionContext execution() {
    return execution;
  }

  /** 収集のドライバ（評価もこれで打つ）。 */
  public CollectionDriver driver() {
    return driver;
  }

  /**
   * 反復 t の局末の値 Φ（SELF_GRP 以外は全ての反復で同じ）。
   *
   * @throws IOException SELF_GRP で既に効いている項目の GRP が読めないとき
   */
  public ActivePhi phi(int t) throws IOException {
    return resolver == null ? fixed : resolver.resolve(t);
  }

  /** Φ・GRP・ドライバ・実行の資源の順に閉じる（どれかが失敗しても残りを閉じる）。 */
  @Override
  public void close() {
    try (execution;
        driver;
        grp;
        HandValue value = fixed == null ? null : fixed.value();
        resolver) {
      // 宣言の逆の順に閉じる
    }
  }
}
