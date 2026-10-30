package com.epsilon.prism.run;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.collect.FixedGrpPhi;
import com.epsilon.prism.collect.HandValue;
import com.epsilon.prism.collect.PhiCalibration;
import com.epsilon.spi.PolicyExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * train の系譜（{@link Lineage}）が反復をまたいで 1 つずつ持つ収集と評価の資源: 符号化と GPU の実行の資源、収集のドライバ、
 * Φ（phi = grp なら固定の GRP、score-rank は GRP を開かずに局末の点数の順位とみなす試験用の代用。{@link PrismConfig#validateTraining}）。閉じる
 * 順は開いた順の逆。
 */
public final class TrainingResources implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(TrainingResources.class);

  /** Φ のメモの上限。 */
  private static final int PHI_MEMO = 1 << 16;

  private final PolicyExecutionContext execution;
  private final CollectionDriver driver;

  /** 固定の GRP（score-rank は null）。 */
  private final EpsilonGrpTrainingSession grp;

  private final HandValue phi;

  private TrainingResources(
      PolicyExecutionContext execution,
      CollectionDriver driver,
      EpsilonGrpTrainingSession grp,
      HandValue phi) {
    this.execution = execution;
    this.driver = driver;
    this.grp = grp;
    this.phi = phi;
  }

  /**
   * run の設定から資源を開く（途中で失敗したら開いた分を閉じる）。
   *
   * @param profile 順位効用（Φ と半荘末の効用）
   * @throws IOException GRP か Φ の較正の表が読めないとき
   */
  public static TrainingResources open(PrismConfig config, EpsilonUtilityProfile profile)
      throws IOException {
    var execution =
        new PolicyExecutionContext(config.encoders(), config.slots(), config.readyBatches());
    CollectionDriver driver = null;
    EpsilonGrpTrainingSession grp = null;
    try {
      driver =
          new CollectionDriver(
              new CollectionDriver.Settings(
                  config.maxActiveGames(),
                  config.advanceWorkers(),
                  new InferenceBatchingSettings(config.batchWaitMicros())));
      grp =
          config.phi().equals(PrismConfig.PHI_GRP)
              ? EpsilonGrpTrainingSession.openInferenceOnly(
                  Path.of(config.grpCheckpoint()), PrismConfig.device(config.grpDevice()))
              : null;
      HandValue phi =
          grp == null
              ? HandValue.scoreRank(profile)
              : new FixedGrpPhi(
                  grp.inference(),
                  profile,
                  driver.executor(),
                  config.grpBatch(),
                  PHI_MEMO,
                  config.phiTemperatureTable(),
                  config.hasPhiCalibration()
                      ? PhiCalibration.load(Path.of(config.phiCalibration()))
                      : null);
      if (grp == null)
        LOG.warn(
            "phi = {}: hand ends are valued by the final score ranking, not the fixed GRP (tests"
                + " only)",
            config.phi());
      return new TrainingResources(execution, driver, grp, phi);
    } catch (IOException | RuntimeException | Error failure) {
      try (execution;
          CollectionDriver openedDriver = driver;
          EpsilonGrpTrainingSession openedGrp = grp) {
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

  /** 局末の値 Φ。 */
  public HandValue phi() {
    return phi;
  }

  /** Φ・GRP・ドライバ・実行の資源の順に閉じる（どれかが失敗しても残りを閉じる）。 */
  @Override
  public void close() {
    try (execution;
        driver;
        grp;
        phi) {
      // 宣言の逆の順に閉じる
    }
  }
}
