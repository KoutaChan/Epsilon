package com.epsilon.prism.luck;

import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * GPU の煙試験（rocm の組）。{@link LuckCohorts} の打ち方で 1 コホートの標本を集め、幅 256 の白紙の Ω を GPU 0・BF16 で当てる。
 *
 * <ol>
 *   <li>既定の計画（塊 8192、学習率 2e-4 一定、減衰 0.01、保留 5%）で、Y = 局末の点。損失と保留の R² が有限で、本番の塊の 1 歩の時間を出す。
 *   <li>Y を結果の関数にした同じ標本（{@link LuckCohorts#outcomeTarget}）で、保留の R² が 1 に近づく（BF16 の道で結果の軸の並びがそろって
 *       いる）。
 * </ol>
 */
public class LuckFitGpuSmokeTest {
  private static final Logger log = LoggerFactory.getLogger(LuckFitGpuSmokeTest.class);

  private static final long SEED = 52_000;
  private static final int GAMES = Integer.getInteger("luck.games", 400);
  private static final double RATE = 0.5;
  private static final int WIDTH = 256;

  @Test(groups = "rocm")
  public void fitsOnGpu() throws IOException {
    if (Engine.getInstance().getGpuCount() == 0) {
      throw new SkipException("no GPU");
    }
    Path root = Files.createTempDirectory("luck-fit-gpu");
    try {
      Path real = LuckCohorts.write(root.resolve("1"), SEED, GAMES, RATE);
      Path outcome = LuckCohorts.outcomeTarget(real, root.resolve("2"));
      LuckFit.Report standard =
          fit("score", LuckSamples.load(List.of(real)), LuckFit.Plan.standard(WIDTH, 4, 3));
      assertTrue(Double.isFinite(standard.lastLoss()));
      assertTrue(Double.isFinite(standard.holdout().all().r2()));
      LuckFit.Report learned =
          fit(
              "outcome",
              LuckSamples.load(List.of(outcome)),
              new LuckFit.Plan(LuckNetwork.shape(WIDTH), 3, 1024, 1e-3f, 0.01f, 0.05, 5));
      double r2 = learned.holdout().all().r2();
      assertTrue(r2 > 0.5, "holdout R2 " + r2);
    } finally {
      try (var files = Files.walk(root)) {
        for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
          Files.delete(file);
        }
      }
    }
  }

  private static LuckFit.Report fit(String target, LuckSamples samples, LuckFit.Plan plan) {
    try (NDManager manager = NDManager.newBaseManager(Device.gpu(0))) {
      LuckNetwork network = LuckNetwork.blank(plan.shape(), manager, plan.seed());
      LuckFit.Report report = LuckFit.run(samples, network, manager, plan, 8);
      log.info(
          "LUCK-FIT-GPU target={} games={} samples={} train={} batch={} steps={} loss {} -> {}"
              + " holdout={} R2={} reduction={} byEvent={} actorMeanZ={} fit={}s ms/step={}",
          target,
          GAMES,
          samples.size(),
          report.trainSamples(),
          plan.batchSize(),
          report.steps(),
          String.format("%.5f", report.firstLoss()),
          String.format("%.5f", report.lastLoss()),
          report.holdout().all().samples(),
          String.format("%.4f", report.holdout().all().r2()),
          String.format("%.5f", report.holdout().all().reduction()),
          report.holdout().byEvent(),
          String.format("%.2f", report.holdout().actorMeanZ()),
          String.format("%.1f", report.seconds()),
          String.format("%.1f", report.seconds() * 1e3 / report.steps()));
      return report;
    }
  }
}
