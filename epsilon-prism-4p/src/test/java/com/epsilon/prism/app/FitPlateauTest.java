package com.epsilon.prism.app;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;

import ai.djl.Device;
import com.epsilon.agent.app.Span;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.play.PublishedModel;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class FitPlateauTest {
  @Test
  public void cumulativeGainUsesTheRecordedPreviousR2ForEveryIteration() {
    var tempo = TestSettings.of().noise().tempo();
    double first = tempo.gain(761, 0.22), second = tempo.gain(762, 0.28);
    double expected = 1 - (1 - first) * (1 - second);
    var schedule =
        new FitPlateau.Schedule(
            761, List.of(0.22, 0.28), List.of(first, second), expected, 0.0012, 6e-5);
    Assert.assertEquals(schedule.combined(Span.of("cohorts", "761..762"), tempo), expected);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> schedule.combined(Span.of("cohorts", "760..761"), tempo));
    var wrong =
        new FitPlateau.Schedule(761, List.of(0.28, 0.22), schedule.gains(), expected, 0.0012, 6e-5);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> wrong.combined(Span.of("cohorts", "761..762"), tempo));
  }

  @Test
  public void pooledGainControlsTheTeacherAndUsesTheCappedLineageLearningRate() {
    var tempo = TestSettings.of().noise().tempo();
    double history = 0.43104149620680277;
    Assert.assertEquals(FitPlateau.Update.of(history, null, null, tempo).teacherGain(), history);
    for (double gain : new double[] {0.25, history, 0.69, 1.0, 0.125}) {
      var update = FitPlateau.Update.of(history, gain, null, tempo);
      Assert.assertEquals(update.teacherGain(), gain);
      Assert.assertEquals(update.scheduledGain(), history);
      Assert.assertEquals(update.lrScale(), Math.min(1.0, gain / 0.25));
    }
    for (double bad : new double[] {0, -0.1, 1.01, Double.NaN, Double.POSITIVE_INFINITY}) {
      Assert.expectThrows(
          IllegalArgumentException.class, () -> FitPlateau.Update.of(history, bad, null, tempo));
    }
  }

  @Test
  public void explicitLearningRateScaleLeavesTheTeacherGainAlone() {
    var settings = TestSettings.of();
    double history = 0.43104149620680277;
    double scale = 0.1385311808692502;
    var update = FitPlateau.Update.of(history, null, scale, settings.noise().tempo());
    Assert.assertEquals(update.teacherGain(), history);
    Assert.assertEquals(update.lrScale(), scale);
    var learner = settings.fit().fit(12597, update.lrScale());
    Assert.assertEquals(learner.lrPeak(), (float) (settings.fit().lrPeak() * scale), 1e-10);
    Assert.assertEquals(learner.lrFloor(), (float) (settings.fit().lrFloor() * scale), 1e-11);
    for (double bad : new double[] {0, -0.1, Double.NaN, Double.POSITIVE_INFINITY}) {
      Assert.expectThrows(
          IllegalArgumentException.class,
          () -> FitPlateau.Update.of(history, null, bad, settings.noise().tempo()));
    }
  }

  @Test
  public void copiedV760PreservesPublishedEagerScores() throws Exception {
    checkPublishedCopy(false);
  }

  @Test(groups = "rocm")
  public void copiedV760PreservesPublishedHipScores() throws Exception {
    checkPublishedCopy(true);
  }

  private void checkPublishedCopy(boolean hip) throws Exception {
    var reference = PublishedModel.fourPlayer(760);
    var original = Snapshot.model(hip ? reference.requiredSnapshot() : reference.snapshot());
    var root = Files.createTempDirectory("plateau-v760");
    try {
      var shape = PrismNetwork.readShape(original);
      var target = root.resolve("copy");
      try (var model = Fit.load(original, Device.cpu())) {
        Snapshot.write(target, (PrismNetwork) model.getBlock(), PolicyParams.of(760));
      }
      if (hip) reference.assertNativeScores(target, shape);
      else reference.assertEagerScores(target, shape);
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void distilledSnapshotsKeepTheirOwnCoreShapeFormatAndWeights() throws Exception {
    var root = Files.createTempDirectory("plateau-distilled");
    try {
      var shapes =
          List.of(
              ShapeArm.V4_S.shape(),
              ShapeArm.V4_M.shape(),
              ShapeArm.V3.shape(),
              new PrismShape.V4(256, 512, 2, PrismShape.EventAttention.BIDIRECTIONAL));
      for (var shape : shapes) {
        var path = root.resolve("shape-" + shapes.indexOf(shape));
        try (var source = Fit.blank(shape.inputFormat(FOUR_PLAYER), shape, Device.cpu(), 11)) {
          ((PrismNetwork) source.getBlock()).save(path);
        }
        // 開始モデルの重みの配置を読み込み後も保つ。
        var header = PrismNetwork.readHeader(path);
        try (var loaded = Fit.load(path, Device.cpu())) {
          var target = path.resolveSibling(path.getFileName() + "-loaded");
          ((PrismNetwork) loaded.getBlock()).save(target);
          Assert.assertEquals(PrismNetwork.readHeader(target), header);
          Assert.assertEquals(
              Files.mismatch(path.resolve("weights.bin"), target.resolve("weights.bin")), -1L);
        }
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
