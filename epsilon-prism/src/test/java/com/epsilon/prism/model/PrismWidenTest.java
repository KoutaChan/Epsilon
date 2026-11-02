package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.Pair;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism.model.WidthPlan.Axis;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 幅を広げる道（CPU、FP32）。F 512→1024 と H 256→384（頭 8→12）で s の差が 1e-5 以下、広げた網の新しいパラメーターに 2 歩までに
 * 勾配が届く。深さ 2 と 3 で確かめる。
 */
public class PrismWidenTest {
  private static final Logger log = LoggerFactory.getLogger(PrismWidenTest.class);

  private static final float TOLERANCE = 1e-5f;

  @DataProvider
  public Object[][] depths() {
    return new Object[][] {{2}, {3}};
  }

  @Test(dataProvider = "depths")
  public void wideningKeepsScores(int depth) {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager scope = manager.newSubManager()) {
      manager.getEngine().setRandomSeed(101);
      PrismNetwork narrow = new PrismNetwork(PrismShape.standard(depth));
      narrow.initialize(manager);
      Batches.perturb(narrow, 0.02f, 103);
      ParameterStore store = new ParameterStore(manager, false);
      Batch batch = Batches.build(Batches.allKinds(6), 107);
      float[] base = scores(narrow, store, scope, batch);
      Assert.assertTrue(Batches.maxAbs(base) > 1e-3f, "scores must be non-trivial");

      PrismNetwork ffn = PrismWiden.widen(narrow, narrow.shape().withFfn(1024), manager);
      assertClose(base, scores(ffn, store, scope, batch), "F 512 -> 1024");
      PrismNetwork hidden = PrismWiden.widen(narrow, narrow.shape().withHidden(384), manager);
      Assert.assertEquals(hidden.shape().heads(), 12);
      assertClose(base, scores(hidden, store, scope, batch), "H 256 -> 384");
      PrismNetwork both = PrismWiden.widen(ffn, ffn.shape().withHidden(384), manager);
      assertClose(base, scores(both, store, scope, batch), "F then H");
    }
  }

  @Test
  public void newParametersReceiveGradientWithinTwoSteps() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu());
        NDManager scope = manager.newSubManager()) {
      manager.getEngine().setRandomSeed(109);
      PrismNetwork narrow = new PrismNetwork(PrismShape.standard(2));
      narrow.initialize(manager);
      Batches.perturb(narrow, 0.02f, 113);
      PrismNetwork wide =
          PrismWiden.widen(narrow, PrismShape.standard(2).withHidden(384).withFfn(1024), manager);
      ParameterStore store = new ParameterStore(manager, false);
      Batch batch = Batches.build(Batches.allKinds(6), 127);
      List<ValueLabel> labels = Batches.values(batch, 131);
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 137);
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      Optimizer optimizer = Batches.adam(1e-3f);
      WidthPlan plan = wide.widthPlan();
      Set<String> reached = new HashSet<>();
      Set<String> grown = new HashSet<>();
      for (int step = 0; step < 2; step++) {
        try (NDManager stepScope = scope.newSubManager();
            GradientCollector collector = manager.getEngine().newGradientCollector()) {
          Outputs outputs =
              wide.forward(
                  store, batch.inputs(stepScope), Pass.training(Precision.FLOAT32, measured, auxiliary));
          collector.backward(loss.total(stepScope, outputs, labels, auxiliary, 1f, labels.size(), 1f));
        }
        for (Pair<String, Parameter> pair : wide.getParameters()) {
          Shape before = narrow.getParameters().get(pair.getKey()).getArray().getShape();
          NDArray array = pair.getValue().getArray();
          if (before.equals(array.getShape())) {
            continue;
          }
          grown.add(pair.getKey());
          boolean[] fresh = newElements(before, array.getShape(), plan.axes(pair.getKey()));
          try (NDArray gradient = array.getGradient()) {
            float[] values = gradient.toFloatArray();
            for (int i = 0; i < values.length; i++) {
              if (fresh[i] && values[i] != 0) {
                reached.add(pair.getKey());
                break;
              }
            }
          }
        }
        Batches.step(wide, optimizer);
      }
      List<String> missing = new ArrayList<>();
      for (String name : grown) {
        if (!reached.contains(name)) {
          missing.add(name);
        }
      }
      Assert.assertFalse(grown.isEmpty());
      Assert.assertTrue(missing.isEmpty(), "new elements without gradient: " + missing);
    }
  }

  private static float[] scores(
      PrismNetwork network, ParameterStore store, NDManager scope, Batch batch) {
    return network
        .forward(store, batch.inputs(scope), Pass.inference(Precision.FLOAT32))
        .scores()
        .toFloatArray();
  }

  private static void assertClose(float[] expected, float[] actual, String what) {
    Assert.assertEquals(actual.length, expected.length, what);
    float worst = 0;
    for (int i = 0; i < expected.length; i++) {
      worst = Math.max(worst, Math.abs(expected[i] - actual[i]));
    }
    log.info(
        "PRISM-T6 {} max|ds|={} max|s|={}",
        what,
        String.format("%.3g", worst),
        String.format("%.3g", Batches.maxAbs(expected)));
    Assert.assertTrue(worst <= TOLERANCE, what + ": max |ds| = " + worst);
  }

  /** 広げた要素の印（どれかの軸の添字が新しい要素）。 */
  private static boolean[] newElements(Shape before, Shape after, List<Axis> axes) {
    int dimensions = after.dimension();
    boolean[][] fresh = new boolean[dimensions][];
    for (int axis = 0; axis < dimensions; axis++) {
      int blocks = axes.get(axis).blocks();
      int oldBlock = (int) before.get(axis) / blocks;
      int newBlock = (int) after.get(axis) / blocks;
      fresh[axis] = new boolean[(int) after.get(axis)];
      for (int index = 0; index < fresh[axis].length; index++) {
        fresh[axis][index] = index % newBlock >= oldBlock;
      }
    }
    boolean[] result = new boolean[(int) after.size()];
    int[] position = new int[dimensions];
    for (int element = 0; element < result.length; element++) {
      for (int axis = 0; axis < dimensions; axis++) {
        result[element] |= fresh[axis][position[axis]];
      }
      for (int axis = dimensions - 1; axis >= 0; axis--) {
        if (++position[axis] < after.get(axis)) {
          break;
        }
        position[axis] = 0;
      }
    }
    return result;
  }
}
