package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 候補の決定行の順伝播、公開順の勾配、有限差分を確かめる。eager の学習の集め方は行を直接集める演算と同じ勾配になり、どちらの集め方も
 * 自分が読むインデックスだけを載せる。
 */
public class CandidateDecisionRowsTest {
  @Test
  public void roundsOnceAfterAddingInPublishedOrder() {
    int[] ids = {2, 1, 2, 1, 2, 1};
    float[] upstream = {256, 16777216, 1, -16777216, -256, 1};
    for (Precision precision : Precision.values()) {
      check(4, 1, ids, upstream, precision);
      check(4, 1, ids, new float[] {1, 16777216, -256, 1, 256, -16777216}, precision);
    }
  }

  @Test
  public void emptyUniqueAndInterleavedRowsKeepShapeAndGradient() {
    for (Precision precision : Precision.values()) {
      check(3, 5, new int[0], new float[0], precision);
      check(0, 5, new int[0], new float[0], precision);
      check(6, 1, new int[] {5, 1, 3}, new float[] {-0f, 1, -7}, precision);
      for (int[] fixture : new int[][] {{800, 8, 384, 70}, {3127, 5, 17, 31}, {9041, 13, 3, 103}}) {
        Random random = new Random(fixture[0]);
        int[] ids = random.ints(fixture[3], 0, fixture[1] - 1).toArray();
        float[] upstream = new float[ids.length * fixture[2]];
        for (int i = 0; i < upstream.length; i++) {
          upstream[i] = Math.scalb((float) (random.nextInt(255) - 127), random.nextInt(30) - 15);
        }
        check(fixture[1], fixture[2], ids, upstream, precision);
      }
    }
  }

  /**
   * eager の学習の集め方は、行を直接集める演算と同じ値と勾配を出す。公開順に足す集め方も同じ値を出し、和が FP32 で丸めを生まない
   * 入力では勾配もビットで一致する。
   */
  @Test
  public void bothTrainingPathsGatherTheSameRows() {
    int[] ids = {2, 1, 2, 1, 2, 1, 0, 2};
    int rows = 3;
    int width = 4;
    float[] upstream = new float[ids.length * width];
    for (int i = 0; i < upstream.length; i++) {
      upstream[i] = (i * 37 % 97) - 48;
    }
    for (Precision precision : Precision.values()) {
      Gathered direct = gather(rows, width, ids, upstream, precision, null);
      Gathered eager = gather(rows, width, ids, upstream, precision, false);
      Gathered ordered = gather(rows, width, ids, upstream, precision, true);
      Assert.assertEquals(eager.values(), direct.values());
      Assert.assertEquals(ordered.values(), direct.values());
      Assert.assertEquals(eager.gradient(), direct.gradient());
      if (precision == Precision.FLOAT32) {
        Assert.assertEquals(ordered.gradient(), eager.gradient());
      }
    }
  }

  /** eager の学習は候補ごとの行だけを、公開順に足す学習は同じ行の候補があるときだけ層と並べ直しの添字を載せる。 */
  @Test
  public void eachTrainingPathStagesOnlyTheIndicesItReads() {
    for (int[] ids : new int[][] {{2, 1, 2, 1, 2, 1}, {5, 1, 3}, {0, 0, 0, 0}, {4}}) {
      int rows = Arrays.stream(ids).max().orElseThrow() + 1;
      boolean repeated = Arrays.stream(ids).distinct().count() < ids.length;
      var eager = new Staging();
      CandidateDecisionRows.stage(eager, rows, ids, false);
      Assert.assertEquals(eager.length(Staging.Kind.LONG), ids.length);
      var ordered = new Staging();
      CandidateDecisionRows.stage(ordered, rows, ids, true);
      Assert.assertEquals(
          ordered.length(Staging.Kind.LONG), repeated ? 2 * ids.length : ids.length);
    }
    var empty = new Staging();
    Assert.assertSame(
        CandidateDecisionRows.stage(empty, 3, new int[0], true), CandidateDecisionRows.EMPTY);
    Assert.assertEquals(empty.length(Staging.Kind.LONG), 0);
  }

  private static void check(int rows, int width, int[] ids, float[] upstream, Precision precision) {
    Gathered direct = gather(rows, width, ids, upstream, precision, null);
    Gathered ordered = gather(rows, width, ids, upstream, precision, true);
    Assert.assertEquals(ordered.shape(), new Shape(ids.length, width));
    Assert.assertEquals(ordered.type(), precision.compute());
    Assert.assertEquals(ordered.values(), direct.values());
    float[] expected = new float[rows * width];
    for (int c = 0; c < ids.length; c++) {
      for (int j = 0; j < width; j++) {
        expected[ids[c] * width + j] += upstream[c * width + j];
      }
    }
    for (int i = 0; i < expected.length; i++) {
      int bits = Float.floatToRawIntBits(expected[i]);
      if (precision == Precision.BFLOAT16) {
        bits = (bits + 0x7fff + ((bits >>> 16) & 1)) & 0xffff0000;
      }
      Assert.assertEquals(ordered.gradient()[i], bits, "gradient " + i);
    }
  }

  /**
   * 候補の決定行を集め、上流の勾配を流す。
   *
   * @param ordered 公開順に足す集め方なら true、eager の学習の集め方なら false、行を直接集める演算なら null
   * @return 集めた値のバイト、形、型と、決定行の勾配の FP32 のビット
   */
  private static Gathered gather(
      int rows, int width, int[] ids, float[] upstream, Precision precision, Boolean ordered) {
    try (var manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
        var gc = manager.getEngine().newGradientCollector()) {
      var staging = new Staging();
      var selection =
          ordered == null ? null : CandidateDecisionRows.stage(staging, rows, ids, ordered);
      staging.send(PinnedUploads.on(manager), manager);
      var context = new Context(null, manager, precision, true);
      float[] values = new float[rows * width * 2];
      for (int i = 0; i < values.length; i++) {
        values[i] = i % 5 == 0 ? -0f : (float) Math.sin(i);
      }
      var states =
          manager.create(values, new Shape(rows, 2, width)).toType(precision.compute(), false);
      var decision = states.get(":, 0, :");
      decision.setRequiresGradient(true);
      NDArray selected =
          selection == null
              ? NDArrays.gatherRows(
                  decision, manager.create(Arrays.stream(ids).asLongStream().toArray()))
              : selection.gather(context, decision);
      ByteBuffer bytes = selected.toByteBuffer();
      var incoming =
          manager.create(upstream, selected.getShape()).toType(precision.compute(), false);
      JniUtils.backward((PtNDArray) selected, (PtNDArray) incoming, false, false);
      NDArray actual = decision.getGradient();
      Assert.assertEquals(actual.getShape(), decision.getShape());
      Assert.assertEquals(actual.getDataType(), decision.getDataType());
      float[] floats = actual.toType(DataType.FLOAT32, false).toFloatArray();
      int[] gradient = new int[floats.length];
      for (int i = 0; i < floats.length; i++) {
        gradient[i] = Float.floatToRawIntBits(floats[i]);
      }
      byte[] copied = new byte[bytes.remaining()];
      bytes.get(copied);
      return new Gathered(copied, selected.getShape(), selected.getDataType(), gradient);
    }
  }

  private record Gathered(byte[] values, Shape shape, DataType type, int[] gradient) {}

  @Test
  public void gradientMatchesCentralDifferencesOfTheSelectedRows() {
    int[] ids = {2, 0, 2, 2, 0};
    float[] x = {-1, 2, -3, 4, 5, -6};
    try (var manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
        var gc = manager.getEngine().newGradientCollector()) {
      var staging = new Staging();
      var selection = CandidateDecisionRows.stage(staging, 3, ids, true);
      staging.send(PinnedUploads.on(manager), manager);
      var context = new Context(null, manager, Precision.FLOAT32, true);
      var decision = manager.create(x, new Shape(3, 2));
      decision.setRequiresGradient(true);
      var coefficients = manager.arange(10).reshape(5, 2);
      gc.backward(selection.gather(context, decision).mul(coefficients).sum());
      float[] actual = decision.getGradient().toFloatArray();
      float h = 0.125f;
      for (int i = 0; i < x.length; i++) {
        x[i] += h;
        float above = objective(context, manager, selection, x, coefficients);
        x[i] -= 2 * h;
        float below = objective(context, manager, selection, x, coefficients);
        x[i] += h;
        Assert.assertEquals(actual[i], (above - below) / (2 * h));
      }
    }
  }

  private static float objective(
      Context context,
      NDManager manager,
      CandidateDecisionRows selection,
      float[] values,
      NDArray coefficients) {
    var selected = selection.gather(context, manager.create(values, new Shape(3, 2)));
    return selected.mul(coefficients).sum().getFloat();
  }
}
