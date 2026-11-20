package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.kernel.CandidateKernels;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import org.testng.Assert;

/** ステップ数と局面に依存しない固定入力で、ゲート勾配の丸めと行の順序を検査する。 */
final class GateGradientFixture {
  private static final int WIDTH = 16, CODES = 23, ROW_LANES = 8;
  private final int rows, tiles;
  private final int[] owners, ids, counts;
  private final float[] x, dy, logits;

  GateGradientFixture(int seed, int rows, int tiles, int transitions) {
    this.rows = rows;
    this.tiles = tiles;
    var random = new Random(seed);
    owners = new int[transitions];
    counts = new int[rows];
    for (int t = 0; t < transitions; t++) counts[random.nextInt(rows - 1)]++;
    for (int r = 0, t = 0; r < rows; r++) for (int i = 0; i < counts[r]; i++) owners[t++] = r;
    ids = new int[transitions * tiles];
    for (int i = 0; i < ids.length; i++) ids[i] = random.nextInt(i % 7 == 0 ? CODES : 3);
    x = values(random, rows * tiles * WIDTH);
    dy = values(random, transitions * WIDTH);
    logits = new float[CODES * WIDTH];
    for (int i = 0; i < logits.length; i++) logits[i] = (random.nextFloat() - .5f) * 4f;
  }

  private static float[] values(Random r, int size) {
    float[] values = new float[size];
    for (int i = 0; i < size; i++)
      values[i] = Math.scalb((float) (r.nextInt(255) - 127), r.nextInt(12) - 8);
    return values;
  }

  float[] reduce(float[] slope, boolean fused, boolean shortFirst) {
    float[] result = new float[logits.length];
    for (int begin = 0; begin < rows; begin += ROW_LANES) {
      Integer[] group = new Integer[Math.min(rows - begin, ROW_LANES)];
      for (int i = 0; i < group.length; i++) group[i] = begin + i;
      if (shortFirst) Arrays.sort(group, Comparator.comparingInt(r -> counts[r]));
      for (int row : group) {
        float[] sum = new float[result.length];
        for (int start = 0; start < tiles; start += ROW_LANES)
          for (int t = 0; t < owners.length; t++) {
            if (owners[t] != row) continue;
            for (int k = start; k < Math.min(tiles, start + ROW_LANES); k++)
              for (int c = 0; c < WIDTH; c++) {
                int at = ids[t * tiles + k] * WIDTH + c;
                float product = dy[t * WIDTH + c] * x[(row * tiles + k) * WIDTH + c];
                sum[at] =
                    fused ? Math.fma(product, slope[at], sum[at]) : sum[at] + product * slope[at];
              }
          }
        for (int i = 0; i < sum.length; i++) result[i] += sum[i];
      }
    }
    return result;
  }

  float[] hostSlope() {
    float[] slope = new float[logits.length];
    for (int i = 0; i < slope.length; i++) {
      float gate = 1f / (1f + (float) StrictMath.exp(-logits[i]));
      slope[i] = gate * (1f - gate);
    }
    return slope;
  }

  /** 各コードに一項だけ渡し、同じデバイスの exp と逆数で丸めた係数を取り出す。 */
  float[] deviceSlope(NDManager m, CandidateKernels kernels, NDArray gate) {
    int[] sequence = new int[CODES];
    Arrays.setAll(sequence, i -> i);
    var ids = m.create(sequence);
    var unit = m.ones(new Shape(CODES, 1, WIDTH), DataType.BFLOAT16);
    var slope = m.zeros(gate.getShape());
    kernels.countGradient(
        unit,
        gate,
        ids,
        ids.reshape(CODES, 1),
        unit.reshape(CODES, WIDTH),
        m.create(unit.getShape(), DataType.BFLOAT16),
        slope);
    float[] actual = slope.toFloatArray();
    float[] host = hostSlope();
    for (int i = 0; i < host.length; i++) Assert.assertEquals(actual[i], host[i], 2e-7f);
    return actual;
  }

  void record(
      NDManager m, CandidateKernels kernels, TrainingTensorTrace trace, String label, boolean check)
      throws Exception {
    var gate = m.create(logits, new Shape(CODES, WIDTH));
    var slope = deviceSlope(m, kernels, gate);
    var projected = m.create(x, new Shape(rows, tiles, WIDTH)).toType(DataType.BFLOAT16, false);
    var upstream = m.create(dy, new Shape(owners.length, WIDTH)).toType(DataType.BFLOAT16, false);
    var result = m.ones(gate.getShape());
    kernels.countGradient(
        projected,
        gate,
        m.create(owners),
        m.create(ids, new Shape(owners.length, tiles)),
        upstream,
        m.create(projected.getShape(), DataType.BFLOAT16),
        result);
    float[] expected = reduce(slope, false, false);
    Assert.assertFalse(Arrays.equals(expected, reduce(slope, true, false)));
    if (check) Assert.assertEquals(result.toFloatArray(), expected);
    if (trace != null) {
      trace.tensor(label + "/slope", m.create(slope));
      trace.tensor(label + "/rows", m.create(owners));
      trace.tensor(label + "/ids", m.create(ids));
      trace.tensor(label + "/projected", projected);
      trace.tensor(label + "/upstream", upstream);
      trace.tensor(label + "/observed", result);
      trace.tensor(label + "/rowOrder", m.create(expected, gate.getShape()));
      trace.tensor(label + "/shortFirst", m.create(reduce(slope, false, true), gate.getShape()));
    }
  }

  static int[][] cases() {
    return new int[][] {{1709, 11, 34, 97}, {4637, 19, 17, 83}, {8521, 7, 9, 53}};
  }
}
