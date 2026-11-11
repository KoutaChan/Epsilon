package com.epsilon.grp;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * GruSinkhorn を確かめる。周辺確率の行と列の和、Φ の範囲と和、GRU の式（double で書いた参照と比べる）、Sinkhorn の出力ヘッド（対数で
 * 正規化したものと比べる）とクリップ、状態のメモの当たりと外れ、履歴の全ての局を読むこと、重みの並び、重み・作業領域・履歴の検査。
 */
public class GruSinkhornTest {

  private static final double TOLERANCE = 1e-12;

  @Test
  public void marginalsSumToOneByRowAndColumn() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      for (int layers = 1; layers <= 2; layers++) {
        for (double scale : new double[] {0.5, 3.0}) {
          GruShape shape = new GruShape(variant, 8, layers);
          GruSinkhorn gru = GrpFixtures.gru(shape, 3 + layers, scale);
          GruWorkspace workspace = GruWorkspace.of(gru);
          double[] marginals = new double[seats * seats];
          Random random = new Random(7);
          for (int trial = 0; trial < 300; trial++) {
            RoundHistory history = GrpFixtures.history(variant, random, 1 + random.nextInt(12));
            gru.marginals(history, GrpFixtures.state(variant, random), workspace, marginals);
            for (int i = 0; i < seats; i++) {
              double row = 0;
              double column = 0;
              for (int j = 0; j < seats; j++) {
                Assert.assertTrue(marginals[i * seats + j] > 0, Arrays.toString(marginals));
                row += marginals[i * seats + j];
                column += marginals[j * seats + i];
              }
              Assert.assertEquals(row, 1, TOLERANCE);
              Assert.assertEquals(column, 1, TOLERANCE);
            }
          }
        }
      }
    }
  }

  @Test
  public void phiStaysWithinTheUtilitiesAndKeepsTheirSum() {
    for (Variant variant : Variant.values()) {
      GruShape shape = new GruShape(variant, 8, 2);
      GruSinkhorn gru = GrpFixtures.gru(shape, 13, 2.0);
      GruWorkspace workspace = GruWorkspace.of(gru);
      TableState table = new TableState(variant);
      double[] phi = new double[variant.seats()];
      for (RankUtility utility : RankUtility.values()) {
        double lowest = utility.value(variant, variant.seats() - 1);
        double highest = utility.value(variant, 0);
        double sum = 0;
        for (int rank = 0; rank < variant.seats(); rank++) {
          sum += utility.value(variant, rank);
        }
        Random random = new Random(17);
        for (int trial = 0; trial < 300; trial++) {
          RoundHistory history = GrpFixtures.history(variant, random, 1 + random.nextInt(10));
          GrpFixtures.state(variant, random).copyInto(table);
          gru.phi(history, table, utility, workspace, phi);
          double total = 0;
          for (double value : phi) {
            Assert.assertTrue(value >= lowest - TOLERANCE && value <= highest + TOLERANCE, Double.toString(value));
            total += value;
          }
          Assert.assertEquals(total, sum, TOLERANCE);
        }
      }
    }
  }

  /** 参照は同じ式を double で添字どおりに書いたもの。float の丸めの差だけ違い、ゲートの順や b_hn の位置を違えると大きく外れる。 */
  @Test
  public void theGruFollowsTheReferenceEquations() {
    for (Variant variant : Variant.values()) {
      GruShape shape = new GruShape(variant, 5, 2);
      GruSinkhorn gru = GrpFixtures.gru(shape, 19, 0.4);
      GruWorkspace workspace = GruWorkspace.of(gru);
      double[] actual = new double[shape.marginals()];
      Random random = new Random(23);
      for (int trial = 0; trial < 100; trial++) {
        RoundHistory history = GrpFixtures.history(variant, random, 1 + random.nextInt(8));
        GrpState state = GrpFixtures.state(variant, random);
        gru.marginals(history, state, workspace, actual);
        double[] expected = reference(gru, history, state);
        for (int index = 0; index < actual.length; index++) {
          Assert.assertEquals(actual[index], expected[index], 1e-4, variant + " trial " + trial + " " + index);
        }
      }
    }
  }

  /** GRU と投影の重みを 0 にするとロジットは最後の層のバイアスそのものになり、周辺確率は対数で正規化したものと丸めの差を除いて同じ。 */
  @Test
  public void theSinkhornHeadMatchesTheLogSpaceNormalization() {
    for (Variant variant : Variant.values()) {
      GruShape shape = new GruShape(variant, 4, 1);
      int bias = GruTensor.starts(shape)[GruTensor.LOGIT_BIAS.ordinal()][0];
      double[] actual = new double[shape.marginals()];
      Random random = new Random(29);
      for (int trial = 0; trial < 200; trial++) {
        double spread = trial < 100 ? 2 : 60;
        float[] weights = new float[GruTensor.count(shape)];
        double[] logits = new double[shape.marginals()];
        for (int index = 0; index < logits.length; index++) {
          weights[bias + index] = (float) (random.nextGaussian() * spread);
          logits[index] = weights[bias + index];
        }
        GruSinkhorn gru = GruSinkhorn.of(shape, weights);
        RoundHistory history = GrpFixtures.history(variant, random, 2);
        gru.marginals(history, GrpFixtures.state(variant, random), GruWorkspace.of(gru), actual);
        double[] expected = logSpaceSinkhorn(logits, variant.seats());
        for (int index = 0; index < actual.length; index++) {
          Assert.assertEquals(actual[index], expected[index], TOLERANCE, variant + " trial " + trial + " " + index);
        }
      }
    }
  }

  /**
   * ロジットが 1000·単位行列のとき、交互作用は対角が大きく正、ほかが大きく負で、クリップで ±2 になる。行列は巡回行列なので、周辺確率の
   * 対角は e^2 / (e^2 ＋ (S − 1)·e^−2) = e^4 / (e^4 ＋ S − 1)。
   */
  @Test
  public void interactionsAreClippedToTwo() {
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      GruShape shape = new GruShape(variant, 4, 1);
      int bias = GruTensor.starts(shape)[GruTensor.LOGIT_BIAS.ordinal()][0];
      float[] weights = new float[GruTensor.count(shape)];
      for (int seat = 0; seat < seats; seat++) {
        weights[bias + seat * seats + seat] = 1000f;
      }
      GruSinkhorn gru = GruSinkhorn.of(shape, weights);
      double[] marginals = new double[shape.marginals()];
      Random random = new Random(31);
      gru.marginals(
          GrpFixtures.history(variant, random, 1), GrpFixtures.state(variant, random), GruWorkspace.of(gru), marginals);
      double diagonal = StrictMath.exp(4) / (StrictMath.exp(4) + seats - 1);
      for (int seat = 0; seat < seats; seat++) {
        for (int rank = 0; rank < seats; rank++) {
          double expected = seat == rank ? diagonal : (1 - diagonal) / (seats - 1);
          Assert.assertEquals(marginals[seat * seats + rank], expected, TOLERANCE, seat + " " + rank);
        }
      }
    }
  }

  /** バージョンが同じ履歴（コピーも）は状態のメモに当たり、新しい作業領域で読み直した値とビット単位で同じ。局を足すとバージョンが替わって外れる。 */
  @Test
  public void stateMemoHitsGiveTheSameBits() {
    Variant variant = Variant.TENHOU_4P;
    GruShape shape = new GruShape(variant, 8, 2);
    GruSinkhorn gru = GrpFixtures.gru(shape, 37, 1.0);
    GruWorkspace shared = GruWorkspace.of(gru);
    double[] cached = new double[shape.marginals()];
    double[] fresh = new double[shape.marginals()];
    Random random = new Random(41);
    RoundHistory history = new RoundHistory(variant);
    int rounds = 12;
    int ends = 20;
    for (int round = 0; round < rounds; round++) {
      history.add(GrpFixtures.state(variant, random));
      RoundHistory fork = history.copy();
      for (int end = 0; end < ends; end++) {
        GrpState state = GrpFixtures.state(variant, random);
        gru.marginals(end % 2 == 0 ? history : fork, state, shared, cached);
        gru.marginals(history, state, GruWorkspace.of(gru), fresh);
        assertBits(cached, fresh);
      }
    }
    Assert.assertEquals(shared.stateMisses(), rounds);
    Assert.assertEquals(shared.stateHits(), rounds * (ends - 1L));
  }

  /** 前の局を 1 つ替えると周辺確率が変わる。中身が同じなら、別々に積んだ履歴（バージョンが違う）でもビット単位で同じ。 */
  @Test
  public void everyRoundOfTheHistoryCounts() {
    Variant variant = Variant.TENHOU_4P;
    GruShape shape = new GruShape(variant, 8, 2);
    GruSinkhorn gru = GrpFixtures.gru(shape, 43, 1.0);
    GruWorkspace workspace = GruWorkspace.of(gru);
    Random random = new Random(47);
    GrpState first = GrpFixtures.state(variant, random);
    GrpState other = GrpFixtures.state(variant, random);
    GrpState middle = GrpFixtures.state(variant, random);
    GrpState end = GrpFixtures.state(variant, random);
    RoundHistory history = new RoundHistory(variant);
    RoundHistory same = new RoundHistory(variant);
    RoundHistory changed = new RoundHistory(variant);
    for (GrpState state : new GrpState[] {first, middle}) {
      history.add(state);
      same.add(state);
    }
    changed.add(other);
    changed.add(middle);
    Assert.assertNotEquals(same.stamp(), history.stamp());
    double[] expected = new double[shape.marginals()];
    double[] actual = new double[shape.marginals()];
    gru.marginals(history, end, workspace, expected);
    gru.marginals(same, end, workspace, actual);
    assertBits(actual, expected);
    gru.marginals(changed, end, workspace, actual);
    Assert.assertFalse(Arrays.equals(actual, expected), Arrays.toString(actual));
  }

  /**
   * 重みの数と並び。状態の幅 64・2 層の GRU を四人打ちで作ると、層 0 が (7 ＋ 64) × 192 ＋ 2 × 192、層 1 が (64 ＋ 64) × 192 ＋
   * 2 × 192、投影が 64 × 64 ＋ 64、ロジットが 64 × 16 ＋ 16 で、合わせて 44,176。テンソルは宣言の順に並び、層ごとに分かれたテンソルの
   * 中は層の順に並ぶ。各テンソルの先頭の位置を並べた表は、最後の行が重みの数になる。
   */
  @Test
  public void tensorsFollowTheDeclarationOrder() {
    GruShape shape = new GruShape(Variant.TENHOU_4P, 64, 2);
    Assert.assertEquals(GruTensor.count(shape), 44_176);
    int[][] starts = GruTensor.starts(shape);
    Assert.assertEquals(starts[GruTensor.values().length], new int[] {44_176});
    Assert.assertEquals(starts[GruTensor.INPUT_WEIGHTS.ordinal()], new int[] {0, 7 * 192});
    Assert.assertEquals(starts[GruTensor.STATE_WEIGHTS.ordinal()], new int[] {7 * 192 + 64 * 192, 7 * 192 + 2 * 64 * 192});
    Assert.assertEquals(starts[GruTensor.PROJECTION_WEIGHTS.ordinal()], new int[] {44_176 - 16 - 64 * 16 - 64 - 64 * 64});
    Assert.assertEquals(starts[GruTensor.LOGIT_BIAS.ordinal()], new int[] {44_176 - 16});
    Assert.assertEquals(new GruShape(Variant.TENHOU_3P, 64, 2).inputs(), 6);
  }

  @Test
  public void weightsMustFitTheShapeAndBeFinite() {
    GruShape shape = new GruShape(Variant.TENHOU_3P, 3, 2);
    Assert.assertThrows(IllegalArgumentException.class, () -> GruSinkhorn.of(shape, new float[5]));
    float[] weights = GrpFixtures.gru(shape, 53, 1.0).weights();
    weights[11] = Float.NaN;
    Assert.assertThrows(IllegalArgumentException.class, () -> GruSinkhorn.of(shape, weights));
    weights[11] = Float.NEGATIVE_INFINITY;
    Assert.assertThrows(IllegalArgumentException.class, () -> GruSinkhorn.of(shape, weights));
    Assert.assertThrows(IllegalArgumentException.class, () -> new GruShape(Variant.TENHOU_4P, 0, 1));
    Assert.assertThrows(IllegalArgumentException.class, () -> new GruShape(Variant.TENHOU_4P, 4, 0));
  }

  @Test
  public void modelsKeepTheirOwnWeights() {
    GruShape shape = new GruShape(Variant.TENHOU_4P, 3, 1);
    float[] weights = GrpFixtures.gru(shape, 59, 1.0).weights();
    GruSinkhorn gru = GruSinkhorn.of(shape, weights);
    float first = weights[0];
    weights[0] = first + 1;
    gru.weights()[1] = 5;
    float[] kept = gru.weights();
    Assert.assertEquals(kept[0], first);
    Assert.assertEquals(kept[1], weights[1]);
  }

  /** ほかのモデルの作業領域は、形が同じで、状態のメモに同じ履歴のバージョンがあっても拒む。 */
  @Test
  public void historiesWorkspacesAndOutputsMustFit() {
    Variant variant = Variant.TENHOU_4P;
    GruShape shape = new GruShape(variant, 4, 1);
    GruSinkhorn gru = GrpFixtures.gru(shape, 61, 1.0);
    GruWorkspace workspace = GruWorkspace.of(gru);
    GrpState state = new GrpState(variant, 0, 0, 0, new int[] {25_000, 25_000, 25_000, 25_000});
    RoundHistory history = GrpFixtures.history(variant, new Random(67), 2);
    double[] out = new double[16];
    Assert.assertThrows(IllegalArgumentException.class, () -> gru.marginals(null, state, workspace, out));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> gru.marginals(new RoundHistory(variant), state, workspace, out));
    RoundHistory three = GrpFixtures.history(Variant.TENHOU_3P, new Random(71), 2);
    Assert.assertThrows(IllegalArgumentException.class, () -> gru.marginals(three, state, workspace, out));
    GrpState threeState = new GrpState(Variant.TENHOU_3P, 0, 0, 0, new int[] {35_000, 35_000, 35_000});
    Assert.assertThrows(IllegalArgumentException.class, () -> gru.marginals(history, threeState, workspace, out));
    GruSinkhorn twin = GrpFixtures.gru(shape, 62, 1.0);
    GruWorkspace other = GruWorkspace.of(twin);
    twin.marginals(history, state, other, out);
    Assert.assertThrows(IllegalArgumentException.class, () -> gru.marginals(history, state, other, out));
    Assert.assertThrows(
        IllegalArgumentException.class, () -> gru.phi(history, state, RankUtility.TENHOU, other, new double[4]));
    Assert.assertThrows(IllegalArgumentException.class, () -> gru.marginals(history, state, workspace, new double[4]));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> gru.phi(history, GrpFixtures.finished(), RankUtility.TENHOU, workspace, new double[4]));
  }

  /**
   * 対数で計算する Sinkhorn。クリップした交互作用から行と列の logsumexp を交互に {@code ITERATIONS} 回引き、最後に行の logsumexp を
   * 引いてから、列の和の不足分を足す。
   */
  static double[] logSpaceSinkhorn(double[] logits, int seats) {
    double[] rows = new double[seats];
    double[] columns = new double[seats];
    double mean = 0;
    for (int seat = 0; seat < seats; seat++) {
      for (int rank = 0; rank < seats; rank++) {
        rows[seat] += logits[seat * seats + rank] / seats;
        columns[rank] += logits[seat * seats + rank] / seats;
        mean += logits[seat * seats + rank] / (seats * seats);
      }
    }
    double[] log = new double[seats * seats];
    for (int seat = 0; seat < seats; seat++) {
      for (int rank = 0; rank < seats; rank++) {
        double interaction = logits[seat * seats + rank] - rows[seat] - columns[rank] + mean;
        log[seat * seats + rank] = 2 * Math.tanh(interaction / 2);
      }
    }
    for (int iteration = 0; iteration < GruSinkhorn.ITERATIONS; iteration++) {
      for (int seat = 0; seat < seats; seat++) {
        subtractLogSum(log, seat * seats, 1, seats);
      }
      for (int rank = 0; rank < seats; rank++) {
        subtractLogSum(log, rank, seats, seats);
      }
    }
    for (int seat = 0; seat < seats; seat++) {
      subtractLogSum(log, seat * seats, 1, seats);
    }
    double[] marginals = new double[seats * seats];
    for (int rank = 0; rank < seats; rank++) {
      double sum = 0;
      for (int seat = 0; seat < seats; seat++) {
        marginals[seat * seats + rank] = Math.exp(log[seat * seats + rank]);
        sum += marginals[seat * seats + rank];
      }
      for (int seat = 0; seat < seats; seat++) {
        marginals[seat * seats + rank] += (1 - sum) / seats;
      }
    }
    return marginals;
  }

  private static void subtractLogSum(double[] values, int offset, int stride, int count) {
    double max = Double.NEGATIVE_INFINITY;
    for (int index = 0; index < count; index++) {
      max = Math.max(max, values[offset + index * stride]);
    }
    double sum = 0;
    for (int index = 0; index < count; index++) {
      sum += Math.exp(values[offset + index * stride] - max);
    }
    double normalizer = max + Math.log(sum);
    for (int index = 0; index < count; index++) {
      values[offset + index * stride] -= normalizer;
    }
  }

  /**
   * PyTorch の GRU の式を double で書いた参照。入力は局の行をスケールで割ったもので、状態は 0 から始め、層 0 から順に進める。その後に
   * 投影・ReLU・ロジット・Sinkhorn を通す。
   */
  private static double[] reference(GruSinkhorn gru, RoundHistory history, GrpState end) {
    GruShape shape = gru.shape();
    float[] weights = gru.weights();
    int[][] starts = GruTensor.starts(shape);
    int seats = shape.variant().seats();
    int hidden = shape.hidden();
    int gates = shape.gates();
    List<double[]> rows = new ArrayList<>();
    for (int index = 0; index < history.size(); index++) {
      int[] scores = new int[seats];
      for (int seat = 0; seat < seats; seat++) {
        scores[seat] = history.score(index, seat);
      }
      rows.add(row(history.round(index), history.honba(index), history.kyotaku(index), scores));
    }
    rows.add(row(end.round(), end.honba(), end.sticks(), end.scores()));
    double[][] state = new double[shape.layers()][hidden];
    for (double[] input : rows) {
      double[] x = input;
      for (int layer = 0; layer < shape.layers(); layer++) {
        int inputWeights = starts[GruTensor.INPUT_WEIGHTS.ordinal()][layer];
        int stateWeights = starts[GruTensor.STATE_WEIGHTS.ordinal()][layer];
        int inputBias = starts[GruTensor.INPUT_BIAS.ordinal()][layer];
        int stateBias = starts[GruTensor.STATE_BIAS.ordinal()][layer];
        double[] next = new double[hidden];
        for (int unit = 0; unit < hidden; unit++) {
          double[] fromInput = new double[3];
          double[] fromState = new double[3];
          for (int gate = 0; gate < 3; gate++) {
            int column = gate * hidden + unit;
            fromInput[gate] = weights[inputBias + column];
            for (int i = 0; i < x.length; i++) {
              fromInput[gate] += weights[inputWeights + i * gates + column] * x[i];
            }
            fromState[gate] = weights[stateBias + column];
            for (int k = 0; k < hidden; k++) {
              fromState[gate] += weights[stateWeights + k * gates + column] * state[layer][k];
            }
          }
          double reset = 1 / (1 + Math.exp(-(fromInput[0] + fromState[0])));
          double update = 1 / (1 + Math.exp(-(fromInput[1] + fromState[1])));
          double candidate = Math.tanh(fromInput[2] + reset * fromState[2]);
          next[unit] = (1 - update) * candidate + update * state[layer][unit];
        }
        state[layer] = next;
        x = next;
      }
    }
    int projectionWeights = starts[GruTensor.PROJECTION_WEIGHTS.ordinal()][0];
    int projectionBias = starts[GruTensor.PROJECTION_BIAS.ordinal()][0];
    double[] projected = new double[hidden];
    for (int unit = 0; unit < hidden; unit++) {
      double value = weights[projectionBias + unit];
      for (int k = 0; k < hidden; k++) {
        value += weights[projectionWeights + k * hidden + unit] * state[shape.layers() - 1][k];
      }
      projected[unit] = Math.max(0, value);
    }
    int logitWeights = starts[GruTensor.LOGIT_WEIGHTS.ordinal()][0];
    int logitBias = starts[GruTensor.LOGIT_BIAS.ordinal()][0];
    double[] logits = new double[seats * seats];
    for (int index = 0; index < logits.length; index++) {
      logits[index] = weights[logitBias + index];
      for (int k = 0; k < hidden; k++) {
        logits[index] += weights[logitWeights + k * logits.length + index] * projected[k];
      }
    }
    return logSpaceSinkhorn(logits, seats);
  }

  /** 局の行をスケールで割ったもの（局と本場は 8、供託は 4、点は 10,000）。 */
  private static double[] row(int round, int honba, int kyotaku, int[] scores) {
    double[] row = new double[3 + scores.length];
    row[0] = round / 8.0;
    row[1] = honba / 8.0;
    row[2] = kyotaku / 4.0;
    for (int seat = 0; seat < scores.length; seat++) {
      row[3 + seat] = scores[seat] / 10_000.0;
    }
    return row;
  }

  private static void assertBits(double[] actual, double[] expected) {
    for (int index = 0; index < expected.length; index++) {
      Assert.assertEquals(
          Double.doubleToRawLongBits(actual[index]), Double.doubleToRawLongBits(expected[index]), "index " + index);
    }
  }
}
