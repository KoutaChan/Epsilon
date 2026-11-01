package com.epsilon.prism.luck;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.util.Pair;
import com.epsilon.core.GameState;
import com.epsilon.prism.luck.LuckSchema.HeaderBits;
import com.epsilon.prism.model.Precision;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 運の網 Ω（CPU、FP32、小さな網）: 結果を入力に入れない、制御変量はプールの上で平均 0、数歩の学習で損失が下がる。当てはめの全体の道は、Y を
 * 結果の関数にした標本で保留の R² が上がる（結果の軸の並びの確かめ）ことと、保存した網が同じ値を出すことで確かめる。標本は
 * {@link LuckObservationTest} と同じ打ち方で集めた主経路の出来事（{@link LuckCohorts}）。
 */
public class LuckFitTest {
  private static final Logger log = LoggerFactory.getLogger(LuckFitTest.class);

  private static final long SEED = 41_000;
  private static final int GAMES = 24;
  private static final double RATE = 0.25;
  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int WIDTH = 64;

  private Path root;
  private Path cohort;
  private LuckSamples samples;
  private LuckSamples outcomeSamples;

  @BeforeClass
  public void collect() throws IOException {
    root = Files.createTempDirectory("luck-fit");
    cohort = LuckCohorts.write(root.resolve("1"), SEED, GAMES, RATE);
    samples = LuckSamples.load(List.of(cohort));
    assertTrue(samples.size() > 2_000, "samples " + samples.size());
    outcomeSamples =
        LuckSamples.load(List.of(LuckCohorts.outcomeTarget(cohort, root.resolve("2"))));
  }

  @AfterClass(alwaysRun = true)
  public void delete() throws IOException {
    if (root == null) {
      return;
    }
    try (var files = Files.walk(root)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(file);
      }
    }
  }

  /** 結果の牌だけが違う O° から、ビットで同じ入力・プール・網の出力を作る（結果の番号だけが違う）。 */
  @Test
  public void outcomeIsNotAnInput() throws IOException {
    int rows = 256;
    LuckRows actual = new LuckRows(rows);
    LuckRows changed = new LuckRows(rows);
    actual.resize(rows);
    changed.resize(rows);
    LuckTokenizer tokenizer = new LuckTokenizer();
    LuckObservation read = new LuckObservation();
    float[] y = new float[SEATS];
    try (LuckSampleStore.Cursor cursor = LuckSampleStore.open(cohort).cursor()) {
      for (int row = 0; row < rows; row++) {
        assertTrue(cursor.next());
        long[] original = Arrays.copyOf(cursor.words(), cursor.observation().longs());
        for (int seat = 0; seat < SEATS; seat++) {
          y[seat] = cursor.y(seat);
        }
        read.bind(original, 0);
        int other = otherOutcome(read);
        actual.put(row, tokenizer, read, y, 0, cursor.weight(), cursor.game());
        long[] moved = original.clone();
        LuckSchema.HEADER.put(moved, 0, 0, HeaderBits.OUTCOME_TILE, LuckSchema.outcomeTile(other));
        LuckSchema.HEADER.put(
            moved, 0, 0, HeaderBits.OUTCOME_AKA, LuckSchema.outcomeAka(other) ? 1 : 0);
        changed.put(row, tokenizer, read.bind(moved, 0), y, 0, cursor.weight(), cursor.game());
        assertTrue(actual.chosen[row] != changed.chosen[row], "outcome did not move");
      }
    }
    assertEquals(changed.round, actual.round);
    assertEquals(changed.seat, actual.seat);
    assertEquals(changed.tile, actual.tile);
    assertEquals(changed.codes.array(), actual.codes.array());
    assertTrue(Arrays.equals(changed.roundScalar, actual.roundScalar));
    assertTrue(Arrays.equals(changed.seatScalar, actual.seatScalar));
    assertTrue(Arrays.equals(changed.pool, actual.pool));
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      LuckNetwork network = LuckNetwork.blank(LuckNetwork.shape(WIDTH), manager, 3);
      randomizeHead(network, 5);
      ParameterStore store = new ParameterStore(manager, false);
      float[] left =
          network.forward(store, actual.inputs(manager), Precision.FLOAT32, false).toFloatArray();
      float[] right =
          network.forward(store, changed.inputs(manager), Precision.FLOAT32, false).toFloatArray();
      assertTrue(Arrays.equals(left, right), "network output depends on the outcome");
    }
  }

  /** Σ_t p(t)·c(t) = 0: どの重みの Ω でも、制御変量はプールの上の結果について平均 0。白紙の Ω の保留の R² は厳密に 0。 */
  @Test
  public void controlVariateAveragesToZeroOverThePool() {
    int rows = 128;
    LuckRows batch = batch(rows);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      LuckNetwork network = LuckNetwork.blank(LuckNetwork.shape(WIDTH), manager, 7);
      ParameterStore store = new ParameterStore(manager, false);
      LuckFit.Holdout blank = LuckFit.score(samples, network, manager, 0.25, 64, 2);
      assertEquals(blank.all().r2(), 0.0, 0.0);
      randomizeHead(network, 9);
      NDArray values = network.forward(store, batch.inputs(manager), Precision.FLOAT32, false);
      LuckTargets targets = batch.targets(manager);
      double scale = 0;
      for (float value : values.toFloatArray()) {
        scale = Math.max(scale, Math.abs(value));
      }
      assertTrue(scale > 1e-2, "head output " + scale);
      double[] mean = new double[rows * SEATS];
      for (int outcome = 0; outcome < LuckRows.OUTCOMES; outcome++) {
        long[] index = new long[rows];
        for (int row = 0; row < rows; row++) {
          index[row] = (long) row * LuckRows.OUTCOMES + outcome;
        }
        float[] c = ControlVariate.of(values, targets.pool(), manager.create(index)).toFloatArray();
        for (int row = 0; row < rows; row++) {
          double total = 0;
          for (int other = 0; other < LuckRows.OUTCOMES; other++) {
            total += batch.pool[row * LuckRows.OUTCOMES + other];
          }
          double p = batch.pool[row * LuckRows.OUTCOMES + outcome] / total;
          for (int seat = 0; seat < SEATS; seat++) {
            mean[row * SEATS + seat] += p * c[row * SEATS + seat];
          }
        }
      }
      double worst = 0;
      for (double value : mean) {
        worst = Math.max(worst, Math.abs(value));
      }
      log.info(
          "LUCK-FIT pool mean of c: max |E_p[c]| = {} (|V| <= {})",
          String.format("%.3g", worst),
          String.format("%.3g", scale));
      assertTrue(worst <= 1e-6 * scale, "E_p[c] = " + worst);
    }
  }

  /**
   * 同じ塊で数歩当てると損失が下がる。当てはめの全体の道（並べ替え・先読み・保留の採点）で、Y が結果の関数なら保留の R² が上がり（網の結果の軸・
   * 選ぶ結果の番号・プールの並びがそろっている）、保存した網が同じ値を出す。
   */
  @Test
  public void fittingLowersTheLossAndLearnsTheOutcome() throws IOException {
    LuckRows batch = batch(128);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      LuckNetwork network = LuckNetwork.blank(LuckNetwork.shape(WIDTH), manager, 11);
      ParameterStore store = new ParameterStore(manager, false);
      Optimizer optimizer = LuckFit.optimizer(1e-3f, LuckFit.Plan.WEIGHT_DECAY);
      double first = LuckFit.step(network, store, manager, batch, Precision.FLOAT32, optimizer);
      double last = first;
      for (int step = 1; step < 30; step++) {
        last = LuckFit.step(network, store, manager, batch, Precision.FLOAT32, optimizer);
      }
      log.info("LUCK-FIT fixed batch loss {} -> {}", String.format("%.5f", first), String.format("%.5f", last));
      assertTrue(last < 0.8 * first, "loss " + first + " -> " + last);
    }
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      LuckFit.Plan plan =
          new LuckFit.Plan(
              LuckNetwork.shape(WIDTH), 4.0, 64, 3e-3f, LuckFit.Plan.WEIGHT_DECAY, 0.25, 13);
      LuckNetwork network = LuckNetwork.blank(plan.shape(), manager, plan.seed());
      LuckFit.Report report = LuckFit.run(outcomeSamples, network, manager, plan, 2);
      log.info("LUCK-FIT report {}", report);
      assertEquals(report.steps(), Math.ceilDiv(4 * report.trainSamples(), 64L));
      assertTrue(report.holdout().all().samples() > 0);
      double r2 = report.holdout().all().r2();
      assertTrue(r2 > 0.5, "holdout R2 " + r2);
      Path saved = root.resolve("network");
      network.save(saved);
      LuckNetwork loaded = LuckNetwork.load(saved, manager);
      ParameterStore store = new ParameterStore(manager, false);
      LuckRows probe = batch(32);
      float[] before =
          network.forward(store, probe.inputs(manager), Precision.FLOAT32, false).toFloatArray();
      float[] after =
          loaded.forward(store, probe.inputs(manager), Precision.FLOAT32, false).toFloatArray();
      assertTrue(Arrays.equals(before, after), "loaded network differs");
    }
  }

  /** 表の先頭の標本で塊を作る。 */
  private LuckRows batch(int rows) {
    LuckRows batch = new LuckRows(rows);
    batch.resize(rows);
    LuckTokenizer tokenizer = new LuckTokenizer();
    LuckObservation read = new LuckObservation();
    for (int row = 0; row < rows; row++) {
      samples.put(row, batch, row, tokenizer, read);
    }
    return batch;
  }

  /** 実際の結果と違い、プールにある結果。 */
  private static int otherOutcome(LuckObservation o) {
    int actual = o.outcome();
    for (int outcome = 0; outcome < LuckSchema.OUTCOMES; outcome++) {
      if (outcome != actual && o.poolOf(outcome) > 0) {
        return outcome;
      }
    }
    throw new IllegalStateException("pool has a single outcome");
  }

  /** 頭の最後の層（白紙では 0）を乱数にする。 */
  private static void randomizeHead(LuckNetwork network, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    for (Pair<String, Parameter> pair : network.getParameters()) {
      if (pair.getKey().endsWith("head_output") || pair.getKey().endsWith("head_outputBias")) {
        NDArray array = pair.getValue().getArray();
        float[] values = new float[Math.toIntExact(array.size())];
        for (int index = 0; index < values.length; index++) {
          values[index] = (float) (random.nextGaussian() * 0.5);
        }
        array.set(values);
      }
    }
  }
}
