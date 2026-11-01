package com.epsilon.prism.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.StagedBatch;
import com.epsilon.prism.model.Uploads;
import com.epsilon.prism.rows.CohortWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 2 つのデバイスへの歩の詰め方（CPU）。詰めた塊の数が 2 の倍数でない歩は、最後の塊が 2 つの部分になるか、そのまま見積もりの少ない
 * デバイスに置かれる。行の数と分母 B_C は分けない参照の作り手の歩と同じで、部分は元の塊の行の割合と補助の頭の分母（部分の分母の和）を
 * 持ち、ほかの塊は自分の割合と分母を持つ。小さな網で取った 1 歩の損失も分けない歩と同じ（部分は網の詰め物の S が違うので、CPU の行列積の分け方の違いを許す）。
 * 塊の数の偶奇は行の S とトークンの上限で決まるので、上限を変えながら比べ、分けた歩が 1 つ以上あることも確かめる。
 *
 * <p>前もって並べて送った入力（{@link StagedBatch}）が、その場で作って送る入力と s・損失でビットで同じことも
 * 確かめる。
 */
public class StepFeedTest {
  private static final Logger log = LoggerFactory.getLogger(StepFeedTest.class);

  private static final PrismShape SHAPE = new PrismShape(64, 128, 64, 2, false);
  private static final int LANES = 2;
  private static final int STEP_ROWS = 160;

  private Path root;

  @BeforeClass
  public void writeCohort() throws IOException {
    root = Files.createTempDirectory("prism-feed");
    SyntheticCohort.write(CohortWriter.directory(root, 1), 1, SyntheticCohort.rows(1300, 1, 3));
  }

  @AfterClass(alwaysRun = true)
  public void deleteCohort() throws IOException {
    if (root == null) {
      return;
    }
    try (var files = Files.walk(root)) {
      for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(file);
      }
    }
  }

  @Test(timeOut = 600_000)
  public void balancedLanesSplitTheLastPieceAndKeepTheStepLoss() throws IOException {
    int compared = 0;
    int split = 0;
    try (CohortStore store = new CohortStore();
        Model model = Fit.blank(SHAPE, Device.cpu(), 7);
        NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] selected = store.select(1, 1, Selection.LABELS);
      int[] rows = Arrays.copyOf(selected, Math.min(selected.length, STEP_ROWS));
      PrismNetwork network = (PrismNetwork) model.getBlock();
      perturb(network, 0.05f, 13);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      for (int budget = 1_000; budget <= 9_000; budget += 250) {
        FitPlan plan = plan(budget);
        try (StepFeed packed = new StepFeed(store, plan, LANES, 2, false);
            StepFeed balanced = new StepFeed(store, plan, LANES, 2, true)) {
          StepFeed.Step reference = packed.prepare(rows, Teacher.RAW).join();
          StepFeed.Step candidate = balanced.prepare(rows, Teacher.RAW).join();
          if (compare(reference, candidate)) {
            split++;
            double expected = stepLoss(reference, network, loss, manager);
            double actual = stepLoss(candidate, network, loss, manager);
            assertEquals(
                actual, expected, 1e-3 * Math.max(1, Math.abs(expected)), "step loss at budget " + budget);
          }
          compared++;
          packed.release(reference);
          balanced.release(candidate);
        }
      }
    }
    log.info("PRISM-FEED budgets={} split={}", compared, split);
    assertTrue(split > 0, "no budget packed a step into an odd number of pieces");
  }

  @Test(timeOut = 600_000)
  public void stagedInputsMatchInputsSentInPlace() throws IOException {
    try (CohortStore store = new CohortStore();
        Model model = Fit.blank(SHAPE, Device.cpu(), 7);
        NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] selected = store.select(1, 1, Selection.LABELS);
      int[] rows = Arrays.copyOf(selected, Math.min(selected.length, STEP_ROWS));
      PrismNetwork network = (PrismNetwork) model.getBlock();
      perturb(network, 0.05f, 13);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      try (StepFeed feed = new StepFeed(store, plan(2_000), LANES, 2, false)) {
        StepFeed.Step step = feed.prepare(rows, Teacher.RAW).join();
        int parts = 0;
        for (Microbatch part : microbatches(step)) {
          try (NDManager scope = manager.newSubManager()) {
            Pass pass = Pass.training(Precision.FLOAT32, part.measured(), part.auxiliary());
            ParameterStore parameters = new ParameterStore(scope, false);
            Outputs inline = network.forward(parameters, part.inputs(scope, Uploads.on(scope)), pass);
            float expected =
                loss.total(
                        scope,
                        inline,
                        part.labels(),
                        part.auxiliary(),
                        1f,
                        step.denominator(),
                        part.share())
                    .getFloat();
            StagedBatch.Sent sent = part.send(scope, null);
            Outputs staged = network.forward(parameters, sent.inputs(), pass);
            float actual =
                loss.total(staged, sent.loss(), 1f, step.denominator(), part.share()).getFloat();
            assertEquals(staged.scores().toFloatArray(), inline.scores().toFloatArray());
            assertEquals(Float.floatToRawIntBits(actual), Float.floatToRawIntBits(expected));
          }
          parts++;
        }
        feed.release(step);
        assertTrue(parts > 1, "the step has " + parts + " microbatches");
      }
    }
  }

  /** 分けない歩と分ける歩の微小バッチを照らし、最後の塊を分けたなら true を返す。 */
  private static boolean compare(StepFeed.Step reference, StepFeed.Step candidate) {
    List<Microbatch> whole = microbatches(reference);
    List<Microbatch> parts = microbatches(candidate);
    assertEquals(candidate.rows(), reference.rows());
    assertEquals(candidate.denominator(), reference.denominator());
    assertEquals(rows(whole), reference.rows());
    assertEquals(rows(parts), candidate.rows());
    for (Microbatch part : whole) {
      assertOwn(part, reference.rows());
    }
    List<Microbatch> pieces = new ArrayList<>();
    for (Microbatch part : parts) {
      if (part.share() == (float) part.rows() / candidate.rows()) {
        assertOwn(part, candidate.rows());
      } else {
        pieces.add(part);
      }
    }
    if (pieces.isEmpty()) {
      assertEquals(parts.size(), whole.size());
      return false;
    }
    assertEquals(whole.size() % LANES, 1, "only a step of an odd number of pieces is split");
    assertEquals(pieces.size(), LANES);
    assertEquals(parts.size(), whole.size() + LANES - 1);
    AuxiliaryLoss.Denominators pieceDenominators = summed(pieces);
    int pieceRows = rows(pieces);
    for (Microbatch piece : pieces) {
      assertEquals(
          Float.floatToRawIntBits(piece.share()),
          Float.floatToRawIntBits((float) pieceRows / candidate.rows()),
          "the parts carry the share of the piece");
      assertTrue(
          same(piece.denominators(), pieceDenominators), "the parts carry the denominators of the piece");
    }
    assertTrue(
        whole.stream()
            .anyMatch(part -> part.rows() == pieceRows && same(part.denominators(), pieceDenominators)),
        "the split piece is not a piece of the reference step");
    return true;
  }

  /** 分けない塊の割合は自分の行の数 / 1 歩の行の数、分母は自分の教師から数えた数。 */
  private static void assertOwn(Microbatch part, int stepRows) {
    assertEquals(
        Float.floatToRawIntBits(part.share()), Float.floatToRawIntBits((float) part.rows() / stepRows));
    assertTrue(
        same(part.denominators(), AuxiliaryLoss.Denominators.of(part.auxiliary())),
        "a whole piece carries its own denominators");
  }

  /** 部分の教師から数えた分母の和。 */
  private static AuxiliaryLoss.Denominators summed(List<Microbatch> parts) {
    AuxiliaryLoss.Denominators sum = null;
    for (Microbatch part : parts) {
      AuxiliaryLoss.Denominators own = AuxiliaryLoss.Denominators.of(part.auxiliary());
      sum = sum == null ? own : sum.plus(own);
    }
    return sum;
  }

  private static boolean same(AuxiliaryLoss.Denominators a, AuxiliaryLoss.Denominators b) {
    return Arrays.equals(a.candidate(), b.candidate()) && Arrays.equals(a.reading(), b.reading());
  }

  /** 1 歩の損失 Σ_微小バッチ（学習器と同じく、前もって並べた入力と教師を順伝播の前に送り、割合と分母は微小バッチのもの）。 */
  private static double stepLoss(
      StepFeed.Step step, PrismNetwork network, PrismLoss loss, NDManager manager) {
    double total = 0;
    for (Microbatch part : microbatches(step)) {
      try (NDManager scope = manager.newSubManager()) {
        StagedBatch.Sent sent = part.send(scope, null);
        Outputs outputs =
            network.forward(
                new ParameterStore(scope, false),
                sent.inputs(),
                Pass.training(Precision.FLOAT32, part.measured(), part.auxiliary()));
        total += loss.total(outputs, sent.loss(), 1f, step.denominator(), part.share()).getFloat();
      }
    }
    return total;
  }

  private static List<Microbatch> microbatches(StepFeed.Step step) {
    List<Microbatch> all = new ArrayList<>();
    for (List<Microbatch> lane : step.lanes()) {
      all.addAll(lane);
    }
    return all;
  }

  private static int rows(List<Microbatch> parts) {
    return parts.stream().mapToInt(Microbatch::rows).sum();
  }

  private static FitPlan plan(int budget) {
    return new FitPlan(
        new FitSettings(1, STEP_ROWS, 3e-3, 3e-4, 1f, 0.01, 0.02),
        SHAPE,
        AuxiliaryLoss.standard(),
        budget,
        5);
  }

  /** 白紙の網は s ≡ 0 なので、全てのパラメーターを少し動かして s を 0 でなくする。 */
  private static void perturb(PrismNetwork network, float scale, long seed) {
    Random random = new Random(seed);
    for (Pair<String, Parameter> pair : network.getParameters()) {
      NDArray array = pair.getValue().getArray();
      float[] values = array.toFloatArray();
      for (int i = 0; i < values.length; i++) {
        values[i] += (float) random.nextGaussian() * scale;
      }
      array.set(values);
    }
  }
}
