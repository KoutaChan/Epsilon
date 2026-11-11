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
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.StagedBatch;
import com.epsilon.prism.model.ValueLabel;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism4p.input.InputSchema;
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
 * 1 ステップ分のバッチを 2 つのデバイスに詰める方法を確かめる（CPU）。詰めたバッチの数が 2 の倍数でないステップでは、最後のバッチを
 * 2 つの部分に分ける。参照は 1 つのデバイスに詰めたステップで、バッチを分けない。1 つのデバイスではバッチの数の見積もりを 2 の倍数に
 * 丸めないので、見積もりが奇数の上限では詰め方が違うことがある。見積もりが偶数の上限では詰め方は同じで、分ける前のバッチが行（教師の
 * 中身で見分ける）と補助のヘッドの分母で参照のバッチと 1 対 1 に対応しなければ失敗にする。見積もりが奇数の上限は、対応するステップだけを参照と比べる。
 * 行の数と分母 B_C は参照のステップと同じ。分けたステップの部分は 2 つで、元のバッチの行の割合と補助のヘッドの分母（部分の分母の和）を
 * 持ち、ほかのバッチは自分の割合と分母を持つ（参照と対応しないステップでも確かめる）。小さなネットワークで求めた 1 ステップの損失も、
 * 参照のステップと同じ（部分はパディングした S が違うので、CPU の行列積の分け方の違いは許す）。バッチの数の偶奇は行の S とトークンの
 * 上限で決まるので、上限を変えながら比べ、分けたステップが 1 つ以上あることも確かめる。
 *
 * <p>前もって並べて送った入力（{@link StagedBatch}）が、その場で作って送る入力と、s と損失でビットで同じことも
 * 確かめる。分けた部分は元のバッチの分母を持ち、その場で求める損失は部分の教師から分母を数えるので、こちらはバッチを分けない 1 つの
 * デバイスで確かめる。
 */
public class StepFeedTest {
  private static final Logger log = LoggerFactory.getLogger(StepFeedTest.class);

  private static final PrismShape SHAPE = new PrismShape(64, 128, 64, 2, null);
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
        Model model = Fit.blank(InputSchema.FORMAT, SHAPE, Device.cpu(), 7);
        NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] selected = store.select(1, 1, Selection.LABELS);
      int[] rows = Arrays.copyOf(selected, Math.min(selected.length, STEP_ROWS));
      PrismNetwork network = (PrismNetwork) model.getBlock();
      perturb(network, 0.05f, 13);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      for (int budget = 1_000; budget <= 9_000; budget += 250) {
        FitPlan plan = plan(budget);
        try (StepFeed single = new StepFeed(store, plan, 1, 2);
            StepFeed balanced = new StepFeed(store, plan, LANES, 2)) {
          StepFeed.Step reference = single.prepare(rows, Teacher.RAW).join();
          StepFeed.Step candidate = balanced.prepare(rows, Teacher.RAW).join();
          Packing packing = compare(reference, candidate);
          if (Math.ceilDiv(tokens(reference), budget) % LANES == 0) {
            assertTrue(
                packing != Packing.DIFFERENT,
                "the unsplit pieces differ from the reference step at budget " + budget);
          }
          if (packing == Packing.SPLIT) {
            split++;
            double expected = stepLoss(reference, network, loss, manager);
            double actual = stepLoss(candidate, network, loss, manager);
            assertEquals(
                actual, expected, 1e-3 * Math.max(1, Math.abs(expected)), "step loss at budget " + budget);
          }
          if (packing != Packing.DIFFERENT) {
            compared++;
          }
          single.release(reference);
          balanced.release(candidate);
        }
      }
    }
    log.info("PRISM-FEED budgets={} split={}", compared, split);
    assertTrue(split > 0, "no budget split a step packed like the single-lane reference");
  }

  @Test(timeOut = 600_000)
  public void stagedInputsMatchInputsSentInPlace() throws IOException {
    try (CohortStore store = new CohortStore();
        Model model = Fit.blank(InputSchema.FORMAT, SHAPE, Device.cpu(), 7);
        NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      store.addCohort(CohortWriter.directory(root, 1));
      int[] selected = store.select(1, 1, Selection.LABELS);
      int[] rows = Arrays.copyOf(selected, Math.min(selected.length, STEP_ROWS));
      PrismNetwork network = (PrismNetwork) model.getBlock();
      perturb(network, 0.05f, 13);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      try (StepFeed feed = new StepFeed(store, plan(2_000), 1, 2)) {
        StepFeed.Step step = feed.prepare(rows, Teacher.RAW).join();
        int parts = 0;
        for (Microbatch part : microbatches(step)) {
          try (NDManager scope = manager.newSubManager()) {
            Pass pass = Pass.training(Precision.FLOAT32, part.measured(), part.auxiliary());
            ParameterStore parameters = new ParameterStore(scope, false);
            Outputs inline =
                network.forward(parameters, part.inputs(scope, PinnedUploads.on(scope)), pass);
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
            StagedBatch.Sent sent = part.send(scope, null).receive();
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

  /** 1 つのデバイスに詰めた参照と比べた、2 つのデバイスに詰めたステップの詰め方。 */
  private enum Packing {
    /** 分ける前のバッチが参照のバッチと対応しない（バッチの数の見積もりが 2 の倍数の上限では失敗）。 */
    DIFFERENT,
    /** 参照と同じに詰め、バッチを分けていない。 */
    WHOLE,
    /** 参照と同じに詰め、最後のバッチを分けた。 */
    SPLIT
  }

  /** 1 つのデバイスに詰めた参照のステップと、2 つのデバイスに詰めたステップのマイクロバッチを照合する。 */
  private static Packing compare(StepFeed.Step reference, StepFeed.Step candidate) {
    List<Microbatch> whole = microbatches(reference);
    List<Microbatch> parts = microbatches(candidate);
    assertEquals(candidate.rows(), reference.rows());
    assertEquals(candidate.denominator(), reference.denominator());
    assertEquals(rows(whole), reference.rows());
    assertEquals(rows(parts), candidate.rows());
    for (Microbatch part : whole) {
      assertOwn(part, reference.rows());
    }
    List<Microbatch> own = new ArrayList<>();
    List<Microbatch> pieces = new ArrayList<>();
    for (Microbatch part : parts) {
      if (part.share() == (float) part.rows() / candidate.rows()) {
        assertOwn(part, candidate.rows());
        own.add(part);
      } else {
        pieces.add(part);
      }
    }
    AuxiliaryLoss.Denominators pieceDenominators = null;
    int pieceRows = rows(pieces);
    if (!pieces.isEmpty()) {
      assertEquals(own.size() % LANES, 0, "only a step of an odd number of pieces is split");
      assertEquals(pieces.size(), LANES);
      pieceDenominators = summed(pieces);
      for (Microbatch piece : pieces) {
        assertEquals(
            Float.floatToRawIntBits(piece.share()),
            Float.floatToRawIntBits((float) pieceRows / candidate.rows()),
            "the parts carry the share of the piece");
        assertTrue(
            same(piece.denominators(), pieceDenominators), "the parts carry the denominators of the piece");
      }
    }
    List<Microbatch> left = new ArrayList<>(whole);
    for (Microbatch part : own) {
      if (!remove(left, labels(List.of(part)), part.denominators())) {
        return Packing.DIFFERENT;
      }
    }
    if (pieces.isEmpty()) {
      return left.isEmpty() ? Packing.WHOLE : Packing.DIFFERENT;
    }
    return remove(left, labels(pieces), pieceDenominators) && left.isEmpty() ? Packing.SPLIT : Packing.DIFFERENT;
  }

  /** 同じ行（教師の中身の多重集合）と補助のヘッドの分母を持つマイクロバッチを 1 つ取り除き、あれば true を返す。 */
  private static boolean remove(
      List<Microbatch> batches, List<String> labels, AuxiliaryLoss.Denominators denominators) {
    for (int index = 0; index < batches.size(); index++) {
      Microbatch batch = batches.get(index);
      if (same(batch.denominators(), denominators) && labels(List.of(batch)).equals(labels)) {
        batches.remove(index);
        return true;
      }
    }
    return false;
  }

  /** マイクロバッチの行の教師を、中身を表す文字列にして並べ替えたもの。合成のコホートの教師は行ごとに違う。 */
  private static List<String> labels(List<Microbatch> parts) {
    List<String> labels = new ArrayList<>();
    for (Microbatch part : parts) {
      for (ValueLabel label : part.labels()) {
        labels.add(Arrays.toString(label.values()) + " " + label.lambda() + " " + label.weight());
      }
    }
    labels.sort(null);
    return labels;
  }

  /** 分けないバッチの割合が自分の行の数 / 1 ステップの行の数で、分母が自分の教師から数えた数であることを確かめる。 */
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

  /** 1 ステップの損失（マイクロバッチの和）。学習器と同じく、前もって並べた入力と教師を順伝播の前に送り、割合と分母はマイクロバッチのものを使う。 */
  private static double stepLoss(
      StepFeed.Step step, PrismNetwork network, PrismLoss loss, NDManager manager) {
    double total = 0;
    for (Microbatch part : microbatches(step)) {
      try (NDManager scope = manager.newSubManager()) {
        StagedBatch.Sent sent = part.send(scope, null).receive();
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

  /** 1 ステップの、パディングを除いたトークン数（バッチの数の見積もりの元）。 */
  private static long tokens(StepFeed.Step step) {
    long total = 0;
    for (Microbatch part : microbatches(step)) {
      BatchPlan plan = part.batch().plan();
      for (int row = 0; row < plan.rows(); row++) {
        total += plan.format().fixedTokens() + plan.events(row);
      }
    }
    return total;
  }

  private static FitPlan plan(int budget) {
    return new FitPlan(
        new FitSettings(1, STEP_ROWS, 3e-3, 3e-4, 1f, 0.01, 0.02),
        SHAPE,
        AuxiliaryLoss.standard(),
        budget,
        5);
  }

  /** 初期化したままのネットワークは s ≡ 0 なので、全てのパラメーターを少し動かして s を 0 でなくする。 */
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
