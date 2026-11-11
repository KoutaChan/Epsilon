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
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.prism.Limits;
import com.epsilon.prism.model.Batches.Batch;
import com.epsilon.prism4p.input.InputSchema;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * ネットワークを確かめる（CPU、FP32）。見るのは、初期化したままで s ≡ 0 になること、アンカーの勾配の表、乱数の Q を学習できること、
 * 2 ステップで全てのパラメーターに勾配が届くこと、補助のヘッドの有無で s がビットで同じこと、補助と差分整合の勾配の分離、イベント 0 個と
 * 最長の行で値が有限なこと、保存と読み込み。深さ 2 と 3 の両方で確かめる。
 */
public class PrismNetworkTest {
  private static final Logger log = LoggerFactory.getLogger(PrismNetworkTest.class);

  @DataProvider
  public Object[][] depths() {
    return new Object[][] {{2}, {3}};
  }

  /** ネットワークとパラメーターのマネージャー。 */
  private static final class Net implements AutoCloseable {

    final NDManager manager = NDManager.newBaseManager(Device.cpu());
    final PrismNetwork network;
    final ParameterStore store;

    Net(PrismShape shape, int seed) {
      manager.getEngine().setRandomSeed(seed);
      network = new PrismNetwork(InputSchema.FORMAT, shape);
      network.initialize(manager);
      store = new ParameterStore(manager, false);
    }

    Outputs forward(NDManager scope, Batch batch, Pass pass) {
      return network.forward(store, batch.inputs(scope), pass);
    }

    @Override
    public void close() {
      manager.close();
    }
  }

  @Test(dataProvider = "depths")
  public void blankNetworkScoresZeroForEveryDecisionKind(int depth) {
    try (Net net = new Net(PrismShape.standard(depth), 1);
        NDManager scope = net.manager.newSubManager()) {
      Batch batch = Batches.build(Batches.allKinds(5), 11);
      float[] scores =
          net.forward(scope, batch, Pass.inference(Precision.FLOAT32)).scores().toFloatArray();
      for (float score : scores) {
        Assert.assertEquals(score, 0f, "blank score");
      }
    }
  }

  @Test
  public void anchorGradientsFollowTheMeasuredSet() {
    // 手番の行: 0 ダマ x、1 y、2 z、3 立直 x、4 暗槓、5 ツモ、6 九種、7 y と同値な打牌。遷移は手 0〜6 に 1 つずつ。
    // ゲートは 0 立直、1 暗槓、2 ツモ、3 九種。Δ_立直 < 0 なので M の最大はダマ x（d*）。
    float[] discardValues = {0.30f, 0.10f, -0.20f, 0f, 0f, 0f, 0f};
    float[] gateValues = {-0.10f, -0.05f, 0.20f, -0.30f};
    float nan = Float.NaN;
    // C = {ダマ x, 立直 x}: B(x) の勾配は g_d + g_r = 0、Δ_立直 は g_r
    Anchor both = anchor(discardValues, gateValues, new float[] {0.4f, nan, nan, 0.1f, nan, nan, nan, nan});
    Assert.assertEquals(both.discard[0], 0f, 1e-6f);
    Assert.assertEquals(both.gate[0], both.score[3], 1e-6f);
    Assert.assertTrue(Math.abs(both.score[3]) > 1e-3f);
    // C = {立直 x, y}: ダマを測っていないので B(x) は 0（sg）、Δ_立直 は g_r、B(y) は g_y
    Anchor noDama = anchor(discardValues, gateValues, new float[] {nan, 0.2f, nan, 0.3f, nan, nan, nan, nan});
    Assert.assertEquals(noDama.discard[0], 0f, 0f);
    Assert.assertEquals(noDama.gate[0], noDama.score[3], 1e-6f);
    Assert.assertEquals(noDama.discard[1], noDama.score[1], 1e-6f);
    // C = {ダマ x, 立直 x, y}: B(x) は g_d + g_r = −g_y
    Anchor three = anchor(discardValues, gateValues, new float[] {0.4f, 0.2f, nan, 0.1f, nan, nan, nan, nan});
    Assert.assertEquals(three.discard[0], three.score[0] + three.score[3], 1e-6f);
    Assert.assertEquals(three.discard[0], -three.score[1], 1e-6f);
    Assert.assertTrue(Math.abs(three.discard[0]) > 1e-3f);
    // C = {d*, ツモ}: B(d*) は g_d + g_t = 0、Δ_ツモ は g_t
    Anchor tsumo = anchor(discardValues, gateValues, new float[] {0.4f, nan, nan, nan, nan, 0.1f, nan, nan});
    Assert.assertEquals(tsumo.discard[0], 0f, 1e-6f);
    Assert.assertEquals(tsumo.gate[2], tsumo.score[5], 1e-6f);
    Assert.assertTrue(Math.abs(tsumo.score[5]) > 1e-3f);
  }

  /**
   * 同値な打牌は、組のどちらの手を測っても同じ勾配になる。測った印がツモ切りの側（番号の大きい手）だけにあっても、M と立直のアンカーは
   * 組が C にあるものとして勾配を流す。
   */
  @Test
  public void anchorGradientsTreatTwinsAsOneAction() {
    float nan = Float.NaN;
    // Batches.turn の行: 最大の打牌は y（手 1。手 7 が同値）。C = {y, ツモ}: B(y) は g_y + g_t = 0、Δ_ツモ は g_t
    float[] discardValues = {0.10f, 0.30f, -0.20f, 0f, 0f, 0f, 0f};
    float[] gateValues = {-0.10f, -0.05f, 0.20f, -0.30f};
    Anchor fromHand = anchor(discardValues, gateValues, new float[] {nan, 0.4f, nan, nan, nan, 0.1f, nan, nan});
    Anchor fromDraw = anchor(discardValues, gateValues, new float[] {nan, nan, nan, nan, nan, 0.1f, nan, 0.4f});
    for (Anchor twin : List.of(fromHand, fromDraw)) {
      Assert.assertEquals(twin.discard[1], 0f, 1e-6f);
      Assert.assertEquals(twin.gate[2], twin.score[5], 1e-6f);
      Assert.assertTrue(Math.abs(twin.score[5]) > 1e-3f);
    }
    Assert.assertEquals(fromDraw.score[5], fromHand.score[5], 1e-6f);

    // 手番の行: 0 ダマ x、1 その同値な打牌、2 y、3 立直 x（基準は 0）、4 その同値な立直（基準は 1）、5 ツモ。遷移は手 0・2・3・5 に 1 つずつ。
    // ゲートは 0・1 立直、2 ツモ。Δ_立直 > 0 なので M の最大は立直 x。
    // C = {ダマ x, 立直 x, ツモ}: B(x) は g_d + g_r + g_t = 0、Δ_立直 は 2 つのゲートを合わせて g_r + g_t、Δ_ツモ は g_t
    Batch riichi =
        Batches.build(
            List.of(
                new Batches.Row(
                    EngineDecisionKind.TURN,
                    3,
                    List.of(
                        new Batches.Move(Action.Type.DAHAI, 1, -1, 0),
                        new Batches.Move(Action.Type.DAHAI, 0, -1, 0),
                        new Batches.Move(Action.Type.DAHAI, 1, -1, 2),
                        new Batches.Move(Action.Type.RIICHI_DAHAI, 1, 0, 3),
                        new Batches.Move(Action.Type.RIICHI_DAHAI, 0, 1, 3),
                        new Batches.Move(Action.Type.TSUMO_AGARI, 1, -1, 5)))),
            5);
    float[] riichiDiscards = {0.30f, 0.10f, 0f, 0f};
    float[] riichiGates = {0.15f, 0.15f, 0.20f};
    Anchor riichiFromHand = anchor(riichi, riichiDiscards, riichiGates, new float[] {0.4f, nan, nan, 0.1f, nan, 0.2f});
    Anchor riichiFromDraw = anchor(riichi, riichiDiscards, riichiGates, new float[] {nan, 0.4f, nan, nan, 0.1f, 0.2f});
    Assert.assertEquals(riichiFromHand.gate[0], riichiFromHand.score[3] + riichiFromHand.score[5], 1e-6f);
    Assert.assertEquals(riichiFromDraw.gate[0], riichiFromDraw.score[5], 1e-6f);
    Assert.assertEquals(riichiFromDraw.gate[1], riichiFromDraw.score[4], 1e-6f);
    for (Anchor twin : List.of(riichiFromHand, riichiFromDraw)) {
      Assert.assertEquals(twin.discard[0], 0f, 1e-6f);
      Assert.assertEquals(twin.gate[2], twin.score[5], 1e-6f);
      Assert.assertTrue(Math.abs(twin.score[5]) > 1e-3f);
    }
    Assert.assertEquals(
        riichiFromDraw.gate[0] + riichiFromDraw.gate[1], riichiFromHand.gate[0] + riichiFromHand.gate[1], 1e-6f);
  }

  /**
   * 行の平均のアンカー。差分整合は行の定数に依らず、アンカーは行の定数だけを 0 に引き止める。アンカーは測った打牌 D だけで取り、勾配は
   * 行の D の手で全て同じ（D の外、応答の行、パディングは 0）。D の上の s の平均が 0 ならアンカーは 0 で、そこから D のある行に定数 c を
   * 足すとアンカーは w·c² になる。
   */
  @Test
  public void rowMeanPinsOnlyTheRowConstant() {
    Batch batch = Batches.build(List.of(Batches.turn(3), Batches.response(2)), 5);
    List<ValueLabel> labels = Batches.values(batch, 23);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      NetworkInputs inputs = batch.inputs(manager);
      BatchIndex index = BatchIndex.send(inputs, Pass.training(Precision.FLOAT32, ValueLabel.measured(labels), null));
      int rows = labels.size();
      int capacity = 0;
      for (ValueLabel label : labels) {
        capacity = Math.max(capacity, label.values().length);
      }
      Random random = new Random(29);
      float[] exact = new float[rows * capacity];
      float[] shifted = new float[rows * capacity];
      float[] noisy = new float[rows * capacity];
      float[] shifts = {0.7f, -1.3f};
      for (int row = 0; row < rows; row++) {
        float[] values = labels.get(row).values();
        // 測った打牌の上の平均を 0 にした s（ほかの手は乱数のまま）
        double sum = 0;
        int count = 0;
        for (int a = 0; a < values.length; a++) {
          if (!Float.isNaN(values[a]) && index.discardValueSlots[row * capacity + a]) {
            sum += values[a];
            count++;
          }
        }
        float mean = count == 0 ? 0f : (float) (sum / count);
        for (int a = 0; a < values.length; a++) {
          float base = Float.isNaN(values[a]) ? (float) random.nextGaussian() : values[a] - mean;
          exact[row * capacity + a] = base;
          shifted[row * capacity + a] = base + shifts[row];
          noisy[row * capacity + a] = base + (float) random.nextGaussian() * 0.1f;
        }
      }
      Shape shape = new Shape(rows, capacity);
      float atTarget = loss.rowMean(manager, outputs(manager.create(exact, shape), index), labels).getFloat();
      Assert.assertEquals(atTarget, 0f, 1e-10f);
      float pinned = loss.rowMean(manager, outputs(manager.create(shifted, shape), index), labels).getFloat();
      double expected = 0;
      for (int row = 0; row < rows; row++) {
        if (hasMeasuredDiscard(labels.get(row), index, row, capacity)) {
          expected += labels.get(row).weight() * shifts[row] * shifts[row];
        }
      }
      Assert.assertTrue(expected > 0, "the turn row must have a measured discard");
      Assert.assertEquals(pinned, (float) expected, 1e-5f);
      float valueExact = loss.value(manager, outputs(manager.create(exact, shape), index), labels).getFloat();
      float valueShifted = loss.value(manager, outputs(manager.create(shifted, shape), index), labels).getFloat();
      Assert.assertEquals(valueShifted, valueExact, 1e-5f);
      NDArray leaf = manager.create(noisy, shape);
      leaf.setRequiresGradient(true);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        collector.backward(loss.rowMean(manager, outputs(leaf, index), labels));
      }
      float[] gradient = leaf.getGradient().toFloatArray();
      for (int row = 0; row < rows; row++) {
        float[] values = labels.get(row).values();
        Float common = null;
        for (int a = 0; a < capacity; a++) {
          float g = gradient[row * capacity + a];
          if (a >= values.length || Float.isNaN(values[a]) || !index.discardValueSlots[row * capacity + a]) {
            Assert.assertEquals(g, 0f, 0f, "row " + row + " action " + a + " is outside D");
          } else if (common == null) {
            common = g;
          } else {
            Assert.assertEquals(g, common, 1e-6f, "row " + row + " action " + a);
          }
        }
      }
    }
  }

  /** 行に測った打牌があるか。 */
  private static boolean hasMeasuredDiscard(ValueLabel label, BatchIndex index, int row, int capacity) {
    float[] values = label.values();
    for (int a = 0; a < values.length; a++) {
      if (!Float.isNaN(values[a]) && index.discardValueSlots[row * capacity + a]) {
        return true;
      }
    }
    return false;
  }

  /** 葉の s をネットワークの出力の形にする（行の手の数と合法手はバッチのインデックスから取る）。 */
  private static Outputs outputs(NDArray scores, BatchIndex index) {
    return new Outputs(scores, index.legal.array(), index.rowActions, index.discardValueSlots, null);
  }

  /** アンカーのテストの結果: B・Δ の勾配と、同じ s を葉にしたときの s の勾配 g。 */
  private record Anchor(float[] discard, float[] gate, float[] score) {}

  private static Anchor anchor(float[] discardValues, float[] gateValues, float[] targets) {
    return anchor(Batches.build(List.of(Batches.turn(3)), 5), discardValues, gateValues, targets);
  }

  private static Anchor anchor(Batch batch, float[] discardValues, float[] gateValues, float[] targets) {
    List<ValueLabel> labels = List.of(new ValueLabel(targets, 0.1f, 1f));
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      NetworkInputs inputs = batch.inputs(manager);
      BatchIndex index = BatchIndex.send(inputs, Pass.training(Precision.FLOAT32, ValueLabel.measured(labels), null));
      NDArray discard = manager.create(discardValues);
      NDArray gate = manager.create(gateValues);
      discard.setRequiresGradient(true);
      gate.setRequiresGradient(true);
      float[] composed;
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray scores = ValueGraphHead.compose(manager, index, discard, gate);
        composed = scores.toFloatArray();
        collector.backward(
            loss.value(manager, new Outputs(scores, index.legal.array(), index.rowActions, index.discardValueSlots, null), labels));
      }
      NDArray leaf = manager.create(composed, new Shape(1, composed.length));
      leaf.setRequiresGradient(true);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        collector.backward(
            loss.value(manager, new Outputs(leaf, index.legal.array(), index.rowActions, index.discardValueSlots, null), labels));
      }
      return new Anchor(
          discard.getGradient().toFloatArray(),
          gate.getGradient().toFloatArray(),
          leaf.getGradient().toFloatArray());
    }
  }

  @Test(dataProvider = "depths")
  public void fitsRandomValues(int depth) {
    try (Net net = new Net(Batches.small(depth), 3);
        NDManager scope = net.manager.newSubManager()) {
      Batch batch = Batches.build(Batches.allKinds(4), 13);
      List<ValueLabel> labels = new ArrayList<>(Batches.values(batch, 17));
      // 同値な手は同じ s になるので、教師にも同じ Q を置く
      float[] turn = labels.get(0).values();
      turn[7] = turn[1];
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      Optimizer optimizer = Batches.adam(3e-3f);
      for (int step = 0; step < 400; step++) {
        try (NDManager stepScope = scope.newSubManager();
            GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
          Outputs outputs = net.forward(stepScope, batch, Pass.training(Precision.FLOAT32, measured, null));
          collector.backward(loss.total(stepScope, outputs, labels, null, 1f, labels.size(), 1f));
        }
        Batches.step(net.network, optimizer);
      }
      float[] scores =
          net.forward(scope, batch, Pass.inference(Precision.FLOAT32)).scores().toFloatArray();
      int capacity = scores.length / labels.size();
      float worst = 0;
      for (int row = 0; row < labels.size(); row++) {
        float[] target = labels.get(row).values();
        double mean = 0;
        for (int a = 0; a < target.length; a++) {
          mean += scores[row * capacity + a] - target[a];
        }
        mean /= target.length;
        for (int a = 0; a < target.length; a++) {
          worst = Math.max(worst, (float) Math.abs(scores[row * capacity + a] - target[a] - mean));
        }
      }
      log.info("PRISM-T4 depth={} fit centered residual max={}", depth, String.format("%.4f", worst));
      Assert.assertTrue(worst < 0.05f, "centered residual " + worst);
    }
  }

  @Test(dataProvider = "depths")
  public void everyParameterReceivesGradientWithinTwoSteps(int depth) {
    try (Net net = new Net(Batches.small(depth), 5);
        NDManager scope = net.manager.newSubManager()) {
      Batch batch = Batches.build(Batches.allKinds(6), 19);
      List<ValueLabel> labels = Batches.values(batch, 23);
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 29);
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      Optimizer optimizer = Batches.adam(1e-3f);
      Set<String> reached = new HashSet<>();
      for (int step = 0; step < 2; step++) {
        try (NDManager stepScope = scope.newSubManager();
            GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
          Outputs outputs =
              net.forward(stepScope, batch, Pass.training(Precision.FLOAT32, measured, auxiliary));
          collector.backward(loss.total(stepScope, outputs, labels, auxiliary, 1f, labels.size(), 1f));
        }
        for (Pair<String, Parameter> pair : net.network.getParameters()) {
          try (NDArray gradient = pair.getValue().getArray().getGradient()) {
            float[] values = gradient.toFloatArray();
            if (Batches.maxAbs(values) > 0) {
              reached.add(pair.getKey());
            }
          }
        }
        Batches.step(net.network, optimizer);
      }
      List<String> missing = new ArrayList<>();
      for (Pair<String, Parameter> pair : net.network.getParameters()) {
        if (!reached.contains(pair.getKey())) {
          missing.add(pair.getKey());
        }
      }
      Assert.assertTrue(missing.isEmpty(), "no gradient: " + missing);
    }
  }

  @Test(dataProvider = "depths")
  public void auxiliaryHeadsDoNotChangeScores(int depth) {
    try (Net net = new Net(Batches.small(depth), 7);
        NDManager scope = net.manager.newSubManager()) {
      Batches.perturb(net.network, 0.05f, 31);
      Batch batch = Batches.build(Batches.allKinds(5), 37);
      List<ValueLabel> labels = Batches.values(batch, 41);
      boolean[] measured = ValueLabel.measured(labels);
      float[] plain =
          net.forward(scope, batch, Pass.inference(Precision.FLOAT32)).scores().toFloatArray();
      float[] trained =
          net.forward(scope, batch, Pass.training(Precision.FLOAT32, measured, null))
              .scores()
              .toFloatArray();
      Outputs withHeads =
          net.forward(
              scope,
              batch,
              Pass.training(Precision.FLOAT32, measured, Batches.auxiliary(batch, 43)));
      Assert.assertNotNull(withHeads.auxiliary());
      Assert.assertTrue(Batches.maxAbs(plain) > 1e-4f, "scores must be non-trivial");
      Assert.assertTrue(Arrays.equals(plain, trained));
      Assert.assertTrue(Arrays.equals(plain, withHeads.scores().toFloatArray()));
    }
  }

  @Test(dataProvider = "depths")
  public void auxiliaryAndValueGradientsAreSeparated(int depth) {
    try (Net net = new Net(Batches.small(depth), 9);
        NDManager scope = net.manager.newSubManager()) {
      Batches.perturb(net.network, 0.05f, 47);
      Batch batch = Batches.build(Batches.allKinds(5), 53);
      List<ValueLabel> labels = Batches.values(batch, 59);
      AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 61);
      boolean[] measured = ValueLabel.measured(labels);
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());

      try (NDManager stepScope = scope.newSubManager();
          GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
        Outputs outputs =
            net.forward(stepScope, batch, Pass.training(Precision.FLOAT32, measured, auxiliary));
        collector.backward(
            loss.auxiliary().terms(stepScope, outputs.auxiliary(), auxiliary, 1f).total());
      }
      Map<String, float[]> fromAuxiliary = Batches.takeGradients(net.network);
      try (NDManager stepScope = scope.newSubManager();
          GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
        Outputs outputs =
            net.forward(stepScope, batch, Pass.training(Precision.FLOAT32, measured, auxiliary));
        collector.backward(loss.value(stepScope, outputs, labels));
      }
      Map<String, float[]> fromValue = Batches.takeGradients(net.network);

      for (Map.Entry<String, float[]> entry : fromAuxiliary.entrySet()) {
        String name = entry.getKey();
        float auxiliaryMax = Batches.maxAbs(entry.getValue());
        float valueMax = Batches.maxAbs(fromValue.get(name));
        if (name.contains("discardValue_") || name.contains("gates_")) {
          Assert.assertEquals(auxiliaryMax, 0f, name + " from the auxiliary loss");
          Assert.assertTrue(valueMax > 0, name + " from the value loss");
        } else if (name.contains("reading_") || name.contains("outcome_")) {
          Assert.assertEquals(valueMax, 0f, name + " from the value loss");
          Assert.assertTrue(auxiliaryMax > 0, name + " from the auxiliary loss");
        }
      }
      Assert.assertTrue(Batches.maxAbs(fromAuxiliary.get(firstOf(fromAuxiliary, "encoder0_qkv"))) > 0);
    }
  }

  @Test(dataProvider = "depths")
  public void emptyAndLongestRowsStayFinite(int depth) {
    try (Net net = new Net(Batches.small(depth), 11);
        NDManager scope = net.manager.newSubManager()) {
      Batches.perturb(net.network, 0.05f, 67);
      List<List<Batches.Row>> shapes =
          List.of(
              List.of(
                  Batches.turn(0),
                  Batches.turn(Limits.MAX_EVENT_TOKENS),
                  Batches.response(0),
                  Batches.postCall(Limits.MAX_EVENT_TOKENS)),
              List.of(Batches.turn(0), Batches.chankan(0)));
      PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
      for (List<Batches.Row> rows : shapes) {
        Batch batch = Batches.build(rows, 71);
        List<ValueLabel> labels = Batches.values(batch, 73);
        AuxiliaryLabels auxiliary = Batches.auxiliary(batch, 79);
        try (NDManager stepScope = scope.newSubManager();
            GradientCollector collector = net.manager.getEngine().newGradientCollector()) {
          Outputs outputs =
              net.forward(
                  stepScope,
                  batch,
                  Pass.training(Precision.FLOAT32, ValueLabel.measured(labels), auxiliary));
          assertFinite("scores", outputs.scores().toFloatArray());
          assertFinite("candidates", outputs.auxiliary().candidates().toFloatArray());
          assertFinite("reading", outputs.auxiliary().reading().toFloatArray());
          NDArray total = loss.total(stepScope, outputs, labels, auxiliary, 1f, labels.size(), 1f);
          Assert.assertTrue(Float.isFinite(total.getFloat()));
          collector.backward(total);
        }
        for (Map.Entry<String, float[]> entry : Batches.takeGradients(net.network).entrySet()) {
          assertFinite(entry.getKey(), entry.getValue());
        }
      }
    }
  }

  @Test
  public void saveAndLoadKeepScoresAndCheckFingerprints() throws IOException {
    Path root = Files.createTempDirectory("prism-model");
    Path directory = root.resolve("model");
    try (Net net = new Net(Batches.small(3), 13);
        NDManager scope = net.manager.newSubManager()) {
      Batches.perturb(net.network, 0.05f, 83);
      Batch batch = Batches.build(Batches.allKinds(4), 89);
      float[] before =
          net.forward(scope, batch, Pass.inference(Precision.FLOAT32)).scores().toFloatArray();
      net.network.save(directory);
      Assert.assertEquals(PrismNetwork.readShape(directory), Batches.small(3));
      PrismNetwork loaded = PrismNetwork.load(directory, net.manager);
      float[] after =
          loaded
              .forward(net.store, batch.inputs(scope), Pass.inference(Precision.FLOAT32))
              .scores()
              .toFloatArray();
      Assert.assertTrue(Arrays.equals(before, after));
      for (Pair<String, Parameter> pair : loaded.inferenceParameters()) {
        Assert.assertFalse(pair.getKey().contains("reading_") || pair.getKey().contains("outcome_"));
      }
      Path metadata = directory.resolve("model.json");
      String json = Files.readString(metadata);
      Files.writeString(metadata, json.replace(InputSchema.fingerprint(), "0000000000000000"));
      Assert.assertThrows(IOException.class, () -> PrismNetwork.readShape(directory));
    } finally {
      try (var files = Files.list(directory)) {
        for (Path file : files.toList()) {
          Files.delete(file);
        }
      }
      Files.delete(directory);
      Files.delete(root);
    }
  }

  private static String firstOf(Map<String, float[]> gradients, String suffix) {
    for (String name : gradients.keySet()) {
      if (name.endsWith(suffix)) {
        return name;
      }
    }
    throw new AssertionError("no parameter " + suffix);
  }

  private static void assertFinite(String what, float[] values) {
    for (float value : values) {
      Assert.assertTrue(Float.isFinite(value), what + " is not finite");
    }
  }
}
