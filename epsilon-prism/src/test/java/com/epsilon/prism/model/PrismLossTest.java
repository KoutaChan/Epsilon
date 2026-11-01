package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.model.Batches.Batch;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 損失の 2 段（CPU の FP32）。
 *
 * <ul>
 *   <li>順伝播の前に送る教師（{@link LossInputs}）の損失と、s・補助の頭の出力への勾配が、損失の中で教師を作る参照の道（{@link ReferenceLoss}）と
 *       ビットで同じ（小さな網の出力を葉にする。補助の頭のある順伝播とない順伝播）。GPU は {@link PrismLossGpuTest}。
 *   <li>詰め方の 1 塊を分けた部分が元の塊の分母と割合を持てば、補助の項の和が元の塊の項と頭ごとに相対 1e-6 で同じで、差分整合・行の平均の
 *       錨・1 歩の損失も同じ（FP32 の和の順の違いだけ）。
 * </ul>
 */
public class PrismLossTest {
  private static final Logger log = LoggerFactory.getLogger(PrismLossTest.class);

  /** 比べる損失の TEMPO のゲイン K′・B_C・割合（1 でない値で掛け算と割り算も通す）。 */
  private static final float GAIN = 0.7f;

  private static final int DENOMINATOR = 13;
  private static final float SHARE = 0.375f;

  /** 行の類の温度 λ_c の例。 */
  private static final float[] LAMBDAS = {0.05f, 0.1f, 0.2f};

  /** 行の出来事の数の例。 */
  private static final int[] EVENTS = {0, 4, 9, 17, 30};

  /** 自分の和了も放銃もない局の終わり方。 */
  private static final Outcome[] QUIET = {
    Outcome.OTHER_TSUMO, Outcome.OTHER_RON, Outcome.DRAW_TENPAI, Outcome.DRAW_NOTEN, Outcome.ABORTIVE
  };

  private static final int READING_WIDTH = AuxiliaryLabels.OPPONENTS * ReadingTarget.WIDTH;

  @Test
  public void stagedLossIsBitwiseTheInlineLoss() {
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      manager.getEngine().setRandomSeed(3);
      PrismNetwork network = new PrismNetwork(Batches.small(2));
      network.initialize(manager);
      Batches.perturb(network, 0.05f, 7);
      Batch batch = Batches.build(rows(24), 5);
      List<ValueLabel> labels = labels(batch, 11, -0.5f);
      assertStagedLossMatchesTheReference(
          network, manager, Precision.FLOAT32, batch, labels, Batches.auxiliary(batch, 13));
      assertStagedLossMatchesTheReference(network, manager, Precision.FLOAT32, batch, labels, null);
    }
  }

  /**
   * 前の 8 行は打牌だけの行（手の容量 4）で、その候補には自分の和了も放銃もなく、相手は聴牌でなく放銃の牌もないので、前の部分のその頭の
   * 項は 0。網の出力は行・候補・読みの行ごとに決めた乱数で、元の塊と部分で同じ値。Y は正（ℓ_PRISM が全ての行で負になり、和が打ち消し
   * 合わない）。
   */
  @Test
  public void splitPiecesKeepTheReferenceLoss() {
    int cut = 8;
    List<Batches.Row> rows = new ArrayList<>();
    for (int r = 0; r < cut; r++) {
      rows.add(Batches.postCall(EVENTS[r % EVENTS.length]));
    }
    rows.addAll(rows(16));
    int end = rows.size();
    Piece piece = Piece.of(rows, cut, new Random(17));
    int denominator = 0;
    for (ValueLabel label : piece.labels()) {
      if (PrismLoss.contributes(label)) {
        denominator++;
      }
    }
    float share = (float) end / (end + 40);
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
      Part all = piece.part(manager, 0, end);
      Part first = piece.part(manager, 0, cut);
      Part second = piece.part(manager, cut, end);
      AuxiliaryLoss.Denominators reference = AuxiliaryLoss.Denominators.of(all.auxiliary());
      AuxiliaryLoss.Denominators summed =
          AuxiliaryLoss.Denominators.of(first.auxiliary())
              .plus(AuxiliaryLoss.Denominators.of(second.auxiliary()));
      Assert.assertTrue(Arrays.equals(summed.candidate(), reference.candidate()), "candidate denominators");
      Assert.assertTrue(Arrays.equals(summed.reading(), reference.reading()), "reading denominators");
      LossInputs allInputs = all.inputs(manager, reference);
      LossInputs firstInputs = first.inputs(manager, reference);
      LossInputs secondInputs = second.inputs(manager, reference);

      AuxiliaryLoss.Terms allTerms = all.terms(loss, manager, allInputs);
      AuxiliaryLoss.Terms firstTerms = first.terms(loss, manager, firstInputs);
      AuxiliaryLoss.Terms secondTerms = second.terms(loss, manager, secondInputs);
      for (CandidateTarget target : CandidateTarget.values()) {
        assertSplit(
            target.name(),
            allTerms.candidate().get(target),
            firstTerms.candidate().get(target),
            secondTerms.candidate().get(target));
      }
      for (ReadingTarget target : ReadingTarget.values()) {
        assertSplit(
            target.name(),
            allTerms.reading().get(target),
            firstTerms.reading().get(target),
            secondTerms.reading().get(target));
      }
      assertSplit("auxiliary", allTerms.total(), firstTerms.total(), secondTerms.total());
      for (NDArray empty :
          List.of(
              firstTerms.candidate().get(CandidateTarget.YAKU),
              firstTerms.candidate().get(CandidateTarget.DEAL_IN_POINTS),
              firstTerms.reading().get(ReadingTarget.WAITS),
              firstTerms.reading().get(ReadingTarget.DANGER_POINTS))) {
        Assert.assertEquals(Float.floatToRawIntBits(empty.getFloat()), 0, "a head without units in the part");
      }

      // 差分整合と行の平均の錨は行の和（分母は 1 歩の B_C）なので、分けても和の順の違いだけ
      NDArray value = loss.value(manager, all.outputs(), all.labels());
      NDArray rowMean = loss.rowMean(manager, all.outputs(), all.labels());
      assertSplit(
          "value",
          value,
          loss.value(manager, first.outputs(), first.labels()),
          loss.value(manager, second.outputs(), second.labels()));
      assertSplit(
          "row mean",
          rowMean,
          loss.rowMean(manager, first.outputs(), first.labels()),
          loss.rowMean(manager, second.outputs(), second.labels()));

      float expected = loss.total(all.outputs(), allInputs, GAIN, denominator, share).getFloat();
      double actual =
          (double) loss.total(first.outputs(), firstInputs, GAIN, denominator, share).getFloat()
              + loss.total(second.outputs(), secondInputs, GAIN, denominator, share).getFloat();
      // 部分ごとの項はどれも符号が一定なので、許す差は項の大きさの和に比べる
      double magnitude =
          (Math.abs(value.getFloat()) + PrismLoss.ROW_MEAN * Math.abs(rowMean.getFloat())) / denominator
              + share * allTerms.total().getFloat();
      log.info(
          "PRISM-SPLIT step loss {} vs {} (magnitude {})",
          String.format("%.9f", actual),
          String.format("%.9f", expected),
          String.format("%.6f", magnitude));
      Assert.assertEquals(actual, expected, 2e-6 * magnitude, "step loss");
    }
  }

  /**
   * 網の出力を葉にして、送った教師の道と参照の道（{@link ReferenceLoss}）の損失と、s・補助の頭の出力への勾配がビットで同じことを確かめる。
   * 網の中の逆伝播は葉の勾配の決まった関数なので、比べるのは葉まで（GPU の index_add の原子的な足し算の順に左右されない）。
   *
   * @param network 網
   * @param manager パラメーターの管理元
   * @param precision 順伝播の計算の型
   * @param batch 塊
   * @param labels 行の順の教師
   * @param auxiliary 補助の頭の教師（補助の頭を計算しないなら null）
   */
  static void assertStagedLossMatchesTheReference(
      PrismNetwork network,
      NDManager manager,
      Precision precision,
      Batch batch,
      List<ValueLabel> labels,
      AuxiliaryLabels auxiliary) {
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    try (NDManager scope = manager.newSubManager()) {
      NetworkInputs inputs = batch.inputs(scope);
      Staging staging = new Staging();
      LossInputs staged =
          LossInputs.stage(
              inputs,
              labels,
              auxiliary,
              auxiliary == null ? null : AuxiliaryLoss.Denominators.of(auxiliary),
              staging);
      staging.send(inputs.uploads(), scope);
      Outputs outputs =
          network.forward(
              new ParameterStore(manager, false),
              inputs,
              Pass.training(precision, ValueLabel.measured(labels), auxiliary));
      Leaves inline = Leaves.of(scope, outputs);
      Leaves sent = Leaves.of(scope, outputs);
      float expected;
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray total =
            ReferenceLoss.total(
                scope, loss.auxiliary(), inline.outputs(), labels, auxiliary, GAIN, DENOMINATOR, SHARE);
        collector.backward(total);
        expected = total.getFloat();
      }
      float actual;
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray total = loss.total(sent.outputs(), staged, GAIN, DENOMINATOR, SHARE);
        collector.backward(total);
        actual = total.getFloat();
      }
      log.info(
          "PRISM-LOSS {} auxiliary={} loss {} vs {}",
          precision,
          auxiliary != null,
          String.format("%.9g", actual),
          String.format("%.9g", expected));
      Assert.assertEquals(
          Float.floatToRawIntBits(actual),
          Float.floatToRawIntBits(expected),
          "loss " + actual + " vs " + expected);
      assertSameGradient("scores", inline.scores(), sent.scores());
      if (auxiliary != null) {
        assertSameGradient("candidates", inline.candidates(), sent.candidates());
        assertSameGradient("reading", inline.reading(), sent.reading());
      }
    }
  }

  /**
   * 4 種の判断の行を、出来事の数を変えて並べる。
   *
   * @param count 行の数
   * @return 行の形
   */
  static List<Batches.Row> rows(int count) {
    List<Batches.Row> rows = new ArrayList<>(count);
    for (int r = 0; r < count; r++) {
      int events = EVENTS[r % EVENTS.length];
      rows.add(
          switch (r % 4) {
            case 0 -> Batches.turn(events);
            case 1 -> Batches.response(events);
            case 2 -> Batches.chankan(events);
            default -> Batches.postCall(events);
          });
    }
    return rows;
  }

  /**
   * 乱数の教師。Y は [low, low + 1)。行の 5 つに 1 つは測った手がなく（|C| = 0）、5 つに 1 つは測った手が 1 つ（|C| = 1）、ほかは手の 3 割を
   * 測らない（NaN）。λ と行の重みも行で変える。
   *
   * @param batch 塊
   * @param seed 乱数の種
   * @param low Y の下限
   * @return 行の順の教師
   */
  static List<ValueLabel> labels(Batch batch, long seed, float low) {
    Random random = new Random(seed);
    List<ValueLabel> labels = new ArrayList<>(batch.rows().size());
    for (int r = 0; r < batch.rows().size(); r++) {
      float[] values = new float[batch.rows().get(r).moves().size()];
      for (int a = 0; a < values.length; a++) {
        boolean measured =
            switch (r % 5) {
              case 0 -> false;
              case 1 -> a == 0;
              default -> random.nextFloat() >= 0.3f;
            };
        values[a] = measured ? low + random.nextFloat() : Float.NaN;
      }
      labels.add(new ValueLabel(values, LAMBDAS[r % LAMBDAS.length], 0.5f + 1.5f * random.nextFloat()));
    }
    return labels;
  }

  /** 部分の項の和が元の塊の項と相対 1e-6 で同じ。 */
  private static void assertSplit(String what, NDArray whole, NDArray first, NDArray second) {
    float expected = whole.getFloat();
    double actual = (double) first.getFloat() + second.getFloat();
    Assert.assertEquals(actual, expected, 1e-6 * Math.abs(expected), what + " split " + actual + " vs " + expected);
  }

  /** 2 つの葉の勾配がビットで同じで、0 でない。 */
  private static void assertSameGradient(String what, NDArray expected, NDArray actual) {
    try (NDArray left = expected.getGradient();
        NDArray right = actual.getGradient()) {
      float[] reference = left.toFloatArray();
      float[] staged = right.toFloatArray();
      Assert.assertEquals(staged.length, reference.length, what);
      for (int i = 0; i < reference.length; i++) {
        Assert.assertEquals(
            Float.floatToRawIntBits(staged[i]),
            Float.floatToRawIntBits(reference[i]),
            what + " gradient " + i + ": " + staged[i] + " vs " + reference[i]);
      }
      Assert.assertTrue(Batches.maxAbs(reference) > 0, what + " gradients must be non-trivial");
    }
  }

  /** 網の出力を葉にしたもの（s と補助の頭の出力をホストを通して写し、勾配を要求する）。 */
  private record Leaves(NDArray scores, NDArray candidates, NDArray reading, Outputs outputs) {

    static Leaves of(NDManager scope, Outputs outputs) {
      NDArray scores = leaf(scope, outputs.scores());
      Outputs.Auxiliary heads = outputs.auxiliary();
      NDArray candidates = heads == null ? null : leaf(scope, heads.candidates());
      NDArray reading = heads == null ? null : leaf(scope, heads.reading());
      return new Leaves(
          scores,
          candidates,
          reading,
          new Outputs(
              scores,
              outputs.legal(),
              outputs.actions(),
              outputs.discards(),
              heads == null ? null : new Outputs.Auxiliary(candidates, reading)));
    }

    private static NDArray leaf(NDManager scope, NDArray array) {
      if (array == null) {
        return null;
      }
      NDArray leaf = scope.create(array.toFloatArray(), array.getShape());
      leaf.setRequiresGradient(true);
      return leaf;
    }
  }

  /**
   * 試験の 1 つの微小バッチ: 網の出力（s と補助の頭の出力は与えた値、行の手の数・打牌の印は塊の索引から）と教師。
   *
   * @param outputs 網の出力
   * @param labels 行の順の教師
   * @param auxiliary 補助の頭の教師
   */
  private record Part(Outputs outputs, List<ValueLabel> labels, AuxiliaryLabels auxiliary) {

    LossInputs inputs(NDManager manager, AuxiliaryLoss.Denominators denominators) {
      return LossInputs.of(manager, outputs, labels, auxiliary, denominators);
    }

    AuxiliaryLoss.Terms terms(PrismLoss loss, NDManager manager, LossInputs inputs) {
      return loss.auxiliary().terms(manager, outputs.auxiliary(), inputs.auxiliary(), GAIN);
    }
  }

  /**
   * 試験の元の塊: 行の形・教師と、行・候補・読みの行ごとに決めた網の出力の値。部分（{@link #part}）は同じ値を切り出すので、元の塊と部分で
   * 行・候補・読みの行の出力が同じ。
   *
   * @param rows 行の形
   * @param batch 塊
   * @param labels 行の順の教師
   * @param auxiliary 補助の頭の教師
   * @param scores 行ごとの s（行の手の数）
   * @param candidates 候補ごとの補助の頭の出力（{@link CandidateTarget#WIDTH}）
   * @param readings 読みの行ごとの補助の頭の出力（3 × {@link ReadingTarget#WIDTH}）
   */
  private record Piece(
      List<Batches.Row> rows,
      Batch batch,
      List<ValueLabel> labels,
      AuxiliaryLabels auxiliary,
      float[][] scores,
      float[][] candidates,
      float[][] readings) {

    /** 行 cut より前を、自分の和了・放銃も聴牌の相手・放銃の牌もない行にした元の塊（Y は正、出力は乱数）。 */
    static Piece of(List<Batches.Row> rows, int cut, Random random) {
      Batch batch = Batches.build(rows, 5);
      AuxiliaryLabels auxiliary = quietBefore(Batches.auxiliary(batch, 13), batch, cut);
      float[][] scores = new float[rows.size()][];
      for (int r = 0; r < scores.length; r++) {
        scores[r] = gaussians(random, rows.get(r).moves().size());
      }
      float[][] candidates = new float[auxiliary.candidateActions().length][];
      for (int c = 0; c < candidates.length; c++) {
        candidates[c] = gaussians(random, CandidateTarget.WIDTH);
      }
      float[][] readings = new float[auxiliary.readingRows().length][];
      for (int k = 0; k < readings.length; k++) {
        readings[k] = gaussians(random, READING_WIDTH);
      }
      return new Piece(
          rows, batch, PrismLossTest.labels(batch, 11, 0.5f), auxiliary, scores, candidates, readings);
    }

    /** 行 [from, to) の微小バッチ（手と行の番号は部分の中の番号。行の手の数と打牌の印は部分の塊の索引から）。 */
    Part part(NDManager manager, int from, int to) {
      int[] picked = candidatesOf(auxiliary, batch, from, to);
      int[] read = readingsOf(auxiliary, from, to);
      List<ValueLabel> partLabels = labels.subList(from, to);
      AuxiliaryLabels partAuxiliary = select(auxiliary, picked, read, batch.firstAction(from), from);
      BatchIndex index =
          BatchIndex.send(
              Batches.build(rows.subList(from, to), 5).inputs(manager),
              Pass.training(Precision.FLOAT32, ValueLabel.measured(partLabels), partAuxiliary));
      int count = partLabels.size();
      int capacity = index.actionCapacity;
      float[] values = new float[count * capacity];
      for (int r = 0; r < count; r++) {
        System.arraycopy(scores[from + r], 0, values, r * capacity, scores[from + r].length);
      }
      Outputs outputs =
          new Outputs(
              manager.create(values, new Shape(count, capacity)),
              index.legal.array(),
              index.rowActions,
              index.discardValueSlots,
              new Outputs.Auxiliary(
                  picked.length == 0
                      ? null
                      : manager.create(
                          gather(candidates, picked), new Shape(picked.length, CandidateTarget.WIDTH)),
                  read.length == 0
                      ? null
                      : manager.create(
                          gather(readings, read),
                          new Shape(read.length, AuxiliaryLabels.OPPONENTS, ReadingTarget.WIDTH))));
      return new Part(outputs, partLabels, partAuxiliary);
    }
  }

  /** 行 [from, to) の手の候補（教師の候補の並びの中の番号）。 */
  private static int[] candidatesOf(AuxiliaryLabels labels, Batch batch, int from, int to) {
    int first = batch.firstAction(from);
    int end = batch.firstAction(to);
    int[] actions = labels.candidateActions();
    return IntStream.range(0, actions.length)
        .filter(c -> actions[c] >= first && actions[c] < end)
        .toArray();
  }

  /** 行 [from, to) の読み（教師の読みの並びの中の番号）。 */
  private static int[] readingsOf(AuxiliaryLabels labels, int from, int to) {
    int[] rows = labels.readingRows();
    return IntStream.range(0, rows.length).filter(k -> rows[k] >= from && rows[k] < to).toArray();
  }

  /** 選んだ候補と読みの教師（手と行の番号は部分の中の番号）。 */
  private static AuxiliaryLabels select(
      AuxiliaryLabels labels, int[] candidates, int[] readings, int firstAction, int firstRow) {
    int width = CandidateTarget.WIDTH;
    int[] actions = new int[candidates.length];
    int[] outcomes = new int[candidates.length];
    float[] values = new float[candidates.length * width];
    for (int i = 0; i < candidates.length; i++) {
      int c = candidates[i];
      actions[i] = labels.candidateActions()[c] - firstAction;
      outcomes[i] = labels.outcomes()[c];
      System.arraycopy(labels.candidateValues(), c * width, values, i * width, width);
    }
    int[] rows = new int[readings.length];
    float[] reading = new float[readings.length * READING_WIDTH];
    for (int i = 0; i < readings.length; i++) {
      int k = readings[i];
      rows[i] = labels.readingRows()[k] - firstRow;
      System.arraycopy(labels.readingValues(), k * READING_WIDTH, reading, i * READING_WIDTH, READING_WIDTH);
    }
    return new AuxiliaryLabels(actions, outcomes, values, rows, reading);
  }

  /** 行 cut より前の候補を自分の和了も放銃もない終わり方にし、その行の相手を聴牌でなく放銃の牌もない形にする。 */
  private static AuxiliaryLabels quietBefore(AuxiliaryLabels labels, Batch batch, int cut) {
    int width = CandidateTarget.WIDTH;
    int end = batch.firstAction(cut);
    int[] outcomes = labels.outcomes().clone();
    float[] values = labels.candidateValues().clone();
    for (int c = 0; c < outcomes.length; c++) {
      if (labels.candidateActions()[c] < end) {
        Outcome outcome = QUIET[c % QUIET.length];
        outcomes[c] = outcome.ordinal();
        int base = c * width + CandidateTarget.OUTCOME.offset();
        Arrays.fill(values, base, base + CandidateTarget.OUTCOME.width(), 0f);
        values[base + outcome.ordinal()] = 1f;
      }
    }
    float[] reading = labels.readingValues().clone();
    for (int k = 0; k < labels.readingRows().length; k++) {
      if (labels.readingRows()[k] < cut) {
        for (int opponent = 0; opponent < AuxiliaryLabels.OPPONENTS; opponent++) {
          int base = (k * AuxiliaryLabels.OPPONENTS + opponent) * ReadingTarget.WIDTH;
          reading[base + ReadingTarget.TENPAI.offset()] = 0f;
          int danger = base + ReadingTarget.DANGER.offset();
          Arrays.fill(reading, danger, danger + ReadingTarget.DANGER.width(), 0f);
        }
      }
    }
    return new AuxiliaryLabels(labels.candidateActions(), outcomes, values, labels.readingRows(), reading);
  }

  /** 選んだ行をつなぐ。 */
  private static float[] gather(float[][] rows, int[] picks) {
    int width = rows.length == 0 ? 0 : rows[0].length;
    float[] values = new float[picks.length * width];
    for (int i = 0; i < picks.length; i++) {
      System.arraycopy(rows[picks[i]], 0, values, i * width, width);
    }
    return values;
  }

  private static float[] gaussians(Random random, int count) {
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      values[i] = (float) random.nextGaussian();
    }
    return values;
  }
}
