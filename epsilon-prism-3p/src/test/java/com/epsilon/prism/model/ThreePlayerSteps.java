package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.hip.TrainingKernels;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.RowColumn;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * 三人打ちの学習のテスト用のヘルパー。エンジンで打った三人打ちの判断のバッチに乱数のラベルを付け、順伝播と逆伝播を 1 回ずつ実行して、
 * 損失・s・全てのパラメーターの勾配を取る。ラベルはバッチの形と種だけで決まるので、同じバッチなら経路を変えても同じラベルになる。
 */
final class ThreePlayerSteps {

  private static final TargetSchema HEADS = TargetSchema.of(Variant.TENHOU_3P);

  /** 価値のラベルの種。 */
  private static final long VALUE_SEED = 11;

  /** 補助のヘッドのラベルの種。 */
  private static final long AUXILIARY_SEED = 13;

  private ThreePlayerSteps() {}

  /**
   * 1 回の順伝播と逆伝播の結果。
   *
   * @param loss 損失
   * @param scores s（[行, 手の容量] を行優先で平らにした形）
   * @param gradients パラメーターごとの勾配の写し
   * @param millis 順伝播を始めてから勾配が揃うまでの時間（ms）
   */
  record Outcome(float loss, float[] scores, Map<String, float[]> gradients, double millis) {}

  /**
   * 順伝播と逆伝播を 1 回ずつ実行し、勾配の写しを取ってから勾配を 0 に戻す。
   *
   * @param network ネットワーク
   * @param manager パラメーターを持つマネージャー（入力は、この下に作る作業用のマネージャーに置く）
   * @param batch 確定した三人打ちのバッチ
   * @param kernels Attention と正規化の演算（Attention の記録を使う）。null なら djl の演算の経路
   * @param precision 計算の精度
   * @param intermediates 区切りの途中の配列の扱い
   * @return 結果
   */
  static Outcome run(
      PrismNetwork network,
      NDManager manager,
      HostBatch batch,
      TrainingKernels kernels,
      Precision precision,
      Context.Intermediates intermediates) {
    PrismLoss loss = new PrismLoss(AuxiliaryLoss.standard());
    long started = System.nanoTime();
    float value;
    float[] scores;
    try (NDManager step = manager.newSubManager();
        GradientCollector collector = manager.getEngine().newGradientCollector()) {
      NetworkInputs inputs = NetworkInputs.of(step, PinnedUploads.on(step), batch);
      List<ValueLabel> labels = values(inputs);
      AuxiliaryLabels auxiliary = auxiliary(inputs);
      AttentionTape tape = kernels == null ? null : new AttentionTape(kernels);
      Outputs outputs =
          network.forward(
              new ParameterStore(manager, false),
              inputs,
              Pass.training(precision, ValueLabel.measured(labels), auxiliary, tape),
              intermediates);
      NDArray total = loss.total(step, outputs, labels, auxiliary, 1f, labels.size(), 1f);
      if (tape == null) {
        collector.backward(total);
      } else {
        tape.backward(collector, total);
      }
      value = total.getFloat();
      scores = outputs.scores().toType(DataType.FLOAT32, false).toFloatArray();
    }
    Map<String, float[]> gradients = new LinkedHashMap<>();
    for (Pair<String, Parameter> pair : network.getParameters()) {
      try (NDArray gradient = pair.getValue().getArray().getGradient()) {
        gradients.put(pair.getKey(), gradient.toFloatArray());
        gradient.fillI(0);
      }
    }
    return new Outcome(value, scores, gradients, (System.nanoTime() - started) / 1e6);
  }

  /** 全ての手を測ったことにした、乱数の Q のラベル（λ = 0.1）。 */
  private static List<ValueLabel> values(NetworkInputs inputs) {
    Random random = new Random(VALUE_SEED);
    List<ValueLabel> labels = new ArrayList<>();
    for (int r = 0; r < inputs.rows(); r++) {
      float[] values = new float[inputs.row(r, RowColumn.ACTIONS)];
      for (int m = 0; m < values.length; m++) {
        values[m] = random.nextFloat() - 0.5f;
      }
      labels.add(new ValueLabel(values, 0.1f, 1f));
    }
    return labels;
  }

  /** 候補のラベルを持つ全ての手と全ての行に付ける、補助のヘッドのラベル（幅は三人打ちのヘッドの形から決める）。 */
  private static AuxiliaryLabels auxiliary(NetworkInputs inputs) {
    Random random = new Random(AUXILIARY_SEED);
    List<Integer> candidates = new ArrayList<>();
    for (int a = 0; a < inputs.actions(); a++) {
      if (TargetSchema.hasCandidateTargets(inputs.format().roles().get(inputs.action(a, ActionColumn.ROLE)))) {
        candidates.add(a);
      }
    }
    int n = candidates.size();
    int[] actions = new int[n];
    int[] outcomes = new int[n];
    float[] values = new float[n * HEADS.candidateWidth()];
    for (int c = 0; c < n; c++) {
      actions[c] = candidates.get(c);
      outcomes[c] = c % TargetSchema.Outcome.values().length;
      int base = c * HEADS.candidateWidth();
      values[base + HEADS.offset(CandidateTarget.OUTCOME) + outcomes[c]] = 1f;
      values[base + HEADS.offset(CandidateTarget.TENPAI)] = random.nextInt(2);
      for (int y = 0; y < HEADS.width(CandidateTarget.YAKU); y++) {
        values[base + HEADS.offset(CandidateTarget.YAKU) + y] = random.nextFloat() < 0.1f ? 1f : 0f;
      }
      values[base + HEADS.offset(CandidateTarget.WIN_POINTS)] = 1000 + random.nextInt(11000);
      values[base + HEADS.offset(CandidateTarget.DEAL_IN_POINTS)] = 1000 + random.nextInt(11000);
      values[base + HEADS.offset(CandidateTarget.YAKU_TENPAI)] = random.nextInt(2);
      values[base + HEADS.offset(CandidateTarget.POINT_DELTA)] = random.nextInt(16001) - 8000;
      values[base + HEADS.offset(CandidateTarget.WINNER) + random.nextInt(HEADS.width(CandidateTarget.WINNER))] = 1f;
    }
    int rows = inputs.rows();
    int[] readingRows = new int[rows];
    float[] reading = new float[rows * HEADS.opponents() * HEADS.readingWidth()];
    for (int r = 0; r < rows; r++) {
      readingRows[r] = r;
      for (int j = 0; j < HEADS.opponents(); j++) {
        int base = (r * HEADS.opponents() + j) * HEADS.readingWidth();
        reading[base + HEADS.offset(ReadingTarget.TENPAI)] = random.nextFloat() < 0.5f ? 1f : 0f;
        reading[base + HEADS.offset(ReadingTarget.SHANTEN) + random.nextInt(TargetSchema.SHANTEN_CLASSES)] = 1f;
        for (int t = 0; t < HEADS.width(ReadingTarget.DANGER); t++) {
          reading[base + HEADS.offset(ReadingTarget.DANGER) + t] = random.nextFloat() < 0.2f ? 1f : 0f;
          reading[base + HEADS.offset(ReadingTarget.WAITS) + t] = random.nextFloat() < 0.2f ? 1f : 0f;
          reading[base + HEADS.offset(ReadingTarget.DANGER_POINTS) + t] = 1000 + random.nextInt(11000);
        }
      }
    }
    return new AuxiliaryLabels(HEADS, actions, outcomes, values, readingRows, reading);
  }

  /** 勾配がちょうど 0 のパラメーター。 */
  static List<String> silent(Outcome outcome) {
    List<String> names = new ArrayList<>();
    for (Map.Entry<String, float[]> entry : outcome.gradients().entrySet()) {
      if (maxAbs(entry.getValue()) == 0) {
        names.add(entry.getKey());
      }
    }
    return names;
  }

  /** 全ての勾配を合わせた、相対の L2 の差。 */
  static double gradients(Outcome reference, Outcome other) {
    double error = 0;
    double norm = 0;
    for (Map.Entry<String, float[]> entry : reference.gradients().entrySet()) {
      norm += squared(entry.getValue());
      error += difference(entry.getValue(), other.gradients().get(entry.getKey()));
    }
    return Math.sqrt(error / Math.max(norm, 1e-30));
  }

  /** 相対の L2 の差。 */
  static double relative(float[] expected, float[] actual) {
    return Math.sqrt(difference(expected, actual) / Math.max(squared(expected), 1e-30));
  }

  static float maxAbs(float[] values) {
    float max = 0;
    for (float value : values) {
      max = Math.max(max, Math.abs(value));
    }
    return max;
  }

  static double squared(float[] values) {
    double sum = 0;
    for (float value : values) {
      sum += (double) value * value;
    }
    return sum;
  }

  static double difference(float[] expected, float[] actual) {
    double sum = 0;
    for (int i = 0; i < expected.length; i++) {
      double d = (double) expected[i] - actual[i];
      sum += d * d;
    }
    return sum;
  }
}
