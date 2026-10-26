package com.epsilon.prism.learn;

import com.epsilon.prism.model.Distill;
import com.epsilon.prism.model.DistillExample;
import com.epsilon.prism.model.EncodedRows;
import com.epsilon.prism.model.PrismModel;
import com.epsilon.prism.rows.PrismRow;
import com.epsilon.prism.rows.RowSource;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 蒸留（論文 3.4・3.6 の「π_new ← mean(π^A, π^B) を 1 本へ CE で蒸留する」）。生徒は呼び出し側が π_t の重みから学習用に読んだ網で、両方の組の TRAIN
 * のラベル行（MAIN・ROLLIN、|C| に依らない）に {@link Distill} を steps 歩当てる。
 *
 * <p>教師の分布は行ごとに、全ての合法手の上で t = (softmax(s_A/λ_c) + softmax(s_B/λ_c))/2（{@link #target}）。s_A・s_B は凍結した生徒 A・B
 * （推論用に読んだ網）で、バッチごとに先読みのスレッドで採点する（{@link PrismModel#scores(java.util.List)}）。λ_c は行の類の温度。輪は {@link
 * StudentFit#train} を使う。
 */
public final class DistillFit {
  private DistillFit() {}

  /**
   * 蒸留する。重みは student に残る（保存は呼び出し側）。
   *
   * @param teacherA 生徒 A（推論用）
   * @param teacherB 生徒 B（推論用）
   * @param rows 使う行（両方の組の TRAIN のラベル行。呼び出し側が索引で選ぶ）
   * @param orderSeed 行の並べ替えの種
   * @return 名前 "distill" の記録
   */
  public static FitResult distill(
      RowSource source,
      BatchAssembler assembler,
      PrismModel student,
      PrismModel teacherA,
      PrismModel teacherB,
      int[] rows,
      FitSettings settings,
      long orderSeed) {
    long started = System.nanoTime();
    String name = "distill";
    if (rows.length == 0 || settings.steps() <= 0) return FitResult.empty(name, rows.length);
    int[] order = BatchAssembler.shuffle(rows, orderSeed);
    FitResult trained;
    try (PrismModel.Fit<DistillExample> fit =
        student.fit(StudentFit.schedule(settings), settings.gradientClip(), new Distill())) {
      trained =
          StudentFit.train(
              name,
              fit,
              source,
              assembler,
              order,
              settings.steps(),
              settings.batchSize(),
              decoded -> examples(decoded, teacherA, teacherB, settings));
    }
    return StudentFit.withSeconds(trained, started);
  }

  /** 展開した行の教師。入力は 1 回だけまとめ、A と B で採点する（先読みのスレッドで呼ぶ）。 */
  private static List<DistillExample> examples(
      BatchAssembler.Decoded decoded,
      PrismModel teacherA,
      PrismModel teacherB,
      FitSettings settings) {
    EncodedRows encoded = EncodedRows.of(Arrays.asList(decoded.hosts()));
    float[][] scoresA = teacherA.scores(encoded), scoresB = teacherB.scores(encoded);
    List<DistillExample> examples = new ArrayList<>(decoded.rows().length);
    for (int j = 0; j < decoded.rows().length; j++) {
      PrismRow row = decoded.rows()[j];
      if (scoresA[j].length != row.actions() || scoresB[j].length != row.actions())
        throw new IllegalStateException(
            "Row of game "
                + row.gameId()
                + " has "
                + row.actions()
                + " actions but the teachers scored "
                + scoresA[j].length
                + " and "
                + scoresB[j].length);
      float lambda = settings.lambda(row.decisionClass());
      examples.add(
          new DistillExample(decoded.hosts()[j], target(scoresA[j], scoresB[j], lambda), lambda));
    }
    return examples;
  }

  /**
   * 教師の分布 (softmax(a/λ) + softmax(b/λ))/2（FP64 で最大値を引いて計算し、FP32 で返す）。
   *
   * @param scoresA 合法手ごとの s_A
   * @param scoresB 合法手ごとの s_B
   */
  public static float[] target(float[] scoresA, float[] scoresB, float lambda) {
    if (scoresA.length == 0 || scoresA.length != scoresB.length)
      throw new IllegalArgumentException(
          "Teacher scores must be non-empty and of equal length: "
              + scoresA.length
              + " vs "
              + scoresB.length);
    if (!(lambda > 0) || !Float.isFinite(lambda))
      throw new IllegalArgumentException("lambda must be positive: " + lambda);
    double[] a = softmax(scoresA, lambda), b = softmax(scoresB, lambda);
    float[] target = new float[a.length];
    for (int i = 0; i < a.length; i++) target[i] = (float) (0.5 * (a[i] + b[i]));
    return target;
  }

  /** softmax(s/λ)（FP64、最大値を引く）。 */
  private static double[] softmax(float[] scores, float lambda) {
    double peak = Double.NEGATIVE_INFINITY;
    for (float score : scores) {
      if (!Float.isFinite(score)) throw new IllegalStateException("Nonfinite teacher score " + score);
      peak = Math.max(peak, score);
    }
    double[] p = new double[scores.length];
    double sum = 0;
    for (int i = 0; i < scores.length; i++) {
      p[i] = Math.exp((scores[i] - peak) / lambda);
      sum += p[i];
    }
    for (int i = 0; i < p.length; i++) p[i] /= sum;
    return p;
  }
}
