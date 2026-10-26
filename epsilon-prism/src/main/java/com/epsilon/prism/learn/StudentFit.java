package com.epsilon.prism.learn;

import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.prism.model.Example;
import com.epsilon.prism.model.LabelExample;
import com.epsilon.prism.model.LearningRateSchedule;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.PrismModel;
import com.epsilon.prism.rows.PrismRow;
import com.epsilon.prism.rows.RowSource;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Function;

/**
 * 生徒の当てはめ（論文 3.4 M3、3.6 の「3 生徒」）。生徒は呼び出し側が π_t の重みから学習用に読んだ網（温め込み）で、組の TRAIN のラベル行
 * （MAIN・ROLLIN、|C| ≥ 2）に差分整合の損失（{@link PrismLoss}、κ = settings.kappa）を steps 歩当てる。
 *
 * <p>行は {@link BatchAssembler#shuffle}（種 orderSeed）で並べ替え、batchSize 行ずつ順に取る（足りなければ巡回）。展開・教師・結合（{@link
 * PrismModel.Fit#pack}）は先読みのスレッドで最大 3 バッチ先まで作り、学習のスレッドは GPU だけを回す。教師は行ごとに
 * {@link #example}。
 */
public final class StudentFit {
  /** 先に展開しておくバッチの数。 */
  static final int PREFETCH = 3;

  private StudentFit() {}

  /**
   * 生徒を当てる。重みは student に残る（保存は呼び出し側）。
   *
   * @param name 記録の名前（"A" か "B"）
   * @param rows 使う行（組の TRAIN のラベル行で測った手が 2 つ以上。呼び出し側が索引で選ぶ）
   * @param orderSeed 行の並べ替えの種（生徒ごとに違う）
   */
  public static FitResult fit(
      String name,
      RowSource source,
      BatchAssembler assembler,
      PrismModel student,
      int[] rows,
      FitSettings settings,
      long orderSeed) {
    long started = System.nanoTime();
    if (rows.length == 0 || settings.steps() <= 0) return FitResult.empty(name, rows.length);
    int[] order = BatchAssembler.shuffle(rows, orderSeed);
    FitResult trained;
    try (PrismModel.Fit<LabelExample> fit =
        student.fit(
            schedule(settings), settings.gradientClip(), new PrismLoss((float) settings.kappa()))) {
      trained =
          train(
              name,
              fit,
              source,
              assembler,
              order,
              settings.steps(),
              settings.batchSize(),
              decoded -> examples(decoded, settings));
    }
    // 秒数は学習器を閉じる（更新の完了を待つ）までを含める
    return withSeconds(trained, started);
  }

  /** ラベル行の教師: values はそのまま、KL の基準は {@link FitSettings#reference}、λ は行の類の λ_c。行の合法手の数が payload と違えば例外。 */
  public static LabelExample example(PrismRow row, DecisionHostBatch host, FitSettings settings) {
    if (host.legalActionCount(0) != row.actions())
      throw new IllegalStateException(
          "Row of game "
              + row.gameId()
              + " has "
              + row.actions()
              + " actions but its payload has "
              + host.legalActionCount(0));
    return new LabelExample(
        host, row.values(), settings.reference(row.prior()), settings.lambda(row.decisionClass()));
  }

  /** 当てはめの学習率（始めの {@link FitSettings#WARMUP} で lrPeak まで上げ、cosine で lrFloor へ）。 */
  static LearningRateSchedule schedule(FitSettings settings) {
    return LearningRateSchedule.of(
        settings.lrPeak(), settings.lrFloor(), FitSettings.WARMUP, settings.steps());
  }

  /** 記録の秒数を started からの経過に替える。 */
  static FitResult withSeconds(FitResult result, long started) {
    return new FitResult(
        result.name(),
        result.rows(),
        result.steps(),
        result.clippedSteps(),
        result.loss(),
        result.firstLoss(),
        result.lastLoss(),
        (System.nanoTime() - started) / 1e9,
        result.stageSeconds());
  }

  private static List<LabelExample> examples(BatchAssembler.Decoded decoded, FitSettings settings) {
    List<LabelExample> examples = new ArrayList<>(decoded.rows().length);
    for (int j = 0; j < decoded.rows().length; j++)
      examples.add(example(decoded.rows()[j], decoded.hosts()[j], settings));
    return examples;
  }

  /**
   * order の行を batchSize ずつ（足りなければ巡回して）steps 歩当てる共通の輪（{@link DistillFit} も使う）。examples は展開した行から教師を作る
   * （先読みのスレッドで呼ぶ）。
   *
   * @return 名前 name の記録（rows は order の長さ）
   */
  static <E extends Example> FitResult train(
      String name,
      PrismModel.Fit<E> fit,
      RowSource source,
      BatchAssembler assembler,
      int[] order,
      int steps,
      int batchSize,
      Function<BatchAssembler.Decoded, List<E>> examples) {
    long started = System.nanoTime();
    int n = order.length;
    if (n == 0 || steps <= 0) return FitResult.empty(name, n);
    if (batchSize < 1) throw new IllegalArgumentException("batchSize must be positive: " + batchSize);
    double sum = 0, first = Double.NaN, last = Double.NaN;
    int measured = 0, from = 0, queued = 0;
    // 展開・教師・結合（レーンへの割り当てと微小バッチの連結）までを先読みのスレッドでやり、学習のスレッドは GPU だけを回す
    var pending = new ArrayDeque<CompletableFuture<PrismModel.Packed<E>>>(PREFETCH);
    for (int step = 0; step < steps; step++) {
      while (pending.size() < PREFETCH && queued < steps) {
        pending.add(
            assembler
                .decode(source, BatchAssembler.batch(order, from, batchSize))
                .thenApply(decoded -> fit.pack(examples.apply(decoded))));
        from = from + batchSize < n ? from + batchSize : 0;
        queued++;
      }
      double loss = fit.train(await(pending.remove()));
      if (!Double.isNaN(loss)) {
        sum += loss;
        measured++;
        if (Double.isNaN(first)) first = loss;
        last = loss;
      }
    }
    return new FitResult(
        name,
        n,
        steps,
        fit.clippedSteps(),
        measured > 0 ? sum / measured : Double.NaN,
        first,
        last,
        (System.nanoTime() - started) / 1e9,
        fit.stageSeconds());
  }

  /** 先読みの完了を待つ。先読みのスレッドの例外は包みを外して投げる。 */
  static <T> T await(CompletableFuture<T> future) {
    try {
      return future.join();
    } catch (CompletionException failure) {
      if (failure.getCause() instanceof RuntimeException cause) throw cause;
      if (failure.getCause() instanceof Error cause) throw cause;
      throw failure;
    }
  }
}
