package com.epsilon.grp.train;

import java.util.SplittableRandom;
import java.util.function.BooleanSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 学習 1 回の進め方。GRP、局の結果、prism の Sinkhorn の 3 つのネットワークの学習で共通に使う。学習器が用意するのは、順伝播と
 * 損失（Graph）と、行の詰め方（Batch）だけ。エポックごとに学習用の行を並べ替え、設定の行数ずつバッチに詰めて 1 ステップ進める。
 * 行の重みは、和が行数になるように w × 行数 / Σ w に直す。エポックの半ばと終わりに早期終了用の検証の行で Σ w·NLL / Σ w を測り、
 * 最良の重みを残す。始めの重みも候補に入る。この損失が良くならない回が続き、その回数が決めた数に達したら学習率を半分にし、別に
 * 決めた数に達したら止める（どちらの数も GrpTrainingRule の設定）。停止のフラグは毎ステップの前に見て、立っていれば STOPPED で
 * 止める。
 */
public final class TrainingLoop {

  private static final Logger LOG = LoggerFactory.getLogger(TrainingLoop.class);

  /** 学習に使う行の並び（GrpRows と OutcomeRows）。 */
  public interface Rows {

    int size();

    double totalWeight();
  }

  /** 行をネットワークの入力に詰めるバッチ（学習器ごとに作る）。配列は作るときに 1 度だけ確保し、{@code clear} で使い回す。 */
  public interface Batch<R extends Rows> {

    /** 1 度に詰められる行の数。 */
    int capacity();

    void clear();

    /** 行を 1 つ、その重みに倍率を掛けて足す。 */
    void add(R rows, int index, float scale);
  }

  /** バッチで学習するネットワーク。学習器ごとに DJL で組む。 */
  public interface Graph<B> {

    /** バッチの Σ w·NLL / 行数 が下がる向きに、重みを 1 ステップ更新する（{@code AdamParameters.step}）。 */
    TrainingStep step(B batch, double learningRate, double hiddenDecay, double clipNorm);

    /** バッチの Σ w·NLL を返す。勾配は作らない。 */
    double weightedNll(B batch);

    /** 今の重みを、テンソルの宣言の順に並べて返す。 */
    float[] weights();
  }

  /**
   * 学習の結果。
   *
   * @param weights 最良の重み。STOPPED で終わったら null
   * @param halfEpochs 検証の行で損失を測った回数（半エポックの数）
   * @param stopNll 最良の重みでの、早期終了用の検証の行の NLL
   * @param clippedSteps 勾配を縮めたステップの数
   * @param ending 学習の終わり方
   */
  public record Result(float[] weights, int halfEpochs, double stopNll, long clippedSteps, TrainingEnding ending) {}

  private TrainingLoop() {}

  /**
   * 学習用の行でネットワークを学習する。
   *
   * @param name ログの見出し
   * @param initial ネットワークの始めの重み。最良の重みの候補に入る
   * @param shuffler 学習用の行を並べ替える乱数
   * @param closing 停止のフラグ
   * @throws IllegalArgumentException 学習用の行か、早期終了用の検証の行が無いとき
   */
  public static <R extends Rows, B extends Batch<R>> Result run(
      String name,
      Graph<B> graph,
      B batch,
      R train,
      R stop,
      float[] initial,
      GrpTrainingRule rule,
      SplittableRandom shuffler,
      BooleanSupplier closing) {
    int rows = train.size();
    if (rows == 0 || stop.size() == 0) {
      throw new IllegalArgumentException(name + " training needs TRAIN and STOP rows: " + train + ", " + stop);
    }
    float scale = (float) (rows / train.totalWeight());
    double best = stopNll(graph, batch, stop);
    float[] bestWeights = initial;
    LOG.info("{} {} start: STOP NLL {} over {} rows", name, rule.init(), best, rows);
    int[] order = new int[rows];
    for (int row = 0; row < order.length; row++) {
      order[row] = row;
    }
    int steps = (rows + rule.batchRows() - 1) / rule.batchRows();
    int half = (steps + 1) / 2;
    double learningRate = rule.learningRate();
    int halfEpochs = 0;
    int stale = 0;
    long clipped = 0;
    double trainLoss = 0;
    int trainSteps = 0;
    training:
    for (int epoch = 0; epoch < rule.maxEpochs(); epoch++) {
      shuffle(order, shuffler);
      for (int step = 0; step < steps; step++) {
        if (closing.getAsBoolean()) {
          LOG.info("{} training stopped after {} half-epochs", name, halfEpochs);
          return new Result(null, halfEpochs, best, clipped, TrainingEnding.STOPPED);
        }
        int from = step * rule.batchRows();
        batch.clear();
        for (int index = from; index < Math.min(rows, from + rule.batchRows()); index++) {
          batch.add(train, order[index], scale);
        }
        TrainingStep result = graph.step(batch, learningRate, rule.hiddenDecay(), rule.clipNorm());
        clipped += result.clipped() ? 1 : 0;
        trainLoss += result.loss();
        trainSteps++;
        if (step + 1 != half && step + 1 != steps) {
          continue;
        }
        halfEpochs++;
        double nll = stopNll(graph, batch, stop);
        if (nll < best) {
          best = nll;
          bestWeights = graph.weights();
          stale = 0;
        } else {
          stale++;
          if (stale == rule.halvingPatience()) {
            learningRate /= 2;
          }
        }
        LOG.info("{} half-epoch {}: TRAIN loss {}, STOP NLL {} (best {}), learning rate {}, clipped steps {}",
            name, halfEpochs, trainLoss / trainSteps, nll, best, learningRate, clipped);
        trainLoss = 0;
        trainSteps = 0;
        if (stale >= rule.stopPatience()) {
          break training;
        }
      }
    }
    return new Result(bestWeights, halfEpochs, best, clipped, TrainingEnding.COMPLETED);
  }

  /** 早期終了用の検証の行で Σ w·NLL / Σ w を測る。行はバッチの容量ずつ詰める。 */
  private static <R extends Rows, B extends Batch<R>> double stopNll(Graph<B> graph, B batch, R stop) {
    double sum = 0;
    for (int from = 0; from < stop.size(); from += batch.capacity()) {
      batch.clear();
      for (int row = from; row < Math.min(stop.size(), from + batch.capacity()); row++) {
        batch.add(stop, row, 1f);
      }
      sum += graph.weightedNll(batch);
    }
    return sum / stop.totalWeight();
  }

  private static void shuffle(int[] order, SplittableRandom random) {
    for (int index = order.length - 1; index > 0; index--) {
      int other = random.nextInt(index + 1);
      int held = order[index];
      order[index] = order[other];
      order[other] = held;
    }
  }
}
