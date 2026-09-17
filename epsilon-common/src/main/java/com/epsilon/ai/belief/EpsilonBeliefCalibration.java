package com.epsilon.ai.belief;

import com.epsilon.core.Tile;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * 他家状態を予測するBeliefモデルの精度と確率の較正を評価する。
 *
 * <ul>
 *   <li>手牌1枚あたりの対数尤度: 牌の残数とロジットから牌種ごとの確率を求め、実際の他家手牌の各牌について対数確率を平均する。全ロジットを0とした比較基準との差を求める。
 *   <li>テンパイのBrierスコア: テンパイ確率と教師のテンパイ判定（シャンテン数0以下）の二乗誤差を求める。小さいほどよい。
 *   <li>待ち牌の上位候補の的中率: テンパイ教師が1で、待ち牌マスクが空でない他家について、予測上位の牌種に実際の待ちが含まれる割合を求める。
 * </ul>
 */
public final class EpsilonBeliefCalibration {

  private static final int EVALUATE_CHUNK = 512;

  private EpsilonBeliefCalibration() {}

  /**
   * 教師サンプルを使い、他家状態の予測精度と確率の較正指標を計算する。
   *
   * @param samples 完全情報教師データを持つ評価サンプル群
   * @param evaluator 評価する推論器。{@code null}なら全ロジットを0とした比較基準を測る
   * @return 手牌の尤度、テンパイのBrierスコア（二乗誤差）、待ち牌の予測上位k候補の正解率を集約した報告
   */
  public static <I> Report evaluate(
      List<EpsilonBeliefSample<I>> samples, EpsilonBeliefEvaluator<I> evaluator) {
    Accumulator acc = new Accumulator();
    for (int start = 0; start < samples.size(); start += EVALUATE_CHUNK) {
      int end = Math.min(start + EVALUATE_CHUNK, samples.size());
      List<EpsilonBeliefSample<I>> chunk = samples.subList(start, end);
      List<EpsilonBeliefPrior> priors = priorsFor(chunk, evaluator);
      for (int i = 0; i < chunk.size(); i++) {
        accumulate(acc, chunk.get(i).target(), priors.get(i));
      }
    }
    return acc.toReport(samples.size());
  }

  private static <I> List<EpsilonBeliefPrior> priorsFor(
      List<EpsilonBeliefSample<I>> chunk, EpsilonBeliefEvaluator<I> evaluator) {
    if (evaluator == null) {
      List<EpsilonBeliefPrior> priors = new ArrayList<>(chunk.size());
      for (int i = 0; i < chunk.size(); i++) {
        priors.add(EpsilonBeliefPrior.uniform());
      }
      return priors;
    }
    List<I> inputs = new ArrayList<>(chunk.size());
    for (EpsilonBeliefSample<I> sample : chunk) {
      inputs.add(sample.input());
    }
    return evaluator.evaluateBatch(inputs);
  }

  private static void accumulate(
      Accumulator acc, EpsilonBeliefTarget target, EpsilonBeliefPrior prior) {
    for (int opponent = 0; opponent < EpsilonBeliefLayout.OPPONENT_COUNT; opponent++) {
      int base = opponent * Tile.NUM_TILE_TYPES;
      accumulateHandLogLikelihood(acc, target, prior, base);
      accumulateTenpaiBrier(acc, target, prior, opponent);
      accumulateWaitTopK(acc, target, prior, opponent, base);
    }
  }

  /** 牌の残数とロジットの指数関数の積を正規化して牌種ごとの確率を求め、実手牌の各牌について対数確率を合計する。後で手牌枚数により平均する。 */
  private static void accumulateHandLogLikelihood(
      Accumulator acc, EpsilonBeliefTarget target, EpsilonBeliefPrior prior, int base) {
    double modelTotal = 0.0;
    double uniformTotal = 0.0;
    for (int t = 0; t < Tile.NUM_TILE_TYPES; t++) {
      double hidden = target.hiddenTileCounts()[t];
      if (hidden <= 0.0) {
        continue;
      }
      modelTotal += hidden * boundedExp(prior.opponentHandLogits()[base + t]);
      uniformTotal += hidden;
    }
    if (!(modelTotal > 0.0) || !(uniformTotal > 0.0)) {
      return;
    }
    for (int t = 0; t < Tile.NUM_TILE_TYPES; t++) {
      float count = target.opponentHandCounts()[base + t];
      if (count <= 0.0f) {
        continue;
      }
      double hidden = Math.max(1.0e-9, target.hiddenTileCounts()[t]);
      double modelProb = hidden * boundedExp(prior.opponentHandLogits()[base + t]) / modelTotal;
      double uniformProb = hidden / uniformTotal;
      acc.handLogLik += count * Math.log(Math.max(1.0e-12, modelProb));
      acc.uniformHandLogLik += count * Math.log(Math.max(1.0e-12, uniformProb));
      acc.handTiles += count;
    }
  }

  private static void accumulateTenpaiBrier(
      Accumulator acc, EpsilonBeliefTarget target, EpsilonBeliefPrior prior, int opponent) {
    double probability = sigmoid(prior.opponentScalar(opponent, EpsilonBeliefLayout.SCALAR_TENPAI));
    double actual = target.opponentTenpai()[opponent] > 0.5f ? 1.0 : 0.0;
    acc.tenpaiBrier += (probability - actual) * (probability - actual);
    acc.opponents++;
  }

  private static void accumulateWaitTopK(
      Accumulator acc,
      EpsilonBeliefTarget target,
      EpsilonBeliefPrior prior,
      int opponent,
      int base) {
    if (target.opponentTenpai()[opponent] <= 0.5f || waitCount(target, base) == 0) {
      return;
    }
    int[] topTiles = topTilesByLogit(prior.opponentWaitLogits(), base, 3);
    acc.tenpaiOpponents++;
    if (target.opponentWaitMask()[base + topTiles[0]] > 0.5f) {
      acc.waitTop1Hits++;
    }
    for (int rank = 0; rank < topTiles.length; rank++) {
      if (target.opponentWaitMask()[base + topTiles[rank]] > 0.5f) {
        acc.waitTop3Hits++;
        break;
      }
    }
  }

  private static int waitCount(EpsilonBeliefTarget target, int base) {
    int count = 0;
    for (int t = 0; t < Tile.NUM_TILE_TYPES; t++) {
      if (target.opponentWaitMask()[base + t] > 0.5f) {
        count++;
      }
    }
    return count;
  }

  private static int[] topTilesByLogit(float[] logits, int base, int k) {
    int[] top = new int[k];
    boolean[] used = new boolean[Tile.NUM_TILE_TYPES];
    for (int rank = 0; rank < k; rank++) {
      int best = -1;
      for (int t = 0; t < Tile.NUM_TILE_TYPES; t++) {
        if (used[t]) {
          continue;
        }
        if (best < 0 || logits[base + t] > logits[base + best]) {
          best = t;
        }
      }
      top[rank] = best;
      used[best] = true;
    }
    return top;
  }

  private static double boundedExp(float logit) {
    return Math.exp(Math.max(-30.0, Math.min(30.0, logit)));
  }

  private static double sigmoid(float logit) {
    double x = Math.max(-30.0, Math.min(30.0, logit));
    return 1.0 / (1.0 + Math.exp(-x));
  }

  private static final class Accumulator {
    double handLogLik;
    double uniformHandLogLik;
    double handTiles;
    double tenpaiBrier;
    int opponents;
    int tenpaiOpponents;
    int waitTop1Hits;
    int waitTop3Hits;

    Report toReport(int samples) {
      return new Report(
          samples,
          opponents,
          tenpaiOpponents,
          (float) (handLogLik / Math.max(1.0, handTiles)),
          (float) (uniformHandLogLik / Math.max(1.0, handTiles)),
          (float) (tenpaiBrier / Math.max(1, opponents)),
          (float) ((double) waitTop1Hits / Math.max(1, tenpaiOpponents)),
          (float) ((double) waitTop3Hits / Math.max(1, tenpaiOpponents)));
    }
  }

  /**
   * Belief 検証全体の較正指標。
   *
   * @param samples 評価サンプル数
   * @param opponents 評価した他家数(サンプル × 3)
   * @param tenpaiOpponents テンパイ教師が1かつ待ち牌マスクが空でない、待ち予測の評価対象となる他家数
   * @param handLogLikelihoodPerTile 手牌1枚あたりの平均対数尤度(高いほど良い)
   * @param uniformHandLogLikelihoodPerTile 全ロジットを0とした比較基準の対数尤度
   * @param tenpaiBrier テンパイ予測の Brierスコア(低いほど良い)
   * @param waitTop1HitRate 評価対象の他家の待ちが予測上位1牌種に含まれる割合
   * @param waitTop3HitRate 評価対象の他家の待ちが予測上位3牌種に含まれる割合
   */
  public record Report(
      int samples,
      int opponents,
      int tenpaiOpponents,
      float handLogLikelihoodPerTile,
      float uniformHandLogLikelihoodPerTile,
      float tenpaiBrier,
      float waitTop1HitRate,
      float waitTop3HitRate) {

    /**
     * 全ロジットを0とした比較基準に対する手牌対数尤度の改善幅を返す。
     *
     * @return 正なら他家状態の推定が比較基準を上回る1枚あたりの改善幅
     */
    public float handLogLikelihoodGain() {
      return handLogLikelihoodPerTile - uniformHandLogLikelihoodPerTile;
    }

    /**
     * 全較正指標を一行の機械可読文字列へ整形する。
     *
     * @return ロケール非依存の要約
     */
    public String summary() {
      return String.format(
          Locale.ROOT,
          "samples=%d opponents=%d tenpaiOpponents=%d handLogLik/tile=%.5f uniformLogLik/tile=%.5f "
              + "gain=%.5f tenpaiBrier=%.5f waitTop1=%.4f waitTop3=%.4f",
          samples,
          opponents,
          tenpaiOpponents,
          handLogLikelihoodPerTile,
          uniformHandLogLikelihoodPerTile,
          handLogLikelihoodGain(),
          tenpaiBrier,
          waitTop1HitRate,
          waitTop3HitRate);
    }
  }
}
