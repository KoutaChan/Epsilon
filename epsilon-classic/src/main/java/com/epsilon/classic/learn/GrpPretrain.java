package com.epsilon.classic.learn;

import com.epsilon.agent.learn.GrpTraining;
import com.epsilon.grp.GrpField;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.RankPermutations;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.GateCohort;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.GrpRows;
import com.epsilon.grp.train.GrpTrained;
import com.epsilon.grp.train.GrpTrainer;
import com.epsilon.grp.train.GrpTrainingRule;
import com.epsilon.grp.train.LogGames;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.mahjong.rules.Variant;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.function.BooleanSupplier;

/**
 * 牌譜の半荘で GRP をゼロから学習する（epsilon-grp の学習器を使う）。半荘は 1 つのコホート（{@link LogGames#COHORT}）にまとめる。
 * 早期終了用の検証の半荘は、半荘 ID のハッシュで決まる（全体の 1/16）。モデルの形は、隠れ層 128、期ごとの出力のヘッドなし、点数の
 * フィールドだけ。
 *
 * <p>比べる基準として、検証の行での NLL を 2 つ返す。1 つは一様分布の NLL（log S!）。もう 1 つは、学習用の行の順列の頻度だけで予測した
 * ときの NLL で、数が 0 の順列は 0.5 と数える（ゼロから学習するときの初めの値と同じ）。
 */
public final class GrpPretrain {

  /** 隠れ層の幅。 */
  public static final int HIDDEN = 128;

  public static final Set<GrpField> FIELDS = EnumSet.range(GrpField.ROUND, GrpField.GAP);

  private static final double ABSENT_COUNT = 0.5;

  /**
   * 学習の結果。
   *
   * @param uniformNll 検証の行での、一様分布の NLL
   * @param frequencyNll 検証の行での、学習用の行の順列の頻度で予測した NLL
   */
  public record Result(GrpTrained trained, int trainRows, int stopRows, double uniformNll, double frequencyNll) {}

  private GrpPretrain() {}

  /**
   * @param games 牌譜の半荘（{@link LogGames#of}）
   * @param training 学習の設定（ゼロから学習する設定を使う）
   * @param seed 並べ替えと重みの初期値に使う乱数の種
   * @param threads torch の演算スレッドの数
   * @param closing 停止のフラグ
   */
  public static Result train(
      Variant variant, List<GameRecord> games, GrpTraining training, long seed, int threads, BooleanSupplier closing) {
    GrpRows.Split split =
        GrpRows.builder(variant, RecencyWindow.uniform(LogGames.COHORT, 1), GateCohort.NEXT)
            .add(LogGames.COHORT, games)
            .build();
    GrpShape shape = new GrpShape(variant, FIELDS, HIDDEN, false);
    GrpTrainingRule rule =
        new GrpTrainingRule(
            GrpInit.SCRATCH,
            training.scratchRate(),
            0,
            training.scratchEpochs(),
            training.batchRows(),
            training.clipNorm(),
            training.halvingPatience(),
            training.stopPatience(),
            seed);
    GrpTrained trained = GrpTrainer.train(shape, null, split.train(), split.stop(), rule, threads, closing);
    int outputs = RankPermutations.of(variant).count();
    return new Result(
        trained, split.train().size(), split.stop().size(), Math.log(outputs), frequencyNll(split, outputs));
  }

  private static double frequencyNll(GrpRows.Split split, int outputs) {
    double[] counts = new double[outputs];
    for (int row = 0; row < split.train().size(); row++) {
      counts[split.train().permutation(row)]++;
    }
    double total = 0;
    for (int k = 0; k < outputs; k++) {
      counts[k] = counts[k] > 0 ? counts[k] : ABSENT_COUNT;
      total += counts[k];
    }
    double nll = 0;
    for (int row = 0; row < split.stop().size(); row++) {
      nll -= Math.log(counts[split.stop().permutation(row)] / total);
    }
    return nll / split.stop().size();
  }
}
