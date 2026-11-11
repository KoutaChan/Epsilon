package com.epsilon.grp;

import com.epsilon.grp.train.PhiStamp;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.Random;
import java.util.Set;

/** テストで使うネットワーク・卓・履歴。com.epsilon.grp・grp.train・grp.rules のテストが共有する。 */
public final class GrpFixtures {

  /** 数の列のヘッダーに書く Φ（GRP のモード）。 */
  public static final PhiStamp STAMP = new PhiStamp("GRP", Double.NaN);

  private GrpFixtures() {}

  /** 全てのフィールド。幅は四人打ちで 40、三人打ちで 26。 */
  public static Set<GrpField> plainFields() {
    return EnumSet.range(GrpField.ROUND, GrpField.GAP);
  }

  public static GrpShape shape(Variant variant, int hidden, boolean phaseHeads) {
    return new GrpShape(variant, plainFields(), hidden, phaseHeads);
  }

  /**
   * OutcomeNetwork を作る。重みは平均 0 の正規分布から引き、乱数の種で決まる。
   *
   * @param scale 重みの標準偏差
   */
  public static OutcomeNetwork outcomes(OutcomeShape shape, long seed, double scale) {
    Random random = new Random(seed);
    float[] weights = new float[OutcomeTensor.starts(shape)[OutcomeTensor.values().length]];
    for (int index = 0; index < weights.length; index++) {
      weights[index] = (float) (random.nextGaussian() * scale);
    }
    return OutcomeNetwork.of(shape, weights);
  }

  /**
   * 卓に依らない OutcomeNetwork を作る。出力層のバイアスだけを持ち、ほかの重みは 0。
   *
   * @param logits HandOutcomes の並びのロジット
   */
  public static OutcomeNetwork biased(OutcomeShape shape, double[] logits) {
    int[] starts = OutcomeTensor.starts(shape);
    float[] weights = new float[starts[OutcomeTensor.values().length]];
    for (int index = 0; index < logits.length; index++) {
      weights[starts[OutcomeTensor.B3.ordinal()] + index] = (float) logits[index];
    }
    return OutcomeNetwork.of(shape, weights);
  }

  /**
   * 区分・和了者・2 人目の和了者・放銃者・聴牌した席のビットが全て合う終わり方の番号を返す。いない席には HandOutcomes.NONE を、
   * 聴牌した席が無ければ 0 を渡す。合う終わり方が無ければ例外を投げる。
   */
  public static int main(HandOutcomes hands, OutcomeSection section, int winner, int second, int loser, int tenpai) {
    for (int main = 0; main < hands.mainCount(); main++) {
      if (hands.section(main) == section
          && hands.winner(main) == winner
          && hands.secondWinner(main) == second
          && hands.loser(main) == loser
          && hands.tenpai(main) == tenpai) {
        return main;
      }
    }
    throw new IllegalArgumentException("No main outcome " + section + " " + winner + " " + second + " " + loser);
  }

  /**
   * GrpNetwork を作る。重みは平均 0 の正規分布から引き、乱数の種で決まる。
   *
   * @param scale 重みの標準偏差
   */
  public static GrpNetwork network(GrpShape shape, long seed, double scale) {
    Random random = new Random(seed);
    float[] weights = new float[GrpTensor.starts(shape)[GrpTensor.values().length]];
    for (int index = 0; index < weights.length; index++) {
      weights[index] = (float) (random.nextGaussian() * scale);
    }
    return GrpNetwork.of(shape, weights);
  }

  /**
   * GruSinkhorn を作る。重みは平均 0 の正規分布から引き、乱数の種で決まる。
   *
   * @param scale 重みの標準偏差
   */
  public static GruSinkhorn gru(GruShape shape, long seed, double scale) {
    Random random = new Random(seed);
    float[] weights = new float[GruTensor.count(shape)];
    for (int index = 0; index < weights.length; index++) {
      weights[index] = (float) (random.nextGaussian() * scale);
    }
    return GruSinkhorn.of(shape, weights);
  }

  /** 四人打ちの終局の卓。南 4 の親（絶対席 3）が不聴で流れ、トップが返し点に届いて終わる。点は [40,000, 20,000, 20,000, 20,000]。 */
  public static TableState finished() {
    TableState start = new TableState(Variant.TENHOU_4P);
    start.set(new RoundStart(Variant.TENHOU_4P, 7, 0, 0, new int[] {40_000, 20_000, 20_000, 20_000}));
    RoundOutcome draw = new RoundOutcome(Variant.TENHOU_4P);
    int[] delta = new int[4];
    Settlement.apply(Variant.TENHOU_4P, start, draw, delta);
    TableState end = new TableState(Variant.TENHOU_4P);
    Progression.next(start, draw, delta, end);
    return end;
  }

  /** 指定した数の局を積んだ履歴を作る。各局の卓は {@link #state} と同じ作り方で作る。 */
  public static RoundHistory history(Variant variant, Random random, int rounds) {
    RoundHistory history = new RoundHistory(variant);
    for (int round = 0; round < rounds; round++) {
      history.add(state(variant, random));
    }
    return history;
  }

  /** 半荘が続く卓をランダムに作る。点は 100 点の倍数で、和は席の数 × 配給点 − 供託の点。4 回に 1 回は、2 席を同点か 100 点差にする。 */
  public static GrpState state(Variant variant, Random random) {
    int seats = variant.seats();
    int sticks = random.nextInt(4);
    int units = (seats * variant.startPoints() - Variant.DEPOSIT * sticks) / 100;
    int[] cuts = new int[seats + 1];
    for (int cut = 1; cut < seats; cut++) {
      cuts[cut] = random.nextInt(units + 1);
    }
    cuts[seats] = units;
    Arrays.sort(cuts, 1, seats);
    int[] scores = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      scores[seat] = (cuts[seat + 1] - cuts[seat]) * 100;
    }
    if (random.nextInt(4) == 0) {
      int a = random.nextInt(seats);
      int b = (a + 1 + random.nextInt(seats - 1)) % seats;
      int total = scores[a] + scores[b];
      scores[a] = total / 200 * 100;
      scores[b] = total - scores[a];
    }
    return new GrpState(variant, random.nextInt(variant.finalRound() + 1), random.nextInt(6), sticks, scores);
  }
}
