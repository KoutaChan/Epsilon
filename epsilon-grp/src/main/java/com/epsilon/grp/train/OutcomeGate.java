package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeTensor;
import com.epsilon.grp.OutcomeWorkspace;
import java.util.function.IntPredicate;

/**
 * 局の結果のネットワーク q の候補を、関門のコホートの局末で判定する。測るのは、FinalRound が数え上げる局末
 * （{@code FinalRound.covers}）のうち、遷移を再現する結果を持つものだけ。測る量は局末あたりの周辺 NLL −log Σ_o q(o|s) の差
 * （候補 − 比べる q）で、標準誤差は半荘をクラスタとして求める。関門のコホートの局末には、局を数えたファイルから作ったもの
 * （{@code NextHands.of(HandEnds, CountedHands)}。結果は記録した 1 つ）を渡す。
 *
 * <p>使用中の q があれば、それと比べて差が +2 SE 以下なら（悪くなっていなければ）通す。学習し直すたびの改善は、コホート 1 つで測る
 * SE より小さいので、良くなったことまでは求めない。使用中の q が無ければ、一様な q（重みが全て 0 で、終わり方と打点は一様、立直は
 * 1/2）と比べて、差が負なら通す。
 */
public final class OutcomeGate {

  /** 使用中の q より悪くてよい幅（SE の倍数）。 */
  static final double STANDARD_ERRORS = 2;

  private OutcomeGate() {}

  /** 関門の種類。 */
  public enum Kind {
    /** 使用中の q と比べる。差が +2 SE 以下なら通す。 */
    REFIT,
    /** 使用中の q が無いので、一様な q と比べる。差が負なら通す。 */
    FIRST;

    /** 差が基準を満たせば true。測った局末が無い（NaN）ときは満たさない。 */
    boolean accepts(Paired versus) {
      return switch (this) {
        case REFIT -> versus.mean() <= STANDARD_ERRORS * versus.standardError();
        case FIRST -> versus.mean() < 0;
      };
    }
  }

  /**
   * 関門の判定。
   *
   * @param kind 関門の種類
   * @param passed 通ったか
   * @param versus 局末あたりの周辺 NLL の差（候補 − 比べる q）の平均と、その標準誤差
   * @param handEnds 測った局末の数
   * @param skipped FinalRound が数え上げる局末のうち、遷移を局の結果のネットワークで表せず測らなかった数
   */
  public record Report(Kind kind, boolean passed, Paired versus, long handEnds, long skipped) {}

  /**
   * 使用中の q と比べる。
   *
   * @param next 関門のコホートの局末と、その次の局の結果
   * @param used 使用中の q
   * @param candidate 候補の q
   * @throws IllegalArgumentException どちらかの q のルールが局末と違うとき、上の席を抜くのに要る打点のフィールドを持つとき
   */
  public static Report refit(NextHands next, OutcomeNetwork used, OutcomeNetwork candidate) {
    return judge(Kind.REFIT, next, used, candidate);
  }

  /**
   * 使用中の q が無いときに、一様な q と比べる。
   *
   * @param next 関門のコホートの局末と、その次の局の結果
   * @param candidate 候補の q
   * @throws IllegalArgumentException q のルールが局末と違うとき、上の席を抜くのに要る打点のフィールドを持つとき
   */
  public static Report first(NextHands next, OutcomeNetwork candidate) {
    return judge(Kind.FIRST, next, uniform(candidate.shape()), candidate);
  }

  private static Report judge(Kind kind, NextHands next, OutcomeNetwork reference, OutcomeNetwork candidate) {
    HandEnds ends = next.ends();
    IntPredicate covered = h -> FinalRound.covers(ends.state(h));
    double[] ofCandidate = next.logLikelihoods(covered, candidate, () -> OutcomeWorkspace.of(candidate.shape()));
    double[] ofReference = next.logLikelihoods(covered, reference, () -> OutcomeWorkspace.of(reference.shape()));
    Clusters differences = new Clusters(1);
    long skipped = 0;
    for (int game = 0; game < ends.games(); game++) {
      for (int h = ends.firstHand(game); h < ends.firstHand(game + 1); h++) {
        if (!covered.test(h)) {
          continue;
        }
        if (next.found(h) == 0) {
          skipped++;
          continue;
        }
        differences.add(0, ofReference[h] - ofCandidate[h]);
      }
      differences.endGame();
    }
    Paired versus = differences.paired(0);
    return new Report(kind, kind.accepts(versus), versus, (long) differences.count(0), skipped);
  }

  /** 重みが全て 0 の q。終わり方と打点は一様、立直は 1/2。 */
  private static OutcomeNetwork uniform(OutcomeShape shape) {
    return OutcomeNetwork.of(shape, new float[OutcomeTensor.starts(shape)[OutcomeTensor.values().length]]);
  }
}
