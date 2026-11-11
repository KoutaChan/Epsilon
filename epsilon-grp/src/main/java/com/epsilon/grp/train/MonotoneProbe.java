package com.epsilon.grp.train;

import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.Phi;
import com.epsilon.grp.RoundHistory;
import java.util.ArrayList;
import java.util.List;

/**
 * 点を動かして、Φ が点について単調かを調べる。局末を乱数の種で選び、その卓（次の局の始まりの卓）で、席 j から席 i へ
 * δ ∈ {1,000, 4,000, 12,000} 点を動かす。j が 0 点未満になる動かし方は、半荘が続かないので除く。受け取った席の Φ_i が 1e-6 を
 * 超えて下がるか、渡した席の Φ_j が 1e-6 を超えて上がれば、逆向きに動いたと数える。動かし方 1 つにつき、受け取った席と渡した席の
 * 2 回を調べ、そのうち逆向きだった割合を出す。GrpNetwork 以外の Φ も、{@link #moves()} の卓の値を渡せば同じ動かし方で測れる。
 */
public final class MonotoneProbe {

  private static final int[] POINTS = {1_000, 4_000, 12_000};
  private static final int SAMPLES = 4_096;
  private static final long SEED = 0x6A7E_5EEDL;

  /** Φ がこの値を超えて逆向きに変われば、逆向きに動いたとみなす。丸め誤差より大きい値にする。 */
  private static final double TOLERANCE = 1e-6;

  /** 点を動かしていない元の卓で、受け取る席と渡す席に入れる値。 */
  private static final int NONE = -1;

  /**
   * 調べる卓。
   *
   * @param handEnd 元の局末の番号
   * @param state 局末の卓。点を動かした卓なら、渡す席から受け取る席へ点を動かした後の卓
   * @param receiver 点を受け取る席。元の卓なら −1
   * @param giver 点を渡す席。元の卓なら −1
   */
  public record Move(int handEnd, GrpState state, int receiver, int giver) {}

  private final HandEnds ends;
  private final int samples;

  /** 選んだ局末ごとに、まず元の卓を置き、続けて点を動かした卓を動かし方の順に並べる。 */
  private final List<Move> moves;

  private MonotoneProbe(HandEnds ends, int samples, List<Move> moves) {
    this.ends = ends;
    this.samples = samples;
    this.moves = moves;
  }

  /** 局末を重複なく最大 4,096 個、決まった乱数の種で選ぶ。 */
  public static MonotoneProbe of(HandEnds ends) {
    return of(ends, SAMPLES, SEED);
  }

  static MonotoneProbe of(HandEnds ends, int samples, long seed) {
    int[] chosen = Samples.choose(ends.handEnds(), samples, seed);
    int seats = ends.seats();
    List<Move> moves = new ArrayList<>();
    for (int handEnd : chosen) {
      GrpState state = ends.state(handEnd);
      moves.add(new Move(handEnd, state, NONE, NONE));
      for (int points : POINTS) {
        for (int receiver = 0; receiver < seats; receiver++) {
          for (int giver = 0; giver < seats; giver++) {
            if (receiver == giver || state.score(giver) < points) {
              continue;
            }
            int[] scores = state.scores();
            scores[receiver] += points;
            scores[giver] -= points;
            GrpState shifted = new GrpState(state.variant(), state.round(), state.honba(), state.sticks(), scores);
            moves.add(new Move(handEnd, shifted, receiver, giver));
          }
        }
      }
    }
    return new MonotoneProbe(ends, chosen.length, List.copyOf(moves));
  }

  public int samples() {
    return samples;
  }

  /** 調べる卓。選んだ局末ごとに、まず元の卓を置き、続けて点を動かした卓を動かし方の順に並べる。 */
  public List<Move> moves() {
    return moves;
  }

  /**
   * 渡した Φ が逆向きに動いた割合を求める。動かし方が無ければ 0。卓の数が {@link #moves()} と違えば IllegalArgumentException を
   * 投げる。
   *
   * @param values 卓ごと・絶対席ごとの Φ。卓は {@link #moves()} の順
   */
  public double violationRate(double[][] values) {
    if (values.length != moves.size()) {
      throw new IllegalArgumentException(moves.size() + " tables need values, got " + values.length);
    }
    long violations = 0;
    long checks = 0;
    double[] base = null;
    for (int index = 0; index < values.length; index++) {
      Move move = moves.get(index);
      if (move.receiver() == NONE) {
        base = values[index];
        continue;
      }
      double[] moved = values[index];
      if (moved[move.receiver()] - base[move.receiver()] < -TOLERANCE) {
        violations++;
      }
      if (moved[move.giver()] - base[move.giver()] > TOLERANCE) {
        violations++;
      }
      checks += 2;
    }
    return checks == 0 ? 0 : violations / (double) checks;
  }

  /**
   * Φ の項目が逆向きに動く割合。局の履歴が要る項目（GruSinkhorn など）には、卓ごとに、元の局末までの局の履歴を積んで渡す。項目の
   * ルールが局末と違えば IllegalArgumentException を投げる。
   */
  public double violationRate(Phi phi) {
    if (phi.variant() != ends.variant()) {
      throw new IllegalArgumentException("A " + phi.variant() + " Φ for " + ends.variant() + " hand ends");
    }
    RoundHistory history = phi.needsHistory() ? new RoundHistory(ends.variant()) : null;
    double[][] values = new double[moves.size()][ends.seats()];
    double[] ranks = new double[ends.seats() * ends.seats()];
    for (int index = 0; index < values.length; index++) {
      Move move = moves.get(index);
      if (history != null) {
        ends.history(move.handEnd(), history);
      }
      phi.evaluate(history, move.state(), values[index], ranks);
    }
    return violationRate(values);
  }

  /** ネットワークの Φ が逆向きに動く割合。ネットワークのルールが局末と違えば IllegalArgumentException を投げる。 */
  public double violationRate(GrpNetwork network, GrpWorkspace workspace) {
    if (network.variant() != ends.variant()) {
      throw new IllegalArgumentException("A " + network.variant() + " network for " + ends.variant() + " hand ends");
    }
    double[][] values = new double[moves.size()][ends.seats()];
    for (int index = 0; index < values.length; index++) {
      network.phi(moves.get(index).state(), ends.utility(), workspace, values[index]);
    }
    return violationRate(values);
  }
}
