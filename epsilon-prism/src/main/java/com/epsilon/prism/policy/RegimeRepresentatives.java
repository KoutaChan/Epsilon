package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.Threats;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;

/**
 * 体制の代表（候補集合の「体制の代表(o)」）。確約点の両側の行き先を候補に必ず入れる。o・s・p0 と判断の席から見える公開情報だけで決める。
 *
 * <p>返す順（当てはまる体制だけ。各組は「確約する側、しない側」の順。同じ手が重なっても除かない。双子をまとめるのは呼び出し側の {@link
 * com.epsilon.prism.collect.CandidateSet}）:
 *
 * <ol>
 *   <li>鳴き: CHI か PON が合法なら、CHI・PON の中の s の最大の手と、PASS。
 *   <li>立直: RIICHI_DAHAI が合法なら、RIICHI_DAHAI の中の s の最大の手と、DAHAI の中の s の最大の手（DAHAI がなければ省く）。
 *   <li>槓: ANKAN・KAKAN・DAIMINKAN のどれかが合法なら、その中の s の最大の手と、槓でない手の中の s の最大の手。
 *   <li>脅威: 公開の脅威が 1 つ以上で打牌（DAHAI・RIICHI_DAHAI）が合法なら、打牌の中の p0 の最大の手（効率の最善）と、{@link PublicSafety#safest}
 *       （公開情報だけで測った安全度の最善）。
 * </ol>
 *
 * s の最大は {@link Greedy#before} の比べ方（同点は p0 の大きい手、それも同じなら番号の小さい手）、p0 の最大の同点は番号の小さい手。
 */
public final class RegimeRepresentatives {
  private static final Predicate<Action.Type> CALL =
      type -> type == Action.Type.CHI || type == Action.Type.PON;
  private static final Predicate<Action.Type> KAN =
      type ->
          type == Action.Type.ANKAN || type == Action.Type.KAKAN || type == Action.Type.DAIMINKAN;
  private static final Predicate<Action.Type> DISCARD =
      type -> type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI;

  private RegimeRepresentatives() {}

  /**
   * 体制の代表の手の位置（上の順。当てはまる体制がなければ空）。
   *
   * @param state 判断の時点の state（commit の前。席 seat から見える公開情報だけを読む）
   * @param scores 合法手ごとの s
   * @param prior 合法手ごとの p0
   * @param threats 判断の席から見た公開の脅威の数（{@link Threats#count}）
   */
  public static int[] of(
      GameState state, int seat, List<Action> legal, float[] scores, float[] prior, int threats) {
    int[] chosen = new int[8];
    int size = 0;
    int call = best(legal, scores, prior, CALL);
    if (call >= 0) {
      chosen[size++] = call;
      int pass = best(legal, scores, prior, type -> type == Action.Type.PASS);
      if (pass >= 0) chosen[size++] = pass;
    }
    int riichi = best(legal, scores, prior, type -> type == Action.Type.RIICHI_DAHAI);
    if (riichi >= 0) {
      chosen[size++] = riichi;
      int dama = best(legal, scores, prior, type -> type == Action.Type.DAHAI);
      if (dama >= 0) chosen[size++] = dama;
    }
    int kan = best(legal, scores, prior, KAN);
    if (kan >= 0) {
      chosen[size++] = kan;
      int other = best(legal, scores, prior, KAN.negate());
      if (other >= 0) chosen[size++] = other;
    }
    if (threats > 0) {
      int efficient = best(legal, prior, null, DISCARD);
      if (efficient >= 0) {
        chosen[size++] = efficient;
        chosen[size++] = PublicSafety.safest(state, seat, legal, prior);
      }
    }
    return Arrays.copyOf(chosen, size);
  }

  /**
   * 種類が member の手のうち values の最大の手の位置（{@link Greedy#before}: 同点は tieBreak の大きい手、それも同じか tieBreak が null なら番号の
   * 小さい手。なければ -1）。
   */
  private static int best(
      List<Action> legal, float[] values, float[] tieBreak, Predicate<Action.Type> member) {
    int best = -1;
    for (int a = 0, n = legal.size(); a < n; a++)
      if (member.test(legal.get(a).type())
          && (best < 0 || Greedy.before(a, best, values, tieBreak))) best = a;
    return best;
  }
}
