package com.epsilon.prism.collect;

import com.epsilon.agent.observe.Observation;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.policy.GreedyAnswer;
import com.epsilon.prism.policy.RegimeRepresentatives;
import java.util.Arrays;

/**
 * 候補集合 C(o)。o・π_t・p0・乱数だけで決まり、局の結果には依らない。ε_C の乱数は、H を固定しても一様に分布する（{@link CrnSeeds}）。
 *
 * <p>{@link #build} は、次の優先順に手を足していく。すでに入っている手と結果が同じ手（{@link Twins#same}）は飛ばし、上限に達したら止める。
 * 上限は {@code maxCandidates} で、ε_C の乱数が当たったときだけ、探索の 1 手を足すために 1 つ減らす。
 *
 * <ol>
 *   <li>s が最大の手（b の貪欲な手。比べ方は {@link Greedy#before} で、同点なら p0 の大きい手）
 *   <li>体制の代表（{@link RegimeRepresentatives#of} が返す順）
 *   <li>p0 が最大の手
 *   <li>s が 2 番目の手
 *   <li>p0 が 2 番目の手
 *   <li>s が 3 番目の手
 * </ol>
 *
 * ε_C の乱数が当たったとき（exploreCoin &lt; ε_C）は、探索の 1 手を足す。ここまでのどの手とも結果が同じでない合法手を、同じ手の組ごとに
 * 番号の最も小さい手で代表させて並べ、{@code floor(exploreDraw × 代表の数)} 番目を足す。代表が無ければ足さない。
 *
 * <p>打った手が C の中のある手と結果が同じなら、C の中のその手を打った手に置き換える。打った手には主経路の値をそのまま使えるので、
 * 分岐が要らない。打った手がどの手とも結果が違うなら C はそのままで、値を測る手は C に打った手を加えた C ∪ {played} になる
 * （{@link #measured}）。一様分布を混ぜた分で選んだ手にも、こうして追加の分岐なしで値が付く。ほかの手の組では、代表の手を、
 * 分岐で打つときと行に書くときに、組の中で切る方の手（{@link TwinDiscard}）に替える。
 *
 * <p>s の順位は {@link Greedy#before} で決める。同点なら p0 の大きい手、それも同じなら番号の小さい手を上にする。p0 の順位の同点は、
 * 番号の小さい手を上にする。「k 番目の手」は、結果が同じ手をまとめた順位で数える（上位に同じ手がある手は数えない）。そのため s の上位 3 手は、
 * 互いに結果の違う 3 つの候補になる。s の 1 番目は b の貪欲な手そのもので、方策の貪欲な手 {@link GreedyAnswer} と同じ比べ方で選ぶ。
 *
 * @param support C(o)（合法手の位置の昇順）。根の記録では、逸脱の分布 q の台になる
 * @param measured ラベル行で値を測る手 C ∪ {played}（昇順）
 * @param explored ε_C の探索の 1 手を足したなら true
 */
record CandidateSet(int[] support, int[] measured, boolean explored) {
  /**
   * 候補集合を作る。
   *
   * @param o 判断の時点の記録。判断する席から見える公開情報だけを読む（体制の代表を選ぶときの安全度）
   * @param legal 合法手の行動の番号（{@code o} の並び）
   * @param scores 合法手ごとの s（FP32）
   * @param prior 合法手ごとの p0
   * @param threats 判断する席から見た、公開情報で分かる脅威の数
   * @param played その経路で打つ手の位置
   * @param maxCandidates 候補の数の上限
   * @param epsilonC 探索の 1 手を足す確率 ε_C
   * @param exploreCoin {@link CollectContext#exploreCoin}
   * @param exploreDraw {@link CollectContext#exploreDraw}
   * @return 候補集合
   * @throws IllegalArgumentException 合法手が 2 つ未満のとき。{@code played} が合法手の範囲に無いとき
   */
  static CandidateSet build(
      Observation o,
      int[] legal,
      float[] scores,
      float[] prior,
      int threats,
      int played,
      int maxCandidates,
      double epsilonC,
      double exploreCoin,
      double exploreDraw) {
    int n = legal.length;
    if (n < 2) throw new IllegalArgumentException("Candidate sets need two or more legal actions");
    if (played < 0 || played >= n) throw new IllegalArgumentException("Played action " + played);
    boolean explore = exploreCoin < epsilonC;
    var members = new Members(legal, explore ? maxCandidates - 1 : maxCandidates);
    int[] all = new int[n];
    for (int a = 0; a < n; a++) all[a] = a;
    int[] byScore = distinct(legal, all, ranking(all, scores, prior));
    int[] byPrior = distinct(legal, all, ranking(all, prior, null));
    members.add(byScore, 0);
    for (int a : RegimeRepresentatives.of(o, legal, scores, prior, threats))
      members.add(a);
    members.add(byPrior, 0);
    members.add(byScore, 1);
    members.add(byPrior, 1);
    members.add(byScore, 2);
    boolean explored = false;
    if (explore) {
      // ここまでのどの手とも結果が同じでない手を、同じ手の組ごとに番号の最も小さい手で代表させ、番号順に並べる
      int[] pool = new int[n];
      int size = 0;
      for (int a = 0; a < n; a++)
        if (!members.coveredBy(a) && Twins.lowest(legal, a) == a) pool[size++] = a;
      if (size > 0) {
        int pick = Math.min(size - 1, (int) Math.floor(exploreDraw * size));
        members.force(pool[pick]);
        explored = true;
      }
    }
    int[] support = members.toArray();
    // 打った手と結果が同じ手は、打った手に置き換える。値は経路の結果がそのまま使えるので、分岐が要らない
    boolean inside = false;
    for (int i = 0; i < support.length; i++) {
      if (Twins.same(legal, support[i], played)) {
        support[i] = played;
        inside = true;
      }
    }
    Arrays.sort(support);
    int[] measured;
    if (inside) {
      measured = support.clone();
    } else {
      measured = Arrays.copyOf(support, support.length + 1);
      measured[support.length] = played;
      Arrays.sort(measured);
    }
    return new CandidateSet(support, measured, explored);
  }

  /**
   * 全ての合法手を候補にした集合を作る（{@link Sampling#measuresAll} のクラスで使う）。結果が同じ手の組ごとに、番号の最も小さい手を代表にする。
   * 打った手の組だけは打った手そのものを代表にするので、打った手はいつも C に入り、値を測る手は C そのものになる。
   */
  static CandidateSet allActions(int[] legal, int played) {
    int n = legal.length;
    if (n < 2) throw new IllegalArgumentException("Candidate sets need two or more legal actions");
    if (played < 0 || played >= n) throw new IllegalArgumentException("Played action " + played);
    int[] support = new int[n];
    int size = 0;
    for (int a = 0; a < n; a++)
      if (Twins.lowest(legal, a) == a) support[size++] = Twins.same(legal, a, played) ? played : a;
    support = Arrays.copyOf(support, size);
    Arrays.sort(support);
    return new CandidateSet(support, support.clone(), false);
  }

  /** {@link #ranking(int[], float[], float[])} で、同点を番号だけで割るもの。 */
  static int[] ranking(int[] actions, float[] values) {
    return ranking(actions, values, null);
  }

  /**
   * 手を値の大きい順に並べ、その順を {@code actions} の中の添字で返す。比べ方は方策の貪欲な手と同じ {@link Greedy#before} で、
   * 同点なら {@code tieBreak} の大きい手を上にする。それも同じか {@code tieBreak} が null なら、番号の小さい手を上にする。
   * 挿入ソートなので、NaN が混じっても順列を返す。
   *
   * @param values 合法手ごとの、並べる基準の値（s か p0）
   * @param tieBreak 同点を割るための合法手ごとの値（s の順位なら p0、p0 の順位なら null）
   */
  static int[] ranking(int[] actions, float[] values, float[] tieBreak) {
    int m = actions.length;
    int[] order = new int[m];
    for (int i = 0; i < m; i++) {
      int j = i - 1;
      while (j >= 0 && Greedy.before(actions[i], actions[order[j]], values, tieBreak)) {
        order[j + 1] = order[j];
        j--;
      }
      order[j + 1] = i;
    }
    return order;
  }

  /** 順位（{@code actions} の中の添字）を順にたどり、上位に結果が同じ手がある手を除いて、手の位置を並べて返す。 */
  private static int[] distinct(int[] legal, int[] actions, int[] order) {
    int[] result = new int[order.length];
    int size = 0;
    for (int index : order) {
      int a = actions[index];
      boolean twin = false;
      for (int k = 0; k < size && !twin; k++) twin = Twins.same(legal, result[k], a);
      if (!twin) result[size++] = a;
    }
    return Arrays.copyOf(result, size);
  }

  /** 結果が同じ手をまとめながら、上限まで手を足していく入れ物。手は足した順に持つ。 */
  private static final class Members {
    private final int[] legal;
    private final int limit;
    private final int[] members;
    private int size;

    Members(int[] legal, int limit) {
      this.legal = legal;
      this.limit = limit;
      members = new int[Math.max(1, limit) + 1];
    }

    /** 順位の k 番目の手を足す。k 番目が無ければ何もしない。 */
    void add(int[] ranked, int k) {
      if (k < ranked.length) add(ranked[k]);
    }

    /** 上限に達しておらず、入っている手と結果が同じでもなければ、手を足す。範囲外の位置（-1 など）は無視する。 */
    void add(int a) {
      if (a < 0 || a >= legal.length || size >= limit || coveredBy(a)) return;
      members[size++] = a;
    }

    /** 上限に関わらず手を足す。ε_C の探索の 1 手に使い、そのぶん上限を 1 つ減らしてある。 */
    void force(int a) {
      members[size++] = a;
    }

    /** 入っている手のどれかと同じ手か、結果が同じ手なら true。 */
    boolean coveredBy(int a) {
      for (int k = 0; k < size; k++) if (Twins.same(legal, members[k], a)) return true;
      return false;
    }

    int[] toArray() {
      return Arrays.copyOf(members, size);
    }
  }

  /** 打った手が C に入っているなら true。入っていなければ、一様分布を混ぜた分で選んだ手。 */
  boolean playedInside(int played) {
    for (int a : support) if (a == played) return true;
    return false;
  }

  /** 合法手の位置の並びを、行に書く形（{@link RowDraft#support}）に変える。 */
  static byte[] bytes(int[] positions) {
    byte[] bytes = new byte[positions.length];
    for (int i = 0; i < positions.length; i++) bytes[i] = (byte) positions[i];
    return bytes;
  }
}
