package com.epsilon.prism.collect;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.policy.GreedyAnswer;
import com.epsilon.prism.policy.RegimeRepresentatives;
import java.util.Arrays;
import java.util.List;

/**
 * 候補集合 C(o)。o・π_t・p0・硬貨だけで決め、結果に依らない。硬貨（ε_C）は H を固定しても一様（{@link CrnSeeds}）。
 *
 * <p>作り方（{@link #build}）: 次の優先の順に手を足し、すでに入っている手の双子（{@link Twins#same}）は飛ばす。上限は ε_C の硬貨が当たれば
 * maxCandidates − 1、外れれば maxCandidates で、上限に達したら止める。
 *
 * <ol>
 *   <li>s の最大の手（b の貪欲の手。{@link Greedy#before} の比べ方で、同点は p0 の大きい手）
 *   <li>体制の代表（{@link RegimeRepresentatives#of} が返す順）
 *   <li>p0 の最大の手
 *   <li>s の 2 番目の手
 *   <li>p0 の 2 番目の手
 *   <li>s の 3 番目の手
 * </ol>
 *
 * ε_C の硬貨（exploreCoin &lt; ε_C）が当たれば、ここまでの手のどれとも双子でない合法手のうち、双子の組ごとに番号の最も小さい手を代表として並べ、
 * {@code floor(exploreDraw × 代表の数)} 番目を足す（代表がなければ足さない）。
 *
 * <p>双子の置き換え: 打った手 played がある手 c の双子なら、C の中の c を played に置き換える（打った手は主経路の値を流用するので分岐が要らない）。
 * played がどの手とも双子でなければ C はそのままで、測る手は C ∪ {played}（{@link #measured}。床 ε_b で選んだ手は値がただで付く）。
 *
 * <p>s の順位は {@link Greedy#before}（同点は p0 の大きい手、それも同じなら番号の小さい手）、p0 の順位の同点は番号の小さい手を上にする。「k 番目の
 * 手」は双子をまとめた順位で数える（上位に双子がある手は数えない）ので、s の上位 3 手は双子でない 3 つの候補になる。s の 1 番目は b の貪欲の手そのもの
 * （方策の貪欲 {@link GreedyAnswer} と同じ比べ方）。
 *
 * @param support C(o)（合法手の位置の昇順）。根の記録では逸脱の分布 q の台
 * @param measured ラベル行で値を測る手 C ∪ {played}（昇順）
 * @param explored ε_C の 1 手を足したか
 */
record CandidateSet(int[] support, int[] measured, boolean explored) {
  /**
   * 候補集合を作る。state は判断の時点（commit の前）で、判断の席から見える公開情報だけを読む（体制の代表の安全度）。
   *
   * @param scores 合法手ごとの s（FP32）
   * @param prior 合法手ごとの p0
   * @param threats 判断の席から見た公開の脅威の数
   * @param played その経路で打つ手の位置
   * @param exploreCoin {@link CollectContext#exploreCoin}
   * @param exploreDraw {@link CollectContext#exploreDraw}
   */
  static CandidateSet build(
      GameState state,
      int seat,
      List<Action> legal,
      float[] scores,
      float[] prior,
      int threats,
      int played,
      int maxCandidates,
      double epsilonC,
      double exploreCoin,
      double exploreDraw) {
    int n = legal.size();
    if (n < 2) throw new IllegalArgumentException("Candidate sets need two or more legal actions");
    if (played < 0 || played >= n) throw new IllegalArgumentException("Played action " + played);
    boolean explore = exploreCoin < epsilonC;
    var members = new Members(legal, explore ? maxCandidates - 1 : maxCandidates);
    int[] all = new int[n];
    for (int a = 0; a < n; a++) all[a] = a;
    int[] byScore = distinct(legal, all, ranking(all, scores, prior));
    int[] byPrior = distinct(legal, all, ranking(all, prior, null));
    members.add(byScore, 0);
    for (int a : RegimeRepresentatives.of(state, seat, legal, scores, prior, threats))
      members.add(a);
    members.add(byPrior, 0);
    members.add(byScore, 1);
    members.add(byPrior, 1);
    members.add(byScore, 2);
    boolean explored = false;
    if (explore) {
      // ここまでの手のどれとも双子でない手を、双子の組ごとに番号の最も小さい手で代表させて番号順に並べる
      int[] pool = new int[n];
      int size = 0;
      for (int a = 0; a < n; a++)
        if (!members.coveredBy(a) && lowestTwin(legal, a)) pool[size++] = a;
      if (size > 0) {
        int pick = Math.min(size - 1, (int) Math.floor(exploreDraw * size));
        members.force(pool[pick]);
        explored = true;
      }
    }
    int[] support = members.toArray();
    // 打った手の双子は打った手に置き換える（値は経路のものがただで付き、分岐が要らない）
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
   * 全ての合法手の候補集合（{@link Sampling#measuresAll} の類）。双子の組ごとに番号の最も小さい手を代表にし、打った手の組は打った手そのものを
   * 代表にするので、打った手はいつも C の中で、測る手は C そのもの。
   */
  static CandidateSet allActions(List<Action> legal, int played) {
    int n = legal.size();
    if (n < 2) throw new IllegalArgumentException("Candidate sets need two or more legal actions");
    if (played < 0 || played >= n) throw new IllegalArgumentException("Played action " + played);
    int[] support = new int[n];
    int size = 0;
    for (int a = 0; a < n; a++)
      if (lowestTwin(legal, a)) support[size++] = Twins.same(legal, a, played) ? played : a;
    support = Arrays.copyOf(support, size);
    Arrays.sort(support);
    return new CandidateSet(support, support.clone(), false);
  }

  /** {@link #ranking(int[], float[], float[])} の同点を番号だけで割る形。 */
  static int[] ranking(int[] actions, float[] values) {
    return ranking(actions, values, null);
  }

  /**
   * actions を values の降順に並べた、actions の中の添字。比べ方は {@link Greedy#before}（同点は tieBreak の大きい手、それも同じか tieBreak が
   * null なら番号の小さい手）で、方策の貪欲と同じ。挿入ソートなので、NaN が混じっても順列を返す。
   *
   * @param values 合法手ごとの主の値（s か p0）
   * @param tieBreak 同点を割る合法手ごとの値（s の順位なら p0、p0 の順位なら null）
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

  /** 順位 order（actions の中の添字）から、上位に双子がある手を除いた手の位置の列。 */
  private static int[] distinct(List<Action> legal, int[] actions, int[] order) {
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

  /** a が自分の双子の組の中で番号の最も小さい手か。 */
  private static boolean lowestTwin(List<Action> legal, int a) {
    for (int b = 0; b < a; b++) if (Twins.same(legal, b, a)) return false;
    return true;
  }

  /** 双子をまとめて上限まで手を足す入れ物（足した順に持つ）。 */
  private static final class Members {
    private final List<Action> legal;
    private final int limit;
    private final int[] members;
    private int size;

    Members(List<Action> legal, int limit) {
      this.legal = legal;
      this.limit = limit;
      members = new int[Math.max(1, limit) + 1];
    }

    /** ranked の k 番目の手を足す（なければ何もしない）。 */
    void add(int[] ranked, int k) {
      if (k < ranked.length) add(ranked[k]);
    }

    /** 上限に達しておらず、入っている手の双子でなければ a を足す。範囲外の位置（-1 など）は無視する。 */
    void add(int a) {
      if (a < 0 || a >= legal.size() || size >= limit || coveredBy(a)) return;
      members[size++] = a;
    }

    /** 上限に依らず a を足す（ε_C の 1 手。上限は 1 枠空けてある）。 */
    void force(int a) {
      members[size++] = a;
    }

    /** a が入っている手のどれかと同じ手か双子か。 */
    boolean coveredBy(int a) {
      for (int k = 0; k < size; k++) if (Twins.same(legal, members[k], a)) return true;
      return false;
    }

    int[] toArray() {
      return Arrays.copyOf(members, size);
    }
  }

  /** played が support に入っているか（入っていなければ床 ε_b の手）。 */
  boolean playedInside(int played) {
    for (int a : support) if (a == played) return true;
    return false;
  }

  /** 行に書く support（{@link RowDraft#support}。位置の昇順）。 */
  byte[] supportBytes() {
    byte[] bytes = new byte[support.length];
    for (int i = 0; i < support.length; i++) bytes[i] = (byte) support[i];
    return bytes;
  }
}
