package com.epsilon.prism.policy;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.core.Twins;
import java.util.Arrays;
import java.util.function.Predicate;

/**
 * 方針の分かれ目ごとの代表の手。鳴くか、立直するか、槓するか、押すか降りるかのように、選ぶと後の局面が変わる分かれ目について、両側の手を
 * 必ず候補集合に入れる。O・s・p0 だけで決めるので、ほかの席の伏せた牌には依らない。
 *
 * <p>返す順は次のとおり。当てはまる分かれ目だけを返し、各組は「踏み込む側、踏み込まない側」の順に並べる。同じ手が重なっても除かない。同じ
 * 結果になる手をまとめるのは、呼び出し側の {@link com.epsilon.prism.collect.CandidateSet} の仕事。
 *
 * <ol>
 *   <li>鳴き: チーかポンが合法なら、チー・ポンの中で s が最大の手と、パス。
 *   <li>立直: 立直の打牌が合法なら、その中で s が最大の手と、ふつうの打牌の中で s が最大の手（ふつうの打牌がなければ省く）。
 *   <li>槓: 暗槓・加槓・大明槓のどれかが合法なら、その中で s が最大の手と、槓でない手の中で s が最大の手。
 *   <li>脅威: 公開情報で見た脅威が 1 つ以上あり、打牌（ふつうの打牌か立直の打牌）が合法なら、打牌の中で p0 が最大の手（牌効率の最善）と、
 *       {@link #safest}（公開情報だけで測った安全度の最善）。
 * </ol>
 *
 * s の最大は {@link Greedy#before} の比べ方で決める（同点は p0 の大きい手、それも同じなら番号の小さい手）。p0 の最大が同点なら番号の小さい手。
 *
 * <p>安全度の最善は次のように選ぶ。打牌ごとに (立直している相手のうち現物でない相手の数, 脅威の相手の危険度の最大, −p0, 合法手の位置) を
 * 辞書式に比べ、最小の打牌を選ぶ。現物と危険度は、O から求めた席ごとの公開情報（河と、立直の後に通った種類。字牌は見えている枚数）で決める。
 */
public final class RegimeRepresentatives {
  private static final Predicate<Action.Type> CALL =
      type -> type == Action.Type.CHI || type == Action.Type.PON;
  private static final Predicate<Action.Type> KAN =
      type ->
          type == Action.Type.ANKAN || type == Action.Type.KAKAN || type == Action.Type.DAIMINKAN;
  private static final Predicate<Action.Type> DISCARD =
      type -> type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI;

  /** 公開情報を求めるための、スレッドごとの作業領域。ルールが替わったら作り直す。 */
  private static final ThreadLocal<Tools> TOOLS = new ThreadLocal<>();

  private RegimeRepresentatives() {}

  /**
   * 分かれ目ごとの代表の手を選ぶ。
   *
   * @param o 判断する席の観測 O
   * @param legal 合法手の行動の番号（O の並び）
   * @param scores 合法手ごとの s
   * @param prior 合法手ごとの p0
   * @param threats 判断する席から見た、公開情報での脅威の数（{@link Threats#count}）
   * @return 代表の手の、合法手の中の位置（上に書いた順。当てはまる分かれ目がなければ空）
   */
  public static int[] of(Observation o, int[] legal, float[] scores, float[] prior, int threats) {
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
        chosen[size++] = safest(o, legal, prior);
      }
    }
    return Arrays.copyOf(chosen, size);
  }

  /**
   * 安全度が最も高い打牌を選ぶ。
   *
   * @param o 判断する席の観測 O
   * @param legal 合法手の行動の番号（O の並び）
   * @param prior 合法手ごとの p0
   * @return 選んだ打牌の、合法手の中の位置。打牌がなければ -1。脅威の相手がいなければ p0 が最大の打牌
   */
  public static int safest(Observation o, int[] legal, float[] prior) {
    Tools tools = tools(o);
    SeatFacts facts = tools.facts;
    int seats = o.variant().seats();
    boolean anyThreat = Threats.count(o, facts) > 0;
    int best = -1, bestUnsafeRiichi = 0, bestDanger = 0;
    for (int a = 0; a < legal.length; a++) {
      Action action = Action.of(legal[a]);
      if (!Twins.discard(action)) continue;
      int unsafeRiichi = 0, danger = 0;
      if (anyThreat) {
        int kind = action.tile();
        for (int seat = 1; seat < seats; seat++) {
          if (!Threats.isThreat(o, facts, seat)) continue;
          if (o.riichi(seat) != RiichiState.NONE && !Tiles.contains(facts.safe(seat), kind)) unsafeRiichi++;
          danger = Math.max(danger, facts.danger(seat, kind, tools.counts));
        }
      }
      // 辞書式に比べる: 現物でない立直者の数 → 危険度の最大 → p0 の大きい手 → 番号の小さい手（全て同じなら先の手を残す）
      if (best < 0
          || unsafeRiichi < bestUnsafeRiichi
          || (unsafeRiichi == bestUnsafeRiichi
              && (danger < bestDanger || (danger == bestDanger && prior[a] > prior[best])))) {
        best = a;
        bestUnsafeRiichi = unsafeRiichi;
        bestDanger = danger;
      }
    }
    return best;
  }

  /**
   * 種類が条件に合う手のうち、値が最大の手の位置を返す。比べ方は {@link Greedy#before} と同じ。同点なら同点用の値が大きい手を選び、それも
   * 同じか同点用の値が無ければ番号の小さい手を選ぶ。当てはまる手が無ければ -1。
   */
  private static int best(int[] legal, float[] values, float[] tieBreak, Predicate<Action.Type> member) {
    int best = -1;
    for (int a = 0; a < legal.length; a++)
      if (member.test(Action.of(legal[a]).type())
          && (best < 0 || Greedy.before(a, best, values, tieBreak))) best = a;
    return best;
  }

  /** 呼んだスレッドの作業領域に観測 O を読み込む。 */
  private static Tools tools(Observation o) {
    Tools tools = TOOLS.get();
    if (tools == null || tools.counts.variant() != o.variant()) {
      tools = new Tools(o);
      TOOLS.set(tools);
    }
    tools.facts.load(o);
    tools.counts.load(o);
    return tools;
  }

  private static final class Tools {
    final SeatFacts facts;
    final TileCounts counts;

    Tools(Observation o) {
      facts = new SeatFacts(o.variant());
      counts = new TileCounts(o.variant());
    }
  }
}
