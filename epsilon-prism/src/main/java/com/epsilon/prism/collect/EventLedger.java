package com.epsilon.prism.collect;

import com.epsilon.calculate.scoring.HandScoreBatchBuffer;
import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.calculate.shape.HandShapeAnalysisBuffer;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.DoraState;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.RoundResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.ScorePayments;
import com.epsilon.engine.WinClaim;
import com.epsilon.engine.WinLegality;
import com.epsilon.prism.core.CrnKeys;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.luck.LuckSink;
import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * 真の牌山の出来事の制御変量（運の精算）。ラベルの 1 枠（分岐の候補 a、または打った手の主経路。{@link Slot}）ごとに、判断 o の後から
 * 局末までの偶然の出来事の増分 δ を、ラベルの席 i から見た向きで帳簿 L_a（{@value #FEATURES} 欄、{@link #NAMES}）に足す。どの増分も、その出来事の
 * 直前までに全情報の状態で明らかになったこと F_k（4 人の手牌・河・副露・公開済みの表示牌・全ての手と硬貨の結果、網の出力 s と行動の分布 b）を
 * 条件として平均 0 なので（マルチンゲールの増分）、E[L_a | o] = 0。β を行と独立に決めれば Ỹ_a = Y_a − β·L_a も E[Ỹ_a | o] = Q(o, a)。
 *
 * <ul>
 *   <li>ツモ（生きた山・嶺上、誰のツモでも）: δ_f = c_f(x, d) − Σ_t n_t·c_f(x, t) / N。n_t は全員から見えない牌（4 − 公開済み − 4 人の手牌、{@link
 *       #pool}）で、x はツモの前の状態。牌山は一様な並べ替えで、開く位置（WASO の席別ツモ列を含む）は開いた履歴と牌山に独立な乱数だけで決まるので、
 *       まだ開いていない牌は交換可能で、引く牌はこのプールの上で一様（{@link GameState#reshuffleUnseen} と同じ集合）。
 *   <li>槓ドラの表示: 同じ形（表示牌の種類について。表示の前のプール）。
 *   <li>局末の裏ドラ（立直の和了があるときだけ）: 各枠の牌が周辺で一様なので、実現 − 枠の数 × プールの上の平均。
 *   <li>方策の偶然（o の後の強制でない判断）: δ = s̄_A − Σ_a b(a)·s̄_a（b は {@link CrnKeys#select} に実際に渡した分布、
 *       s̄ は s を ±{@value #POLICY_CLIP} で切ったもの）。硬貨は判断ごとに新しい鍵で、牌山と過去の硬貨と独立。
 * </ul>
 *
 * <p>厳密さの条件: c は F_k の関数でなければならない。裏ドラ（{@link DoraState#uraDoraTileTypesPacked}）・牌山の中身・未公開の
 * 表示牌は読まず（採点は常に includeUra = false。裏ドラの表示牌は局末の実現値にだけ使う）、境界の state はツモの後なので、ツモった席の手は引いた
 * 牌を外した形で読む。c は牌の種類だけの関数（引いた牌が赤かは使わない）。出来事は {@link GameRunner} の境界ごとにまとめて復元する（{@link
 * Tracker#observe}。表示は古い順、ツモはその後）。最後の境界の後に起きて局が終わる出来事は数えない（起きるかが牌に依らないので平均 0 は保つ）。
 *
 * <p>帳簿の計算が有限でない値を出したら {@link IllegalStateException} にする（行の NaN は「帳簿を付けなかった」ことだけを意味させる）。行に書く
 * ときは偏りのない確率的丸め（{@link #half}）で fp16 にする。
 */
public final class EventLedger {
  /** 帳簿の欄の数。 */
  public static final int FEATURES = 17;

  /**
   * 欄の名前（行に書く順）。並びは行の形式の一部で、コホートの目録（{@link CohortWriter.Manifest#ledgerFeatures}）は
   * 欄の数だけを持つ。
   */
  public static final List<String> NAMES =
      List.of(
          "ownImprove1",
          "ownImprove2",
          "ownImprove3",
          "ownTsumo",
          "ownForcedDealIn",
          "ownDrawnDanger",
          "oppImprove1",
          "oppImprove2",
          "oppTsumo",
          "oppForcedToSelf",
          "oppForcedToOther",
          "oppDrawnChance",
          "doraSelf",
          "doraOpponents",
          "ura",
          "policySelf",
          "policyOpponent");

  /** 報告の部分 R² の組: 欄の範囲 [from, to)。 */
  public record Group(String name, int from, int to) {}

  /** 組（ツモ・槓ドラ・裏ドラ・方策）。 */
  public static final List<Group> GROUPS =
      List.of(new Group("draw", 0, 12), new Group("dora", 12, 14), new Group("ura", 14, 15), new Group("policy", 15, 17));

  /** 点の項の上限（千点）。 */
  public static final double CLIP = 32;

  /** 方策の項の s の上限（効用の単位。|δ| ≤ 2·これ）。 */
  public static final double POLICY_CLIP = 8;

  static final int OWN_IMPROVE = 0,
      OWN_TSUMO = 3,
      OWN_FORCED_DEAL_IN = 4,
      OWN_DRAWN_DANGER = 5,
      OPP_IMPROVE = 6,
      OPP_TSUMO = 8,
      OPP_FORCED_TO_SELF = 9,
      OPP_FORCED_TO_OTHER = 10,
      OPP_DRAWN_CHANCE = 11,
      DORA_SELF = 12,
      DORA_OPPONENTS = 13,
      URA = 14,
      POLICY_SELF = 15,
      POLICY_OPPONENT = 16;

  private static final int TILES = Tile.NUM_TILE_TYPES, SEATS = GameState.NUM_PLAYERS;

  /** 表示牌の種類 → ドラの種類。 */
  private static final int[] DORA_OF = new int[TILES];

  static {
    for (int t = 0; t < TILES; t++) DORA_OF[t] = Tile.doraFrom(t);
  }

  private static final ThreadLocal<Tools> TOOLS = ThreadLocal.withInitial(Tools::new);

  private EventLedger() {}

  /** 進行ワーカーのスレッドごとの道具（解析器・採点器はスレッド安全でない）。 */
  static final class Tools {
    final HandShapeAnalyzer shapes = new HandShapeAnalyzer();
    final HandScoreEvaluator scores = new HandScoreEvaluator();

    /** ツモった席のツモの前の形。 */
    final HandShapeState before = new HandShapeState();

    /** 席の表の stamp を作る写し。 */
    final HandShapeState scratch = new HandShapeState();

    final HandShapeAnalysisBuffer analysis = new HandShapeAnalysisBuffer();
    final HandScoreBatchBuffer batch = new HandScoreBatchBuffer(1);
    final HandScoreBuffer score = new HandScoreBuffer();
    final int[] pool = new int[TILES];

    /** 席ごとの所持（手牌と副露）の枚数。 */
    final int[][] owned = new int[SEATS][TILES];

    /** 席ごとの、この境界の増分。 */
    final double[][] delta = new double[SEATS][FEATURES];

    /** 種類ごとの c の値（マスクの上だけ書いて読む）。 */
    final double[] values = new double[TILES];

    /** ツモの和了の基本点（ツモのマスクの上だけ）。 */
    final int[] tsumoBase = new int[TILES];

    /** このツモで使う席の表（{@link Tracker} の表への参照。中身は変えない）。 */
    final SeatTable[] seen = new SeatTable[SEATS];
  }

  /**
   * 席 w の栄和の表（作った後に変えない。写しと分岐は親と共有する）。stamp は表が依る状態のハッシュ（{@link Tracker#stamp}）、tenpai は 13 枚が
   * 聴牌か、ron は w が栄和できる種類のマスク（役・振聴・五枚目を含む {@link WinLegality#canRon}）、ronBase はその基本点（裏を数えない。ron の外は
   * 0）。
   */
  record SeatTable(long stamp, boolean tenpai, long ron, int[] ronBase) {}

  /**
   * 境界で復元した出来事（試験の {@link Tracker#trace} だけ）。
   *
   * @param draw ツモ（嶺上を含む）なら true、槓ドラの表示なら false
   * @param seat ツモった席（表示は -1）
   * @param tile 引いた牌か表示牌の種類
   * @param pool 出来事の前のプール
   * @param hands 出来事の前の 4 席の手牌（副露・暗槓を除く）の枚数
   */
  record Event(boolean draw, int seat, int tile, int[] pool, int[][] hands) {}

  /**
   * 牌の種類ごとの、全員から見えない枚数 n_t = 4 − 公開済み − 4 人の手牌（{@link GameState#reshuffleUnseen} が並べ替える集合と同じ）。0 未満なら
   * 数え方の誤りなので例外。
   *
   * @return N = Σ n_t
   */
  static int pool(GameState state, int[] pool) {
    int total = 0;
    for (int tile = 0; tile < TILES; tile++) {
      int left = Tile.TILES_PER_TYPE - state.publicState().visibleTileCount(tile);
      for (int seat = 0; seat < SEATS; seat++) left -= state.hand(seat).count(tile);
      if (left < 0) throw new IllegalStateException("Negative unseen count " + left + " of tile " + tile);
      pool[tile] = left;
      total += left;
    }
    return total;
  }

  /** 千点に直して ±{@value #CLIP} で切った点。 */
  static double kappa(int points) {
    return Math.max(-CLIP, Math.min(CLIP, points / 1000.0));
  }

  /** ±{@value #POLICY_CLIP} で切った s。 */
  static double clipScore(float score) {
    return Math.max(-POLICY_CLIP, Math.min(POLICY_CLIP, score));
  }

  /** 増分を足す。有限でなければ帳簿の誤りなので例外（黙って NaN を行に入れない）。 */
  static void add(double[] delta, int feature, double value) {
    if (!Double.isFinite(value))
      throw new IllegalStateException("Non-finite ledger term " + NAMES.get(feature) + " = " + value);
    delta[feature] += value;
  }

  /**
   * 方策の項を足すか: b(a) &gt; 0 の手（{@code p > 0}。{@link CrnKeys#select} と同じ判定で、NaN・0・負の手は重み 0）の s が
   * 全て有限で、重みの和が正。足すかは硬貨の前に決まる。
   */
  static boolean policyCounts(float[] scores, float[] p, int n) {
    double sum = 0;
    for (int a = 0; a < n; a++) {
      float probability = p[a];
      if (!(probability > 0)) continue;
      if (!Float.isFinite(scores[a])) return false;
      sum += probability;
    }
    return sum > 0;
  }

  /**
   * 方策の項 δ = s̄_A − Σ_a b(a)·s̄_a（b(a) = w_a / Σ w、w_a = p_a &gt; 0 ? p_a : 0。s̄ は s を ±{@value #POLICY_CLIP} で切ったもの）。{@link
   * #policyCounts} が真のときだけ呼ぶ。
   */
  static double policyDelta(float[] scores, float[] p, int chosen, int n) {
    double sum = 0, weighted = 0;
    for (int a = 0; a < n; a++) {
      float probability = p[a];
      double weight = probability > 0 ? probability : 0;
      if (weight == 0) continue;
      sum += weight;
      weighted += weight * clipScore(scores[a]);
    }
    return clipScore(scores[chosen]) - weighted / sum;
  }

  /**
   * 偏りのない確率的丸め（fp16 でも平均 0）: x を fp16 で表せる隣り合う 2 つの値 low ≤ x &lt; low + ulp に、確率 (x − low)/ulp で上へ、残りで下へ
   * 丸める（E[結果 | x] = x）。0 と NaN はそのまま。|x| ≥ 65504 は例外（帳簿は有界で起きない）。
   *
   * @param unit [0, 1) の一様な硬貨（{@link #coin}）
   */
  static float half(double x, double unit) {
    if (x == 0 || Double.isNaN(x)) return (float) x;
    if (!(Math.abs(x) < 65504))
      throw new IllegalStateException("Ledger value outside the fp16 range: " + x);
    int exponent = Math.max(Math.getExponent(Math.abs(x)), -14);
    double ulp = Math.scalb(1.0, exponent - 10);
    double low = Math.floor(x / ulp) * ulp;
    double fraction = (x - low) / ulp;
    return (float) (unit < fraction ? low + ulp : low);
  }

  /** 丸めの硬貨 unit(derive(コホートの種, LEDGER, 半荘, 判断, 手·{@value #FEATURES} + 欄))。牌山と手の硬貨と独立。 */
  static double coin(long iterationSeed, long gameId, long decisionId, int action, int feature) {
    return CrnSeeds.unit(
        CrnSeeds.derive(
            iterationSeed, CrnSeeds.Domain.LEDGER, gameId, decisionId, (long) action * FEATURES + feature));
  }

  /**
   * 1 本の経路（{@link GameRunner}）の帳簿の計算器。1 度に 1 スレッドだけが触る。席の表（{@link SeatTable}）を stamp で使い回し、写しと分岐は
   * 親の表の配列を写して始める（表そのものは作った後に変えないので、分岐が進んで作った表が親に混ざらない）。表は経路の過去の状態だけの関数なので、
   * ハッシュの衝突で古い表を使っても平均 0 は崩れず、結果は実行順・ワーカー数に依らない。運の網の標本を集める収集では、帳簿を付けなくても出来事を
   * 復元するために作る（{@link #observe} の luck。枠に帳簿がなければ帳簿の計算はしない）。
   */
  static final class Tracker {
    private final SeatTable[] tables;

    /** 前の境界で見た表示牌の数（-1 は未知。最初の境界で数えずに合わせる）。 */
    private int last;

    /** 計測（{@link StageTimer.Stage#LEDGER}。{@link #flush} で 1 回だけ足す）。 */
    private long nanos, calls;

    private List<Event> trace;

    /** 主経路。 */
    Tracker() {
      tables = new SeatTable[SEATS];
      last = -1;
    }

    /** 写しか分岐。state は親の commit の前の状態（o の境界）。 */
    Tracker(Tracker parent, GameState state) {
      tables = parent.tables.clone();
      last = state.doraState().indicatorCount();
    }

    /** 復元した出来事を events に足す（試験だけ。本番は null）。 */
    void trace(List<Event> events) {
      trace = events;
    }

    /** 主経路が次局へ移る（表示の数を未知に戻し、席の表を捨てる）。 */
    void reset() {
      last = -1;
      Arrays.fill(tables, null);
    }

    /** 計測した時間と呼び出しの数を timer に足して空にする（{@link GameRunner#advance} の終わりに 1 回）。 */
    void flush(StageTimer timer) {
      if (calls == 0) return;
      timer.add(StageTimer.Stage.LEDGER, nanos, calls);
      nanos = 0;
      calls = 0;
    }

    /**
     * 境界ごとに 1 回（{@link GameRunner} の prepare）。前の境界からの出来事（新しい表示牌は古い順、TURN の境界でツモの直後ならそのツモ）を、
     * 出来事の前の状態に戻して計算し、枠の席ごとに増分を求めて、その席の全ての枠の帳簿に足す。枠のない進行では表示の数だけを合わせる。
     *
     * @param luck 復元した出来事を同じ順に受け取る運の網の標本の捕捉（標本を集めない収集は null）。帳簿の枠の有無より前に渡すので、標本は
     *     枠の有無で選ばれない
     */
    void observe(GameState state, List<EngineDecisionPoint> points, List<Slot> targets, LuckSink luck) {
      int count = state.doraState().indicatorCount();
      int fresh = last < 0 ? 0 : count - last;
      last = count;
      if (fresh < 0) throw new IllegalStateException("Dora indicators went back to " + count);
      TurnEvent.Draw draw = freshDraw(state, points);
      if (luck != null) {
        for (int k = count - fresh; k < count; k++) luck.reveal(state, k, draw);
        if (draw != null) luck.draw(state, draw);
      }
      int seats = seats(targets);
      if (seats == 0) return;
      if (draw == null && fresh == 0) return;
      long started = System.nanoTime();
      Tools tools = TOOLS.get();
      int drawer = draw == null ? -1 : draw.player(), drawn = draw == null ? -1 : draw.tileType();
      int[] pool = tools.pool;
      pool(state, pool);
      // ツモの前のプール（引いた牌を戻す）
      if (draw != null) pool[drawn]++;
      double[][] delta = tools.delta;
      for (double[] row : delta) Arrays.fill(row, 0);
      if (fresh > 0) {
        owned(state, drawer, drawn, tools.owned);
        // 最初の新しい表示の前のプール（新しい表示牌を全て戻す）
        for (int k = count - fresh; k < count; k++) pool[state.doraState().indicatorTileType(k)]++;
        for (int k = count - fresh; k < count; k++) {
          int indicator = state.doraState().indicatorTileType(k);
          if (trace != null) trace.add(new Event(false, -1, indicator, pool.clone(), hands(state, drawer, drawn)));
          reveal(pool, indicator, tools.owned, seats, delta);
          pool[indicator]--;
        }
      }
      if (draw != null) {
        if (trace != null) trace.add(new Event(true, drawer, drawn, pool.clone(), hands(state, drawer, drawn)));
        draw(state, tools, draw, pool, seats, delta);
      }
      apply(delta, targets, 0, URA);
      nanos += System.nanoTime() - started;
      calls++;
    }

    /**
     * o の後の強制でない判断（{@link GameRunner} の commit の手を選ぶ輪、record より前）。s がない判断（inline の相手）と、b(a) &gt; 0 の手の s が
     * 有限でない判断は足さない（{@link #policyCounts}）。枠の席 = 判断の席なら policySelf、ほかは policyOpponent。
     *
     * @param p {@link CrnKeys#select} に渡した分布
     * @param chosen 選んだ手の位置
     */
    void policy(int decider, float[] scores, float[] p, int chosen, int n, List<Slot> targets) {
      if (scores == null) return;
      int seats = seats(targets);
      if (seats == 0) return;
      long started = System.nanoTime();
      if (policyCounts(scores, p, n)) {
        double value = policyDelta(scores, p, chosen, n);
        double[][] delta = TOOLS.get().delta;
        for (int seat = 0; seat < SEATS; seat++) {
          if ((seats & (1 << seat)) == 0) continue;
          Arrays.fill(delta[seat], 0);
          add(delta[seat], seat == decider ? POLICY_SELF : POLICY_OPPONENT, value);
        }
        apply(delta, targets, POLICY_SELF, FEATURES);
      }
      nanos += System.nanoTime() - started;
      calls++;
    }

    /**
     * 局末（{@link GameRunner} の endHand の最初、次局を始める前）: 立直の和了者 w ごとに、和了の手のドラの数え g_w(t) = own_w(doraFrom(t)) の
     * 裏ドラの枠 k 個の実現 Σ g_w(裏表示牌) から、期待 k·Σ_t n_t·g_w(t)/N を引き、席 i から見た重み ω_i(w)（i = w なら +1、栄和の放銃者 −1、
     * ツモは子の和了で i が親なら −1/2・子なら −1/4、親の和了なら −1/3）を掛けて ura に足す。手牌が 13 枚なら手番のイベントの牌（栄和の牌）を
     * 足し、合わなければその w の項は足さない（足すかは裏を開く前に決まる）。
     */
    void handEnd(GameState state, RoundSettlement settlement, List<Slot> targets) {
      if (!(settlement.result() instanceof RoundResult.Winning winning) || !winning.hasRiichiWinner()) return;
      int seats = seats(targets);
      if (seats == 0) return;
      long started = System.nanoTime();
      Tools tools = TOOLS.get();
      int[] pool = tools.pool;
      int total = pool(state, pool);
      if (total <= 0) throw new IllegalStateException("Empty pool at the hand end");
      int indicators = state.doraState().indicatorCount(), dealer = state.getOya();
      double[][] delta = tools.delta;
      for (double[] row : delta) Arrays.fill(row, 0);
      int[] counts = tools.owned[0];
      for (WinClaim claim : winning.claims()) {
        if (!claim.hasRiichi()) continue;
        int winner = claim.winner();
        if (!winningHand(state, winner, counts)) continue;
        double realized = 0, expected = 0;
        for (int slot = 0; slot < indicators; slot++)
          realized += counts[DORA_OF[Tile.typeOf(state.uraDoraIndicatorPhysicalTileId(slot))]];
        for (int tile = 0; tile < TILES; tile++) expected += pool[tile] * counts[DORA_OF[tile]];
        double term = realized - indicators * expected / total;
        for (int seat = 0; seat < SEATS; seat++)
          if ((seats & (1 << seat)) != 0) add(delta[seat], URA, uraWeight(claim, seat, dealer) * term);
      }
      apply(delta, targets, URA, URA + 1);
      nanos += System.nanoTime() - started;
      calls++;
    }

    /** ツモの出来事（ツモった席 j、引いた種類 d、ツモの前のプール pool）の 0〜11 の増分を、枠の席ごとに delta へ足す。 */
    private void draw(
        GameState state, Tools tools, TurnEvent.Draw draw, int[] pool, int seats, double[][] delta) {
      int drawer = draw.player(), drawn = draw.tileType();
      int total = 0;
      for (int count : pool) total += count;
      if (total <= 0) throw new IllegalStateException("Empty pool before a draw");
      Hand hand = state.hand(drawer);
      hand.copyShapeInto(tools.before);
      tools.before.remove(drawn);
      tools.shapes.analyzeInto(tools.before, tools.analysis);
      int shanten = tools.analysis.minimumShanten();
      long accepting = tools.analysis.ukeireTileTypeMask();
      // ツモの和了（聴牌のときだけ）: 採点はツモの前の形に t を足す。赤の所持はツモの前（引いた赤は外す）。プールにない種類（4 枚とも
      // 見えているか誰かの手にある。副露を含めると 5 枚目になることもある）は引けず増分に効かないので採点しない
      long tsumo = 0;
      if (shanten == 0) {
        int aka = hand.ownedAkaMask();
        if (draw.isAkaTile()) aka = AkaTileMask.excludeTile(aka, drawn);
        tools.batch.configure(0, WinMethod.TSUMO, WinConditions.fromGameState(state, drawer, draw), aka).size(1);
        for (long bits = accepting; bits != 0; bits &= bits - 1) {
          int tile = Long.numberOfTrailingZeros(bits);
          if (pool[tile] == 0) continue;
          tools.scores.scoreAfterAddingBatch(tools.before, hand, tile, state.doraState(), false, tools.batch);
          HandScoreBuffer result = tools.batch.result(0);
          if (!result.available()) continue;
          tsumo |= 1L << tile;
          tools.tsumoBase[tile] = result.basePoints();
        }
      }
      boolean riichi = state.isRiichi(drawer);
      int dealer = state.getOya(), honba = state.getHonba(), sticks = state.getKyotakuCount();
      // 栄和の表: ツモった席が枠の席なら（放銃）全ての他家、ツモった席が立直なら（ほかの席への放銃）全ての他家、ほかは枠の席だけ
      boolean everyone = (seats & (1 << drawer)) != 0 || riichi;
      SeatTable[] seen = tools.seen;
      for (int seat = 0; seat < SEATS; seat++)
        seen[seat] = seat != drawer && (everyone || (seats & (1 << seat)) != 0) ? table(state, tools, seat) : null;
      double[] values = tools.values;
      for (int seat = 0; seat < SEATS; seat++) {
        if ((seats & (1 << seat)) == 0) continue;
        double[] row = delta[seat];
        if (seat == drawer) {
          if (shanten >= 1)
            add(row, OWN_IMPROVE + Math.min(shanten, 3) - 1, indicator(accepting, drawn, pool, total));
          if (tsumo != 0) {
            for (long bits = tsumo; bits != 0; bits &= bits - 1) {
              int tile = Long.numberOfTrailingZeros(bits);
              values[tile] = kappa(tsumoGain(tools.tsumoBase[tile], drawer, dealer, honba, sticks));
            }
            add(row, OWN_TSUMO, centred(tsumo, values, drawn, pool, total));
          }
          // 引いた牌を打てば他家に栄和される（ツモの和了の牌は除く）
          long ron = 0;
          for (int other = 0; other < SEATS; other++) if (other != drawer) ron |= seen[other].ron();
          ron &= ~tsumo;
          if (ron != 0) {
            for (long bits = ron; bits != 0; bits &= bits - 1) {
              int tile = Long.numberOfTrailingZeros(bits);
              double loss = 0;
              for (int other = 0; other < SEATS; other++)
                if (other != drawer && (seen[other].ron() & (1L << tile)) != 0)
                  loss += kappa(ronPayment(seen[other].ronBase()[tile], other, dealer, honba));
              values[tile] = -loss;
            }
            add(row, riichi ? OWN_FORCED_DEAL_IN : OWN_DRAWN_DANGER, centred(ron, values, drawn, pool, total));
          }
        } else {
          if (shanten >= 1)
            add(row, OPP_IMPROVE + (shanten == 1 ? 0 : 1), indicator(accepting, drawn, pool, total));
          if (tsumo != 0) {
            for (long bits = tsumo; bits != 0; bits &= bits - 1) {
              int tile = Long.numberOfTrailingZeros(bits);
              values[tile] = -kappa(tsumoPayment(tools.tsumoBase[tile], drawer, seat, dealer, honba));
            }
            add(row, OPP_TSUMO, centred(tsumo, values, drawn, pool, total));
          }
          SeatTable own = seen[seat];
          long gain = own.ron() & ~tsumo;
          if (gain != 0) {
            for (long bits = gain; bits != 0; bits &= bits - 1) {
              int tile = Long.numberOfTrailingZeros(bits);
              values[tile] = kappa(ronPayment(own.ronBase()[tile], seat, dealer, honba) + 1000 * sticks);
            }
            add(row, riichi ? OPP_FORCED_TO_SELF : OPP_DRAWN_CHANCE, centred(gain, values, drawn, pool, total));
          }
          if (riichi) {
            long ron = 0;
            for (int other = 0; other < SEATS; other++)
              if (other != drawer && other != seat) ron |= seen[other].ron();
            ron &= ~tsumo;
            if (ron != 0) {
              for (long bits = ron; bits != 0; bits &= bits - 1) {
                int tile = Long.numberOfTrailingZeros(bits);
                double paid = 0;
                for (int other = 0; other < SEATS; other++)
                  if (other != drawer && other != seat && (seen[other].ron() & (1L << tile)) != 0)
                    paid += kappa(ronPayment(seen[other].ronBase()[tile], other, dealer, honba));
                values[tile] = paid;
              }
              add(row, OPP_FORCED_TO_OTHER, centred(ron, values, drawn, pool, total));
            }
          }
        }
      }
    }

    /**
     * 席 w の栄和の表。stamp が前の表と同じなら使い回し、違えば新しい表を作って席の枠に入れる（前の表は変えない）。表は打った席に依らない
     * （{@link WinLegality#canRon} の判定と採点は打った席を読まないので、打った席は w の下家に固定する）。
     */
    private SeatTable table(GameState state, Tools tools, int seat) {
      long stamp = stamp(state, tools, seat);
      SeatTable table = tables[seat];
      if (table != null && table.stamp() == stamp) return table;
      Hand hand = state.hand(seat);
      tools.shapes.analyzeInto(hand, tools.analysis);
      boolean tenpai = tools.analysis.minimumShanten() == 0;
      long ron = 0;
      int[] base = new int[TILES];
      if (tenpai) {
        int discarder = (seat + 1) % SEATS;
        for (long bits = tools.analysis.agariTileTypeMask(); bits != 0; bits &= bits - 1) {
          int tile = Long.numberOfTrailingZeros(bits);
          var discard = new TurnEvent.Discard(discarder, tile, false);
          if (!WinLegality.canRon(state, seat, discard, tools.scores, tools.shapes)) continue;
          if (!tools.scores.scoreAfterAdding(
              hand,
              tile,
              WinMethod.RON,
              WinConditions.fromGameState(state, seat, discard),
              state.doraState(),
              false,
              hand.ownedAkaMask(),
              tools.score)) continue;
          ron |= 1L << tile;
          base[tile] = tools.score.basePoints();
        }
      }
      table = new SeatTable(stamp, tenpai, ron, base);
      tables[seat] = table;
      return table;
    }

    /**
     * 席 w の表が依る状態のハッシュ: 手牌の枚数・副露・赤の所持・立直（ダブルを含む）・一発・同巡の振聴・河の種類（振聴）・山の尽き（河底）・
     * 第一巡（和了の条件）・ドラ表示牌。
     */
    private static long stamp(GameState state, Tools tools, int seat) {
      Hand hand = state.hand(seat);
      hand.copyShapeInto(tools.scratch);
      long flags =
          (state.isRiichi(seat) ? 1 : 0)
              | (state.isDoubleRiichi(seat) ? 2 : 0)
              | (state.isIppatsu(seat) ? 4 : 0)
              | (state.isTemporaryFuriten(seat) ? 8 : 0)
              | (state.isWallExhausted() ? 16 : 0)
              | (state.isFirstDraw(seat) ? 32 : 0)
              | (state.isFirstTurnCallOccurred() ? 64 : 0)
              | ((long) state.doraState().indicatorCount() << 8)
              | ((long) hand.ownedAkaMask() << 16)
              | ((long) tools.scratch.meldCount() << 24);
      long h = SeedMixer.mix64(tools.scratch.packedConcealedTileCountsLow());
      h = SeedMixer.mix64(h ^ tools.scratch.packedConcealedTileCountsHigh());
      h = SeedMixer.mix64(h ^ hand.canonicalMeldSignature());
      h = SeedMixer.mix64(h ^ flags);
      h = SeedMixer.mix64(h ^ state.publicState().riverTileTypeMask(seat));
      return SeedMixer.mix64(h ^ state.doraState().doraTileTypesPacked());
    }
  }

  /** 枠の席のビット（帳簿を持つ枠の判断の席）。 */
  private static int seats(List<Slot> targets) {
    int seats = 0;
    for (Slot slot : targets)
      if (slot.ledger() != null && slot.target().seat() >= 0) seats |= 1 << slot.target().seat();
    return seats;
  }

  /** 席ごとの増分の欄 [from, to) を、その席の全ての枠の帳簿に足す。 */
  private static void apply(double[][] delta, List<Slot> targets, int from, int to) {
    for (Slot slot : targets) {
      double[] ledger = slot.ledger();
      int seat = slot.target().seat();
      if (ledger == null || seat < 0) continue;
      double[] row = delta[seat];
      for (int feature = from; feature < to; feature++) ledger[feature] += row[feature];
    }
  }

  /** TURN の境界でツモの直後なら、そのツモ（境界の手番のイベントが判断の席の Draw で、牌が分かる）。ほかは null。 */
  private static TurnEvent.Draw freshDraw(GameState state, List<EngineDecisionPoint> points) {
    if (points.size() != 1) return null;
    EngineDecisionPoint point = points.getFirst();
    if (point.kind() != EngineDecisionKind.TURN) return null;
    return state.turnEvent() instanceof TurnEvent.Draw draw
            && draw.player() == point.player()
            && draw.tileType() >= 0
        ? draw
        : null;
  }

  /** 席ごとの所持（手牌と副露）の枚数。drawer ≥ 0 ならその席の手から引いた牌 drawn を外す（ツモの前）。 */
  private static void owned(GameState state, int drawer, int drawn, int[][] owned) {
    for (int seat = 0; seat < SEATS; seat++) {
      Hand hand = state.hand(seat);
      hand.copyConcealedTileCountsTo(owned[seat]);
      if (seat == drawer) owned[seat][drawn]--;
      addMelds(hand, owned[seat]);
    }
  }

  private static void addMelds(Hand hand, int[] counts) {
    for (int m = 0; m < hand.meldCount(); m++) {
      Meld meld = hand.meld(m);
      for (int k = 0; k < meld.size(); k++) counts[meld.tileAt(k)]++;
    }
  }

  /** 出来事の前の 4 席の手牌（副露・暗槓を除く）の枚数（試験の記録だけ）。 */
  private static int[][] hands(GameState state, int drawer, int drawn) {
    int[][] hands = new int[SEATS][TILES];
    for (int seat = 0; seat < SEATS; seat++) {
      state.hand(seat).copyConcealedTileCountsTo(hands[seat]);
      if (seat == drawer) hands[seat][drawn]--;
    }
    return hands;
  }

  /**
   * 槓ドラの表示の増分（表示の前のプール pool、表示牌の種類 indicator）: doraSelf は own_i(doraFrom(u)) の、doraOpponents は Σ_{w≠i}
   * own_w(doraFrom(u)) の、プールの上の平均からのずれ。
   */
  private static void reveal(int[] pool, int indicator, int[][] owned, int seats, double[][] delta) {
    int total = 0;
    for (int count : pool) total += count;
    if (total <= 0) throw new IllegalStateException("Empty pool before a dora reveal");
    double[] deviation = new double[SEATS];
    int dora = DORA_OF[indicator];
    for (int seat = 0; seat < SEATS; seat++) {
      double expected = 0;
      for (int tile = 0; tile < TILES; tile++) expected += pool[tile] * owned[seat][DORA_OF[tile]];
      deviation[seat] = owned[seat][dora] - expected / total;
    }
    for (int seat = 0; seat < SEATS; seat++) {
      if ((seats & (1 << seat)) == 0) continue;
      double others = 0;
      for (int other = 0; other < SEATS; other++) if (other != seat) others += deviation[other];
      add(delta[seat], DORA_SELF, deviation[seat]);
      add(delta[seat], DORA_OPPONENTS, others);
    }
  }

  /** 指示関数の増分 [d ∈ mask] − Σ_{t∈mask} n_t / N。 */
  private static double indicator(long mask, int drawn, int[] pool, int total) {
    int live = 0;
    for (long bits = mask; bits != 0; bits &= bits - 1) live += pool[Long.numberOfTrailingZeros(bits)];
    return ((mask >>> drawn) & 1) - (double) live / total;
  }

  /** c が mask の上だけ values で 0 でない関数の増分 c(d) − Σ_{t∈mask} n_t·c(t) / N。 */
  private static double centred(long mask, double[] values, int drawn, int[] pool, int total) {
    double expected = 0;
    for (long bits = mask; bits != 0; bits &= bits - 1) {
      int tile = Long.numberOfTrailingZeros(bits);
      expected += pool[tile] * values[tile];
    }
    double realized = ((mask >>> drawn) & 1) != 0 ? values[drawn] : 0;
    return realized - expected / total;
  }

  /** ツモの和了者 j の得（{@link ScorePayments}。包は無視）: 3 人の支払い + 300·本場 + 1000·供託。 */
  static int tsumoGain(int base, int winner, int dealer, int honba, int sticks) {
    int paid =
        winner == dealer
            ? 3 * ScorePayments.tsumoFromChild(base, true)
            : ScorePayments.tsumoFromDealer(base) + 2 * ScorePayments.tsumoFromChild(base, false);
    return paid + 300 * honba + 1000 * sticks;
  }

  /** ツモで payer が払う点 + 100·本場。 */
  static int tsumoPayment(int base, int winner, int payer, int dealer, int honba) {
    int paid =
        winner == dealer
            ? ScorePayments.tsumoFromChild(base, true)
            : payer == dealer ? ScorePayments.tsumoFromDealer(base) : ScorePayments.tsumoFromChild(base, false);
    return paid + 100 * honba;
  }

  /** 栄和で放銃者が払う点（和了者 winner）+ 300·本場。和了者の得はこれに 1000·供託を足したもの。 */
  static int ronPayment(int base, int winner, int dealer, int honba) {
    return ScorePayments.ronPoints(base, winner == dealer) + 300 * honba;
  }

  /** 裏ドラの項の席 seat から見た重み ω（和了 claim、親 dealer）。 */
  static double uraWeight(WinClaim claim, int seat, int dealer) {
    int winner = claim.winner();
    if (seat == winner) return 1;
    if (claim instanceof WinClaim.Ron) return seat == claim.from() ? -1 : 0;
    if (winner == dealer) return -1.0 / 3;
    return seat == dealer ? -0.5 : -0.25;
  }

  /**
   * 和了者 w の和了の手の所持（手牌と副露）の枚数。14 枚ならそのまま、13 枚なら手番のイベントの牌（栄和・搶槓の牌）を足す。合わなければ false。
   */
  private static boolean winningHand(GameState state, int winner, int[] counts) {
    Hand hand = state.hand(winner);
    hand.copyConcealedTileCountsTo(counts);
    addMelds(hand, counts);
    int effective = hand.effectiveTileCount();
    if (effective == 14) return true;
    if (effective == 13
        && state.turnEvent() instanceof TurnEvent.ResponseSource source
        && source.tileType() >= 0) {
      counts[source.tileType()]++;
      return true;
    }
    return false;
  }

  // 測定の付録: システムプロパティ prism.ledger に出力のパスがあり、帳簿が働くときだけ、ラベル行の測った手ごとに 1 行の TSV を書く。

  private static final String PROPERTY = "prism.ledger";
  private static final String DUMP_PATH = System.getProperty(PROPERTY);

  /** TSV を書くか。 */
  static boolean dumping() {
    return DUMP_PATH != null && !DUMP_PATH.isBlank();
  }

  /** ラベル行がそろったら、測った手ごとに 1 行（{@link #dumpLines}）を書く。 */
  static void dump(RowDraft draft, float[] values, float[] ledger) {
    String lines = dumpLines(draft, values, ledger);
    Writer writer = Dump.WRITER;
    synchronized (Dump.class) {
      try {
        writer.write(lines);
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
    }
  }

  /** TSV の見出しの行（{@link #dumpLines} の欄の名前）。 */
  static String dumpHeader() {
    return "game\tdecision\tseat\tclass\tkind\tsource\tplayed\taction\ty\t" + String.join("\t", NAMES) + "\n";
  }

  /** 測った手ごとに 1 行（半荘・判断・席・類・種別・行の種類・打った手・手・Y・帳簿の欄。値は行に書くのと同じ丸めた値）。 */
  static String dumpLines(RowDraft draft, float[] values, float[] ledger) {
    StringBuilder lines = new StringBuilder();
    for (int action = 0; action < values.length; action++) {
      if (Float.isNaN(values[action])) continue;
      lines
          .append(draft.gameId())
          .append('\t')
          .append(draft.decisionId())
          .append('\t')
          .append(draft.seat())
          .append('\t')
          .append(draft.decisionClass())
          .append('\t')
          .append(draft.kind())
          .append('\t')
          .append(draft.source())
          .append('\t')
          .append(draft.played())
          .append('\t')
          .append(action)
          .append('\t')
          .append(values[action]);
      for (int feature = 0; feature < FEATURES; feature++)
        lines.append('\t').append(ledger[action * FEATURES + feature]);
      lines.append('\n');
    }
    return lines.toString();
  }

  /** TSV の書き手（初めて書くときに開く）。 */
  private static final class Dump {
    static final Writer WRITER = open();

    private static Writer open() {
      try {
        Writer writer = Files.newBufferedWriter(Path.of(DUMP_PATH), StandardCharsets.UTF_8);
        writer.write(dumpHeader());
        Runtime.getRuntime()
            .addShutdownHook(
                new Thread(
                    () -> {
                      synchronized (Dump.class) {
                        try {
                          writer.close();
                        } catch (IOException ignored) {
                          // 終了時の閉じ損ないは記録しない
                        }
                      }
                    }));
        return writer;
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
    }
  }
}
