package com.epsilon.prism.learn;

import com.epsilon.core.Action;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.model.PrismModel;
import com.epsilon.prism.rows.PrismRow;
import com.epsilon.prism.rows.RowSource;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * 交差した既知傾向の認証（論文 3.5 S、命題 10）。生徒 A の乗り換えの 1 次の改善 Î_A を、組 B の今のコホートの根の記録（ROOT）から Horvitz–Thompson
 * で推定する。Î_B は組 A の根と生徒 B で同じ。生徒は自分の組の行だけで学ぶので、相手の組の結果と独立。
 *
 * <p>根は、b が貪欲の手 π_t(o) を打った主経路の判断だけにある（床 ε_b で別の手を打った判断は収集で根を作らない。{@code GameRunner}）。根 r（判断
 * o_r、打った手 b_r = played ≡ π_t(o_r)、逸脱 d_r、q_r = q(d_r|o_r)、傾向 p_r = β_c·q_r、y_main = Y[played]、y_br = Y[d_r]）と生徒 X について
 *
 * <pre>
 * T_X(o) = argmax_{合法手} s_X(o, ·)                 （{@link Greedy#argmax}: 同点は p0 の大きい手、それも同じなら番号の小さい手）
 * hit_r  = 1{T_X(o_r) が played と同じ手でも双子でもなく、d_r と同じ手か双子}       （{@link com.epsilon.prism.core.Twins#same}）
 * term_r = hit_r · (y_br − y_main) / p_r
 * w_r    = 1/β_c = q_r / p_r                         （根 1 つが代表する判断の数。行の deviationProbability / propensity）
 * Î_X,c  = Σ_{r ∈ 相手の組, 類 c} term_r / Σ_r w_r
 * </pre>
 *
 * E[term_r | o_r] = β_c·Σ_d q(d)·1{d ≡ T}·ΔQ/(β_c·q(d)) = 1{T ≢ π_t(o)}·(Q_b(o, T) − Q_b(o, π_t(o)))、E[w_r] の和は根を置けた判断の数なので、
 * Î は判断 1 つあたりの 1 次の改善（効用の単位）の比の推定で、certifyTolerance も判断あたりで読める。類の中では β が一定なので Î_c = Σ term·β_c / 根の
 * 数（= Σ hit·Δ/q / 根の数）。全体（all）と確約の類の合算は、β の違う類を判断の数で重み付けした平均になる。
 *
 * <p>標準誤差は半荘で束ねる: 半荘 g の和 T_g = Σ term、重み n_g = Σ w として Î = ΣT_g / Σn_g、se² = G/(G−1)·Σ_g (T_g − Î·n_g)² / (Σn_g)²（G は
 * 半荘の数。G &lt; 2 なら NaN）。類ごと・全体・確約の類を合わせたもの（{@link com.epsilon.prism.core.DecisionClass#commitment()}）を報告する。
 *
 * <p>判定の量: 確約の類の Ī = (Î_A,commit + Î_B,commit)/2、se = √(se_A² + se_B²)/2、下限 = Ī − 2·se。どちらかの組に確約の根がなければ NaN
 * （判定は不合格）。
 *
 * <p>q の台は C(o) なので、T_X(o) が C(o) の外の乗り換えは hit が起こりえず、その改善（悪化）は Î に入らない（0 に偏る）。そこで確約の類の乗り換えの
 * うち C の外だった割合（両方の生徒の和。{@link Report#outsideShare}）が上限を超えたら不合格にする（{@link Report#passes}）。行の support（{@link
 * PrismRow#support}）と双子で比べる。行の合法手は payload を復号した入力の legalActionId から {@code Action.fromIndex} で得る。
 *
 * <p>T_X(o) が played の双子なら生徒は b と同じ手を打つので、乗り換えではなく改善は 0 とする（候補集合は双子をまとめ played の双子を played に置き換える
 * ので、d が played の双子になることはない）。d = played の根は写しがなく term は 0 だが、重みは数える。
 */
public final class Certification {
  private Certification() {}

  /**
   * 1 つの区分の推定。
   *
   * @param roots 根の数
   * @param decisions 根が代表する判断の数 Σ w = Σ 1/β_c（Î の分母）
   * @param games 根のある半荘の数 G
   * @param switches T_X(o) が played と同じ手でも双子でもない根の数（乗り換え）
   * @param hits hit の根の数（d = played の根を除く）
   * @param outside 乗り換えのうち T_X(o) が C(o) の外だった根の数（q の台の外で測れない）
   * @param value Î（判断あたりの改善。根がなければ NaN）
   * @param error 半荘で束ねた標準誤差（G &lt; 2 なら NaN）
   */
  public record Estimate(
      long roots,
      double decisions,
      long games,
      long switches,
      long hits,
      long outside,
      double value,
      double error) {}

  /**
   * 一方の生徒の推定。
   *
   * @param student "A" か "B"
   * @param evaluatedOn 根を取った組（生徒 A なら B）
   * @param all 全ての類（判断の数で重み付け）
   * @param commitment 確約の類を合わせたもの
   * @param byClass 類ごと（{@link com.epsilon.prism.core.DecisionClass} の名前。根のない類は載せない）
   */
  public record Side(
      String student,
      PrismRow.Group evaluatedOn,
      Estimate all,
      Estimate commitment,
      Map<String, Estimate> byClass) {}

  /**
   * 報告。
   *
   * @param commitmentMean 確約の類の Ī
   * @param commitmentError その se
   * @param lowerBound Ī − 2·se（NaN なら不合格）
   * @param outsideShare 確約の類の乗り換えのうち C の外だった割合（両方の生徒の和。乗り換えがなければ 0）
   */
  public record Report(
      Side a,
      Side b,
      double commitmentMean,
      double commitmentError,
      double lowerBound,
      double outsideShare,
      double seconds) {
    /**
     * 受け入れてよいか: 下限が −tolerance を超え（NaN は偽）、かつ C の外の乗り換えの割合が maxOutsideShare 以下（測れない乗り換えが多い生徒は、
     * 下限が良くても悪化を見逃しうるので通さない）。
     */
    public boolean passes(double tolerance, double maxOutsideShare) {
      return lowerBound > -tolerance && outsideShare <= maxOutsideShare;
    }
  }

  /**
   * 認証の量を求める。
   *
   * @param studentA 生徒 A（推論用か学習用。{@link PrismModel#scores(java.util.List)} だけを使う）
   * @param studentB 生徒 B
   * @param roots 今のコホートの根の記録（両方の組・両方の用途。呼び出し側が索引で選ぶ）
   * @param batchSize 採点のバッチの行数
   */
  public static Report certify(
      RowSource source,
      BatchAssembler assembler,
      PrismModel studentA,
      PrismModel studentB,
      int[] roots,
      int batchSize) {
    long started = System.nanoTime();
    if (batchSize < 1) throw new IllegalArgumentException("batchSize must be positive: " + batchSize);
    // 生徒 A は組 B の根で、生徒 B は組 A の根で測る
    var byA = new Estimator();
    var byB = new Estimator();
    int n = roots.length;
    if (n > 0) {
      CompletableFuture<BatchAssembler.Decoded> next =
          assembler.decode(source, BatchAssembler.batch(roots, 0, batchSize));
      for (int from = 0; from < n; from += batchSize) {
        BatchAssembler.Decoded decoded = StudentFit.await(next);
        if (from + batchSize < n)
          next = assembler.decode(source, BatchAssembler.batch(roots, from + batchSize, batchSize));
        List<Integer> groupB = new ArrayList<>(), groupA = new ArrayList<>();
        for (int j = 0; j < decoded.rows().length; j++) {
          PrismRow row = decoded.rows()[j];
          if (row.source() != PrismRow.Source.ROOT)
            throw new IllegalArgumentException(
                "Certification needs root records, got " + row.source() + " of game " + row.gameId());
          (row.group() == PrismRow.Group.B ? groupB : groupA).add(j);
        }
        score(studentA, decoded, groupB, byA);
        score(studentB, decoded, groupA, byB);
      }
    }
    return report(
        byA.side("A", PrismRow.Group.B),
        byB.side("B", PrismRow.Group.A),
        (System.nanoTime() - started) / 1e9);
  }

  /** 二人の生徒の推定から判定の量（確約の類の Ī・se・下限・C の外の乗り換えの割合）を作る。 */
  static Report report(Side a, Side b, double seconds) {
    double mean = (a.commitment().value() + b.commitment().value()) / 2;
    double error =
        Math.sqrt(
                a.commitment().error() * a.commitment().error()
                    + b.commitment().error() * b.commitment().error())
            / 2;
    long switches = a.commitment().switches() + b.commitment().switches();
    long outside = a.commitment().outside() + b.commitment().outside();
    double share = switches == 0 ? 0 : outside / (double) switches;
    return new Report(a, b, mean, error, mean - 2 * error, share, seconds);
  }

  /** 展開したバッチの members 番目の行を model で採点し、estimator に足す。 */
  private static void score(
      PrismModel model, BatchAssembler.Decoded decoded, List<Integer> members, Estimator estimator) {
    if (members.isEmpty()) return;
    List<DecisionHostBatch> hosts = new ArrayList<>(members.size());
    for (int j : members) hosts.add(decoded.hosts()[j]);
    float[][] scores = model.scores(hosts);
    for (int k = 0; k < members.size(); k++) {
      int j = members.get(k);
      estimator.add(decoded.rows()[j], legal(decoded.hosts()[j]), scores[k]);
    }
  }

  /** 行の合法手（復号した 1 行の入力の legalActionId から）。 */
  static List<Action> legal(DecisionHostBatch host) {
    int n = host.legalActionCount(0);
    List<Action> legal = new ArrayList<>(n);
    for (int slot = 0; slot < n; slot++) legal.add(Action.fromIndex(host.legalActionId(0, slot)));
    return legal;
  }

  /** 一方の生徒の推定。根を 1 つずつ足し（{@link #add}）、{@link #side} で区分ごとの推定にする。網にも入力の復号にも依らない。 */
  static final class Estimator {
    private final Sums all = new Sums(), commitment = new Sums();
    private final EnumMap<DecisionClass, Sums> classes = new EnumMap<>(DecisionClass.class);

    /**
     * 根を 1 つ足す。
     *
     * @param legal 根の判断の合法手（行の並び。双子の判定に使う）
     * @param scores 生徒の合法手ごとの s
     */
    void add(PrismRow row, List<Action> legal, float[] scores) {
      int actions = row.actions(), played = row.played(), deviation = row.deviation();
      if (row.source() != PrismRow.Source.ROOT)
        throw new IllegalArgumentException(
            "Certification needs root records, got " + row.source() + " of game " + row.gameId());
      if (scores.length != actions || legal.size() != actions)
        throw new IllegalStateException(
            "Root of game "
                + row.gameId()
                + " has "
                + actions
                + " actions but the payload has "
                + legal.size()
                + " and the student scored "
                + scores.length);
      if (played < 0 || played >= actions || deviation < 0 || deviation >= actions)
        throw new IllegalStateException(
            "Root of game " + row.gameId() + " has played " + played + " deviation " + deviation);
      float propensity = row.propensity(), probability = row.deviationProbability();
      if (!(propensity > 0)
          || !Float.isFinite(propensity)
          || !(probability > 0)
          || !(probability <= 1)
          || !(propensity <= probability))
        throw new IllegalStateException(
            "Root of game "
                + row.gameId()
                + " has propensity "
                + propensity
                + " and q "
                + probability);
      // 根 1 つが代表する判断の数 1/β_c（p = β_c·q）
      double weight = (double) probability / propensity;
      int target = Greedy.argmax(scores, row.prior(), actions);
      boolean switched = !Twins.same(legal, target, played);
      boolean outside = switched && !inSupport(row, legal, target);
      boolean hit = switched && Twins.same(legal, deviation, target);
      double term = 0;
      if (hit) {
        double branch = row.deviationValue(), main = row.playedValue();
        if (Double.isNaN(branch - main))
          throw new IllegalStateException(
              "Root of game " + row.gameId() + " has values " + branch + " / " + main);
        term = (branch - main) / propensity;
      }
      all.add(row.gameId(), term, weight, switched, hit, outside);
      if (row.decisionClass().commitment())
        commitment.add(row.gameId(), term, weight, switched, hit, outside);
      classes
          .computeIfAbsent(row.decisionClass(), ignored -> new Sums())
          .add(row.gameId(), term, weight, switched, hit, outside);
    }

    /** target が support（C(o)）のどれかと同じ手か双子。 */
    private static boolean inSupport(PrismRow row, List<Action> legal, int target) {
      for (byte member : row.support())
        if (Twins.same(legal, Byte.toUnsignedInt(member), target)) return true;
      return false;
    }

    Side side(String student, PrismRow.Group evaluatedOn) {
      var byClass = new LinkedHashMap<String, Estimate>();
      for (var entry : classes.entrySet())
        byClass.put(entry.getKey().name(), entry.getValue().estimate());
      return new Side(student, evaluatedOn, all.estimate(), commitment.estimate(), byClass);
    }
  }

  /** 1 つの区分の和。半荘ごとの {Σ term, Σ w}。 */
  private static final class Sums {
    private long roots, switches, hits, outside;
    private double total, weight;
    private final Map<Long, double[]> games = new HashMap<>();

    void add(
        long gameId, double term, double w, boolean switched, boolean hit, boolean outsideSupport) {
      roots++;
      if (switched) switches++;
      if (hit) hits++;
      if (outsideSupport) outside++;
      total += term;
      weight += w;
      double[] game = games.computeIfAbsent(gameId, ignored -> new double[2]);
      game[0] += term;
      game[1] += w;
    }

    Estimate estimate() {
      if (roots == 0) return new Estimate(0, 0, 0, 0, 0, 0, Double.NaN, Double.NaN);
      double value = total / weight;
      long count = games.size();
      double error = Double.NaN;
      if (count >= 2) {
        double squares = 0;
        for (double[] game : games.values()) {
          double residual = game[0] - value * game[1];
          squares += residual * residual;
        }
        error = Math.sqrt(count / (count - 1.0) * squares) / weight;
      }
      return new Estimate(roots, weight, count, switches, hits, outside, value, error);
    }
  }
}
