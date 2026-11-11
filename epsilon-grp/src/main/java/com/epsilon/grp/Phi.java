package com.epsilon.grp;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * 局末の卓から、絶対席ごとの Φ を求める。Φ を出す項目ごとに 1 つ作り、どのスレッドからも呼べる。終盤のメモは全てのスレッドで
 * 共有し、ほかのメモと作業領域はスレッドごとに持つ。どれもこの Phi と一緒に捨てる。
 *
 * <p>項目は 5 種類ある。GrpNetwork は局末の卓だけを読む。GruSinkhorn は局末の卓に加えて、各局が始まったときの卓の履歴
 * （RoundHistory、終わった局まで）も読む。GruSinkhorn に PhiTable を重ねると、周辺確率に次局の温度を掛けてから Φ にし、終わった局
 * （履歴の最後の局）の補正を足す。点数の順位は、局末の点をそのまま最終の点とみなす。点数のなめらかな関数は、最終の点を局末の点の
 * 周りの正規分布とみなす。この 2 つはモデルを持たない。終盤の数え上げは、次局が南場の最後の局から先の卓（{@code FinalRound.covers}）
 * を最後の局の数え上げ（FinalRound）で求め、ほかの卓を葉の GrpNetwork で求める。数え上げない卓の値は、葉の GrpNetwork だけの項目と
 * ビット単位で同じ。
 *
 * <p>半荘が続く卓では、キーを作ってメモを引き、外れたら項目で求める。メモのキーは、卓のキー（GrpKey）と印の組。印は、GruSinkhorn
 * では履歴のバージョン、ほかは 0。数え上げる卓は 1 回の計算が重いので、安い卓に追い出されないように別のメモ（終盤のメモ）に覚える。
 * 終盤のメモは全てのスレッドで共有するので、ほかのスレッドが数え上げた卓にも当たる。終盤のメモは Phi ごとに持つので、q のネットワーク・
 * 葉のネットワーク・打ち切りの値・効用のどれかが違う Φ を返すことはない。キーに収まらない卓は、メモを通さずに求める。Φ は卓と、印の
 * もとになる履歴だけで決まる。そのため、スレッド・メモ・呼び出しの順に依らず、ビット単位で同じになる。終局の卓では項目を使わず、順位の
 * 効用 u をそのまま返す（厳密な値）。表を重ねないときは、丸めの差を除いて Σ_i Φ_i = Σ_r u_r が成り立つ。
 */
public final class Phi {

  /** Φ を出す項目の種類。 */
  public enum Kind {
    /** GrpNetwork。局末の卓だけを読む。 */
    NETWORK,
    /** GruSinkhorn。局の履歴も読み、PhiTable を重ねてよい。 */
    GRU,
    /** 局末の点の順位をそのまま最終順位とみなす。周辺確率は 0 か 1。 */
    SCORE_RANK,
    /** 最終の点を、局末の点を中心とする独立な正規分布とみなす。 */
    SCORE_SMOOTH,
    /** 数え上げる卓は最後の局の数え上げ（FinalRound）、ほかの卓は葉の GrpNetwork。 */
    LATE;

    /** モデルを持たず、点数だけで決まる項目なら true。 */
    public boolean scoreOnly() {
      return this == SCORE_RANK || this == SCORE_SMOOTH;
    }
  }

  /**
   * 全てのスレッドを合わせた呼び出しの集計。メモの当たり・外れの数、メモを通さなかった数、時間には、終盤のメモと数え上げの分も含む。
   *
   * @param requests 呼び出しの数（終局の卓を含む）
   * @param hits メモに当たった数
   * @param misses メモに外れて項目で求めた数
   * @param bypasses キーに収まらず、メモを通さずに求めた数
   * @param nanos 項目で求めた時間の合計（ns）
   * @param late 数え上げる卓の分の集計。終盤の数え上げでない項目では全て 0
   */
  public record Stats(long requests, long hits, long misses, long bypasses, long nanos, LateStats late) {

    /** 前に取った集計 {@code earlier} から増えた分を返す。 */
    public Stats since(Stats earlier) {
      return new Stats(
          requests - earlier.requests, hits - earlier.hits, misses - earlier.misses, bypasses - earlier.bypasses,
          nanos - earlier.nanos, late.since(earlier.late));
    }
  }

  /**
   * 終盤の数え上げで、数え上げる卓を呼んだ分の集計。
   *
   * @param requests 数え上げる卓の呼び出しの数
   * @param hits 終盤のメモに当たった数（ほかのスレッドが数え上げた卓に当たった分も含む）
   * @param misses 終盤のメモに外れて数え上げた数
   * @param bypasses キーに収まらず、メモを通さずに数え上げた数
   * @param nanos 数え上げた時間の合計（ns）
   * @param settled 精算して局を進めた組の数
   * @param leafNetworks 葉のネットワークで求めた卓の数（打ち切った確率を埋める分は数えない）
   * @param leafHits 葉の値を葉のメモから読んだ数
   * @param filled 打ち切って葉のネットワークで埋めた確率を、数え上げた卓について足したもの
   */
  public record LateStats(
      long requests, long hits, long misses, long bypasses, long nanos, long settled, long leafNetworks,
      long leafHits, double filled) {

    /** 前に取った集計 {@code earlier} から増えた分を返す。 */
    public LateStats since(LateStats earlier) {
      return new LateStats(
          requests - earlier.requests, hits - earlier.hits, misses - earlier.misses, bypasses - earlier.bypasses,
          nanos - earlier.nanos, settled - earlier.settled, leafNetworks - earlier.leafNetworks,
          leafHits - earlier.leafHits, filled - earlier.filled);
    }
  }

  /** スレッドごとのメモのスロット数。 */
  private static final int MEMO_SLOTS = 1 << 12;

  /** 履歴を読まない項目で、メモの印に使う値。どの履歴でも同じ値にする。 */
  private static final long NO_HISTORY = 0;

  private final Kind kind;
  private final RankUtility utility;
  private final Variant variant;

  /** 点数だけの項目では null。 */
  private final GrpModel model;

  /** 表を重ねないときは null。 */
  private final PhiTable phiTable;

  /** 終盤の数え上げのときだけ持つ。 */
  private final FinalRound finalRound;

  /** 全てのスレッドで共有する終盤のメモ。終盤の数え上げのときだけ持つ。 */
  private final SharedPhiMemo lateMemo;

  private final ThreadLocal<Local> locals = ThreadLocal.withInitial(this::local);

  /** 全てのスレッドの持ち物の一覧。集計を足し合わせるのに使う。 */
  private final ConcurrentLinkedQueue<Local> registered = new ConcurrentLinkedQueue<>();

  /**
   * スレッドごとの持ち物。作業領域は項目の種類に合うものだけを作り、ほかは null。集計はそのスレッドだけが書く。
   *
   * @param probabilities 順列ごとの確率（ネットワークと終盤の数え上げだけ）
   * @param marginals 周辺確率の書き先（点数だけの項目だけ）
   * @param late 数え上げの作業領域（終盤の数え上げだけ）
   */
  private record Local(
      PhiMemo memo, GrpWorkspace network, GruWorkspace gru, double[] probabilities, double[] marginals,
      FinalRoundWorkspace late, Counts counts) {}

  /**
   * スレッドごとの呼び出しの数と時間。スレッドごとのメモの当たり・外れはメモ自身が数え、共有する終盤のメモの分はここで数える。
   * そのスレッドだけが書く。
   */
  private static final class Counts {
    long requests;
    long nanos;
    long lateRequests;
    long lateHits;
    long lateMisses;
    long lateBypasses;
    long lateNanos;
    long settled;
    long leafNetworks;
    long leafHits;
    double filled;
  }

  private Phi(
      Kind kind, Variant variant, RankUtility utility, GrpModel model, PhiTable phiTable, FinalRound finalRound,
      int lateSlots) {
    this.kind = kind;
    this.variant = Objects.requireNonNull(variant, "variant");
    this.utility = Objects.requireNonNull(utility, "utility");
    this.model = model;
    this.phiTable = phiTable;
    this.finalRound = finalRound;
    lateMemo = finalRound == null ? null : new SharedPhiMemo(lateSlots, variant.seats());
  }

  /** GrpNetwork か GruSinkhorn の Φ を作る。 */
  public static Phi of(GrpModel model, RankUtility utility) {
    return switch (model) {
      case GrpNetwork network -> new Phi(Kind.NETWORK, network.variant(), utility, network, null, null, 0);
      case GruSinkhorn gru -> new Phi(Kind.GRU, gru.variant(), utility, gru, null, null, 0);
    };
  }

  /** GruSinkhorn に PhiTable を重ねる。ルールが違えば IllegalArgumentException を投げる。 */
  public static Phi of(GruSinkhorn model, RankUtility utility, PhiTable phiTable) {
    if (phiTable.variant() != model.variant()) {
      throw new IllegalArgumentException("Variant mismatch: " + model + ", table " + phiTable.variant());
    }
    return new Phi(Kind.GRU, model.variant(), utility, model, phiTable, null, 0);
  }

  /** 局末の点の順位をそのまま最終順位とみなす Φ。 */
  public static Phi scoreRank(Variant variant, RankUtility utility) {
    return new Phi(Kind.SCORE_RANK, variant, utility, null, null, null, 0);
  }

  /** 点数のなめらかな関数の Φ（{@link ScoreSmooth}）。点のばらつきの係数を測っていないルールでは IllegalArgumentException を投げる。 */
  public static Phi scoreSmooth(Variant variant, RankUtility utility) {
    ScoreSmooth.requireMeasured(variant);
    return new Phi(Kind.SCORE_SMOOTH, variant, utility, null, null, null, 0);
  }

  /**
   * 終盤だけを数え上げる Φ を作る。数え上げる卓は最後の局の数え上げで、ほかの卓は葉のネットワークで求める。ルールは数え上げと同じ。
   *
   * @param lateSlots 終盤のメモのスロットの数。終盤のメモは数え上げた卓の Φ を覚え、全てのスレッドで共有する。
   *     1 スロットは (S ＋ 1) × 8 バイト
   * @throws IllegalArgumentException スロットの数が 2 の冪でないとき、スロットの数と席の数の積が int に収まらないとき
   */
  public static Phi late(FinalRound finalRound, RankUtility utility, int lateSlots) {
    if (lateSlots < 1 || Integer.bitCount(lateSlots) != 1) {
      throw new IllegalArgumentException("Late memo slots must be a power of two: " + lateSlots);
    }
    return new Phi(Kind.LATE, finalRound.variant(), utility, finalRound.leaf(), null, finalRound, lateSlots);
  }

  public Kind kind() {
    return kind;
  }

  /** 項目のモデルを返す。終盤の数え上げでは葉のネットワーク。点数だけの項目では null。 */
  public GrpModel model() {
    return model;
  }

  /** 最後の局の数え上げを返す。終盤の数え上げでない項目では null。 */
  public FinalRound finalRound() {
    return finalRound;
  }

  public RankUtility utility() {
    return utility;
  }

  public Variant variant() {
    return variant;
  }

  /** 局の履歴を読む項目（GruSinkhorn）なら true。そのときだけ、呼び出し元は RoundHistory に局を積んで渡す。 */
  public boolean needsHistory() {
    return kind == Kind.GRU;
  }

  /** GrpNetwork の新しい作業領域を返す。ネットワークの項目と終盤の数え上げでだけ使え、ほかの項目では IllegalStateException を投げる。 */
  public GrpWorkspace networkWorkspace() {
    if (!(model instanceof GrpNetwork network)) {
      throw new IllegalStateException("Not a network entry: " + kind);
    }
    return GrpWorkspace.of(network.shape());
  }

  /** 終盤の数え上げの新しい作業領域を返す。ほかの項目では IllegalStateException を投げる。 */
  public FinalRoundWorkspace finalRoundWorkspace() {
    if (finalRound == null) {
      throw new IllegalStateException("Not a late entry: " + kind);
    }
    return FinalRoundWorkspace.of(finalRound);
  }

  /**
   * 卓の Φ を書く。終局の卓には順位の効用をそのまま書く。GruSinkhorn の項目では、半荘が続く卓に履歴が要るので
   * IllegalArgumentException を投げる（履歴を渡す方を使う）。
   *
   * @param out 出力先（長さ S）
   */
  public void values(TableState table, double[] out) {
    values(null, table, out);
  }

  /**
   * 半荘が続く卓の Φ を書く。GruSinkhorn の項目では履歴が要るので、IllegalArgumentException を投げる（履歴を渡す方を使う）。
   *
   * @param out 出力先（長さ S）
   */
  public void values(GrpState state, double[] out) {
    values(null, state, out);
  }

  /**
   * 卓の Φ を書く。終局の卓には順位の効用をそのまま書く。
   *
   * @param history 各局が始まったときの卓の履歴（終わった局まで）。GruSinkhorn だけが読み、1 局以上が要る。ほかの項目では null でもよい
   * @param out 出力先（長さ S）
   */
  public void values(RoundHistory history, TableState table, double[] out) {
    tableValues(history, table, out);
  }

  /**
   * 半荘が続く卓の Φ を書く。
   *
   * @param history {@link #values(RoundHistory, TableState, double[])} と同じ
   * @param out 出力先（長さ S）
   */
  public void values(RoundHistory history, GrpState state, double[] out) {
    stateValues(history, state, out);
  }

  /**
   * 半荘が続く卓の Φ と、席ごとの最終順位の周辺確率を書く。GruSinkhorn に表を重ねたときの周辺確率は、温度を掛けた後で補正を足す前の
   * 値。記録した局末の評価に使うので、メモを通さず、集計にも入れない。Φ は {@link #values(RoundHistory, GrpState, double[])} と
   * ビット単位で同じ。
   *
   * @param history {@link #values(RoundHistory, TableState, double[])} と同じ
   * @param phiOut Φ の出力先（長さ S）
   * @param marginalsOut 周辺確率の出力先（長さ S²。並びは [絶対席 × S ＋ 順位]）
   */
  public void evaluate(RoundHistory history, GrpState state, double[] phiOut, double[] marginalsOut) {
    requireTable(state.variant(), phiOut);
    int seats = variant.seats();
    if (marginalsOut.length != seats * seats) {
      throw new IllegalArgumentException(variant + " needs " + seats * seats + " marginals, got " + marginalsOut.length);
    }
    Local local = locals.get();
    switch (kind) {
      case NETWORK -> {
        RankPermutations table = RankPermutations.of(variant);
        ((GrpNetwork) model).permutations(state, local.network, local.probabilities);
        table.phi(local.probabilities, state.dealer(), utility, phiOut);
        table.marginals(local.probabilities, state.dealer(), marginalsOut);
      }
      case GRU -> {
        stamp(history);
        ((GruSinkhorn) model).forward(history, state, local.gru);
        overlay(history, state.round(), local.gru.marginals, phiOut);
        System.arraycopy(local.gru.marginals, 0, marginalsOut, 0, marginalsOut.length);
      }
      case SCORE_RANK, SCORE_SMOOTH -> {
        scoreMarginals(state, marginalsOut);
        GruSinkhorn.expectedUtilities(variant, marginalsOut, utility, phiOut);
      }
      case LATE -> {
        RankPermutations table = RankPermutations.of(variant);
        if (FinalRound.covers(state)) {
          finalRound.permutations(state, local.late, local.probabilities);
        } else {
          ((GrpNetwork) model).permutations(state, local.network, local.probabilities);
        }
        table.phi(local.probabilities, state.dealer(), utility, phiOut);
        table.marginals(local.probabilities, state.dealer(), marginalsOut);
      }
    }
  }

  private void tableValues(RoundHistory history, TableState table, double[] out) {
    requireTable(table.variant(), out);
    Local local = locals.get();
    local.counts.requests++;
    if (table.gameOver()) {
      for (int seat = 0; seat < out.length; seat++) {
        out[seat] = utility.value(variant, table.rank(seat));
      }
      return;
    }
    long stamp = stamp(history);
    long key = GrpKey.of(table);
    if (finalRound != null && FinalRound.covers(table)) {
      if (!lateFound(key, local, out)) {
        enumerate(key, table, local, out);
      }
      return;
    }
    if (key != GrpKey.NONE && local.memo.find(stamp, key, out)) {
      return;
    }
    long started = System.nanoTime();
    switch (kind) {
      case NETWORK, LATE -> ((GrpNetwork) model).phi(table, utility, local.network, out);
      case GRU -> {
        ((GruSinkhorn) model).forward(history, table, local.gru);
        overlay(history, table.round(), local.gru.marginals, out);
      }
      case SCORE_RANK, SCORE_SMOOTH -> {
        scoreMarginals(GrpState.of(table), local.marginals);
        GruSinkhorn.expectedUtilities(variant, local.marginals, utility, out);
      }
    }
    local.counts.nanos += System.nanoTime() - started;
    remember(stamp, key, local.memo, out);
  }

  private void stateValues(RoundHistory history, GrpState state, double[] out) {
    requireTable(state.variant(), out);
    Local local = locals.get();
    local.counts.requests++;
    long stamp = stamp(history);
    long key = state.key();
    if (finalRound != null && FinalRound.covers(state)) {
      if (!lateFound(key, local, out)) {
        enumerate(key, local.late.load(state), local, out);
      }
      return;
    }
    if (key != GrpKey.NONE && local.memo.find(stamp, key, out)) {
      return;
    }
    long started = System.nanoTime();
    switch (kind) {
      case NETWORK, LATE -> ((GrpNetwork) model).phi(state, utility, local.network, out);
      case GRU -> {
        ((GruSinkhorn) model).forward(history, state, local.gru);
        overlay(history, state.round(), local.gru.marginals, out);
      }
      case SCORE_RANK, SCORE_SMOOTH -> {
        scoreMarginals(state, local.marginals);
        GruSinkhorn.expectedUtilities(variant, local.marginals, utility, out);
      }
    }
    local.counts.nanos += System.nanoTime() - started;
    remember(stamp, key, local.memo, out);
  }

  /** 数え上げる卓の呼び出しを数え、終盤のメモに当たれば Φ をコピーして true を返す。 */
  private boolean lateFound(long key, Local local, double[] out) {
    Counts counts = local.counts;
    counts.lateRequests++;
    if (key == GrpKey.NONE) {
      return false;
    }
    if (lateMemo.find(key, out)) {
      counts.lateHits++;
      return true;
    }
    counts.lateMisses++;
    return false;
  }

  /** 数え上げる卓の Φ を数え上げて書き、終盤のメモに覚える。キーに収まらない卓は覚えず、メモを通さなかった数に入れる。 */
  private void enumerate(long key, TableState table, Local local, double[] out) {
    FinalRoundWorkspace workspace = local.late;
    Counts counts = local.counts;
    long settled = workspace.settled();
    long leafNetworks = workspace.networkLeaves();
    long leafHits = workspace.memoHits();
    long started = System.nanoTime();
    double filled = finalRound.phi(table, utility, workspace, out);
    long nanos = System.nanoTime() - started;
    counts.nanos += nanos;
    counts.lateNanos += nanos;
    counts.settled += workspace.settled() - settled;
    counts.leafNetworks += workspace.networkLeaves() - leafNetworks;
    counts.leafHits += workspace.memoHits() - leafHits;
    counts.filled += filled;
    if (key == GrpKey.NONE) {
      counts.lateBypasses++;
    } else {
      lateMemo.store(key, out);
    }
  }

  /** 呼んだスレッドのメモを返す。当たり・外れの回数を読むのに使う。 */
  public PhiMemo memo() {
    return locals.get().memo;
  }

  /**
   * 全てのスレッドの集計を足し合わせて返す。集計は各スレッドが自分で書くので、そのスレッドの呼び出しが終わった後（収集を待った後
   * など）に読む。
   */
  public Stats stats() {
    long requests = 0;
    long hits = 0;
    long misses = 0;
    long bypasses = 0;
    long nanos = 0;
    long lateRequests = 0;
    long lateHits = 0;
    long lateMisses = 0;
    long lateBypasses = 0;
    long lateNanos = 0;
    long settled = 0;
    long leafNetworks = 0;
    long leafHits = 0;
    double filled = 0;
    for (Local local : registered) {
      Counts counts = local.counts;
      requests += counts.requests;
      nanos += counts.nanos;
      hits += local.memo.hits();
      misses += local.memo.misses();
      bypasses += local.memo.bypasses();
      lateRequests += counts.lateRequests;
      lateHits += counts.lateHits;
      lateMisses += counts.lateMisses;
      lateBypasses += counts.lateBypasses;
      lateNanos += counts.lateNanos;
      settled += counts.settled;
      leafNetworks += counts.leafNetworks;
      leafHits += counts.leafHits;
      filled += counts.filled;
    }
    return new Stats(
        requests, hits + lateHits, misses + lateMisses, bypasses + lateBypasses, nanos,
        new LateStats(
            lateRequests, lateHits, lateMisses, lateBypasses, lateNanos, settled, leafNetworks, leafHits, filled));
  }

  @Override
  public String toString() {
    String entry = finalRound != null ? " " + finalRound : model == null ? "" : " " + model;
    return "Phi[" + kind + " " + variant + " " + utility + entry + (phiTable == null ? "" : " with table") + "]";
  }

  /**
   * メモの印を返す。GruSinkhorn では履歴のバージョン、ほかは 0。GruSinkhorn では、履歴が null でも空でもなく、ルールが同じことも
   * 確かめる。
   */
  private long stamp(RoundHistory history) {
    return model instanceof GruSinkhorn gru ? gru.requireHistory(history).stamp() : NO_HISTORY;
  }

  /**
   * 周辺確率から Φ を求めて書く。表があれば、先に周辺確率に次局の温度を掛け、後で Φ に履歴の最後の局の補正を足す。このとき
   * 周辺確率は、温度を掛けた値に書き換わる。
   */
  private void overlay(RoundHistory history, int nextRound, double[] marginals, double[] out) {
    if (phiTable != null) {
      phiTable.temper(nextRound, marginals);
    }
    GruSinkhorn.expectedUtilities(variant, marginals, utility, out);
    if (phiTable != null) {
      phiTable.calibrate(history.round(history.size() - 1), out);
    }
  }

  /** 点数だけの項目の周辺確率を書く。 */
  private void scoreMarginals(GrpState state, double[] marginals) {
    if (kind == Kind.SCORE_RANK) {
      int seats = variant.seats();
      Arrays.fill(marginals, 0);
      for (int seat = 0; seat < seats; seat++) {
        marginals[seat * seats + state.rank(seat)] = 1;
      }
    } else {
      ScoreSmooth.marginals(state, marginals);
    }
  }

  private static void remember(long stamp, long key, PhiMemo memo, double[] out) {
    if (key == GrpKey.NONE) {
      memo.bypass();
    } else {
      memo.store(stamp, key, out);
    }
  }

  private Local local() {
    int seats = variant.seats();
    int permutations = RankPermutations.of(variant).count();
    PhiMemo memo = new PhiMemo(MEMO_SLOTS, seats);
    Counts counts = new Counts();
    Local local = switch (kind) {
      case NETWORK -> new Local(memo, networkWorkspace(), null, new double[permutations], null, null, counts);
      case GRU -> new Local(memo, null, GruWorkspace.of((GruSinkhorn) model), null, null, null, counts);
      case SCORE_RANK, SCORE_SMOOTH -> new Local(memo, null, null, null, new double[seats * seats], null, counts);
      case LATE -> new Local(
          memo, networkWorkspace(), null, new double[permutations], null, FinalRoundWorkspace.of(finalRound), counts);
    };
    registered.add(local);
    return local;
  }

  private void requireTable(Variant tableVariant, double[] out) {
    if (tableVariant != variant) {
      throw new IllegalArgumentException("Variant mismatch: Phi " + variant + ", table " + tableVariant);
    }
    if (out.length != variant.seats()) {
      throw new IllegalArgumentException(variant + " needs " + variant.seats() + " values, got " + out.length);
    }
  }
}
