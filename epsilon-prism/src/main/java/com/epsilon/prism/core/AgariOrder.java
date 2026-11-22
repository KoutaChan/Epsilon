package com.epsilon.prism.core;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.WinSettlementProjection.SettlementAssumption;
import com.epsilon.mahjong.rules.DealerOutcome;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.input.WinFacts;
import com.epsilon.prism.luck.UraDistribution;
import java.util.Arrays;
import java.util.Objects;

/**
 * 公開観測と自手から、ルールで証明できる和了の終局順位と順序辺を求める。
 *
 * <p>ツモは同時和了が無いので首位確定（和了 ≽ 全他手）と最下位確定（全他手 ≽ 和了）の両方で順序辺を作る。ロンは他家の同時和了
 * （ダブロン）により放銃者が沈んで自分の順位が上がり得るため、最下位確定は順序辺を作らず {@link Kind#CERT_MID} と同じ扱いとし、
 * 単独精算で首位確定となるロンだけ順序辺を作る（他家の同時ロンの有無によらず自分のロンは他手より弱くならない）。
 *
 * <p>裏ドラが乗り得る非役満では、裏 0 枚での首位終局・翻の上限到達・最大裏ドラでの非終局や非辺判定・ツモ最下位確定を定数時間で
 * 判定し、境界を跨ぐ行だけ公開残数プールに対する到達可能な裏ドラ枚数をビットマスク DP で調べる。
 *
 * <p>推論時の厳密な FP32 クランプ（{@link #clamp}）と、教師に対する星型 DAG の二乗距離射影（{@link #project}）を提供する。
 * 作業領域を使い回すのでスレッドセーフではない。
 */
public final class AgariOrder {

  /** 証明できる和了の順序を教師・学習・推論に反映するか。 */
  public enum Mode {
    /** 従来の挙動。和了の順序制約を掛けない。 */
    LEGACY,
    /** 証明できる和了の順序を教師の射影・学習の勾配遮断・推論のクランプに反映する。 */
    CERTIFIED
  }

  /** 公開情報から証明できた和了の終局順位の区分。 */
  public enum Kind {
    /** 終局しないか、裏ドラ次第で終局しない可能性がある。 */
    NONE,
    /** 必ず終局し、常に 1 位（全他手より優位。ツモ・ロン共通）。 */
    CERT_TOP,
    /** ツモ和了で必ず終局し、常に最下位（全他手より劣位。ロンでは返さない）。 */
    CERT_BOTTOM,
    /** 必ず終局するが、順序辺を作らない（中位、順位変動、またはロンの単独最下位）。 */
    CERT_MID;

    /** 順序辺（和了 ≽ 他手 または 他手 ≽ 和了）を持つか。 */
    public boolean hasEdge() {
      return this == CERT_TOP || this == CERT_BOTTOM;
    }
  }

  /**
   * 1 つの判断における和了の証明結果。
   *
   * @param action 和了の合法手の位置（非終局または和了無しなら −1）
   * @param kind 終局順位の区分
   * @param bestRank 終局時に取り得る最も良い順位（0 始まり。非終局なら −1）
   * @param worstRank 終局時に取り得る最も悪い順位（0 始まり。非終局なら −1）
   * @param tsumo ツモ和了なら true
   */
  public record Certification(int action, Kind kind, int bestRank, int worstRank, boolean tsumo) {

    /** 証明できる終局和了が無いことを表す結果。 */
    public static final Certification NONE = new Certification(-1, Kind.NONE, -1, -1, false);

    /** 順序辺を持つか（首位確定またはツモ最下位確定）。 */
    public boolean hasEdge() {
      return kind.hasEdge();
    }

    /** 首位確定（和了 ≽ 他手）なら true。 */
    public boolean top() {
      return kind == Kind.CERT_TOP;
    }
  }

  /** {@code ObservationSchema.encodeFu} の復号表（0 は役満、1 は 20 符、2 は 25 符、3〜11 は 30〜110 符）。 */
  private static final int[] FU_TABLE = {0, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110};

  /** 満貫以上（5 翻以上）で {@link ScoreMath#base} に渡す有効な符。 */
  private static final int MANGAN_FU = 30;

  /** 満貫の下限の翻。 */
  private static final int MANGAN_HAN = 5;

  /** 槓の 4 枚目の増分と北抜きを除いた、和了形の枚数。 */
  private static final int HAND_TILES = Variant.HAND + 1;

  private final Variant variant;
  private final TileSet tileSet;
  private final long validKinds;
  private final int seats;
  private final int lastRound;
  private final int finalRound;
  private final int[] doraOf = new int[Tiles.KINDS];
  private final int[] basePool = new int[Tiles.KINDS];
  private final int[] absolute0;
  private final int[] absolute;
  private final int[] pool = new int[Tiles.KINDS];
  private final int[] owned = new int[Tiles.KINDS];
  private final int[] uraCounts = new int[Tiles.KINDS];
  private final int[] grouped = new int[Tiles.COPIES + 1];
  private final int[] reach = new int[UraDistribution.MAX_INDICATORS + 1];
  private final Observation.EventRecord eventRecord = new Observation.EventRecord();
  private UraDistribution ura;
  private int lastBestRank;
  private int lastWorstRank;
  private boolean lastTsumo;

  /**
   * ルールを指定して作業領域を作る。
   *
   * @param variant ルール
   */
  public AgariOrder(Variant variant) {
    this.variant = Objects.requireNonNull(variant, "variant");
    tileSet = variant.tiles();
    validKinds = tileSet.kinds();
    seats = variant.seats();
    lastRound = variant.lastRound();
    finalRound = variant.finalRound();
    absolute0 = new int[seats];
    absolute = new int[seats];
    Arrays.fill(doraOf, -1);
    for (long bits = validKinds; bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      doraOf[kind] = tileSet.doraOf(kind);
      basePool[kind] = tileSet.copies(kind);
    }
  }

  /**
   * 判断の記録にある和了を判定する。和了が合法でないときは {@link Certification#NONE} を返す。
   *
   * @param o 判断の記録
   * @return 和了の証明結果
   */
  public Certification certify(Observation o) {
    return certify(o, true);
  }

  /**
   * 順序辺の有無だけを判定し、順序辺があるときだけ {@link Certification} を返す（学習エンコード用）。順序辺の無い行は
   * {@link Certification#NONE} を返し、オブジェクトを確保しない。
   *
   * @param o 判断の記録
   * @return 順序辺を持つ証明結果、または {@link Certification#NONE}
   */
  public Certification certifyEdge(Observation o) {
    Kind kind = classify(o, true, true);
    if (!kind.hasEdge()) {
      return Certification.NONE;
    }
    int rank = kind == Kind.CERT_TOP ? 0 : seats - 1;
    return new Certification(findWinAction(o), kind, rank, rank, lastTsumo);
  }

  /**
   * 高速経路の有効・無効を指定して和了を判定する（テストと監査で全列挙との一致を確かめるために使う）。
   *
   * @param o 判断の記録
   * @param fastPaths 高速経路とビットマスク到達判定を使うなら true、全裏ドラ確率列挙なら false
   * @return 和了の証明結果
   */
  Certification certify(Observation o, boolean fastPaths) {
    Kind kind = classify(o, fastPaths, false);
    if (kind == Kind.NONE) {
      return Certification.NONE;
    }
    return new Certification(findWinAction(o), kind, lastBestRank, lastWorstRank, lastTsumo);
  }

  /**
   * 推論のスコア列に、証明辺に基づく厳密な FP32 クランプをその場で適用する。証明辺が無い行は触らず、オブジェクトを確保しない。
   *
   * @param o 判断の記録
   * @param scores 合法手ごとのスコア（書き換える）
   * @return スコアを書き換えたら true
   */
  public boolean clamp(Observation o, float[] scores) {
    if (o.wins() == 0) {
      return false;
    }
    if (scores.length != o.legalActions()) {
      throw new IllegalArgumentException(
          "Score length " + scores.length + " differs from legal actions " + o.legalActions());
    }
    Kind kind = classify(o, true, true);
    if (!kind.hasEdge()) {
      return false;
    }
    return clamp(scores, findWinAction(o), kind == Kind.CERT_TOP);
  }

  /**
   * 証明辺のある 1 手に対して、スコア列を厳密な FP32 順序にクランプする。首位確定は他手の最大の {@link Math#nextUp} 以上、
   * 最下位確定は他手の最小の {@link Math#nextDown} 以下にする。非有限値は例外を投げる。
   *
   * @param scores 合法手ごとのスコア（書き換える）
   * @param winAction 和了の合法手の位置
   * @param top 首位確定なら true、最下位確定なら false
   * @return スコアを書き換えたら true
   */
  public static boolean clamp(float[] scores, int winAction, boolean top) {
    Objects.checkIndex(winAction, scores.length);
    if (top) {
      float peak = Float.NEGATIVE_INFINITY;
      for (int a = 0; a < scores.length; a++) {
        float score = scores[a];
        if (!Float.isFinite(score)) {
          throw new IllegalStateException("Nonfinite score at action " + a + ": " + score);
        }
        if (a != winAction && score > peak) {
          peak = score;
        }
      }
      if (scores.length < 2) {
        return false;
      }
      float bound = Math.nextUp(peak);
      if (!Float.isFinite(bound)) {
        bound = peak;
      }
      if (scores[winAction] < bound) {
        scores[winAction] = bound;
        return true;
      }
      return false;
    }
    float floor = Float.POSITIVE_INFINITY;
    for (int a = 0; a < scores.length; a++) {
      float score = scores[a];
      if (!Float.isFinite(score)) {
        throw new IllegalStateException("Nonfinite score at action " + a + ": " + score);
      }
      if (a != winAction && score < floor) {
        floor = score;
      }
    }
    if (scores.length < 2) {
      return false;
    }
    float bound = Math.nextDown(floor);
    if (!Float.isFinite(bound)) {
      bound = floor;
    }
    if (scores[winAction] > bound) {
      scores[winAction] = bound;
      return true;
    }
    return false;
  }

  /**
   * 測定済みの教師列が証明辺に反するかを返す。和了が未測定（NaN）のときは false。
   *
   * @param values 合法手ごとの教師（未測定は NaN）
   * @param winAction 和了の合法手の位置
   * @param top 首位確定なら true、最下位確定なら false
   * @return 証明辺に反する測定済みの他手があれば true
   */
  public static boolean violates(float[] values, int winAction, boolean top) {
    Objects.checkIndex(winAction, values.length);
    float win = values[winAction];
    if (Float.isNaN(win)) {
      return false;
    }
    if (!Float.isFinite(win)) {
      throw new IllegalStateException("Nonfinite win target: " + win);
    }
    for (int a = 0; a < values.length; a++) {
      if (a == winAction) {
        continue;
      }
      float other = values[a];
      if (Float.isNaN(other)) {
        continue;
      }
      if (!Float.isFinite(other)) {
        throw new IllegalStateException("Nonfinite action target at " + a + ": " + other);
      }
      if (top ? other > win : other < win) {
        return true;
      }
    }
    return false;
  }

  /**
   * 星型 DAG の順序錐へ、測定済みの教師列を二乗距離で射影する（閉形式 KKT 解、FP32 の 1 ULP 補正込み）。未測定（NaN）の要素は
   * 変えない。既に全辺を満たすときは何も書き換えず false を返す。
   *
   * @param values 合法手ごとの教師（未測定は NaN。書き換える）
   * @param winAction 和了の合法手の位置
   * @param top 首位確定（和了 ≥ 他手）なら true、最下位確定（和了 ≤ 他手）なら false
   * @return 教師を書き換えたら true
   */
  public static boolean project(float[] values, int winAction, boolean top) {
    Objects.checkIndex(winAction, values.length);
    float win = values[winAction];
    if (Float.isNaN(win)) {
      return false;
    }
    if (!Float.isFinite(win)) {
      throw new IllegalStateException("Nonfinite win target: " + win);
    }
    int n = values.length;
    double[] violators = new double[n];
    int m = 0;
    for (int a = 0; a < n; a++) {
      if (a == winAction) {
        continue;
      }
      float other = values[a];
      if (Float.isNaN(other)) {
        continue;
      }
      if (!Float.isFinite(other)) {
        throw new IllegalStateException("Nonfinite action target at " + a + ": " + other);
      }
      if (top ? other > win : other < win) {
        violators[m++] = other;
      }
    }
    if (m == 0) {
      return false;
    }
    Arrays.sort(violators, 0, m);
    double sum = win;
    double mu = sum;
    if (top) {
      for (int k = 1; k <= m; k++) {
        sum += violators[m - k];
        mu = sum / (k + 1);
        if (k == m || mu >= violators[m - k - 1]) {
          break;
        }
      }
      float level = (float) mu;
      values[winAction] = level;
      for (int a = 0; a < n; a++) {
        if (a != winAction && !Float.isNaN(values[a]) && values[a] > level) {
          values[a] = level;
        }
      }
      for (int a = 0; a < n; a++) {
        if (a != winAction && !Float.isNaN(values[a]) && values[winAction] < values[a]) {
          values[winAction] = Math.nextUp(values[winAction]);
          if (values[winAction] < values[a]) {
            values[winAction] = values[a];
          }
        }
      }
    } else {
      for (int k = 1; k <= m; k++) {
        sum += violators[k - 1];
        mu = sum / (k + 1);
        if (k == m || mu <= violators[k]) {
          break;
        }
      }
      float level = (float) mu;
      values[winAction] = level;
      for (int a = 0; a < n; a++) {
        if (a != winAction && !Float.isNaN(values[a]) && values[a] < level) {
          values[a] = level;
        }
      }
      for (int a = 0; a < n; a++) {
        if (a != winAction && !Float.isNaN(values[a]) && values[winAction] > values[a]) {
          values[winAction] = Math.nextDown(values[winAction]);
          if (values[winAction] > values[a]) {
            values[winAction] = values[a];
          }
        }
      }
    }
    return true;
  }

  /**
   * 判断の和了を区分し、結果を {@link #lastBestRank}・{@link #lastWorstRank}・{@link #lastTsumo} に書く。
   * {@code edgeOnly} が true のときは順序辺の有無だけを厳密に保ち、非辺（{@link Kind#CERT_MID} と {@link Kind#NONE}）を
   * 区別せず {@link Kind#NONE} で早期終了する。
   */
  private Kind classify(Observation o, boolean fastPaths, boolean edgeOnly) {
    int wins = o.wins();
    if (wins == 0) {
      return Kind.NONE;
    }
    if (wins != 1) {
      throw new IllegalStateException("Multiple win actions in one observation");
    }
    if (o.variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + variant + ", got " + o.variant());
    }
    SettlementAssumption settlement = o.winSettlement(0);
    boolean tsumo = settlement == SettlementAssumption.EXACT_TSUMO;
    lastTsumo = tsumo;
    boolean finishes0 = WinFacts.settlesGame(o, 0, absolute0);
    int rank0 = o.winRank(0, 0);
    int draws = o.winUraIndicators(0);
    if (!o.winUraEligible(0) || o.winYakuman(0) > 0 || draws == 0) {
      if (!finishes0) {
        return Kind.NONE;
      }
      lastBestRank = rank0;
      lastWorstRank = rank0;
      return kindOf(rank0, rank0, tsumo);
    }
    if (o.winPao(0)) {
      throw new IllegalStateException("Non-yakuman win cannot have pao");
    }
    if (fastPaths && finishes0 && rank0 == 0) {
      lastBestRank = 0;
      lastWorstRank = 0;
      return Kind.CERT_TOP;
    }
    int han0 = o.winHan(0);
    int fu = decodeFu(han0, o.winFuCode(0));
    int kyoku = o.kyoku();
    int observer = o.observer();
    DealerOutcome dealerOutcome = o.dealer() == 0 ? DealerOutcome.CONTINUES : DealerOutcome.PASSES;
    if (fastPaths) {
      int maxBase = ScoreMath.base(han0 + Tiles.COPIES * draws, fu, 0);
      if (ScoreMath.base(han0, fu, 0) == maxBase) {
        if (!finishes0) {
          return Kind.NONE;
        }
        lastBestRank = rank0;
        lastWorstRank = rank0;
        return kindOf(rank0, rank0, tsumo);
      }
      settleBase(o, settlement, maxBase, observer, absolute);
      boolean finishesMax = Progression.finishes(variant, kyoku, absolute, dealerOutcome);
      if (!finishes0 && !finishesMax) {
        return Kind.NONE;
      }
      int rankMax = Placement.rank(absolute, observer);
      if (edgeOnly) {
        if (!tsumo) {
          if (!finishesMax || rankMax != 0) {
            return Kind.NONE;
          }
        } else if (rank0 < seats - 1 && rankMax > 0) {
          return Kind.NONE;
        }
      }
      if (tsumo
          && finishes0
          && rank0 == seats - 1
          && rankMax == seats - 1
          && bottomFinishesAlways(kyoku, dealerOutcome, observer)) {
        lastBestRank = seats - 1;
        lastWorstRank = seats - 1;
        return Kind.CERT_BOTTOM;
      }
    }
    buildPublicPool(o, settlement, draws);
    if (fastPaths) {
      return classifyWithBitmask(
          o,
          settlement,
          draws,
          han0,
          fu,
          kyoku,
          observer,
          dealerOutcome,
          tsumo,
          finishes0,
          rank0,
          edgeOnly);
    }
    return classifyWithFullUra(
        o, settlement, draws, han0, fu, kyoku, observer, dealerOutcome, tsumo);
  }

  /** ツモ最下位が裏 0 枚でも最大裏ドラでも最下位のとき、間の全ての裏ドラ枚数でも終局する十分条件。 */
  private boolean bottomFinishesAlways(int kyoku, DealerOutcome dealerOutcome, int observer) {
    for (int seat = 0; seat < seats; seat++) {
      if (seat != observer && Progression.busts(absolute0[seat])) {
        return true;
      }
    }
    if (kyoku >= lastRound && dealerOutcome == DealerOutcome.PASSES) {
      return kyoku == finalRound
          || Progression.reachesReturn(variant, absolute[Placement.top(absolute)]);
    }
    return false;
  }

  private Kind classifyWithBitmask(
      Observation o,
      SettlementAssumption settlement,
      int draws,
      int han0,
      int fu,
      int kyoku,
      int observer,
      DealerOutcome dealerOutcome,
      boolean tsumo,
      boolean finishes0,
      int rank0,
      boolean edgeOnly) {
    int reachableMask = reachableUraMask(draws);
    if ((reachableMask & 1) != 0) {
      if (!finishes0) {
        return Kind.NONE;
      }
      if (edgeOnly) {
        if (!tsumo && rank0 > 0) {
          return Kind.NONE;
        }
        if (tsumo && rank0 > 0 && rank0 < seats - 1) {
          return Kind.NONE;
        }
      }
    }
    int bestRank = seats - 1;
    int worstRank = 0;
    int previousBase = -1;
    for (int mask = reachableMask; mask != 0; mask &= mask - 1) {
      int k = Integer.numberOfTrailingZeros(mask);
      int base = ScoreMath.base(han0 + k, fu, 0);
      if (base == previousBase) {
        continue;
      }
      previousBase = base;
      int rank;
      if (k == 0) {
        if (!finishes0) {
          return Kind.NONE;
        }
        rank = rank0;
      } else {
        settleBase(o, settlement, base, observer, absolute);
        if (!Progression.finishes(variant, kyoku, absolute, dealerOutcome)) {
          return Kind.NONE;
        }
        rank = Placement.rank(absolute, observer);
      }
      if (rank < bestRank) {
        bestRank = rank;
      }
      if (rank > worstRank) {
        worstRank = rank;
      }
      if (edgeOnly) {
        if (!tsumo && worstRank > 0) {
          return Kind.NONE;
        }
        if (tsumo && worstRank > 0 && bestRank < seats - 1) {
          return Kind.NONE;
        }
      }
    }
    lastBestRank = bestRank;
    lastWorstRank = worstRank;
    return kindOf(bestRank, worstRank, tsumo);
  }

  private Kind classifyWithFullUra(
      Observation o,
      SettlementAssumption settlement,
      int draws,
      int han0,
      int fu,
      int kyoku,
      int observer,
      DealerOutcome dealerOutcome,
      boolean tsumo) {
    if (ura == null) {
      ura = new UraDistribution();
    }
    ura.single(pool, uraCounts, draws);
    int bestRank = seats - 1;
    int worstRank = 0;
    int previousBase = -1;
    for (int k = 0; k <= ura.maxFirst(); k++) {
      if (ura.probability(k) == 0.0) {
        continue;
      }
      int base = ScoreMath.base(han0 + k, fu, 0);
      if (base == previousBase) {
        continue;
      }
      previousBase = base;
      settleBase(o, settlement, base, observer, absolute);
      if (!Progression.finishes(variant, kyoku, absolute, dealerOutcome)) {
        return Kind.NONE;
      }
      int rank = Placement.rank(absolute, observer);
      if (rank < bestRank) {
        bestRank = rank;
      }
      if (rank > worstRank) {
        worstRank = rank;
      }
    }
    lastBestRank = bestRank;
    lastWorstRank = worstRank;
    return kindOf(bestRank, worstRank, tsumo);
  }

  /** 公開残数プールから {@code draws} 枚を引いて到達可能な裏ドラ枚数 k のビットマスクを返す（k ≤ 20）。 */
  private int reachableUraMask(int draws) {
    Arrays.fill(grouped, 0);
    for (long bits = validKinds; bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      int tiles = pool[kind];
      if (tiles > 0) {
        grouped[uraCounts[kind]] += tiles;
      }
    }
    if (draws == 1) {
      int mask = 0;
      for (int v = 0; v <= Tiles.COPIES; v++) {
        if (grouped[v] > 0) {
          mask |= 1 << v;
        }
      }
      return mask;
    }
    Arrays.fill(reach, 0, draws + 1, 0);
    reach[0] = 1;
    for (int v = 0; v <= Tiles.COPIES; v++) {
      int tiles = grouped[v];
      if (tiles == 0) {
        continue;
      }
      for (int d = draws; d >= 1; d--) {
        int most = Math.min(d, tiles);
        int mask = reach[d];
        for (int j = 1; j <= most; j++) {
          mask |= reach[d - j] << (j * v);
        }
        reach[d] = mask;
      }
    }
    return reach[draws];
  }

  private Kind kindOf(int bestRank, int worstRank, boolean tsumo) {
    if (bestRank == 0 && worstRank == 0) {
      return Kind.CERT_TOP;
    }
    if (tsumo && bestRank == seats - 1 && worstRank == seats - 1) {
      return Kind.CERT_BOTTOM;
    }
    return Kind.CERT_MID;
  }

  /** 公開残数プールと和了者所有牌から、表示牌ごとの裏ドラ増加枚数を計算する。 */
  private void buildPublicPool(Observation o, SettlementAssumption settlement, int draws) {
    System.arraycopy(basePool, 0, pool, 0, Tiles.KINDS);
    o.handCounts(owned);
    for (long bits = validKinds; bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      pool[kind] -= owned[kind];
    }
    int indicators = o.indicators();
    for (int i = 0; i < indicators; i++) {
      pool[o.indicatorTile(i)]--;
    }
    int ownKans = 0;
    int ownNukis = 0;
    int events = o.events();
    for (int e = 0; e < events; e++) {
      o.readEvent(e, eventRecord);
      EventKind kind = eventRecord.kind();
      int author = eventRecord.author();
      if (kind == EventKind.DISCARD) {
        pool[eventRecord.discardTile()]--;
      } else if (kind == EventKind.MELD) {
        Meld meld = eventRecord.meld();
        switch (meld.type()) {
          case CHI -> {
            for (int j = 0; j < 3; j++) {
              int tile = meld.tile(j);
              if (tile != meld.called()) {
                pool[tile]--;
              }
              if (author == 0) {
                owned[tile]++;
              }
            }
          }
          case PON -> {
            pool[meld.base()] -= 2;
            if (author == 0) {
              owned[meld.base()] += 3;
            }
          }
          case DAIMINKAN -> {
            pool[meld.base()] -= 3;
            if (author == 0) {
              owned[meld.base()] += 4;
              ownKans++;
            }
          }
          case ANKAN -> {
            pool[meld.base()] -= 4;
            if (author == 0) {
              owned[meld.base()] += 4;
              ownKans++;
            }
          }
          case KAKAN -> {
            pool[meld.base()] -= 1;
            if (author == 0) {
              owned[meld.base()] += 1;
              ownKans++;
            }
          }
        }
      } else if (kind == EventKind.NUKI) {
        pool[Tiles.PEI]--;
        if (author == 0) {
          owned[Tiles.PEI]++;
          ownNukis++;
        }
      }
    }
    if (settlement == SettlementAssumption.SOLE_RON) {
      owned[o.lastEventTile()]++;
    }
    int ownedSum = 0;
    int poolSum = 0;
    Arrays.fill(uraCounts, 0);
    for (long bits = validKinds; bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      if (pool[kind] < 0) {
        throw new IllegalStateException(
            "Negative public pool count " + pool[kind] + " for tile " + kind);
      }
      poolSum += pool[kind];
      ownedSum += owned[kind];
      uraCounts[kind] = owned[doraOf[kind]];
    }
    if (ownedSum - ownKans - ownNukis != HAND_TILES) {
      throw new IllegalStateException(
          "Winner hand tile count invariant violated: " + (ownedSum - ownKans - ownNukis));
    }
    if (poolSum < draws) {
      throw new IllegalStateException(
          "Public pool has " + poolSum + " tiles for " + draws + " ura indicators");
    }
  }

  /** 基本点 {@code basePoints} で非役満の和了を精算し、絶対席順の精算後点数を書く。 */
  private void settleBase(
      Observation o, SettlementAssumption settlement, int basePoints, int observer, int[] into) {
    int dealer = observer + o.dealer();
    if (dealer >= seats) {
      dealer -= seats;
    }
    Arrays.fill(into, 0);
    if (settlement == SettlementAssumption.EXACT_TSUMO) {
      Settlement.tsumo(
          variant, basePoints, observer, dealer, o.honba(), RoundOutcome.NO_SEAT, into);
    } else {
      int loser = observer + o.lastEventSeat();
      if (loser >= seats) {
        loser -= seats;
      }
      Settlement.ron(
          variant, basePoints, observer, loser, dealer, o.honba(), RoundOutcome.NO_SEAT, into);
    }
    for (int seat = 0; seat < seats; seat++) {
      int abs = observer + seat;
      if (abs >= seats) {
        abs -= seats;
      }
      into[abs] += o.score(seat);
    }
    into[observer] += o.kyotaku() * Variant.DEPOSIT;
  }

  private static int findWinAction(Observation o) {
    int actions = o.legalActions();
    for (int a = 0; a < actions; a++) {
      Action.Type type = o.action(a).type();
      if (type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI) {
        return a;
      }
    }
    throw new IllegalStateException(
        "Observation declares " + o.wins() + " wins without a win action");
  }

  private static int decodeFu(int han, int fuCode) {
    if (han >= MANGAN_HAN) {
      return MANGAN_FU;
    }
    if (fuCode <= 0 || fuCode >= FU_TABLE.length) {
      throw new IllegalArgumentException("Invalid fu code " + fuCode + " for " + han + " han");
    }
    return FU_TABLE[fuCode];
  }
}
