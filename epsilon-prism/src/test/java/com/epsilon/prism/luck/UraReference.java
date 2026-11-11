package com.epsilon.prism.luck;

import com.epsilon.grp.Phi;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.HypotheticalEnd;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * 裏ドラの期待値のテストで使う基準。裏ドラ表示牌の種類の組を全て数え上げ、組ごとに基本点を書き直して精算し、精算後の卓を Φ で値にする。
 * 動的計画法は使わず、基本点が同じ組もまとめない。精算は {@link HypotheticalEnd} のツモ・ロン・ダブロンのメソッドで行う。
 *
 * <p>裏ドラ表示牌の枚数は、局の偶然のイベントの記録にある裏ドラのイベントの数。プールは、誰にも見えない牌に、めくった裏ドラ表示牌を
 * 戻したもの。
 */
final class UraReference {

  /** 数え上げる裏ドラ表示牌の枚数の上限。これより多い局末は求めない。 */
  static final int MAX_DRAWS = 3;

  private static final double EPSILON = 1e-12;

  private final Phi phi;
  private final int seats;
  private final TileSet set;
  private final HypotheticalEnd hypothetical;
  private final HandShape hand = new HandShape();
  private final double[] scratch;
  private final int[] owned = new int[Tiles.KINDS];

  /** 立直していて役満でない和了者が、表示牌の種類ごとに数える裏ドラの数。添字は [その和了者の何人目か][表示牌の種類]。 */
  private final int[][] counts = new int[2][Tiles.KINDS];

  /** 立直していて役満でない和了者が、局の結果で何番目の和了か。 */
  private final int[] varying = new int[2];

  private final int[] picked = new int[Tiles.KINDS];
  private final int[] kinds = new int[Tiles.KINDS];
  private final int[] bases = new int[2];
  private final Set<Long> distinctBases = new HashSet<>();

  /** 種類ごとの、プールの枚数。 */
  final int[] pool = new int[Tiles.KINDS];

  /** 種類ごとの、めくった裏ドラ表示牌の枚数。 */
  final int[] realizedPick = new int[Tiles.KINDS];

  final double[] expected;
  final double[] variance;

  /** めくった裏ドラ表示牌の組で精算した値。実際の値と一致することをテストで確かめる。 */
  final double[] realized;

  int uraEvents;
  int indicatorEvents;
  int varyingCount;
  double totalWeight;
  private RoundOutcome outcome;
  private RoundEnd end;
  private TableState start;
  private int kindCount;

  UraReference(Phi phi) {
    this.phi = phi;
    Variant variant = phi.variant();
    seats = variant.seats();
    set = variant.tiles();
    hypothetical = new HypotheticalEnd(variant);
    scratch = new double[seats];
    expected = new double[seats];
    variance = new double[seats];
    realized = new double[seats];
  }

  /**
   * 立直の和了の局末について、期待値・分散と、めくった組での値を求める。
   *
   * @return 求めたら true。立直の和了でないか、裏ドラ表示牌が上限より多ければ false
   */
  boolean compute(FullView full, RoundEnd end, TableState start) {
    this.end = end;
    this.start = start;
    outcome = end.outcome();
    if (!outcome.kind().win()) {
      return false;
    }
    boolean riichi = false;
    varyingCount = 0;
    for (int i = 0; i < outcome.winnerCount(); i++) {
      int winner = outcome.winner(i);
      if (full.seat(winner).riichi(winner) == RiichiState.NONE) {
        continue;
      }
      riichi = true;
      if (end.score(i).yakuman() == 0) {
        varying[varyingCount++] = i;
      }
    }
    if (!riichi) {
      return false;
    }
    ChanceLog log = full.chance();
    Arrays.fill(pool, 0);
    Arrays.fill(realizedPick, 0);
    uraEvents = 0;
    indicatorEvents = 0;
    for (int index = 0; index < log.size(); index++) {
      if (log.kind(index) == ChanceKind.URA) {
        int kind = Tiles.kind(log.tile(index));
        pool[kind]++;
        realizedPick[kind]++;
        uraEvents++;
      } else if (log.kind(index) == ChanceKind.INDICATOR) {
        indicatorEvents++;
      }
    }
    if (uraEvents > MAX_DRAWS) {
      return false;
    }
    kindCount = 0;
    for (long bits = set.kinds(); bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      pool[kind] += full.unseen(kind);
      if (pool[kind] > 0) {
        kinds[kindCount++] = kind;
      }
    }
    for (int v = 0; v < varyingCount; v++) {
      loadCounts(full, outcome.winner(varying[v]), counts[v]);
    }
    Arrays.fill(expected, 0);
    Arrays.fill(variance, 0);
    distinctBases.clear();
    totalWeight = 0;
    Arrays.fill(picked, 0);
    double norm = choose(total(), uraEvents);
    enumerate(0, uraEvents, 1 / norm);
    for (int seat = 0; seat < seats; seat++) {
      variance[seat] = Math.max(0, variance[seat] - expected[seat] * expected[seat]);
    }
    System.arraycopy(realizedPick, 0, picked, 0, Tiles.KINDS);
    value(realized);
    return true;
  }

  /** 基本点の組の数（確率が 0 でない組だけを数える）。 */
  int groups() {
    return varyingCount == 0 ? 1 : distinctBases.size();
  }

  /** 立直していて役満でない {@code v} 人目の和了者について、めくった裏ドラ表示牌で数えた裏ドラの数。 */
  int realizedUra(int v) {
    int ura = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      ura += realizedPick[kind] * counts[v][kind];
    }
    return ura;
  }

  /** 立直していて役満でない {@code v} 人目の和了者が、局の結果で何番目の和了か。 */
  int varying(int v) {
    return varying[v];
  }

  private int total() {
    int total = 0;
    for (int index = 0; index < kindCount; index++) {
      total += pool[kinds[index]];
    }
    return total;
  }

  private void enumerate(int index, int left, double weight) {
    if (left == 0) {
      value(scratch);
      for (int seat = 0; seat < seats; seat++) {
        expected[seat] += weight * scratch[seat];
        variance[seat] += weight * scratch[seat] * scratch[seat];
      }
      totalWeight += weight;
      distinctBases.add(((long) bases[0] << 32) | (bases[1] & 0xffffffffL));
      return;
    }
    if (index == kindCount) {
      return;
    }
    int kind = kinds[index];
    for (int c = 0; c <= Math.min(left, pool[kind]); c++) {
      picked[kind] = c;
      enumerate(index + 1, left - c, weight * choose(pool[kind], c));
    }
    picked[kind] = 0;
  }

  /** 選んだ表示牌の組で基本点を書き直して精算し、精算後の卓の Φ を書く。 */
  private void value(double[] out) {
    bases[0] = 0;
    bases[1] = 0;
    int[] winnerBases = new int[outcome.winnerCount()];
    for (int i = 0; i < winnerBases.length; i++) {
      winnerBases[i] = outcome.base(i);
    }
    for (int v = 0; v < varyingCount; v++) {
      int ura = 0;
      for (int kind = 0; kind < Tiles.KINDS; kind++) {
        ura += picked[kind] * counts[v][kind];
      }
      WinScore score = end.score(varying[v]);
      bases[v] = ScoreMath.base(score.han() - score.ura() + ura, score.fu(), 0);
      winnerBases[varying[v]] = bases[v];
    }
    int mask = outcome.riichiMask();
    TableState next;
    if (outcome.kind() == EndKind.TSUMO) {
      next = hypothetical.tsumo(start, mask, outcome.winner(0), winnerBases[0], outcome.pao(0));
    } else if (outcome.winnerCount() == 1) {
      next = hypothetical.ron(start, mask, outcome.loser(), outcome.winner(0), winnerBases[0], outcome.pao(0));
    } else if (outcome.pao(0) == RoundOutcome.NO_SEAT && outcome.pao(1) == RoundOutcome.NO_SEAT) {
      next =
          hypothetical.doubleRon(
              start, mask, outcome.loser(), outcome.winner(0), winnerBases[0], outcome.winner(1), winnerBases[1]);
    } else {
      RoundOutcome copy = hypothetical.outcome();
      copy.setRon(winnerBases[0], outcome.winner(0), outcome.loser(), outcome.pao(0), mask);
      copy.addRonWinner(winnerBases[1], outcome.winner(1), outcome.pao(1));
      next = hypothetical.settle(start);
    }
    phi.values(null, next, out);
  }

  /** 和了の 14 枚（純手牌・面子・抜いた北・ロン牌）について、表示牌の種類ごとに数える裏ドラの数を書く。 */
  private void loadCounts(FullView full, int winner, int[] out) {
    Arrays.fill(owned, 0);
    full.copyHand(winner, hand);
    int tiles = 0;
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      owned[kind] = hand.count(kind);
      tiles += owned[kind];
    }
    SeatView view = full.seat(winner);
    for (int m = 0; m < view.meldCount(winner); m++) {
      Meld meld = Meld.of(view.meld(winner, m));
      for (int k = 0; k < meld.size(); k++) {
        owned[meld.tile(k)]++;
      }
      // 槓の 4 枚目は和了形の 14 枚の数に入れない（裏ドラは 4 枚とも数える）
      tiles += Math.min(meld.size(), 3);
    }
    owned[Tiles.PEI] += view.nuki(winner);
    if (tiles == HandShape.MAX_SIZE - 1) {
      owned[view.last().tile()]++;
    } else if (tiles != HandShape.MAX_SIZE) {
      throw new AssertionError("Seat " + winner + " won with " + tiles + " tiles");
    }
    Arrays.fill(out, 0);
    for (long bits = set.kinds(); bits != 0; bits &= bits - 1) {
      int kind = Long.numberOfTrailingZeros(bits);
      out[kind] = owned[set.doraOf(kind)];
    }
  }

  /** 数え上げた組の確率の和が 1 か。 */
  boolean normalized() {
    return Math.abs(totalWeight - 1) <= EPSILON;
  }

  private static double choose(int n, int k) {
    if (k < 0 || k > n) {
      return 0;
    }
    double value = 1;
    for (int j = 1; j <= k; j++) {
      value = value * (n - k + j) / j;
    }
    return value;
  }
}
