package com.epsilon.prism.luck;

import com.epsilon.grp.Phi;
import com.epsilon.grp.RoundHistory;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.HypotheticalEnd;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.Objects;

/**
 * 点の列を Φ の単位で付けるときの、出来事の値。局の始まりの卓と今立直している席から、仮定した結果（ツモの和了か栄和）で局を精算し、
 * 次の卓の席ごとの Φ（半荘が終われば順位の効用）と、基準の卓の Φ の差を返す。基準は、立直した席だけが聴牌の荒牌流局で終わった卓。
 * 包は考えない。
 *
 * <p>値は、局の始まりの卓・立直した席・仮定した結果・Φ だけで決まる。牌山も引いた牌も読まない。どのスレッドから呼んでもよく、作業領域は
 * スレッドごとに持つ。
 */
public final class PhiStakes {

  private final ThreadLocal<Table> tables;

  /** @param phi 卓の値を求める Φ */
  public PhiStakes(Phi phi) {
    Objects.requireNonNull(phi, "phi");
    tables = ThreadLocal.withInitial(() -> new Table(phi));
  }

  /**
   * このスレッドの作業領域を、卓と立直に合わせて返す（このスレッドで次に呼ぶまで有効）。
   *
   * @param start 局の始まりの卓（返した作業領域を使う間は変えない）
   * @param history 終わった局までの各局の始まりの卓。Φ が履歴を読まなければ null でよい
   * @param riichiMask この局で立直が成立した席のマスク
   */
  Table bind(TableState start, RoundHistory history, int riichiMask) {
    Table table = tables.get();
    table.bind(start, history, riichiMask);
    return table;
  }

  /**
   * 1 スレッドの作業領域。求めた結果の値を控え、卓と立直が前に合わせたときと同じなら使い回す。卓は中身を全て比べるので、控えの値は
   * 求め直した値とビット単位で同じ。Φ が履歴を読むときは、合わせるたびに控えを捨てる。
   */
  static final class Table {
    private final Phi phi;
    private final Variant variant;
    private final int seats;
    private final boolean keeps;
    private final HypotheticalEnd end;

    private final int[] boundScores;
    private int boundRound = -1, boundHonba, boundKyotaku;

    private TableState start;
    private RoundHistory history;
    private int riichiMask;

    /** 基準の卓の席ごとの Φ（{@code referenced} のときだけ有効）。 */
    private final double[] reference;

    private boolean referenced;
    private final double[] next;

    /** 栄和の和了者の基本点（席ごと。和了しない席は 0）。 */
    private final int[] ronBases;

    /** 控えた結果。結果ごとに、和了者と放銃者の符号・席ごとの基本点・席ごとの値を持つ。 */
    private int[] heads = new int[16];

    private int[] bases;
    private double[] values;
    private int used;

    private Table(Phi phi) {
      this.phi = phi;
      variant = phi.variant();
      seats = variant.seats();
      keeps = !phi.needsHistory();
      end = new HypotheticalEnd(variant);
      boundScores = new int[seats];
      reference = new double[seats];
      next = new double[seats];
      ronBases = new int[seats];
      bases = new int[heads.length * seats];
      values = new double[heads.length * seats];
    }

    private void bind(TableState start, RoundHistory history, int riichiMask) {
      boolean same = keeps && riichiMask == this.riichiMask && sameTable(start);
      this.start = start;
      this.history = history;
      if (same) return;
      this.riichiMask = riichiMask;
      boundRound = start.round();
      boundHonba = start.honba();
      boundKyotaku = start.kyotaku();
      for (int seat = 0; seat < seats; seat++) boundScores[seat] = start.score(seat);
      used = 0;
      referenced = false;
    }

    private boolean sameTable(TableState table) {
      if (table.round() != boundRound
          || table.honba() != boundHonba
          || table.kyotaku() != boundKyotaku) return false;
      for (int seat = 0; seat < seats; seat++)
        if (table.score(seat) != boundScores[seat]) return false;
      return true;
    }

    /**
     * ツモの和了の値を返す。
     *
     * @param seat 値を受け取る席
     * @param winner 和了する席
     * @param base 基本点
     */
    double tsumo(int seat, int winner, int base) {
      Arrays.fill(ronBases, 0);
      ronBases[winner] = base;
      int head = head(1 << winner, RoundOutcome.NO_SEAT);
      int at = find(head);
      if (at < 0) {
        reference();
        at = remember(head, end.tsumo(start, riichiMask, winner, base, RoundOutcome.NO_SEAT));
      }
      return values[at * seats + seat];
    }

    /** 栄和の和了者を空にする。{@link #winner} で和了者を足してから {@link #ron} を呼ぶ。 */
    void clearWinners() {
      Arrays.fill(ronBases, 0);
    }

    /** 栄和の和了者を足す。 */
    void winner(int seat, int base) {
      ronBases[seat] = base;
    }

    /**
     * 足した和了者の全員が栄和したときの値を返す。三家和のあるルールで和了者がその人数なら、三家和の途中流局の値を返す。
     *
     * @param seat 値を受け取る席
     * @param loser 放銃する席
     */
    double ron(int seat, int loser) {
      int winners = 0;
      for (int other = 0; other < seats; other++) if (ronBases[other] != 0) winners |= 1 << other;
      int head = head(winners, loser);
      int at = find(head);
      if (at < 0) {
        reference();
        at = remember(head, settleRon(winners, loser));
      }
      return values[at * seats + seat];
    }

    private TableState settleRon(int winners, int loser) {
      if (Integer.bitCount(winners) == AbortiveDraw.TRIPLE_RON_WINNERS
          && variant.aborts(AbortiveDraw.TRIPLE_RON))
        return end.abortive(start, riichiMask, AbortiveDraw.TRIPLE_RON);
      RoundOutcome outcome = end.outcome();
      boolean first = true;
      for (int other = 0; other < seats; other++) {
        if ((winners & (1 << other)) == 0) continue;
        if (first) outcome.setRon(ronBases[other], other, loser, RoundOutcome.NO_SEAT, riichiMask);
        else outcome.addRonWinner(ronBases[other], other, RoundOutcome.NO_SEAT);
        first = false;
      }
      return end.settle(start);
    }

    /** 和了者のマスクと放銃者（ツモなら席なし）を 1 つの符号にする。 */
    private int head(int winners, int loser) {
      return winners << 3 | (loser + 1);
    }

    /** 今の和了者の基本点と符号が同じ控えの位置を返す。無ければ −1。 */
    private int find(int head) {
      for (int at = 0; at < used; at++) {
        if (heads[at] != head) continue;
        int from = at * seats;
        if (Arrays.equals(bases, from, from + seats, ronBases, 0, seats)) return at;
      }
      return -1;
    }

    /** 精算した卓の値と基準の値の差を控え、控えの位置を返す。 */
    private int remember(int head, TableState table) {
      phi.values(history, table, next);
      if (used == heads.length) {
        heads = Arrays.copyOf(heads, 2 * used);
        bases = Arrays.copyOf(bases, 2 * used * seats);
        values = Arrays.copyOf(values, 2 * used * seats);
      }
      int at = used++;
      heads[at] = head;
      System.arraycopy(ronBases, 0, bases, at * seats, seats);
      for (int seat = 0; seat < seats; seat++)
        values[at * seats + seat] = next[seat] - reference[seat];
      return at;
    }

    private void reference() {
      if (referenced) return;
      phi.values(history, end.exhaustive(start, riichiMask, riichiMask, 0), reference);
      referenced = true;
    }
  }
}
