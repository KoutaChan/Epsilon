package com.epsilon.mahjong.rules;

import java.util.Objects;

/**
 * 局の結果。精算（Settlement）と局の進め方（Progression）への入力で、エンジンの局末も、対局の記録も、GRP の数え上げも、この形で
 * 渡す。使い回す可変の値で、ルールは作るときに決まる。同時に使えるのは 1 つのスレッドだけ。作った直後は、全員が不聴の荒牌（点が動かない）。
 *
 * <p>席のマスクは、ビット s が席 s を表す。その終わり方に関係しないフィールドは、席なら {@link #NO_SEAT}、マスクと数なら 0、途中流局の
 * 種類なら null。結果を書くメソッドは値を確かめてから書くので、例外を投げたときは前の結果がそのまま残る。
 */
public final class RoundOutcome {

  /** 席が無いことを表す（包の無い和了の包の席、ロン以外の局の放銃者）。 */
  public static final int NO_SEAT = -1;

  private final Variant variant;

  /** 和了ごとの和了者・基本点・包の席。放銃者から手番が近い順に並ぶ。 */
  private final int[] winners;
  private final int[] bases;
  private final int[] paos;

  private EndKind kind;
  private AbortiveDraw abortive;
  private int winnerCount;
  private int loser;
  private int riichiMask;
  private int tenpaiMask;
  private int nagashiMask;

  public RoundOutcome(Variant variant) {
    this.variant = Objects.requireNonNull(variant, "variant");
    int capacity = variant.seats() - 1;
    if (variant.aborts(AbortiveDraw.TRIPLE_RON)) {
      capacity = Math.min(capacity, AbortiveDraw.TRIPLE_RON_WINNERS - 1);
    }
    winners = new int[capacity];
    bases = new int[capacity];
    paos = new int[capacity];
    setExhaustive(0, 0, 0);
  }

  public Variant variant() {
    return variant;
  }

  /**
   * ツモ和了の結果にする。
   *
   * @param base 複合役満を含む和了全体の基本点（{@link BaseClass} の値）
   * @param winner 和了者
   * @param pao 包の席。無ければ {@link #NO_SEAT}
   * @param riichiMask この局で立直が成立した席（供託を払った席）
   */
  public void setTsumo(int base, int winner, int pao, int riichiMask) {
    requireWin(base, winner, pao);
    requireMask(riichiMask);
    reset(EndKind.TSUMO, riichiMask);
    winnerCount = 1;
    put(0, base, winner, pao);
  }

  /**
   * 打牌（加槓・抜いた北を含む）へのロンの結果にする。ほかの和了者は {@link #addRonWinner} で足す。包の席が放銃したロンは、
   * 包の無いロンとして書く。
   *
   * @param base 複合役満を含む和了全体の基本点（{@link BaseClass} の値）
   * @param winner 和了者
   * @param loser 放銃者
   * @param pao 包の席。無ければ {@code NO_SEAT}。放銃者を渡すと IllegalArgumentException
   * @param riichiMask この局で立直が成立した席（供託を払った席）
   */
  public void setRon(int base, int winner, int loser, int pao, int riichiMask) {
    requireWin(base, winner, pao);
    Objects.checkIndex(loser, variant.seats());
    requireNotLoser(winner, loser);
    Settlement.requireLoserNotLiable(loser, pao);
    requireMask(riichiMask);
    reset(EndKind.RON, riichiMask);
    this.loser = loser;
    winnerCount = 1;
    put(0, base, winner, pao);
  }

  /**
   * ロンの和了者を足す。和了者は、足した順に関係なく放銃者から手番が近い順に並ぶ。三家和のあるルールでは、和了者は 2 人まで。
   *
   * @param base 複合役満を含む和了全体の基本点（{@link BaseClass} の値）
   * @param winner 和了者
   * @param pao 包の席。無ければ {@link #NO_SEAT}
   * @throws IllegalStateException ロンの結果でないとき
   * @throws IllegalArgumentException 包の席が放銃者のとき（包の席が放銃したロンは、包の無いロンとして書く）、その席がすでに
   *     和了者のとき、和了者が上限を超えるとき
   */
  public void addRonWinner(int base, int winner, int pao) {
    if (kind != EndKind.RON) {
      throw new IllegalStateException("Only a ron takes more winners, this outcome is " + kind);
    }
    requireWin(base, winner, pao);
    requireNotLoser(winner, loser);
    Settlement.requireLoserNotLiable(loser, pao);
    for (int i = 0; i < winnerCount; i++) {
      if (winners[i] == winner) {
        throw new IllegalArgumentException("Seat " + winner + " already won this ron");
      }
    }
    if (winnerCount == winners.length) {
      throw new IllegalArgumentException(variant + " settles at most " + winners.length + " ron winners");
    }
    int index = winnerCount++;
    while (index > 0 && turnsFromLoser(winners[index - 1]) > turnsFromLoser(winner)) {
      put(index, bases[index - 1], winners[index - 1], paos[index - 1]);
      index--;
    }
    put(index, base, winner, pao);
  }

  /**
   * 荒牌の流局の結果にする。
   *
   * @param tenpaiMask 聴牌した席
   * @param nagashiMask 流し満貫になった席
   * @param riichiMask この局で立直が成立した席
   */
  public void setExhaustive(int tenpaiMask, int nagashiMask, int riichiMask) {
    requireMask(tenpaiMask);
    requireMask(nagashiMask);
    requireMask(riichiMask);
    reset(EndKind.EXHAUSTIVE, riichiMask);
    this.tenpaiMask = tenpaiMask;
    this.nagashiMask = nagashiMask;
  }

  /** 途中流局の結果にする。このルールで起きない途中流局を渡すと IllegalArgumentException。 */
  public void setAbortive(AbortiveDraw draw, int riichiMask) {
    if (draw == null || !variant.aborts(draw)) {
      throw new IllegalArgumentException(variant + " has no abortive draw " + draw);
    }
    requireMask(riichiMask);
    reset(EndKind.ABORTIVE, riichiMask);
    abortive = draw;
  }

  /** 別の結果をコピーする。ルールが違えば IllegalArgumentException。 */
  public void copyFrom(RoundOutcome other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    System.arraycopy(other.winners, 0, winners, 0, winners.length);
    System.arraycopy(other.bases, 0, bases, 0, bases.length);
    System.arraycopy(other.paos, 0, paos, 0, paos.length);
    kind = other.kind;
    abortive = other.abortive;
    winnerCount = other.winnerCount;
    loser = other.loser;
    riichiMask = other.riichiMask;
    tenpaiMask = other.tenpaiMask;
    nagashiMask = other.nagashiMask;
  }

  public EndKind kind() {
    return kind;
  }

  /** 途中流局の種類。途中流局でなければ null。 */
  public AbortiveDraw abortive() {
    return abortive;
  }

  /** 和了者の数。ツモは 1、ロンは 1 以上、流局は 0。 */
  public int winnerCount() {
    return winnerCount;
  }

  /**
   * 和了者。放銃者から手番が近い順に並ぶ。本場と場の供託は最初の和了者が受け取る。
   *
   * @param i 何番目の和了か（0 始まり）
   */
  public int winner(int i) {
    return winners[Objects.checkIndex(i, winnerCount)];
  }

  /** 放銃者。ロンでなければ {@link #NO_SEAT}。 */
  public int loser() {
    return loser;
  }

  /**
   * 複合役満を含む和了全体の基本点。
   *
   * @param i 何番目の和了か（0 始まり）
   */
  public int base(int i) {
    return bases[Objects.checkIndex(i, winnerCount)];
  }

  /**
   * 包の席。無ければ {@link #NO_SEAT}。
   *
   * @param i 何番目の和了か（0 始まり）
   */
  public int pao(int i) {
    return paos[Objects.checkIndex(i, winnerCount)];
  }

  /** この局で立直が成立した席（供託を払った席）。 */
  public int riichiMask() {
    return riichiMask;
  }

  /** 荒牌で聴牌した席。 */
  public int tenpaiMask() {
    return tenpaiMask;
  }

  /** 荒牌で流し満貫になった席。 */
  public int nagashiMask() {
    return nagashiMask;
  }

  private void reset(EndKind kind, int riichiMask) {
    this.kind = kind;
    this.riichiMask = riichiMask;
    abortive = null;
    winnerCount = 0;
    loser = NO_SEAT;
    tenpaiMask = 0;
    nagashiMask = 0;
  }

  private void put(int index, int base, int winner, int pao) {
    bases[index] = base;
    winners[index] = winner;
    paos[index] = pao;
  }

  private int turnsFromLoser(int seat) {
    return Math.floorMod(seat - loser, variant.seats());
  }

  private void requireWin(int base, int winner, int pao) {
    BaseClass.of(base);
    Objects.checkIndex(winner, variant.seats());
    Settlement.requirePao(variant, base, winner, pao);
  }

  private static void requireNotLoser(int winner, int loser) {
    if (winner == loser) {
      throw new IllegalArgumentException("Seat " + winner + " cannot ron its own discard");
    }
  }

  private void requireMask(int mask) {
    if ((mask & ~variant.seatMask()) != 0) {
      throw new IllegalArgumentException("Mask " + Integer.toBinaryString(mask) + " has seats outside " + variant);
    }
  }

  @Override
  public String toString() {
    StringBuilder text = new StringBuilder("RoundOutcome[").append(variant).append(' ').append(kind);
    if (abortive != null) {
      text.append(' ').append(abortive);
    }
    for (int i = 0; i < winnerCount; i++) {
      text.append(" winner=").append(winners[i]).append(" base=").append(bases[i]);
      if (paos[i] != NO_SEAT) {
        text.append(" pao=").append(paos[i]);
      }
    }
    if (loser != NO_SEAT) {
      text.append(" loser=").append(loser);
    }
    return text.append(" riichi=").append(Integer.toBinaryString(riichiMask))
        .append(" tenpai=").append(Integer.toBinaryString(tenpaiMask))
        .append(" nagashi=").append(Integer.toBinaryString(nagashiMask))
        .append(']')
        .toString();
  }
}
