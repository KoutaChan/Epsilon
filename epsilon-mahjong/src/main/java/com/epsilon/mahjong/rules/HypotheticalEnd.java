package com.epsilon.mahjong.rules;

import java.util.Objects;

/**
 * 仮定した終わり方で局を精算する。呼び出し元が渡す基本点と席から局の結果を作り、エンジンの局末と同じく {@link Settlement#apply}
 * と {@link Progression#next} で点の動きと次局の卓を求める。牌山は読まない。局の結果・点の動き・次局の卓は使い回すので、
 * 次の呼び出しまで有効。どの呼び出しもメモリを確保しない。同時に使えるのは 1 つのスレッドだけ。
 *
 * <p>局の始まりの卓には、精算の前の卓（{@code Engine.handStart()}）を渡す。この局で成立した立直は、立直した席のマスクで渡す
 * （供託は精算の中で数える）。ダブロンの結果を自分で組むときは、{@link #outcome} に和了者を足してから {@link #settle} を呼ぶ。
 */
public final class HypotheticalEnd {

  private final Variant variant;
  private final RoundOutcome outcome;
  private final int[] delta;
  private final TableState next;

  public HypotheticalEnd(Variant variant) {
    this.variant = Objects.requireNonNull(variant, "variant");
    outcome = new RoundOutcome(variant);
    delta = new int[variant.seats()];
    next = new TableState(variant);
  }

  public Variant variant() {
    return variant;
  }

  /** 仮定した局の結果。書き換えてから {@link #settle} を呼んでよい。 */
  public RoundOutcome outcome() {
    return outcome;
  }

  /**
   * 仮定した局の結果を精算して、次局の卓を求める。
   *
   * @param start 局の始まりの卓（読むだけ）
   * @return 次局の始まりか終局の卓。次の呼び出しまで有効
   * @throws IllegalArgumentException ルールが合わないとき、または終局の卓を渡したとき
   */
  public TableState settle(TableState start) {
    Settlement.apply(variant, start, outcome, delta);
    Progression.next(start, outcome, delta, next);
    return next;
  }

  /**
   * 仮定したツモ和了を精算する。
   *
   * @param start 局の始まりの卓
   * @param riichiMask この局で立直が成立した席のマスク
   * @param winner 和了者
   * @param base 基本点
   * @param pao 包の席。無ければ {@link RoundOutcome#NO_SEAT}
   * @return 次局の卓。次の呼び出しまで有効
   */
  public TableState tsumo(TableState start, int riichiMask, int winner, int base, int pao) {
    outcome.setTsumo(base, winner, pao, riichiMask);
    return settle(start);
  }

  /**
   * 仮定したロン（和了者は 1 人）を精算する。
   *
   * @param start 局の始まりの卓
   * @param riichiMask この局で立直が成立した席のマスク。立直の宣言牌へのロンでは、宣言した席を含めない
   * @param loser 放銃者
   * @param winner 和了者
   * @param base 基本点
   * @param pao 包の席。無ければ {@link RoundOutcome#NO_SEAT}
   * @return 次局の卓。次の呼び出しまで有効
   */
  public TableState ron(TableState start, int riichiMask, int loser, int winner, int base, int pao) {
    outcome.setRon(base, winner, loser, pao, riichiMask);
    return settle(start);
  }

  /**
   * 仮定したダブロンを精算する。本場と場の供託は、放銃者から手番が近いほうの和了者が受け取る。
   *
   * @param start 局の始まりの卓
   * @param riichiMask この局で立直が成立した席のマスク
   * @param loser 放銃者
   * @param first 1 人目の和了者
   * @param firstBase 1 人目の基本点
   * @param second 2 人目の和了者
   * @param secondBase 2 人目の基本点
   * @return 次局の卓。次の呼び出しまで有効
   */
  public TableState doubleRon(
      TableState start, int riichiMask, int loser, int first, int firstBase, int second, int secondBase) {
    outcome.setRon(firstBase, first, loser, RoundOutcome.NO_SEAT, riichiMask);
    outcome.addRonWinner(secondBase, second, RoundOutcome.NO_SEAT);
    return settle(start);
  }

  /**
   * 仮定した荒牌の流局を精算する。
   *
   * @param start 局の始まりの卓
   * @param riichiMask この局で立直が成立した席のマスク
   * @param tenpaiMask 聴牌した席のマスク
   * @param nagashiMask 流し満貫になった席のマスク
   * @return 次局の卓。次の呼び出しまで有効
   */
  public TableState exhaustive(TableState start, int riichiMask, int tenpaiMask, int nagashiMask) {
    outcome.setExhaustive(tenpaiMask, nagashiMask, riichiMask);
    return settle(start);
  }

  /**
   * 仮定した途中流局（三家和など）を精算する。
   *
   * @param start 局の始まりの卓
   * @param riichiMask この局で立直が成立した席のマスク
   * @param draw 途中流局の種類
   * @return 次局の卓。次の呼び出しまで有効
   */
  public TableState abortive(TableState start, int riichiMask, AbortiveDraw draw) {
    outcome.setAbortive(draw, riichiMask);
    return settle(start);
  }

  /** 直前の精算で、その席の点が局の始まりからいくら動いたか。 */
  public int delta(int seat) {
    return delta[Objects.checkIndex(seat, delta.length)];
  }

  /** 直前の精算で求めた次局の卓。 */
  public TableState next() {
    return next;
  }
}
