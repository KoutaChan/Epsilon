package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import java.util.Objects;

/**
 * 局末の情報。局の結果、和了ごとの点、点の動き、各席の待ち、精算した後の卓を持つ。エンジンが持ち、局ごとに書き直す。エンジンが
 * 局末か終局で止まっている間だけ有効。点の動きは局の結果から {@code Settlement.apply} で求め、次の卓は {@code Progression.next}
 * で求める。
 */
public final class RoundEnd {

  private final Variant variant;
  private final RoundOutcome outcome;

  /** 和了ごとの点。 */
  private final WinScore[] scores;

  private final int[] delta;
  private final long[] waits;
  private final TableState next;

  RoundEnd(Variant variant) {
    this.variant = variant;
    outcome = new RoundOutcome(variant);
    scores = new WinScore[variant.seats() - 1];
    for (int i = 0; i < scores.length; i++) {
      scores[i] = new WinScore();
    }
    delta = new int[variant.seats()];
    waits = new long[variant.seats()];
    next = new TableState(variant);
  }

  public RoundOutcome outcome() {
    return outcome;
  }

  /**
   * 和了の翻・符・役・ドラ・赤・裏・抜きと基本点。立直の和了は裏ドラを含む。
   *
   * @param i 何番目の和了か（局の結果の和了者の順）
   */
  public WinScore score(int i) {
    return scores[Objects.checkIndex(i, outcome.winnerCount())];
  }

  /** 局の始めの卓から見た、その席の点の動き。本場・供託・罰符と、この局で払った立直の供託を含む。 */
  public int delta(int seat) {
    return delta[seat];
  }

  /**
   * 局末のその席の待ち（役と振聴は見ない、形の上の和了牌）。ツモの後の 14 枚相当の手（ツモ和了・九種九牌）では、ツモった牌を
   * 除いた 13 枚の待ち。
   */
  public long waits(int seat) {
    return waits[seat];
  }

  /** 精算した後の卓。半荘が続くなら次の局の始めの卓、終わるなら終局の卓。 */
  public TableState next() {
    return next;
  }

  /** 和了の点をコピーする。局の結果より先に書いてよい。 */
  void setScore(int i, WinScore score) {
    scores[i].copyFrom(score);
  }

  /** {@code Settlement.apply} が書く、点の動きの配列。 */
  int[] delta() {
    return delta;
  }

  void setWaits(int seat, long kinds) {
    waits[seat] = kinds;
  }

  void copyFrom(RoundEnd other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    outcome.copyFrom(other.outcome);
    for (int i = 0; i < scores.length; i++) {
      scores[i].copyFrom(other.scores[i]);
    }
    System.arraycopy(other.delta, 0, delta, 0, delta.length);
    System.arraycopy(other.waits, 0, waits, 0, waits.length);
    next.copyFrom(other.next);
  }
}
