package com.epsilon.mahjong.rules;

import java.util.Arrays;

/**
 * 局の始まり（精算の後）か、終局の卓。Progression が書き、GRP が読む。使い回す可変の値で、ルールは作るときに決まる。同時に使えるのは 1 つの
 * スレッドだけ。
 */
public final class TableState {

  private final Variant variant;
  private final int[] scores;
  private int round;
  private int honba;
  private int kyotaku;
  private boolean gameOver;

  /** 半荘の始め（東 1 局 0 本場、供託なし、全員が配給点）の卓を作る。 */
  public TableState(Variant variant) {
    this.variant = variant;
    scores = new int[variant.seats()];
    Arrays.fill(scores, variant.startPoints());
  }

  public Variant variant() {
    return variant;
  }

  /** 局の番号。連荘なら前の局と同じ。 */
  public int round() {
    return round;
  }

  public int dealer() {
    return variant.dealer(round);
  }

  public int honba() {
    return honba;
  }

  public int kyotaku() {
    return kyotaku;
  }

  /** その席の点。終局なら、残った供託をトップに渡した後の点。 */
  public int score(int seat) {
    return scores[seat];
  }

  public boolean gameOver() {
    return gameOver;
  }

  /** その席の順位（0 始まり、点の多い順）。同点なら起家に近い席が上（Placement と同じ）。 */
  public int rank(int seat) {
    return Placement.rank(scores, seat);
  }

  /** トップ（順位 0）の席。 */
  public int top() {
    return Placement.top(scores);
  }

  /** 渡した局の始まりと同じ卓なら true。終局でなく、ルール・局・本場・供託・点が全て同じときだけ true になる。 */
  public boolean startsAt(RoundStart start) {
    if (gameOver
        || start.variant() != variant
        || round != start.round()
        || honba != start.honba()
        || kyotaku != start.kyotaku()) {
      return false;
    }
    for (int seat = 0; seat < scores.length; seat++) {
      if (scores[seat] != start.score(seat)) {
        return false;
      }
    }
    return true;
  }

  /**
   * 終局の卓で、点が渡した最終の点と同じなら true。
   *
   * @param finalScores 絶対席の順に並べた最終の点
   */
  public boolean endsWith(int[] finalScores) {
    return gameOver && Arrays.equals(scores, finalScores);
  }

  /** 渡した局の始まりの卓にする。ルールが違えば IllegalArgumentException。 */
  public void set(RoundStart start) {
    requireVariant(start.variant());
    round = start.round();
    honba = start.honba();
    kyotaku = start.kyotaku();
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = start.score(seat);
    }
    gameOver = false;
  }

  /**
   * 局の始まりの卓にする。{@link #set(RoundStart)} と同じだが、メモリを確保しない（GRP が局末の卓をコピーするときに使う）。
   * 値の不変条件は {@link RoundStart} と同じで、破っていれば IllegalArgumentException。点はコピーする。
   */
  public void set(int round, int honba, int kyotaku, int[] scores) {
    RoundStart.check(variant, round, honba, kyotaku, scores);
    assign(round, honba, kyotaku, scores);
  }

  /**
   * もしもの局の始まりの卓にする。局の番号・本場・供託・点の数は RoundStart と同じように確かめるが、点と供託の和が配給点の和に
   * なることは求めない。点はコピーし、メモリは確保しない。どれかが合わなければ IllegalArgumentException を投げ、卓は変えない。
   *
   * @param round 局の通し番号
   * @param honba 本場
   * @param kyotaku 供託の本数
   * @param scores 絶対席の順に並べた点
   */
  public void setScenario(int round, int honba, int kyotaku, int[] scores) {
    RoundStart.checkRanges(variant, round, honba, kyotaku, scores);
    assign(round, honba, kyotaku, scores);
  }

  /** 別の卓をコピーする。ルールが違えば IllegalArgumentException。 */
  public void copyFrom(TableState other) {
    requireVariant(other.variant);
    round = other.round;
    honba = other.honba;
    kyotaku = other.kyotaku;
    System.arraycopy(other.scores, 0, scores, 0, scores.length);
    gameOver = other.gameOver;
  }

  /** 点の配列そのもの。Progression がコピーせずに読むためのもので、呼び出し元は書き換えない。 */
  int[] scoreArray() {
    return scores;
  }

  /** 局・本場・供託を書き、終局でない卓にする。点は変えない。 */
  void setRound(int round, int honba, int kyotaku) {
    this.round = round;
    this.honba = honba;
    this.kyotaku = kyotaku;
    gameOver = false;
  }

  void setScore(int seat, int points) {
    scores[seat] = points;
  }

  /** 終局にする。残った供託はトップに渡す（Placement.awardDeposits）。 */
  void finish() {
    Placement.awardDeposits(scores, kyotaku);
    kyotaku = 0;
    gameOver = true;
  }

  private void requireVariant(Variant other) {
    if (other != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other);
    }
  }

  /** 確かめた後の値で、局の始まりの卓にする。点はコピーする。 */
  private void assign(int round, int honba, int kyotaku, int[] scores) {
    this.round = round;
    this.honba = honba;
    this.kyotaku = kyotaku;
    System.arraycopy(scores, 0, this.scores, 0, this.scores.length);
    gameOver = false;
  }

  @Override
  public String toString() {
    return "TableState[" + variant + " round=" + round + " honba=" + honba + " kyotaku=" + kyotaku
        + " scores=" + Arrays.toString(scores) + (gameOver ? " game over" : "") + "]";
  }
}
