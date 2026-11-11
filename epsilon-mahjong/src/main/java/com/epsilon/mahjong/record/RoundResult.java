package com.epsilon.mahjong.record;

import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 牌譜の局の結果。配列はコピーして持つ。包の席が放銃したロンは、包の無いロンとして書く。放銃者を包の席にした和了は
 * IllegalArgumentException。
 *
 * <p>記録が内訳を持たない結果もある。点を持たない和了（{@code Win.unscored}）と、終わり方を持たない流局（終わり方が null）の 2 つ。
 * どちらの場合も {@code ReplayDriver} は和了者・放銃者と点の移動だけを照合し、内訳はエンジンの局末から取る。
 *
 * @param kind 局の終わり方。終わり方を持たない流局は null
 * @param abortive 途中流局（ABORTIVE）の種類。ほかは null
 * @param wins 和了。放銃者から見て手番の近い順
 * @param loser ロン（RON）の放銃者。ほかは {@code RoundOutcome.NO_SEAT}
 * @param tenpaiMask 荒牌（EXHAUSTIVE）で聴牌だった席のマスク。ほかは 0
 * @param nagashiMask 荒牌（EXHAUSTIVE）で流し満貫だった席のマスク。ほかは 0
 * @param deltas 点の移動（絶対席の順）。本場・供託・罰符・この局の立直の供託を含む
 * @param uraIndicators 局末にめくった裏ドラ表示牌の物理 id（表示牌の順）。立直した和了の局だけ持つ
 */
public record RoundResult(
    EndKind kind,
    AbortiveDraw abortive,
    List<Win> wins,
    int loser,
    int tenpaiMask,
    int nagashiMask,
    int[] deltas,
    int[] uraIndicators) {

  /**
   * 和了 1 つ。役満なら役は役満の役だけで、翻・符・ドラ・赤・裏は 0。点を持たない和了は和了者の席だけを持ち、ほかは全て 0 で、
   * 包の席は無い。
   *
   * @param seat 和了者の席
   * @param base 和了全体の基本点（複合役満を含む）
   * @param yaku 役のビット（{@code ScoringYaku.bit}）の論理和
   * @param han 翻
   * @param fu 符
   * @param dora 表ドラと抜きドラの翻の和（天鳳の記録の数え方）
   * @param aka 赤ドラの翻
   * @param ura 裏ドラの翻
   * @param pao 包の席。無ければ {@code RoundOutcome.NO_SEAT}
   */
  public record Win(int seat, int base, long yaku, int han, int fu, int dora, int aka, int ura, int pao) {

    public Win {
      if (seat < 0 || pao < RoundOutcome.NO_SEAT || pao == seat) {
        throw new IllegalArgumentException("Bad winner " + seat + " or liable seat " + pao);
      }
      if (base != 0) {
        BaseClass.of(base);
      } else if (yaku != 0 || han != 0 || fu != 0 || dora != 0 || aka != 0 || ura != 0 || pao != RoundOutcome.NO_SEAT) {
        throw new IllegalArgumentException("A win without points holds only the seat " + seat);
      }
    }

    /**
     * 記録が和了者しか持たない、点を持たない和了。
     *
     * @param seat 和了者の席
     */
    public static Win unscored(int seat) {
      return new Win(seat, 0, 0, 0, 0, 0, 0, 0, RoundOutcome.NO_SEAT);
    }

    /** 記録が点・役・包を持つ和了か。 */
    public boolean scored() {
      return base != 0;
    }

    /**
     * 記録の翻・符か役満の倍率から、基本点を求めた和了を作る。役満なら翻・符・ドラ・赤・裏を 0 にする。
     *
     * @param yakuman 役満の倍率。役満でなければ 0
     */
    public static Win of(int seat, long yaku, int han, int fu, int dora, int aka, int ura, int yakuman, int pao) {
      if (yakuman > 0) {
        return new Win(seat, ScoreMath.base(0, 0, yakuman), yaku, 0, 0, 0, 0, 0, pao);
      }
      return new Win(seat, ScoreMath.base(han, fu, 0), yaku, han, fu, dora, aka, ura, pao);
    }

    /**
     * 本場・供託・包を除いた和了点。ツモなら払う席全ての払いの和で、三人打ちはツモ損になる。点を持たない和了は
     * IllegalStateException。
     *
     * @param variant 払い方を決めるルール（四人打ちか三人打ちか）
     * @param dealer 親の席
     * @param tsumo ツモ和了なら true、ロンなら false
     * @return 和了者が受け取る点
     */
    public int points(Variant variant, int dealer, boolean tsumo) {
      if (!scored()) {
        throw new IllegalStateException("The win of seat " + seat + " has no points in the record");
      }
      int[] delta = new int[variant.seats()];
      if (tsumo) {
        Settlement.tsumo(variant, base, seat, dealer, 0, RoundOutcome.NO_SEAT, delta);
      } else {
        Settlement.ron(variant, base, seat, (seat + 1) % variant.seats(), dealer, 0, RoundOutcome.NO_SEAT, delta);
      }
      return delta[seat];
    }
  }

  public RoundResult {
    wins = List.copyOf(wins);
    if (kind == null) {
      if (abortive != null || !wins.isEmpty() || loser != RoundOutcome.NO_SEAT || (tenpaiMask | nagashiMask) != 0) {
        throw new IllegalArgumentException("A draw without details holds only the point moves");
      }
    } else {
      check(kind, abortive, wins, loser, tenpaiMask, nagashiMask);
    }
    deltas = deltas.clone();
    uraIndicators = uraIndicators.clone();
  }

  /**
   * 終わり方（荒牌か途中流局か、聴牌と流し満貫の席）を記録が持たない流局。
   *
   * @param deltas 絶対席の順の点の移動
   */
  public static RoundResult drawWithoutDetails(int[] deltas) {
    return new RoundResult(null, null, List.of(), RoundOutcome.NO_SEAT, 0, 0, deltas, new int[0]);
  }

  /** 和了で終わったか（終わり方を持たない流局は和了でない）。 */
  public boolean win() {
    return kind != null && kind.win();
  }

  /** 記録が内訳を持つか。終わり方があり、どの和了も点を持てば true。 */
  public boolean detailed() {
    if (kind == null) {
      return false;
    }
    for (Win win : wins) {
      if (!win.scored()) {
        return false;
      }
    }
    return true;
  }

  /** コピーを返す。 */
  @Override
  public int[] deltas() {
    return deltas.clone();
  }

  public int uraCount() {
    return uraIndicators.length;
  }

  /** 指定した番号の表示牌に対応する、裏ドラ表示牌の物理 id。 */
  public int ura(int index) {
    return uraIndicators[index];
  }

  /** コピーを返す。 */
  @Override
  public int[] uraIndicators() {
    return uraIndicators.clone();
  }

  /**
   * この結果を、エンジンの局の結果（{@code RoundOutcome}）に書く。記録が内訳を持たなければ IllegalStateException。
   *
   * @param riichiMask この局で立直が成立した席のマスク
   * @param into 書く先
   */
  public void outcome(int riichiMask, RoundOutcome into) {
    if (!detailed()) {
      throw new IllegalStateException("The record has no details of " + this);
    }
    switch (kind) {
      case TSUMO -> {
        Win win = wins.get(0);
        into.setTsumo(win.base(), win.seat(), win.pao(), riichiMask);
      }
      case RON -> {
        Win first = wins.get(0);
        into.setRon(first.base(), first.seat(), loser, first.pao(), riichiMask);
        for (int i = 1; i < wins.size(); i++) {
          Win win = wins.get(i);
          into.addRonWinner(win.base(), win.seat(), win.pao());
        }
      }
      case EXHAUSTIVE -> into.setExhaustive(tenpaiMask, nagashiMask, riichiMask);
      case ABORTIVE -> into.setAbortive(abortive, riichiMask);
    }
  }

  /**
   * エンジンの局末が、立直の成立した席・局の結果・和了の点・点の移動のどれもこの結果と同じか。終わり方を持たない流局では、立直の
   * 席のほかに照合するのは、流局であることと点の移動だけ。点を持たない和了では和了者の席だけを照合し、点・役・包は照合しない。
   *
   * @param end エンジンの局末
   * @param riichiMask この局で立直が成立した席のマスク
   */
  public boolean matches(RoundEnd end, int riichiMask) {
    RoundOutcome outcome = end.outcome();
    if (outcome.riichiMask() != riichiMask) {
      return false;
    }
    if (kind == null) {
      return !outcome.kind().win() && sameDeltas(end);
    }
    if (outcome.kind() != kind
        || outcome.abortive() != abortive
        || outcome.winnerCount() != wins.size()
        || outcome.loser() != loser
        || outcome.tenpaiMask() != tenpaiMask
        || outcome.nagashiMask() != nagashiMask) {
      return false;
    }
    for (int i = 0; i < wins.size(); i++) {
      Win win = wins.get(i);
      if (outcome.winner(i) != win.seat()) {
        return false;
      }
      if (!win.scored()) {
        continue;
      }
      WinScore score = end.score(i);
      if (outcome.base(i) != win.base()
          || outcome.pao(i) != win.pao()
          || score.yaku() != win.yaku()
          || score.han() != win.han()
          || score.fu() != win.fu()
          || score.dora() + score.nuki() != win.dora()
          || score.aka() != win.aka()
          || score.ura() != win.ura()) {
        return false;
      }
    }
    return sameDeltas(end);
  }

  /** エンジンの局末の点の移動が、この結果と同じか。 */
  public boolean sameDeltas(RoundEnd end) {
    for (int seat = 0; seat < deltas.length; seat++) {
      if (end.delta(seat) != deltas[seat]) {
        return false;
      }
    }
    return true;
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof RoundResult result
        && kind == result.kind
        && abortive == result.abortive
        && wins.equals(result.wins)
        && loser == result.loser
        && tenpaiMask == result.tenpaiMask
        && nagashiMask == result.nagashiMask
        && Arrays.equals(deltas, result.deltas)
        && Arrays.equals(uraIndicators, result.uraIndicators);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        kind, abortive, wins, loser, tenpaiMask, nagashiMask, Arrays.hashCode(deltas), Arrays.hashCode(uraIndicators));
  }

  /** 終わり方のある結果で、各フィールドがその終わり方と合うかを確かめる。合わなければ IllegalArgumentException。 */
  private static void check(
      EndKind kind, AbortiveDraw abortive, List<Win> wins, int loser, int tenpaiMask, int nagashiMask) {
    if ((kind == EndKind.ABORTIVE) != (abortive != null)) {
      throw new IllegalArgumentException("Only an abortive draw names one: " + kind + " " + abortive);
    }
    boolean winsMatch =
        switch (kind) {
          case TSUMO -> wins.size() == 1;
          case RON -> !wins.isEmpty();
          case EXHAUSTIVE, ABORTIVE -> wins.isEmpty();
        };
    if (!winsMatch) {
      throw new IllegalArgumentException(kind + " cannot have " + wins.size() + " wins");
    }
    if (kind == EndKind.RON ? loser < 0 : loser != RoundOutcome.NO_SEAT) {
      throw new IllegalArgumentException(kind + " cannot have loser " + loser);
    }
    for (int i = 0; kind == EndKind.RON && i < wins.size(); i++) {
      if (wins.get(i).pao() == loser) {
        throw new IllegalArgumentException("Seat " + loser + " deals in, so it cannot also be liable for the win");
      }
    }
    if (kind != EndKind.EXHAUSTIVE && (tenpaiMask | nagashiMask) != 0) {
      throw new IllegalArgumentException("Only an exhaustive draw has tenpai or nagashi seats");
    }
  }

  @Override
  public String toString() {
    return "RoundResult[" + (kind == null ? "DRAW" : kind) + (abortive == null ? "" : " " + abortive) + " wins=" + wins
        + (loser == RoundOutcome.NO_SEAT ? "" : " loser=" + loser)
        + " tenpai=" + Integer.toBinaryString(tenpaiMask) + " nagashi=" + Integer.toBinaryString(nagashiMask)
        + " deltas=" + Arrays.toString(deltas) + " ura=" + Arrays.toString(uraIndicators) + "]";
  }
}
