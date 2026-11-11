package com.epsilon.prism.input;

import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.facts.WaitScores;
import com.epsilon.agent.facts.WaitScores.Method;
import com.epsilon.agent.facts.WaitScores.WinningTile;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.score.WinScore;

/**
 * 手を選んだ後に聴牌している手を、待ちごとに採点する。待ちごとに、（ロン・ツモ）×（ダマ・その遷移の立直）×（赤でない和了牌・
 * 赤 5 の和了牌）の点を {@link WaitScores} で求める（裏ドラは数えない）。採点の条件はダマと立直で 1 つずつ持ち、立直の条件で採点するのは
 * 立直する遷移だけ。
 *
 * <p>採点の条件は公開された情報だけで作る（自風・場風・ドラ表示牌・立直。一発・海底などは数えない）。赤 5 の和了牌で採点するのは、
 * その色の赤がまだ見えていない 5 の待ちだけ。メモリを割り当てない。
 */
public final class WaitScoring {

  /** ダマか立直か。 */
  public enum Stance {
    DAMA,
    /** その遷移の立直の状態（立直・ダブル立直）。立直しない遷移では採点しない。 */
    RIICHI
  }

  private final WaitScores dama;
  private final WaitScores declared;
  private boolean riichi;

  public WaitScoring(Variant variant) {
    dama = new WaitScores(variant);
    declared = new WaitScores(variant);
  }

  /**
   * 待ちを採点する。
   *
   * @param hand 手を選んだ後の手（13 枚相当で聴牌）
   * @param waitMask 形の上の待ち
   * @param tiles 種類ごとの枚数（見えていない赤を読む）
   * @param o 判断の記録（ドラ表示牌を読む）
   * @param seatWind 自風の牌の種類
   * @param transitionRiichi その遷移の立直の状態
   */
  public void score(
      AfterHand hand,
      long waitMask,
      TileCounts tiles,
      Observation o,
      int seatWind,
      RiichiState transitionRiichi) {
    riichi = transitionRiichi != RiichiState.NONE;
    load(dama, hand, waitMask, tiles, o, seatWind, RiichiState.NONE);
    if (riichi) {
      load(declared, hand, waitMask, tiles, o, seatWind, transitionRiichi);
    }
  }

  /** 採点した待ちの数。 */
  public int waits() {
    return dama.waits();
  }

  /** {@code wait} 番目の待ち牌（牌の番号の小さい順）。 */
  public int waitTile(int wait) {
    return dama.wait(wait);
  }

  /** 和了できるか（役があるか）。 */
  public boolean available(int wait, Stance stance, Method method, WinningTile tile) {
    return basePoints(wait, stance, method, tile) > 0;
  }

  /** 基本点（役がなければ 0）。 */
  public int basePoints(int wait, Stance stance, Method method, WinningTile tile) {
    return score(wait, stance, method, tile).base();
  }

  /** 翻数（裏ドラなし。役満は 0）。 */
  public int han(int wait, Stance stance, Method method, WinningTile tile) {
    return score(wait, stance, method, tile).han();
  }

  private WinScore score(int wait, Stance stance, Method method, WinningTile tile) {
    if (stance == Stance.RIICHI && !riichi) {
      throw new IllegalStateException("riichi stance was not scored");
    }
    return (stance == Stance.DAMA ? dama : declared).score(wait, method, tile);
  }

  private static void load(
      WaitScores scores,
      AfterHand hand,
      long waitMask,
      TileCounts tiles,
      Observation o,
      int seatWind,
      RiichiState riichi) {
    WinContext context = scores.context();
    context.clear();
    hand.writeTo(context);
    context.setWinds(seatWind, o.roundWindTile());
    for (int index = 0; index < o.indicators(); index++) {
      context.addIndicator(o.indicatorTile(index));
    }
    context.setRiichi(riichi, false);
    scores.load(waitMask, tiles.unseenRed());
  }
}
