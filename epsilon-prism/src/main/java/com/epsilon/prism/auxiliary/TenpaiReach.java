package com.epsilon.prism.auxiliary;

import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.River;
import java.util.Arrays;
import java.util.Objects;

/**
 * 経路ごと・席ごとに、最後に聴牌だった出来事と、最後にロンの役がある聴牌（立直は役に数える）だった出来事の番号を持つ（補助の頭の聴牌への
 * 到達の教師。{@link HandEndFacts#reachedTenpai}）。
 *
 * <p>出来事の番号はエンジンの公開の出来事の番号（打牌・副露・加槓の連番で、記録 O の出来事と同じ数え方）。打牌のたびに {@link #discarded}
 * を呼ぶと、その打牌の後の 13 枚を見て、聴牌ならその打牌の番号を書く。判断の時点の出来事の数（O の頭）を e とすると、番号が e 以上なら判断の
 * 後に届いた。聴牌と役の決まりは {@link TenpaiProbe} のとおりで、役の有無はドラに依らない。
 *
 * <p>分岐は親の写し（{@link #TenpaiReach(TenpaiReach)}）で始める。局が替わると番号は 0 から数え直すので、局を続ける経路（主経路）は次局の
 * 前に {@link #reset} する。打牌の確かめは割り当てをしない（確かめる道具はスレッドごとに 1 つ持つ）。
 */
public final class TenpaiReach {

  /** 番号がない（この局でまだその形になっていない）ことを表す値。 */
  public static final int NONE = -1;

  private static final ThreadLocal<TenpaiProbe> PROBES = ThreadLocal.withInitial(TenpaiProbe::new);

  private final int[] tenpai = new int[GameState.NUM_PLAYERS];
  private final int[] yakuTenpai = new int[GameState.NUM_PLAYERS];

  /** 番号のない状態（局の始め）で作る。 */
  public TenpaiReach() {
    reset();
  }

  /**
   * 写しを作る（分岐を始めるとき、局末の事実に残すとき）。
   *
   * @param other 写す元
   */
  public TenpaiReach(TenpaiReach other) {
    System.arraycopy(other.tenpai, 0, tenpai, 0, GameState.NUM_PLAYERS);
    System.arraycopy(other.yakuTenpai, 0, yakuTenpai, 0, GameState.NUM_PLAYERS);
  }

  /** 全ての席の番号を消す（次局の前）。 */
  public void reset() {
    Arrays.fill(tenpai, NONE);
    Arrays.fill(yakuTenpai, NONE);
  }

  /**
   * 席 {@code seat} の打牌が終わった局面を見て、その打牌の後の手が聴牌なら打牌の番号を書く。打牌の後、席がまた牌を引く前に呼ぶ（その打牌で
   * 局が終わったときも、局末の事実を作る前に呼ぶ）。
   *
   * @param state 打牌の後の局面
   * @param seat 打牌した席
   * @throws IllegalStateException 席の河が空か、手が 13 枚でないか、打牌に出来事の番号がないとき
   */
  public void discarded(GameState state, int seat) {
    River river = state.river(seat);
    if (river.size() == 0) {
      throw new IllegalStateException("seat " + seat + " has not discarded");
    }
    Hand hand = state.hand(seat);
    if (hand.effectiveTileCount() != 13) {
      throw new IllegalStateException(
          "seat " + seat + " holds " + hand.effectiveTileCount() + " tiles after a discard");
    }
    int event = river.event(river.size() - 1);
    TenpaiProbe probe = PROBES.get();
    if (!probe.tenpai(hand)) {
      return;
    }
    tenpai[seat] = event;
    if (probe.yakuTenpai(state, seat)) {
      yakuTenpai[seat] = event;
    }
  }

  /**
   * 席が最後に聴牌だった出来事の番号を返す。
   *
   * @param seat 席
   * @return 番号（なければ {@link #NONE}）
   */
  public int lastTenpaiEvent(int seat) {
    return tenpai[Objects.checkIndex(seat, GameState.NUM_PLAYERS)];
  }

  /**
   * 席が最後にロンの役がある聴牌だった出来事の番号を返す。
   *
   * @param seat 席
   * @return 番号（なければ {@link #NONE}）
   */
  public int lastYakuTenpaiEvent(int seat) {
    return yakuTenpai[Objects.checkIndex(seat, GameState.NUM_PLAYERS)];
  }
}
