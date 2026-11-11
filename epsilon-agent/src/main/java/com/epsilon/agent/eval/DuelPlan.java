package com.epsilon.agent.eval;

import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * 決闘の計画。2 スロットの卓を判定の時点ごとに見て、止め方と判定に使う量から止めるかを決める（{@link Duel}）。
 *
 * @param arena 卓の計画。スロット 0 が候補、スロット 1 が相手
 * @param stop 止め方
 * @param score 判定に使う量
 */
public record DuelPlan(ArenaPlan arena, StopRule stop, Score score) {

  /** 止め方が固定のときのチャンクの牌山の数（3 と 4 の倍数）。長い卓でも、チャンクの切れ目で止めて続きから打てる。 */
  public static final int FIXED_CHUNK = 6144;

  public DuelPlan {
    Objects.requireNonNull(arena, "arena");
    Objects.requireNonNull(score, "score");
    check(arena.variant(), arena.slots().size(), arena.seating(), arena.walls(), stop);
  }

  /** @return チャンクの牌山の数。止め方が判定か精度なら判定の間隔、固定なら {@link #FIXED_CHUNK} と牌山の数の小さい方 */
  public int chunk() {
    return chunk(stop, arena.walls());
  }

  /**
   * 決闘の形を確かめる。
   *
   * @param slots 卓のスロットの数
   * @param walls 牌山の数
   * @throws IllegalArgumentException スロットが 2 つでないとき、牌山が 2 つより少ないとき、牌山 1 つで 1 半荘ずつ打つ卓で止め方が
   *     固定でないか牌山の数が席の数の倍数でないとき、判定の間隔が 2 牌山より短いか牌山の数を割り切れないとき
   */
  static void check(Variant variant, int slots, Seating seating, int walls, StopRule stop) {
    Objects.requireNonNull(variant, "variant");
    Objects.requireNonNull(seating, "seating");
    Objects.requireNonNull(stop, "stop");
    if (slots != 2) {
      throw new IllegalArgumentException("A duel seats two slots, not " + slots);
    }
    if (walls < 2) {
      throw new IllegalArgumentException("A duel needs at least two walls for a standard error, not " + walls);
    }
    if (seating == Seating.BY_GAME) {
      if (!(stop instanceof StopRule.Fixed)) {
        throw new IllegalArgumentException("A " + seating + " duel plays all its walls, so it cannot stop by " + stop);
      }
      if (walls % variant.seats() != 0) {
        throw new IllegalArgumentException(
            "A " + seating + " duel plays walls in multiples of the " + variant.seats() + " seats, not " + walls);
      }
    }
    int checkEvery = stop.checkEvery(walls);
    if (checkEvery < 2 || walls % checkEvery != 0) {
      throw new IllegalArgumentException(
          walls + " walls are not a multiple of the check interval " + checkEvery + " of at least two walls");
    }
  }

  /** @return チャンクの牌山の数 */
  static int chunk(StopRule stop, int walls) {
    return stop instanceof StopRule.Fixed ? Math.min(FIXED_CHUNK, walls) : stop.checkEvery(walls);
  }
}
