package com.epsilon.agent.play;

import com.epsilon.mahjong.util.SeedMixer;

/**
 * 卓の乱数の種を作る。牌山の種と局のキーは、run の種、コホートの種、牌山の番号、局の序数だけで決まる。割り当ての番号、実行の順、
 * 同時に進める半荘の数、ワーカーの数には依らない。ソルトを変えると、同じ種と牌山の番号でも牌山と手の抽選が変わる。
 *
 * <pre>
 * derive(base, salt, c1, …, cn) = mix64(… mix64(mix64(base ^ salt) ^ mix64(c1)) … ^ mix64(cn))
 * </pre>
 */
public final class TableSeeds {

  private static final long WALL = 0x57414c4cL, HAND_KEY = 0x48414e44L;

  private TableSeeds() {}

  /** 半荘の牌山の種（{@code derive(runSeed, WALL, wall)}）。 */
  public static long wall(long runSeed, long wall) {
    return next(SeedMixer.mix64(runSeed ^ WALL), wall);
  }

  /**
   * 局のキー（{@code derive(iterationSeed, HAND_KEY, wall, handOrdinal, 0)}）。主経路と、その局の写し・分岐が共有する。末尾の 0 も
   * キーの式の一部で、変えると全てのキーが変わる。
   */
  public static long handKey(long iterationSeed, long wall, int handOrdinal) {
    return next(next(next(SeedMixer.mix64(iterationSeed ^ HAND_KEY), wall), handOrdinal), 0);
  }

  private static long next(long state, long coordinate) {
    return SeedMixer.mix64(state ^ SeedMixer.mix64(coordinate));
  }
}
