package com.epsilon.agent.play;

import java.util.ArrayList;
import java.util.List;

/**
 * 卓で打つ 1 半荘。牌山、局のキー、手の抽選のキーは牌山の番号から作り、割り当ての番号は使わない。割り当ての番号が決めるのは、どの席に
 * どの打ち手が座るかだけ。同じ牌山を別の割り当てで打つ半荘は、同じ牌山が配られ、同じ判断の時点で同じ乱数を引く。
 *
 * @param wall 牌山の番号（卓の中で、割り当てとの組が一意）
 * @param placement 割り当ての番号（0 以上）
 */
public record GameKey(long wall, int placement) {

  public GameKey {
    if (placement < 0) {
      throw new IllegalArgumentException("Placement must not be negative: " + placement);
    }
  }

  /**
   * 牌山ごとに、割り当て 0 の半荘を 1 つずつ並べる。
   *
   * @param walls 牌山の番号（打つ順）
   * @return 半荘（牌山と同じ順）
   */
  public static List<GameKey> onePerWall(List<Long> walls) {
    List<GameKey> games = new ArrayList<>(walls.size());
    for (long wall : walls) {
      games.add(new GameKey(wall, 0));
    }
    return games;
  }
}
