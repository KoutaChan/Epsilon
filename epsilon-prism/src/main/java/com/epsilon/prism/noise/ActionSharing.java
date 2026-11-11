package com.epsilon.prism.noise;

import com.epsilon.agent.play.CrnScheme;

/** 判断で手を引く乱数を、親と分岐でどう共有するか。 */
public enum ActionSharing {
  /** 乱数のキーでは、判断を手番・応答・鳴いた後の打牌の 3 つのクラスに分ける。一様分布を混ぜるかを 2 段の乱数で決めて引く（{@link CrnScheme#TWO_STAGE_FLOOR}）。 */
  TWO_STAGE_FLOOR(CrnScheme.TWO_STAGE_FLOOR),
  /** 乱数のキーでは、判断を手番系と応答系の 2 つのクラスに分ける。一様分布を混ぜた分布の上で 1 回の指数競争をする（{@link CrnScheme#SINGLE_RACE}）。 */
  SINGLE_RACE(CrnScheme.SINGLE_RACE),
  /** 分岐の局のキーにソルトを混ぜ、分岐の残りの手を親と独立に引く。 */
  INDEPENDENT(null);

  private final CrnScheme scheme;

  ActionSharing(CrnScheme scheme) {
    this.scheme = scheme;
  }

  /**
   * 卓のコアに渡す共通乱数の使い方を返す。
   *
   * @return 共通乱数の使い方
   * @throws IllegalStateException 卓のコアの使い方で表せない共有のしかた（{@link #INDEPENDENT}）のとき
   */
  public CrnScheme scheme() {
    if (scheme == null) {
      throw new IllegalStateException(this + " has no table scheme");
    }
    return scheme;
  }
}
