package com.epsilon.mahjong.engine;

/** エンジンが止まる時点。エンジンを進める呼び出しは、止まった時点をこのどれかで返す。 */
public enum Boundary {
  DECISIONS,
  /** 局の精算の後。 */
  ROUND_END,
  GAME_END
}
