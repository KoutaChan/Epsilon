package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;

/**
 * 状態のエンコーダーの出力。方策と価値の集約は同じ値を読み、方策ヘッドはさらに牌と席の記憶も読む。
 *
 * @param round 局の表現 [行, 幅]
 * @param entities 局・4 席の要約・34 種類の牌・河 96・面子 16 のトークン [行, 151, 幅]
 * @param entityMask トークンが実在すれば 1 [行, 151]
 * @param tiles 牌の種類の表現 [行, 34, 幅]
 * @param players 席ごとの記憶。Nano では要約・河 24・面子 4 のトークン [行, 4, 29, 幅]、Major ではそれを射影したキーとバリュー
 *     [行, 4, 29, 2 × 文脈の幅]
 * @param playerMask 席ごとの記憶のトークンが実在すれば 1 [行, 4, 29]
 */
record Memory(
    NDArray round, NDArray entities, NDArray entityMask, NDArray tiles, NDArray players, NDArray playerMask) {

  /** 値はそのままで、勾配を止めた記憶を返す。価値の集約に渡すときに使う。 */
  Memory detached() {
    return new Memory(
        round.stopGradient(),
        entities.stopGradient(),
        entityMask,
        tiles.stopGradient(),
        players.stopGradient(),
        playerMask);
  }
}
