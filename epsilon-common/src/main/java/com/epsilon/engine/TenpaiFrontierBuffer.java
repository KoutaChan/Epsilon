package com.epsilon.engine;

/** 一ツモと打牌を経て、役あり聴牌へ到達できる牌種を保持する再利用結果。 */
public final class TenpaiFrontierBuffer {
  private long furitenFreeRon;
  private long furitenOnlyRon;
  private long tsumo;

  void clear() {
    furitenFreeRon = 0L;
    furitenOnlyRon = 0L;
    tsumo = 0L;
  }

  void add(int tile, boolean freeRon, boolean furitenRon, boolean tsumoReachable) {
    long bit = 1L << tile;
    if (freeRon) furitenFreeRon |= bit;
    else if (furitenRon) furitenOnlyRon |= bit;
    if (tsumoReachable) tsumo |= bit;
  }

  public long furitenFreeRon() {
    return furitenFreeRon;
  }

  public long furitenOnlyRon() {
    return furitenOnlyRon;
  }

  public long tsumo() {
    return tsumo;
  }
}
