package com.epsilon.reviewer.engine.view;

import com.epsilon.core.Tile;
import java.util.HashMap;
import java.util.Map;

/**
 * 牌の記号。萬・筒・索は数字と m・p・s、赤の 5 は後ろに r、字牌は E S W N P F C。34 の牌種と 3 枚の赤の 37 の記号を定数で持ち、
 * 局面・候補・副露の写しは新しい文字列を作らない。
 */
public final class TileCode {
  private static final String HONORS = "ESWNPFC";
  private static final String SUITS = "mps";
  private static final String[] PLAIN = new String[Tile.NUM_TILE_TYPES];
  private static final String[] RED = new String[SUITS.length()];

  /** 記号 → 牌種 × 2 + 赤でない（赤は偶数）。37 の記号だけを持つ。 */
  private static final Map<String, Integer> CODES = new HashMap<>();

  static {
    for (int type = 0; type < Tile.NUM_TILE_TYPES; type++) {
      PLAIN[type] =
          type >= Tile.TON
              ? String.valueOf(HONORS.charAt(type - Tile.TON))
              : (type % Tile.TILE_TYPES_PER_SUIT + 1) + String.valueOf(SUITS.charAt(type / 9));
      CODES.put(PLAIN[type], type * 2 + 1);
    }
    for (int suit = 0; suit < RED.length; suit++) {
      RED[suit] = "5" + SUITS.charAt(suit) + "r";
      CODES.put(RED[suit], (suit * Tile.TILE_TYPES_PER_SUIT + 4) * 2);
    }
  }

  private TileCode() {}

  /** 牌種と赤の記号。赤は 5 だけ。 */
  public static String of(int type, boolean red) {
    if (!red) return PLAIN[type];
    if (!Tile.canBeAka(type)) throw new IllegalArgumentException("Only fives can be red: " + type);
    return RED[Tile.suitOf(type)];
  }

  /** 37 の記号のどれかか（[1-9][mps]、赤は 5 だけ r 付き、字牌は ESWNPFC）。 */
  public static boolean isValid(String code) {
    return code != null && CODES.containsKey(code);
  }

  public static int type(String code) {
    return order(code) / 2;
  }

  public static boolean red(String code) {
    return order(code) % 2 == 0;
  }

  /** 手牌の並びの順の値（protocol/result.ts の tileOrder と同じ）。牌種の順で、赤の 5 は同じ 5 の前。 */
  public static int order(String code) {
    Integer order = CODES.get(code);
    if (order == null) throw new IllegalArgumentException("Invalid tile code: " + code);
    return order;
  }
}
