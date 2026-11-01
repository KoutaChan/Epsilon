package com.epsilon.prism.luck;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.prism.core.Stage;

/**
 * 出来事の層の鍵を出す。鍵は観測 O° の出来事の前の状態だけの関数で、結果を読まない。
 *
 * <p>層は {@value #COUNT} 個: 出来事の種類 3 × 行為者の向聴の類 4（0・1・2・3 以上） × 聴牌している他家の数 3（0・1・2 以上） × 他家の
 * 立直の有無 2 × 山の段 3（{@link Stage}）。鍵はこの順の混合基数の番号。向聴は手（副露を除く）と副露の数から出す最小の向聴で、行為者は
 * 出来事の前の手（ツモの前の 13 枚）で数える。
 *
 * <p>解析器を持つので、スレッドごとに 1 つ持つ。割り当てをしない。
 */
public final class LuckStratum {

  public static final int COUNT = 216;

  /** 定義の記述（標本の形式の指紋に入れる）。 */
  static final String DEFINITION =
      "event x actorShanten(0,1,2,3+) x tenpaiOpponents(0,1,2+) x opponentRiichi x Stage";

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int SHANTEN_CLASSES = 4;
  private static final int TENPAI_CLASSES = 3;
  private static final int STAGES = Stage.values().length;

  static {
    if (LuckEvent.values().length * SHANTEN_CLASSES * TENPAI_CLASSES * 2 * STAGES != COUNT) {
      throw new IllegalStateException("stratum count does not match its dimensions");
    }
  }

  private final HandShapeAnalyzer shapes = new HandShapeAnalyzer();
  private final int[] counts = new int[Tile.NUM_TILE_TYPES];

  /**
   * 出来事の層の鍵を返す。
   *
   * @param observation 出来事の O°
   * @return 鍵（{@value #COUNT} 未満）
   */
  public int of(LuckObservation observation) {
    int shanten = Math.min(shanten(observation, 0), SHANTEN_CLASSES - 1);
    int tenpai = 0;
    boolean riichi = false;
    for (int seat = 1; seat < SEATS; seat++) {
      if (shanten(observation, seat) == 0) {
        tenpai++;
      }
      riichi |= observation.riichi(seat) != RiichiState.NONE;
    }
    int key = observation.event().ordinal();
    key = key * SHANTEN_CLASSES + shanten;
    key = key * TENPAI_CLASSES + Math.min(tenpai, TENPAI_CLASSES - 1);
    key = key * 2 + (riichi ? 1 : 0);
    return key * STAGES + Stage.of(observation.remainingWall()).ordinal();
  }

  /** 相対席 {@code seat} の最小の向聴（和了の形でも 0 にそろえる）。 */
  private int shanten(LuckObservation observation, int seat) {
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      counts[tile] = observation.handCount(seat, tile);
    }
    return Math.max(0, shapes.calculateMinimum(counts, observation.meldCount(seat)));
  }
}
