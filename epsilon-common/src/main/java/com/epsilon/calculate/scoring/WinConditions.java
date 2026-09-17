package com.epsilon.calculate.scoring;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import java.util.Objects;

/**
 * 和了評価に必要な局況だけを切り出した値。
 *
 * <p>同じ値の組み合わせはキャッシュから返し、対局中に文脈オブジェクトを生成しない。フリテン・点棒・本場・供託は役と符の成立条件ではないため含めない。
 */
public record WinConditions(
    int jikaze,
    int bakaze,
    RiichiState riichiStatus,
    boolean ippatsu,
    WinSource source,
    boolean uninterruptedFirstDraw) {

  private static final int WIND_COUNT = 4;
  private static final WinSource[] SOURCES = WinSource.values();
  private static final WinConditions[] CONTEXTS = createContexts();

  public WinConditions {
    if (jikaze < Tile.TON || jikaze > Tile.PEI || bakaze < Tile.TON || bakaze > Tile.PEI)
      throw new IllegalArgumentException("invalid seat or round wind");
    Objects.requireNonNull(riichiStatus, "riichiStatus");
    Objects.requireNonNull(source, "source");
  }

  /** 公開情報だけで評価する AI 入力用文脈を返す。 */
  public static WinConditions publicDecision(int jikaze, int bakaze, RiichiState riichiStatus) {
    return cached(jikaze, bakaze, riichiStatus, false, WinSource.NORMAL, false);
  }

  /** 現在局面と和了イベントから完全な評価文脈を返す。 */
  public static WinConditions fromGameState(GameState state, int player, TurnEvent event) {
    return cached(
        state.getJikaze(player),
        state.getBakaze(),
        riichiStatus(state, player),
        state.isIppatsu(player),
        WinSource.from(event, state.isWallExhausted()),
        state.isFirstDraw(player) && !state.isFirstTurnCallOccurred());
  }

  /** 和了者が親かを返す。 */
  public boolean oya() {
    return jikaze == Tile.TON;
  }

  private static WinConditions cached(
      int jikaze,
      int bakaze,
      RiichiState riichiStatus,
      boolean ippatsu,
      WinSource source,
      boolean uninterruptedFirstDraw) {
    if (jikaze < Tile.TON || jikaze > Tile.PEI || bakaze < Tile.TON || bakaze > Tile.PEI)
      throw new IllegalArgumentException("invalid seat or round wind");
    return CONTEXTS[index(jikaze, bakaze, riichiStatus, ippatsu, source, uninterruptedFirstDraw)];
  }

  private static int index(
      int jikaze,
      int bakaze,
      RiichiState riichi,
      boolean ippatsu,
      WinSource source,
      boolean firstDraw) {
    int index = (jikaze - Tile.TON) * WIND_COUNT + bakaze - Tile.TON;
    index = index * 3 + riichi.ordinal();
    index = index * 2 + (ippatsu ? 1 : 0);
    index = index * SOURCES.length + source.ordinal();
    return index * 2 + (firstDraw ? 1 : 0);
  }

  private static WinConditions[] createContexts() {
    WinConditions[] contexts =
        new WinConditions[WIND_COUNT * WIND_COUNT * 3 * 2 * SOURCES.length * 2];
    for (int seat = Tile.TON; seat <= Tile.PEI; seat++)
      for (int round = Tile.TON; round <= Tile.PEI; round++)
        for (RiichiState riichi : RiichiState.values())
          for (int ippatsu = 0; ippatsu < 2; ippatsu++)
            for (WinSource source : SOURCES)
              for (int firstDraw = 0; firstDraw < 2; firstDraw++) {
                int index = index(seat, round, riichi, ippatsu != 0, source, firstDraw != 0);
                contexts[index] =
                    new WinConditions(seat, round, riichi, ippatsu != 0, source, firstDraw != 0);
              }
    return contexts;
  }

  private static RiichiState riichiStatus(GameState state, int player) {
    if (state.isDoubleRiichi(player)) {
      return RiichiState.DOUBLE_RIICHI;
    }
    return state.isRiichi(player) ? RiichiState.RIICHI : RiichiState.NONE;
  }
}
