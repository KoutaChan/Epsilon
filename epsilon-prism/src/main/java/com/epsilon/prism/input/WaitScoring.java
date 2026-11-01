package com.epsilon.prism.input;

import com.epsilon.calculate.scoring.HandScoreBatchBuffer;
import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.core.AkaTileMask;
import com.epsilon.prism.Limits;

/**
 * 聴牌の後の手の待ちごとの採点。待ちごとに、ロン・ツモ ×（立直なし・その遷移の立直の状態）×（赤でない和了牌・赤の 5 の和了牌）の役の有無・
 * 翻・符・基本点を、1 回の分解で求める（{@link HandScoreEvaluator} の表ドラを基本型で受ける口。裏ドラは数えない）。
 *
 * <p>条件はエンジンの判断の解析と同じ公開の条件（自風・場風・立直だけ。一発・海底などは数えない）。赤でない和了牌の値は、エンジンの
 * {@code analyzeTransitionWaits} の待ちの点と同じになる。赤の 5 の和了牌は、その色の赤がまだ見えていない 5 の待ちだけ採点する。割り当てを
 * しない。
 */
final class WaitScoring {

  /** 立直の構え。 */
  enum Stance {
    DAMA,
    /** その遷移の立直の状態（立直・ダブル立直。立直していない遷移では採点しない）。 */
    RIICHI
  }

  enum WinningTile {
    /** 赤でない牌。 */
    PLAIN,
    /** 赤の 5。 */
    AKA
  }

  private static final WinMethod[] METHODS = WinMethod.values();
  private static final int STANCES = Stance.values().length;
  private static final int TILES = WinningTile.values().length;
  private static final int RESULTS = STANCES * METHODS.length * TILES;

  private final HandScoreEvaluator evaluator = new HandScoreEvaluator();
  private final HandScoreBatchBuffer batch = new HandScoreBatchBuffer(RESULTS);
  private final int[] waitTile = new int[Limits.MAX_WAIT_KINDS];
  private final int[] basePoints = new int[Limits.MAX_WAIT_KINDS * RESULTS];
  private final int[] han = new int[Limits.MAX_WAIT_KINDS * RESULTS];
  private final int[] fu = new int[Limits.MAX_WAIT_KINDS * RESULTS];
  private int waits;
  private boolean riichi;

  /**
   * 待ちを採点する。
   *
   * @param hand 後の手（13 枚相当で聴牌）
   * @param waitMask 形の待ち
   * @param tiles 牌ごとの枚数（表ドラと見えた赤を読む）
   * @param seatWind 自風の牌
   * @param roundWind 場風の牌
   * @param transitionRiichi その遷移の立直の状態
   */
  void score(
      AfterHand hand,
      long waitMask,
      TileCounts tiles,
      int seatWind,
      int roundWind,
      RiichiState transitionRiichi) {
    riichi = transitionRiichi != RiichiState.NONE;
    int stances = riichi ? STANCES : 1;
    int plainSlots = stances * METHODS.length;
    WinConditions dama = WinConditions.publicDecision(seatWind, roundWind, RiichiState.NONE);
    WinConditions declared =
        riichi ? WinConditions.publicDecision(seatWind, roundWind, transitionRiichi) : dama;
    int owned = hand.ownedAkaMask();
    waits = 0;
    for (long rest = waitMask; rest != 0L; rest &= rest - 1) {
      int tile = Long.numberOfTrailingZeros(rest);
      boolean aka = tiles.akaUnseen(tile);
      for (int stance = 0; stance < stances; stance++) {
        WinConditions conditions = stance == Stance.DAMA.ordinal() ? dama : declared;
        for (WinMethod method : METHODS) {
          int slot = stance * METHODS.length + method.ordinal();
          batch.configure(slot, method, conditions, owned);
          if (aka) {
            batch.configure(plainSlots + slot, method, conditions, AkaTileMask.includeTile(owned, tile));
          }
        }
      }
      batch.size(aka ? 2 * plainSlots : plainSlots);
      evaluator.scoreAfterAddingBatch(
          hand.shape(), hand, tile, tiles.doraPacked(), tiles.indicators(), batch);
      int wait = waits++;
      waitTile[wait] = tile;
      for (int result = 0; result < RESULTS; result++) {
        basePoints[wait * RESULTS + result] = 0;
      }
      for (int slot = 0, size = batch.size(); slot < size; slot++) {
        HandScoreBuffer score = batch.result(slot);
        if (!score.available()) {
          continue;
        }
        int variant = slot / plainSlots;
        int stance = slot % plainSlots / METHODS.length;
        int method = slot % METHODS.length;
        int at = wait * RESULTS + index(stance, method, variant);
        basePoints[at] = score.basePoints();
        han[at] = score.han();
        fu[at] = score.fu();
      }
    }
  }

  /** 採点した待ちの数。 */
  int waits() {
    return waits;
  }

  /** wait 番目の待ちの牌（牌の番号の小さい順）。 */
  int waitTile(int wait) {
    return waitTile[wait];
  }

  /** 立直の構え（{@link Stance#RIICHI}）を採点した。 */
  boolean riichi() {
    return riichi;
  }

  /** 和了できる（役がある）。 */
  boolean available(int wait, Stance stance, WinMethod method, WinningTile tile) {
    return basePoints(wait, stance, method, tile) > 0;
  }

  /** 基本点（役がなければ 0）。 */
  int basePoints(int wait, Stance stance, WinMethod method, WinningTile tile) {
    return basePoints[at(wait, stance, method, tile)];
  }

  /** 翻（裏なし。役満は 0）。 */
  int han(int wait, Stance stance, WinMethod method, WinningTile tile) {
    return han[at(wait, stance, method, tile)];
  }

  /** 符（役満は 0）。 */
  int fu(int wait, Stance stance, WinMethod method, WinningTile tile) {
    return fu[at(wait, stance, method, tile)];
  }

  private int at(int wait, Stance stance, WinMethod method, WinningTile tile) {
    if (stance == Stance.RIICHI && !riichi) {
      throw new IllegalStateException("riichi stance was not scored");
    }
    return wait * RESULTS + index(stance.ordinal(), method.ordinal(), tile.ordinal());
  }

  private static int index(int stance, int method, int tile) {
    return (stance * METHODS.length + method) * TILES + tile;
  }
}
