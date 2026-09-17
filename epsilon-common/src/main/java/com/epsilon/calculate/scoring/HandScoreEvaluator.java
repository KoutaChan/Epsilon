package com.epsilon.calculate.scoring;

import com.epsilon.calculate.shape.HandShapeView;
import com.epsilon.calculate.shape.WinningDecompositionCursor;
import com.epsilon.core.DoraState;
import com.epsilon.core.HandView;
import com.epsilon.core.Tile;
import java.util.Objects;

/**
 * 一つのワーカーが所有する役・符・得点評価器。複数スレッドから同時に利用できない。
 *
 * <p>完成形、固定面子、和了条件を分離し、一回の分解を全条件へ共有する。 出力は呼び出し元のバッファへ書き込み、候補ごとのオブジェクト・配列・複製は生成しない。
 * フリテンと支払い精算はこの評価器の責務に含めない。
 */
public final class HandScoreEvaluator {
  public static final int RON = 1;
  public static final int TSUMO = 2;
  public static final int BOTH = RON | TSUMO;

  private final WinningHandFacts hand = new WinningHandFacts();
  private final DecompositionFacts decomposition = new DecompositionFacts();
  private final WinningDecompositionCursor cursor = new WinningDecompositionCursor();
  private long[] handYaku = new long[8];
  private long[] handYakuman = new long[8];
  private int[] redCounts = new int[8];
  private int[] uraCounts = new int[8];
  private int remainingPlacements;
  private WaitShape currentWait;
  private long currentPatternYaku;

  public HandScoreEvaluator() {}

  /** 完成手を採点する。不成立の場合も出力を消去する。 */
  public boolean scoreCompleted(
      HandView source,
      int tile,
      WinMethod method,
      WinConditions conditions,
      DoraState dora,
      boolean includeUra,
      int ownedAkaMask,
      HandScoreBuffer output) {
    return scoreSingle(
        source, tile, false, method, conditions, dora, includeUra, ownedAkaMask, output);
  }

  /** 13枚相当の手に和了牌を仮想追加して採点する。入力は変更しない。 */
  public boolean scoreAfterAdding(
      HandView source,
      int tile,
      WinMethod method,
      WinConditions conditions,
      DoraState dora,
      boolean includeUra,
      int ownedAkaMask,
      HandScoreBuffer output) {
    return scoreSingle(
        source, tile, true, method, conditions, dora, includeUra, ownedAkaMask, output);
  }

  /** 完成形を一度分解し、各条件について独立した最良結果を求める。 */
  public void scoreCompletedBatch(
      HandShapeView shape,
      HandView fixedHand,
      int tile,
      DoraState dora,
      boolean includeUra,
      HandScoreBatchBuffer output) {
    score(shape, fixedHand, tile, false, dora, includeUra, output);
  }

  /**
   * 牌を仮に1枚追加した手牌を一度だけ面子に分解し、ロン・ツモ・立直・赤牌の条件ごとに得点を計算する。
   *
   * <p>副露と暗槓を除く手牌の形は{@code shape}から、確定済みの面子は{@code fixedHand}から読む。各条件の自風と場風は同じでなければならない。各{@code
   * ownedAkaMask}は確定済み面子も含む赤牌の保有状況を表し、対応する五の牌を実際に含む必要がある。
   */
  public void scoreAfterAddingBatch(
      HandShapeView shape,
      HandView fixedHand,
      int tile,
      DoraState dora,
      boolean includeUra,
      HandScoreBatchBuffer output) {
    score(shape, fixedHand, tile, true, dora, includeUra, output);
  }

  /** 指定された和了方法に役があるかだけを調べ、成立した方法のマスクを返す。 */
  public int probeCompleted(
      HandView source, int tile, int requestedMethods, WinConditions conditions) {
    return probe(source, source, tile, false, requestedMethods, conditions);
  }

  public int probeAfterAdding(
      HandView source, int tile, int requestedMethods, WinConditions conditions) {
    return probe(source, source, tile, true, requestedMethods, conditions);
  }

  /** 形状だけを更新する探索カーソルにも使える、符・ドラを計算しない役存在判定。 */
  public int probeAfterAdding(
      HandShapeView shape,
      HandView fixedHand,
      int tile,
      int requestedMethods,
      WinConditions conditions) {
    return probe(shape, fixedHand, tile, true, requestedMethods, conditions);
  }

  /** 単独評価はプリミティブ型の局所値で比較する。バッチ設定・配列アクセス・一時出力の転記を挟まない。 分解列挙、役、符、最良候補の比較はバッチと共用する。 */
  private boolean scoreSingle(
      HandView source,
      int tile,
      boolean adding,
      WinMethod method,
      WinConditions conditions,
      DoraState dora,
      boolean includeUra,
      int ownedAkaMask,
      HandScoreBuffer output) {
    Objects.requireNonNull(output, "output").clear();
    Objects.requireNonNull(method, "method");
    load(source, source, tile, adding, conditions, dora, includeUra);
    validateAkaMask(ownedAkaMask, source, adding ? tile : -1);
    long yakuman =
        YakuRules.evaluateHandYakumanMask(hand)
            | YakuRules.evaluateContextYakumanMask(method, conditions);
    long normalYaku =
        yakuman == 0L
            ? YakuRules.evaluateHandYakuMask(hand)
                | YakuRules.evaluateContextYakuMask(hand.menzen, method, conditions)
            : 0L;
    int red = Integer.bitCount(ownedAkaMask);
    int ura = uraCount(conditions, includeUra);
    if (hand.thirteenOrphansComplete || hand.sevenPairsComplete) {
      consider(
          output, specialYaku(normalYaku, yakuman), hand.sevenPairsComplete ? 25 : 0, red, ura);
    }
    if (hand.thirteenOrphansComplete || !hand.standardPatternMayExist) return output.available();
    bind(source, tile, adding);
    while (nextPlacement(tile)) {
      long bits = candidateYaku(method, normalYaku, yakuman);
      if (bits != 0L) consider(output, bits, candidateFu(method, tile, bits), red, ura);
    }
    return output.available();
  }

  private void score(
      HandShapeView shape,
      HandView fixedHand,
      int tile,
      boolean adding,
      DoraState dora,
      boolean includeUra,
      HandScoreBatchBuffer output) {
    Objects.requireNonNull(output, "output");
    for (HandScoreBuffer result : output.results) result.clear();
    if (output.size == 0) throw new IllegalArgumentException("empty scoring batch");
    WinConditions common = Objects.requireNonNull(output.conditions[0], "unconfigured condition");
    load(shape, fixedHand, tile, adding, common, dora, includeUra);
    if (hand.thirteenOrphansComplete || hand.sevenPairsComplete && !hand.standardPatternMayExist) {
      for (int i = 0; i < output.size; i++) {
        validateCondition(output, i, shape, adding ? tile : -1, common);
      }
      scoreSpecialBatch(output, includeUra);
      return;
    }
    ensureCapacity(output.size);
    long[] contextYaku = hand.menzen ? output.closedContextYakuMasks : output.openContextYakuMasks;
    long structuralYakuman = YakuRules.evaluateHandYakumanMask(hand);
    long commonYaku = 0L;
    boolean commonYakuPrepared = false;
    for (int i = 0; i < output.size; i++) {
      WinConditions conditions = validateCondition(output, i, shape, adding ? tile : -1, common);
      handYakuman[i] = structuralYakuman | output.contextYakumanMasks[i];
      if (handYakuman[i] == 0L) {
        if (!commonYakuPrepared) {
          commonYaku = YakuRules.evaluateHandYakuMask(hand);
          commonYakuPrepared = true;
        }
        handYaku[i] = commonYaku | contextYaku[i];
      } else {
        handYaku[i] = 0L;
      }
      redCounts[i] = Integer.bitCount(output.ownedAkaMasks[i]);
      uraCounts[i] = uraCount(conditions, includeUra);
    }

    // 特殊形も成立の確認が先。牌種・局況だけで未完成形を役満にしない。
    if (hand.thirteenOrphansComplete || hand.sevenPairsComplete) {
      for (int i = 0; i < output.size; i++) {
        consider(
            output.results[i],
            specialYaku(handYaku[i], handYakuman[i]),
            hand.sevenPairsComplete ? 25 : 0,
            redCounts[i],
            uraCounts[i]);
      }
    }

    if (hand.thirteenOrphansComplete || !hand.standardPatternMayExist) return;
    bind(shape, tile, adding);
    while (nextPlacement(tile)) {
      for (int i = 0; i < output.size; i++) {
        WinMethod method = output.methods[i];
        long bits = candidateYaku(method, handYaku[i], handYakuman[i]);
        if (bits != 0L)
          consider(
              output.results[i], bits, candidateFu(method, tile, bits), redCounts[i], uraCounts[i]);
      }
    }
  }

  private WinConditions validateCondition(
      HandScoreBatchBuffer output,
      int index,
      HandShapeView shape,
      int addedTile,
      WinConditions common) {
    WinConditions conditions =
        Objects.requireNonNull(output.conditions[index], "unconfigured condition");
    Objects.requireNonNull(output.methods[index], "unconfigured method");
    if (conditions.jikaze() != common.jikaze() || conditions.bakaze() != common.bakaze())
      throw new IllegalArgumentException("batch winds must match");
    validateAkaMask(output.ownedAkaMasks[index], shape, addedTile);
    return conditions;
  }

  /** 通常形と競合しない特殊形は、全条件の検証後に一時配列を介さず採点する。 */
  private void scoreSpecialBatch(HandScoreBatchBuffer output, boolean includeUra) {
    long[] contextYaku = hand.menzen ? output.closedContextYakuMasks : output.openContextYakuMasks;
    long structuralYakuman = YakuRules.evaluateHandYakumanMask(hand);
    long commonYaku = 0L;
    boolean commonYakuPrepared = false;
    int fu = hand.sevenPairsComplete ? 25 : 0;
    for (int i = 0; i < output.size; i++) {
      long yakuman = structuralYakuman | output.contextYakumanMasks[i];
      long normalYaku = 0L;
      if (yakuman == 0L) {
        if (!commonYakuPrepared) {
          commonYaku = YakuRules.evaluateHandYakuMask(hand);
          commonYakuPrepared = true;
        }
        normalYaku = commonYaku | contextYaku[i];
      }
      consider(
          output.results[i],
          specialYaku(normalYaku, yakuman),
          fu,
          Integer.bitCount(output.ownedAkaMasks[i]),
          uraCount(output.conditions[i], includeUra));
    }
  }

  private long specialYaku(long normalYaku, long yakuman) {
    return yakuman != 0L
        ? yakuman
        : YakuBits.effective(normalYaku | ScoringYaku.CHIITOITSU.bit(), hand.menzen);
  }

  /** 一つの分解を読み、和了牌を割り当てられる各待ち形を順に返す。 */
  private boolean nextPlacement(int tile) {
    while (remainingPlacements == 0) {
      if (!cursor.next()) return false;
      remainingPlacements = decomposition.load(hand, cursor, tile);
      if (remainingPlacements != 0)
        currentPatternYaku = YakuRules.evaluatePatternYakuMask(hand, decomposition);
    }
    currentWait = WaitShape.fromOrdinal(Integer.numberOfTrailingZeros(remainingPlacements));
    remainingPlacements &= remainingPlacements - 1;
    return true;
  }

  private long candidateYaku(WinMethod method, long normalYaku, long yakuman) {
    return YakuRules.evaluateCandidateYakuMask(
        hand, decomposition, method, currentWait, normalYaku, yakuman, currentPatternYaku);
  }

  private int candidateFu(WinMethod method, int tile, long bits) {
    return YakuBits.hasYakuman(bits)
        ? 0
        : FuCalculator.calculate(hand, decomposition, method, currentWait, tile, bits);
  }

  private int uraCount(WinConditions conditions, boolean includeUra) {
    return includeUra && hand.menzen && conditions.riichiStatus() != RiichiState.NONE
        ? hand.uraDoraCount
        : 0;
  }

  private int probe(
      HandShapeView shape,
      HandView fixedHand,
      int tile,
      boolean adding,
      int requestedMethods,
      WinConditions conditions) {
    if (requestedMethods == 0 || (requestedMethods & ~BOTH) != 0)
      throw new IllegalArgumentException("invalid requested method mask");
    load(shape, fixedHand, tile, adding, conditions, null, false);
    if (hand.thirteenOrphansComplete || hand.sevenPairsComplete) return requestedMethods;
    if (!hand.standardPatternMayExist) return 0;
    long structuralYakuman = YakuRules.evaluateHandYakumanMask(hand);
    long ronYaku = 0L, ronYakuman = 0L, tsumoYaku = 0L, tsumoYakuman = 0L;
    if ((requestedMethods & RON) != 0) {
      ronYakuman =
          structuralYakuman | YakuRules.evaluateContextYakumanMask(WinMethod.RON, conditions);
    }
    if ((requestedMethods & TSUMO) != 0) {
      tsumoYakuman =
          structuralYakuman | YakuRules.evaluateContextYakumanMask(WinMethod.TSUMO, conditions);
    }
    if (((requestedMethods & RON) != 0 && ronYakuman == 0L)
        || ((requestedMethods & TSUMO) != 0 && tsumoYakuman == 0L)) {
      long commonYaku = YakuRules.evaluateHandYakuMask(hand);
      if ((requestedMethods & RON) != 0 && ronYakuman == 0L) {
        ronYaku =
            commonYaku | YakuRules.evaluateContextYakuMask(hand.menzen, WinMethod.RON, conditions);
      }
      if ((requestedMethods & TSUMO) != 0 && tsumoYakuman == 0L) {
        tsumoYaku =
            commonYaku
                | YakuRules.evaluateContextYakuMask(hand.menzen, WinMethod.TSUMO, conditions);
      }
    }
    int found = 0;
    bind(shape, tile, adding);
    while (nextPlacement(tile)) {
      int remaining = requestedMethods & ~found;
      if ((remaining & RON) != 0 && candidateYaku(WinMethod.RON, ronYaku, ronYakuman) != 0L)
        found |= RON;
      if ((remaining & TSUMO) != 0 && candidateYaku(WinMethod.TSUMO, tsumoYaku, tsumoYakuman) != 0L)
        found |= TSUMO;
      if (found == requestedMethods) return found;
    }
    return found;
  }

  private void consider(HandScoreBuffer best, long bits, int fu, int redCount, int uraCount) {
    if (bits == 0L) return;
    int multiplier = YakuBits.yakumanCount(bits);
    int omote = multiplier == 0 ? hand.omoteDoraCount : 0;
    int aka = multiplier == 0 ? redCount : 0;
    int ura = multiplier == 0 ? uraCount : 0;
    int han = multiplier == 0 ? YakuBits.han(bits, hand.menzen) + omote + aka + ura : 0;
    int normalizedFu = multiplier == 0 ? fu : 0;
    int points = multiplier > 0 ? 8000 * multiplier : ScoreMath.normalBasePoints(han, normalizedFu);
    if (best.available()) {
      if ((multiplier > 0) != best.isYakuman()) {
        if (multiplier == 0) return;
      } else if (points < best.basePoints()
          || points == best.basePoints()
              && (han < best.han() || han == best.han() && normalizedFu <= best.fu())) return;
    }
    best.accept(hand.menzen, bits, han, normalizedFu, omote, aka, ura, multiplier, points);
  }

  private void load(
      HandShapeView shape,
      HandView fixedHand,
      int tile,
      boolean adding,
      WinConditions conditions,
      DoraState dora,
      boolean includeUra) {
    Objects.requireNonNull(shape, "shape");
    Objects.requireNonNull(fixedHand, "fixedHand");
    Objects.requireNonNull(conditions, "conditions");
    if (tile < 0 || tile >= Tile.NUM_TILE_TYPES)
      throw new IllegalArgumentException("invalid winning tile");
    if (shape.meldCount() != fixedHand.meldCount()
        || shape.meldCount() < 0
        || shape.meldCount() > 4)
      throw new IllegalArgumentException("fixed meld count does not match shape");
    int effective = shape.concealedTileCount() + 3 * shape.meldCount();
    if (effective != (adding ? 13 : 14))
      throw new IllegalArgumentException("expected " + (adding ? 13 : 14) + " effective tiles");
    if (!adding && shape.count(tile) == 0)
      throw new IllegalArgumentException("winning tile absent");
    if (adding && shape.count(tile) >= 4)
      throw new IllegalArgumentException("cannot add a fifth tile");
    hand.load(
        shape,
        fixedHand,
        adding ? tile : -1,
        conditions,
        dora == null ? 0L : dora.doraTileTypesPacked(),
        dora == null ? 0 : dora.indicatorCount(),
        dora == null || !includeUra ? 0L : dora.uraDoraTileTypesPacked(),
        dora == null || !includeUra ? 0 : dora.uraKnownMask());
  }

  private void validateAkaMask(int mask, HandShapeView shape, int addedTile) {
    if ((mask & ~7) != 0 || (mask & hand.fixedAkaMask) != hand.fixedAkaMask)
      throw new IllegalArgumentException("invalid red tile mask");
    for (int concealedReds = mask & ~hand.fixedAkaMask;
        concealedReds != 0;
        concealedReds &= concealedReds - 1) {
      // 確定面子の通常五を仮想的な赤五へ置き換えてはならない。
      int five = Integer.numberOfTrailingZeros(concealedReds) * 9 + 4;
      if (shape.count(five) + (addedTile == five ? 1 : 0) == 0)
        throw new IllegalArgumentException("red tile absent from concealed shape");
    }
  }

  private void bind(HandShapeView shape, int tile, boolean adding) {
    remainingPlacements = 0;
    if (adding) cursor.bindAfterAdding(shape, tile);
    else cursor.bindCompleted(shape);
  }

  private void ensureCapacity(int size) {
    if (size <= handYaku.length) return;
    int capacity = Math.max(size, handYaku.length * 2);
    handYaku = new long[capacity];
    handYakuman = new long[capacity];
    redCounts = new int[capacity];
    uraCounts = new int[capacity];
  }
}
