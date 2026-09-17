package com.epsilon.major.ai.decision.input;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;

/**
 * 行動 ID や牌の関係などの意味を持つ値を、保存用のカテゴリ値へ変換する。
 *
 * <p>物理連続バッファのオフセットやテンソル形状は扱わない。行動 ID、複合牌種間の関係、役ビットまとまりなど、単一フィールドへ格納する値の 符号化/復号だけを所有する。
 */
public final class DecisionFeatureCodec {

  private static final Action.Group[] ACTION_GROUPS = Action.Group.values();
  private static final int TILE_COUNT_CARDINALITY = Tile.TILES_PER_TYPE + 1;
  private static final int SHAPE_WAIT_MASK = 1 << 5;
  private static final int RON_YAKU_WAIT_MASK = 1 << 4;
  private static final int TSUMO_YAKU_WAIT_MASK = 1 << 3;
  private static final int DISCARDED_TILE_MASK = 1 << 2;
  private static final int CALLED_INTO_MELD_MASK = 1 << 1;
  private static final int AKA_WINNING_TILE_AVAILABLE_MASK = 1;

  private DecisionFeatureCodec() {}

  /**
   * 行動種別を、0をパディングに予約した保存カテゴリへ変換する。
   *
   * @param type エンジン行動種別
   * @return {@code ordinal + 1}
   */
  public static int actionType(Action.Type type) {
    return type.ordinal() + 1;
  }

  /**
   * 行動グループを、0をパディングに予約した保存カテゴリへ変換する。
   *
   * @param group エンジン行動グループ
   * @return {@code ordinal + 1}
   */
  public static int actionGroup(Action.Group group) {
    return group.ordinal() + 1;
  }

  /**
   * 保存済み行動グループカテゴリが応答判断に属するかを返す。
   *
   * @param storedGroup 0をパディングに予約したグループカテゴリ
   * @return 他家行動への応答グループなら {@code true}
   */
  public static boolean isResponseActionGroup(int storedGroup) {
    return ACTION_GROUPS[storedGroup - 1].isResponse();
  }

  /**
   * エンジン行動 IDを、0をパディングに予約した保存カテゴリへ変換する。
   *
   * @param action 符号化する標準形式の行動
   * @return {@code action.toIndex() + 1}
   */
  public static int actionId(Action action) {
    return action.toIndex() + 1;
  }

  /**
   * 保存カテゴリからエンジン行動 IDを復元する。
   *
   * @param storedActionId 0をパディングに予約した行動カテゴリ
   * @return {@code 0 <= id < Action.ACTION_SPACE_SIZE} の固定行動 ID
   */
  public static int decodeActionId(int storedActionId) {
    if (storedActionId <= DecisionInputSchema.PAD_ID || storedActionId > Action.ACTION_SPACE_SIZE) {
      throw new IllegalArgumentException("stored action id out of range: " + storedActionId);
    }
    return storedActionId - 1;
  }

  /**
   * 一牌種ぶんの遷移関係を単一カテゴリへ可逆圧縮する。
   *
   * <p>0はパディング専用。実値は結果副露に含まれない手牌枚数、改善牌残り枚数、形待ち、RON役あり待ち、TSUMO役あり待ち、打牌、 新設面子への使用、赤5和了牌の残存可能性の直積を使う。
   *
   * @param concealedCount 遷移後の副露に含まれない手牌にある同牌種枚数
   * @param ukeireRemainingCount 公開情報から算出した同牌種の残り受入れ枚数
   * @param shapeWait 同牌種で純粋な和了形になるなら {@code true}
   * @param ronYakuWait 同牌種で役ありRONが可能なら {@code true}
   * @param tsumoYakuWait 同牌種で役ありTSUMOが可能なら {@code true}
   * @param discardedTile 遷移でこの牌種を捨てるなら {@code true}
   * @param calledIntoMeld 行動候補で新しい面子へこの牌種を使うなら {@code true}
   * @param akaWinningTileAvailable 同牌種の赤5を和了牌として利用できるなら {@code true}
   * @return 0と衝突しない可逆カテゴリ
   */
  public static int transitionTile(
      int concealedCount,
      int ukeireRemainingCount,
      boolean shapeWait,
      boolean ronYakuWait,
      boolean tsumoYakuWait,
      boolean discardedTile,
      boolean calledIntoMeld,
      boolean akaWinningTileAvailable) {
    if (concealedCount < 0 || concealedCount > Tile.TILES_PER_TYPE) {
      throw new IllegalArgumentException("concealedCount out of range: " + concealedCount);
    }
    if (ukeireRemainingCount < 0 || ukeireRemainingCount > Tile.TILES_PER_TYPE) {
      throw new IllegalArgumentException(
          "ukeireRemainingCount out of range: " + ukeireRemainingCount);
    }
    int code = concealedCount;
    code = code * TILE_COUNT_CARDINALITY + ukeireRemainingCount;
    code = code * 2 + (shapeWait ? 1 : 0);
    code = code * 2 + (ronYakuWait ? 1 : 0);
    code = code * 2 + (tsumoYakuWait ? 1 : 0);
    code = code * 2 + (discardedTile ? 1 : 0);
    code = code * 2 + (calledIntoMeld ? 1 : 0);
    code = code * 2 + (akaWinningTileAvailable ? 1 : 0);
    return code + 1;
  }

  /**
   * 遷移牌カテゴリから結果副露に含まれない手牌枚数を復元する。
   *
   * @param storedCode {@link #transitionTile} が返したカテゴリ
   * @return 同牌種の副露に含まれない手牌枚数
   */
  public static int transitionConcealedCount(int storedCode) {
    return requireTransitionTile(storedCode)
        / DecisionInputSchema.ACTION_TRANSITION_TILE_FLAG_COMBINATION_COUNT
        / TILE_COUNT_CARDINALITY;
  }

  /**
   * 遷移牌カテゴリから残り受入れ枚数を復元する。
   *
   * @param storedCode {@link #transitionTile} が返したカテゴリ
   * @return 同牌種の残り受入れ枚数
   */
  public static int transitionUkeireRemainingCount(int storedCode) {
    return requireTransitionTile(storedCode)
        / DecisionInputSchema.ACTION_TRANSITION_TILE_FLAG_COMBINATION_COUNT
        % TILE_COUNT_CARDINALITY;
  }

  /**
   * 遷移牌カテゴリが形待ちを表すかを返す。
   *
   * @param storedCode {@link #transitionTile} が返したカテゴリ
   * @return 形待ちなら {@code true}
   */
  public static boolean transitionIsShapeWait(int storedCode) {
    return (requireTransitionTile(storedCode) & SHAPE_WAIT_MASK) != 0;
  }

  /**
   * 遷移牌カテゴリが役ありRON待ちを表すかを返す。
   *
   * @param storedCode {@link #transitionTile} が返したカテゴリ
   * @return 役ありRON待ちなら {@code true}
   */
  public static boolean transitionIsRonYakuWait(int storedCode) {
    return (requireTransitionTile(storedCode) & RON_YAKU_WAIT_MASK) != 0;
  }

  /**
   * 遷移牌カテゴリが役ありTSUMO待ちを表すかを返す。
   *
   * @param storedCode {@link #transitionTile} が返したカテゴリ
   * @return 役ありTSUMO待ちなら {@code true}
   */
  public static boolean transitionIsTsumoYakuWait(int storedCode) {
    return (requireTransitionTile(storedCode) & TSUMO_YAKU_WAIT_MASK) != 0;
  }

  /**
   * 遷移牌カテゴリが捨て牌を表すかを返す。
   *
   * @param storedCode {@link #transitionTile} が返したカテゴリ
   * @return 遷移で捨てる牌種なら {@code true}
   */
  public static boolean transitionIsDiscarded(int storedCode) {
    return (requireTransitionTile(storedCode) & DISCARDED_TILE_MASK) != 0;
  }

  /**
   * 遷移牌カテゴリが新設面子への使用を表すかを返す。
   *
   * @param storedCode {@link #transitionTile} が返したカテゴリ
   * @return 行動候補で面子へ使う牌種なら {@code true}
   */
  public static boolean transitionIsCalledIntoMeld(int storedCode) {
    return (requireTransitionTile(storedCode) & CALLED_INTO_MELD_MASK) != 0;
  }

  /**
   * 遷移牌カテゴリが赤5和了牌の残存可能性を表すかを返す。
   *
   * @param storedCode {@link #transitionTile} が返したカテゴリ
   * @return 赤5を和了牌として利用できるなら {@code true}
   */
  public static boolean transitionIsAkaWinningTileAvailable(int storedCode) {
    return (requireTransitionTile(storedCode) & AKA_WINNING_TILE_AVAILABLE_MASK) != 0;
  }

  /**
   * 役ビット集合の指定まとまりを、パディング 0と衝突しないカテゴリへ変換する。
   *
   * @param yakuBits 成立可能役を表すビット集合
   * @param chunk 取り出すまとまりインデックス
   * @return まとまり値へ1を加えたカテゴリ
   */
  public static int waitYakuChunk(long yakuBits, int chunk) {
    long mask = (1L << DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK) - 1L;
    return (int) ((yakuBits >>> (chunk * DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK)) & mask) + 1;
  }

  /**
   * 保存済みカテゴリから指定まとまりの役ビットを元の位置へ復元する。
   *
   * @param storedCode 0をパディングに予約したまとまりカテゴリ
   * @param chunk 復元先まとまりインデックス
   * @return 元のビット位置へ定数加算した役ビット集合
   */
  public static long decodeWaitYakuChunk(int storedCode, int chunk) {
    if (storedCode <= DecisionInputSchema.PAD_ID
        || storedCode >= DecisionInputSchema.ACTION_TRANSITION_WAIT_YAKU_DICTIONARY_SIZE) {
      throw new IllegalArgumentException("wait yaku code out of range: " + storedCode);
    }
    requireWaitYakuChunk(chunk);
    return (long) (storedCode - 1) << (chunk * DecisionInputSchema.WAIT_YAKU_BITS_PER_CHUNK);
  }

  /**
   * RON/TSUMOとまとまり番号を待ち役テンソル末尾次元へ写す。
   *
   * @param winType RONまたはTSUMO
   * @param chunk 役ビットまとまりインデックス
   * @return 待ち役テンソルの特徴量インデックス
   */
  public static int waitYakuFeature(DecisionInputSchema.WaitWinType winType, int chunk) {
    return winType.ordinal() * DecisionInputSchema.WAIT_YAKU_CHUNKS_PER_WIN_TYPE + chunk;
  }

  private static int requireTransitionTile(int storedCode) {
    if (storedCode <= DecisionInputSchema.PAD_ID
        || storedCode >= DecisionInputSchema.ACTION_TRANSITION_TILE_DICTIONARY_SIZE) {
      throw new IllegalArgumentException("transition tile code out of range: " + storedCode);
    }
    return storedCode - 1;
  }

  private static void requireWaitYakuChunk(int chunk) {
    if (chunk < 0 || chunk >= DecisionInputSchema.WAIT_YAKU_CHUNKS_PER_WIN_TYPE) {
      throw new IllegalArgumentException("wait yaku chunk out of range: " + chunk);
    }
  }
}
