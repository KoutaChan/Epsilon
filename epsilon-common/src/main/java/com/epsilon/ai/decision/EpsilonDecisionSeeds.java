package com.epsilon.ai.decision;

import com.epsilon.util.SeedMixer;

/**
 * 学習・評価・行動選択の用途ごとに、固定値と連番から再現可能な乱数シードを生成する。
 *
 * <p>並列処理の実行順序が変わっても、同じ用途・連番・席の組み合わせからは同じシードを得る。
 */
public final class EpsilonDecisionSeeds {

  private static final long TRAIN_GAME_SALT = 0xD1B54A32D192ED03L;
  private static final long EVAL_VS_GAME_SALT = 0xABC98388FB8FAC03L;
  private static final long PROMOTION_ATTEMPT_SALT = 0x4D4F4E4F544F4E45L;
  private static final long PLAYER_SALT = 0x89E182857D9ED689L;
  private static final long ADAPTIVE_PERCENTILE_SALT = 0xA24BAED4963EE407L;
  private static final long BRANCH_SALT = 0xC13FA9A902A6328FL;

  private EpsilonDecisionSeeds() {}

  /** 元対局の行動抽選を消費しない比較専用乱数列。 */
  public static long branch(long gameSeed, long decisionId) {
    return SeedMixer.indexed(gameSeed, BRANCH_SALT, decisionId);
  }

  /** 自己対戦の対局インデックスに対応するシードを返す。 */
  public static long trainGame(long seedBase, long gameIndex) {
    return SeedMixer.indexed(seedBase, TRAIN_GAME_SALT, gameIndex);
  }

  /** 同じ牌山と席順を使う比較評価の牌山シードインデックスに対応するシードを返す。 */
  public static long evalVsGame(long seedBase, long seedIndex) {
    return SeedMixer.indexed(seedBase, EVAL_VS_GAME_SALT, seedIndex);
  }

  /** モデルの採用判定ごとに独立した牌山系列の基点を返す。 */
  public static long promotionAttempt(long seedBase, long attempt) {
    return SeedMixer.indexed(seedBase, PROMOTION_ATTEMPT_SALT, attempt);
  }

  /** 同じ対局内のプレイヤー席ごとに独立した行動シードを返す。 */
  public static long player(long gameSeed, int seat) {
    return SeedMixer.indexed(gameSeed, PLAYER_SALT, seat);
  }

  /** 行動抽選の乱数系列を変えずに、累積分布の区間内での位置を分散させるプレイヤー固有のシードを返す。 */
  public static long adaptivePercentile(long playerSeed) {
    return SeedMixer.indexed(playerSeed, ADAPTIVE_PERCENTILE_SALT, 0L);
  }
}
