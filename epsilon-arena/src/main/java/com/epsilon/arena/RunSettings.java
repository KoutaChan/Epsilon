package com.epsilon.arena;

import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsPrefix;
import com.epsilon.util.SeedMixer;

/** 同じ牌山の乱数シードで4通りの席替えを行う対局群を、1組として実行するための設定。 */
@SettingsPrefix("arena")
public record RunSettings(
    @Setting("games") int games,
    @Setting("concurrentGames") int concurrentGames,
    @Setting("workers") int workers,
    @Setting("seed") long seed,
    @Setting("maxBatchWaitMicros") long maxBatchWaitMicros,
    @Setting("firstWallFamily") long firstWallFamily,
    @Setting("slotsPerDevice") int slotsPerDevice,
    @Setting("readyBatchesPerDevice") int readyBatchesPerDevice) {
  private static final long EVAL_VS_WALL_SALT = 0xABC98388FB8FAC03L;

  public RunSettings(
      int games, int concurrentGames, int workers, long seed, long maxBatchWaitMicros) {
    this(games, concurrentGames, workers, seed, maxBatchWaitMicros, 0L, 2, 4);
  }

  public RunSettings(
      int games,
      int concurrentGames,
      int workers,
      long seed,
      long maxBatchWaitMicros,
      long firstWallFamily) {
    this(games, concurrentGames, workers, seed, maxBatchWaitMicros, firstWallFamily, 2, 4);
  }

  public RunSettings {
    if (games < 4 || games % 4 != 0)
      throw new IllegalArgumentException("games must be a positive multiple of 4");
    if (concurrentGames < 1 || workers < 1 || maxBatchWaitMicros < 0)
      throw new IllegalArgumentException(
          "concurrentGames/workers must be positive and batch wait nonnegative");
    if (firstWallFamily < 0 || firstWallFamily > Long.MAX_VALUE - games / 4)
      throw new IllegalArgumentException("firstWallFamily is outside the valid wall range");
    if (slotsPerDevice < 1 || slotsPerDevice > 4 || readyBatchesPerDevice < slotsPerDevice)
      throw new IllegalArgumentException(
          "slotsPerDevice must be 1..4 and readyBatchesPerDevice >= slotsPerDevice");
  }

  /** 旧 eval-vs と同じシード系列を使い、実行順序に依存しない牌山を割り当てる。 */
  long wallSeed(int gameIndex) {
    return SeedMixer.indexed(seed, EVAL_VS_WALL_SALT, firstWallFamily + gameIndex / 4);
  }

  /** 対戦環境に同梱した実行既定値を型付きで取得する。 */
  public static RunSettings defaults() {
    return ArenaConfig.defaultSettings().bind(RunSettings.class);
  }
}
