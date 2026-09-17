package com.epsilon.training;

import com.epsilon.engine.Player;
import com.epsilon.engine.SynchronousGameRunner;
import com.epsilon.engine.player.RandomPlayer;
import com.epsilon.util.FormatUtils;
import java.util.AbstractList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** マルチスレッドで対局シミュレーションを並列実行する。 */
public final class ParallelSimulator {

  private static final Logger log = LoggerFactory.getLogger(ParallelSimulator.class);

  private final int numThreads;
  private final ExecutorService executor;

  /** 利用可能な論理プロセッサ数をワーカー数としてシミュレーターを生成する。 */
  public ParallelSimulator() {
    this(Runtime.getRuntime().availableProcessors());
  }

  /**
   * 指定したワーカー数でシミュレーターを生成する。
   *
   * @param numThreads 並列に対局を処理するワーカー数
   */
  public ParallelSimulator(int numThreads) {
    if (numThreads <= 0) {
      throw new IllegalArgumentException("numThreads must be positive: " + numThreads);
    }
    this.numThreads = numThreads;
    this.executor = Executors.newFixedThreadPool(numThreads);
    log.info("ParallelSimulator initialized with {} threads", numThreads);
  }

  /**
   * 指定数の対局を並列実行する。
   *
   * @param numGames 対局数
   * @param playerFactory 各対局用のプレイヤー配列を生成するファクトリ
   * @return 各対局の結果スコア
   */
  public List<int[]> runGames(int numGames, PlayerFactory playerFactory) {
    if (numGames < 0) {
      throw new IllegalArgumentException("numGames must not be negative: " + numGames);
    }
    int[][] results = new int[numGames][];
    int workerCount = Math.min(numThreads, numGames);
    Future<?>[] workers = new Future<?>[workerCount];
    for (int worker = 0; worker < workerCount; worker++) {
      int workerIndex = worker;
      workers[worker] =
          executor.submit(
              () -> {
                for (int game = workerIndex; game < numGames; game += workerCount) {
                  Player[] players = playerFactory.createPlayers();
                  results[game] = new SynchronousGameRunner(players).playHanchan();
                }
              });
    }
    for (Future<?> worker : workers) {
      try {
        worker.get();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException("Game simulation interrupted", e);
      } catch (ExecutionException e) {
        throw new IllegalStateException("Game simulation failed", e.getCause());
      }
    }

    return new ResultView(results);
  }

  /**
   * ランダムプレイヤー同士の半荘を実行し、1秒あたりの対局数をログへ出力する。
   *
   * @param numGames 測定する半荘数
   */
  public void benchmark(int numGames) {
    log.info("Running benchmark with {} games on {} threads...", numGames, numThreads);
    long start = System.currentTimeMillis();

    List<int[]> results =
        runGames(
            numGames,
            () -> {
              Player[] players = new Player[4];
              for (int i = 0; i < 4; i++) {
                players[i] = new RandomPlayer();
              }
              return players;
            });

    long elapsed = System.currentTimeMillis() - start;
    int completedGames = results.size();
    double gamesPerSec = gamesPerSecond(completedGames, elapsed);

    log.info(
        "Benchmark: {} completed / {} requested in {} ms ({} games/sec)",
        completedGames,
        numGames,
        elapsed,
        FormatUtils.fixed0(gamesPerSec));
  }

  static double gamesPerSecond(int completedGames, long elapsedMs) {
    if (completedGames <= 0 || elapsedMs <= 0) {
      return 0.0;
    }
    return (double) completedGames / elapsedMs * 1000;
  }

  /** 新規対局の受付を止め、実行中のワーカーが終了するまで最大60秒待機する。 */
  public void shutdown() {
    executor.shutdown();
    try {
      executor.awaitTermination(60, TimeUnit.SECONDS);
    } catch (InterruptedException e) {
      executor.shutdownNow();
      Thread.currentThread().interrupt();
    }
  }

  /** 対局ごとに独立した4席分のプレイヤー配列を生成する。 */
  @FunctionalInterface
  public interface PlayerFactory {
    /**
     * 一つの半荘で使用する4席分のプレイヤーを生成する。
     *
     * @return 席順どおりに並んだ4要素のプレイヤー配列
     */
    Player[] createPlayers();
  }

  /** 完了順にかかわらず、対局番号順に結果配列を参照するリスト。配列は複製しない。 */
  private static final class ResultView extends AbstractList<int[]> {
    private final int[][] results;

    private ResultView(int[][] results) {
      this.results = results;
    }

    @Override
    public int[] get(int index) {
      return results[index];
    }

    @Override
    public int size() {
      return results.length;
    }
  }
}
