package com.epsilon.calculate.benchmark;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 固定入力を使い、和了形の参照と役・符・得点計算の速度を測る。ニューラルネットワークは使用しない。 */
public final class HandScoreEvaluatorBenchmark {

  private static final Logger log = LoggerFactory.getLogger(HandScoreEvaluatorBenchmark.class);
  private static final ThreadMXBean THREADS = (ThreadMXBean) ManagementFactory.getThreadMXBean();

  private HandScoreEvaluatorBenchmark() {}

  public static void main(String[] args) {
    Locale.setDefault(Locale.ROOT);
    int warmup = args.length > 0 ? Integer.parseInt(args[0]) : 500_000;
    int measurement = args.length > 1 ? Integer.parseInt(args[1]) : 5_000_000;
    String notation = args.length > 2 ? args[2] : "123456789m123p44p";
    int agariTile = args.length > 3 ? Integer.parseInt(args[3]) : Tile.P3;
    Hand complete = hand(notation);
    Hand ready = complete.snapshotCopy();
    ready.remove(agariTile);
    HandScoreEvaluator evaluator = new HandScoreEvaluator();
    HandScoreBuffer result = new HandScoreBuffer();
    WinConditions context = WinConditions.publicDecision(Tile.NAN, Tile.TON, RiichiState.RIICHI);

    run(complete, ready, agariTile, context, evaluator, result, warmup);
    long threadId = Thread.currentThread().threadId();
    long allocatedBefore = THREADS.getThreadAllocatedBytes(threadId);
    long started = System.nanoTime();
    long checksum = run(complete, ready, agariTile, context, evaluator, result, measurement);
    long elapsed = System.nanoTime() - started;
    long allocated = THREADS.getThreadAllocatedBytes(threadId) - allocatedBefore;
    log.info(
        String.format(
            Locale.ROOT,
            "hand=%s agari_tile=%d evaluations=%d elapsed_ns=%d evaluations_per_sec=%.3f"
                + " ns_per_evaluation=%.3f bytes_allocated=%d bytes_per_evaluation=%.6f"
                + " checksum=%d",
            notation,
            agariTile,
            measurement,
            elapsed,
            measurement * 1_000_000_000.0 / elapsed,
            (double) elapsed / measurement,
            allocated,
            (double) allocated / measurement,
            checksum));
  }

  private static long run(
      Hand complete,
      Hand ready,
      int agariTile,
      WinConditions context,
      HandScoreEvaluator evaluator,
      HandScoreBuffer result,
      int evaluations) {
    long checksum = 0L;
    for (int index = 0; index < evaluations; index++) {
      boolean evaluated =
          (index & 1) == 0
              ? evaluator.scoreCompleted(
                  complete, agariTile, WinMethod.RON, context, null, false, 0, result)
              : evaluator.scoreAfterAdding(
                  ready, agariTile, WinMethod.RON, context, null, false, 0, result);
      if (!evaluated) throw new IllegalStateException("benchmark fixture has no score");
      checksum += result.basePoints();
    }
    return checksum;
  }

  private static Hand hand(String notation) {
    Hand hand = new Hand();
    int numberStart = 0;
    for (int index = 0; index < notation.length(); index++) {
      char value = notation.charAt(index);
      if (value >= '1' && value <= '9') continue;
      int base =
          switch (value) {
            case 'm' -> Tile.M1;
            case 'p' -> Tile.P1;
            case 's' -> Tile.S1;
            case 'z' -> Tile.TON;
            default -> throw new IllegalArgumentException("invalid notation: " + notation);
          };
      for (int numberIndex = numberStart; numberIndex < index; numberIndex++) {
        hand.add(base + notation.charAt(numberIndex) - '1');
      }
      numberStart = index + 1;
    }
    return hand;
  }
}
