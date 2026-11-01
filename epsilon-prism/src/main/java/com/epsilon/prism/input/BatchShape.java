package com.epsilon.prism.input;

import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputSchema.Axis;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.LongAdder;

/**
 * 封じた塊の形の度数。行のトークンの数 T = {@link BatchPlan#FIXED_TOKENS} + 出来事の数 を、注意の核が受け持つ 16 トークンの区切りの数
 * B = ⌈T/16⌉ ごとに数え（行ごとの注意の対の区切りは B² 個）、出来事・遷移・手の和も持つ。本番の塊の形を測りの塊の形と比べるためのもの。
 *
 * <p>{@link #add} は塊ごとに 1 回呼ぶ。行ごとの仕事は出来事の数の差 1 つと配列の加算 1 つで、度数はスレッドの配列に数えてから区切りごとに
 * 1 回だけ足す。スレッド安全。
 */
public final class BatchShape {

  /** 注意の核の 1 区切りのトークンの数。 */
  public static final int BLOCK = 16;

  /** 区切りの数の最大（最も長い行）。 */
  static final int MAX_BLOCKS = blocks(BatchPlan.FIXED_TOKENS + Limits.MAX_EVENT_TOKENS);

  /** これを超える区切りの数の行は T > 128（長い行の道を通る）。 */
  private static final int LONG_BLOCKS = 128 / BLOCK;

  /** 区切りの数 B ごとの行の数（添字 B）。 */
  private final AtomicLongArray rows = new AtomicLongArray(MAX_BLOCKS + 1);

  private final LongAdder batches = new LongAdder();
  private final LongAdder events = new LongAdder();
  private final LongAdder transitions = new LongAdder();
  private final LongAdder actions = new LongAdder();

  /**
   * 塊の行を数える。
   *
   * @param plan 封じた計画
   */
  public void add(BatchPlan plan) {
    int[] counts = new int[MAX_BLOCKS + 1];
    int count = plan.rows();
    for (int row = 0; row < count; row++) {
      counts[blocks(BatchPlan.FIXED_TOKENS + plan.events(row))]++;
    }
    for (int b = 0; b <= MAX_BLOCKS; b++) {
      if (counts[b] != 0) {
        rows.addAndGet(b, counts[b]);
      }
    }
    batches.increment();
    events.add(plan.count(Axis.EVENT));
    transitions.add(plan.count(Axis.TRANSITION));
    actions.add(plan.count(Axis.ACTION));
  }

  /**
   * ここまでの度数。
   *
   * @return 度数（数えている途中なら、塊の切れ目でそろわない値が混じりうる）
   */
  public Summary summary() {
    long[] histogram = new long[MAX_BLOCKS + 1];
    for (int b = 0; b <= MAX_BLOCKS; b++) {
      histogram[b] = rows.get(b);
    }
    return new Summary(
        batches.sum(), histogram, events.sum(), transitions.sum(), actions.sum());
  }

  /** トークンの数 T の行の区切りの数 ⌈T/16⌉。 */
  static int blocks(int tokens) {
    return (tokens + BLOCK - 1) / BLOCK;
  }

  /**
   * 塊の形の度数。
   *
   * @param batches 塊の数
   * @param histogram 区切りの数 B ごとの行の数（添字 B）
   * @param events 出来事のトークンの和
   * @param transitions 遷移の和
   * @param actions 手の和
   */
  public record Summary(
      long batches, long[] histogram, long events, long transitions, long actions) {

    /** 行の数。 */
    public long rows() {
      long rows = 0;
      for (long count : histogram) {
        rows += count;
      }
      return rows;
    }

    /** 行あたりのトークン（T の平均）。 */
    public double tokensPerRow() {
      long rows = rows();
      return rows == 0 ? 0 : BatchPlan.FIXED_TOKENS + (double) events / rows;
    }

    /** 行あたりの注意の対の区切りの数（Σ⌈T/16⌉² / 行）。 */
    public double blockPairsPerRow() {
      long rows = rows();
      long pairs = 0;
      for (int b = 0; b < histogram.length; b++) {
        pairs += (long) b * b * histogram[b];
      }
      return rows == 0 ? 0 : (double) pairs / rows;
    }

    /** T > 128 の行の数。 */
    public long longRows() {
      long rows = 0;
      for (int b = LONG_BLOCKS + 1; b < histogram.length; b++) {
        rows += histogram[b];
      }
      return rows;
    }

    /**
     * 記録の 1 行にする（rows・tokens/row・blocks²/row・T>128 の行・遷移と手の行あたり・B ごとの行の割合 %）。
     *
     * @return 行
     */
    public String line() {
      long rows = rows();
      StringBuilder blocks = new StringBuilder();
      for (int b = 0; b < histogram.length; b++) {
        if (histogram[b] != 0) {
          blocks
              .append(blocks.isEmpty() ? "" : ",")
              .append(b)
              .append(':')
              .append(String.format(Locale.ROOT, "%.2f", 100.0 * histogram[b] / rows));
        }
      }
      return String.format(
          Locale.ROOT,
          "rows=%d rows/batch=%.1f tokens/row=%.2f blocks2/row=%.3f rowsT>128=%d (%.3f%%)"
              + " transitions/row=%.3f actions/row=%.3f blocks%%=%s",
          rows,
          batches == 0 ? 0 : (double) rows / batches,
          tokensPerRow(),
          blockPairsPerRow(),
          longRows(),
          rows == 0 ? 0 : 100.0 * longRows() / rows,
          rows == 0 ? 0 : (double) transitions / rows,
          rows == 0 ? 0 : (double) actions / rows,
          blocks);
    }
  }
}
