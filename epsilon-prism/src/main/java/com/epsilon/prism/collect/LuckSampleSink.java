package com.epsilon.prism.collect;

import com.epsilon.prism.luck.LuckBatch;
import com.epsilon.prism.luck.LuckSampleWriter;
import com.epsilon.prism.luck.LuckSampling;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;

/**
 * 収集が行と並べて集める、運の網 Ω の標本の受け口（luck-fit の材料）。硬貨（{@link LuckSampling}）と書き手（{@link LuckSampleWriter}）を
 * 持つ。経路ごとの捕捉（{@link LuckRecorder}）が局ごとに標本の束を作り、ドライバがその経路の局末の Φ で {@link #settle} して書き手に渡す。
 *
 * <p>標本は収集の脇道なので、捕捉・局末・書き手のどこかが失敗しても収集は止めない。最初の失敗を持ち（{@link #failure}）、以後このコホートでは標本を
 * 集めない。失敗したコホートの標本は封印しない（呼び出し側が {@link #failure} と、捕捉した数と書いた数の一致を見て決める）。
 */
public final class LuckSampleSink {

  /**
   * 1 コホートの件数。
   *
   * @param events 硬貨を引いた出来事（全ての経路のツモと槓ドラの表示）
   * @param captured 捕捉した標本（局末に束で渡した数）
   * @param samples 局末の Φ を付けて書き手に渡した標本
   * @param luckOnlyHands 標本のためだけに局末の Φ を求めた局（枠のない主経路の、半荘の続く局）
   * @param settleSeconds ドライバが標本に Y を付けて書き手に渡した時間（枠の圧縮を含む）
   */
  public record Counts(
      long events, long captured, long samples, long luckOnlyHands, double settleSeconds) {}

  private final LuckSampling sampling;
  private final LuckSampleWriter writer;
  private final AtomicReference<Exception> failure = new AtomicReference<>();
  private final LongAdder events = new LongAdder();
  private final LongAdder captured = new LongAdder();
  private final LongAdder samples = new LongAdder();
  private final LongAdder luckOnlyHands = new LongAdder();
  private final LongAdder settleNanos = new LongAdder();

  /**
   * 受け口を作る。
   *
   * @param sampling 標本を引く硬貨（種はコホートの種）
   * @param writer コホートの標本の書き手（封印と閉じるのは呼び出し側）
   */
  public LuckSampleSink(LuckSampling sampling, LuckSampleWriter writer) {
    this.sampling = Objects.requireNonNull(sampling, "sampling");
    this.writer = Objects.requireNonNull(writer, "writer");
  }

  /**
   * 最初の失敗を返す。
   *
   * @return 失敗（なければ null）
   */
  public Exception failure() {
    return failure.get();
  }

  /** 件数の写し。 */
  public Counts counts() {
    return new Counts(
        events.sum(),
        captured.sum(),
        samples.sum(),
        luckOnlyHands.sum(),
        settleNanos.sum() / 1e9);
  }

  LuckSampling sampling() {
    return sampling;
  }

  /** まだ失敗していない（標本を集め続ける）。 */
  boolean active() {
    return failure.get() == null;
  }

  /** 失敗を記録し、以後の標本をやめる（最初の失敗だけを持つ）。 */
  void fail(Exception problem) {
    failure.compareAndSet(null, problem);
  }

  /**
   * 経路の 1 局の終わり（進行ワーカー）: 硬貨を引いた出来事の数と、捕捉した標本の数を足す。
   *
   * @param drawn その局で硬貨を引いた出来事の数
   * @param batch その局の標本（なければ null）
   */
  void ended(long drawn, LuckBatch batch) {
    events.add(drawn);
    if (batch != null) {
      captured.add(batch.size());
    }
  }

  /**
   * 経路の局末の Φ を Y にして、その局の標本を書き手に渡す（ドライバのスレッド）。失敗したら記録して、以後の標本をやめる。
   *
   * @param batch その局の標本
   * @param phi 局末の Φ（絶対席の順。半荘末は最終効用）
   * @param luckOnly 枠がなく、標本のためだけに局末の Φ を求めた局か
   */
  void settle(LuckBatch batch, double[] phi, boolean luckOnly) {
    if (luckOnly) {
      luckOnlyHands.increment();
    }
    if (!active()) {
      return;
    }
    long started = System.nanoTime();
    try {
      batch.settle(phi, writer);
      samples.add(batch.size());
    } catch (IOException | RuntimeException problem) {
      fail(problem);
    }
    settleNanos.add(System.nanoTime() - started);
  }
}
