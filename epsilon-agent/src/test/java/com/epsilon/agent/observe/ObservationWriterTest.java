package com.epsilon.agent.observe;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.observe.ObservationSchema.HeaderBits;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/** 観測を書くクラスのテスト。ルールや席の食い違いと、観測のフィールドの範囲を外れる値を拒む。書き足したイベントのセクションは初めから書いたものと同じ。 */
public class ObservationWriterTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  @Test
  public void rejectsAnotherVariant() {
    Engine engine = firstDecision(RoundStart.first(VARIANT));
    Decisions decisions = engine.decisions();
    SeatView view = engine.view(decisions.seat(0));
    ObservationWriter writer = new ObservationWriter(VARIANT);
    ObservationBatch three = new ObservationBatch(Variant.TENHOU_3P, 64);
    expectThrows(IllegalArgumentException.class, () -> writer.capture(view, decisions, 0, three));

    ObservationBatch four = new ObservationBatch(VARIANT, 64);
    int index = writer.capture(view, decisions, 0, four);
    expectThrows(
        IllegalArgumentException.class, () -> new Observation(Variant.TENHOU_3P).bind(four, index));
    Observation read = new Observation(VARIANT).bind(four, index);
    expectThrows(IllegalArgumentException.class, () -> three.add(read));
    assertEquals(four.add(read), index + 1);
  }

  @Test
  public void rejectsAViewOfAnotherSeat() {
    Engine engine = firstDecision(RoundStart.first(VARIANT));
    Decisions decisions = engine.decisions();
    SeatView other = engine.view((decisions.seat(0) + 1) % VARIANT.seats());
    ObservationBatch batch = new ObservationBatch(VARIANT, 64);
    expectThrows(
        IllegalArgumentException.class,
        () -> new ObservationWriter(VARIANT).capture(other, decisions, 0, batch));
  }

  /**
   * 点数が 100 点単位でない卓と、本場が観測の上限を超える卓は書けず、バッチに何も残さない。観測が配列に収まっていなければ読めない。
   * フィールドの範囲を外れる値は書けない。
   */
  @Test
  public void rejectsValuesOutsideTheFields() {
    ObservationWriter writer = new ObservationWriter(VARIANT);
    ObservationBatch batch = new ObservationBatch(VARIANT, 64);
    expectThrows(
        IllegalArgumentException.class,
        () ->
            capture(
                writer,
                new RoundStart(VARIANT, 0, 0, 0, new int[] {25_050, 24_950, 25_000, 25_000}),
                batch));
    int[] even = {25_000, 25_000, 25_000, 25_000};
    expectThrows(
        IllegalArgumentException.class,
        () ->
            capture(
                writer,
                new RoundStart(VARIANT, 0, ObservationSchema.MAX_HONBA + 1, 0, even),
                batch));
    assertEquals(batch.size(), 0);

    int index =
        capture(writer, new RoundStart(VARIANT, 0, ObservationSchema.MAX_HONBA, 0, even), batch);
    long[] truncated =
        Arrays.copyOfRange(
            batch.words(), batch.start(index), batch.start(index) + batch.longs(index) - 1);
    expectThrows(
        IndexOutOfBoundsException.class, () -> new Observation(VARIANT).bind(truncated, 0));

    ObservationSchema schema = ObservationSchema.of(VARIANT);
    long[] header = new long[schema.header.longs(1)];
    schema.header.put(header, 0, 0, HeaderBits.WALL, VARIANT.liveWall());
    expectThrows(
        IllegalArgumentException.class,
        () -> schema.header.put(header, 0, 0, HeaderBits.WALL, VARIANT.liveWall() + 1));
    expectThrows(
        IllegalArgumentException.class,
        () -> schema.header.put(header, 0, 0, HeaderBits.INDICATORS, 0));
    expectThrows(
        IllegalArgumentException.class,
        () -> schema.header.get(header, 0, 0, HeaderBits.INDICATOR));
  }

  /** 卓の最初の判断で、判断する 1 番目の席の観測を書き、バッチの中の番号を返す。 */
  private static int capture(ObservationWriter writer, RoundStart start, ObservationBatch batch) {
    Engine engine = firstDecision(start);
    Decisions decisions = engine.decisions();
    return writer.capture(engine.view(decisions.seat(0)), decisions, 0, batch);
  }

  /**
   * 経路のイベントのセクションに書き足して作った O は、全てのイベントを書いた O とビット単位で同じで、付けたリーダーはバッチの O を
   * 読み直したリーダーと同じ値を返す。局の変わり目（経路がセクションを捨てる）、
   * 鳴かれた打牌、写した局面を局の終わりまで進める分岐（親のセクションの写しから始める）を含む。
   */
  @Test
  public void appendedEventsMatchAFullWrite() {
    RandomPlay play = new RandomPlay(VARIANT, 31);
    ObservationWriter writer = new ObservationWriter(VARIANT);
    EventRecords path = new EventRecords(VARIANT);
    int compared = 0;
    for (int step = 0; step < 30_000; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        compared += compareAll(writer, play.engine, path);
        if (step % 41 == 0) {
          Engine fork = play.engine.copy();
          EventRecords forked = new EventRecords(path);
          int[] first = new int[VARIANT.seats() - 1];
          for (Boundary at = Boundary.DECISIONS;
              at == Boundary.DECISIONS;
              at = fork.commit(first, play.ws)) {
            compared += compareAll(writer, fork, forked);
          }
        }
      }
      Boundary before = play.boundary();
      play.step();
      if (before != Boundary.DECISIONS) path.reset();
    }
    Assert.assertTrue(compared > 20_000, compared + " observations compared");
  }

  /** 判断の時点の全ての判断について、経路のセクションに書き足した O と全て書いた O を比べ、比べた数を返す。 */
  private static int compareAll(ObservationWriter writer, Engine engine, EventRecords path) {
    Decisions decisions = engine.decisions();
    ObservationBatch appended = new ObservationBatch(VARIANT, 256),
        full = new ObservationBatch(VARIANT, 256);
    Observation attached = new Observation(VARIANT), read = new Observation(VARIANT);
    for (int slot = 0; slot < decisions.size(); slot++) {
      SeatView view = engine.view(decisions.seat(slot));
      int a = writer.capture(view, decisions, slot, appended, path, attached);
      var tip = path.prefix(view.seat(), attached.kanDoraIndicators());
      appended.rememberEvents(a, tip);
      Assert.assertSame(appended.eventPrefix(a), tip);
      if (tip != null) {
        Assert.assertEquals(tip.records, attached.events());
        Assert.assertEquals(tip.end, attached.publicEvents());
        if (slot == 0 && tip.end % 19 == 0) {
          EventRecords fork = new EventRecords(path);
          Assert.assertSame(fork.prefix(view.seat(), attached.kanDoraIndicators()), tip);
          fork.reset();
          Assert.assertNull(fork.prefix(view.seat(), 0));
          Assert.assertSame(path.prefix(view.seat(), attached.kanDoraIndicators()), tip);
        }
      }
      int f = writer.capture(view, decisions, slot, full);
      assertEquals(
          Arrays.copyOfRange(
              appended.words(), appended.start(a), appended.start(a) + appended.longs(a)),
          Arrays.copyOfRange(full.words(), full.start(f), full.start(f) + full.longs(f)),
          "slot " + slot);
      read.bind(appended, a);
      assertEquals(attached.longs(), read.longs(), "slot " + slot);
      assertEquals(attached.events(), read.events(), "slot " + slot);
      assertEquals(attached.wins(), read.wins(), "slot " + slot);
      assertEquals(attached.transitions(), read.transitions(), "slot " + slot);
      for (int action = 0; action < read.legalActions(); action++) {
        assertEquals(
            attached.action(action), read.action(action), "slot " + slot + " action " + action);
        assertEquals(attached.representative(action), read.representative(action), "slot " + slot);
      }
    }
    return decisions.size();
  }

  private static Engine firstDecision(RoundStart start) {
    Engine engine = new Engine(VARIANT);
    assertEquals(new Scenes(start).start(engine, new RuleWorkspace(VARIANT)), Boundary.DECISIONS);
    return engine;
  }

  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 100_000;

  /**
   * 確保の数え方そのものの揺れとして許すバイト数。測るのは 10^5 ステップなので、1 ステップに 1 回でも確保（16 バイト以上）があれば、
   * 合計は MB 単位になる。
   */
  private static final long SLACK_BYTES = 4_096;

  private final RandomPlay play = new RandomPlay(VARIANT, 23);
  private final ObservationWriter writer = new ObservationWriter(VARIANT);
  private final ObservationBatch batch = new ObservationBatch(VARIANT, 1_024);
  private final Observation read = new Observation(VARIANT);

  @Test
  public void capturesAllocateNothingOnceWarm() {
    ThreadMXBean threads =
        ManagementFactory.getThreadMXBean() instanceof ThreadMXBean bean ? bean : null;
    if (threads == null
        || !threads.isThreadAllocatedMemorySupported()
        || !threads.isThreadAllocatedMemoryEnabled()) {
      throw new SkipException("Per-thread allocation counters are not available");
    }
    long checksum = run(WARM_UP);
    long before = threads.getCurrentThreadAllocatedBytes();
    checksum += run(MEASURED);
    long allocated = threads.getCurrentThreadAllocatedBytes() - before;
    Assert.assertTrue(
        allocated < SLACK_BYTES,
        allocated + " bytes over " + MEASURED + " steps (checksum " + checksum + ")");
  }

  /**
   * 自己対局を決まったステップ数だけ進め、判断の時点ごとに、判断する全ての席の観測を書いて読む。最適化で処理が消されないように、
   * 読んだ値の和を返す。
   */
  private long run(int steps) {
    long checksum = 0;
    for (int step = 0; step < steps; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        Decisions decisions = play.engine.decisions();
        batch.clear();
        for (int slot = 0; slot < decisions.size(); slot++) {
          int index =
              writer.capture(play.engine.view(decisions.seat(slot)), decisions, slot, batch);
          read.bind(batch, index);
          checksum += read.transitions() + read.events() + read.wins();
        }
      }
      play.step();
    }
    return checksum;
  }
}
