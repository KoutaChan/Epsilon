package com.epsilon.prism.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.EventRecords.Prefix;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Locale;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/** 実戦の判断を符号化し、再利用するライターの割り当て量と、前の判断の履歴の続きから書いた行が全てを書いた行と同じことを確かめる。 */
public class TokenizerWalkTest {
  private static final Logger log = LoggerFactory.getLogger(TokenizerWalkTest.class);

  private static final int SMALL_GAMES = 8;
  private static final int LARGE_GAMES = 100;

  /** 確保の数え方そのものの揺れを許す幅（バイト）。測る判断は 10^5 以上なので、1 判断に 1 つでも確保すれば MB に届く。 */
  private static final long SLACK_BYTES = 4_096;

  @Test(dataProvider = "formats", dataProviderClass = InputFormatTest.class, timeOut = 600_000)
  public void everyDecisionPassesTheWriterChecks(InputFormat format) {
    Walk walk = new Walk(format, o -> {});
    play(walk, SMALL_GAMES);
    log.info(walk.summary());
    assertTrue(walk.decisions >= 3_000, "decisions " + walk.decisions);
    if (format.variant() == TENHOU_3P) assertTrue(walk.nukis > 0, walk.summary());
    assertTrue(walk.kinds[EngineDecisionKind.TURN.ordinal()] > 0, walk.summary());
    assertTrue(walk.kinds[EngineDecisionKind.RESPONSE.ordinal()] > 0, walk.summary());
    assertTrue(walk.kinds[EngineDecisionKind.POST_CALL_DAHAI.ordinal()] > 0, walk.summary());
    assertTrue(walk.calls > 0 && walk.wins > 0, walk.summary());
  }

  @Test(
      dataProvider = "formats",
      dataProviderClass = InputFormatTest.class,
      groups = "slow",
      timeOut = 1_800_000)
  public void everyDecisionPassesTheWriterChecksWithoutAllocating(InputFormat format) {
    checkAllocations(format, o -> {});
  }

  public static void checkAllocations(InputFormat format, Consumer<Observation> extra) {
    // 同じ半荘を同じヘルパーで 2 回通し、2 回目を測る。初めて通るコードでの 1 回だけの確保（作業領域の伸び、enum の switch の表のクラスの初期化など）は 1 回目に済む。
    Walk walk = new Walk(format, extra);
    int games = format.variant() == TENHOU_3P ? LARGE_GAMES : 50;
    play(walk, games);
    walk.startMeasuring();
    play(walk, games);
    log.info(walk.summary());
    assertTrue(walk.decisions >= 100_000, "decisions " + walk.decisions);
    assertTrue(
        walk.calls > 0
            && walk.kans > 0
            && walk.wins > 0
            && (format.variant() != TENHOU_3P || walk.nukis > 0),
        walk.summary());
    assertTrue(
        walk.maxEvents > (format.variant() == TENHOU_3P ? 60 : 100),
        "longest public record " + walk.maxEvents);
    if (format.variant() != TENHOU_3P) {
      for (var kind : EngineDecisionKind.values()) {
        if (kind.in(format.variant()))
          assertTrue(walk.kinds[kind.ordinal()] > 0, "no " + kind + " decision");
      }
    }
    assertTrue(walk.allocated < SLACK_BYTES, "allocated bytes " + walk.summary());
  }

  /**
   * 観測席ごとの履歴の続きから書いた行が、全てを書いた行と blob の全てのバイトで同じ。分岐の写しから続ける判断、別の観測席の履歴を
   * 渡した判断（確かめで捨てる）、局が変わっても捨てない履歴を混ぜる。公開の時点で確定する形式では、写したイベントのバイトが前の判断と
   * 同じことも確かめる。
   */
  @Test(dataProvider = "formats", dataProviderClass = InputFormatTest.class, timeOut = 600_000)
  public void writingFromTheHistoryMatchesAFullWrite(InputFormat format) {
    RowEncoder full = new RowEncoder(format);
    InputFormat.RowWriter writer = format.writer();
    HostInputs host = HostInputs.allocate(format, RowEncoder.ONE_ROW, ByteBuffer::allocate);
    int seats = format.seats();
    EventHistory[] histories = new EventHistory[seats];
    Arrays.setAll(histories, seat -> new EventHistory());
    byte[][] previous = new byte[seats][];
    long[] counts = new long[5];
    TableWalk.Visitor visitor =
        new TableWalk.Visitor() {
          @Override
          public void decision(Engine engine, int slot) {
            Observation o = full.encode(engine, slot);
            int seat = o.observer();
            long decision = counts[0]++;
            if (decision % 11 == 0) histories[seat] = histories[seat].copy();
            boolean swapped = decision % 97 == 0;
            EventHistory history = swapped ? histories[(seat + 1) % seats] : histories[seat];
            BatchPlan plan = host.plan();
            plan.clear();
            assertEquals(plan.add(o), 0);
            plan.seal();
            writer.write(o, history, host, 0);
            HostBatch batch = host.input();
            assertEquals(batch.bytes(), full.batch.bytes());
            assertEquals(
                batch
                    .blob()
                    .slice(0, batch.bytes())
                    .mismatch(full.batch.blob().slice(0, full.batch.bytes())),
                -1,
                "decision " + decision + " differs from a full write");
            int prefix = batch.eventPrefix(0);
            if (swapped) {
              previous[(seat + 1) % seats] = null;
              return;
            }
            if (prefix > 0) {
              counts[1]++;
              counts[2] += prefix;
            }
            counts[3] += o.publicEvents();
            byte[] events = eventBytes(batch, format, o.publicEvents());
            if (format.eventsSettleOnPublication() && previous[seat] != null && prefix > 0) {
              int width = format.event().width() + format.relation().width();
              assertEquals(
                  Arrays.mismatch(events, 0, prefix * width, previous[seat], 0, prefix * width),
                  -1,
                  "settled event bytes changed at decision " + decision);
              counts[4]++;
            }
            previous[seat] = events;
          }

          @Override
          public void roundEnd(Engine engine) {
            if (counts[0] % 2 == 0) {
              for (EventHistory history : histories) history.clear();
            }
          }
        };
    for (TableWalk.Policy policy : TableWalk.Policy.values()) {
      new TableWalk(format.variant(), policy).run(7_300 + policy.ordinal(), SMALL_GAMES, visitor);
    }
    log.info(
        "HISTORY walk {}: decisions {} reused {} copied events {} of {} settled checks {}",
        format.fingerprint(),
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4]);
    assertTrue(
        counts[1] > counts[0] * 3 / 4,
        "rows that reused a history: " + counts[1] + " of " + counts[0]);
    assertTrue(counts[2] > counts[3] * 3 / 4, "copied events: " + counts[2] + " of " + counts[3]);
    if (format.eventsSettleOnPublication())
      assertTrue(counts[4] > counts[0] / 2, "settled checks " + counts[4]);
  }

  /** 共有した公開順を分岐後に追記し、既存の全トークンを再符号化せずに通常版と同じ入力を復元できる。 */
  @Test(dataProvider = "variants", dataProviderClass = InputFormatTest.class)
  public void publishedRangesSurviveForksAndRoundChanges(
      com.epsilon.mahjong.rules.Variant variant) {
    var format = InputFormat.of(variant, java.util.EnumSet.allOf(InputFeature.class));
    checkPublishedRanges(format, false);
    Runnable close = PublishedEventStore.register(format, 128);
    try {
      checkPublishedRanges(format, true);
    } finally {
      close.run();
    }
  }

  private static void checkPublishedRanges(InputFormat format, boolean limited) {
    var variant = format.variant();
    var full = new RowEncoder(format);
    var writer = format.writer();
    var host = HostInputs.allocate(format, RowEncoder.ONE_ROW, ByteBuffer::allocate);
    var batch = host.input();
    batch.enablePublishedEvents();
    var tips = new Prefix[variant.seats()];
    int[] reused = {0}, uncached = {0};
    new TableWalk(variant, TableWalk.Policy.CALLS)
        .run(
            7301,
            1,
            new TableWalk.Visitor() {
              @Override
              public void decision(Engine engine, int slot) {
                var o = full.encode(engine, slot);
                int seat = o.observer();
                var previous = tips[seat];
                var current = Prefix.append(previous, o.events(), o.kanDoraIndicators());
                var sibling = Prefix.append(previous, o.events(), o.kanDoraIndicators());
                tips[seat] = current;
                if (current != previous && previous != null) previous.release();
                var plan = batch.plan();
                plan.clear();
                assertEquals(plan.add(o), 0);
                plan.seal();
                // 親から同時に分かれた 2 経路が、それぞれ親の末尾から追記する。
                int expectedPrefix = previous == null ? 0 : previous.end;
                for (var tip : new Prefix[] {sibling, current}) {
                  batch.publishedPrefix(0, tip);
                  writer.write(o, host, 0);
                  int archived = batch.eventPrefix(0);
                  assertTrue(archived <= expectedPrefix);
                  reused[0] += archived;
                  uncached[0] += expectedPrefix - archived;
                  assertEquals(
                      batch
                          .blob()
                          .slice(0, plan.bytes())
                          .mismatch(full.batch.blob().slice(0, plan.bytes())),
                      -1);
                }
                if (sibling != null && sibling != previous) sibling.release();
                batch.releaseEvents();
              }

              @Override
              public void roundEnd(Engine engine) {
                for (int seat = 0; seat < tips.length; seat++) {
                  if (tips[seat] != null) tips[seat].release();
                  tips[seat] = null;
                }
              }
            });
    // 最終局は半荘終了の境界へ直接進む場合もある。
    for (Prefix tip : tips) if (tip != null) tip.release();
    assertTrue(reused[0] > 0);
    assertEquals(uncached[0] > 0, limited);
  }

  /** 1 行のバッチの、イベントのトークンのカテゴリの行と関係の行を、イベントごとに並べて返す。 */
  private static byte[] eventBytes(HostBatch batch, InputFormat format, int events) {
    int eventWidth = format.event().width();
    int relationWidth = format.relation().width();
    byte[] out = new byte[events * (eventWidth + relationWidth)];
    ByteBuffer blob = batch.blob();
    int categories = batch.plan().offset(Tensor.EVENT_CATEGORY);
    int relations =
        batch.plan().offset(Tensor.RELATION) + batch.plan().tokenStart(Axis.EVENT) * relationWidth;
    for (int p = 0; p < events; p++) {
      blob.get(categories + p * eventWidth, out, p * (eventWidth + relationWidth), eventWidth);
      blob.get(
          relations + p * relationWidth,
          out,
          p * (eventWidth + relationWidth) + eventWidth,
          relationWidth);
    }
    return out;
  }

  private static void play(Walk walk, int games) {
    for (TableWalk.Policy policy : TableWalk.Policy.values()) {
      new TableWalk(walk.encoder.format.variant(), policy)
          .run(
              (walk.encoder.format.variant() == TENHOU_3P ? 7_100 : 7_000) + policy.ordinal(),
              games,
              walk);
    }
  }

  private static final class Walk implements TableWalk.Visitor {
    private final RowEncoder encoder;
    private final Consumer<Observation> extra;

    Walk(InputFormat format, Consumer<Observation> extra) {
      encoder = new RowEncoder(format);
      this.extra = extra;
    }

    private final ThreadMXBean threads = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    private final long[] kinds = new long[EngineDecisionKind.values().length];
    private boolean measure;
    private long decisions;
    private long calls;
    private long kans;
    private long wins;
    private long nukis;
    private int maxEvents;
    private long allocated;

    void startMeasuring() {
      measure = true;
      decisions = 0;
      calls = 0;
      kans = 0;
      wins = 0;
      nukis = 0;
      maxEvents = 0;
      Arrays.fill(kinds, 0L);
    }

    @Override
    public void decision(Engine engine, int slot) {
      long before = measure ? threads.getCurrentThreadAllocatedBytes() : 0L;
      Observation o = encoder.encode(engine, slot);
      extra.accept(o);
      if (measure) {
        allocated += threads.getCurrentThreadAllocatedBytes() - before;
      }
      decisions++;
      kinds[o.decisionKind().ordinal()]++;
      maxEvents = Math.max(maxEvents, o.publicEvents());
      for (int action = 0; action < o.legalActions(); action++) {
        switch (o.action(action).type()) {
          case CHI, PON -> calls++;
          case DAIMINKAN, ANKAN, KAKAN -> kans++;
          case TSUMO_AGARI, RON_AGARI -> wins++;
          case NUKI -> nukis++;
          default -> {}
        }
      }
    }

    String summary() {
      return String.format(
          Locale.ROOT,
          "TOKENIZER walk: decisions %d kinds %s calls %d kans %d wins %d nukis %d maxPublicEvents"
              + " %d allocated %d B",
          decisions,
          Arrays.toString(kinds),
          calls,
          kans,
          wins,
          nukis,
          maxEvents,
          allocated);
    }
  }
}
