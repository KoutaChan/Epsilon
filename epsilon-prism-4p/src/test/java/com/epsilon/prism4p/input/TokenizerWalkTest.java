package com.epsilon.prism4p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.prism.TableWalk;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * ランダムに打つ自己対局（3 通りの打ち方）の全ての判断で、観測 O を {@link Tokenizer} と {@link FrontierFeatures} に通すと、ライターの
 * 確かめ（キーのコード・手のインデックス・語彙）を通る。ウォームアップの後は、取り込み・Tokenizer・バッチの確定・フロンティアの計算で
 * メモリを確保しない。
 */
public class TokenizerWalkTest {
  private static final Logger log = LoggerFactory.getLogger(TokenizerWalkTest.class);

  private static final int GAMES = 50;

  /**
   * メモリの確保を数える仕組みそのものの揺れ（JIT の最適化が取り消されて、消えていた確保が戻るなど）を許す幅。測る判断は 10^5 以上あるので、
   * 1 判断に 1 つでも確保すれば MB に届く。
   */
  private static final long SLACK_BYTES = 4_096;

  @Test(timeOut = 1_800_000)
  public void everyDecisionPassesTheWriterChecksWithoutAllocating() {
    // 同じ半荘を同じヘルパーで 2 回通し、2 回目を測る。初めて通るコードでの 1 回だけの確保（作業領域の伸び、enum の switch の表を持つ
    // クラスの初期化など）は 1 回目で済む。
    Walk walk = new Walk();
    play(walk);
    walk.startMeasuring();
    play(walk);
    log.info(walk.summary());
    assertTrue(walk.decisions >= 100_000, "decisions " + walk.decisions);
    for (EngineDecisionKind kind : walk.vocabulary) {
      assertTrue(walk.kinds[kind.ordinal()] > 0, "no " + kind + " decision");
    }
    assertTrue(walk.calls > 0 && walk.kans > 0 && walk.wins > 0, walk.summary());
    assertTrue(walk.maxEvents > 100, "longest public record " + walk.maxEvents);
    assertTrue(walk.allocated < SLACK_BYTES, "allocated bytes " + walk.summary());
  }

  private static void play(Walk walk) {
    for (TableWalk.Policy policy : TableWalk.Policy.values()) {
      new TableWalk(TENHOU_4P, policy).run(7_000 + policy.ordinal(), GAMES, walk);
    }
  }

  private static final class Walk implements TableWalk.Visitor {
    private final RowEncoder encoder = new RowEncoder();
    private final FrontierFeatures frontier = new FrontierFeatures();
    private final ThreadMXBean threads = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    private final EngineDecisionKind[] vocabulary = {
      EngineDecisionKind.TURN, EngineDecisionKind.RESPONSE, EngineDecisionKind.CHANKAN, EngineDecisionKind.POST_CALL_DAHAI
    };
    private final long[] kinds = new long[EngineDecisionKind.values().length];
    private boolean measure;
    private long decisions;
    private long calls;
    private long kans;
    private long wins;
    private int maxEvents;
    private long allocated;

    void startMeasuring() {
      measure = true;
      decisions = 0;
      calls = 0;
      kans = 0;
      wins = 0;
      maxEvents = 0;
      Arrays.fill(kinds, 0L);
    }

    @Override
    public void decision(Engine engine, int slot) {
      long before = measure ? threads.getCurrentThreadAllocatedBytes() : 0L;
      Observation o = encoder.encode(engine, slot);
      frontier.values(o);
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
          default -> {}
        }
      }
    }

    String summary() {
      return String.format(
          Locale.ROOT,
          "TOKENIZER walk: decisions %d kinds %s calls %d kans %d wins %d maxPublicEvents %d allocated %d B",
          decisions,
          Arrays.toString(kinds),
          calls,
          kans,
          wins,
          maxEvents,
          allocated);
    }
  }
}
