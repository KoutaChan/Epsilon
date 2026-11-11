package com.epsilon.prism3p.input;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;
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
 * ランダムに打つ三人打ちの自己対局（3 つの打ち方）で、全ての判断の O を {@link Tokenizer} に通し、ライターの確かめ（キーのコード・
 * 手のインデックス・語彙）を通ることを確かめる。ふだんの組は少ない半荘の数で回し、10 万判断の規模とメモリを確保しないことは slow の組で見る。
 */
public class TokenizerWalkTest {
  private static final Logger log = LoggerFactory.getLogger(TokenizerWalkTest.class);

  private static final int SMALL_GAMES = 8;
  private static final int LARGE_GAMES = 100;

  /** 確保の数え方そのものの揺れを許す幅（バイト）。測る判断は 10^5 以上なので、1 判断に 1 つでも確保すれば MB に届く。 */
  private static final long SLACK_BYTES = 4_096;

  @Test(timeOut = 600_000)
  public void everyDecisionPassesTheWriterChecks() {
    Walk walk = new Walk();
    play(walk, SMALL_GAMES);
    log.info(walk.summary());
    assertTrue(walk.decisions >= 3_000, "decisions " + walk.decisions);
    assertTrue(walk.nukis > 0, walk.summary());
    assertTrue(walk.kinds[EngineDecisionKind.TURN.ordinal()] > 0, walk.summary());
    assertTrue(walk.kinds[EngineDecisionKind.RESPONSE.ordinal()] > 0, walk.summary());
    assertTrue(walk.kinds[EngineDecisionKind.POST_CALL_DAHAI.ordinal()] > 0, walk.summary());
    assertTrue(walk.calls > 0 && walk.wins > 0, walk.summary());
  }

  @Test(groups = "slow", timeOut = 1_800_000)
  public void everyDecisionPassesTheWriterChecksWithoutAllocating() {
    // 同じ半荘を同じヘルパーで 2 回通し、2 回目を測る。初めて通るコードでの 1 回だけの確保（作業領域の伸び、enum の switch の表のクラスの初期化など）は 1 回目に済む。
    Walk walk = new Walk();
    play(walk, LARGE_GAMES);
    walk.startMeasuring();
    play(walk, LARGE_GAMES);
    log.info(walk.summary());
    assertTrue(walk.decisions >= 100_000, "decisions " + walk.decisions);
    assertTrue(walk.calls > 0 && walk.kans > 0 && walk.wins > 0 && walk.nukis > 0, walk.summary());
    assertTrue(walk.maxEvents > 60, "longest public record " + walk.maxEvents);
    assertTrue(walk.allocated < SLACK_BYTES, "allocated bytes " + walk.summary());
  }

  private static void play(Walk walk, int games) {
    for (TableWalk.Policy policy : TableWalk.Policy.values()) {
      new TableWalk(TENHOU_3P, policy).run(7_100 + policy.ordinal(), games, walk);
    }
  }

  private static final class Walk implements TableWalk.Visitor {
    private final RowEncoder encoder = new RowEncoder();
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
      if (measure) {
        allocated += threads.getCurrentThreadAllocatedBytes() - before;
      }
      decisions++;
      kinds[o.decisionKind().ordinal()]++;
      maxEvents = Math.max(maxEvents, o.publicEvents());
      for (int action = 0; action < o.legalActions(); action++) {
        switch (o.action(action).type()) {
          case PON -> calls++;
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
          "TOKENIZER 3p walk: decisions %d kinds %s pons %d kans %d wins %d nukis %d maxPublicEvents %d allocated %d B",
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
