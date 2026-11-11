package com.epsilon.classic.rulebot;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 4 席ともルールベースの打ち手にして、シードを決めた半荘を終局まで打つ。同じシードなら同じ半荘になる。ウォームアップの後は、
 * 手を選んでもメモリを確保しない。
 */
public class RuleBotGameTest {

  private static final Logger LOG = LoggerFactory.getLogger(RuleBotGameTest.class);

  private static final Variant VARIANT = Variant.TENHOU_4P;

  /** 1 つの半荘のステップ数の上限（局末と判断の時点をそれぞれ 1 ステップと数える）。 */
  private static final int GAME_STEPS = 100_000;

  private static final int WARM_UP = 2_000;
  private static final int MEASURED = 4_000;

  /**
   * メモリ確保の計測そのものの誤差を見込んだ余裕（バイト）。測るのは 4000 ステップなので、1 ステップごとに 16 バイト以上を 1 回
   * 確保するだけで 64 KB になり、余裕を大きく超える。
   */
  private static final long SLACK_BYTES = 4_096;

  /** どの半荘も持ち点と供託の合計が変わらず、3 つの半荘を通して和了と立直が 1 回以上ある。 */
  @Test
  public void ruleBotsPlaySeededGamesToTheEnd() {
    int rounds = 0;
    int wins = 0;
    int riichi = 0;
    for (long seed = 1; seed <= 3; seed++) {
      BotTable table = new BotTable(new RuleBot(VARIANT), seed);
      table.finishGame(GAME_STEPS);
      TableState last = table.engine().roundEnd().next();
      assertTrue(last.gameOver());
      int total = Variant.DEPOSIT * last.kyotaku();
      int[] scores = new int[VARIANT.seats()];
      for (int seat = 0; seat < scores.length; seat++) {
        scores[seat] = last.score(seat);
        total += scores[seat];
      }
      assertEquals(total, VARIANT.seats() * VARIANT.startPoints(), Arrays.toString(scores));
      LOG.info(
          "Rule bot game {}: {} decisions, {} rounds, {} wins, {} riichi, final scores {}",
          seed,
          table.decisions(),
          table.rounds(),
          table.wins(),
          table.riichi(),
          Arrays.toString(scores));
      rounds += table.rounds();
      wins += table.wins();
      riichi += table.riichi();
    }
    assertTrue(wins > 0 && riichi > 0, wins + " wins and " + riichi + " riichi in " + rounds + " rounds");
  }

  @Test
  public void aSeedPlaysTheSameGameTwice() {
    BotTable first = new BotTable(new RuleBot(VARIANT), 7);
    BotTable second = new BotTable(new RuleBot(VARIANT), 7);
    first.finishGame(GAME_STEPS);
    second.finishGame(GAME_STEPS);
    assertEquals(second.decisions(), first.decisions());
    assertEquals(second.digest(), first.digest());
    for (int seat = 0; seat < VARIANT.seats(); seat++) {
      assertEquals(second.engine().roundEnd().next().score(seat), first.engine().roundEnd().next().score(seat));
    }
  }

  @Test(timeOut = 900_000)
  public void actAllocatesNothingOnceWarm() {
    ThreadMXBean threads = ManagementFactory.getThreadMXBean() instanceof ThreadMXBean bean ? bean : null;
    if (threads == null || !threads.isThreadAllocatedMemorySupported() || !threads.isThreadAllocatedMemoryEnabled()) {
      throw new SkipException("Per-thread allocation counters are not available");
    }
    BotTable table = new BotTable(new RuleBot(VARIANT), 11);
    for (int step = 0; step < WARM_UP; step++) {
      table.step();
    }
    long before = threads.getCurrentThreadAllocatedBytes();
    for (int step = 0; step < MEASURED; step++) {
      table.step();
    }
    long allocated = threads.getCurrentThreadAllocatedBytes() - before;
    assertTrue(
        allocated < SLACK_BYTES,
        allocated + " bytes over " + MEASURED + " steps (" + table.decisions() + " decisions, digest " + table.digest()
            + ")");
  }
}
