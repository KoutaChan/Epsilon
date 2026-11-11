package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * アラートの予算を確かめる。PRISM の 1,000 バージョン分の回数では、予算は設計で見積もった値になる。監視に偶然の判定と偶然の対の差を流すと、
 * 出たアラートの数が予算に合う（基準の低下の予算は幅を掛けない上限なので、実際の数はそれ以下）。
 */
public class AlarmBudgetTest {

  private static final Logger log = LoggerFactory.getLogger(AlarmBudgetTest.class);

  private static final Watch WATCH = new Watch(3, 0.01, new OnWorseSettings.Report(2));

  private Path root;
  private RunDirectory run;

  @BeforeMethod
  public void createRun() throws Exception {
    root = Files.createTempDirectory("agent-alarm-budget");
    run = new RunDirectory(root);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws Exception {
    DurableFiles.delete(root);
  }

  /** 王者との決闘 999 回、ラダー 124 回、基準 121 回、ルールベースの打ち手の卓 1,000 回で、有意水準は全て 0.01。 */
  @Test
  public void aThousandVersionsGiveTheDesignTable() {
    AlarmBudget budget = AlarmBudget.of(WATCH, new AlarmBudget.Exposure(999, 0.01, 124, 0.01, 121, 1000));
    double tail = Normal.upper(3);
    assertEquals(budget.promotions(), 4.995, 1e-12);
    Map<Signal, Double> alarms = budget.alarms();
    assertEquals(alarms.keySet(), EnumSet.allOf(Signal.class));
    assertEquals(alarms.get(Signal.CHAMPION_WORSE), 4.995, 1e-12);
    assertEquals(alarms.get(Signal.WORSE_STREAK), 998 * 0.005 * 0.005, 1e-12);
    assertEquals(alarms.get(Signal.LADDER_WORSE), 0.62, 1e-12);
    assertEquals(alarms.get(Signal.BASELINE_DROP), 121 * tail, 1e-12);
    assertEquals(alarms.get(Signal.ANCHOR_DROP), 1000 * tail, 1e-12);
    assertEquals(alarms.get(Signal.ANCHOR_DROP), 1.35, 0.005);
    assertEquals(alarms.get(Signal.WORSE_STREAK), 0.025, 0.0005);
    assertThrows(UnsupportedOperationException.class, () -> alarms.put(Signal.LADDER_WORSE, 0.0));
  }

  @Test
  public void theStreakBudgetFollowsTheStreakLength() {
    AlarmBudget.Exposure exposure = new AlarmBudget.Exposure(10, 0.1, 0, 0.01, 0, 0);
    assertEquals(of(1, exposure).alarms().get(Signal.WORSE_STREAK), 10 * 0.05, 1e-12);
    assertEquals(of(3, exposure).alarms().get(Signal.WORSE_STREAK), 8 * Math.pow(0.05, 3), 1e-15);
    assertEquals(of(10, exposure).alarms().get(Signal.WORSE_STREAK), Math.pow(0.05, 10), 1e-25);
    assertEquals(of(11, exposure).alarms().get(Signal.WORSE_STREAK), 0.0, 0);
  }

  private static AlarmBudget of(int streak, AlarmBudget.Exposure exposure) {
    return AlarmBudget.of(new Watch(3, 0.01, new OnWorseSettings.Report(streak)), exposure);
  }

  @Test
  public void invalidExposuresAreRejected() {
    assertThrows(IllegalArgumentException.class, () -> new AlarmBudget.Exposure(-1, 0.01, 0, 0.01, 0, 0));
    assertThrows(IllegalArgumentException.class, () -> new AlarmBudget.Exposure(0, 0.01, -1, 0.01, 0, 0));
    assertThrows(IllegalArgumentException.class, () -> new AlarmBudget.Exposure(0, 0.01, 0, 0.01, -1, 0));
    assertThrows(IllegalArgumentException.class, () -> new AlarmBudget.Exposure(0, 0.01, 0, 0.01, 0, -1));
    assertThrows(IllegalArgumentException.class, () -> new AlarmBudget.Exposure(0, 0, 0, 0.01, 0, 0));
    assertThrows(IllegalArgumentException.class, () -> new AlarmBudget.Exposure(0, 1, 0, 0.01, 0, 0));
    assertThrows(IllegalArgumentException.class, () -> new AlarmBudget.Exposure(0, 0.01, 0, Double.NaN, 0, 0));
    assertThrows(NullPointerException.class, () -> new AlarmBudget(0, null));
  }

  /**
   * 判定が独立に確率 p で悪いとき、王者との決闘 200 回の系列を 5,000 本流すと、アラートの数の平均は、両側の有意水準を 2p とした予算から
   * 3 SE 以内に入る（p = 0.1、悪いが続く回数は 2 と 3）。
   */
  @Test
  public void randomVerdictsRaiseTheBudgetedDuelAlarms() {
    int duels = 200;
    int series = 5000;
    double worse = 0.1;
    for (int streak : new int[] {2, 3}) {
      Watch watch = new Watch(3, 0.01, new OnWorseSettings.Report(streak));
      AlarmBudget budget = AlarmBudget.of(watch, new AlarmBudget.Exposure(duels, 2 * worse, 0, 0.01, 0, 0));
      Alarms alarms = new Alarms(watch, run);
      Estimate even = Estimate.of(0, 1, 3);
      WallMean champion = new WallMean();
      WallMean streaks = new WallMean();
      SplittableRandom random = new SplittableRandom(streak);
      for (int s = 0; s < series; s++) {
        int inARow = 0;
        int championCount = 0;
        int streakCount = 0;
        for (int duel = 0; duel < duels; duel++) {
          Verdict verdict = random.nextDouble() < worse ? Verdict.WORSE : Verdict.EVEN;
          inARow = Alarms.streak(inARow, verdict);
          for (Alarm alarm : alarms.championDuel("v00001-champion", duel, verdict, even, inARow, List.of())) {
            if (alarm.signal() == Signal.CHAMPION_WORSE) {
              championCount++;
            } else {
              streakCount++;
            }
          }
        }
        champion.add(championCount);
        streaks.add(streakCount);
      }
      log.info(
          "ALARM-BUDGET streak {}: champion {} ({}) budget {}, streak {} ({}) budget {}",
          streak,
          champion.mean(),
          champion.error(),
          budget.alarms().get(Signal.CHAMPION_WORSE),
          streaks.mean(),
          streaks.error(),
          budget.alarms().get(Signal.WORSE_STREAK));
      assertEquals(champion.mean(), budget.alarms().get(Signal.CHAMPION_WORSE), 3 * champion.error());
      assertEquals(streaks.mean(), budget.alarms().get(Signal.WORSE_STREAK), 3 * streaks.error());
    }
  }

  /**
   * 真の差が 0 の対の差（牌山 256）を、毎回別の王者として 4,000 回流す。しきい値 z = 2 で、ルールベースの打ち手の卓の低下の率は
   * P(Z > 2) から 3 SE 以内に入る。基準の低下は幅 0.01 を掛けるので、その上限以下になる。
   */
  @Test
  public void randomDifferencesRaiseTheBudgetedDrops() throws Exception {
    int comparisons = 4000;
    int walls = 256;
    Watch watch = new Watch(2, 0.01, new OnWorseSettings.Report(2));
    AlarmBudget budget =
        AlarmBudget.of(watch, new AlarmBudget.Exposure(0, 0.01, 0, 0.01, comparisons, comparisons));
    Alarms alarms = new Alarms(watch, run);
    SplittableRandom random = new SplittableRandom(23);
    int anchors = 0;
    int baselines = 0;
    for (int c = 0; c < comparisons; c++) {
      double[] champion = new double[walls];
      double[] current = new double[walls];
      for (int i = 0; i < walls; i++) {
        double luck = 0.3 * random.nextGaussian();
        champion[i] = luck + 0.1 * random.nextGaussian();
        current[i] = luck + 0.1 * random.nextGaussian();
      }
      anchors += alarms.anchor("v00001-rx2", c, c, current, champion).size();
      baselines += alarms.baseline("v00001-reference", c, c, current, champion).size();
    }
    double expected = budget.alarms().get(Signal.ANCHOR_DROP);
    double error = Math.sqrt(expected * (1 - expected / comparisons));
    log.info("ALARM-BUDGET drops: anchor {} baseline {} budget {} ({})", anchors, baselines, expected, error);
    assertEquals(anchors, expected, 3 * error);
    assertTrue(baselines <= anchors, baselines + " > " + anchors);
    assertEquals(budget.alarms().get(Signal.BASELINE_DROP), expected, 1e-9);
  }
}
