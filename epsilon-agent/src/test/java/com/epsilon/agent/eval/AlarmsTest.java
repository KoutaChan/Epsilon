package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.eval.Alarm.Handling;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.RunDirectory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * リグレッションのアラートを確かめる。王者との決闘で「悪い」の判定が出たら、毎回アラートを出す。「悪い」が設定の回数以上続いている間は、
 * 続いたことのアラートも出す。続いたときの扱いは設定の形で決まり、王者へ戻す形は、幅の中で戻した回数が上限に届くと一時停止に替わる。
 * ラダーのアラートは、区間の上端が 0 より小さいときだけ出す。基準の卓の低下は幅と z の両方で、ルールベースの打ち手の卓の低下は z だけで、
 * 同じ牌山の対の差を見て判定する。どちらも同じ卓で同じ王者の間は 1 回だけ出す（書いた行から読む）。
 */
public class AlarmsTest {

  private static final Estimate BAD = Estimate.of(-0.03, 0.005, 3);
  private static final Watch REPORT = new Watch(3, 0.01, new OnWorseSettings.Report(2));
  private static final Watch PAUSE = new Watch(3, 0.01, new OnWorseSettings.Pause(2));

  /** 悪いが 2 回続いたら戻し、10 バージョンの幅の中で 1 回まで戻す。 */
  private static final Watch ROLLBACK = new Watch(3, 0.01, new OnWorseSettings.Rollback(2, 1, 10));

  private Path root;
  private RunDirectory run;

  @BeforeMethod
  public void createRun() throws Exception {
    root = Files.createTempDirectory("agent-alarms");
    run = new RunDirectory(root);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws Exception {
    DurableFiles.delete(root);
  }

  @Test
  public void theStreakCountsWorseDuelsAndAnyOtherVerdictResetsIt() {
    assertEquals(Alarms.streak(0, Verdict.WORSE), 1);
    assertEquals(Alarms.streak(2, Verdict.WORSE), 3);
    for (Verdict reset : new Verdict[] {Verdict.BETTER, Verdict.EVEN, Verdict.UNDECIDED}) {
      assertEquals(Alarms.streak(5, reset), 0, reset.toString());
    }
  }

  @Test
  public void aWorseStreakIsRaisedFromTheSetCountWhileItLasts() {
    Alarms alarms = new Alarms(REPORT, run);
    Verdict[] verdicts = {Verdict.WORSE, Verdict.WORSE, Verdict.WORSE, Verdict.EVEN, Verdict.WORSE};
    List<List<Signal>> expected =
        List.of(
            List.of(Signal.CHAMPION_WORSE),
            List.of(Signal.CHAMPION_WORSE, Signal.WORSE_STREAK),
            List.of(Signal.CHAMPION_WORSE, Signal.WORSE_STREAK),
            List.of(),
            List.of(Signal.CHAMPION_WORSE));
    int streak = 0;
    for (int duel = 0; duel < verdicts.length; duel++) {
      int version = 10 + duel;
      streak = Alarms.streak(streak, verdicts[duel]);
      List<Alarm> raised = alarms.championDuel(duel(version), version, verdicts[duel], BAD, streak, List.of());
      assertEquals(raised.stream().map(Alarm::signal).toList(), expected.get(duel), "duel " + duel);
      for (Alarm alarm : raised) {
        assertEquals(alarm, Alarm.of(duel(version), version, alarm.signal(), BAD, Handling.REPORT, null));
      }
    }
  }

  @Test
  public void aPauseStreakAsksForAPause() {
    Alarms alarms = new Alarms(PAUSE, run);
    assertEquals(
        handlings(alarms.championDuel(duel(5), 5, Verdict.WORSE, BAD, 1, List.of())), List.of(Handling.REPORT));
    assertEquals(
        handlings(alarms.championDuel(duel(6), 6, Verdict.WORSE, BAD, 2, List.of())),
        List.of(Handling.REPORT, Handling.PAUSE));
  }

  /** 戻しを数えるのは、バージョン − 幅より後で、そのバージョンまでの戻し。 */
  @Test
  public void aRollbackTurnsIntoAPauseOnceTheRollbacksWithinTheSpanReachTheLimit() {
    Alarms alarms = new Alarms(ROLLBACK, run);
    assertEquals(streakHandling(alarms, 5, List.of()), Handling.ROLLBACK);
    assertEquals(streakHandling(alarms, 5, List.of(4)), Handling.PAUSE);
    assertEquals(streakHandling(alarms, 13, List.of(4)), Handling.PAUSE);
    assertEquals(streakHandling(alarms, 14, List.of(4)), Handling.ROLLBACK, "4 is not after 14 - 10");
    assertEquals(streakHandling(alarms, 13, List.of(20)), Handling.ROLLBACK, "a later rollback");
    Alarms twice = new Alarms(new Watch(3, 0.01, new OnWorseSettings.Rollback(2, 2, 10)), run);
    assertEquals(streakHandling(twice, 13, List.of(4)), Handling.ROLLBACK);
    assertEquals(streakHandling(twice, 13, List.of(4, 9)), Handling.PAUSE);
    // 続いた回数が設定に届かなければ、戻しも一時停止もしない
    assertEquals(
        handlings(alarms.championDuel(duel(5), 5, Verdict.WORSE, BAD, 1, List.of(4))), List.of(Handling.REPORT));
  }

  /** evaluate は悪いが続いた回数を数えないので、1 回で届く設定でも王者に悪いだけを出す。 */
  @Test
  public void withoutAStreakOnlyTheChampionAlarmIsRaised() {
    Alarms alarms = new Alarms(new Watch(3, 0.01, new OnWorseSettings.Pause(1)), run);
    assertEquals(
        alarms.championDuel(duel(7), 7, Verdict.WORSE, BAD),
        List.of(Alarm.of(duel(7), 7, Signal.CHAMPION_WORSE, BAD, Handling.REPORT, null)));
    for (Verdict other : new Verdict[] {Verdict.BETTER, Verdict.EVEN, Verdict.UNDECIDED}) {
      assertTrue(alarms.championDuel(duel(7), 7, other, BAD).isEmpty(), other.toString());
    }
    assertEquals(
        handlings(alarms.championDuel(duel(7), 7, Verdict.WORSE, BAD, 1, List.of())),
        List.of(Handling.REPORT, Handling.PAUSE));
  }

  @Test
  public void aLadderAlarmNeedsTheWholeIntervalBelowZero() {
    Alarms alarms = new Alarms(REPORT, run);
    Estimate below = new Estimate(-0.02, 0.005, -0.033, -0.0001);
    String job = "v00005-ladder8";
    assertEquals(
        alarms.ladder(job, 5, below),
        List.of(new Alarm(job + ":LADDER_WORSE", job, 5, Signal.LADDER_WORSE, below, Handling.REPORT, null)));
    assertTrue(alarms.ladder(job, 5, new Estimate(-0.013, 0.005, -0.026, 0)).isEmpty());
    assertTrue(alarms.ladder(job, 5, new Estimate(0, 0.005, -0.013, 0.013)).isEmpty());
  }

  /** 対の差が −0.03 で標準誤差が小さければ低下のアラートを出す。平均は対の差、区間は平均 ± 3 × 標準誤差。牌山の運は対の差で消える。 */
  @Test
  public void aBaselineDropIsJudgedOnThePairedDifference() throws Exception {
    double[] champion = luck(100);
    Alarm alarm =
        new Alarms(REPORT, run).baseline("v00020-reference", 20, 9, shifted(champion, -0.03, 0.01), champion).getFirst();
    double error = 0.01 / Math.sqrt(99);
    assertEquals(alarm.alarm(), "v00020-reference:BASELINE_DROP");
    assertEquals(alarm.version(), 20);
    assertEquals(alarm.signal(), Signal.BASELINE_DROP);
    assertEquals(alarm.handling(), Handling.REPORT);
    assertEquals(alarm.champion(), Integer.valueOf(9));
    assertEquals(alarm.estimate().mean(), -0.03, 1e-12);
    assertEquals(alarm.estimate().error(), error, 1e-12);
    assertEquals(alarm.estimate().low(), -0.03 - 3 * error, 1e-12);
    assertEquals(alarm.estimate().high(), -0.03 + 3 * error, 1e-12);
  }

  @Test
  public void aBaselineDropNeedsBothTheMarginAndTheLine() throws Exception {
    double[] champion = luck(100);
    Alarms alarms = new Alarms(REPORT, run);
    String job = "v00020-reference";
    assertTrue(alarms.baseline(job, 20, 9, shifted(champion, -0.005, 0.0001), champion).isEmpty(), "inside the margin");
    assertTrue(alarms.baseline(job, 20, 9, shifted(champion, -0.03, 0.5), champion).isEmpty(), "below the line");
    assertTrue(alarms.baseline(job, 20, 9, shifted(champion, 0.03, 0.0001), champion).isEmpty(), "better");
    assertEquals(alarms.baseline(job, 20, 9, shifted(champion, -0.03, 0.01), champion).size(), 1);
  }

  /** ルールベースの打ち手の卓には幅が無く、幅より小さい低下でも z がしきい値を超えれば出る。 */
  @Test
  public void anAnchorDropIgnoresTheMargin() throws Exception {
    double[] champion = luck(100);
    Alarms alarms = new Alarms(REPORT, run);
    Alarm alarm = alarms.anchor("v00030-rx2", 30, 9, shifted(champion, -0.005, 0.0001), champion).getFirst();
    assertEquals(alarm.signal(), Signal.ANCHOR_DROP);
    assertEquals(alarm.estimate().mean(), -0.005, 1e-12);
    assertTrue(alarms.anchor("v00030-rx2", 30, 9, shifted(champion, 0.005, 0.0001), champion).isEmpty());
  }

  /** 長さの違う卓は、短い方の長さまでを対にする。 */
  @Test
  public void aDropUsesTheCommonWalls() throws Exception {
    double[] champion = luck(100);
    double[] longer = Arrays.copyOf(shifted(champion, -0.03, 0.01), 150);
    Arrays.fill(longer, 100, 150, 5.0);
    Alarm alarm = new Alarms(REPORT, run).baseline("v00020-reference", 20, 9, longer, champion).getFirst();
    assertEquals(alarm.estimate().mean(), -0.03, 1e-12);
  }

  /**
   * 書いた低下は、同じ卓で同じ王者の間はもう出さない。どの王者で出したかは書いた行から読むので、作り直した Alarms でも同じに決まる。
   * 別の卓、別の印、替わった王者では改めて出る。
   */
  @Test
  public void aWrittenDropIsNotRaisedAgainForTheSameTableAndChampion() throws Exception {
    double[] champion = luck(100);
    double[] current = shifted(champion, -0.03, 0.01);
    Alarms alarms = new Alarms(REPORT, run);
    List<Alarm> first = alarms.baseline("v00020-reference", 20, 9, current, champion);
    assertEquals(first.size(), 1);
    assertEquals(alarms.baseline("v00021-reference", 21, 9, current, champion).size(), 1, "not written yet");
    alarms.write(first);

    Alarms restarted = new Alarms(REPORT, run);
    assertTrue(restarted.baseline("v00021-reference", 21, 9, current, champion).isEmpty(), "the same champion");
    assertEquals(restarted.baseline("v00021-second", 21, 9, current, champion).size(), 1, "another table");
    assertEquals(restarted.anchor("v00021-reference", 21, 9, current, champion).size(), 1, "another signal");
    assertEquals(restarted.baseline("v00023-reference", 23, 12, current, champion).size(), 1, "a new champion");
  }

  /** 名前ごとに 1 度だけ書き、読むと書いた順に戻る。低下の行だけが王者を持つ。 */
  @Test
  public void eachAlarmIsWrittenOnceAndReadBackInOrder() throws Exception {
    Alarms alarms = new Alarms(PAUSE, run);
    assertEquals(alarms.read(), List.of());
    List<Alarm> duel = alarms.championDuel(duel(4), 4, Verdict.WORSE, BAD, 2, List.of());
    double[] champion = luck(100);
    List<Alarm> drop = alarms.anchor("v00004-rx2", 4, 3, shifted(champion, -0.03, 0.01), champion);
    alarms.write(duel);
    alarms.write(drop);
    alarms.write(duel);
    List<Alarm> written = new ArrayList<>(duel);
    written.addAll(drop);
    assertEquals(alarms.read(), written);
    List<String> lines = DurableFiles.lines(run.report(Alarm.FILE));
    assertEquals(lines.size(), 3);
    assertFalse(lines.get(1).contains("\"champion\""), lines.get(1));
    assertTrue(lines.get(2).endsWith(",\"handling\":\"REPORT\",\"champion\":3}"), lines.get(2));
  }

  @Test
  public void invalidInputsAreRejected() {
    OnWorseSettings report = new OnWorseSettings.Report(2);
    assertThrows(IllegalArgumentException.class, () -> new Watch(0, 0.01, report));
    assertThrows(IllegalArgumentException.class, () -> new Watch(Double.NaN, 0.01, report));
    assertThrows(IllegalArgumentException.class, () -> new Watch(Double.POSITIVE_INFINITY, 0.01, report));
    assertThrows(IllegalArgumentException.class, () -> new Watch(3, -0.01, report));
    assertThrows(IllegalArgumentException.class, () -> new Watch(3, Double.NaN, report));
    assertThrows(IllegalArgumentException.class, () -> new Watch(3, 0.01, new OnWorseSettings.Report(0)));
    assertThrows(NullPointerException.class, () -> new Watch(3, 0.01, null));
    assertThrows(NullPointerException.class, () -> new Alarms(null, run));
    assertThrows(IllegalArgumentException.class, () -> Alarms.streak(-1, Verdict.WORSE));
    Alarms alarms = new Alarms(REPORT, run);
    assertThrows(
        IllegalArgumentException.class, () -> alarms.championDuel(duel(1), 1, Verdict.WORSE, BAD, 0, List.of()));
    assertThrows(
        IllegalArgumentException.class, () -> alarms.championDuel(duel(1), 1, Verdict.EVEN, BAD, 1, List.of()));
    double[] none = {};
    assertThrows(IllegalArgumentException.class, () -> alarms.baseline("v00001-reference", 1, 0, none, none));
    assertThrows(IllegalArgumentException.class, () -> alarms.anchor("rx2", 1, 0, luck(10), luck(10)));
  }

  private static String duel(int version) {
    return RunDirectory.jobName(version, "champion");
  }

  private static List<Handling> handlings(List<Alarm> alarms) {
    return alarms.stream().map(Alarm::handling).toList();
  }

  /** 悪いが 2 回続いた決闘の、悪いが続くの扱い。 */
  private static Handling streakHandling(Alarms alarms, int version, List<Integer> rollbacks) {
    List<Alarm> raised = alarms.championDuel(duel(version), version, Verdict.WORSE, BAD, 2, rollbacks);
    assertEquals(raised.size(), 2, raised.toString());
    assertEquals(raised.getLast().signal(), Signal.WORSE_STREAK);
    return raised.getLast().handling();
  }

  /** @return 牌山ごとに大きく揺れる値。対の差では消える */
  private static double[] luck(int walls) {
    double[] values = new double[walls];
    for (int i = 0; i < walls; i++) {
      values[i] = 0.3 * (i % 7) - 0.9;
    }
    return values;
  }

  /** @return 王者の値に {@code shift} を足し、さらに ±{@code spread} を交互に足した値 */
  private static double[] shifted(double[] champion, double shift, double spread) {
    double[] values = new double[champion.length];
    for (int i = 0; i < champion.length; i++) {
      values[i] = champion[i] + shift + (i % 2 == 0 ? spread : -spread);
    }
    return values;
  }
}
