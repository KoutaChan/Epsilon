package com.epsilon.prism.eval;

import com.epsilon.agent.app.App;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.eval.Arena;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Placements;
import com.epsilon.agent.eval.PlayStats;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.Seating;
import com.epsilon.agent.eval.Split;
import com.epsilon.agent.eval.StopRule;
import com.epsilon.agent.eval.WallSet;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.app.PrismApp;
import com.epsilon.prism.core.TwinDiscard;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 2 つの評価のジョブを同じ牌山で対にして比べる処理（{@link JobComparison}）と、report の --compare。答えの分かる 3 人の 1v2 の games.tsv を
 * 書き、次を確かめる。
 *
 * <ul>
 *   <li>牌山ごとの δ と υ の差（2 つ目 − 1 つ目）の平均と対の標準誤差、それぞれのジョブの平均と標準誤差が手計算と合う。
 *   <li>片方にしか無い牌山は対にせず、数だけを残す。回っている途中のジョブは、job.json に書いた済んだ牌山までを読む。
 *   <li>候補のほかの計画（ルール・相手・牌山の種・最初の牌山・席の方式と方式・ツモの順・着順の効用）が違う組は、違いごとの英語の文で
 *       拒む。相手は、系譜のバージョンなら指定（run のスナップショットのパス）とバージョンで、系譜の外のチェックポイントなら SHA-256 で
 *       比べる。候補・牌山の数・止め方・判定に使う量・jar だけが違う組と、系譜の外の相手のチェックポイントの場所だけが違う組は対にする。
 *   <li>--other で別の run のジョブと比べる。別の系譜の同じ番号のバージョンは中身の違う相手なので拒み、同じ系譜のバージョンを相手にした
 *       ジョブとは対にする。同じジョブどうし、無いジョブ、まだ計画の無いジョブ、打たずに初めの王者にしたジョブは拒む。
 * </ul>
 */
public class JobComparisonTest {

  private static final Variant THREE = Variant.TENHOU_3P;
  private static final String TABLE = "x1n2";
  private static final int WALLS = 12;

  /** 牌山 1 つの半荘の数（1v2 の全ての割り当て）。 */
  private static final int PER_WALL = 3;

  /** 着順ごとの点（1 位・2 位・3 位）。 */
  private static final int[] POINTS = {50_000, 35_000, 20_000};

  private static final String OPPONENT_SHA = "a".repeat(64);
  private static final Instant CREATED = Instant.parse("2026-11-10T00:00:00Z");

  private Path root;
  private RunDirectory run;

  /** ジョブの定義（比べは読まない）。 */
  record Definition(String table) {}

  /** 候補の着順（0 始まり）。 */
  @FunctionalInterface
  interface Places {

    /**
     * @param wall 牌山の添字（0 始まり）
     * @param game 牌山の中の半荘の番号
     */
    int place(int wall, int game);
  }

  @BeforeMethod
  public void createRoot() throws IOException {
    root = Files.createTempDirectory("job-comparison");
    run = new RunDirectory(root.resolve("a"));
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRoot() throws IOException {
    DurableFiles.delete(root);
  }

  /** 牌山と半荘ごとに引いた着順。 */
  private static Places drawn(long salt) {
    return (wall, game) -> (int) Math.floorMod(SeedMixer.mix64(salt * 1_000 + wall * PER_WALL + game), 3L);
  }

  /** run の系譜のバージョン。 */
  private static DuelSetup.Entrant version(RunDirectory run, int version) {
    return new DuelSetup.Entrant("prism:" + run.snapshot(version), EvaluationJobs.label(version), version, null);
  }

  private static DuelSetup.Entrant candidate(RunDirectory run, int version) {
    return version(run, version);
  }

  /** run の系譜のバージョン 1 の相手。別の run の系譜のバージョン 1 は、番号が同じでも別の相手。 */
  private static DuelSetup.Entrant opponent(RunDirectory run) {
    return version(run, 1);
  }

  /** 系譜の外のチェックポイントの相手。 */
  private static DuelSetup.Entrant outside(String path, String sha256) {
    return new DuelSetup.Entrant("nano:" + path, "nano:v760", null, sha256);
  }

  private static DuelSetup duel(
      DuelSetup.Entrant candidate,
      DuelSetup.Entrant opponent,
      Variant variant,
      Seating seating,
      List<String> splits,
      DrawOrder draws,
      long seed,
      long base,
      int walls,
      StopRule stop,
      Score score,
      RankUtility utility) {
    List<Split> parsed = splits.stream().map(split -> Split.parse(split, 2, variant)).toList();
    return DuelSetup.of(
        variant, candidate, opponent, seating, parsed, draws, WallSet.FIXED, seed, base, walls, stop, score, utility);
  }

  /** 基準の計画: 3 人の 1v2 の全ての割り当て、席ごとのツモ列、評価の牌山、全ての牌山を打つ。 */
  private static DuelSetup duel(DuelSetup.Entrant candidate, DuelSetup.Entrant opponent) {
    return duel(
        candidate,
        opponent,
        THREE,
        Seating.EVERY_PLACEMENT,
        List.of("1v2"),
        DrawOrder.SEAT_STREAMS,
        Arena.EVALUATION_SEED,
        Arena.WALL_BASE,
        WALLS,
        new StopRule.Fixed(0.01),
        Score.RANK,
        RankUtility.TENHOU);
  }

  /** 1v2 の卓の games.tsv の行。候補の席に着順の点を、ほかの 2 席に残りの点を席の順に置く。 */
  private static List<String> lines(DuelSetup duel, int walls, Places places) {
    Placements layout = duel.layout();
    int[][] slotTable = layout.slotTable();
    List<String> lines = new ArrayList<>();
    for (int index = 0; index < walls; index++) {
      long wall = duel.wallBase() + index;
      for (int game = 0; game < layout.gamesPerWall(); game++) {
        int placement = layout.placement(wall, game);
        int[] slots = slotTable[placement];
        int place = places.place(index, game);
        StringBuilder line = new StringBuilder().append(wall).append('\t').append(placement).append('\t');
        for (int slot : slots) {
          line.append(slot);
        }
        int next = 0;
        for (int slot : slots) {
          if (slot == 0) {
            line.append('\t').append(POINTS[place]);
          } else {
            next += next == place ? 1 : 0;
            line.append('\t').append(POINTS[next++]);
          }
        }
        line.append('\t').append((double) (place + 1));
        line.append('\t').append(duel.utility().value(duel.variant(), place));
        new PlayStats(2).appendTo(line);
        lines.add(line.toString());
      }
    }
    return lines;
  }

  /**
   * ジョブの記録と games.tsv を書く。
   *
   * @param recorded job.json に書く済んだ牌山の数
   * @param lines games.tsv の行（済んだ牌山より後ろの行も書ける）
   */
  private static void job(
      RunDirectory run, int version, EvaluationPlan plan, JobFile.State state, int recorded, List<String> lines)
      throws IOException {
    String name = RunDirectory.jobName(version, TABLE);
    Path home = run.job(name);
    Files.createDirectories(home);
    StringBuilder text = new StringBuilder();
    long bytes = 0;
    for (int index = 0; index < lines.size(); index++) {
      String line = lines.get(index) + "\n";
      text.append(line);
      if (index < recorded * PER_WALL) {
        bytes += line.getBytes(StandardCharsets.UTF_8).length;
      }
    }
    Files.writeString(home.resolve(JobOutputs.GAMES), text.toString(), StandardCharsets.UTF_8);
    DurableFiles.write(
        home.resolve(Jobs.JOB),
        new JobFile(
            name, version, TABLE, 0, 1, RecordFile.tree(new Definition(TABLE)), RecordFile.tree(plan), state, 1, 0,
            recorded, bytes, 0, CREATED, null));
  }

  /** 同値な打牌を既定の切り方（ツモ切り）で切る計画。 */
  private static EvaluationPlan plan(DuelSetup.Entrant candidate, DuelSetup duel, String jar) {
    return new EvaluationPlan(candidate, duel, jar, TwinDiscard.TSUMOGIRI);
  }

  /** 基準の計画で全ての牌山を打った、済んだジョブを書く。 */
  private static void done(RunDirectory run, int version, Places places, String jar) throws IOException {
    done(run, run, version, places, jar);
  }

  /**
   * 基準の計画で全ての牌山を打った、済んだジョブを書く。
   *
   * @param home ジョブを書く run
   * @param lineage 候補と相手のバージョンを持つ run（evaluate の --out のジョブは、別の run の系譜のバージョンを測る）
   */
  private static void done(RunDirectory home, RunDirectory lineage, int version, Places places, String jar)
      throws IOException {
    DuelSetup duel = duel(candidate(lineage, version), opponent(lineage));
    job(home, version, plan(duel.candidate(), duel, jar), JobFile.State.DONE, WALLS,
        lines(duel, WALLS, places));
  }

  /** 相手の違いの句に出す、run の系譜のバージョンの書き方（名前と指定）。 */
  private static String named(RunDirectory run, int version) {
    return EvaluationJobs.label(version) + " prism:" + run.snapshot(version);
  }

  /** 牌山ごとの δ（3 半荘の候補の着順の平均 − 2）。 */
  private static double[] rank(Places places, int walls) {
    double[] values = new double[walls];
    for (int wall = 0; wall < walls; wall++) {
      int sum = 0;
      for (int game = 0; game < PER_WALL; game++) {
        sum += places.place(wall, game) + 1;
      }
      values[wall] = sum / 3.0 - 2;
    }
    return values;
  }

  /** 牌山ごとの υ（3 半荘の候補の効用の平均 − 着順ごとの効用の平均）。 */
  private static double[] utility(Places places, int walls) {
    double average = 0;
    for (int place = 0; place < 3; place++) {
      average += RankUtility.TENHOU.value(THREE, place) / 3;
    }
    double[] values = new double[walls];
    for (int wall = 0; wall < walls; wall++) {
      double sum = 0;
      for (int game = 0; game < PER_WALL; game++) {
        sum += RankUtility.TENHOU.value(THREE, places.place(wall, game));
      }
      values[wall] = sum / PER_WALL - average;
    }
    return values;
  }

  /** 牌山ごとの a − b（短い方の長さまで）。 */
  private static double[] minus(double[] a, double[] b) {
    double[] difference = new double[Math.min(a.length, b.length)];
    for (int index = 0; index < difference.length; index++) {
      difference[index] = a[index] - b[index];
    }
    return difference;
  }

  private static double mean(double[] values) {
    double sum = 0;
    for (double value : values) {
      sum += value;
    }
    return sum / values.length;
  }

  /** 平均の標準誤差の手計算（偏差の平方和を牌山の数 − 1 で割った分散から）。 */
  private static double error(double[] values) {
    double mean = mean(values);
    double squares = 0;
    for (double value : values) {
      squares += (value - mean) * (value - mean);
    }
    return Math.sqrt(squares / (values.length - 1) / values.length);
  }

  private static void assertValue(JobComparison.Value value, double[] values, String label) {
    Assert.assertEquals(value.mean(), mean(values), 1e-12, label + " mean");
    Assert.assertEquals(value.error(), error(values), 1e-12, label + " error");
  }

  @Test
  public void theDifferenceAndThePairedErrorMatchAHandCalculation() throws Exception {
    Places shared = drawn(2);
    // 4 牌山ごとに、最初の半荘で 1 つ目は 3 位、2 つ目は 1 位。ほかの半荘は同じ着順なので、差は 12 牌山のうち 3 牌山で δ が −2/3
    Places first = (wall, game) -> wall % 4 == 0 && game == 0 ? 2 : shared.place(wall, game);
    Places second = (wall, game) -> wall % 4 == 0 && game == 0 ? 0 : shared.place(wall, game);
    done(run, 2, first, "jar-a");
    done(run, 3, second, "jar-a");

    JobComparison comparison = JobComparison.of(TABLE, run, 2, run, 3);
    Assert.assertEquals(comparison.walls(), WALLS);
    // 差の平均 −3 × (2/3) / 12 = −1/6、偏差の平方和 3 × (1/2)² + 9 × (1/6)² = 1 なので標準誤差は √(1 / 11 / 12)。効用は −2 倍
    Assert.assertEquals(comparison.rank().mean(), -1 / 6.0, 1e-12);
    Assert.assertEquals(comparison.rank().error(), Math.sqrt(1 / 132.0), 1e-12);
    Assert.assertEquals(comparison.utility().mean(), 1 / 3.0, 1e-12);
    Assert.assertEquals(comparison.utility().error(), 2 * Math.sqrt(1 / 132.0), 1e-12);
    double[] firstRank = rank(first, WALLS);
    double[] secondRank = rank(second, WALLS);
    assertValue(comparison.rank(), minus(secondRank, firstRank), "rank difference");
    assertValue(comparison.utility(), minus(utility(second, WALLS), utility(first, WALLS)), "utility difference");
    assertValue(comparison.first().rank(), firstRank, "first rank");
    assertValue(comparison.second().rank(), secondRank, "second rank");
    assertValue(comparison.first().utility(), utility(first, WALLS), "first utility");
    assertValue(comparison.second().utility(), utility(second, WALLS), "second utility");
    // 同じ牌山の運が両方に同じ向きに効くので、対の標準誤差は別々の標準誤差を合わせたものより小さい
    Assert.assertTrue(
        comparison.rank().error() < Math.hypot(error(firstRank), error(secondRank)),
        comparison.rank() + " against " + error(firstRank) + ", " + error(secondRank));
    Assert.assertEquals(comparison.first().job().name(), "v00002-x1n2");
    Assert.assertEquals(comparison.second().plan().candidate().label(), "v00003");
    Assert.assertEquals(comparison.second().run(), run);

    // 向きを入れ替えると差の符号が反り、標準誤差は同じ
    JobComparison reversed = JobComparison.of(TABLE, run, 3, run, 2);
    Assert.assertEquals(reversed.rank().mean(), -comparison.rank().mean(), 1e-12);
    Assert.assertEquals(reversed.rank().error(), comparison.rank().error(), 1e-12);
  }

  @Test
  public void wallsOfOnlyOneJobAreLeftOutOfThePairs() throws Exception {
    Places first = drawn(5);
    Places second = drawn(7);
    done(run, 2, first, "jar-a");
    // 回っている途中のジョブ: job.json は 6 牌山。games.tsv には、その後ろに書いた 2 牌山と書きかけの行もある
    DuelSetup running = duel(candidate(run, 3), opponent(run));
    job(run, 3, plan(running.candidate(), running, "jar-a"), JobFile.State.RUNNING, 6,
        lines(running, 8, second));
    Files.writeString(
        run.job("v00003-x1n2").resolve(JobOutputs.GAMES),
        (Arena.WALL_BASE + 8) + "\t0\t011\t50000",
        StandardCharsets.UTF_8,
        StandardOpenOption.APPEND);

    JobComparison comparison = JobComparison.of(TABLE, run, 2, run, 3);
    Assert.assertEquals(comparison.walls(), 6);
    assertValue(comparison.rank(), minus(rank(second, 6), rank(first, 6)), "rank difference");
    assertValue(comparison.utility(), minus(utility(second, 6), utility(first, 6)), "utility difference");
    assertValue(comparison.first().rank(), rank(first, 6), "first rank on the paired walls");
    Assert.assertEquals(comparison.first().job().walls() - comparison.walls(), WALLS - 6);
    Assert.assertEquals(comparison.second().job().walls() - comparison.walls(), 0);
    Assert.assertEquals(comparison.second().job().state(), JobFile.State.RUNNING);
    JobComparison reversed = JobComparison.of(TABLE, run, 3, run, 2);
    Assert.assertEquals(reversed.walls(), 6);
    Assert.assertEquals(reversed.rank().mean(), -comparison.rank().mean(), 1e-12);

    // 済んだ牌山が 1 つだけのジョブとは、対の標準誤差が出せない
    DuelSetup single = duel(candidate(run, 4), opponent(run));
    job(run, 4, plan(single.candidate(), single, "jar-a"), JobFile.State.RUNNING, 1,
        lines(single, 1, second));
    SettingsException few =
        Assert.expectThrows(SettingsException.class, () -> JobComparison.of(TABLE, run, 2, run, 4));
    Assert.assertEquals(
        few.problems(),
        List.of(
            "Cannot pair v00002-x1n2 with v00004-x1n2: a paired standard error needs at least two walls finished in"
                + " both jobs, not 1"));

    // job.json に書いた牌山の行を失った games.tsv は読まない
    DuelSetup lost = duel(candidate(run, 5), opponent(run));
    job(run, 5, plan(lost.candidate(), lost, "jar-a"), JobFile.State.RUNNING, 6, lines(lost, 5, second));
    IOException shorter = Assert.expectThrows(IOException.class, () -> JobComparison.of(TABLE, run, 2, run, 5));
    Assert.assertTrue(
        shorter.getMessage().contains("holds 15 games, but v00005-x1n2 recorded 6 walls of 3 games"),
        shorter.getMessage());
  }

  @Test
  public void plansThatDifferBesidesTheCandidateAreRefused() throws Exception {
    Places first = drawn(9);
    done(run, 2, first, "jar-a");
    DuelSetup.Entrant candidate = candidate(run, 3);
    DuelSetup.Entrant opponent = opponent(run);
    long seed = Arena.EVALUATION_SEED;
    long base = Arena.WALL_BASE;
    StopRule fixed = new StopRule.Fixed(0.01);
    List<String> one = List.of("1v2");

    // 拒む計画と、違いの句
    record Apart(DuelSetup duel, List<String> because) {}
    List<Apart> apart =
        List.of(
            new Apart(
                duel(candidate, opponent, Variant.TENHOU_4P, Seating.EVERY_PLACEMENT, List.of("1v3"),
                    DrawOrder.SEAT_STREAMS, seed, base, WALLS, fixed, Score.RANK, RankUtility.TENHOU),
                List.of(
                    "the variant differs (TENHOU_3P and TENHOU_4P)",
                    "the seating differs (EVERY_PLACEMENT [1v2] and EVERY_PLACEMENT [1v3])")),
            new Apart(
                duel(candidate, version(run, 2), THREE, Seating.EVERY_PLACEMENT, one, DrawOrder.SEAT_STREAMS, seed,
                    base, WALLS, fixed, Score.RANK, RankUtility.TENHOU),
                List.of("the opponent differs (" + named(run, 1) + " and " + named(run, 2) + ")")),
            new Apart(
                duel(candidate, outside("/elsewhere/00001", OPPONENT_SHA), THREE, Seating.EVERY_PLACEMENT, one,
                    DrawOrder.SEAT_STREAMS, seed, base, WALLS, fixed, Score.RANK, RankUtility.TENHOU),
                List.of("the opponent differs (" + named(run, 1) + " and nano:v760 sha256 " + OPPONENT_SHA + ")")),
            new Apart(
                duel(candidate, new DuelSetup.Entrant(EvaluationTable.HEURISTIC, "heuristic", null, null), THREE,
                    Seating.EVERY_PLACEMENT, one, DrawOrder.SEAT_STREAMS, seed, base, WALLS, fixed, Score.RANK,
                    RankUtility.TENHOU),
                List.of("the opponent differs (" + named(run, 1) + " and heuristic)")),
            new Apart(
                duel(candidate, opponent, THREE, Seating.EVERY_PLACEMENT, one, DrawOrder.SEAT_STREAMS, seed + 1,
                    base, WALLS, fixed, Score.RANK, RankUtility.TENHOU),
                List.of("the wall seed differs (" + seed + " and " + (seed + 1) + ")")),
            new Apart(
                duel(candidate, opponent, THREE, Seating.EVERY_PLACEMENT, one, DrawOrder.SEAT_STREAMS, seed,
                    base + WALLS, WALLS, fixed, Score.RANK, RankUtility.TENHOU),
                List.of("the first wall differs (" + base + " and " + (base + WALLS) + ")")),
            new Apart(
                duel(candidate, opponent, THREE, Seating.BY_GAME, one, DrawOrder.SEAT_STREAMS, seed, base, WALLS,
                    fixed, Score.RANK, RankUtility.TENHOU),
                List.of("the seating differs (EVERY_PLACEMENT [1v2] and BY_GAME [1v2])")),
            new Apart(
                duel(candidate, opponent, THREE, Seating.EVERY_PLACEMENT, List.of("1v2", "2v1"),
                    DrawOrder.SEAT_STREAMS, seed, base, WALLS, fixed, Score.RANK, RankUtility.TENHOU),
                List.of("the seating differs (EVERY_PLACEMENT [1v2] and EVERY_PLACEMENT [1v2, 2v1])")),
            new Apart(
                duel(candidate, opponent, THREE, Seating.EVERY_PLACEMENT, one, DrawOrder.WALL_ORDER, seed, base,
                    WALLS, fixed, Score.RANK, RankUtility.TENHOU),
                List.of("the draw order differs (SEAT_STREAMS and WALL_ORDER)")),
            new Apart(
                duel(candidate, opponent, THREE, Seating.EVERY_PLACEMENT, one, DrawOrder.SEAT_STREAMS, seed, base,
                    WALLS, fixed, Score.RANK, RankUtility.AVERAGE_RANK),
                List.of("the rank utility differs (TENHOU and AVERAGE_RANK)")),
            new Apart(
                duel(candidate, opponent, THREE, Seating.EVERY_PLACEMENT, one, DrawOrder.WALL_ORDER, seed + 2,
                    base, WALLS, fixed, Score.RANK, RankUtility.TENHOU),
                List.of(
                    "the wall seed differs (" + seed + " and " + (seed + 2) + ")",
                    "the draw order differs (SEAT_STREAMS and WALL_ORDER)")));
    for (Apart pair : apart) {
      job(run, 3, plan(candidate, pair.duel(), "jar-a"), JobFile.State.DONE, 0, List.of());
      SettingsException refused =
          Assert.expectThrows(SettingsException.class, () -> JobComparison.of(TABLE, run, 2, run, 3));
      Assert.assertEquals(
          refused.problems(),
          pair.because().stream().map(because -> "Cannot pair v00002-x1n2 with v00003-x1n2: " + because).toList());
    }

    // 候補・牌山の数・止め方・判定に使う量・jar が違っても、相手が同じバージョンなら対にする
    DuelSetup longer =
        duel(candidate, opponent, THREE, Seating.EVERY_PLACEMENT, one, DrawOrder.SEAT_STREAMS, seed, base, 2 * WALLS,
            new StopRule.Precision(0.005, 0.01, WALLS), Score.UTILITY, RankUtility.TENHOU);
    Places second = drawn(11);
    job(run, 3, plan(candidate, longer, "jar-b"), JobFile.State.DONE, 2 * WALLS,
        lines(longer, 2 * WALLS, second));
    JobComparison comparison = JobComparison.of(TABLE, run, 2, run, 3);
    Assert.assertEquals(comparison.walls(), WALLS);
    assertValue(comparison.rank(), minus(rank(second, WALLS), rank(first, WALLS)), "rank difference");
    Assert.assertEquals(comparison.second().job().walls() - comparison.walls(), WALLS);
    Assert.assertEquals(comparison.first().plan().jar(), "jar-a");
    Assert.assertEquals(comparison.second().plan().jar(), "jar-b");

    // 同値な打牌の切り方が違えば、ほかの計画が同じでも対にしない
    job(run, 4, new EvaluationPlan(candidate, longer, "jar-a", TwinDiscard.TEDASHI), JobFile.State.DONE, 0, List.of());
    SettingsException tedashi =
        Assert.expectThrows(SettingsException.class, () -> JobComparison.of(TABLE, run, 2, run, 4));
    Assert.assertEquals(
        tedashi.problems(),
        List.of("Cannot pair v00002-x1n2 with v00004-x1n2: the twin discard differs (TSUMOGIRI and TEDASHI)"));
  }

  /** 系譜の外のチェックポイントの相手は SHA-256 で比べる。場所が違っても中身が同じなら対にし、中身が違えば拒む。 */
  @Test
  public void anOutsideOpponentIsComparedByItsSha256() throws Exception {
    Places first = drawn(29);
    Places second = drawn(31);
    DuelSetup placed = duel(candidate(run, 2), outside("/models/v760", OPPONENT_SHA));
    job(run, 2, plan(placed.candidate(), placed, "jar-a"), JobFile.State.DONE, WALLS,
        lines(placed, WALLS, first));
    DuelSetup copied = duel(candidate(run, 3), outside("/copies/v760", OPPONENT_SHA));
    job(run, 3, plan(copied.candidate(), copied, "jar-a"), JobFile.State.DONE, WALLS,
        lines(copied, WALLS, second));
    JobComparison comparison = JobComparison.of(TABLE, run, 2, run, 3);
    assertValue(comparison.rank(), minus(rank(second, WALLS), rank(first, WALLS)), "rank difference");

    String other = "b".repeat(64);
    DuelSetup changed = duel(candidate(run, 4), outside("/models/v760", other));
    job(run, 4, plan(changed.candidate(), changed, "jar-a"), JobFile.State.DONE, 0, List.of());
    SettingsException refused =
        Assert.expectThrows(SettingsException.class, () -> JobComparison.of(TABLE, run, 2, run, 4));
    Assert.assertEquals(
        refused.problems(),
        List.of(
            "Cannot pair v00002-x1n2 with v00004-x1n2: the opponent differs (nano:v760 sha256 " + OPPONENT_SHA
                + " and nano:v760 sha256 " + other + ")"));
  }

  @Test
  public void anotherRunIsComparedAndUnpairableJobsAreRefused() throws Exception {
    RunDirectory other = new RunDirectory(root.resolve("b"));
    Places mine = drawn(13);
    Places theirs = drawn(17);
    done(run, 5, mine, "jar-a");
    // 別の系譜のジョブ。相手はその系譜のバージョン 1 で、番号は同じでも中身の違うネットワークなので対にしない
    done(other, 4, theirs, "jar-a");
    SettingsException apart =
        Assert.expectThrows(SettingsException.class, () -> JobComparison.of(TABLE, run, 5, other, 4));
    Assert.assertEquals(
        apart.problems(),
        List.of(
            "Cannot pair " + run.job("v00005-x1n2") + " with " + other.job("v00004-x1n2")
                + ": the opponent differs (" + named(run, 1) + " and " + named(other, 1) + ")"));

    // 別の run に書いた、同じ系譜のバージョンを測るジョブ（相手も同じ系譜のバージョン 1）とは対にする
    done(other, run, 5, theirs, "jar-a");
    JobComparison comparison = JobComparison.of(TABLE, run, 5, other, 5);
    assertValue(comparison.rank(), minus(rank(theirs, WALLS), rank(mine, WALLS)), "rank difference");
    assertValue(comparison.utility(), minus(utility(theirs, WALLS), utility(mine, WALLS)), "utility difference");
    Assert.assertEquals(comparison.first().run(), run);
    Assert.assertEquals(comparison.second().run(), other);

    // 同じジョブどうし
    SettingsException itself =
        Assert.expectThrows(SettingsException.class, () -> JobComparison.of(TABLE, run, 5, run, 5));
    Assert.assertEquals(
        itself.problems(), List.of("Both jobs are " + run.job("v00005-x1n2") + "; compare two different jobs"));

    // 無いジョブとまだ計画の無いジョブは、理由をまとめて出す
    DurableFiles.write(
        run.job("v00006-x1n2").resolve(Jobs.JOB),
        JobFile.pending(6, TABLE, 0, 1, RecordFile.tree(new Definition(TABLE))));
    SettingsException missing =
        Assert.expectThrows(SettingsException.class, () -> JobComparison.of(TABLE, run, 4, run, 6));
    Assert.assertEquals(
        missing.problems(),
        List.of(
            "There is no job v00004-x1n2 in " + run.root(),
            "The job v00006-x1n2 in " + run.root() + " has not started, so it has no plan yet"));

    // 王者の保存先が空だったので、打たずに候補を初めの王者にしたジョブ
    job(run, 7, plan(candidate(run, 7), null, "jar-a"), JobFile.State.DONE, 0, List.of());
    SettingsException crowned =
        Assert.expectThrows(SettingsException.class, () -> JobComparison.of(TABLE, run, 5, run, 7));
    Assert.assertEquals(
        crowned.problems(),
        List.of("The job v00007-x1n2 in " + run.root() + " played no walls: it made v00007 the first champion"));

    // 別の run との違いは、ジョブのディレクトリで書く
    DuelSetup seeded =
        duel(candidate(other, 8), opponent(run), THREE, Seating.EVERY_PLACEMENT, List.of("1v2"),
            DrawOrder.SEAT_STREAMS, 7, Arena.WALL_BASE, WALLS, new StopRule.Fixed(0.01), Score.RANK,
            RankUtility.TENHOU);
    job(other, 8, plan(seeded.candidate(), seeded, "jar-a"), JobFile.State.DONE, 0, List.of());
    SettingsException across =
        Assert.expectThrows(SettingsException.class, () -> JobComparison.of(TABLE, run, 5, other, 8));
    Assert.assertEquals(
        across.problems(),
        List.of(
            "Cannot pair " + run.job("v00005-x1n2") + " with " + other.job("v00008-x1n2")
                + ": the wall seed differs (" + Arena.EVALUATION_SEED + " and 7)"));
  }

  /** report を回して終了コードを確かめ、出力を返す。 */
  private String report(Exit exit, String... arguments) {
    List<String> all = new ArrayList<>(List.of("report", "--run", run.root().toString()));
    all.addAll(List.of(arguments));
    ByteArrayOutputStream captured = new ByteArrayOutputStream();
    PrintStream out = new PrintStream(captured, true, StandardCharsets.UTF_8);
    int code = App.run(PrismApp.application(), all.toArray(String[]::new), out);
    String printed = captured.toString(StandardCharsets.UTF_8);
    Assert.assertEquals(code, exit.code(), all + "\n" + printed);
    return printed;
  }

  /** 出力の、start で始まる最初の行。 */
  private static String line(String printed, String start) {
    return printed
        .lines()
        .filter(line -> line.startsWith(start))
        .findFirst()
        .orElseThrow(() -> new AssertionError("No line starts with " + start + " in\n" + printed));
  }

  @Test
  public void theReportCommandPrintsThePairedDifferenceAndRefusesBadArguments() throws Exception {
    Places first = drawn(19);
    Places second = drawn(23);
    done(run, 2, first, "jar-a");
    done(run, 3, second, "jar-b");

    String printed = report(Exit.OK, "--compare", TABLE, "--versions", "2,3");
    Assert.assertTrue(printed.startsWith("x1n2: v00003 against v00002 on 12 paired walls (36 games)"), printed);
    double[] ranks = minus(rank(second, WALLS), rank(first, WALLS));
    double[] utilities = minus(utility(second, WALLS), utility(first, WALLS));
    Assert.assertTrue(
        line(printed, "rank ")
            .contains(String.format(Locale.ROOT, "second - first %.6f, paired error %.6f", mean(ranks), error(ranks))),
        printed);
    Assert.assertTrue(
        line(printed, "utility ")
            .contains(
                String.format(
                    Locale.ROOT, "second - first %.6f, paired error %.6f", mean(utilities), error(utilities))),
        printed);
    Assert.assertTrue(
        line(printed, "first ").contains("v00002-x1n2 in " + run.root() + ": DONE, 12 walls (0 unpaired)"), printed);
    Assert.assertTrue(line(printed, "first ").contains("candidate v00002, jar jar-a"), printed);
    Assert.assertTrue(line(printed, "second ").contains("candidate v00003, jar jar-b"), printed);

    // --other の run に書いた、同じ系譜の同じバージョンを測るジョブ
    RunDirectory other = new RunDirectory(root.resolve("b"));
    done(other, run, 2, second, "jar-a");
    String across = report(Exit.OK, "--compare", TABLE, "--versions", "2,2", "--other", other.root().toString());
    Assert.assertTrue(line(across, "second ").contains("v00002-x1n2 in " + other.root()), across);

    // 引数の誤りと、対にできない組は使い方の誤り
    report(Exit.USAGE, "--compare", TABLE);
    report(Exit.USAGE, "--versions", "2,3");
    report(Exit.USAGE, "--other", other.root().toString());
    report(Exit.USAGE, "--compare", TABLE, "--versions", "2");
    report(Exit.USAGE, "--compare", TABLE, "--versions", "0,3");
    report(Exit.USAGE, "--compare", TABLE, "--versions", "2,3", "--set", "schedule.iterations=2");
    report(Exit.USAGE, "--compare", TABLE, "--versions", "2,3", "--last", "1");
    report(Exit.USAGE, "--compare", TABLE, "--versions", "2,3", "--json", "true");
    report(Exit.USAGE, "--compare", TABLE, "--versions", "2,9");
    report(Exit.USAGE, "--compare", TABLE, "--versions", "2,2");
  }
}
