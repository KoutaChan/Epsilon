package com.epsilon.prism.app;

import com.epsilon.agent.app.Exit;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.ScoreSmooth;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.Main;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiEntry;
import com.epsilon.prism.grp.PhiHistory;
import com.epsilon.prism.grp.PhiSetup;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;
import java.util.SplittableRandom;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 三人打ちの run で GRP を測るコマンドのテスト。grp-sigma は確定済みのコホートの半荘の記録から σ の係数を測り、決めた σ で点を動かした
 * 記録からはその σ が戻ってくる。grp-trust は、点数だけで決まる Φ で集めたコホートで候補を学習し、関門用の半荘で 2 SE の判定に要る半荘の
 * 数を出す（乱数の種が同じなら出力も同じ）。勧める trustGames は、候補を学習した半荘の数から決めた学び直しの範囲の半荘の数で、1 コホートの
 * 半荘が判定に足りるときだけ勧める。三人打ちで点数のなめらかな関数を Φ にする設定は、settings check で問題として報告される。
 */
public class ThreePlayerGrpCommandsTest {

  private static final Variant THREE = Variant.TENHOU_3P;

  /** 決めた σ の係数（点）。 */
  private static final double SIGMA = 8_000;

  /** 半荘の最終の点数の組。 */
  private static final int[] FINALS = {50_000, 35_000, 20_000};

  private Path run;
  private GrpFiles files;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-3p-grp-commands");
    files = new GrpFiles(run);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  /** コマンドを実行した結果（終了コードと標準出力）。 */
  private record Ran(int exit, String out) {}

  private static Ran prism(String... arguments) {
    PrintStream saved = System.out;
    ByteArrayOutputStream captured = new ByteArrayOutputStream();
    int exit;
    System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8));
    try {
      exit = Main.run(arguments);
    } finally {
      System.setOut(saved);
    }
    return new Ran(exit, captured.toString(StandardCharsets.UTF_8));
  }

  private static void seal(GrpFiles files, int t) throws IOException {
    Files.createDirectories(files.cohort(t));
    Files.writeString(files.cohort(t).resolve(CohortWriter.MANIFEST), "{}");
  }

  private static void records(GrpFiles files, int t, String phiId, List<GameRecord> games) throws IOException {
    Files.createDirectories(files.records(t).getParent());
    GameRecordCodec.write(files.records(t), new GameCohort(THREE, t, phiId, games));
  }

  /** 局末の点数を、最終の点数から σ·√(残りの局の数)·z（z は標準正規分布に従う）だけずらした三人打ちの半荘を作る。 */
  private static List<GameRecord> moves(int count, long firstId, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    List<GameRecord> games = new ArrayList<>(count);
    for (int game = 0; game < count; game++) {
      int[] finals = new int[3];
      for (int seat = 0; seat < 3; seat++) {
        finals[seat] = FINALS[(seat + game) % 3];
      }
      int hands = 1 + random.nextInt(8);
      List<GrpState> handEnds = new ArrayList<>(hands);
      for (int hand = 0; hand < hands; hand++) {
        int round = random.nextInt(THREE.finalRound() + 1);
        double scale = SIGMA * Math.sqrt(ScoreSmooth.remainingRounds(THREE, round));
        int[] scores = new int[3];
        for (int seat = 0; seat < 3; seat++) {
          scores[seat] = finals[seat] - (int) Math.round(scale * random.nextGaussian());
        }
        handEnds.add(new GrpState(THREE, round, 0, 0, scores));
      }
      games.add(new GameRecord(THREE, firstId + game, handEnds, finals));
    }
    return games;
  }

  private static long seatHands(List<GameRecord> games) {
    long count = 0;
    for (GameRecord game : games) {
      count += 3L * game.handEnds().size();
    }
    return count;
  }

  private static final Pattern SIGMA_LINE =
      Pattern.compile(
          "TENHOU_3P over (\\d+) cohorts \\((\\d+)\\.\\.(\\d+)\\), (\\d+) games, (\\d+) seat-hands: sigma per hand"
              + " ([0-9.]+) \\(in use: none\\)");

  @Test
  public void grpSigmaGivesBackTheSigmaOfTheMoves() throws Exception {
    List<GameRecord> all = new ArrayList<>();
    for (int t = 1; t <= 3; t++) {
      List<GameRecord> games = moves(1_000, 1_000L * t, t);
      seal(files, t);
      records(files, t, PhiEntry.INITIAL_ID, games);
      all.addAll(games);
    }
    // 数のファイルをまだ移していないコホートは測らない
    seal(files, 4);
    records(files, 4, PhiEntry.INITIAL_ID, moves(1_000, 4_000, 4));
    Files.writeString(files.unmovedCounts(4), "");
    // 行のコホートを消した後も、残っている半荘の記録は測る
    Snapshot.deleteTree(files.cohort(1));
    Ran ran = prism("grp-sigma", "--run", run.toString());
    Assert.assertEquals(ran.exit(), Exit.OK.code(), ran.out());
    Matcher matcher = SIGMA_LINE.matcher(ran.out());
    Assert.assertTrue(matcher.find(), ran.out());
    Assert.assertEquals(Integer.parseInt(matcher.group(1)), 3);
    Assert.assertEquals(matcher.group(2) + ".." + matcher.group(3), "1..3");
    Assert.assertEquals(Integer.parseInt(matcher.group(4)), 3_000);
    long seatHands = seatHands(all);
    Assert.assertEquals(Long.parseLong(matcher.group(5)), seatHands);
    double measured = Double.parseDouble(matcher.group(6));
    Assert.assertEquals(measured, SIGMA, 4 * SIGMA / Math.sqrt(2.0 * seatHands), ran.out());
    // 次の局ごとの行は、東 1 局から西 3 局まで出る
    for (String name : List.of("E1", "E2", "E3", "S1", "S2", "S3", "W1", "W2", "W3")) {
      Assert.assertTrue(ran.out().contains("next " + name + ": remaining "), name + " in " + ran.out());
    }
    Assert.assertTrue(ran.out().contains("next E1: remaining 6,"), ran.out());
    Assert.assertTrue(ran.out().contains("next S3: remaining 1,"), ran.out());

    Ran part = prism("grp-sigma", "--run", run.toString(), "--cohorts", "2..3");
    Assert.assertEquals(part.exit(), Exit.OK.code(), part.out());
    Matcher partial = SIGMA_LINE.matcher(part.out());
    Assert.assertTrue(partial.find(), part.out());
    Assert.assertEquals(partial.group(2) + ".." + partial.group(3), "2..3");
    Assert.assertEquals(Integer.parseInt(partial.group(4)), 2_000);
    Assert.assertEquals(prism("grp-sigma", "--run", run.toString(), "--cohorts", "5..9").exit(), Exit.USAGE.code());
  }

  /** 勧める trustGames は、学び直しの範囲に要る半荘の数以上で最小の 2 の冪。int に収まらなければ空。 */
  @Test
  public void suggestedTrustGamesIsTheNextPowerOfTwo() {
    Assert.assertEquals(GrpTrust.suggestion(0.2), OptionalInt.of(1));
    Assert.assertEquals(GrpTrust.suggestion(1), OptionalInt.of(1));
    Assert.assertEquals(GrpTrust.suggestion(1.01), OptionalInt.of(2));
    Assert.assertEquals(GrpTrust.suggestion(24.3), OptionalInt.of(32));
    Assert.assertEquals(GrpTrust.suggestion(32), OptionalInt.of(32));
    Assert.assertEquals(GrpTrust.suggestion(131_072), OptionalInt.of(131_072));
    Assert.assertEquals(GrpTrust.suggestion(131_073), OptionalInt.of(262_144));
    Assert.assertEquals(GrpTrust.suggestion(1 << 30), OptionalInt.of(1 << 30));
    Assert.assertEquals(GrpTrust.suggestion((1 << 30) + 1.0), OptionalInt.empty());
    Assert.assertEquals(GrpTrust.suggestion(Double.POSITIVE_INFINITY), OptionalInt.empty());
  }

  /** 三人打ちで点数のなめらかな関数を Φ にすると、settings check で問題になる（終了コード 4）。点数の順位と、自分の GRP の既定の設定は問題ない。 */
  @Test
  public void settingsCheckRefusesThreePlayerScoreSmooth() throws Exception {
    Assert.assertEquals(check("rank.toml", "[phi]\nsource = \"SCORE_RANK\"\n"), Exit.OK.code());
    Assert.assertEquals(check("self.toml", "[phi]\nsource = \"SELF_GRP\"\n"), Exit.OK.code());
    Assert.assertEquals(check("smooth.toml", "[phi]\nsource = \"SCORE_SMOOTH\"\n"), Exit.PROBLEMS.code());
    Assert.assertEquals(
        check("self-smooth.toml", "[phi]\nsource = \"SELF_GRP\"\n\n[phi.start]\nkind = \"SCORE_SMOOTH\"\n"),
        Exit.PROBLEMS.code());
  }

  private int check(String name, String body) throws IOException {
    Path file = Files.writeString(run.resolve(name), body + "\n[run]\nvariant = \"TENHOU_3P\"\nseed = 1\n");
    return prism("settings", "check", "--settings", file.toString()).exit();
  }

  /** 自分の GRP を点数の順位から始める三人打ちの run を作る（g000 と設定の 1 行目を書く）。 */
  private PhiEntry selfRun(int next, String... more) throws IOException {
    List<String> assignments =
        new ArrayList<>(
            List.of(
                "phi.source=SELF_GRP",
                "phi.start.kind=SCORE_RANK",
                "phi.grp.monitor=RECORD",
                "phi.grp.refit=MANUAL",
                "phi.grp.switching=PROPOSE",
                "phi.grp.trustGames=64",
                "phi.grp.threads=1"));
    assignments.addAll(List.of(more));
    PrismSettings settings = ThreePlayerSettings.create(run, List.of(), assignments.toArray(String[]::new));
    Files.writeString(run.resolve("state.json"), "{\"format\": 1, \"next\": " + next + "}");
    return PhiHistory.initialize(files, new PhiSetup(settings.phi(), settings.run().variant(), settings.run().seed()));
  }

  /** 最終の順位が局末の点数と無関係な三人打ちの半荘を作る（局末は 2〜8 個で、次の局は 1 局ずつ進む）。 */
  private static List<GameRecord> shuffled(int count, long firstId, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    List<GameRecord> games = new ArrayList<>(count);
    for (int game = 0; game < count; game++) {
      int[] finals = FINALS.clone();
      for (int index = 2; index > 0; index--) {
        int other = random.nextInt(index + 1);
        int held = finals[index];
        finals[index] = finals[other];
        finals[other] = held;
      }
      int hands = 2 + random.nextInt(7);
      List<GrpState> handEnds = new ArrayList<>(hands);
      for (int hand = 0; hand < hands; hand++) {
        int first = 35_000 + 100 * (random.nextInt(201) - 100);
        int second = 35_000 + 100 * (random.nextInt(201) - 100);
        handEnds.add(new GrpState(THREE, hand + 1, 0, 0, new int[] {first, second, 105_000 - first - second}));
      }
      games.add(new GameRecord(THREE, firstId + game, handEnds, finals));
    }
    return games;
  }

  private static final Pattern HELD =
      Pattern.compile(
          "held out: (\\d+) games with hand ends, ([0-9.]+) hand ends per game, paired MSE candidate - used"
              + " ([-+][0-9.]+) \\(SE ([0-9.]+), z [-+][0-9.]+\\), bootstrap SD ([0-9.]+) over (\\d+) resamples");

  private static final Pattern SPLIT =
      Pattern.compile(": (\\d+) games learned from scratch \\([^)]*\\), (\\d+) games held out");

  private static final Pattern JUDGMENT =
      Pattern.compile(
          "2 SE judgment: (\\d+) games at the line, (\\d+) games with 95% power, counted in the cohort after the"
              + " refit range \\(a cohort of the range has (\\d+)\\); the candidate learned from (\\d+) games of the"
              + " range; suggested trustGames (\\d+) \\(the run has 64\\)");

  /**
   * 点数の順位で集めた三人打ちのコホートで grp-trust を実行する。乱数の種が同じなら出力も同じ。出力される必要な半荘の数は、同じく出力される差と
   * 標準偏差から G·(2·SD/d)² で求めた値と合う。その数と比べるのは 1 コホートの半荘の数（ここでは 200）。勧める trustGames は、候補を学習した
   * 半荘の数以上で最小の 2 の冪。範囲を省くと最新のコホート 1 つを測る。
   */
  @Test(groups = "native", timeOut = 900_000)
  public void grpTrustIsDeterministicAndFollowsTheFormula() throws Exception {
    PhiEntry initial = selfRun(3);
    for (int t = 1; t <= 2; t++) {
      seal(files, t);
      records(files, t, initial.id(), shuffled(200, 1_000L * t, t));
    }
    String[] arguments = {"grp-trust", "--run", run.toString(), "--cohorts", "1..2", "--resamples", "400"};
    Ran first = prism(arguments);
    Assert.assertEquals(first.exit(), Exit.OK.code(), first.out());
    Ran again = prism(arguments);
    Assert.assertEquals(again.exit(), Exit.OK.code(), again.out());
    Assert.assertEquals(again.out(), first.out(), "the same run and range give the same output");
    Assert.assertTrue(
        first.out().contains("cohorts 1..2 of TENHOU_3P collected with " + initial.id() + " (SCORE_RANK)"), first.out());
    Matcher held = HELD.matcher(first.out());
    Assert.assertTrue(held.find(), first.out());
    int games = Integer.parseInt(held.group(1));
    double difference = Double.parseDouble(held.group(3));
    double sd = Double.parseDouble(held.group(5));
    Assert.assertEquals(Integer.parseInt(held.group(6)), 400);
    Assert.assertTrue(games > 50 && games < 200, "held-out games " + games);
    // 点数の順位の MSE は 16/3、候補は 8/3 に近いはず。差は −8/3 のまわり
    Assert.assertTrue(difference < -1, first.out());
    Matcher judgment = JUDGMENT.matcher(first.out());
    Assert.assertTrue(judgment.find(), first.out());
    double line = games * Math.pow(2 * sd / difference, 2);
    double power = games * Math.pow((2 + 1.6448536269514722) * sd / difference, 2);
    // 出力の値は丸めてあるので、1 半荘のずれまで許す
    Assert.assertEquals(Long.parseLong(judgment.group(1)), (long) Math.ceil(line), 1.0, first.out());
    Assert.assertEquals(Long.parseLong(judgment.group(2)), (long) Math.ceil(power), 1.0, first.out());
    // 学び直しの関門が比べるのは次のコホートの半荘なので、判定に要る数と比べるのは 1 コホートの半荘（どの半荘も局末を持つ）
    Assert.assertEquals(Integer.parseInt(judgment.group(3)), 200, first.out());
    Assert.assertTrue(Long.parseLong(judgment.group(2)) <= 200, first.out());
    // 範囲の半荘は、学習した半荘と関門用に残した半荘を合わせた 400。学び直しの範囲の半荘に当たるのは、学習した半荘
    Matcher split = SPLIT.matcher(first.out());
    Assert.assertTrue(split.find(), first.out());
    int learned = Integer.parseInt(split.group(1));
    Assert.assertEquals(learned + Integer.parseInt(split.group(2)), 400, first.out());
    Assert.assertEquals(Integer.parseInt(judgment.group(4)), learned, first.out());
    int suggested = Integer.parseInt(judgment.group(5));
    Assert.assertEquals(Integer.bitCount(suggested), 1);
    Assert.assertTrue(suggested >= learned && suggested / 2 < learned, first.out());

    // 範囲を省くと、点数だけで決まる Φ で集めた確定済みのコホートのうち、最新のものを測る
    Ran latest = prism("grp-trust", "--run", run.toString(), "--resamples", "100");
    Assert.assertEquals(latest.exit(), Exit.OK.code(), latest.out());
    Assert.assertTrue(latest.out().contains("cohorts 2..2 of TENHOU_3P"), latest.out());
    // 確定していないコホートを含む範囲は測らない
    Assert.assertEquals(prism("grp-trust", "--run", run.toString(), "--cohorts", "1..3").exit(), Exit.FAILED.code());
  }

  private static final Pattern POWER = Pattern.compile("(\\d+) games with 95% power");

  private static final Pattern SUGGESTED = Pattern.compile("suggested trustGames (\\S+) \\(the run has 64\\)");

  /**
   * 1 コホートの半荘が 2 SE の判定に要る数より少ない run では、trustGames を勧めない。学び直しの関門が比べるのは、範囲の次のコホートの
   * 全ての半荘。trustGames を大きくして範囲を広げても、関門で比べる半荘は増えない。
   */
  @Test(groups = "native", timeOut = 900_000)
  public void grpTrustSuggestsNothingWhenOneCohortIsSmallerThanTheJudgmentNeeds() throws Exception {
    int cohorts = 200;
    PhiEntry initial = selfRun(cohorts + 1);
    List<GameRecord> games = new ArrayList<>(shuffled(200, 1_000, 1));
    games.addAll(shuffled(200, 2_000, 2));
    for (int t = 1; t <= cohorts; t++) {
      seal(files, t);
      records(files, t, initial.id(), List.copyOf(games.subList(2 * (t - 1), 2 * t)));
    }
    Ran ran = prism("grp-trust", "--run", run.toString(), "--cohorts", "1.." + cohorts, "--resamples", "400");
    Assert.assertEquals(ran.exit(), Exit.OK.code(), ran.out());
    Matcher power = POWER.matcher(ran.out());
    Assert.assertTrue(power.find(), ran.out());
    Assert.assertTrue(Long.parseLong(power.group(1)) > 2, ran.out());
    Matcher suggested = SUGGESTED.matcher(ran.out());
    Assert.assertTrue(suggested.find(), ran.out());
    Assert.assertEquals(suggested.group(1), "none", ran.out());
    Assert.assertTrue(
        ran.out().contains("a cohort of the range has 2 games with hand ends, fewer than the 2 SE judgment needs"),
        ran.out());
  }

  /** 自分の GRP を使わない run では、使い方の誤りになる。 */
  @Test
  public void grpTrustNeedsASelfGrpRun() throws Exception {
    ThreePlayerSettings.create(run, List.of(), "phi.source=SCORE_RANK");
    Assert.assertEquals(prism("grp-trust", "--run", run.toString()).exit(), Exit.USAGE.code());
  }
}
