package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.eval.DuelPairsTest.CommonRandom;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.JobRunner;
import com.epsilon.agent.run.JobWork;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;

/**
 * games.tsv を読むクラス（{@link WallValues}）のテスト。三人打ちの決闘のジョブが書いた行から牌山ごとの順位と効用の対の量を読み、同じ牌山を打って
 * {@link PairScore} で求めた値とビット単位で一致することを確かめる（全ての割り当てを打つ卓と、1 半荘ずつの卓の両方）。牌山の中で割り当ての
 * 順が入れ替わった行、席ごとの打ち手の番号が違う行、4 席の行、途中で切れた行、計画より多い牌山は拒む。同じ方策どうしの決闘では、どちらの
 * ツモの順でも、games.tsv から読んだ値が全ての牌山で 0 になる。対にできる計画の照合（{@link DuelSetup#pairsWith}）も確かめる。
 */
public class WallValuesTest {

  private static final Variant THREE = Variant.TENHOU_3P;
  private static final long SEED = 4_242L, BASE = 7_000L;
  private static final int WALLS = 12;
  private static final List<String> BOTH = List.of("1v2", "2v1"), ONE = List.of("1v2");

  /** ジョブの定義。中身は使わない。 */
  record Definition(String name) {}

  /** 渡した計画をそのまま書くランナー。打ち手は、計画にある打ち手の指定（数の文字列）をソルトにした {@link CommonRandom} で開く。 */
  record Runner(TableLoop table, DuelSetup setup) implements JobRunner {

    @Override
    public Node.Table plan(JobFile job) {
      return RecordFile.tree(setup);
    }

    @Override
    public JobWork open(JobFile job, Path directory) throws Exception {
      DuelSetup planned = RecordFile.read(DuelSetup.class, job.plan()).value();
      return Duel.open(
          job.name(), table, planned, random(planned.candidate()), random(planned.opponent()), directory);
    }
  }

  private static CommonRandom random(DuelSetup.Entrant entrant) {
    return new CommonRandom(THREE, Long.parseLong(entrant.spec()));
  }

  private static DuelSetup.Entrant entrant(long salt) {
    return new DuelSetup.Entrant(Long.toString(salt), "random:" + salt, null, null);
  }

  private static DuelSetup setup(
      Seating seating, List<String> splits, DrawOrder draws, long base, int walls, long candidate, long opponent) {
    List<Split> parsed = splits.stream().map(split -> Split.parse(split, 2, THREE)).toList();
    return DuelSetup.of(
        THREE, entrant(candidate), entrant(opponent), seating, parsed, draws, WallSet.FIXED, SEED, base, walls,
        new StopRule.Fixed(0.01), Score.RANK, RankUtility.TENHOU);
  }

  private static TableLoop table() {
    return new TableLoop(new TableSettings(16, 4, 1000L, 4));
  }

  /** 決闘のジョブを終わるまで回し、ジョブのディレクトリを返す。 */
  private static Path duel(TableLoop table, Path root, DuelSetup setup) throws Exception {
    Jobs jobs = new Jobs(root.resolve("jobs"), root.resolve("evaluations.jsonl"));
    JobFile job = JobFile.pending(1, "pair", 0, 1, RecordFile.tree(new Definition("pair")));
    assertTrue(jobs.create(job));
    assertTrue(jobs.run(job, new Runner(table, setup), RunControl.detached(() -> false)).isEmpty());
    JobFile done = jobs.read(job.name());
    assertEquals(done.state(), JobFile.State.DONE, done.reason());
    return root.resolve("jobs").resolve(job.name());
  }

  /** 同じ計画を卓で打ち、牌山ごとの順位と効用の対の量を {@link PairScore} で求める。[0] が順位、[1] が効用。 */
  private static double[][] played(TableLoop table, DuelSetup setup) {
    ArenaPlan arena = setup.plan(random(setup.candidate()), random(setup.opponent())).arena();
    PairScore pair = new PairScore(arena);
    double[][] values = new double[2][setup.walls()];
    Arena.play(
        table,
        arena,
        (wall, scores, stats) -> {
          int index = (int) (wall - setup.wallBase());
          pair.measure(wall, scores);
          values[0][index] = pair.rank();
          values[1][index] = pair.utility();
          return true;
        },
        null);
    return values;
  }

  private static void assertBits(double[] actual, double[] expected, String label) {
    assertTrue(Arrays.equals(actual, expected), label + ": " + Arrays.toString(actual) + " vs " + Arrays.toString(expected));
  }

  @Test(timeOut = 600_000)
  public void theReaderGivesThePlayedWallValuesBitForBit() throws Exception {
    Path root = Files.createTempDirectory("wall-values");
    try (TableLoop table = table()) {
      for (Seating seating : Seating.values()) {
        List<String> splits = seating == Seating.BY_GAME ? ONE : BOTH;
        DuelSetup setup = setup(seating, splits, DrawOrder.SEAT_STREAMS, BASE, WALLS, 3, 5);
        Path home = duel(table, root.resolve(seating.name()), setup);
        List<String> lines = Files.readAllLines(home.resolve(JobOutputs.GAMES));
        int perWall = setup.layout().gamesPerWall();
        assertEquals(perWall, seating == Seating.BY_GAME ? 1 : 6, seating.name());
        assertEquals(lines.size(), WALLS * perWall, seating.name());

        WallValues values = WallValues.of(setup, lines);
        double[][] expected = played(table, setup);
        assertEquals(values.walls(), WALLS, seating.name());
        assertBits(values.rank(), expected[0], seating + " rank");
        assertBits(values.utility(), expected[1], seating + " utility");
        assertTrue(Arrays.stream(values.rank()).anyMatch(rank -> rank != 0), seating + " all walls are even");
        double[] negated = Arrays.stream(values.rank()).map(rank -> -rank).toArray();
        assertBits(values.better(Score.RANK), negated, seating + " better rank");
        assertBits(values.better(Score.UTILITY), expected[1], seating + " better utility");
        // 決闘の結果の平均は、読んだ牌山ごとの値の平均と一致する
        DuelResult result = Jobs.result(home, DuelResult.class);
        assertEquals(result.rank().mean(), Arrays.stream(values.rank()).average().orElseThrow(), 1e-12, seating.name());
        assertEquals(
            result.utility().mean(), Arrays.stream(values.utility()).average().orElseThrow(), 1e-12, seating.name());
        // 牌山の数が計画より少ない行（止め方で早く止まったジョブ）は、行のある牌山だけを読む
        WallValues head = WallValues.of(setup, lines.subList(0, 4 * perWall));
        assertEquals(head.walls(), 4, seating.name());
        assertBits(head.rank(), Arrays.copyOf(expected[0], 4), seating + " head");
      }
    } finally {
      DurableFiles.delete(root);
    }
  }

  @Test(timeOut = 600_000)
  public void linesThatDoNotMatchThePlanAreRefused() throws Exception {
    Path root = Files.createTempDirectory("wall-values-refused");
    try (TableLoop table = table()) {
      DuelSetup setup = setup(Seating.EVERY_PLACEMENT, BOTH, DrawOrder.SEAT_STREAMS, BASE, WALLS, 3, 5);
      List<String> lines = Files.readAllLines(duel(table, root, setup).resolve(JobOutputs.GAMES));
      assertEquals(WallValues.of(setup, lines).walls(), WALLS);

      // 牌山の中で割り当ての順が入れ替わった行
      List<String> swapped = new ArrayList<>(lines);
      swapped.set(6, lines.get(7));
      swapped.set(7, lines.get(6));
      refused(setup, swapped, "is not game");
      // 席ごとの打ち手の番号が割り当てと違う行
      List<String> slots = new ArrayList<>(lines);
      String[] fields = lines.get(0).split("\t", -1);
      fields[2] = new StringBuilder(fields[2]).reverse().toString();
      assertFalse(fields[2].equals(lines.get(0).split("\t", -1)[2]), fields[2]);
      slots.set(0, String.join("\t", fields));
      refused(setup, slots, "is not game");
      // 4 席の行（席ごとの打ち手の番号と点数が 1 つずつ多い）
      List<String> four = new ArrayList<>(lines);
      List<String> wide = new ArrayList<>(List.of(lines.get(1).split("\t", -1)));
      wide.set(2, wide.get(2) + "1");
      wide.add(DuelTally.SCORE_COLUMN + 3, "25000");
      four.set(1, String.join("\t", wide));
      refused(setup, four, "columns");
      // 途中で切れた行（最後の列が無い）
      List<String> cut = new ArrayList<>(lines);
      String last = lines.get(lines.size() - 1);
      cut.set(lines.size() - 1, last.substring(0, last.lastIndexOf('\t')));
      refused(setup, cut, "columns");
      // 牌山の途中で終わる行
      refused(setup, lines.subList(0, 2 * 6 + 1), "not whole walls");
      // 計画より牌山が多い行
      DuelSetup shorter = setup(Seating.EVERY_PLACEMENT, BOTH, DrawOrder.SEAT_STREAMS, BASE, WALLS / 2, 3, 5);
      refused(shorter, lines, "within " + WALLS / 2 + " walls");
      // 牌山の番号が計画と違う行
      DuelSetup shifted = setup(Seating.EVERY_PLACEMENT, BOTH, DrawOrder.SEAT_STREAMS, BASE + 1, WALLS, 3, 5);
      refused(shifted, lines, "is not game");
      // 1 半荘ずつの計画で読むと、全ての割り当てを打った行は牌山の番号が合わない
      DuelSetup byGame = setup(Seating.BY_GAME, ONE, DrawOrder.SEAT_STREAMS, BASE, WALLS * 6, 3, 5);
      refused(byGame, lines, "is not game");
    } finally {
      DurableFiles.delete(root);
    }
  }

  private static void refused(DuelSetup setup, List<String> lines, String because) {
    IllegalArgumentException failure = expectThrows(IllegalArgumentException.class, () -> WallValues.of(setup, lines));
    assertTrue(failure.getMessage().contains(because), failure.getMessage());
  }

  /** 同じ方策どうしの三人打ちの決闘（1v2 ＋ 2v1）では、どちらのツモの順でも、games.tsv から読んだ値が全ての牌山で 0 になる。 */
  @Test(timeOut = 600_000)
  public void theSamePolicyReadsZeroOnEveryWallInBothDrawOrders() throws Exception {
    Path root = Files.createTempDirectory("wall-values-even");
    try (TableLoop table = table()) {
      for (DrawOrder draws : DrawOrder.values()) {
        DuelSetup setup = setup(Seating.EVERY_PLACEMENT, BOTH, draws, BASE, WALLS, 7, 7);
        Path home = duel(table, root.resolve(draws.name()), setup);
        WallValues values = WallValues.of(setup, Files.readAllLines(home.resolve(JobOutputs.GAMES)));
        assertEquals(values.walls(), WALLS, draws.name());
        double[] zero = new double[WALLS];
        assertBits(values.rank(), zero, draws + " rank");
        assertBits(values.utility(), zero, draws + " utility");
        DuelResult result = Jobs.result(home, DuelResult.class);
        assertEquals(result.rank().mean(), 0.0, draws.name());
      }
    } finally {
      DurableFiles.delete(root);
    }
  }

  /**
   * 候補・牌山の数・止め方・判定に使う値（順位か効用か）が違っても対にできる。相手・牌山・席の割り当て・ツモの順・順位ごとの効用の
   * どれかが違えば対にできない。
   */
  @Test
  public void plansPairOnlyWithTheSameOpponentWallsPlacementsDrawsAndUtility() {
    DuelSetup base = setup(Seating.EVERY_PLACEMENT, ONE, DrawOrder.SEAT_STREAMS, BASE, WALLS, 3, 5);
    DuelSetup other =
        DuelSetup.of(
            THREE, entrant(9), entrant(5), Seating.EVERY_PLACEMENT, List.of(Split.parse("1v2", 2, THREE)),
            DrawOrder.SEAT_STREAMS, WallSet.FIXED, SEED, BASE, 2048, new StopRule.Precision(0.005, 0.01, 512),
            Score.UTILITY, RankUtility.TENHOU);
    assertTrue(base.pairsWith(other));
    assertTrue(other.pairsWith(base));
    assertTrue(base.pairsWith(base));
    List<DuelSetup> apart =
        List.of(
            setup(Seating.EVERY_PLACEMENT, ONE, DrawOrder.SEAT_STREAMS, BASE, WALLS, 3, 6),
            setup(Seating.EVERY_PLACEMENT, BOTH, DrawOrder.SEAT_STREAMS, BASE, WALLS, 3, 5),
            setup(Seating.BY_GAME, ONE, DrawOrder.SEAT_STREAMS, BASE, WALLS, 3, 5),
            setup(Seating.EVERY_PLACEMENT, ONE, DrawOrder.WALL_ORDER, BASE, WALLS, 3, 5),
            setup(Seating.EVERY_PLACEMENT, ONE, DrawOrder.SEAT_STREAMS, BASE + 1, WALLS, 3, 5),
            DuelSetup.of(
                THREE, entrant(3), entrant(5), Seating.EVERY_PLACEMENT, List.of(Split.parse("1v2", 2, THREE)),
                DrawOrder.SEAT_STREAMS, WallSet.FIXED, SEED + 1, BASE, WALLS, new StopRule.Fixed(0.01), Score.RANK,
                RankUtility.TENHOU),
            DuelSetup.of(
                THREE, entrant(3), entrant(5), Seating.EVERY_PLACEMENT, List.of(Split.parse("1v2", 2, THREE)),
                DrawOrder.SEAT_STREAMS, WallSet.FRESH, SEED, BASE, WALLS, new StopRule.Fixed(0.01), Score.RANK,
                RankUtility.TENHOU),
            DuelSetup.of(
                THREE, entrant(3), entrant(5), Seating.EVERY_PLACEMENT, List.of(Split.parse("1v2", 2, THREE)),
                DrawOrder.SEAT_STREAMS, WallSet.FIXED, SEED, BASE, WALLS, new StopRule.Fixed(0.01), Score.RANK,
                RankUtility.AVERAGE_RANK));
    for (DuelSetup setup : apart) {
      assertFalse(base.pairsWith(setup), setup.toString());
      assertFalse(setup.pairsWith(base), setup.toString());
    }
  }
}
