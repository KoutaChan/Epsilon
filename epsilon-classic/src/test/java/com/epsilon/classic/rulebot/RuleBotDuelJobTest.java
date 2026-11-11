package com.epsilon.classic.rulebot;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.eval.Duel;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.eval.Score;
import com.epsilon.agent.eval.Seating;
import com.epsilon.agent.eval.Sequential;
import com.epsilon.agent.eval.Split;
import com.epsilon.agent.eval.StopRule;
import com.epsilon.agent.eval.Stopped;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.eval.WallSet;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
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
import com.epsilon.mahjong.util.SeedMixer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.testng.annotations.Test;

/**
 * ルールベースの打ち手とランダムな打ち手の小さな決闘を、決闘のジョブ（{@link Duel}）として CPU で打つ。四人打ち・三人打ちとも、最初の
 * 判定の時点でルールベースの打ち手のほうが強いと決まり、記録にはその時点までの牌山だけが残る。ルールベースの打ち手どうしでは、どの牌山でも
 * 差が 0 で、最初の判定の時点で互角と決まる。
 */
public class RuleBotDuelJobTest {

  /** 決闘の設定で打ち手を指す名前。 */
  private static final String RULES = "heuristic", RANDOM = "random";

  /** 判定の止め方。牌山 16 個ごとに判定する（牌山 64 個で最大 4 回）。 */
  private static final StopRule TEST = new StopRule.Test(new Sequential(0.01, 3, 16), 0.01, 0.01);

  private static final int WALLS = 64, LOOK = 16;

  /**
   * ランダムな打ち手。判断の時点の記録から一様乱数を作り、確率 1/2 で最初の合法手（和了・槓・鳴き・立直が先に並ぶ）を選ぶ。それ以外は
   * 合法手から一様に選ぶ。同じ判断の時点では同じ値を引くので、選ぶ手は割り当てや実行の順に依らない。
   */
  record RandomPlayer(Variant variant) implements InlinePolicy {

    @Override
    public String label() {
      return RANDOM;
    }

    @Override
    public int act(Observation o) {
      long state = mix(SeedMixer.mix64(7), o.observer());
      state = mix(state, o.kyoku() * 64L + o.honba());
      state = mix(state, o.remainingWall());
      state = mix(state, o.events());
      state = mix(state, o.drawnTile());
      for (int seat = 0; seat < variant.seats(); seat++) {
        state = mix(state, o.score(seat));
      }
      for (int kind = 0; kind < 34; kind++) {
        state = mix(state, o.handCount(kind) * 64L + kind);
      }
      int n = o.legalActions();
      for (int a = 0; a < n; a++) {
        state = mix(state, o.action(a).code());
      }
      double u = (state >>> 11) * 0x1.0p-53;
      return u < 0.5 ? 0 : Math.min(n - 1, (int) ((u - 0.5) * 2 * n));
    }

    private static long mix(long state, long value) {
      return SeedMixer.mix64(state ^ SeedMixer.mix64(value));
    }

    @Override
    public Evaluation evaluate(Observation o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  /** ジョブの定義。 */
  record Table(Variant variant, String candidate, String opponent, List<String> splits) {}

  record Runner(TableLoop table) implements JobRunner {

    @Override
    public Node.Table plan(JobFile job) {
      Table t = RecordFile.read(Table.class, job.definition()).value();
      List<Split> splits = new ArrayList<>();
      for (String split : t.splits()) {
        splits.add(Split.parse(split, 2, t.variant()));
      }
      return RecordFile.tree(
          DuelSetup.of(
              t.variant(),
              new DuelSetup.Entrant(t.candidate(), t.candidate(), null, null),
              new DuelSetup.Entrant(t.opponent(), t.opponent(), null, null),
              Seating.EVERY_PLACEMENT,
              splits,
              DrawOrder.SEAT_STREAMS,
              WallSet.FRESH,
              1234,
              5000,
              WALLS,
              TEST,
              Score.RANK,
              RankUtility.TENHOU));
    }

    @Override
    public JobWork open(JobFile job, Path directory) throws IOException {
      DuelSetup setup = RecordFile.read(DuelSetup.class, job.plan()).value();
      return Duel.open(
          job.name(), table, setup, policy(setup.candidate(), setup.variant()), policy(setup.opponent(), setup.variant()),
          directory);
    }

    private static Policy policy(DuelSetup.Entrant entrant, Variant variant) {
      return entrant.spec().equals(RULES) ? new RuleBot(variant) : new RandomPlayer(variant);
    }
  }

  /**
   * @return 終わったジョブの結果
   */
  private static DuelResult duel(TableLoop table, Path root, Table definition, int games) throws Exception {
    Jobs jobs = new Jobs(root.resolve("jobs"), root.resolve("evaluations.jsonl"));
    JobFile job = JobFile.pending(1, "duel", 0, 1, RecordFile.tree(definition));
    jobs.create(job);
    assertTrue(jobs.run(job, new Runner(table), RunControl.detached(() -> false)).isEmpty());
    JobFile done = jobs.read(job.name());
    assertEquals(done.state(), JobFile.State.DONE, done.reason());
    Path home = root.resolve("jobs").resolve(job.name());
    DuelResult result = Jobs.result(home, DuelResult.class);
    assertEquals(result.stopped(), Stopped.DECIDED, result.summary());
    assertEquals(result.walls(), LOOK, result.summary());
    assertEquals(result.games(), (long) LOOK * games);
    assertEquals(Files.readAllLines(home.resolve(JobOutputs.GAMES)).size(), LOOK * games);
    assertEquals(Files.readAllLines(home.resolve(JobOutputs.LOOKS)).size(), 1);
    return result;
  }

  @Test(timeOut = 600_000)
  public void theRuleBotBeatsARandomPlayerAtTheFirstLook() throws Exception {
    Path root = Files.createTempDirectory("rulebot-duel");
    try (TableLoop table = new TableLoop(new TableSettings(64, 8, 1000L, 4))) {
      for (Variant variant : List.of(Variant.TENHOU_4P, Variant.TENHOU_3P)) {
        int seats = variant.seats();
        List<String> splits = List.of("1v" + (seats - 1), (seats - 1) + "v1");
        DuelResult result =
            duel(table, root.resolve(variant.name()), new Table(variant, RULES, RANDOM, splits), 2 * seats);
        String where = variant + " " + result.summary();
        assertEquals(result.verdict(), Verdict.BETTER, where);
        assertTrue(result.rank().high() < 0, where);
        assertTrue(result.utility().low() > 0, where);
        assertTrue(result.slots().get(0).meanRank() < (seats + 1) / 2.0, where);
        assertTrue(result.slots().get(1).meanRank() > (seats + 1) / 2.0, where);
        assertEquals(result.slots().get(0).shares().size(), seats, where);
        for (DuelResult.SplitValue split : result.splits()) {
          assertTrue(split.rank().mean() < 0, where + " " + split);
        }
      }
    } finally {
      DurableFiles.delete(root);
    }
  }

  @Test(timeOut = 600_000)
  public void theRuleBotAgainstItselfIsEvenAtTheFirstLook() throws Exception {
    Path root = Files.createTempDirectory("rulebot-even");
    try (TableLoop table = new TableLoop(new TableSettings(64, 8, 1000L, 4))) {
      DuelResult result = duel(table, root, new Table(Variant.TENHOU_4P, RULES, RULES, List.of("1v3", "3v1")), 8);
      assertEquals(result.verdict(), Verdict.EVEN, result.summary());
      assertEquals(result.rank().mean(), 0.0);
      assertEquals(result.rank().error(), 0.0);
      assertEquals(result.utility().mean(), 0.0);
      assertEquals(result.identical(), 1.0);
      assertEquals(result.roundSpread(), 0.0);
    } finally {
      DurableFiles.delete(root);
    }
  }
}
