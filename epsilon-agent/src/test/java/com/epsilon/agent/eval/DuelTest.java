package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.eval.DuelPairsTest.CommonRandom;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.Halt;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.JobRunner;
import com.epsilon.agent.run.JobWork;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.UnaryOperator;
import org.testng.annotations.Test;

/**
 * 決闘のジョブ（{@link Duel}）を確かめる。バッチの区切りで止めてから続けても、止めずに打ったときと同じ記録と結果になる（続きは計画に
 * 書いた種と臨界値を読む）。決着した判定の時点より後の牌山は、打ち終わっていても記録にも結果にも入らない。卓は、受け取り先が許す牌山より
 * 先を始めない。結果を書く前に落ちた決着済みのジョブは、打たずに同じ結果を返す。ほかに記録の往復と、計画と合わない記録を拒むことも見る。
 */
public class DuelTest {

  private static final long BASE = 5_000L;

  private static final DuelSetup.Entrant HEURISTIC = new DuelSetup.Entrant("heuristic", "heuristic", null, null);

  /**
   * ジョブの定義。打ち手は {@link CommonRandom} のソルトで書く。牌山の種は定義に書かず、計画を決める {@link Runner} が決める。
   *
   * @param candidate スロット 0 の打ち手のソルト
   * @param opponent スロット 1 の打ち手のソルト
   */
  record Table(
      Variant variant, long candidate, long opponent, Seating seating, List<String> splits, StopRule stop, int walls) {}

  /** 打とうとすると落ちる打ち手。打たずに済むはずの続きに置く。名前は置き換える前の打ち手と同じにする。 */
  record Unplayable(Variant variant, String label) implements InlinePolicy {

    @Override
    public int act(Observation o) {
      throw new AssertionError("The duel played although its record was decided");
    }

    @Override
    public Evaluation evaluate(Observation o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  /** 閉じられたかを覚えておく打ち手。 */
  record Closing(Variant variant, AtomicBoolean closed) implements InlinePolicy {

    @Override
    public String label() {
      return "closing";
    }

    @Override
    public int act(Observation o) {
      return 0;
    }

    @Override
    public Evaluation evaluate(Observation o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {
      closed.set(true);
    }
  }

  /**
   * 決闘のジョブを実際に動かす部分。
   *
   * @param seed 計画を決めるときに使う牌山の種。続きに別の種の {@link Runner} を置き、続きが記録した計画を読むかを確かめるのに使う
   * @param playable false なら、打つと落ちる打ち手を置く
   * @param edit 計画を書く前に、計画に手を加える関数
   */
  record Runner(TableLoop table, long seed, boolean playable, UnaryOperator<DuelSetup> edit) implements JobRunner {

    Runner(TableLoop table, long seed) {
      this(table, seed, true, UnaryOperator.identity());
    }

    @Override
    public Node.Table plan(JobFile job) {
      Table t = RecordFile.read(Table.class, job.definition()).value();
      List<Split> splits = new ArrayList<>();
      for (String split : t.splits()) {
        splits.add(Split.parse(split, 2, t.variant()));
      }
      return RecordFile.tree(
          edit.apply(
              DuelSetup.of(
                  t.variant(),
                  new DuelSetup.Entrant(Long.toString(t.candidate()), "random:" + t.candidate(), null, null),
                  new DuelSetup.Entrant(Long.toString(t.opponent()), "random:" + t.opponent(), null, null),
                  t.seating(),
                  splits,
                  DrawOrder.SEAT_STREAMS,
                  WallSet.FRESH,
                  seed,
                  BASE,
                  t.walls(),
                  t.stop(),
                  Score.RANK,
                  RankUtility.TENHOU)));
    }

    @Override
    public JobWork open(JobFile job, Path directory) throws IOException {
      DuelSetup setup = RecordFile.read(DuelSetup.class, job.plan()).value();
      return Duel.open(
          job.name(), table, setup, policy(setup.variant(), setup.candidate()), policy(setup.variant(), setup.opponent()),
          directory);
    }

    private Policy policy(Variant variant, DuelSetup.Entrant entrant) {
      return playable ? new CommonRandom(variant, Long.parseLong(entrant.spec())) : new Unplayable(variant, entrant.label());
    }
  }

  /** 1 つのジョブのディレクトリ。 */
  record Home(Jobs jobs, String name, Path directory) {

    JobFile job() throws IOException {
      return jobs.read(name);
    }

    List<String> games() throws IOException {
      return Files.readAllLines(directory.resolve(JobOutputs.GAMES));
    }

    List<String> looks() throws IOException {
      return Files.readAllLines(directory.resolve(JobOutputs.LOOKS));
    }

    DuelResult result() throws IOException {
      return Jobs.result(directory, DuelResult.class);
    }

    Path evaluations() {
      return directory.getParent().getParent().resolve("evaluations.jsonl");
    }
  }

  private static Home create(Path root, String table, Table definition) throws IOException {
    Jobs jobs = new Jobs(root.resolve("jobs"), root.resolve("evaluations.jsonl"));
    JobFile job = JobFile.pending(1, table, 0, 1, RecordFile.tree(definition));
    assertTrue(jobs.create(job));
    return new Home(jobs, job.name(), root.resolve("jobs").resolve(job.name()));
  }

  /**
   * ジョブを回す。止める牌山の数ごとに、そのバッチの後のセーフポイントで止め、記録の後にごみの行を足す（落ちたバッチの書きかけの行の
   * 代わり）。次は {@code continuation} で続け、最後は済むまで回す。
   */
  private static Home duel(Path root, String table, Table definition, Runner first, Runner continuation, int... stops)
      throws Exception {
    Home home = create(root, table, definition);
    Runner runner = first;
    for (int stop : stops) {
      RunControl control =
          RunControl.detached(
              () -> {
                try {
                  return home.job().walls() >= stop;
                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                }
              });
      Optional<Halt> halt = home.jobs().run(home.job(), runner, control);
      JobFile after = home.job();
      assertTrue(halt.isPresent(), table + " stops at " + stop);
      assertEquals(after.walls(), stop, table);
      assertEquals(after.state(), JobFile.State.PENDING, table);
      Files.writeString(home.directory().resolve(JobOutputs.GAMES), "garbage after the record\n", StandardOpenOption.APPEND);
      Files.writeString(home.directory().resolve(JobOutputs.LOOKS), "garbage after the record\n", StandardOpenOption.APPEND);
      runner = continuation;
    }
    assertTrue(home.jobs().run(home.job(), runner, RunControl.detached(() -> false)).isEmpty(), table);
    JobFile done = home.job();
    assertEquals(done.state(), JobFile.State.DONE, table + " " + done.reason());
    assertEquals(done.attempts(), stops.length + 1, table);
    assertEquals(done.cleanStops(), stops.length, table);
    return home;
  }

  private static DuelResult withoutSeconds(DuelResult r) {
    return new DuelResult(
        r.walls(), r.games(), r.verdict(), r.stopped(), r.score(), r.critical(), r.rank(), r.utility(), r.splits(),
        r.slots(), r.play(), r.identical(), r.roundSpread(), 0);
  }

  private static TableLoop table() {
    return new TableLoop(new TableSettings(64, 4, 1000L, 4));
  }

  /** 四人打ちの 1v3 ＋ 3v1 で、精度の止め方（目標が 0 なので計画の牌山を全て打つ）。32 牌山を 8 牌山ごとに判定する。 */
  private static Table precision() {
    return new Table(
        Variant.TENHOU_4P, 3, 5, Seating.EVERY_PLACEMENT, List.of("1v3", "3v1"), new StopRule.Precision(0, 0.01, 8), 32);
  }

  @Test(timeOut = 600_000)
  public void stoppingAndContinuingGivesTheSameRecordsAndResultFromThePlannedSeed() throws Exception {
    Path root = Files.createTempDirectory("duel-continue");
    try (TableLoop table = table()) {
      Home straight = duel(root.resolve("straight"), "straight", precision(), new Runner(table, 77), null);
      // 続きの Runner は別の種で計画を決める。ただし計画は始めたときに書いてあるので、続きはそれを読み、同じ牌山を打つ
      Home stopped =
          duel(root.resolve("stopped"), "stopped", precision(), new Runner(table, 77), new Runner(table, 78), 8, 24);
      Home other = duel(root.resolve("other"), "other", precision(), new Runner(table, 78), null);

      DuelResult expected = straight.result();
      assertEquals(expected.stopped(), Stopped.LIMIT);
      assertEquals(expected.walls(), 32);
      assertEquals(expected.games(), 32L * 8);
      assertEquals(straight.games().size(), 32 * 8);
      assertEquals(straight.looks().size(), 4);
      assertEquals(stopped.games(), straight.games());
      assertEquals(stopped.looks(), straight.looks());
      assertEquals(withoutSeconds(stopped.result()), withoutSeconds(expected));
      assertNotEquals(other.games(), straight.games(), "another wall seed plays other games");
      assertEquals(Files.readAllLines(stopped.evaluations()).size(), 1, "one report line");

      // 済んだジョブをもう一度回しても打たず、記録も報告の行も変わらない
      String games = Files.readString(stopped.directory().resolve(JobOutputs.GAMES), StandardCharsets.UTF_8);
      Runner idle = new Runner(table, 77, false, UnaryOperator.identity());
      assertTrue(stopped.jobs().run(stopped.job(), idle, RunControl.detached(() -> false)).isEmpty());
      assertEquals(Files.readString(stopped.directory().resolve(JobOutputs.GAMES), StandardCharsets.UTF_8), games);
      assertEquals(Files.readAllLines(stopped.evaluations()).size(), 1);
    } finally {
      DurableFiles.delete(root);
    }
  }

  /** 計画に書いた臨界値で判定する。計画を書いた後に臨界値の求め方が変わっても、始めた決闘は同じ境界で続ける。 */
  @Test(timeOut = 600_000)
  public void theDuelJudgesWithTheRecordedCriticalValues() throws Exception {
    Path root = Files.createTempDirectory("duel-critical");
    Table definition =
        new Table(
            Variant.TENHOU_4P, 3, 5, Seating.EVERY_PLACEMENT, List.of("1v3"),
            new StopRule.Test(new Sequential(0.01, 3, 8), 10, 10), 24);
    UnaryOperator<DuelSetup> wide =
        setup -> {
          List<DuelSetup.Look> looks = new ArrayList<>();
          for (DuelSetup.Look look : setup.looks()) {
            looks.add(new DuelSetup.Look(look.walls(), 1e9));
          }
          return new DuelSetup(
              setup.variant(), setup.candidate(), setup.opponent(), setup.seating(), setup.splits(), setup.draws(),
              setup.wallSet(), setup.wallSeed(), setup.wallBase(), setup.walls(), setup.stop(), setup.score(),
              setup.utility(), setup.chunk(), looks);
        };
    try (TableLoop table = table()) {
      DuelResult computed = duel(root.resolve("computed"), "computed", definition, new Runner(table, 9), null).result();
      assertEquals(computed.verdict(), Verdict.EVEN, "wide even margins decide at the first look");
      assertEquals(computed.walls(), 8);
      Home recorded =
          duel(root.resolve("recorded"), "recorded", definition, new Runner(table, 9, true, wide), new Runner(table, 9), 8);
      DuelResult result = recorded.result();
      assertEquals(result.verdict(), Verdict.UNDECIDED);
      assertEquals(result.stopped(), Stopped.LIMIT);
      assertEquals(result.walls(), 24);
      assertEquals(result.critical(), 1e9);
      assertEquals(recorded.looks().size(), 3);
    } finally {
      DurableFiles.delete(root);
    }
  }

  /**
   * 三人打ちの 1v2 ＋ 2v1 を判定の止め方で打ち、最初の判定の時点で決着した決闘は、その時点までの牌山だけで結果を出す。卓が先に始めた
   * 牌山は記録にも結果にも入らず、同じ牌山だけを固定の止め方で打った決闘と同じ記録・値になる。
   */
  @Test(timeOut = 600_000)
  public void wallsAfterTheDecidingLookNeverEnterTheResult() throws Exception {
    Path root = Files.createTempDirectory("duel-decided");
    Variant three = Variant.TENHOU_3P;
    List<String> splits = List.of("1v2", "2v1");
    Table decided =
        new Table(three, 3, 5, Seating.EVERY_PLACEMENT, splits, new StopRule.Test(new Sequential(0.01, 3, 8), 10, 10), 64);
    Table fixed = new Table(three, 3, 5, Seating.EVERY_PLACEMENT, splits, new StopRule.Fixed(0.01), 8);
    try (TableLoop table = table()) {
      Home home = duel(root.resolve("decided"), "decided", decided, new Runner(table, 11), null);
      Home only = duel(root.resolve("fixed"), "fixed", fixed, new Runner(table, 11), null);
      DuelResult result = home.result();
      assertEquals(result.verdict(), Verdict.EVEN);
      assertEquals(result.stopped(), Stopped.DECIDED);
      assertEquals(result.walls(), 8);
      assertEquals(result.games(), 8L * 6);
      assertEquals(home.job().walls(), 8);
      assertEquals(home.looks().size(), 1);
      assertTrue(home.looks().getFirst().contains("\tEVEN\tDECIDED"), home.looks().getFirst());
      List<String> games = home.games();
      assertEquals(games.size(), 8 * 6);
      for (String line : games) {
        assertTrue(Long.parseLong(line.split("\t", 2)[0]) < BASE + 8, line);
      }
      assertEquals(games, only.games());
      DuelResult expected = only.result();
      assertEquals(result.rank().mean(), expected.rank().mean());
      assertEquals(result.rank().error(), expected.rank().error());
      assertEquals(result.utility().mean(), expected.utility().mean());
      assertEquals(result.utility().error(), expected.utility().error());
      assertEquals(result.slots(), expected.slots());
      assertEquals(result.play(), expected.play());
      assertEquals(result.identical(), expected.identical());
      assertEquals(result.roundSpread(), expected.roundSpread());
      for (int split = 0; split < splits.size(); split++) {
        assertEquals(result.splits().get(split).rank().mean(), expected.splits().get(split).rank().mean());
      }
    } finally {
      DurableFiles.delete(root);
    }
  }

  /** 卓は、受け取り先が許す番号より前の牌山だけを始める。許す番号が上がるまで、その先は始めない。 */
  @Test(timeOut = 600_000)
  public void theTableStartsNoWallFromTheLimit() {
    Variant four = Variant.TENHOU_4P;
    ArenaPlan plan =
        new ArenaPlan(
            four, List.of(new CommonRandom(four, 3), new CommonRandom(four, 5)), Seating.EVERY_PLACEMENT,
            List.of(Split.oneAgainstRest(four)), DrawOrder.SEAT_STREAMS, WallSet.FRESH, 0, BASE, 24, RankUtility.TENHOU);
    try (TableLoop table = table()) {
      // 牌山を受け取るたびに、許す番号を受け取った牌山の番号より 4 大きくして全てを打つ。始めた牌山は、いつもその時点で許す番号より前にある
      AtomicLong limit = new AtomicLong(BASE + 3);
      AtomicLong late = new AtomicLong(-1);
      AtomicInteger accepted = new AtomicInteger();
      GameObserver watcher =
          new GameObserver() {
            @Override
            public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
              if (game.wall() >= limit.get()) {
                late.set(game.wall());
              }
            }

            @Override
            public void gameEnded(GameKey game, TableState last) {}
          };
      TableLoop.Result played =
          Arena.play(
              table,
              plan,
              new WallSink() {
                @Override
                public boolean accept(long wall, int[][] scores, PlayStats[] stats) {
                  accepted.incrementAndGet();
                  limit.set(wall + 1 + 3);
                  return true;
                }

                @Override
                public long startBefore() {
                  return limit.get();
                }
              },
              watcher);
      assertEquals(accepted.get(), 24);
      assertEquals(played.games(), 24L * 4);
      assertEquals(late.get(), -1L, "a wall started past the limit");

      // 許す番号を上げずに 5 牌山目で止めると、卓はその先を始めない
      AtomicLong highest = new AtomicLong(-1);
      AtomicInteger taken = new AtomicInteger();
      TableLoop.Result capped =
          Arena.play(
              table,
              plan,
              new WallSink() {
                @Override
                public boolean accept(long wall, int[][] scores, PlayStats[] stats) {
                  return taken.incrementAndGet() < 5;
                }

                @Override
                public long startBefore() {
                  return BASE + 5;
                }
              },
              new GameObserver() {
                @Override
                public void gameEnded(GameKey game, TableState last) {
                  highest.accumulateAndGet(game.wall(), Math::max);
                }
              });
      assertEquals(taken.get(), 5);
      assertEquals(capped.games(), 5L * 4);
      assertEquals(highest.get(), BASE + 4);
    }
  }

  /** 答えた判断の数を数える打ち手。 */
  record Counting(CommonRandom player, AtomicLong acts) implements InlinePolicy {

    @Override
    public Variant variant() {
      return player.variant();
    }

    @Override
    public String label() {
      return player.label();
    }

    @Override
    public int act(Observation o) {
      acts.incrementAndGet();
      return player.act(o);
    }

    @Override
    public Evaluation evaluate(Observation o) {
      return player.evaluate(o);
    }

    @Override
    public void close() {}
  }

  /** 数が 2 秒のあいだ変わらなくなるまで待つ。 @return 変わらなくなったときの数 */
  private static long settled(AtomicLong counter) throws InterruptedException {
    long last = counter.get();
    long since = System.nanoTime();
    while (System.nanoTime() - since < 2_000_000_000L) {
      Thread.sleep(20);
      long now = counter.get();
      if (now != last) {
        last = now;
        since = System.nanoTime();
      }
    }
    return last;
  }

  /**
   * ジョブのスレッドが次のチャンクを受け取らない間（run が一時停止している間）、卓は計画の終わりまでは打たず、チャンクを渡す所で待つ。
   * 受け取りを続けると、待たせなかったときと同じ記録になる。
   */
  @Test(timeOut = 600_000)
  public void theTableWaitsWhileNoChunkIsTaken() throws Exception {
    Path root = Files.createTempDirectory("duel-wait");
    Table definition =
        new Table(
            Variant.TENHOU_4P, 3, 5, Seating.EVERY_PLACEMENT, List.of("1v3", "3v1"), new StopRule.Precision(0, 0.01, 8), 64);
    try (TableLoop table = table()) {
      Home straight = duel(root.resolve("straight"), "straight", definition, new Runner(table, 77), null);
      Runner runner = new Runner(table, 77);
      AtomicLong acts = new AtomicLong();
      AtomicLong idle = new AtomicLong(-1);
      JobRunner waiting =
          new JobRunner() {
            @Override
            public Node.Table plan(JobFile job) {
              return runner.plan(job);
            }

            @Override
            public JobWork open(JobFile job, Path directory) throws IOException {
              DuelSetup setup = RecordFile.read(DuelSetup.class, job.plan()).value();
              Duel duel =
                  Duel.open(
                      job.name(), table, setup, new Counting(new CommonRandom(setup.variant(), 3), acts),
                      new Counting(new CommonRandom(setup.variant(), 5), acts), directory);
              return new JobWork() {
                @Override
                public Chunk next(long walls, JobOutputs outputs) throws Exception {
                  Chunk chunk = duel.next(walls, outputs);
                  if (idle.get() < 0) {
                    // 最初のチャンクの後、卓が判断しなくなるまで次のチャンクを受け取らない
                    idle.set(settled(acts));
                  }
                  return chunk;
                }

                @Override
                public void close() {
                  duel.close();
                }
              };
            }
          };
      Home home = create(root.resolve("waiting"), "waiting", definition);
      assertTrue(home.jobs().run(home.job(), waiting, RunControl.detached(() -> false)).isEmpty());
      assertEquals(home.job().state(), JobFile.State.DONE, home.job().reason());
      assertEquals(home.job().walls(), 64);
      assertTrue(
          idle.get() < acts.get(),
          "the table played the whole plan (" + acts.get() + " decisions) while no chunk was taken");
      assertEquals(home.games(), straight.games());
      assertEquals(home.looks(), straight.looks());
    } finally {
      DurableFiles.delete(root);
    }
  }

  /** 決着を games.tsv と job.json に書いた後、result.json を書く前に落ちたジョブは、続きで打たずに同じ結果を書く。 */
  @Test(timeOut = 600_000)
  public void aDecidedDuelThatFellBeforeItsResultReturnsItWithoutPlaying() throws Exception {
    Path root = Files.createTempDirectory("duel-fell");
    Table definition =
        new Table(
            Variant.TENHOU_4P, 3, 5, Seating.EVERY_PLACEMENT, List.of("1v3", "3v1"),
            new StopRule.Test(new Sequential(0.01, 3, 4), 10, 10), 16);
    try (TableLoop table = table()) {
      Home home = duel(root, "fell", definition, new Runner(table, 13), null);
      DuelResult expected = home.result();
      assertEquals(expected.stopped(), Stopped.DECIDED);
      JobFile done = home.job();
      DurableFiles.write(
          home.directory().resolve(Jobs.JOB),
          new JobFile(
              done.name(), done.version(), done.table(), done.order(), done.row(), done.definition(), done.plan(),
              JobFile.State.RUNNING, done.attempts(), done.cleanStops(), done.walls(), done.gamesBytes(),
              done.looksBytes(), done.created(), null));
      Files.delete(home.directory().resolve(Jobs.RESULT));
      Files.delete(home.evaluations());
      List<String> games = home.games();
      List<String> looks = home.looks();

      Runner idle = new Runner(table, 13, false, UnaryOperator.identity());
      assertTrue(home.jobs().run(home.job(), idle, RunControl.detached(() -> false)).isEmpty());
      JobFile again = home.job();
      assertEquals(again.state(), JobFile.State.DONE, again.reason());
      assertEquals(again.walls(), done.walls());
      assertEquals(home.games(), games);
      assertEquals(home.looks(), looks);
      DuelResult result = home.result();
      assertEquals(result.seconds(), 0.0);
      assertEquals(withoutSeconds(result), withoutSeconds(expected));
      assertEquals(Files.readAllLines(home.evaluations()).size(), 1);
    } finally {
      DurableFiles.delete(root);
    }
  }

  /** 系譜のバージョン 12 を候補にした計画。 */
  private static DuelSetup setup(StopRule stop, int walls, DuelSetup.Entrant opponent) {
    Variant four = Variant.TENHOU_4P;
    return DuelSetup.of(
        four,
        version("/runs/a", 12),
        opponent,
        Seating.EVERY_PLACEMENT,
        List.of(Split.oneAgainstRest(four), Split.parse("3v1", 2, four)),
        DrawOrder.WALL_ORDER,
        WallSet.FIXED,
        -7L,
        Arena.WALL_BASE,
        walls,
        stop,
        Score.UTILITY,
        RankUtility.TENHOU);
  }

  /** run の系譜のバージョン。 */
  private static DuelSetup.Entrant version(String run, int version) {
    return new DuelSetup.Entrant(
        String.format(Locale.ROOT, "prism:%s/%05d", run, version), "v" + version, version, null);
  }

  /** 計画は木と JSON を通して同じ値に戻る。系譜のバージョンには SHA-256 を書かず、系譜の外のチェックポイントにだけ書く。 */
  @Test
  public void theRecordsRoundTrip() throws Exception {
    DuelSetup.Entrant outside = new DuelSetup.Entrant("nano:/models/v760", "nano:v760", null, "ab".repeat(32));
    List<DuelSetup> setups =
        List.of(
            setup(new StopRule.Fixed(0.01), 12_288, outside),
            setup(new StopRule.Precision(0.002, 0.01, 512), 4096, HEURISTIC),
            setup(new StopRule.Test(new Sequential(0.01, 3, 512), 0.02, 0.01), 4096, version("/runs/a", 8)));
    assertTrue(RecordFile.json(setups.get(0)).contains("\"sha256\":\"" + outside.sha256() + "\""));
    assertFalse(RecordFile.json(setups.get(2)).contains("sha256"), RecordFile.json(setups.get(2)));
    for (DuelSetup setup : setups) {
      DuelSetup read = RecordFile.read(DuelSetup.class, RecordFile.json(setup), "test").value();
      assertEquals(read, setup);
      assertEquals(RecordFile.read(DuelSetup.class, RecordFile.tree(setup)).value(), setup);
      Stopping fresh = Stopping.of(setup.stop(), setup.walls());
      Stopping recorded = read.stopping();
      assertEquals(recorded.looks(), fresh.looks());
      for (int look = 0; look < fresh.looks(); look++) {
        assertEquals(recorded.wallsAt(look), fresh.wallsAt(look));
        assertEquals(recorded.critical(look), fresh.critical(look));
      }
    }
    assertEquals(setups.get(0).chunk(), DuelPlan.FIXED_CHUNK);
    assertEquals(setups.get(0).looks().size(), 1);
    assertEquals(setups.get(2).chunk(), 512);
    assertEquals(setups.get(2).looks().size(), 8);

    // 済んだ決闘の結果も、木と JSON を通して同じ値に戻る
    Path root = Files.createTempDirectory("duel-records");
    try (TableLoop table = table()) {
      DuelResult result = duel(root, "records", precision(), new Runner(table, 77), null).result();
      assertEquals(RecordFile.read(DuelResult.class, RecordFile.json(result), "test").value(), result);
    } finally {
      DurableFiles.delete(root);
    }
  }

  /** 打ち方を数えた列は、書いて読み直すと同じ数になる。 */
  @Test(timeOut = 600_000)
  public void playStatsColumnsRoundTrip() {
    Variant three = Variant.TENHOU_3P;
    ArenaPlan plan =
        new ArenaPlan(
            three, List.of(new CommonRandom(three, 3), new CommonRandom(three, 5)), Seating.EVERY_PLACEMENT,
            List.of(Split.oneAgainstRest(three)), DrawOrder.SEAT_STREAMS, WallSet.FRESH, 1, BASE, 4, RankUtility.TENHOU);
    PlayStats sum = new PlayStats(2);
    PlayStats read = new PlayStats(2);
    try (TableLoop table = table()) {
      Arena.play(
          table,
          plan,
          (wall, scores, stats) -> {
            for (PlayStats game : stats) {
              StringBuilder line = new StringBuilder();
              game.appendTo(line);
              String[] fields = line.substring(1).split("\t");
              assertEquals(fields.length, game.columns());
              long[] columns = new long[fields.length];
              for (int column = 0; column < columns.length; column++) {
                columns[column] = Long.parseLong(fields[column]);
              }
              read.addColumns(columns);
              sum.add(game);
            }
            return true;
          },
          null);
    }
    StringBuilder readLine = new StringBuilder();
    StringBuilder sumLine = new StringBuilder();
    read.appendTo(readLine);
    sum.appendTo(sumLine);
    assertEquals(readLine.toString(), sumLine.toString());
    assertTrue(sum.seatHands(0) > 0 && sum.rates(1).nuki() > 0, sumLine.toString());
    expectThrows(IllegalArgumentException.class, () -> read.addColumns(new long[read.columns() - 1]));
  }

  /**
   * 系譜のバージョンの相手は、指定とバージョンで見分ける。同じバージョンと打った計画どうしは、止め方や牌山の数が違っても対にし、別の
   * バージョンや別の run の同じ番号とは対にしない。系譜のバージョンに SHA-256 を添えた打ち手は作れない。
   */
  @Test
  public void aDuelAgainstOneVersionNeverPairsWithAnotherVersion() {
    DuelSetup against8 = setup(new StopRule.Fixed(0.01), 4096, version("/runs/a", 8));
    assertTrue(against8.pairsWith(setup(new StopRule.Precision(0.002, 0.01, 512), 2048, version("/runs/a", 8))));
    assertFalse(against8.pairsWith(setup(new StopRule.Fixed(0.01), 4096, version("/runs/a", 9))));
    assertFalse(against8.pairsWith(setup(new StopRule.Fixed(0.01), 4096, version("/runs/b", 8))));
    assertFalse(against8.pairsWith(setup(new StopRule.Fixed(0.01), 4096, HEURISTIC)));
    expectThrows(
        IllegalArgumentException.class, () -> new DuelSetup.Entrant("prism:/runs/a/00008", "v8", 8, "ab".repeat(32)));
  }

  @Test
  public void theSetupRejectsAPlanItsStopRuleWouldNotMake() {
    DuelSetup setup = setup(new StopRule.Test(new Sequential(0.01, 3, 512), 0.02, 0.01), 4096, HEURISTIC);
    List<DuelSetup.Look> shifted = new ArrayList<>(setup.looks());
    shifted.set(2, new DuelSetup.Look(shifted.get(2).walls() + 1, shifted.get(2).critical()));
    expectThrows(IllegalArgumentException.class, () -> withLooks(setup, setup.chunk(), shifted));
    expectThrows(IllegalArgumentException.class, () -> withLooks(setup, setup.chunk(), setup.looks().subList(0, 7)));
    expectThrows(IllegalArgumentException.class, () -> withLooks(setup, setup.chunk() * 2, setup.looks()));
    List<DuelSetup.Look> negative = new ArrayList<>(setup.looks());
    negative.set(0, new DuelSetup.Look(negative.get(0).walls(), -1));
    expectThrows(IllegalArgumentException.class, () -> withLooks(setup, setup.chunk(), negative));
    expectThrows(
        IllegalArgumentException.class,
        () -> Stopping.recorded(new StopRule.Precision(0, 0.01, 8), 32, new double[] {2.58, 2.58, Double.NaN, 2.58}));
    expectThrows(
        IllegalArgumentException.class, () -> Stopping.recorded(new StopRule.Fixed(0.01), 32, new double[] {2.58, 2.58}));
  }

  private static DuelSetup withLooks(DuelSetup s, int chunk, List<DuelSetup.Look> looks) {
    return new DuelSetup(
        s.variant(), s.candidate(), s.opponent(), s.seating(), s.splits(), s.draws(), s.wallSet(), s.wallSeed(),
        s.wallBase(), s.walls(), s.stop(), s.score(), s.utility(), chunk, looks);
  }

  /** 計画と合わない games.tsv・looks.tsv は、開くときに拒み、方策を閉じる。 */
  @Test(timeOut = 600_000)
  public void openingRejectsARecordThatDoesNotFitThePlan() throws Exception {
    Path root = Files.createTempDirectory("duel-broken");
    try (TableLoop table = table()) {
      Home home = duel(root, "broken", precision(), new Runner(table, 77), null);
      DuelSetup setup = RecordFile.read(DuelSetup.class, home.job().plan()).value();
      List<String> games = home.games();
      List<String> looks = home.looks();
      Path directory = home.directory();

      // 牌山の途中で切れた行
      expectRejected(table, setup, directory, games.subList(0, 8 * 8 - 3), looks.subList(0, 1));
      // 牌山の番号の違う行
      List<String> moved = new ArrayList<>(games.subList(0, 8 * 8));
      moved.set(5, moved.get(5).replaceFirst("^" + BASE + "\t", (BASE + 1) + "\t"));
      expectRejected(table, setup, directory, moved, looks.subList(0, 1));
      // 列の欠けた行
      List<String> cut = new ArrayList<>(games.subList(0, 8 * 8));
      cut.set(9, cut.get(9).substring(0, cut.get(9).lastIndexOf('\t')));
      expectRejected(table, setup, directory, cut, looks.subList(0, 1));
      // 判定の時点の行の数が牌山の数と合わない
      expectRejected(table, setup, directory, games.subList(0, 8 * 8), looks.subList(0, 2));
      // 決着の後にも続く行（判定の止め方では、決着した時点より後の牌山は書かない）
      DuelSetup decided =
          DuelSetup.of(
              setup.variant(), setup.candidate(), setup.opponent(), setup.seating(),
              List.of(Split.oneAgainstRest(setup.variant()), Split.parse("3v1", 2, setup.variant())), setup.draws(),
              setup.wallSet(), setup.wallSeed(), setup.wallBase(), setup.walls(),
              new StopRule.Test(new Sequential(0.01, 3, 8), 10, 10), setup.score(), setup.utility());
      expectRejected(table, decided, directory, games.subList(0, 16 * 8), looks.subList(0, 1));
    } finally {
      DurableFiles.delete(root);
    }
  }

  private static void expectRejected(
      TableLoop table, DuelSetup setup, Path directory, List<String> games, List<String> looks) throws IOException {
    Files.write(directory.resolve(JobOutputs.GAMES), games, StandardCharsets.UTF_8);
    Files.write(directory.resolve(JobOutputs.LOOKS), looks, StandardCharsets.UTF_8);
    AtomicBoolean candidate = new AtomicBoolean(), opponent = new AtomicBoolean();
    expectThrows(
        IllegalArgumentException.class,
        () ->
            Duel.open(
                "broken", table, setup, new Closing(setup.variant(), candidate), new Closing(setup.variant(), opponent),
                directory));
    assertTrue(candidate.get() && opponent.get(), "both policies are closed");
  }
}
