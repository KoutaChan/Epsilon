package com.epsilon.agent.play;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Supplier;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 卓のコア（{@link TableLoop}）のテスト。結果は種と打ち手だけで決まり、ワーカーの数、同時に進める半荘の数、打ち手がその場で答えるか
 * 束ねて答えるかには依らない。束は受付の区分ごとに分かれ、上限を超えない。分岐は局の終わりで止まり、同じ手を打ち直した分岐は主経路と
 * 同じ局末になる。フックは経路ごとに決まった順で、一度に 1 つのスレッドから呼ばれる。
 */
public class TableLoopTest {
  private static final int GAMES = 6;
  private static final long RUN_SEED = 77L, ITERATION_SEED = 991L, BASE = 5_000L;

  /** 束ねて答える打ち手が答えを返すスレッド。 */
  private static final ExecutorService ANSWERS = Executors.newFixedThreadPool(3, task -> {
    Thread thread = new Thread(task, "table-test-answer");
    thread.setDaemon(true);
    return thread;
  });

  @AfterClass(alwaysRun = true)
  public void stopAnswers() {
    ANSWERS.shutdownNow();
  }

  private static double unit(long bits) {
    return (bits >>> 11) * 0x1.0p-53;
  }

  /** 観測の合法手と席だけから、打ち手の分布（one-hot でない）と価値を書く。 */
  private static void answer(Query query) {
    int n = query.legalCount();
    float[] sigma = new float[n], scores = new float[n];
    for (int a = 0; a < n; a++) {
      long bits = SeedMixer.mix64(query.legal()[a] * 31L + query.seat());
      sigma[a] = (float) (0.05 + unit(bits) * unit(bits * 7));
      scores[a] = (float) (unit(bits ^ 0x5eed) - 0.5);
    }
    query.sigma = sigma;
    query.scores = scores;
  }

  /** 受付の区分。合法手が 6 つを超えれば 1、それ以外は 0。 */
  private static int admissionOf(int legalActions) {
    return legalActions > 6 ? 1 : 0;
  }

  /** 区分ごとの束の上限。区分 0 は 3 つ、区分 1 は 5 つまで。 */
  private static int limitOf(int admission) {
    return admission == 0 ? 3 : 5;
  }

  /** その場で答える席。 */
  static class InlineSeat implements TableSeat<Query> {
    @Override
    public boolean inline() {
      return true;
    }

    @Override
    public void evaluateInline(Query query) {
      answer(query);
    }

    @Override
    public int admission(Observation observation) {
      return admissionOf(observation.legalActions());
    }
  }

  /** 束ねて受け取り、別のスレッドで少し遅れて答える席。束の区分と大きさが守られているかを確かめる。 */
  static final class BatchedSeat implements TableSeat<Query> {
    final Queue<String> violations;
    final AtomicLong batches = new AtomicLong(), rows = new AtomicLong();
    final AtomicInteger largest = new AtomicInteger();

    BatchedSeat(Queue<String> violations) {
      this.violations = violations;
    }

    @Override
    public boolean inline() {
      return false;
    }

    @Override
    public int admission(Observation observation) {
      return admissionOf(observation.legalActions());
    }

    @Override
    public int maxBatch(int admission) {
      return limitOf(admission);
    }

    @Override
    public CompletableFuture<Void> submit(List<Query> batch, DecisionInferenceIngress ingress) {
      int admission = batch.get(0).admission();
      if (batch.size() > limitOf(admission)) violations.add("batch of " + batch.size() + " in class " + admission);
      for (Query query : batch) {
        if (query.admission() != admission) violations.add("classes " + admission + " and " + query.admission() + " in one batch");
        if (query.admission() != admissionOf(query.legalCount())) violations.add("class of " + query.legalCount() + " actions");
      }
      batches.incrementAndGet();
      rows.addAndGet(batch.size());
      largest.accumulateAndGet(batch.size(), Math::max);
      long delay = SeedMixer.mix64(batch.get(0).wall() * 131 + batch.size()) & 0xffff;
      return CompletableFuture.runAsync(
          () -> {
            LockSupport.parkNanos(delay);
            for (Query query : batch) answer(query);
          },
          ANSWERS);
    }
  }

  /** 半荘ごとの主経路の観測の要約。判断と選んだ手、局末の点数の移動、終局の点数をハッシュに混ぜる。 */
  static final class Digests implements GameObserver {
    final Map<Long, long[]> games = new ConcurrentHashMap<>();
    final Map<Long, long[]> deals = new ConcurrentHashMap<>();

    /** 主経路の判断ごとに観測が受け取った手の行動の番号（キーは半荘・判断 ID・判断の位置）。 */
    final Map<String, Integer> choices = new ConcurrentHashMap<>();

    private long[] of(long gameId) {
      return games.computeIfAbsent(gameId, key -> new long[] {key});
    }

    private static void mix(long[] state, long value) {
      state[0] = SeedMixer.mix64(state[0] ^ SeedMixer.mix64(value));
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {
      long[] state = of(game.wall());
      for (int seat = 0; seat < 4; seat++) mix(state, last.score(seat));
    }

    @Override
    public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
      long[] state = of(game.wall());
      mix(state, decisions.id(slot));
      mix(state, decisions.legal(slot, choice));
      choices.put(game.wall() + ":" + decisions.id(slot) + ":" + slot, decisions.legal(slot, choice));
    }

    @Override
    public void advanced(GameKey game, FullView full) {
      // 最初の配牌（牌山の要約）を控える
      deals.computeIfAbsent(
          game.wall(),
          key -> {
            long[] deal = {key};
            for (int seat = 0; seat < 4; seat++)
              for (int kind = 0; kind < 34; kind++) mix(deal, full.seat(seat).count(kind) * 64L + kind);
            mix(deal, full.seat(0).indicator(0));
            return deal;
          });
    }

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      long[] state = of(game.wall());
      for (int seat = 0; seat < 4; seat++) mix(state, end.delta(seat));
    }

    Map<Long, Long> snapshot() {
      Map<Long, Long> result = new TreeMap<>();
      games.forEach((id, state) -> result.put(id, state[0]));
      return result;
    }
  }

  /** 分岐のキー。半荘・局・判断・判断する席・差し替えの組で分岐を見分け、局末の点数の移動と局の始まりの卓をこのキーで控える。 */
  record BranchKey(long game, int hand, long decision, int slot, int alternative) implements Comparable<BranchKey> {
    @Override
    public int compareTo(BranchKey other) {
      return toString().compareTo(other.toString());
    }
  }

  /**
   * 経路ごとに、フックが決まった順で、一度に 1 つのスレッドから呼ばれるかを確かめる。主経路の判断からは分岐を作る。{@code remap} なら、
   * 引いた手のうち、手と席のハッシュで選んだ手を、合法手の次の手に替えて打つ（{@link RunHooks#play}）。
   */
  static final class Hooks implements TableHooks<Query> {
    final boolean remap;
    final Queue<String> violations = new ConcurrentLinkedQueue<>();

    /** 主経路の判断ごとに decided が受け取った手の行動の番号（キーは半荘・判断 ID・判断の位置）。 */
    final Map<String, Integer> mainChosen = new ConcurrentHashMap<>();

    final AtomicLong mainRemaps = new AtomicLong(), branchRemaps = new AtomicLong();
    final Map<BranchKey, List<Integer>> branches = new ConcurrentHashMap<>();
    final Map<String, List<Integer>> mainEnds = new ConcurrentHashMap<>();

    /**
     * 主経路の局ごとに、局末で読んだ局の始まりの卓と次局の卓を控える。キーは「半荘/局」。分岐の局の始まりの卓は、次のマップに
     * 分岐のキーで控える。
     */
    final Map<String, String> mainStarts = new ConcurrentHashMap<>(), mainNexts = new ConcurrentHashMap<>();
    final Map<BranchKey, String> branchStarts = new ConcurrentHashMap<>();
    final AtomicLong queries = new AtomicLong(), mainRuns = new AtomicLong(), forks = new AtomicLong();
    final AtomicLong settles = new AtomicLong(), mainRounds = new AtomicLong(), branchEnds = new AtomicLong();

    Hooks(boolean remap) {
      this.remap = remap;
    }

    @Override
    public RunHooks<Query> main(Run<Query> run) {
      mainRuns.incrementAndGet();
      if (!run.main()) violations.add("main() for a branch");
      try {
        run.handStart();
        violations.add("handStart() before the first deal at " + run);
      } catch (IllegalStateException expected) {
        // 最初の局を配る前は、局の始まりの卓が無い
      }
      return new Path(null);
    }

    /** 1 本の経路のフック。分岐なら分岐のキーを持ち、主経路ならキーは null。 */
    final class Path implements RunHooks<Query> {
      final BranchKey branch;
      final AtomicBoolean inside = new AtomicBoolean();
      boolean atBoundary, ended;
      int queried;

      /** 判断の位置ごとに、{@link #play} が返した手。 */
      final int[] played = new int[3];

      Path(BranchKey branch) {
        this.branch = branch;
      }

      private void enter(String hook, boolean driver) {
        if (!inside.compareAndSet(false, true)) violations.add(hook + " overlaps another hook of the run");
        boolean worker = Thread.currentThread().getName().startsWith("table-adv-");
        if (worker == driver) violations.add(hook + " on " + Thread.currentThread().getName());
      }

      private void exit() {
        inside.set(false);
      }

      @Override
      public void boundary(Run<Query> run, Decisions decisions, RuleWorkspace ws) {
        enter("boundary", false);
        if (atBoundary || ended) violations.add("boundary at " + run + " (open " + atBoundary + ", ended " + ended + ")");
        atBoundary = true;
        queried = 0;
        exit();
      }

      @Override
      public Query query(
          Run<Query> run,
          Decisions decisions,
          int slot,
          SeatView view,
          int admission,
          ObservationBatch records,
          int record,
          Observation observation) {
        enter("query", false);
        if (!atBoundary) violations.add("query before boundary at " + run);
        if (decisions.legalCount(slot) < 2) violations.add("query of a forced decision at " + run);
        if (observation.legalActions() != decisions.legalCount(slot)) violations.add("O and decisions disagree at " + run);
        queries.incrementAndGet();
        queried++;
        exit();
        return new Query(run, run.game().wall(), slot, admission, records, record, observation);
      }

      @Override
      public int play(Query query, int drawn) {
        enter("play", false);
        if (!atBoundary) violations.add("play outside a boundary");
        if (query.sigma == null) violations.add("play before the answer");
        int n = query.legalCount();
        int action =
            remap && (SeedMixer.mix64(query.legal()[drawn] * 13L + query.seat()) & 3) == 0 ? (drawn + 1) % n : drawn;
        if (action != drawn) (branch == null ? mainRemaps : branchRemaps).incrementAndGet();
        played[query.slot()] = action;
        exit();
        return action;
      }

      @Override
      public void decided(Run<Query> run, Decisions decisions, int[] chosen, RuleWorkspace ws) {
        enter("decided", false);
        if (!atBoundary) violations.add("decided before boundary at " + run);
        atBoundary = false;
        int asked = 0;
        for (int slot = 0; slot < decisions.size(); slot++) {
          boolean forced = decisions.legalCount(slot) == 1;
          Query query = run.query(slot);
          if (forced != (query == null)) violations.add("query of slot " + slot + " at " + run);
          if (branch == null)
            mainChosen.put(run.game().wall() + ":" + decisions.id(slot) + ":" + slot, decisions.legal(slot, chosen[slot]));
          if (query == null) continue;
          asked++;
          if (query.sigma == null) violations.add("decided before the answer at " + run);
          if (chosen[slot] < 0 || chosen[slot] >= decisions.legalCount(slot)) violations.add("choice out of range at " + run);
          if (chosen[slot] != played[slot]) violations.add("decided saw another action than play at " + run);
        }
        if (asked != queried) violations.add("decided saw " + asked + " queries of " + queried);
        if (run.main()) fork(run, decisions, chosen);
        exit();
      }

      /**
       * 判断の ID と牌山の番号のハッシュで選んだ席で、同じ手（差し替え 0）と別の手（差し替え 1）の分岐を作る。分岐する席は、1 つの判断の時点で
       * 1 つまで。
       */
      private void fork(Run<Query> run, Decisions decisions, int[] chosen) {
        for (int slot = 0; slot < decisions.size(); slot++) {
          int n = decisions.legalCount(slot);
          if (n < 2 || (SeedMixer.mix64(decisions.id(slot) * 977 + run.game().wall()) & 15) != 0) continue;
          for (int alternative = 0; alternative < 2; alternative++) {
            int[] choices = Arrays.copyOf(chosen, decisions.size());
            if (alternative == 1) choices[slot] = (chosen[slot] + 1) % n;
            var key = new BranchKey(run.game().wall(), run.handOrdinal(), decisions.id(slot), slot, alternative);
            forks.incrementAndGet();
            run.fork(choices, new Path(key));
          }
          return;
        }
      }

      @Override
      public void roundEnded(Run<Query> run, RoundEnd end, boolean gameEnded) {
        enter("roundEnded", false);
        if (atBoundary || ended) violations.add("roundEnded at " + run + " (open " + atBoundary + ", ended " + ended + ")");
        List<Integer> deltas = new ArrayList<>(4);
        for (int seat = 0; seat < 4; seat++) deltas.add(end.delta(seat));
        String start = table(run.handStart());
        if (branch == null) {
          mainRounds.incrementAndGet();
          String hand = run.game().wall() + "/" + run.handOrdinal();
          mainEnds.put(hand, deltas);
          mainStarts.put(hand, start);
          mainNexts.put(hand, table(end.next()));
          ended = gameEnded;
        } else {
          if (run.main()) violations.add("branch hooks on the main run");
          if (run.handOrdinal() != branch.hand()) violations.add("branch hand " + run.handOrdinal() + " of " + branch);
          branchStarts.put(branch, start);
          if (branches.put(branch, deltas) != null) violations.add("branch ended twice " + branch);
          branchEnds.incrementAndGet();
          ended = true;
        }
        exit();
      }

      @Override
      public void advanced(Run<Query> run, long nanos) {
        enter("advanced", false);
        if (nanos < 0) violations.add("negative advance");
        exit();
      }

      @Override
      public void settle(Run<Query> run) {
        enter("settle", true);
        settles.incrementAndGet();
        exit();
      }
    }
  }

  /** 卓の局・親・本場・供託・点数を / でつないだ文字列。 */
  private static String table(TableState table) {
    StringBuilder text =
        new StringBuilder().append(table.round()).append('/').append(table.dealer()).append('/').append(table.honba())
            .append('/').append(table.kyotaku());
    for (int seat = 0; seat < table.variant().seats(); seat++) text.append('/').append(table.score(seat));
    return text.toString();
  }

  /**
   * 局末で読んだ局の始まりの卓（{@link Run#handStart}）を確かめる。半荘の最初の局は東 1 局 0 本場・供託 0 で全員が同じ点数、ほかの局は
   * 前の局の局末に決まった次局の卓と同じ。分岐は親と同じ局の序数で、局の始まりの卓も親と同じ。
   */
  private static void assertHandStarts(Played played) {
    Hooks hooks = played.hooks();
    Assert.assertEquals(hooks.mainStarts.size(), hooks.mainEnds.size());
    for (var entry : hooks.mainStarts.entrySet()) {
      String[] key = entry.getKey().split("/");
      int hand = Integer.parseInt(key[1]);
      if (hand == 0) {
        String[] fields = entry.getValue().split("/");
        Assert.assertEquals(List.of(fields[0], fields[2], fields[3]), List.of("0", "0", "0"), entry.toString());
        for (int seat = 5; seat < fields.length; seat++) Assert.assertEquals(fields[seat], fields[4], entry.toString());
      } else {
        Assert.assertEquals(entry.getValue(), hooks.mainNexts.get(key[0] + "/" + (hand - 1)), entry.getKey());
      }
    }
    Assert.assertEquals(hooks.branchStarts.size(), hooks.branches.size());
    for (var entry : hooks.branchStarts.entrySet())
      Assert.assertEquals(
          entry.getValue(), hooks.mainStarts.get(entry.getKey().game() + "/" + entry.getKey().hand()), entry.toString());
  }

  /** 1 回の卓の結果と、フックと観測が控えたもの。 */
  record Played(TableLoop.Result result, Hooks hooks, Digests digests, List<BatchedSeat> batched) {}

  /**
   * @param inline 席ごとに、その場で答えるなら true
   */
  private static Played play(
      int workers, int activeGames, boolean[] inline, Behaviour behaviour, long runSeed, int games) {
    return play(workers, activeGames, inline, behaviour, runSeed, games, false);
  }

  /**
   * @param inline 席ごとに、その場で答えるなら true
   * @param remap 引いた手の一部を替えて打つなら true（{@link Hooks}）
   */
  private static Played play(
      int workers, int activeGames, boolean[] inline, Behaviour behaviour, long runSeed, int games, boolean remap) {
    Queue<String> violations = new ConcurrentLinkedQueue<>();
    List<BatchedSeat> batched = new ArrayList<>();
    List<TableSeat<Query>> seats = new ArrayList<>();
    for (boolean each : inline) {
      if (each) {
        seats.add(new InlineSeat());
      } else {
        var seat = new BatchedSeat(violations);
        batched.add(seat);
        seats.add(seat);
      }
    }
    List<Long> ids = new ArrayList<>();
    for (int game = 0; game < games; game++) ids.add(BASE + game);
    var hooks = new Hooks(remap);
    var digests = new Digests();
    var seating = List.copyOf(seats);
    TableLoop.Result result;
    try (var table = new TableLoop(new TableSettings(activeGames, workers, 200L, 4))) {
      result =
          table.play(
              new TablePlan<>(
                  TENHOU_4P,
                  runSeed,
                  ITERATION_SEED,
                  GameKey.onePerWall(ids),
                  DrawOrder.SEAT_STREAMS,
                  behaviour,
                  game -> seating,
                  digests), hooks);
    }
    hooks.violations.addAll(violations);
    return new Played(result, hooks, digests, batched);
  }

  private static final boolean[] INLINE = {true, true, true, true}, BATCHED = {false, false, false, false};
  private static final boolean[] MIXED = {true, false, true, false};

  private static void assertSame(Played actual, Played expected, String label) {
    Assert.assertEquals(actual.digests().snapshot(), expected.digests().snapshot(), label + " games");
    Assert.assertEquals(new TreeMap<>(actual.hooks().branches), new TreeMap<>(expected.hooks().branches), label + " branches");
    Assert.assertEquals(actual.result().games(), expected.result().games(), label);
    Assert.assertEquals(actual.result().rounds(), expected.result().rounds(), label);
    Assert.assertEquals(actual.result().queriedDecisions(), expected.result().queriedDecisions(), label);
    Assert.assertEquals(actual.result().forcedDecisions(), expected.result().forcedDecisions(), label);
  }

  private static void assertConsistent(Played played) {
    Hooks hooks = played.hooks();
    TableLoop.Result result = played.result();
    Assert.assertTrue(hooks.violations.isEmpty(), String.join("\n", hooks.violations));
    Assert.assertEquals(result.games(), GAMES);
    Assert.assertEquals(hooks.mainRuns.get(), GAMES);
    Assert.assertEquals(played.digests().snapshot().size(), GAMES);
    Assert.assertEquals(result.rounds(), hooks.mainRounds.get());
    Assert.assertTrue(result.rounds() >= GAMES, "rounds " + result.rounds());
    Assert.assertEquals(result.queriedDecisions(), hooks.queries.get());
    // 分岐は全て、局の終わりでちょうど 1 度止まる
    Assert.assertTrue(hooks.forks.get() > 0, "no forks");
    Assert.assertEquals(hooks.branchEnds.get(), hooks.forks.get());
    Assert.assertEquals(hooks.branches.size(), hooks.forks.get());
    Assert.assertTrue(hooks.settles.get() > 0, "no settle");
    for (BatchedSeat seat : played.batched()) {
      Assert.assertTrue(seat.batches.get() > 0, "a batched seat answered no batch");
      Assert.assertTrue(seat.largest.get() <= 5, "largest " + seat.largest.get());
    }
    long batchedRows = played.batched().stream().mapToLong(seat -> seat.rows.get()).sum();
    if (!played.batched().isEmpty()) {
      Assert.assertTrue(result.batches() > 0);
      Assert.assertTrue(result.largestBatch() <= 5, "largest " + result.largestBatch());
      long admitted = result.admission().admitted().values().stream().mapToLong(count -> count.rows()).sum();
      Assert.assertEquals(admitted, batchedRows);
      Assert.assertEquals(result.admission().batches(), result.batches());
    }
  }

  /** 同じ手を打ち直した分岐は、主経路と同じ局末になる（共通乱数と牌山を引き継ぐ）。別の手の分岐の少なくとも 1 つは局末が変わる。 */
  private static void assertReplaysMatchMain(Played played) {
    int replays = 0, changed = 0;
    for (var entry : played.hooks().branches.entrySet()) {
      BranchKey key = entry.getKey();
      List<Integer> main = played.hooks().mainEnds.get(key.game() + "/" + key.hand());
      Assert.assertNotNull(main, key.toString());
      if (key.alternative() == 0) {
        Assert.assertEquals(entry.getValue(), main, key.toString());
        replays++;
      } else if (!entry.getValue().equals(main)) {
        changed++;
      }
    }
    Assert.assertTrue(replays > 0);
    Assert.assertTrue(changed > 0, "no alternative changed the round end");
  }

  @Test(timeOut = 600_000)
  public void resultsDoNotDependOnWorkersOrBatching() {
    Played reference = play(1, 1, INLINE, null, RUN_SEED, GAMES);
    assertConsistent(reference);
    assertReplaysMatchMain(reference);
    assertHandStarts(reference);
    Played wide = play(4, 6, BATCHED, null, RUN_SEED, GAMES);
    assertConsistent(wide);
    assertHandStarts(wide);
    assertSame(wide, reference, "4 workers, batched");
    Played mixed = play(3, 16, MIXED, null, RUN_SEED, GAMES);
    assertConsistent(mixed);
    assertSame(mixed, reference, "3 workers, mixed");
    // 同じ設定で 2 回目を回しても同じ
    assertSame(play(4, 6, BATCHED, null, RUN_SEED, GAMES), reference, "again");
    // 種を変えると牌山も半荘も変わる
    Played other = play(2, 4, INLINE, null, RUN_SEED + 1, GAMES);
    assertConsistent(other);
    Map<Long, Long> first = reference.digests().snapshot(), second = other.digests().snapshot();
    for (long id : first.keySet()) {
      Assert.assertNotEquals(second.get(id), first.get(id), "game " + id);
      Assert.assertFalse(Arrays.equals(other.digests().deals.get(id), reference.digests().deals.get(id)), "deal " + id);
    }
  }

  /** 行動方策（一様分布を混ぜるもの、主経路で温度をかけて探索するもの）があっても、結果はワーカーの数と束ね方に依らない。 */
  @Test(timeOut = 600_000)
  public void behaviourDrawsDoNotDependOnWorkers() {
    for (Behaviour behaviour :
        List.of(new Behaviour(0.1, 0, CrnScheme.TWO_STAGE_FLOOR), new Behaviour(0.15, 0.5, CrnScheme.SINGLE_RACE))) {
      Played reference = play(1, 1, INLINE, behaviour, RUN_SEED, GAMES);
      assertConsistent(reference);
      if (behaviour.temperature() == 0) assertReplaysMatchMain(reference);
      Played wide = play(4, 8, MIXED, behaviour, RUN_SEED, GAMES);
      assertConsistent(wide);
      assertSame(wide, reference, behaviour.toString());
      Assert.assertNotEquals(reference.digests().snapshot(), play(1, 1, INLINE, null, RUN_SEED, GAMES).digests().snapshot());
    }
  }

  /**
   * 系列の処理が引いた手を替えて打つと（{@link RunHooks#play}）、decided と主経路の観測が受け取る手も、エンジンが進める手も替えた手になる。
   * 同じ手を打ち直した続きは主経路と同じ局末になる（続きの最初の手も、その後に引いた手も替える）。手を替えた半荘は、替えない半荘と違う。
   */
  @Test(timeOut = 600_000)
  public void runHooksChooseWhichDrawnActionIsPlayed() {
    Behaviour floor = new Behaviour(0.1, 0, CrnScheme.TWO_STAGE_FLOOR);
    Played remapped = play(2, 4, MIXED, floor, RUN_SEED, GAMES, true);
    assertConsistent(remapped);
    assertReplaysMatchMain(remapped);
    Hooks hooks = remapped.hooks();
    Assert.assertTrue(hooks.mainRemaps.get() > 0 && hooks.branchRemaps.get() > 0, hooks.mainRemaps + " " + hooks.branchRemaps);
    Assert.assertEquals(new TreeMap<>(remapped.digests().choices), new TreeMap<>(hooks.mainChosen));
    Assert.assertNotEquals(remapped.digests().snapshot(), play(2, 4, MIXED, floor, RUN_SEED, GAMES).digests().snapshot());
  }

  /** 打ち手が失敗すると卓を止め、同じ例外をそのまま投げる。卓はその後も使える。 */
  @Test(timeOut = 300_000)
  public void seatFailureStopsTheTableAndTheLoopStaysUsable() {
    var failure = new IllegalStateException("seat failed");
    var calls = new AtomicLong();
    TableSeat<Query> failing =
        new InlineSeat() {
          @Override
          public void evaluateInline(Query query) {
            if (calls.incrementAndGet() == 200) throw failure;
            super.evaluateInline(query);
          }
        };
    List<TableSeat<Query>> seats = List.of(failing, new InlineSeat(), new InlineSeat(), new InlineSeat());
    List<Long> ids = List.of(BASE, BASE + 1, BASE + 2);
    List<GameKey> games = GameKey.onePerWall(ids);
    try (var table = new TableLoop(new TableSettings(3, 2, 200L, 4))) {
      try {
        table.play(
            new TablePlan<>(TENHOU_4P, RUN_SEED, ITERATION_SEED, games, DrawOrder.SEAT_STREAMS, null, id -> seats, null),
            TableHooks.plain());
        Assert.fail("the failure did not stop the table");
      } catch (IllegalStateException thrown) {
        Assert.assertSame(thrown, failure);
      }
      List<TableSeat<Query>> healthy = List.of(new InlineSeat(), new InlineSeat(), new InlineSeat(), new InlineSeat());
      TableLoop.Result result =
          table.play(
              new TablePlan<>(TENHOU_4P, RUN_SEED, ITERATION_SEED, games, DrawOrder.SEAT_STREAMS, null, id -> healthy, null),
              TableHooks.plain());
      Assert.assertEquals(result.games(), ids.size());
    }
  }

  /** バッチで答える席が、その場で投げる前に予約をどう扱ったか。 */
  enum Reservation {
    /** 受け取らずに返した。 */
    ABORTED,
    /** 受け取ってから返した。 */
    RELEASED,
    /** 受け取って実行先へ渡した。 */
    MOVED,
    /** 触っていない。 */
    UNTOUCHED
  }

  /** バッチにその場で答える席。決めた回のバッチでは、予約を扱ってからその場で投げる。取った予約と返した予約を数える。 */
  static final class ThrowingSeat implements TableSeat<Query> {
    final Reservation reservation;
    final RuntimeException failure;
    final int failAt;
    int submits, acquired, released;

    ThrowingSeat(Reservation reservation, RuntimeException failure, int failAt) {
      this.reservation = reservation;
      this.failure = failure;
      this.failAt = failAt;
    }

    @Override
    public boolean inline() {
      return false;
    }

    @Override
    public int admission(Observation observation) {
      return 0;
    }

    @Override
    public DecisionInferenceIngress.Attempt tryAcquireIngress(int admission) {
      acquired++;
      return DecisionInferenceIngress.Attempt.acquired(new DecisionInferenceIngress(this, null, () -> released++));
    }

    @Override
    public CompletableFuture<Void> submit(List<Query> batch, DecisionInferenceIngress ingress) {
      if (++submits == failAt) {
        switch (reservation) {
          case ABORTED -> ingress.abort();
          case RELEASED -> ingress.handoff(this).release();
          case MOVED -> ingress.handoff(this).move();
          case UNTOUCHED -> {}
        }
        throw failure;
      }
      ingress.handoff(this).release();
      for (Query query : batch) answer(query);
      return CompletableFuture.completedFuture(null);
    }
  }

  /** 卓を別のスレッドで打ち、時間の内に戻らなければ失敗にする（戻らない卓にテストのスレッドを取られない）。 */
  private static <T> T within(int seconds, Supplier<T> body) {
    var result = new CompletableFuture<T>();
    Thread thread =
        new Thread(
            () -> {
              try {
                result.complete(body.get());
              } catch (Throwable failure) {
                result.completeExceptionally(failure);
              }
            },
            "table-test-driver");
    thread.setDaemon(true);
    thread.start();
    try {
      return result.get(seconds, TimeUnit.SECONDS);
    } catch (TimeoutException hung) {
      throw new AssertionError("The table did not return within " + seconds + " s");
    } catch (InterruptedException | ExecutionException failed) {
      throw new AssertionError(failed);
    }
  }

  @DataProvider
  public Object[][] reservations() {
    return Arrays.stream(Reservation.values()).map(each -> new Object[] {each}).toArray(Object[][]::new);
  }

  /**
   * バッチで答える席がその場で投げると、卓は止まって同じ例外のまま投げる。席が予約を返すか受け取った後でも、卓は待ち続けない。席が
   * 触っていない予約は卓が返す。卓はその後も使える。
   */
  @Test(dataProvider = "reservations")
  public void aSeatThatThrowsFromSubmitStopsTheTable(Reservation reservation) {
    var failure = new IllegalStateException("submit failed");
    var failing = new ThrowingSeat(reservation, failure, 20);
    List<TableSeat<Query>> seats = List.of(failing, new InlineSeat(), new InlineSeat(), new InlineSeat());
    List<Long> ids = List.of(BASE, BASE + 1, BASE + 2);
    List<GameKey> games = GameKey.onePerWall(ids);
    try (var table = new TableLoop(new TableSettings(3, 2, 200L, 4))) {
      Throwable thrown =
          within(
              120,
              () -> {
                try {
                  table.play(
                      new TablePlan<>(
                          TENHOU_4P, RUN_SEED, ITERATION_SEED, games, DrawOrder.SEAT_STREAMS, null, id -> seats, null),
                      TableHooks.plain());
                  return null;
                } catch (RuntimeException | Error stopped) {
                  return stopped;
                }
              });
      Assert.assertSame(thrown, failure);
      Assert.assertEquals(failing.released, failing.acquired - (reservation == Reservation.MOVED ? 1 : 0));
      List<TableSeat<Query>> healthy = List.of(new InlineSeat(), new InlineSeat(), new InlineSeat(), new InlineSeat());
      TableLoop.Result result =
          table.play(
              new TablePlan<>(TENHOU_4P, RUN_SEED, ITERATION_SEED, games, DrawOrder.SEAT_STREAMS, null, id -> healthy, null),
              TableHooks.plain());
      Assert.assertEquals(result.games(), ids.size());
    }
  }
}
