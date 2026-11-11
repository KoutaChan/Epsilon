package com.epsilon.agent.eval;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.BatchedPolicy;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 評価の卓を確かめる。牌山ごとに 1 半荘打つ卓では、候補は牌山の番号で決まる 1 席だけを打つ。牌山の結果は番号の順に届き、同じ種なら
 * 同じ半荘になる。観測のフックは全て呼ばれる。
 */
public class ArenaTest {
  private static final int GAMES = 8;
  private static final long SEED = 4_242L, BASE = 1_000L;

  /**
   * 1 半荘の結果。配列は席の順。
   *
   * @param slotOfSeat 席ごとに、座ったスロット
   * @param scores 最終の点
   */
  record Game(long wall, int placement, int[] slotOfSeat, int[] scores) {

    /** @return スロットが座った最初の席（座っていなければ −1） */
    int seatOf(int slot) {
      for (int seat = 0; seat < slotOfSeat.length; seat++) {
        if (slotOfSeat[seat] == slot) {
          return seat;
        }
      }
      return -1;
    }
  }

  /** 牌山ごとの結果を、半荘ごとに並べて集める（牌山の番号の順。同じ牌山の中は割り当ての順）。 */
  static final class Games implements WallSink {
    private final ArenaPlan plan;
    final List<Game> games = new ArrayList<>();

    Games(ArenaPlan plan) {
      this.plan = plan;
    }

    @Override
    public boolean accept(long wall, int[][] scores, PlayStats[] stats) {
      int[][] slotOfSeat = plan.slotTable();
      for (int game = 0; game < scores.length; game++) {
        int placement = plan.placement(wall, game);
        games.add(new Game(wall, placement, slotOfSeat[placement], scores[game].clone()));
      }
      return true;
    }
  }

  /** 手のコード・観測する席・ソルトから作ったハッシュが最大の手を打つ。 */
  record Hashed(long salt) implements InlinePolicy {

    @Override
    public Variant variant() {
      return TENHOU_4P;
    }

    @Override
    public String label() {
      return "hashed:" + salt;
    }

    @Override
    public int act(Observation observation) {
      int best = 0;
      double top = -1;
      for (int a = 0; a < observation.legalActions(); a++) {
        long bits = SeedMixer.mix64(observation.action(a).code() * 31L + observation.observer() + salt);
        double score = (bits >>> 11) * 0x1.0p-53;
        if (score > top) {
          top = score;
          best = a;
        }
      }
      return best;
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  /** 牌山ごとに 1 半荘打ち、候補の 1 席を牌山の番号で回す計画。 */
  static ArenaPlan byGame(Policy candidate, Policy opponent, int games, long seed, long base) {
    return new ArenaPlan(
        TENHOU_4P,
        List.of(candidate, opponent),
        Seating.BY_GAME,
        List.of(Split.oneAgainstRest(TENHOU_4P)),
        DrawOrder.SEAT_STREAMS,
        WallSet.FIXED,
        seed,
        base,
        games,
        RankUtility.TENHOU);
  }

  /** 観測のフックが呼ばれた回数と、合法手が 2 つ以上ある判断で、実際に選ばれた手がその席の打ち手の選ぶ手と食い違った回数を数える。 */
  static final class Seated implements GameObserver {
    private static final ThreadLocal<Tools> TOOLS = ThreadLocal.withInitial(Tools::new);

    private static final class Tools {
      final ObservationWriter writer = new ObservationWriter(TENHOU_4P);
      final ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
      final Observation reader = new Observation(TENHOU_4P);
    }

    final Hashed candidate, opponent;
    final AtomicLong games = new AtomicLong(), rounds = new AtomicLong(), decisions = new AtomicLong();
    final AtomicLong candidateDecisions = new AtomicLong(), mismatches = new AtomicLong();

    Seated(Hashed candidate, Hashed opponent) {
      this.candidate = candidate;
      this.opponent = opponent;
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {
      games.incrementAndGet();
    }

    @Override
    public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
      this.decisions.incrementAndGet();
      if (decisions.legalCount(slot) < 2) return;
      Tools tools = TOOLS.get();
      tools.batch.clear();
      int index = tools.writer.capture(view, decisions, slot, tools.batch);
      Observation o = tools.reader.bind(tools.batch, index);
      boolean candidateSeat = decisions.seat(slot) == Math.floorMod(game.wall(), TENHOU_4P.seats());
      if ((candidateSeat ? candidate : opponent).act(o) != choice) mismatches.incrementAndGet();
      if (candidateSeat) candidateDecisions.incrementAndGet();
    }

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      rounds.incrementAndGet();
    }
  }

  private static List<Game> play(Hashed candidate, Hashed opponent, GameObserver watcher) {
    return play(new TableSettings(4, 2, 1000L, 4), candidate, opponent, GAMES, watcher);
  }

  @Test(timeOut = 300_000)
  public void candidatePlaysOneRotatingSeat() {
    var candidate = new Hashed(1);
    var opponent = new Hashed(2);
    var watcher = new Seated(candidate, opponent);
    List<Game> games = play(candidate, opponent, watcher);
    Assert.assertEquals(games.size(), GAMES);
    for (int game = 0; game < GAMES; game++) {
      Game played = games.get(game);
      Assert.assertEquals(played.wall(), BASE + game);
      int seat = played.seatOf(0);
      Assert.assertEquals(seat, Math.floorMod(played.wall(), 4));
      Assert.assertEquals(played.placement(), seat);
      for (int other = 0; other < 4; other++) Assert.assertEquals(played.slotOfSeat()[other], other == seat ? 0 : 1);
    }
    // 候補は牌山の番号を席の数で割った余りの席だけ、相手はほかの席だけを打つ
    Assert.assertTrue(watcher.candidateDecisions.get() > 0);
    Assert.assertEquals(watcher.mismatches.get(), 0);
    Assert.assertEquals(watcher.games.get(), GAMES);
    Assert.assertTrue(watcher.rounds.get() >= GAMES, "rounds " + watcher.rounds.get());
    Assert.assertTrue(watcher.decisions.get() > watcher.rounds.get(), "decisions " + watcher.decisions.get());
    // 種と最初の牌山の番号が同じなら、同じ半荘になる
    List<Game> again = play(new Hashed(1), new Hashed(2), null);
    for (int game = 0; game < GAMES; game++) {
      Game first = games.get(game), second = again.get(game);
      Assert.assertEquals(second.wall(), first.wall());
      Assert.assertEquals(second.scores(), first.scores());
    }
  }

  /** 観測をバッチにまとめて答える打ち手。同じ観測には、中の {@link Hashed} と同じ手を返す。答えは別のスレッドで出す。 */
  record Batched(Hashed inner) implements BatchedPolicy {

    @Override
    public Variant variant() {
      return TENHOU_4P;
    }

    @Override
    public String label() {
      return inner.label();
    }

    @Override
    public int admission(Observation observation) {
      return observation.legalActions() > 5 ? 1 : 0;
    }

    @Override
    public int maxBatch(int admission) {
      return 3;
    }

    @Override
    public CompletableFuture<Void> choose(ObservationBatch batch, int[] chosen, DecisionInferenceIngress reservation) {
      var ownership = reservation.handoff(this);
      return CompletableFuture.runAsync(
          () -> {
            Observation reader = new Observation(TENHOU_4P);
            for (int row = 0; row < batch.size(); row++) chosen[row] = inner.act(reader.bind(batch, row));
            ownership.release();
          });
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  /** 半荘ごとの最初の配牌。牌山の要約として比べるのに使う。 */
  static final class Deals implements GameObserver {
    final Map<Long, List<Integer>> deals = new ConcurrentHashMap<>();

    @Override
    public void gameEnded(GameKey game, TableState last) {}

    @Override
    public void advanced(GameKey game, FullView full) {
      deals.computeIfAbsent(
          game.wall(),
          key -> {
            List<Integer> deal = new ArrayList<>(4 * 34 + 1);
            for (int seat = 0; seat < 4; seat++)
              for (int kind = 0; kind < 34; kind++) deal.add(full.seat(seat).count(kind));
            deal.add(full.seat(0).indicator(0));
            return deal;
          });
    }
  }

  private static List<Game> play(
      TableSettings settings, Policy candidate, Policy opponent, int games, GameObserver watcher) {
    try (var table = new TableLoop(settings)) {
      ArenaPlan plan = byGame(candidate, opponent, games, SEED, BASE);
      Games played = new Games(plan);
      Arena.play(table, plan, played, watcher);
      return played.games;
    }
  }

  private static void assertSameGames(List<Game> actual, List<Game> expected, String label) {
    Assert.assertEquals(actual.size(), expected.size(), label);
    for (int game = 0; game < expected.size(); game++) {
      Game first = expected.get(game), second = actual.get(game);
      Assert.assertEquals(second.wall(), first.wall(), label);
      Assert.assertEquals(second.slotOfSeat(), first.slotOfSeat(), label);
      Assert.assertEquals(second.scores(), first.scores(), label);
    }
  }

  /** 候補を替えても牌山と席は同じ。結果はワーカーの数と、打ち手がその場で答えるか束ねて答えるかに依らない。 */
  @Test(timeOut = 300_000)
  public void wallsAndResultsDoNotDependOnTheCandidateOrTheTable() {
    int games = 8;
    var deals = new Deals();
    List<Game> reference = play(new TableSettings(1, 1, 1000L, 4), new Hashed(1), new Hashed(2), games, deals);
    var otherDeals = new Deals();
    List<Game> other =
        play(new TableSettings(4, 2, 1000L, 4), new Hashed(9), new Hashed(2), games, otherDeals);
    Assert.assertEquals(deals.deals.size(), games);
    Assert.assertEquals(otherDeals.deals, deals.deals);
    boolean differs = false;
    for (int game = 0; game < games; game++) {
      Assert.assertEquals(other.get(game).slotOfSeat(), reference.get(game).slotOfSeat());
      differs |= !Arrays.equals(other.get(game).scores(), reference.get(game).scores());
    }
    Assert.assertTrue(differs, "another candidate played the same games");
    assertSameGames(
        play(new TableSettings(8, 4, 1000L, 4), new Hashed(1), new Hashed(2), games, null),
        reference,
        "8 games, 4 workers");
    assertSameGames(
        play(new TableSettings(6, 3, 200L, 4), new Batched(new Hashed(1)), new Hashed(2), games, null),
        reference,
        "batched candidate");
    assertSameGames(
        play(new TableSettings(6, 3, 200L, 4), new Batched(new Hashed(1)), new Batched(new Hashed(2)), games, null),
        reference,
        "batched seats");
  }
}
