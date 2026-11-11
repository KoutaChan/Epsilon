package com.epsilon.agent.eval;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Policy;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 決闘の卓の対の差を確かめる。同じ方策どうしなら、どの分け方でも、どちらのツモの順でも、全ての牌山で 0 になる。鳴き・槓・抜きを
 * しない打ち手（ツモ切りだけ）は、2 つのツモの順で全く同じ半荘を打つ。
 */
public class DuelPairsTest {

  private static final long SEED = 31_337L, BASE = 9_000L;
  private static final int WALLS = 12;

  /**
   * 乱択の打ち手。判断の時点の観測（観測する席・局と本場・山の残り枚数・イベントの数・ツモ牌・点・手牌・合法手）から一様乱数を作る。
   * 確率 1/2 で最初の合法手（和了・槓・鳴き・立直が先に並ぶ）を選び、残りの 1/2 では合法手全体から一様に選ぶ。同じ判断の時点では同じ乱数を引く
   * （共通乱数）ので、割り当てや実行の順に依らない。
   */
  record CommonRandom(Variant variant, long salt) implements InlinePolicy {

    @Override
    public String label() {
      return "random:" + salt;
    }

    @Override
    public int act(Observation o) {
      long state = SeedMixer.mix64(salt);
      state = mix(state, o.observer());
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

  /** ツモ切りだけの打ち手。和了できれば和了し、それ以外はツモ切り（できなければ最初の打牌）か見送り。鳴き・槓・抜き・立直・九種九牌はしない。 */
  record Tsumogiri(Variant variant) implements InlinePolicy {

    @Override
    public String label() {
      return "tsumogiri";
    }

    @Override
    public int act(Observation o) {
      int n = o.legalActions();
      int win = -1, cut = -1, discard = -1, pass = -1;
      for (int a = 0; a < n; a++) {
        Action action = o.action(a);
        switch (action.type()) {
          case TSUMO_AGARI, RON_AGARI -> win = win < 0 ? a : win;
          case DAHAI -> {
            if (action.selection().tsumogiri() && cut < 0) cut = a;
            if (discard < 0) discard = a;
          }
          case PASS -> pass = a;
          default -> {}
        }
      }
      for (int choice : new int[] {win, cut, discard, pass}) {
        if (choice >= 0) return choice;
      }
      throw new IllegalStateException("No tsumogiri choice among " + n + " actions");
    }

    @Override
    public Evaluation evaluate(Observation o) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void close() {}
  }

  /** 半荘ごとの主経路の要約（判断と選んだ手、局末の点の移動、終局の点）と、鳴き・槓・抜きの数を集める。 */
  static final class Digests implements GameObserver {
    final Map<GameKey, long[]> games = new ConcurrentHashMap<>();
    final AtomicInteger calls = new AtomicInteger(), rounds = new AtomicInteger();

    private long[] of(GameKey game) {
      return games.computeIfAbsent(game, key -> new long[] {key.wall() * 131 + key.placement()});
    }

    private static void mix(long[] state, long value) {
      state[0] = SeedMixer.mix64(state[0] ^ SeedMixer.mix64(value));
    }

    @Override
    public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
      long[] state = of(game);
      int code = decisions.legal(slot, choice);
      mix(state, decisions.seat(slot) * 1_000_003L + code);
      switch (Action.of(code).type()) {
        case CHI, PON, DAIMINKAN, ANKAN, KAKAN, NUKI -> calls.incrementAndGet();
        default -> {}
      }
    }

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      long[] state = of(game);
      for (int seat = 0; seat < full.variant().seats(); seat++) mix(state, end.delta(seat));
      rounds.incrementAndGet();
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {
      long[] state = of(game);
      for (int seat = 0; seat < last.variant().seats(); seat++) mix(state, last.score(seat));
    }

    Map<GameKey, Long> snapshot() {
      Map<GameKey, Long> result = new TreeMap<>((a, b) -> a.wall() != b.wall() ? Long.compare(a.wall(), b.wall()) : Integer.compare(a.placement(), b.placement()));
      games.forEach((key, state) -> result.put(key, state[0]));
      return result;
    }
  }

  private static ArenaPlan plan(Variant variant, List<Policy> slots, Seating seating, List<Split> splits, DrawOrder order) {
    return new ArenaPlan(variant, slots, seating, splits, order, WallSet.FIXED, SEED, BASE, WALLS, RankUtility.TENHOU);
  }

  private static List<List<Split>> methods(Variant variant) {
    int seats = variant.seats();
    List<Split> both = List.of(Split.oneAgainstRest(variant), new Split(List.of(seats - 1, 1)));
    return seats == 4 ? List.of(both, List.of(new Split(List.of(2, 2)))) : List.of(both);
  }

  /**
   * 同じ方策の 2 つの打ち手をスロット 0 とスロット 1 に置く。全ての分け方と両方のツモの順で、全ての牌山の対の差（順位・効用、分け方ごとの値）が
   * 丸め誤差もなく 0 になることを確かめる。
   *
   * @param open 方策を開く関数（呼ぶたびに別の打ち手を返す）
   */
  static void assertZeroOnEveryWall(Function<Variant, Policy> open) {
    try (var table = new TableLoop(new TableSettings(16, 4, 1000L, 4))) {
      for (Variant variant : List.of(Variant.TENHOU_4P, Variant.TENHOU_3P)) {
        for (List<Split> splits : methods(variant)) {
          for (DrawOrder order : DrawOrder.values()) {
            ArenaPlan plan = plan(variant, List.of(open.apply(variant), open.apply(variant)), Seating.EVERY_PLACEMENT, splits, order);
            String label = variant + " " + splits + " " + order;
            PairScore score = new PairScore(plan);
            AtomicInteger walls = new AtomicInteger();
            TableLoop.Result result =
                Arena.play(
                    table,
                    plan,
                    (wall, scores, stats) -> {
                      Assert.assertEquals(wall, BASE + walls.getAndIncrement(), label);
                      score.measure(wall, scores);
                      Assert.assertEquals(score.rank(), 0.0, label + " rank of wall " + wall);
                      Assert.assertEquals(score.utility(), 0.0, label + " utility of wall " + wall);
                      for (int split = 0; split < splits.size(); split++) {
                        Assert.assertEquals(score.rank(split), 0.0, label + " split " + split + " of wall " + wall);
                        Assert.assertEquals(score.utility(split), 0.0, label + " split " + split + " of wall " + wall);
                      }
                      return true;
                    },
                    null);
            Assert.assertEquals(walls.get(), WALLS, label);
            Assert.assertEquals(result.games(), (long) WALLS * plan.gamesPerWall(), label);
          }
        }
      }
    }
  }

  @Test(timeOut = 600_000)
  public void theSameRandomPolicyScoresZeroOnEveryWall() {
    assertZeroOnEveryWall(variant -> new CommonRandom(variant, 7));
  }

  private static Map<GameKey, Long> digests(TableLoop table, ArenaPlan plan, Digests digests, Map<Long, int[]> scores) {
    Arena.play(
        table,
        plan,
        (wall, games, stats) -> {
          for (int[] game : games) {
            scores.put(wall, game.clone());
          }
          return true;
        },
        digests);
    return digests.snapshot();
  }

  /** ツモ切りだけの打ち手は、2 つのツモの順で同じ半荘になる。鳴く打ち手では半荘が分かれる。 */
  @Test(timeOut = 600_000)
  public void tsumogiriPlaysTheSameGamesInBothDrawOrders() {
    try (var table = new TableLoop(new TableSettings(16, 4, 1000L, 4))) {
      for (Variant variant : List.of(Variant.TENHOU_4P, Variant.TENHOU_3P)) {
        List<Split> one = List.of(Split.oneAgainstRest(variant));
        Map<DrawOrder, Map<GameKey, Long>> games = new TreeMap<>();
        Map<DrawOrder, Map<Long, int[]>> scores = new TreeMap<>();
        int rounds = 0;
        for (DrawOrder order : DrawOrder.values()) {
          Digests digests = new Digests();
          Map<Long, int[]> finals = new TreeMap<>();
          games.put(order, digests(table, plan(variant, List.of(new Tsumogiri(variant), new Tsumogiri(variant)), Seating.BY_GAME, one, order), digests, finals));
          scores.put(order, finals);
          Assert.assertEquals(digests.calls.get(), 0, variant + " " + order + " called");
          rounds += digests.rounds.get();
        }
        Assert.assertEquals(games.get(DrawOrder.WALL_ORDER).size(), WALLS);
        Assert.assertEquals(games.get(DrawOrder.WALL_ORDER), games.get(DrawOrder.SEAT_STREAMS), variant + " games");
        for (long wall : scores.get(DrawOrder.WALL_ORDER).keySet()) {
          Assert.assertEquals(scores.get(DrawOrder.WALL_ORDER).get(wall), scores.get(DrawOrder.SEAT_STREAMS).get(wall), variant + " wall " + wall);
        }
        Assert.assertTrue(rounds > 2 * WALLS * 2, variant + " rounds " + rounds);
        // 鳴く打ち手では、ツモの順で半荘が分かれる
        Map<DrawOrder, Map<GameKey, Long>> calling = new TreeMap<>();
        for (DrawOrder order : DrawOrder.values()) {
          Digests digests = new Digests();
          calling.put(order, digests(table, plan(variant, List.of(new CommonRandom(variant, 3), new CommonRandom(variant, 3)), Seating.BY_GAME, one, order), digests, new TreeMap<>()));
          Assert.assertTrue(digests.calls.get() > 0, variant + " " + order + " made no call");
        }
        Assert.assertNotEquals(calling.get(DrawOrder.WALL_ORDER), calling.get(DrawOrder.SEAT_STREAMS), variant + " calling games");
      }
    }
  }
}
