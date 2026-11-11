package com.epsilon.agent.eval;

import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.PolicySeat;
import com.epsilon.agent.play.Query;
import com.epsilon.agent.play.Run;
import com.epsilon.agent.play.RunHooks;
import com.epsilon.agent.play.TableHooks;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TablePlan;
import com.epsilon.agent.play.TableSeat;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.util.SeedMixer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * 評価の卓。スロットごとの打ち手を、計画の割り当てどおりに席に置く。卓のコアで牌山を打ち、牌山ごとの結果を番号の順に流す。
 * 評価で牌山を打つのはこのクラスだけ。手は各席の打ち手の分布から引き（行動方策は使わない）、行も分岐も作らない。
 *
 * <p>牌山と局のキーは、牌山の種の素に {@link #EVALUATION_SALT} を混ぜた種と、牌山の番号だけで作る。割り当ての番号は乱数に入らない。
 * そのため、同じ牌山の全ての割り当てが同じ牌山を配り、同じ判断の時点では同じ乱数を引く。収集の牌山は run の種そのもので作るので、
 * 評価の牌山とは重ならない。
 */
public final class Arena {

  /** 評価の牌山の種に混ぜるソルトと、全てのバージョンで同じ牌山を使う卓の、最初の牌山の番号。 */
  public static final long EVALUATION_SALT = 0x4556414cL, WALL_BASE = 1L << 40;

  /** 全ての run とバージョンで同じ牌山で測る卓の、牌山の種の素。run の種に依らないので、系譜をまたいで同じ半荘を対にできる。 */
  public static final long EVALUATION_SEED = 20261010L;

  private Arena() {}

  /**
   * ジョブごとに新しい牌山で測る卓の、牌山の種の素。run の種とジョブの名前だけで決まるので、同じジョブを測り直しても同じ牌山になる。
   *
   * @param runSeed run の種
   * @param job ジョブの名前（{@code v00824-x1n3} など）
   */
  public static long freshSeed(long runSeed, String job) {
    long hash = 0xcbf29ce484222325L;
    for (byte b : job.getBytes(StandardCharsets.UTF_8)) {
      hash = (hash ^ (b & 0xff)) * 0x100000001b3L;
    }
    return SeedMixer.mix64(runSeed ^ SeedMixer.mix64(hash));
  }

  /**
   * 計画の牌山を打ち、牌山ごとの結果を受け取り先に流す。牌山は番号の順に始め、受け取り先が許す番号
   * （{@link WallSink#startBefore}）より前の牌山だけを始める。受け取り先は、牌山の番号の順に、一度に 1 つのスレッドから呼ぶ。
   * 受け取り先が false を返すか、打ち切りを求めたら（{@link WallSink#stopped}）、まだ始めていない牌山は始めない。
   * 打っている途中の半荘は打ち終えてから返し、その結果は捨てる。
   *
   * @param sink 牌山ごとの結果の受け取り先
   * @param watcher 主経路の観測。null なら観測しない
   * @return 卓のコアの結果
   * @throws IllegalStateException 打ち切らずに終わったのに、結果がそろわない牌山があった
   */
  public static TableLoop.Result play(TableLoop table, ArenaPlan plan, WallSink sink, GameObserver watcher) {
    int[][] slotOfSeat = plan.slotTable();
    List<TableSeat<Query>> slotSeats = new ArrayList<>(plan.slots().size());
    for (var policy : plan.slots()) {
      slotSeats.add(new PolicySeat(policy));
    }
    List<List<TableSeat<Query>>> seatsOf = new ArrayList<>(slotOfSeat.length);
    for (int[] slots : slotOfSeat) {
      List<TableSeat<Query>> row = new ArrayList<>(slots.length);
      for (int slot : slots) {
        row.add(slotSeats.get(slot));
      }
      seatsOf.add(List.copyOf(row));
    }
    int perWall = plan.gamesPerWall();
    List<GameKey> games = new ArrayList<>(plan.walls() * perWall);
    for (int index = 0; index < plan.walls(); index++) {
      long wall = plan.wallBase() + index;
      for (int game = 0; game < perWall; game++) {
        games.add(new GameKey(wall, plan.placement(wall, game)));
      }
    }
    Walls walls = new Walls(plan, sink, watcher, slotOfSeat);
    long seed = plan.wallSeed() ^ EVALUATION_SALT;
    TableLoop.Result played =
        table.play(
            new TablePlan<Query>(
                plan.variant(),
                seed,
                seed,
                games,
                plan.drawOrder(),
                null,
                game -> {
                  walls.start(game);
                  return seatsOf.get(game.placement());
                },
                walls),
            walls);
    walls.finish();
    return played;
  }

  /** 1 つの牌山の結果を集める入れ物。最後の半荘が終わるまで、ほかのスレッドは残りの数を減らすだけで、中身は読まない。 */
  private static final class Wall {
    /** [牌山の中の半荘][席] の最終の点。 */
    final int[][] scores;

    /** 牌山の中の半荘ごとの内訳。その半荘を進めるスレッドだけが書く。 */
    final PlayStats[] stats;

    /** まだ終わっていない半荘の数。 */
    final AtomicInteger remaining;

    Wall(int games, int seats, int slots) {
      scores = new int[games][seats];
      stats = new PlayStats[games];
      for (int game = 0; game < games; game++) {
        stats[game] = new PlayStats(slots);
      }
      remaining = new AtomicInteger(games);
    }
  }

  /**
   * 牌山ごとの結果を集めて、番号の順に受け取り先へ渡す。卓の観測と卓のフックを兼ね、始めてよい数と打ち切りを卓に伝える。
   * 牌山の入れ物は、その牌山の最初の半荘を始めるときに作り、受け取り先に渡したら捨てる。
   */
  private static final class Walls implements GameObserver, TableHooks<Query> {
    private final ArenaPlan plan;
    private final WallSink sink;
    private final GameObserver watcher;
    private final int[][] slotOfSeat;
    private final boolean oneGame;
    private final int perWall, seats, slots;
    private final AtomicReferenceArray<Wall> open;
    private final TableHooks<Query> plain = TableHooks.plain();

    private volatile boolean stopped;

    /** 次に受け取り先へ渡す牌山が、計画の何番目か。このオブジェクトのロックで守る。 */
    private int next;

    Walls(ArenaPlan plan, WallSink sink, GameObserver watcher, int[][] slotOfSeat) {
      this.plan = plan;
      this.sink = sink;
      this.watcher = watcher;
      this.slotOfSeat = slotOfSeat;
      oneGame = plan.seating() == Seating.BY_GAME;
      perWall = plan.gamesPerWall();
      seats = plan.variant().seats();
      slots = plan.slots().size();
      open = new AtomicReferenceArray<>(plan.walls());
    }

    /** 半荘を始める。牌山の入れ物が無ければ作る。 */
    void start(GameKey game) {
      int index = index(game);
      if (open.get(index) == null) {
        open.compareAndSet(index, null, new Wall(perWall, seats, slots));
      }
    }

    @Override
    public RunHooks<Query> main(Run<Query> run) {
      return plain.main(run);
    }

    @Override
    public boolean stopped() {
      return stopped || sink.stopped();
    }

    @Override
    public int startable() {
      long before = sink.startBefore();
      if (before <= plan.wallBase()) {
        return 0;
      }
      long walls = Math.min(before - plan.wallBase(), plan.walls());
      return (int) (walls * perWall);
    }

    @Override
    public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
      open.get(index(game)).stats[gameOf(game)].decided(slotOfSeat[game.placement()], view, decisions, slot, choice);
      if (watcher != null) {
        watcher.decided(game, view, decisions, slot, choice);
      }
    }

    @Override
    public void advanced(GameKey game, FullView full) {
      if (watcher != null) {
        watcher.advanced(game, full);
      }
    }

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      open.get(index(game)).stats[gameOf(game)].roundEnded(slotOfSeat[game.placement()], full, end);
      if (watcher != null) {
        watcher.roundEnded(game, full, end);
      }
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {
      if (watcher != null) {
        watcher.gameEnded(game, last);
      }
      Wall wall = open.get(index(game));
      int[] scores = wall.scores[gameOf(game)];
      for (int seat = 0; seat < seats; seat++) {
        scores[seat] = last.score(seat);
      }
      if (wall.remaining.decrementAndGet() == 0) {
        deliver();
      }
    }

    /** 結果がそろった牌山を、番号の順に受け取り先へ渡す。 */
    private synchronized void deliver() {
      while (!stopped() && next < plan.walls()) {
        Wall wall = open.get(next);
        if (wall == null || wall.remaining.get() != 0) {
          return;
        }
        boolean more = sink.accept(plan.wallBase() + next, wall.scores, wall.stats);
        open.set(next, null);
        next++;
        if (!more) {
          stopped = true;
        }
      }
    }

    /** 打ち切らずに終わったなら、全ての牌山を渡したことを確かめる。 */
    synchronized void finish() {
      if (!stopped() && next != plan.walls()) {
        throw new IllegalStateException("Missing result for wall " + (plan.wallBase() + next));
      }
    }

    private int index(GameKey game) {
      return (int) (game.wall() - plan.wallBase());
    }

    private int gameOf(GameKey game) {
      return oneGame ? 0 : game.placement();
    }
  }
}
