package com.epsilon.agent.eval;

import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.JobWork;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 決闘のジョブを続きから打つ。計画の牌山の結果を評価の卓（{@link Arena#play}）から番号の順に受け取り、判定の時点ごとに止めるかを
 * 決める。チャンクごとに、半荘の行（games.tsv）と判定の時点の行（looks.tsv）を渡す。行の列の形は {@link DuelTally} に書いた。
 *
 * <p>卓は別スレッドで、打ち終えた牌山の後ろから計画の終わりまでを 1 回で打つ（チャンクの区切りで卓を空にしない）。止め方が判定か精度
 * なら、まだ決めていない次の判定の時点から判定の間隔の半分より先の牌山は始めない。止めると決めたら、それより後の牌山は打ち終わって
 * いても捨てる。ジョブのスレッドがまだ受け取っていないチャンクが 1 つあるときは、卓は次のチャンクを渡す所で、受け取られるか閉じられる
 * まで待つ（その間は牌山の結果を受け取らず、走っている半荘が終わると卓は止まる）。
 *
 * <p>続きから打つときは、games.tsv にある打ち終えた牌山の行を読み直し、牌山ごとの値を同じ順に足し直す。乱数は牌山の種と番号だけから
 * 作るので、止めた所から続けても止めずに打ったのと同じ半荘になり、同じ値で判定する。打ち終えた牌山で既に決着していれば（結果を書く
 * 前に落ちたとき）、打たずに同じ結果を返す。
 */
public final class Duel implements JobWork {

  private static final Logger LOG = LoggerFactory.getLogger(Duel.class);

  private final String name;
  private final TableLoop table;
  private final DuelPlan plan;
  private final Stopping stopping;
  private final int chunk;
  private final List<Policy> policies;
  private final DuelTally tally;
  private final Sink sink = new Sink();

  /** 卓が渡して、ジョブのスレッドがまだ受け取っていないチャンク（1 つまで）。 */
  private final BlockingQueue<Batch> batches = new LinkedBlockingQueue<>(1);

  /** 打ち終えた牌山で既に決着していたときの結果（最初のチャンクとして返す）。 */
  private Batch recorded;

  private Thread player;

  /** 渡し終えた行に含まれる、打ち終えた牌山の数。 */
  private long walls;

  private Duel(String name, TableLoop table, DuelPlan plan, Stopping stopping, int chunk, List<Policy> policies) {
    this.name = name;
    this.table = table;
    this.plan = plan;
    this.stopping = stopping;
    this.chunk = chunk;
    this.policies = policies;
    tally = new DuelTally(plan);
  }

  /**
   * ジョブを続きから開く。games.tsv と looks.tsv は、打ち終えた牌山までの長さに切り詰めておくこと。
   *
   * @param name ジョブの名前（ログに出す）
   * @param table 卓のコア（この決闘を閉じるまで、ほかで使わない）
   * @param setup 計画の記録
   * @param candidate スロット 0 の方策（閉じるのはこの決闘。開けなかったときはここで閉じる）
   * @param opponent スロット 1 の方策（閉じ方はスロット 0 の方策と同じ）
   * @param directory ジョブのディレクトリ
   * @throws IOException games.tsv か looks.tsv が読めないとき
   * @throws IllegalArgumentException 記録が計画と合わないとき
   */
  public static Duel open(
      String name, TableLoop table, DuelSetup setup, Policy candidate, Policy opponent, Path directory)
      throws IOException {
    List<Policy> policies = List.of(candidate, opponent);
    try {
      DuelPlan plan = setup.plan(candidate, opponent);
      Duel duel = new Duel(name, table, plan, setup.stopping(), setup.chunk(), policies);
      duel.replay(
          DurableFiles.lines(directory.resolve(JobOutputs.GAMES)),
          DurableFiles.lines(directory.resolve(JobOutputs.LOOKS)).size());
      return duel;
    } catch (IOException | RuntimeException | Error failure) {
      closeAll(policies, failure);
      throw failure;
    }
  }

  /** 打ち終えた牌山の行を足し直し、判定の時点をそこまで進める。 */
  private void replay(List<String> games, int looks) {
    ArenaPlan arena = plan.arena();
    int perWall = arena.gamesPerWall();
    if (games.size() % perWall != 0 || games.size() / perWall > arena.walls()) {
      throw new IllegalArgumentException(
          "games.tsv of " + name + " holds " + games.size() + " games, not whole walls of " + perWall + " games within "
              + arena.walls() + " walls");
    }
    int done = games.size() / perWall;
    int[][] scores = new int[perWall][arena.variant().seats()];
    PlayStats[] stats = new PlayStats[perWall];
    for (int game = 0; game < perWall; game++) {
      stats[game] = new PlayStats(arena.slots().size());
    }
    int look = 0;
    for (int index = 0; index < done; index++) {
      tally.add(tally.parse(games, index, scores, stats), scores, stats);
      if (look < stopping.looks() && tally.walls() == stopping.wallsAt(look)) {
        Optional<Stopping.Decision> decision = tally.decide(stopping, look);
        if (decision.isPresent()) {
          if (index + 1 != done) {
            throw new IllegalArgumentException(
                "games.tsv of " + name + " goes on after the duel was decided at look " + look);
          }
          recorded = new Batch(done, List.of(), List.of(), tally.result(decision.get(), stopping.critical(look), 0), null);
        }
        look++;
      }
    }
    if (looks != look) {
      throw new IllegalArgumentException(
          "looks.tsv of " + name + " holds " + looks + " looks, but its " + done + " walls reach " + look);
    }
    walls = done;
    sink.look = look;
    sink.chunkEnd = Math.min(arena.walls(), (done / chunk + 1L) * chunk);
    sink.startBefore = startLimit(look);
  }

  @Override
  public Chunk next(long done, JobOutputs outputs) throws Exception {
    if (done != walls) {
      throw new IllegalStateException("Duel " + name + " has written " + walls + " walls, the job " + done);
    }
    Batch batch;
    if (recorded != null) {
      batch = recorded;
      recorded = null;
    } else {
      if (player == null) {
        start();
      }
      batch = batches.take();
    }
    if (batch.failure() != null) {
      throw thrown(batch.failure());
    }
    for (String line : batch.games()) {
      outputs.game(line);
    }
    for (String line : batch.looks()) {
      outputs.look(line);
    }
    walls = batch.walls();
    if (batch.result() == null) {
      return Chunk.more(walls);
    }
    LOG.info("{} {}", name, batch.result().summary());
    return Chunk.decided(walls, batch.result());
  }

  /** 卓を止めて別スレッドの終わりを待ち（走っている半荘は打ち終えてから捨てる）、スロットの方策を閉じる。 */
  @Override
  public void close() {
    sink.closing = true;
    // チャンクを渡す所で待っている卓を進める（閉じ始めた後に卓が渡すチャンクは 1 つまでなので、空けた後は待たない）
    batches.clear();
    boolean interrupted = false;
    if (player != null) {
      while (true) {
        try {
          player.join();
          break;
        } catch (InterruptedException again) {
          interrupted = true;
        }
      }
    }
    try {
      closeAll(policies, null);
    } finally {
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
    }
  }

  /** 打ち終えた牌山の後ろから計画の終わりまでを、別スレッドで打ち始める。 */
  private void start() {
    ArenaPlan rest = plan.arena().rest(Math.toIntExact(walls));
    sink.started = System.nanoTime();
    player = new Thread(() -> play(rest), "duel-" + name);
    player.setDaemon(true);
    player.start();
  }

  private void play(ArenaPlan rest) {
    try {
      TableLoop.Result played = Arena.play(table, rest, sink, null);
      LOG.info(
          "{} table played {} games after {} recorded walls in {}s",
          name,
          played.games(),
          rest.wallBase() - plan.arena().wallBase(),
          String.format(Locale.ROOT, "%.1f", played.seconds()));
      if (!sink.finished && !sink.closing) {
        hand(failed(new IllegalStateException("The table of " + name + " ended before the duel was decided")));
      }
    } catch (Throwable failure) {
      if (sink.finished || sink.closing) {
        LOG.warn("The table of {} failed after the duel stopped", name, failure);
      } else {
        hand(failed(failure));
      }
    }
  }

  /**
   * チャンクをジョブのスレッドへ渡す。まだ受け取られていないチャンクがあれば、受け取られるか閉じられるまで待つ。
   *
   * @throws IllegalStateException 待っている間に割り込まれたとき
   */
  private void hand(Batch batch) {
    try {
      batches.put(batch);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("The table of " + name + " was interrupted handing over a chunk", interrupted);
    }
  }

  /**
   * @param look まだ決めていない次の判定の時点
   * @return まだ始めない最初の牌山の番号。止め方が判定か精度なら、その時点の牌山の数に判定の間隔の半分を足した所（計画の終わりを
   *     超えない）。止め方が固定か、判定の時点が残っていなければ {@code Long.MAX_VALUE}
   */
  private long startLimit(int look) {
    ArenaPlan arena = plan.arena();
    if (plan.stop() instanceof StopRule.Fixed || look >= stopping.looks()) {
      return Long.MAX_VALUE;
    }
    long ahead = stopping.wallsAt(look) + plan.stop().checkEvery(arena.walls()) / 2;
    return arena.wallBase() + Math.min(arena.walls(), ahead);
  }

  /** 卓から牌山の結果を受け取って判定する。{@link #accept} は卓の進行のスレッドから、1 度に 1 つずつ呼ばれる。 */
  private final class Sink implements WallSink {

    private volatile long startBefore;
    private volatile boolean closing;
    private volatile boolean finished;
    private volatile long started;

    /** まだ決めていない次の判定の時点と、今のチャンクが終わる牌山の数。 */
    private int look;

    private long chunkEnd;
    private List<String> games = new ArrayList<>();
    private List<String> looks = new ArrayList<>();

    @Override
    public boolean accept(long wall, int[][] scores, PlayStats[] stats) {
      if (closing || finished) {
        return false;
      }
      tally.lines(wall, scores, stats, games);
      tally.add(wall, scores, stats);
      int done = tally.walls();
      DuelResult result = null;
      if (look < stopping.looks() && done == stopping.wallsAt(look)) {
        Optional<Stopping.Decision> decision = tally.decide(stopping, look);
        looks.add(tally.look(stopping, look, decision));
        LOG.info(
            "{} look {}/{} after {} walls: {}",
            name,
            look + 1,
            stopping.looks(),
            done,
            decision.map(decided -> decided.verdict() + " " + decided.stopped()).orElse(DuelTally.CONTINUE));
        if (decision.isPresent()) {
          result = tally.result(decision.get(), stopping.critical(look), (System.nanoTime() - started) / 1e9);
          finished = true;
        } else {
          look++;
          startBefore = startLimit(look);
        }
      }
      if (result != null || done == chunkEnd) {
        hand(new Batch(done, games, looks, result, null));
        games = new ArrayList<>();
        looks = new ArrayList<>();
        chunkEnd = Math.min(plan.arena().walls(), chunkEnd + chunk);
      }
      return result == null;
    }

    @Override
    public long startBefore() {
      return startBefore;
    }

    @Override
    public boolean stopped() {
      return closing;
    }
  }

  /**
   * チャンク 1 つの行と、決着したならその結果。
   *
   * @param walls このチャンクまでで打ち終えた牌山の数
   * @param result 決着したときの結果（続けるなら null）
   * @param failure 卓の失敗（あるときは、ほかのフィールドを使わない）
   */
  private record Batch(long walls, List<String> games, List<String> looks, DuelResult result, Throwable failure) {}

  private static Batch failed(Throwable failure) {
    return new Batch(0, List.of(), List.of(), null, failure);
  }

  /** 別スレッドの失敗を、このスレッドで投げ直せる形にする。 */
  private static Exception thrown(Throwable failure) {
    if (failure instanceof Error error) {
      throw error;
    }
    return failure instanceof Exception exception ? exception : new IllegalStateException(failure);
  }

  /** 方策を全て閉じる。閉じるときの失敗は、先の失敗があればそれに添える。無ければ最初の失敗を投げ、2 つ目からはそれに添える。 */
  private static void closeAll(List<Policy> policies, Throwable failure) {
    RuntimeException first = null;
    for (Policy policy : policies) {
      try {
        policy.close();
      } catch (RuntimeException closing) {
        if (failure != null) {
          failure.addSuppressed(closing);
        } else if (first == null) {
          first = closing;
        } else {
          first.addSuppressed(closing);
        }
      }
    }
    if (first != null) {
      throw first;
    }
  }
}
