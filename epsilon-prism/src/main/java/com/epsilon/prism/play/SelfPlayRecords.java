package com.epsilon.prism.play;

import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.PolicySeat;
import com.epsilon.agent.play.Query;
import com.epsilon.agent.play.TableHooks;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TablePlan;
import com.epsilon.agent.play.TableSeat;
import com.epsilon.agent.policy.Policy;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.io.tenhou.TenhouXml;
import com.epsilon.mahjong.record.GameRecorder;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.FourPlayerFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.GZIPOutputStream;

/**
 * 自己対局の半荘を天鳳の牌譜の XML（gzip）に書く。1 半荘を 1 ファイルにする。classic のモデルを牌譜から学習するときの材料になる。
 * 卓のコアで打ち、学習の行は作らない。手は各席の σ から引く。ネットワークの方策の σ は one-hot なので、ネットワークの席は貪欲に打つ。
 *
 * <p>候補の打ち手を指定した数の席に座らせ、残りの席には相手を座らせる。候補の席は、半荘 ID を席の数で割った余りの席から順に続く。
 * 牌山は種と半荘 ID だけで決まる。ファイル名は半荘 ID に .xml.gz を付けたもの。牌譜の id は半荘 ID、打ち手の名前は席に座った打ち手の
 * 名前（prism:00760・heuristic など）にする。ファイルは一時ファイルに書いてから名前を変える。
 */
public final class SelfPlayRecords {

  /** 書いたファイルの名前の末尾。 */
  public static final String SUFFIX = ".xml.gz";

  private static final Variant VARIANT = FourPlayerFormat.VARIANT;

  /**
   * 書いた結果。
   *
   * @param games 書いた半荘の数
   * @param rounds 書いた局の数
   * @param seconds かかった時間（秒）
   */
  public record Result(int games, long rounds, double seconds) {}

  /**
   * 打ち手と、牌譜に書くその名前。
   *
   * @param policy 打ち手の方策
   * @param name 牌譜に書く打ち手の名前
   */
  public record Player(Policy policy, String name) {

    public Player {
      Objects.requireNonNull(policy, "policy");
      Objects.requireNonNull(name, "name");
    }
  }

  private SelfPlayRecords() {}

  /**
   * 半荘を打ち、牌譜のファイルに書く。
   *
   * @param table 半荘を打つ卓
   * @param candidate 候補の打ち手
   * @param opponent 相手の打ち手（候補が全ての席に座るなら使わない）
   * @param seats 候補の席の数（1〜席の数）
   * @param games 打つ半荘の数（1 以上）
   * @param seed 牌山の種
   * @param firstGame 最初の半荘 ID
   * @param out 書き出すディレクトリ（無ければ作る）
   * @return 書いた半荘の数・局の数・かかった時間
   * @throws IOException ファイルを書けなかったとき
   */
  public static Result write(
      TableLoop table,
      Player candidate,
      Player opponent,
      int seats,
      int games,
      long seed,
      long firstGame,
      Path out)
      throws IOException {
    int total = VARIANT.seats();
    if (seats < 1 || seats > total) {
      throw new IllegalArgumentException("The candidate takes 1 to " + total + " seats, not " + seats);
    }
    if (games < 1) {
      throw new IllegalArgumentException("Write at least one game: " + games);
    }
    Files.createDirectories(out);
    long started = System.nanoTime();
    Writer writer = new Writer(candidate, opponent, seats, out);
    List<Long> ids = new ArrayList<>(games);
    for (int game = 0; game < games; game++) {
      ids.add(firstGame + game);
    }
    List<List<TableSeat<Query>>> seating = seating(candidate, opponent, seats);
    table.play(
        new TablePlan<Query>(
            VARIANT,
            seed,
            seed,
            GameKey.onePerWall(ids),
            DrawOrder.SEAT_STREAMS,
            null,
            game -> seating.get(Math.floorMod(game.wall(), VARIANT.seats())),
            writer),
        TableHooks.plain());
    Throwable failure = writer.failure.get();
    if (failure != null) {
      throw new IOException("Could not record the self-play games", failure);
    }
    if (writer.written.get() != games || !writer.recorders.isEmpty()) {
      throw new IllegalStateException(
          "Recorded " + writer.written.get() + " of " + games + " games, " + writer.recorders.size() + " unfinished");
    }
    return new Result(games, writer.rounds.get(), (System.nanoTime() - started) / 1e9);
  }

  /** 各席の打ち手を、半荘 ID を席の数で割った余りごとに並べた表。 */
  private static List<List<TableSeat<Query>>> seating(Player candidate, Player opponent, int seats) {
    int total = VARIANT.seats();
    TableSeat<Query> candidateSeat = new PolicySeat(candidate.policy()), opponentSeat = new PolicySeat(opponent.policy());
    List<List<TableSeat<Query>>> result = new ArrayList<>(total);
    for (int rest = 0; rest < total; rest++) {
      List<TableSeat<Query>> row = new ArrayList<>(total);
      for (int seat = 0; seat < total; seat++) row.add(candidateSeat(rest, seat, seats) ? candidateSeat : opponentSeat);
      result.add(List.copyOf(row));
    }
    return List.copyOf(result);
  }

  /** 候補の席なら true。候補の席は、半荘 ID を席の数で割った余りの席から順に、候補の席の数だけ続く。 */
  static boolean candidateSeat(long gameId, int seat, int seats) {
    int total = VARIANT.seats();
    return Math.floorMod(seat - gameId, total) < seats;
  }

  /** 主経路の進行を半荘ごとに記録し、終局したらファイルに書く。進行ワーカーから並行に呼ばれる。 */
  private static final class Writer implements GameObserver {

    private final Player candidate;
    private final Player opponent;
    private final int seats;
    private final Path out;
    private final Map<Long, GameRecorder> recorders = new ConcurrentHashMap<>();
    private final AtomicLong written = new AtomicLong();
    private final AtomicLong rounds = new AtomicLong();
    private final AtomicReference<Throwable> failure = new AtomicReference<>();

    private Writer(Player candidate, Player opponent, int seats, Path out) {
      this.candidate = candidate;
      this.opponent = opponent;
      this.seats = seats;
      this.out = out;
    }

    @Override
    public void advanced(GameKey game, FullView full) {
      if (failure.get() != null) return;
      try {
        recorders.computeIfAbsent(game.wall(), id -> new GameRecorder(VARIANT)).advanced(full);
      } catch (RuntimeException thrown) {
        failure.compareAndSet(null, thrown);
      }
    }

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      if (failure.get() != null) return;
      try {
        recorder(game.wall()).roundEnded(full, end);
        rounds.incrementAndGet();
      } catch (RuntimeException thrown) {
        failure.compareAndSet(null, thrown);
      }
    }

    @Override
    public void gameEnded(GameKey key, TableState last) {
      if (failure.get() != null) return;
      long gameId = key.wall();
      try {
        GameRecorder recorder = recorder(gameId);
        recorders.remove(gameId);
        String[] names = new String[VARIANT.seats()];
        for (int seat = 0; seat < names.length; seat++) {
          names[seat] = candidateSeat(gameId, seat, seats) ? candidate.name() : opponent.name();
        }
        String id = Long.toString(gameId);
        RecordedGame game =
            recorder.gameEnded(last, new RecordedGame.Meta(id, RecordedGame.Source.TENHOU, Arrays.asList(names)));
        save(out.resolve(id + SUFFIX), TenhouXml.write(game));
        written.incrementAndGet();
      } catch (IOException thrown) {
        failure.compareAndSet(null, new UncheckedIOException(thrown));
      } catch (RuntimeException thrown) {
        failure.compareAndSet(null, thrown);
      }
    }

    private GameRecorder recorder(long gameId) {
      GameRecorder recorder = recorders.get(gameId);
      if (recorder == null) {
        throw new IllegalStateException("Game " + gameId + " ended a round before it started");
      }
      return recorder;
    }

    private static void save(Path file, String xml) throws IOException {
      Path temporary = file.resolveSibling(file.getFileName() + ".tmp");
      try (OutputStream stream = new GZIPOutputStream(Files.newOutputStream(temporary))) {
        stream.write(xml.getBytes(StandardCharsets.UTF_8));
      }
      Files.move(temporary, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    }
  }
}
