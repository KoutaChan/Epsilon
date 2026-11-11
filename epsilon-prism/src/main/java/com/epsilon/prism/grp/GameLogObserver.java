package com.epsilon.prism.grp;

import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.train.CountedHands;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.HandResult;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.collect.CollectionMetrics;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 収集の主経路の半荘を記録するオブザーバー。収集のコホートにだけ付ける。収集では牌山ごとに 1 半荘を打つので、記録の半荘 ID は牌山の番号に
 * なる。
 *
 * <p>主経路の半荘が続く局末ごとに次局の卓を積み、半荘が終わったときの点数を最終の点数にする。全ての局（半荘が終わった局も）の結果も、
 * その局の親から見た生の形のコード（HandResult）で積む。
 *
 * <p>進行ワーカーから並行に呼ばれるので、半荘ごとの記録は ConcurrentHashMap に持ち、半荘が終わったら取り出してキューに積む。ラベル行には
 * 触れない。中で起きた例外は全て捕まえ、記録だけを止めて 1 回だけログに書く。そのときは {@link #seal} と {@link #countedHands} が、空を返さずに
 * 例外を投げる。
 */
public final class GameLogObserver implements GameObserver {
  private static final Logger LOG = LoggerFactory.getLogger(GameLogObserver.class);

  private final int cohort;
  private final PhiValue phi;
  private final double exploreTemperature;

  /** 終わっていない半荘の記録。キーは牌山の番号（記録の半荘 ID）。 */
  private final Map<Long, Playing> open = new ConcurrentHashMap<>();

  /** 終わった半荘（終わった順）。 */
  private final ConcurrentLinkedQueue<Played> ended = new ConcurrentLinkedQueue<>();

  /** 記録を止めた最初の失敗（なければ null）。 */
  private final AtomicReference<Throwable> failure = new AtomicReference<>();

  private GameLogObserver(int cohort, PhiValue phi, double exploreTemperature) {
    this.cohort = cohort;
    this.phi = phi;
    this.exploreTemperature = exploreTemperature;
  }

  /**
   * コホート t の記録を始める。
   *
   * @param cohort コホートの番号 t
   * @param phi このコホートで使う Φ
   * @param exploreTemperature 主経路の探索の温度（数のファイルのヘッダーに書く）
   */
  public static GameLogObserver open(int cohort, PhiValue phi, double exploreTemperature) {
    if (phi == null) throw new IllegalArgumentException("Φ value required");
    return new GameLogObserver(cohort, phi, exploreTemperature);
  }

  @Override
  public void roundEnded(GameKey game, FullView full, RoundEnd end) {
    if (failure.get() != null) return;
    try {
      TableState next = end.next();
      GrpState handEnd = next.gameOver() ? null : GrpState.of(next);
      RoundOutcome outcome = end.outcome();
      open.compute(game.wall(), (id, playing) -> (playing == null ? new Playing() : playing).add(outcome, handEnd));
    } catch (RuntimeException thrown) {
      stop(thrown);
    }
  }

  @Override
  public void gameEnded(GameKey game, TableState last) {
    if (failure.get() != null) return;
    try {
      int[] finalScores = new int[last.variant().seats()];
      for (int seat = 0; seat < finalScores.length; seat++) finalScores[seat] = last.score(seat);
      Playing playing = open.remove(game.wall());
      ended.add((playing == null ? new Playing() : playing).end(phi.entry().variant(), game.wall(), finalScores));
    } catch (RuntimeException thrown) {
      stop(thrown);
    }
  }

  /**
   * 収集の後に、半荘の記録をまとめて返す（半荘 ID の昇順）。
   *
   * @param counts 収集の数（半荘と局末の数を突き合わせる）
   * @return コホートの半荘の記録
   * @throws IllegalStateException 記録が途中で止まったか、半荘か局末の数が収集の数と合わないとき
   */
  public GameCohort seal(CollectionMetrics.Snapshot counts) {
    List<GameRecord> games = new ArrayList<>();
    for (Played game : played(counts)) games.add(game.record());
    PhiEntry entry = phi.entry();
    return new GameCohort(entry.variant(), cohort, entry.id(), games);
  }

  /**
   * 収集の後に、主経路の全ての局について、始まりの卓と結果のコードを返す（半荘 ID の昇順）。ヘッダーには主経路の探索の温度と、使った Φ を
   * 書く。返す前に局ごとに、始まりの卓を結果のコードで精算して局を進め、記録した次局の卓と同じになることを確かめる。半荘の最後の局では、
   * 終局して最終の点数と同じになることを確かめる。
   *
   * @param counts 収集の数（半荘と局末の数を突き合わせる）
   * @throws IllegalStateException 記録が途中で止まったか、数が合わないか、記録を再現しない局があるとき
   */
  public CountedHands countedHands(CollectionMetrics.Snapshot counts) {
    Variant variant = phi.entry().variant();
    CountedHands.Builder builder = CountedHands.builder(variant, cohort, exploreTemperature, phi.stamp());
    TableState first = new TableState(variant);
    TableState start = new TableState(variant);
    TableState settled = new TableState(variant);
    RoundOutcome outcome = new RoundOutcome(variant);
    int[] delta = new int[variant.seats()];
    for (Played game : played(counts)) {
      GameRecord record = game.record();
      List<GrpState> handEnds = record.handEnds();
      long[] codes = game.codes();
      if (codes.length != handEnds.size() + 1)
        throw new IllegalStateException(
            "Game " + record.gameId() + " has " + codes.length + " hand results for " + handEnds.size() + " hand ends");
      start.copyFrom(first);
      for (int hand = 0; hand < codes.length; hand++) {
        builder.add(record.gameId(), start, codes[hand]);
        HandResult.place(codes[hand], start.dealer(), outcome);
        Settlement.apply(variant, start, outcome, delta);
        Progression.next(start, outcome, delta, settled);
        boolean replayed =
            hand < handEnds.size() ? reaches(settled, handEnds.get(hand)) : settled.endsWith(record.finalScores());
        if (!replayed)
          throw new IllegalStateException(
              "Hand " + hand + " of game " + record.gameId() + " does not replay its record, it reaches " + settled);
        start.copyFrom(settled);
      }
    }
    return builder.build();
  }

  /**
   * 終わった半荘を、半荘 ID の昇順で返す。記録が止まっていたか、終わっていない半荘があるか、半荘か局末の数が収集の数と合わなければ例外を
   * 投げる。
   */
  private List<Played> played(CollectionMetrics.Snapshot counts) {
    Throwable stopped = failure.get();
    if (stopped != null) throw new IllegalStateException("The game log stopped: " + stopped, stopped);
    if (!open.isEmpty())
      throw new IllegalStateException(open.size() + " games of cohort " + cohort + " did not end");
    List<Played> games = new ArrayList<>(ended);
    games.sort(Comparator.comparingLong(game -> game.record().gameId()));
    // 半荘が終わった局末は記録に入らないので、半荘ごとに 1 つ足して収集の局末の数と比べる
    long handEnds = games.size();
    for (Played game : games) handEnds += game.record().handEnds().size();
    if (games.size() != counts.games() || handEnds != counts.hands())
      throw new IllegalStateException(
          "The game log of cohort " + cohort + " has " + games.size() + " games and " + handEnds
              + " hand ends, the collection " + counts.games() + " and " + counts.hands());
    return games;
  }

  /** 精算して局を進めた卓が、記録した次局の卓と同じなら true。半荘が終わっていれば false。 */
  private static boolean reaches(TableState settled, GrpState next) {
    if (settled.gameOver()
        || settled.round() != next.round()
        || settled.honba() != next.honba()
        || settled.kyotaku() != next.sticks()) {
      return false;
    }
    for (int seat = 0; seat < next.variant().seats(); seat++) {
      if (settled.score(seat) != next.score(seat)) {
        return false;
      }
    }
    return true;
  }

  /** 記録を止める。ログに書くのは最初の失敗だけ。 */
  private void stop(Throwable thrown) {
    if (failure.compareAndSet(null, thrown))
      LOG.warn("t={} the game log stopped recording: {}", cohort, thrown.toString());
  }

  /**
   * 終わった半荘。
   *
   * @param record 半荘の記録
   * @param codes 局ごとの結果のコード。生の形で打った順に並び、半荘が終わった局も入る
   */
  private record Played(GameRecord record, long[] codes) {}

  /** 終わっていない半荘の記録。同じ半荘には、一度に 1 つのスレッドだけが ConcurrentHashMap の compute の中で書く。 */
  private static final class Playing {
    private static final int INITIAL_HANDS = 16;

    /** 半荘が続いた局末ごとの次局の卓。 */
    private final List<GrpState> handEnds = new ArrayList<>();

    /** 局ごとの結果のコード（生の形、打った順）。 */
    private long[] codes = new long[INITIAL_HANDS];

    private int hands;

    /**
     * 局を 1 つ積む。結果のコードは、その局の親から見た形にする。親は、半荘の最初の局なら東 1 局の親、ほかは前の局末に積んだ次局の卓の親。
     *
     * @param outcome 局の結果
     * @param handEnd 次局の卓（半荘が終わった局なら null）
     */
    Playing add(RoundOutcome outcome, GrpState handEnd) {
      int dealer = handEnds.isEmpty() ? outcome.variant().dealer(0) : handEnds.getLast().dealer();
      if (hands == codes.length) codes = Arrays.copyOf(codes, 2 * hands);
      codes[hands++] = HandResult.raw(outcome, dealer);
      if (handEnd != null) handEnds.add(handEnd);
      return this;
    }

    Played end(Variant variant, long gameId, int[] finalScores) {
      return new Played(new GameRecord(variant, gameId, handEnds, finalScores), Arrays.copyOf(codes, hands));
    }
  }
}
