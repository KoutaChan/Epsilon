package com.epsilon.prism.grp;

import com.epsilon.prism.collect.Behaviour;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.collect.CollectionMetrics;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 収集の主経路の対局を記録する観察の口（phiMonitor が OFF でないとき、学習の JVM の {@code Lineage} が収集のコホートにだけ付ける。x1n3・3×H・
 * 監査の対局には付けない）。進行ワーカーから並行に呼ばれるので、半荘ごとの物を ConcurrentHashMap に持ち、{@link #gameEnded} で取り出して列に
 * 積む（半荘の中の列は compute の中で足すので、ワーカーをまたいでも見え方がずれない）。ラベル行には触れない。中の例外は全て捕まえ、記録だけを
 * 止めて 1 回だけ書き残す（{@link #seal} は空を返さず失敗を投げる）。
 */
public final class GameLogObserver implements CollectionDriver.GameObserver {
  private static final Logger LOG = LoggerFactory.getLogger(GameLogObserver.class);

  private final int cohort;
  private final PhiEntry phi;
  private final Behaviour behaviour;

  /** 終わっていない半荘の、半荘が続く局末ごとの次局の整数。 */
  private final Map<Long, List<HandStart>> open = new ConcurrentHashMap<>();

  /** 終わった半荘（終わった順）。 */
  private final ConcurrentLinkedQueue<RecordedGame> ended = new ConcurrentLinkedQueue<>();

  /** 記録を止めた最初の失敗（なければ null）。 */
  private final AtomicReference<Throwable> failure = new AtomicReference<>();

  private GameLogObserver(int cohort, PhiEntry phi, Behaviour behaviour) {
    this.cohort = cohort;
    this.phi = phi;
    this.behaviour = behaviour;
  }

  /**
   * コホート t の記録を始める。
   *
   * @param phi このコホートで使う Φ の項目
   * @param behaviour 全席の行動方策 b
   */
  public static GameLogObserver open(int cohort, PhiEntry phi, Behaviour behaviour) {
    if (phi == null || behaviour == null) throw new IllegalArgumentException("Φ entry and behaviour required");
    return new GameLogObserver(cohort, phi, behaviour);
  }

  @Override
  public void handProgressed(long gameId, int kyokuIndex, int honba, int kyotakuCount, int[] scores) {
    if (failure.get() != null) return;
    try {
      var start = new HandStart(kyokuIndex, honba, kyotakuCount, scores.clone());
      open.compute(
          gameId,
          (id, starts) -> {
            List<HandStart> list = starts == null ? new ArrayList<>() : starts;
            list.add(start);
            return list;
          });
    } catch (RuntimeException thrown) {
      stop(thrown);
    }
  }

  @Override
  public void gameEnded(long gameId, int[] finalScores) {
    if (failure.get() != null) return;
    try {
      List<HandStart> starts = open.remove(gameId);
      ended.add(new RecordedGame(gameId, starts == null ? List.of() : starts, finalScores.clone()));
    } catch (RuntimeException thrown) {
      stop(thrown);
    }
  }

  /**
   * 収集の後の記録（半荘 ID の昇順）。局末の数が収集の報告の hands と合わなければ投げる。
   *
   * @param counts 収集の数（主経路の判断と貪欲でない手の数を先頭に、hands と突き合わせる）
   * @throws IllegalStateException 記録が途中で止まったか、局末の数が合わないとき
   */
  public CohortGames seal(CollectionMetrics.Snapshot counts) {
    Throwable stopped = failure.get();
    if (stopped != null) throw new IllegalStateException("The game log stopped: " + stopped, stopped);
    if (!open.isEmpty())
      throw new IllegalStateException(open.size() + " games of cohort " + cohort + " did not end");
    List<RecordedGame> games = new ArrayList<>(ended);
    games.sort(Comparator.comparingLong(RecordedGame::gameId));
    long decisions = 0;
    for (long count : counts.mainDecisions()) decisions += count;
    var recorded =
        new CohortGames(
            new GameLogHeader(
                cohort,
                phi.id(),
                phi.sha256(),
                behaviour.epsilon(),
                behaviour.temperature(),
                decisions,
                counts.mainNonGreedy()),
            games);
    if (games.size() != counts.games() || recorded.handEnds() != counts.hands())
      throw new IllegalStateException(
          "The game log of cohort " + cohort + " has " + games.size() + " games and " + recorded.handEnds()
              + " hand ends, the collection " + counts.games() + " and " + counts.hands());
    return recorded;
  }

  /** 記録を止める（最初の失敗だけを書き残す）。 */
  private void stop(Throwable thrown) {
    if (failure.compareAndSet(null, thrown))
      LOG.warn("t={} the game log stopped recording: {}", cohort, thrown.toString());
  }
}
