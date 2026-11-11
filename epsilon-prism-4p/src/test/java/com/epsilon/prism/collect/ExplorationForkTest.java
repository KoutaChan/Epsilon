package com.epsilon.prism.collect;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.Run;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.EventKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.Resolution;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.Row;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 主経路が探索するとき（{@link Behaviour#explore}）、写しと分岐は、同じ判断の時点のほかの席の手も行動方策 b から引くことを確かめる。
 *
 * <p>方策は鳴きを貪欲に選ばず、b は一様分布を混ぜないので、b で打つ席は鳴かない。主経路の探索は見送りと鳴きをほぼ同じ率で引く。打牌に
 * 2 席以上が応答できる時点で、行を作る席が 1 つだけのとき、そこから分けた続きで鳴けるのはその席だけになる。
 */
public class ExplorationForkTest {

  private static final List<Long> GAMES = List.of(300L, 301L, 302L, 303L, 304L, 305L);

  /** 応答の判断の半分をラベル行にする（同じ時点の 2 席のうち 1 席だけが行を作る時点を多く作る）。 */
  private static final Sampling SAMPLING =
      TestSampling.grouped(0.25, 0.03, 0.03, 0.03, 0.5, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  /** 一様分布を混ぜず、主経路だけ温度 0.5 で探索する行動方策。 */
  private static final Behaviour EXPLORING = new Behaviour(0, 0.5, CrnScheme.SINGLE_RACE);

  /**
   * 主経路の、2 席以上が打牌に応答する判断の時点。
   *
   * @param events その時点までの公開イベントの数
   * @param history その時点までの公開イベントの要約
   * @param seats 判断ごとの席
   * @param ids 判断ごとの判断 ID
   * @param called 判断ごとに、主経路が選んだ手が鳴き（チー・ポン・大明槓）なら true
   */
  private record Point(int events, long history, int[] seats, long[] ids, boolean[] called) {}

  /**
   * 分けた続きの最初の問い合わせで見た、分けた時点の直後。
   *
   * @param point 分けた時点のキー
   * @param caller 分けた時点で鳴いた席（誰も鳴かなければ −1）
   */
  private record Continuation(String point, int caller) {}

  /**
   * 鳴きを選ばない inline の方策と、主経路の観測。s は見送りが 0、チー・ポン・大明槓が −0.01、和了が 10、打牌が手と席のハッシュ
   * （0〜1）、ほかは −10 で、σ はその貪欲。分けた続きの最初の問い合わせで、続きを作った判断の時点（{@link Run#resolution} の判断 ID と
   * 公開イベントの並びで見分ける）の直後に誰が鳴いたかを控える。
   */
  private static final class Watcher implements PolicyService, GameObserver {
    final Map<String, Point> points = new ConcurrentHashMap<>();
    final Set<Object> continued = ConcurrentHashMap.newKeySet();
    final Queue<Continuation> continuations = new ConcurrentLinkedQueue<>();

    private static boolean call(Action.Type type) {
      return type == Action.Type.CHI || type == Action.Type.PON || type == Action.Type.DAIMINKAN;
    }

    private static long history(EventLog log, int events) {
      long hash = events;
      for (int event = 0; event < events; event++) {
        int flags = (log.red(event) ? 1 : 0) | (log.tsumogiri(event) ? 2 : 0) | (log.riichi(event) ? 4 : 0);
        for (long field : new long[] {log.kind(event).ordinal(), log.seat(event), log.tile(event), log.meld(event), flags})
          hash = SeedMixer.mix64(hash * 31 + field);
      }
      return hash;
    }

    private static String key(long wall, long[] ids) {
      StringBuilder key = new StringBuilder().append(wall);
      for (long id : ids) key.append(':').append(id);
      return key.toString();
    }

    @Override
    public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
      int count = decisions.size();
      if (count < 2 || decisions.kind(slot) != EngineDecisionKind.RESPONSE) return;
      long[] ids = new long[count];
      int[] seats = new int[count];
      for (int each = 0; each < count; each++) {
        ids[each] = decisions.id(each);
        seats[each] = decisions.seat(each);
      }
      EventLog log = view.events();
      Point point =
          points.computeIfAbsent(
              key(game.wall(), ids),
              created -> new Point(log.size(), history(log, log.size()), seats, ids, new boolean[count]));
      point.called()[slot] = call(Action.of(decisions.legal(slot, choice)).type());
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {}

    @Override
    public int version() {
      return 1;
    }

    @Override
    public boolean inline() {
      return true;
    }

    @Override
    public void evaluateInline(PolicyRequest request) {
      Run<?> run = (Run<?>) request.owner();
      if (!run.main() && continued.add(run)) first(run, request);
      int n = request.legalCount();
      float[] scores = new float[n];
      int best = 0;
      for (int a = 0; a < n; a++) {
        Action.Type type = Action.of(request.legal()[a]).type();
        scores[a] =
            switch (type) {
              case PASS -> 0f;
              case CHI, PON, DAIMINKAN -> -0.01f;
              case TSUMO_AGARI, RON_AGARI -> 10f;
              case DAHAI -> (float) CrnSeeds.unit(SeedMixer.mix64(request.legal()[a] * 31L + request.seat()));
              default -> -10f;
            };
        if (scores[a] > scores[best]) best = a;
      }
      float[] sigma = new float[n];
      sigma[best] = 1f;
      request.scores = scores;
      request.sigma = sigma;
      if (request.retain()) {
        float[] prior = new float[n];
        double sum = 0;
        for (int a = 0; a < n; a++) {
          prior[a] = 1f + (float) CrnSeeds.unit(SeedMixer.mix64(request.legal()[a] * 17L + 5));
          sum += prior[a];
        }
        for (int a = 0; a < n; a++) prior[a] = (float) (prior[a] / sum);
        request.prior = prior;
      }
    }

    /** 分けた続きの最初の問い合わせ。直前に進めた判断の時点が控えた時点なら、その直後のイベントが鳴きかを見る。 */
    private void first(Run<?> run, PolicyRequest request) {
      Resolution resolution = run.resolution();
      long[] ids = new long[resolution.size()];
      for (int slot = 0; slot < ids.length; slot++) ids[slot] = resolution.id(slot);
      String key = key(request.wall(), ids);
      Point point = points.get(key);
      if (point == null) return;
      EventLog log = run.view(request.seat()).events();
      int events = point.events();
      if (log.size() < events || history(log, events) != point.history()) return;
      boolean called = log.size() > events && log.kind(events) == EventKind.MELD;
      continuations.add(new Continuation(key, called ? log.seat(events) : -1));
    }
  }

  @Test(timeOut = 600_000)
  public void continuationsDrawTheOtherSeatsFromTheBehaviourPolicy() {
    var watcher = new Watcher();
    List<Row> rows;
    try (var table = new TableLoop(new TableSettings(4, 2, 1000L, 4));
        var temporary = new CollectorTest.TemporaryCohort(null)) {
      var cohort =
          new Collector.Cohort(
              TENHOU_4P, 7_001L, 41_017L, GAMES, DrawOrder.SEAT_STREAMS, SAMPLING, null, CollectorTest.URA, EXPLORING,
              TwinDiscard.TSUMOGIRI, watcher);
      Collector.collect(table, cohort, watcher, Phi.scoreRank(TENHOU_4P, RankUtility.TENHOU), temporary.sink);
      rows = temporary.sealAndRead();
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    // 主経路の判断で行（ラベル行か根の記録）を作ったもの
    Set<String> retained = new HashSet<>();
    for (Row row : rows)
      if (row.source() != Row.Source.ROLLIN) retained.add(row.gameId() + ":" + row.decisionId());
    int exposed = 0;
    List<String> violations = new ArrayList<>();
    for (Continuation continuation : watcher.continuations) {
      Point point = watcher.points.get(continuation.point());
      String wall = continuation.point().substring(0, continuation.point().indexOf(':'));
      // 行を作った席がちょうど 1 つの時点だけを見る。続きで手を差し替えたのはその席
      int deviator = -1, deviators = 0;
      boolean othersCalled = false;
      for (int slot = 0; slot < point.ids().length; slot++) {
        if (retained.contains(wall + ":" + point.ids()[slot])) {
          deviator = point.seats()[slot];
          deviators++;
        } else if (point.called()[slot]) {
          othersCalled = true;
        }
      }
      if (deviators != 1 || !othersCalled) continue;
      exposed++;
      if (continuation.caller() >= 0 && continuation.caller() != deviator)
        violations.add(
            "seat " + continuation.caller() + " called in a continuation of seat " + deviator + " at " + continuation.point());
    }
    Collections.sort(violations);
    Assert.assertTrue(exposed > 0, "no continuation left a point where the main path explored another seat's call");
    Assert.assertEquals(violations, List.of());
  }
}
