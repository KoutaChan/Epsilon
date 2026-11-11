package com.epsilon.prism.grp;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.agent.play.GameKey;
import com.epsilon.agent.play.GameObserver;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.Phi;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.CountedHands;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.HandResult;
import com.epsilon.grp.train.PhiStamp;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RandomGames;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.collect.CollectionMetrics;
import com.epsilon.prism.collect.Collector;
import com.epsilon.prism.collect.RowSink;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.collect.TestSampling;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 主経路の対局の記録（{@link GameLogObserver}）を、本物の収集（{@link Collector}、CPU の inline の方策、進行のワーカー 3 本）で確かめる。
 * 記録にある次局の卓の列は、収集が渡した列と一致する。局末の数は収集の hands と合い、記録を書いて読み戻しても変わらない。記録を付けても
 * 付けなくても、コホートの行はビットまで一致する。記録の中で起きた失敗は記録だけを止め、確定のときに例外を投げる。
 *
 * <p>数の列（{@link GameLogObserver#countedHands}）は、半荘が終わった局も含む全ての局を持ち、ヘッダーに探索の温度と、使った Φ の印を
 * 持つ。記録を再現しない局（最終の点数の違い、抜けた局末）があれば、数の列を作らずに例外を投げる。
 */
public class GameLogObserverTest {
  private static final long SEED = 9_101L;
  private static final PhiValue SCORE_RANK = PhiValue.grp(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, TENHOU_4P));

  /** q を持つ項目に、LATE のモードの行（打ち切り 1e-5）を当てた値。記録は q を読まないので、パスの先は無くてよい。 */
  private static final PhiValue LATE =
      PhiValue.of(
          new PhiEntry(
              "g003-a1", PhiEntryKind.NETWORK, TENHOU_4P, "/grp/a", "aa", "", "", "/grp/q", "qq", 4, PhiEntry.INITIAL_ID,
              "test"),
          ModeLine.late(1e-5, 4, "x"));

  /** ラベル行、ロールイン、写しの中の行がどれも出るようにした抽出率（本番より高い）。 */
  private static final Sampling SAMPLING =
      TestSampling.grouped(0.25, 0.03, 0.03, 0.03, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  /** 一様分布を ε_b = 0.05 だけ混ぜる行動方策。 */
  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.TWO_STAGE_FLOOR);

  /** 裏ドラは実際の値で学び、局の 5% では期待値も求めて監査する。 */
  private static final UraSettings URA = new UraSettings.Realized(0.05);

  @Test(timeOut = 600_000)
  public void recordedHistoriesMatchTheCollection() throws IOException {
    var tapped = new Tapped(GameLogObserver.open(4, SCORE_RANK, 0));
    Collected collected = collect(tapped, 3, 4);
    CollectionMetrics.Snapshot counts = collected.result().counts();
    GameCohort games = tapped.log.seal(counts);
    Assert.assertEquals(games.games().size(), GAMES);
    Assert.assertEquals(games.cohort(), 4);
    Assert.assertEquals(games.phiId(), PhiEntry.SCORE_RANK_ID);
    long handEnds = games.games().size();
    for (GameRecord game : games.games()) handEnds += game.handEnds().size();
    Assert.assertEquals(handEnds, counts.hands());
    assertHistories(games, tapped.states, tapped.finals);
    // 記録を書いて読み戻しても同じ
    Path directory = Files.createTempDirectory("prism-gamelog");
    try {
      Path file = new GrpFiles(directory).records(4);
      Files.createDirectories(file.getParent());
      GameRecordCodec.write(file, games);
      assertHistories(GameRecordCodec.read(file), tapped.states, tapped.finals);
    } finally {
      Snapshot.deleteTree(directory);
    }
  }

  /**
   * 数の列（{@link GameLogObserver#countedHands}）が収集の全ての局を持つことを確かめる。局の数は収集の hands（半荘が終わった局も入る）と
   * 同じ。各局の始まりの卓は、東 1 局の卓か、収集が渡した前の局の次局の卓。コードは、その局の結果をその卓の親から見たもの。ヘッダーは探索の
   * 温度と使った Φ の印で、数のファイルを書いて読み戻しても変わらない。
   */
  @Test(timeOut = 600_000)
  public void countsHoldEveryHandOfTheCollection() throws IOException {
    var tapped = new Tapped(GameLogObserver.open(4, LATE, 0.25));
    Collected collected = collect(tapped, 3, 4);
    CollectionMetrics.Snapshot counts = collected.result().counts();
    CountedHands hands = tapped.log.countedHands(counts);
    Assert.assertEquals(hands.games(), GAMES);
    Assert.assertEquals(hands.hands(), counts.hands());
    Assert.assertEquals(hands.cohort(), 4);
    Assert.assertEquals(hands.exploreTemperature(), 0.25);
    Assert.assertEquals(hands.phi(), new PhiStamp("LATE", 1e-5));
    Assert.assertEquals(hands.firstMismatch(), -1);
    assertHands(hands, tapped);
    // 数のファイルを書いて読み戻しても同じ
    Path directory = Files.createTempDirectory("prism-counts");
    try {
      Path file = new GrpFiles(directory).counts(4);
      CountFile.write(file, hands);
      CountedHands read = CountFile.read(file);
      Assert.assertEquals(read.phi(), hands.phi());
      Assert.assertEquals(read.exploreTemperature(), 0.25);
      assertHands(read, tapped);
    } finally {
      Snapshot.deleteTree(directory);
    }
  }

  @Test(timeOut = 600_000)
  public void recordingDoesNotChangeTheRows() throws IOException {
    Collected off = collect(null, 3, 4);
    var log = GameLogObserver.open(4, SCORE_RANK, 0);
    Collected on = collect(log, 3, 4);
    Assert.assertTrue(off.rows().rows() > 0, "no rows");
    Assert.assertEquals(on.rows(), off.rows());
    Assert.assertEquals(on.result().counts().hands(), off.result().counts().hands());
    Assert.assertEquals(log.seal(on.result().counts()).games().size(), GAMES);
    Assert.assertEquals(log.countedHands(on.result().counts()).hands(), on.result().counts().hands());
  }

  /** GRP のモードの印では、打ち切りが NaN になる。 */
  @Test
  public void theStampIsTheUsedValue() {
    for (Variant variant : Variant.values()) {
      PhiValue score = PhiValue.grp(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, variant));
      var log = GameLogObserver.open(1, score, 0);
      long played = walk(variant, 11, 2, log, Fault.NONE);
      CountedHands hands = log.countedHands(counts(2, played));
      Assert.assertEquals(hands.variant(), variant);
      Assert.assertEquals(hands.hands(), played);
      Assert.assertEquals(hands.phi(), new PhiStamp("GRP", Double.NaN));
      Assert.assertEquals(hands.phi(), score.stamp());
      Assert.assertEquals(hands.firstMismatch(), -1);
    }
    // LATE を求めても、q の無い項目は GRP のモードになり、印も GRP のモードになる
    PhiValue kept = PhiValue.of(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, TENHOU_4P), ModeLine.late(1e-5, 1, "x"));
    Assert.assertEquals(kept.stamp(), new PhiStamp("GRP", Double.NaN));
  }

  /**
   * 記録を再現しない局があれば、数の列を作らずに例外を投げる（半荘の記録はそのまま確定できる）。入れる誤りは 3 つ。最終の点数が違う。
   * 半荘が続く局末が 1 つ抜けた。半荘が終わった局末が抜けて、局の数が局末の数 ＋ 1 にならない。
   */
  @Test
  public void handsThatDoNotReplayAreRefused() {
    for (Fault fault : new Fault[] {Fault.FINAL_SCORE, Fault.CONTINUING_END, Fault.ENDING_END}) {
      var log = GameLogObserver.open(1, SCORE_RANK, 0);
      long played = walk(TENHOU_4P, 13, 3, log, fault);
      long fed = fault == Fault.CONTINUING_END ? played - 1 : played;
      CollectionMetrics.Snapshot counts = counts(3, fed);
      Assert.assertEquals(log.seal(counts).games().size(), 3, fault.toString());
      IllegalStateException refused = Assert.expectThrows(IllegalStateException.class, () -> log.countedHands(counts));
      String expected = fault == Fault.ENDING_END ? "hand results for" : "does not replay";
      Assert.assertTrue(refused.getMessage().contains(expected), fault + ": " + refused.getMessage());
      Assert.assertTrue(refused.getMessage().contains(" 100 "), fault + ": " + refused.getMessage());
    }
    // 局末の数が収集と合わなければ、数の列も作らない
    var log = GameLogObserver.open(1, SCORE_RANK, 0);
    long played = walk(TENHOU_4P, 13, 3, log, Fault.NONE);
    Assert.assertEquals(log.countedHands(counts(3, played)).hands(), played);
    Assert.expectThrows(IllegalStateException.class, () -> log.countedHands(counts(3, played + 1)));
    Assert.expectThrows(IllegalStateException.class, () -> log.countedHands(counts(2, played)));
  }

  @Test
  public void failureStopsOnlyTheRecording() {
    TableState last = table(TENHOU_4P, 40000, 30000, 20000, 10000);
    var log = GameLogObserver.open(2, SCORE_RANK, 0);
    GrpState next = firstRoundEnd(TENHOU_4P, 3, 1, log);
    log.gameEnded(new GameKey(1, 0), last);
    // 別のルールの卓は記録を止める（例外は外へ出ない）
    Assert.assertEquals(firstRoundEnd(Variant.TENHOU_3P, 4, 2, log).variant(), Variant.TENHOU_3P);
    log.gameEnded(new GameKey(2, 0), last);
    IllegalStateException stopped = Assert.expectThrows(IllegalStateException.class, () -> log.seal(counts(2, 3)));
    Assert.assertTrue(stopped.getMessage().contains("stopped"), stopped.getMessage());
    IllegalStateException counted = Assert.expectThrows(IllegalStateException.class, () -> log.countedHands(counts(2, 3)));
    Assert.assertTrue(counted.getMessage().contains("stopped"), counted.getMessage());
    // 局末の数が収集と合わない記録は確定しない
    var mismatched = GameLogObserver.open(2, SCORE_RANK, 0);
    Assert.assertEquals(firstRoundEnd(TENHOU_4P, 3, 1, mismatched), next);
    mismatched.gameEnded(new GameKey(1, 0), last);
    Assert.assertEquals(mismatched.seal(counts(1, 2)).games().getFirst().handEnds(), List.of(next));
    Assert.assertEquals(mismatched.seal(counts(1, 2)).games().getFirst().finalScores(), new int[] {40000, 30000, 20000, 10000});
    Assert.expectThrows(IllegalStateException.class, () -> mismatched.seal(counts(1, 3)));
    // 半荘が終わった局末を受けていない半荘は、半荘の記録としては確定できても、数の列にはならない
    Assert.expectThrows(IllegalStateException.class, () -> mismatched.countedHands(counts(1, 2)));
    // 終わっていない半荘が残る記録も確定しない
    var unfinished = GameLogObserver.open(2, SCORE_RANK, 0);
    firstRoundEnd(TENHOU_4P, 3, 5, unfinished);
    Assert.expectThrows(IllegalStateException.class, () -> unfinished.seal(counts(0, 1)));
    Assert.expectThrows(IllegalStateException.class, () -> unfinished.countedHands(counts(0, 1)));
    Assert.expectThrows(IllegalArgumentException.class, () -> GameLogObserver.open(2, null, 0));
  }

  /** 終局の卓（東 1 局の形で、点数だけを置く）。 */
  private static TableState table(Variant variant, int... scores) {
    TableState table = new TableState(variant);
    table.set(0, 0, 0, scores);
    return table;
  }

  /**
   * 乱数の種 {@code seed} で打った半荘のうち、半荘が続く最初の局末を、半荘 {@code gameId} として {@code observer} に渡す。戻り値は
   * その次局の卓。
   */
  private static GrpState firstRoundEnd(Variant variant, long seed, long gameId, GameObserver observer) {
    GrpState[] first = new GrpState[1];
    new TableWalk(variant, TableWalk.Policy.GREEDY)
        .run(
            seed,
            1,
            new TableWalk.Visitor() {
              @Override
              public void decision(com.epsilon.mahjong.engine.Engine engine, int slot) {}

              @Override
              public void roundEnd(com.epsilon.mahjong.engine.Engine engine) {
                if (first[0] != null || engine.roundEnd().next().gameOver()) return;
                observer.roundEnded(new GameKey(gameId, 0), engine.full(), engine.roundEnd());
                first[0] = GrpState.of(engine.roundEnd().next());
              }
            });
    Assert.assertNotNull(first[0], "no continuing round end");
    return first[0];
  }

  /** {@link #walk} が最初の半荘に入れる誤り。 */
  private enum Fault {
    /** 誤りを入れない。 */
    NONE,
    /** 最終の点を 100 点ずらして渡す。 */
    FINAL_SCORE,
    /** 半荘が続く 2 つ目の局末を渡さない。 */
    CONTINUING_END,
    /** 半荘が終わった局末を渡さない（終局は渡す）。 */
    ENDING_END
  }

  /**
   * 乱数の種 {@code seed} から、合法手を一様に選んで {@code games} 回の半荘を打つ。半荘 ID 100 から順に、局末（半荘が終わった局も）と
   * 終局を {@code observer} に渡す。最初の半荘にだけ誤り {@code fault} を入れる。
   *
   * @return 打った局の数（渡さなかった局末も数える）
   */
  private static long walk(Variant variant, long seed, int games, GameObserver observer, Fault fault) {
    RandomGames play = RandomGames.start(variant, seed);
    long played = 0;
    int continuing = 0;
    boolean faulted = false;
    for (int game = 0; game < games; ) {
      Boundary at = play.boundary();
      if (at != Boundary.DECISIONS) {
        RoundEnd end = play.engine.roundEnd();
        TableState next = end.next();
        Assert.assertEquals(next.gameOver(), at == Boundary.GAME_END);
        GameKey key = new GameKey(100 + game, 0);
        boolean first = game == 0;
        boolean withheld =
            first && (next.gameOver() ? fault == Fault.ENDING_END : fault == Fault.CONTINUING_END && continuing++ == 1);
        played++;
        faulted |= withheld;
        if (!withheld) observer.roundEnded(key, play.engine.full(), end);
        if (next.gameOver()) {
          boolean shift = first && fault == Fault.FINAL_SCORE;
          faulted |= shift;
          observer.gameEnded(key, shift ? shifted(next) : next);
          if (++game == games) break;
        }
      }
      play.step();
    }
    Assert.assertEquals(faulted, fault != Fault.NONE, "the fault " + fault + " was not placed");
    return played;
  }

  /** 最終の点数で、トップの席から次の席へ 100 点を移した卓。 */
  private static TableState shifted(TableState last) {
    int seats = last.variant().seats();
    int[] scores = new int[seats];
    for (int seat = 0; seat < seats; seat++) scores[seat] = last.score(seat);
    int top = last.top();
    scores[top] -= 100;
    scores[(top + 1) % seats] += 100;
    TableState table = new TableState(last.variant());
    table.setScenario(0, 0, 0, scores);
    return table;
  }

  /**
   * 数の列が、収集が渡したものと一致することを確かめる。半荘ごとに、局の数は次局の卓の数 ＋ 1。最初の局は東 1 局の卓から、ほかの局は
   * 前の局末の次局の卓から始まる。コードは、控えた結果をその局の親から見たもの。
   */
  private static void assertHands(CountedHands hands, Tapped tapped) {
    Variant variant = hands.variant();
    GrpState opening = GrpState.of(new TableState(variant));
    TableState start = new TableState(variant);
    int ended = 0;
    for (int game = 0; game < hands.games(); game++) {
      long id = hands.gameId(game);
      List<GrpState> states = tapped.states.getOrDefault(id, List.of());
      List<RoundOutcome> outcomes = tapped.outcomes.get(id);
      int first = hands.firstHand(game);
      Assert.assertEquals(hands.firstHand(game + 1) - first, states.size() + 1, "game " + id);
      Assert.assertEquals(outcomes.size(), states.size() + 1, "game " + id);
      for (int hand = 0; hand <= states.size(); hand++) {
        hands.start(first + hand, start);
        Assert.assertEquals(GrpState.of(start), hand == 0 ? opening : states.get(hand - 1), "game " + id + " hand " + hand);
        Assert.assertEquals(
            hands.code(first + hand), HandResult.raw(outcomes.get(hand), start.dealer()), "game " + id + " hand " + hand);
      }
      ended++;
    }
    Assert.assertEquals(ended, GAMES);
  }

  /** 収集した半荘の数。 */
  private static final int GAMES = 6;

  /**
   * 1 つの収集の結果と、確定したコホートの行のフィンガープリント。
   *
   * @param result 収集の結果
   * @param rows 行のフィンガープリント
   */
  record Collected(Collector.Result result, RowHash.Digest rows) {}

  /** 半荘 300〜305 を {@code observer} 付きで集め、行を一時ディレクトリのコホートに書いて確定する。 */
  private static Collected collect(GameObserver observer, int workers, int concurrent)
      throws IOException {
    List<Long> ids = new ArrayList<>();
    for (long id = 0; id < GAMES; id++) ids.add(300 + id);
    Path directory = Files.createTempDirectory("prism-rows");
    try (var table = new TableLoop(new TableSettings(concurrent, workers, 1000L, 4));
        var writer = new CohortWriter(directory, TENHOU_4P, 4, 0, 2, CohortWriter.Tap.NONE)) {
      var cohort =
          new Collector.Cohort(
              TENHOU_4P, SEED, SEED * 5, ids, DrawOrder.SEAT_STREAMS, SAMPLING, null, URA, FLOOR, TwinDiscard.TSUMOGIRI,
              observer);
      var result =
          Collector.collect(
              table, cohort, new HashedScorePolicy(), Phi.scoreRank(TENHOU_4P, RankUtility.TENHOU), new RowSink(writer));
      writer.seal();
      return new Collected(result, RowHash.of(directory));
    } finally {
      Snapshot.deleteTree(directory);
    }
  }

  /** 記録の半荘ごとに、次局の卓の列が収集の渡した列と一致し、最終の点数も一致することを確かめる。 */
  private static void assertHistories(GameCohort games, Map<Long, List<GrpState>> states, Map<Long, int[]> finals) {
    int compared = 0;
    for (GameRecord game : games.games()) {
      List<GrpState> expected = states.getOrDefault(game.gameId(), List.of());
      Assert.assertEquals(game.handEnds(), expected, "game " + game.gameId());
      Assert.assertEquals(game.finalScores(), finals.get(game.gameId()));
      compared += expected.size();
    }
    Assert.assertTrue(compared > GAMES, "too few continuing hand ends: " + compared);
  }

  /** 収集の数のうち、記録が突き合わせるフィールド（半荘の数と局末の数）だけを入れたもの。 */
  private static CollectionMetrics.Snapshot counts(long games, long hands) {
    return new CollectionMetrics.Snapshot(
        0, 0, hands, games, 0, 0, new long[1], new long[1], new long[1], new long[1], 0, 0, 0, 0, 0, 0, 0, 0, 0,
        new CollectionMetrics.Breakdown(Map.of(), Map.of(), Map.of(), Map.of(), Map.of(), Map.of()));
  }

  /** 受けたものを記録に流しながら、収集が渡す次局の卓、局ごとの結果（半荘が終わった局も）、最終の点数を控えるオブザーバー。 */
  static final class Tapped implements GameObserver {
    final GameLogObserver log;
    final Map<Long, List<GrpState>> states = new ConcurrentHashMap<>();
    final Map<Long, List<RoundOutcome>> outcomes = new ConcurrentHashMap<>();
    final Map<Long, int[]> finals = new ConcurrentHashMap<>();

    Tapped(GameLogObserver log) {
      this.log = log;
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {
      int[] scores = new int[last.variant().seats()];
      for (int seat = 0; seat < scores.length; seat++) scores[seat] = last.score(seat);
      finals.put(game.wall(), scores);
      log.gameEnded(game, last);
    }

    @Override
    public void roundEnded(GameKey game, FullView full, RoundEnd end) {
      if (!end.next().gameOver())
        states.computeIfAbsent(game.wall(), id -> new ArrayList<>()).add(GrpState.of(end.next()));
      RoundOutcome outcome = new RoundOutcome(end.outcome().variant());
      outcome.copyFrom(end.outcome());
      outcomes.computeIfAbsent(game.wall(), id -> new ArrayList<>()).add(outcome);
      log.roundEnded(game, full, end);
    }
  }

  /**
   * s を手の ID と席から作るハッシュにし、σ をその argmax に置く inline の方策。手の選び方は決定的で、打ち筋は手ごとにばらける。
   * {@code retain} のときは p0 も書く（収集のテストの {@code CollectorTest.HashedScorePolicy} と同じ）。
   */
  static final class HashedScorePolicy implements PolicyService {
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
      int n = request.legalCount();
      float[] scores = new float[n];
      int best = 0;
      for (int a = 0; a < n; a++) {
        long bits = SeedMixer.mix64(request.legal()[a] * 31L + request.seat());
        scores[a] = (float) CrnSeeds.unit(bits);
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
          long bits = SeedMixer.mix64(request.legal()[a] * 17L + 5);
          prior[a] = 1f + (float) CrnSeeds.unit(bits);
          sum += prior[a];
        }
        for (int a = 0; a < n; a++) prior[a] = (float) (prior[a] / sum);
        request.prior = prior;
      }
    }
  }
}
