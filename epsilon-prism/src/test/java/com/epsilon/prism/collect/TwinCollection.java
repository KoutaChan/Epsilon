package com.epsilon.prism.collect;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.agent.play.Behaviour;
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
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.DeviationTarget;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.LongAdder;
import org.testng.Assert;

/**
 * テスト用のヘルパー。同値な打牌に同じ s と p0 を付ける方策（ネットワークと同じ。手は合法手・席・手牌だけで決まり、河のツモ切りの印を
 * 読まない）で 1 コホートを集め、同値な打牌の切り方が全ての経路で効いたかを数える。
 *
 * <p>確かめ方: 方策は経路ごとに手番の合法手を覚え、同じ経路の次の問い合わせの O に出た打牌のイベントのツモ切りの印を、切り方と比べる（主経路の
 * 貪欲・床・探索と、写しと分岐の続きの手）。分岐と写しの最初の手は行の手の位置（打った手・候補集合・逸脱）で比べる。
 */
public final class TwinCollection {

  /**
   * ラベル行・ロールイン・写しの中の行がどれも出る率。終盤の打牌は全ての合法手を測り（{@link CandidateSet#allActions}）、中盤の打牌の逸脱は
   * 打った手のほかから引く（{@link DeviationTarget#OTHER}）。
   */
  public static final Sampling SAMPLING = sampling();

  /** 局末の点の順位をそのまま値にする Φ の効用。 */
  private static final RankUtility UTILITY = RankUtility.TENHOU;

  private static final UraSettings URA = new UraSettings.Realized(0.05);

  /**
   * 集めた結果。
   *
   * @param rows 行（受け口に渡した順ではない）
   * @param main 主経路の判断ごとの打った手を、同値な打牌の組の手出しの番号に直したもの（キーは牌山・判断 ID・判断の位置）
   * @param mainTwins 主経路で、手出しとツモ切りの両方で切れる打牌を切った手番の数
   * @param branchTwins 写しと分岐の続きで、同じ打牌を切った手番の数
   * @param riichiTwins そのうち立直の打牌の数
   * @param violations 切り方と違うツモ切りの印の打牌
   */
  public record Collected(
      List<Row> rows, Map<String, Integer> main, long mainTwins, long branchTwins, long riichiTwins, List<String> violations) {}

  /**
   * 行の手の位置を数えた結果。
   *
   * @param played 打った手が、手出しとツモ切りの両方で切れる打牌だった行の数
   * @param candidates 打った手のほかの候補集合の手（分岐の最初の手）で、同じ打牌だった数
   * @param deviations 逸脱 d（写しの最初の手）で、同じ打牌だった数
   * @param violations 切り方と違う方の手を書いた行
   */
  public record RowTwins(long played, long candidates, long deviations, List<String> violations) {}

  private TwinCollection() {}

  /**
   * TSUMOGIRI と TEDASHI で同じ半荘を集め、全ての経路で切り方のとおりに切ったことと、手と行が同値な打牌の組を除けば同じことを確かめる。
   * 主経路・写しと分岐の続き・立直の打牌・打った手・候補集合・逸脱のどれにも、手出しとツモ切りの両方で切れる打牌が出る。
   *
   * @param behaviour 全席の行動方策 b
   */
  public static void assertEveryPath(Variant variant, Behaviour behaviour, long iterationSeed, List<Long> gameIds) {
    Collected tsumogiri = collect(variant, TwinDiscard.TSUMOGIRI, behaviour, iterationSeed, gameIds);
    Collected tedashi = collect(variant, TwinDiscard.TEDASHI, behaviour, iterationSeed, gameIds);
    assertPlays(tsumogiri, variant, TwinDiscard.TSUMOGIRI);
    assertPlays(tedashi, variant, TwinDiscard.TEDASHI);
    Assert.assertEquals(tsumogiri.main(), tedashi.main());
    Assert.assertEquals(canonical(tsumogiri.rows(), variant), canonical(tedashi.rows(), variant));
  }

  private static void assertPlays(Collected collected, Variant variant, TwinDiscard twins) {
    Assert.assertEquals(collected.violations(), List.of(), twins.name());
    Assert.assertTrue(collected.mainTwins() > 0, twins + " main " + collected.mainTwins());
    Assert.assertTrue(collected.branchTwins() > 0, twins + " branches " + collected.branchTwins());
    Assert.assertTrue(collected.riichiTwins() > 0, twins + " riichi " + collected.riichiTwins());
    RowTwins rows = rowTwins(collected.rows(), variant, twins);
    Assert.assertEquals(rows.violations(), List.of(), twins.name());
    Assert.assertTrue(rows.played() > 0 && rows.candidates() > 0 && rows.deviations() > 0, twins + " " + rows);
  }

  /**
   * 半荘 gameIds の 1 つのコホートを集める（同時に 4 局・進行ワーカー 2 本、帳簿なし）。行は実行順に依らない。
   *
   * @param twinDiscard 同値な打牌のうち切る方
   * @param behaviour 全席の行動方策 b
   */
  public static Collected collect(
      Variant variant, TwinDiscard twinDiscard, Behaviour behaviour, long iterationSeed, List<Long> gameIds) {
    Path directory;
    try {
      directory = Files.createTempDirectory("prism-twins");
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
    var policy = new TwinBlindPolicy(variant, twinDiscard);
    var main = new MainChoices();
    try (var table = new TableLoop(new TableSettings(4, 2, 1000L, 4));
        var writer = new CohortWriter(directory, variant, 1, 0, 2, CohortWriter.Tap.NONE)) {
      var cohort =
          new Collector.Cohort(
              variant, 7_001L, iterationSeed, gameIds, DrawOrder.SEAT_STREAMS, SAMPLING, null, URA, behaviour,
              twinDiscard, main);
      Collector.collect(table, cohort, policy, Phi.scoreRank(variant, UTILITY), new RowSink(writer));
      writer.seal();
      List<Row> rows = read(directory, variant);
      return new Collected(
          rows,
          new TreeMap<>(main.choices),
          policy.mainTwins.sum(),
          policy.branchTwins.sum(),
          policy.riichiTwins.sum(),
          List.copyOf(policy.violations));
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    } finally {
      try {
        Snapshot.deleteTree(directory);
      } catch (IOException failure) {
        throw new UncheckedIOException(failure);
      }
    }
  }

  /**
   * 行の手の位置（打った手・候補集合・逸脱）のうち、手出しとツモ切りの両方で切れる打牌が、切る方の手かを数える。
   *
   * @param twinDiscard 同値な打牌のうち切る方
   */
  public static RowTwins rowTwins(List<Row> rows, Variant variant, TwinDiscard twinDiscard) {
    Observation o = new Observation(variant);
    long played = 0, candidates = 0, deviations = 0;
    List<String> violations = new ArrayList<>();
    for (Row row : rows) {
      int[] legal = legal(o.bind(row.observation(), 0));
      if (pair(legal, row.played())) {
        played++;
        check(legal, row.played(), twinDiscard, row, "played", violations);
      }
      for (int a = 0; a < legal.length; a++) {
        if ((row.support() >>> a & 1) == 0 || a == row.played() || !pair(legal, a)) continue;
        candidates++;
        check(legal, a, twinDiscard, row, "candidate", violations);
      }
      if (row.deviation() >= 0 && row.deviation() != row.played() && pair(legal, row.deviation())) {
        deviations++;
        check(legal, row.deviation(), twinDiscard, row, "deviation", violations);
      }
    }
    return new RowTwins(played, candidates, deviations, violations);
  }

  /**
   * 行を、切り方に依らない形に直す（キーは半荘・判断 ID・行の種類・写しの根）。手の位置は同値な打牌の組の手出しの位置に直し、測った手の値は
   * その位置ごとに持つ。O は河のツモ切りの印が切り方で変わるので含めない。
   */
  public static Map<String, String> canonical(List<Row> rows, Variant variant) {
    Observation o = new Observation(variant);
    Map<String, String> result = new TreeMap<>();
    for (Row row : rows) {
      int[] legal = legal(o.bind(row.observation(), 0));
      StringBuilder text =
          new StringBuilder()
              .append(row.decisionClass())
              .append(' ')
              .append(row.purpose())
              .append(" played=")
              .append(hand(legal, row.played()))
              .append(" deviation=")
              .append(row.deviation() < 0 ? -1 : hand(legal, row.deviation()))
              .append(" q=")
              .append(Float.floatToRawIntBits(row.deviationProbability()))
              .append(" p=")
              .append(Float.floatToRawIntBits(row.propensity()))
              .append(" support=");
      for (int a = 0; a < legal.length; a++) if ((row.support() >>> a & 1) != 0) text.append(hand(legal, a)).append(',');
      text.append(" values=");
      for (int a = 0; a < legal.length; a++)
        if (row.measured(a)) text.append(hand(legal, a)).append(':').append(Float.floatToRawIntBits(row.values()[a])).append(',');
      text.append(" prior=");
      for (float p : row.prior()) text.append(Float.floatToRawIntBits(p)).append(',');
      String key = row.gameId() + ":" + row.decisionId() + ":" + row.source() + ":" + row.rootDecisionId();
      if (result.put(key, text.toString()) != null) throw new AssertionError("two rows share the key " + key);
    }
    return result;
  }

  /** 同値な打牌の組で同じ値（組の手出しの行動の番号）。打牌でない手はその番号。 */
  static int twinKey(int code) {
    Action action = Action.of(code);
    if (!Twins.discard(action)) return code;
    return Action.discard(action.tile(), action.selection().red(), false, action.type() == Action.Type.RIICHI_DAHAI);
  }

  /** 同値な打牌の組の手出しの位置（組の無い手はその位置）。 */
  private static int hand(int[] legal, int a) {
    return TwinDiscard.TEDASHI.play(legal, a);
  }

  private static void check(int[] legal, int a, TwinDiscard twinDiscard, Row row, String what, List<String> violations) {
    boolean tsumogiri = Action.of(legal[a]).selection().tsumogiri();
    if (tsumogiri != (twinDiscard == TwinDiscard.TSUMOGIRI))
      violations.add(row.source() + " " + row.gameId() + ":" + row.decisionId() + " " + what + " " + Action.of(legal[a]));
  }

  private static int[] legal(Observation o) {
    int[] legal = new int[o.legalActions()];
    for (int a = 0; a < legal.length; a++) legal[a] = o.action(a).code();
    return legal;
  }

  private static List<Row> read(Path directory, Variant variant) throws IOException {
    CohortWriter.Manifest manifest = CohortWriter.readManifest(directory);
    RowCodec codec = new RowCodec(manifest.ledgerFeatures(), variant);
    List<Row> rows = new ArrayList<>();
    for (CohortWriter.Shard shard : manifest.shards()) {
      var frames = ByteBuffer.wrap(Files.readAllBytes(directory.resolve(shard.file())));
      while (frames.hasRemaining()) rows.add(codec.decode(frames));
    }
    return rows;
  }

  private static Sampling sampling() {
    Sampling base = TestSampling.grouped(0.25, 0.05, 0.05, 0.05, 0.15, 1.0, 6, false, 0.3, 0.2, 0.2, 0.1, 0.5, 2);
    boolean[] measureAll = new boolean[DecisionClass.COUNT];
    measureAll[DecisionClass.DISCARD_LATE.ordinal()] = true;
    RollInRule[] rollIn = base.rollIn().clone();
    rollIn[DecisionClass.DISCARD_MID.ordinal()] = RollInRule.uniform(0.2, 0.1, DeviationTarget.OTHER, 0.5, 2);
    return new Sampling(base.holdoutFraction(), base.labelRates(), measureAll, base.maxCandidates(), base.epsilonC(), rollIn);
  }

  /** 手出しとツモ切りの両方で切れる打牌か。 */
  private static boolean pair(int[] legal, int a) {
    return TwinDiscard.TSUMOGIRI.play(legal, a) != TwinDiscard.TEDASHI.play(legal, a);
  }

  /**
   * 同値な打牌に同じ s と p0 を付け、σ をその貪欲（同点は p0、それも同じなら番号の小さい手）にする inline の方策。打牌の s は打った後の向聴の
   * 負に、組ごとの小さな揺らぎを足したもの。立直の打牌は、手出しとツモ切りの両方で切れるときだけ 2 上げて、ほかは 0.5 下げる（黙聴の聴牌が
   * 続き、立直の打牌の手出しとツモ切りを多く作る）。和了は必ず選び、鳴き・槓・北抜き・九種九牌は選ばない。経路ごとに、手番の問い合わせの
   * 合法手を覚え、次の問い合わせの O の打牌を確かめる。
   */
  private static final class TwinBlindPolicy implements PolicyService {
    private final TwinDiscard expected;
    private final ThreadLocal<Observation> readers;
    private final ThreadLocal<Shapes> shapes;
    private final Map<Object, Turn> turns = new ConcurrentHashMap<>();
    final LongAdder mainTwins = new LongAdder(), branchTwins = new LongAdder(), riichiTwins = new LongAdder();
    final Queue<String> violations = new ConcurrentLinkedQueue<>();

    /** 手番の問い合わせ。 */
    private record Turn(int[] legal, int seat, int kyoku, int honba, int events) {}

    TwinBlindPolicy(Variant variant, TwinDiscard expected) {
      this.expected = expected;
      readers = ThreadLocal.withInitial(() -> new Observation(variant));
      shapes = ThreadLocal.withInitial(() -> new Shapes(variant));
    }

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
      Observation o = request.bind(readers.get());
      check(request.owner(), o);
      int n = request.legalCount();
      int[] legal = request.legal();
      Shapes hand = shapes.get().load(o);
      float[] scores = new float[n], prior = new float[n];
      double sum = 0;
      for (int a = 0; a < n; a++) {
        long key = twinKey(legal[a]);
        float noise = 0.1f * (float) CrnSeeds.unit(SeedMixer.mix64(key * 31 + request.seat()));
        Action action = Action.of(legal[a]);
        boolean pair = pair(legal, a);
        scores[a] =
            noise
                + switch (action.type()) {
                  case TSUMO_AGARI, RON_AGARI -> 10f;
                  case PASS -> 0f;
                  case DAHAI -> -hand.shantenAfter(action.tile());
                  case RIICHI_DAHAI -> -hand.shantenAfter(action.tile()) + (pair ? 2f : -0.5f);
                  default -> -10f;
                };
        prior[a] = 1f + (float) CrnSeeds.unit(SeedMixer.mix64(key * 17 + 5));
        sum += prior[a];
      }
      for (int a = 0; a < n; a++) prior[a] = (float) (prior[a] / sum);
      float[] sigma = new float[n];
      sigma[Greedy.argmax(scores, prior, n)] = 1f;
      request.scores = scores;
      request.sigma = sigma;
      if (request.retain()) request.prior = prior;
      if (o.decisionKind() == EngineDecisionKind.TURN)
        turns.put(request.owner(), new Turn(legal.clone(), o.observer(), o.kyoku(), o.honba(), o.events()));
    }

    /** 同じ経路の前の手番で切った打牌（O の次のイベント）を確かめる。 */
    private void check(Object owner, Observation o) {
      Turn turn = turns.remove(owner);
      if (turn == null || o.kyoku() != turn.kyoku() || o.honba() != turn.honba() || o.events() <= turn.events()) return;
      int event = turn.events();
      if (o.eventKind(event) != EventKind.DISCARD || o.absoluteSeat(o.eventAuthor(event)) != turn.seat()) return;
      int tile = o.discardTile(event);
      boolean red = o.discardAka(event), riichi = o.discardRiichi(event);
      if (!holds(turn.legal(), Action.discard(tile, red, false, riichi))
          || !holds(turn.legal(), Action.discard(tile, red, true, riichi))) return;
      boolean main = ((Run<?>) owner).main();
      (main ? mainTwins : branchTwins).increment();
      if (riichi) riichiTwins.increment();
      if (o.discardTsumogiri(event) != (expected == TwinDiscard.TSUMOGIRI))
        violations.add((main ? "main " : "branch ") + owner + " event " + event + " " + Action.discard(tile, red, false, riichi));
    }

    private static boolean holds(int[] legal, int code) {
      for (int action : legal) if (action == code) return true;
      return false;
    }
  }

  /** 判断の席の手牌の形と、打った後の向聴。読む O を替えて使い回すので、スレッドごとに 1 つ持つ。 */
  private static final class Shapes {
    private final ShapeAnalyzer analyzer;
    private final HandShape hand = new HandShape(), after = new HandShape();
    private final int[] counts = new int[Tiles.KINDS];

    Shapes(Variant variant) {
      analyzer = new ShapeAnalyzer(variant);
    }

    /** O の純手牌を読み、面子の数を枚数から決める（純手牌は 14 枚か 13 枚から面子ごとに 3 枚減る）。 */
    Shapes load(Observation o) {
      o.handCounts(counts);
      hand.clear();
      int tiles = 0;
      for (int kind = 0; kind < counts.length; kind++)
        for (int copy = 0; copy < counts[kind]; copy++, tiles++) hand.add(kind);
      for (int meld = (HandShape.MAX_SIZE - tiles) / HandShape.MELD_SIZE; meld > 0; meld--) hand.addMeld();
      return this;
    }

    /** 指定した種類を 1 枚打った後の向聴。 */
    int shantenAfter(int kind) {
      after.copyFrom(hand);
      after.remove(kind);
      return analyzer.shanten(after);
    }
  }

  /** 主経路の判断ごとの打った手（同値な打牌の組の手出しの番号）。 */
  private static final class MainChoices implements GameObserver {
    final Map<String, Integer> choices = new ConcurrentHashMap<>();

    @Override
    public void decided(GameKey game, SeatView view, Decisions decisions, int slot, int choice) {
      choices.put(game.wall() + ":" + decisions.id(slot) + ":" + slot, twinKey(decisions.legal(slot, choice)));
    }

    @Override
    public void gameEnded(GameKey game, TableState last) {}
  }
}
