package com.epsilon.classic.rulebot;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.classic.RandomPlay;
import com.epsilon.classic.model.SampleDecisions;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 三人打ちと四人打ちのルールベースの判断・終局・再現性・メモリ確保を検査する。 */
public class RuleBotTest {

  private static final Logger LOG = LoggerFactory.getLogger(RuleBotTest.class);

  private static final Variant VARIANT = Variant.TENHOU_4P;

  /** 1 つの半荘のステップ数の上限（局末と判断の時点をそれぞれ 1 ステップと数える）。 */
  private static final int GAME_STEPS = 100_000;

  private static final int WARM_UP = 2_000;
  private static final int MEASURED = 4_000;

  /**
   * メモリ確保の計測そのものの誤差を見込んだ余裕（バイト）。測るのは 4000 ステップなので、1 ステップごとに 16 バイト以上を 1 回
   * 確保するだけで 64 KB になり、余裕を大きく超える。
   */
  private static final long SLACK_BYTES = 4_096;

  @DataProvider
  public static Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /** どの半荘も持ち点と供託の合計が変わらず、3 つの半荘を通して和了と立直が 1 回以上ある。 */
  @Test(dataProvider = "variants")
  public void ruleBotsPlaySeededGamesToTheEnd(Variant variant) {
    int rounds = 0;
    int nuki = 0;
    int wins = 0;
    int riichi = 0;
    for (long seed = 1; seed <= 3; seed++) {
      BotTable table = new BotTable(new RuleBot(variant), seed);
      table.finishGame(GAME_STEPS);
      TableState last = table.engine().roundEnd().next();
      assertTrue(last.gameOver());
      int total = Variant.DEPOSIT * last.kyotaku();
      int[] scores = new int[variant.seats()];
      for (int seat = 0; seat < scores.length; seat++) {
        scores[seat] = last.score(seat);
        total += scores[seat];
      }
      assertEquals(total, variant.seats() * variant.startPoints(), Arrays.toString(scores));
      LOG.info(
          "Rule bot game {}: {} decisions, {} rounds, {} wins, {} riichi, final scores {}",
          seed,
          table.decisions(),
          table.rounds(),
          table.wins(),
          table.riichi(),
          Arrays.toString(scores));
      rounds += table.rounds();
      nuki += table.nuki();
      wins += table.wins();
      riichi += table.riichi();
    }
    if (variant == Variant.TENHOU_3P) assertTrue(nuki > 0, "no nuki");
    assertTrue(
        wins > 0 && (variant == Variant.TENHOU_3P || riichi > 0),
        wins + " wins and " + riichi + " riichi in " + rounds + " rounds");
  }

  @Test(dataProvider = "variants")
  public void aSeedPlaysTheSameGameTwice(Variant variant) {
    BotTable first = new BotTable(new RuleBot(variant), 7);
    BotTable second = new BotTable(new RuleBot(variant), 7);
    first.finishGame(GAME_STEPS);
    second.finishGame(GAME_STEPS);
    assertEquals(second.decisions(), first.decisions());
    assertEquals(second.digest(), first.digest());
    for (int seat = 0; seat < variant.seats(); seat++) {
      assertEquals(
          second.engine().roundEnd().next().score(seat),
          first.engine().roundEnd().next().score(seat));
    }
  }

  @Test(timeOut = 900_000)
  public void actAllocatesNothingOnceWarm() {
    ThreadMXBean threads =
        ManagementFactory.getThreadMXBean() instanceof ThreadMXBean bean ? bean : null;
    if (threads == null
        || !threads.isThreadAllocatedMemorySupported()
        || !threads.isThreadAllocatedMemoryEnabled()) {
      throw new SkipException("Per-thread allocation counters are not available");
    }
    BotTable table = new BotTable(new RuleBot(VARIANT), 11);
    for (int step = 0; step < WARM_UP; step++) {
      table.step();
    }
    long before = threads.getCurrentThreadAllocatedBytes();
    for (int step = 0; step < MEASURED; step++) {
      table.step();
    }
    long allocated = threads.getCurrentThreadAllocatedBytes() - before;
    assertTrue(
        allocated < SLACK_BYTES,
        allocated
            + " bytes over "
            + MEASURED
            + " steps ("
            + table.decisions()
            + " decisions, digest "
            + table.digest()
            + ")");
  }

  /** ランダムな自己対局のステップ数。 */
  private static final int RANDOM_STEPS = 60_000;

  @Test
  public void everyRandomDecisionGetsALegalChoiceAndNorthIsExtracted() {
    RuleBot bot = new RuleBot(Variant.TENHOU_3P);
    RandomPlay play = new RandomPlay(Variant.TENHOU_3P, 23);
    ObservationWriter writer = new ObservationWriter(Variant.TENHOU_3P);
    ObservationBatch batch = new ObservationBatch(Variant.TENHOU_3P, 1_024);
    Observation observation = new Observation(Variant.TENHOU_3P);
    int decisions = 0;
    int nukiOffered = 0;
    int nukiResponses = 0;
    for (int step = 0; step < RANDOM_STEPS; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        Decisions open = play.engine.decisions();
        for (int slot = 0; slot < open.size(); slot++) {
          batch.clear();
          observation.bind(
              batch, writer.capture(play.engine.view(open.seat(slot)), open, slot, batch));
          int choice = bot.act(observation);
          assertTrue(
              choice >= 0 && choice < open.legalCount(slot),
              choice + " of " + open.legalCount(slot));
          boolean win = false;
          boolean nuki = false;
          for (int index = 0; index < open.legalCount(slot); index++) {
            Action.Type type = Action.of(open.legal(slot, index)).type();
            win |= type == Action.Type.TSUMO_AGARI || type == Action.Type.RON_AGARI;
            nuki |= type == Action.Type.NUKI;
          }
          if (nuki && !win) {
            nukiOffered++;
            assertEquals(Action.of(open.legal(slot, choice)).type(), Action.Type.NUKI);
          }
          nukiResponses += open.kind(slot) == EngineDecisionKind.NUKI_RESPONSE ? 1 : 0;
          decisions++;
        }
      }
      play.step();
    }
    LOG.info(
        "Three-player rule bot: {} random decisions, {} nuki offered, {} nuki responses",
        decisions,
        nukiOffered,
        nukiResponses);
    assertTrue(decisions > 20_000, decisions + " decisions");
    assertTrue(nukiOffered > 0, "no nuki offered");
  }

  private final RuleBot bot = new RuleBot(RuleScene.VARIANT);

  /** 配牌で和了の形になっていれば（天和）、ツモ和了する。 */
  @Test
  public void aCompleteHandWinsByTsumo() {
    RuleScene scene =
        new RuleScene().hand(0, "123m456m789m123p5p").draws("5p").indicator("2z").start();
    assertChoice(scene, 0, "TSUMO_AGARI");
  }

  /** 和了牌が打たれたら、チーできてもロンする。 */
  @Test
  public void aWinningDiscardIsRonnedBeforeAnyCall() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "5p147m258s369s123z")
            .hand(1, "123m456m789m234p5p")
            .draws("4z")
            .indicator("2z")
            .start()
            .play("0:DAHAI(5p)");
    assertTrue(scene.legal(1).contains("CHI(3p4p5p 5p)"), scene.legal(1).toString());
    assertChoice(scene, 1, "RON_AGARI");
  }

  /** ダマではロンできる役が無い聴牌は、立直する。 */
  @Test
  public void aTenpaiWithoutAYakuDeclaresRiichi() {
    RuleScene scene =
        new RuleScene().hand(0, "123m789p555s99s46p").draws("1z").indicator("2z").start();
    assertChoice(scene, 0, "RIICHI_TSUMOGIRI(E)");
  }

  /** ダマのロンで 7700 点以上ある聴牌（ここでは親の断么九・平和・三色）は、立直しない。 */
  @Test
  public void aValuableDamatenStaysDama() {
    RuleScene scene =
        new RuleScene().hand(0, "234m234p234s67s55p").draws("1z").indicator("2z").start();
    assertTrue(scene.legal(0).contains("RIICHI_TSUMOGIRI(E)"), scene.legal(0).toString());
    assertChoice(scene, 0, "DAHAI_TSUMOGIRI(E)");
  }

  /** 他家が立直したら、遠い手（3 向聴）は浮いた字牌ではなく現物を打って降りる。宣言牌はポンせずに見送る。 */
  @Test
  public void aFarHandFoldsToARiichiWithTheSafeTile() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "147m258p369s1357z")
            .hand(1, "123m456m789m11z23p")
            .hand(2, "99s13m57p24s68m157z")
            .hand(3, "258m147p369p2468s")
            .draws("6z9s2z")
            .indicator("3z")
            .start()
            .play("0:DAHAI_TSUMOGIRI(F)")
            .play("1:RIICHI_TSUMOGIRI(9s)");
    assertTrue(scene.legal(2).contains("PON(9s)"), scene.legal(2).toString());
    assertChoice(scene, 2, "PASS");
    scene.play();
    assertChoice(scene, 2, "DAHAI(9s)");
  }

  /** 宣言牌が通る前から、立直を宣言した席を脅威として扱う。脅威が無ければポンする役牌の対子でも、宣言牌はポンしない。 */
  @Test
  public void aRiichiDeclarationTileIsNotCalledEvenForADragonPon() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "147m258p369s2367z")
            .hand(1, "123m456m789m23p15z")
            .hand(2, "55z234m56p78s9m124z")
            .hand(3, "258m147p369p2468s")
            .draws("9p1z")
            .indicator("8p")
            .start()
            .play("0:DAHAI_TSUMOGIRI(9p)")
            .play("1:RIICHI_DAHAI(P)");
    assertTrue(scene.legal(2).contains("PON(P)"), scene.legal(2).toString());
    assertChoice(scene, 2, "PASS");
  }

  /** 役牌の対子は、ポンして向聴が進むならポンする。 */
  @Test
  public void aDragonPairIsPonnedWhenItAdvancesTheHand() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "5z147m258p369s246z")
            .hand(2, "55z234m56p78s9m124z")
            .draws("1s")
            .indicator("8p")
            .start()
            .play("0:DAHAI(P)");
    assertChoice(scene, 2, "PON(P)");
  }

  /** 鳴いても役の見込みが無い手（么九牌があり、3 色にまたがる）は、チーしない。 */
  @Test
  public void aChiWithoutAYakuIsDeclined() {
    RuleScene scene =
        new RuleScene()
            .hand(0, "147m258p369s1357z")
            .hand(1, "46s123m789p13s9m14z")
            .draws("5s")
            .indicator("8p")
            .start()
            .play("0:DAHAI_TSUMOGIRI(5s)");
    assertTrue(scene.legal(1).contains("CHI(4s5s6s 5s)"), scene.legal(1).toString());
    assertChoice(scene, 1, "PASS");
  }

  /**
   * 役が無くてロンできない和了牌を見送っても、立直していなければ次の自分の打牌で見逃しの振聴は消えるので、打牌を比べるときはロンを
   * 数える。6m をポンした聴牌で 3p を引いたとき、3p を打つ両面待ち（役があるのは 4p だけ）より、2p を打つ 5s と 3p のシャンポン待ち
   * （ロンの平均の点数が高い）を選ぶ。見送りが無いときと同じ手になる。
   */
  @Test
  public void aMissedWinDoesNotCostTheRonOfTheNextDiscard() {
    assertChoice(openTenpaiAfter("9p"), 1, "DAHAI(2p)");
    RuleScene missed = openTenpaiAfter("1p");
    assertTrue(missed.observe(1).furitenTemporary(), "the 1p was a wait without a yaku");
    assertChoice(missed, 1, "DAHAI(2p)");
  }

  /**
   * 席 1 が 6m をポンして 1p と 4p の待ちの聴牌になり、下家がツモ切りした後、次の自分のツモで 3p を引いた場面を作る。ドラは 5s。
   *
   * @param shimochaDiscard 下家がツモ切りする牌
   */
  private static RuleScene openTenpaiAfter(String shimochaDiscard) {
    return new RuleScene()
        .hand(0, "147m258p369s1235z")
        .hand(1, "66m234m678s55s23p4z")
        .hand(2, "159m159p19s12345z")
        .hand(3, "258m147p369p2468s")
        .draws("6m" + shimochaDiscard + "7z6z3p")
        .indicator("4s")
        .start()
        .play("0:DAHAI_TSUMOGIRI(6m)")
        .play("1:PON(6m)")
        .play("1:DAHAI(N)")
        .play("2:DAHAI_TSUMOGIRI(" + shimochaDiscard + ")")
        .play("3:DAHAI_TSUMOGIRI(C)")
        .play("0:DAHAI_TSUMOGIRI(F)");
  }

  /**
   * ルールベースの打ち手が、その席の判断で期待どおりの手を選ぶことを確かめる。同じ判断の記録で 2 回選んでも、評価が最も高い手を取っても、
   * 同じ手になることも確かめる。
   *
   * @param scene 判断を待っている場面
   * @param seat 判断する席
   * @param expected 選ぶはずの手の名前（{@code Action.toString} の形）
   */
  private void assertChoice(RuleScene scene, int seat, String expected) {
    Observation o = scene.observe(seat);
    int chosen = bot.act(o);
    assertEquals(o.action(chosen).toString(), expected, scene.legal(seat).toString());
    assertEquals(bot.act(o), chosen);
    assertEquals(bot.evaluate(o).greedy(), chosen);
  }

  /** サンプルに集める判断の、種類ごとの最大数。種類はロン・ツモ・九種九牌・鳴き・槓・立直・打牌だけの 7 つ。 */
  private static final int PER_KIND = 64;

  /** サンプルを集めるときに進めるステップ数の上限。 */
  private static final int STEPS = 400_000;

  /** サンプルを回る回数。1 回目でウォームアップし、2 回目からの確保を数える。 */
  private static final int PASSES = 4;

  /**
   * メモリ確保の計測そのものの誤差を見込んだ余裕（バイト）。確保を数える間に手を選ぶ回数は 1000 回を超えるので、1 回選ぶごとに
   * 16 バイト以上を 1 回確保するだけで 16 KB になり、余裕を大きく超える。
   */
  private static final long WAIT_SECONDS = 300;

  @Test(timeOut = 600_000)
  public void twoThreadsChooseWhatOneThreadChooses() throws Exception {
    SampleDecisions samples = new SampleDecisions(29, PER_KIND, STEPS);
    assertTrue(samples.size() > 250, samples.counts().toString());
    int[] expected = new int[samples.size()];
    RuleBot reference = new RuleBot(SampleDecisions.VARIANT);
    Observation o = new Observation(SampleDecisions.VARIANT);
    for (int row = 0; row < samples.size(); row++) {
      expected[row] = reference.act(samples.bind(o, row));
      assertTrue(expected[row] >= 0 && expected[row] < o.legalActions(), "row " + row);
    }
    RuleBot shared = new RuleBot(SampleDecisions.VARIANT);
    CyclicBarrier start = new CyclicBarrier(2);
    ExecutorService threads = Executors.newFixedThreadPool(2);
    try {
      List<Future<Pass>> passes =
          List.of(
              threads.submit(() -> play(shared, samples, start, false)),
              threads.submit(() -> play(shared, samples, start, true)));
      for (Future<Pass> future : passes) {
        Pass pass = future.get(WAIT_SECONDS, TimeUnit.SECONDS);
        assertEquals(pass.chosen(), expected);
        assertEquals(pass.changes(), 0, "moves that changed between passes");
        if (pass.allocated() >= 0) {
          assertTrue(
              pass.allocated() < SLACK_BYTES,
              pass.allocated() + " bytes over " + pass.measured() + " acts");
        }
      }
    } finally {
      threads.shutdownNow();
    }
  }

  /**
   * 1 つのスレッドの結果。
   *
   * @param chosen 1 回目に選んだ手（サンプルの番号の順）
   * @param changes 2 回目から、1 回目と違う手を選んだ数
   * @param allocated 2 回目から確保したバイト数（スレッドごとの数を読めなければ −1）
   * @param measured 2 回目から手を選んだ回数
   */
  private record Pass(int[] chosen, int changes, long allocated, int measured) {}

  /** 1 つのスレッドで、サンプルの全ての判断を番号の順か逆の順に決まった回数だけ回り、手を選ぶ。 */
  private static Pass play(
      RuleBot bot, SampleDecisions samples, CyclicBarrier start, boolean reverse) throws Exception {
    ThreadMXBean counter =
        ManagementFactory.getThreadMXBean() instanceof ThreadMXBean bean ? bean : null;
    boolean counted =
        counter != null
            && counter.isThreadAllocatedMemorySupported()
            && counter.isThreadAllocatedMemoryEnabled();
    Observation o = new Observation(SampleDecisions.VARIANT);
    int size = samples.size();
    int[] chosen = new int[size];
    int changes = 0;
    long before = 0;
    start.await(WAIT_SECONDS, TimeUnit.SECONDS);
    for (int pass = 0; pass < PASSES; pass++) {
      if (pass == 1 && counted) {
        before = counter.getCurrentThreadAllocatedBytes();
      }
      for (int step = 0; step < size; step++) {
        int row = reverse ? size - 1 - step : step;
        int move = bot.act(samples.bind(o, row));
        if (pass == 0) {
          chosen[row] = move;
        } else if (move != chosen[row]) {
          changes++;
        }
      }
    }
    long allocated = counted ? counter.getCurrentThreadAllocatedBytes() - before : -1;
    return new Pass(chosen, changes, allocated, (PASSES - 1) * size);
  }
}
