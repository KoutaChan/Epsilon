package com.epsilon.prism.collect;

import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.prism.core.CrnKeys;
import com.epsilon.prism.core.CrnScheme;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.rows.Row;
import com.epsilon.util.SeedMixer;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.SplittableRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 運の精算の帳簿（{@link EventLedger}）の平均 0: 多くの局で全ての席・欄の局で束ねた平均が 0 に近い、方策の項は b の上で平均 0 で選んだ頻度が b に
 * 合う、確率的丸めは偏らず fp16 で厳密、s の切りと足さない判断、有限でない増分は例外、TSV の形。
 */
public class EventLedgerTest {
  private static final Logger log = LoggerFactory.getLogger(EventLedgerTest.class);

  /** 試験の s: 手の ID と席のハッシュ（[-1, 1)。どの s でも方策の項は平均 0 のはず）。 */
  private static float score(Action action, int seat) {
    return (float) (2 * CrnSeeds.unit(SeedMixer.mix64(action.toIndex() * 131L + seat)) - 1);
  }

  /** 1 局を打って帳簿を付けた結果。 */
  private record Hand(List<Slot> slots, int draws) {}

  /**
   * 半荘の最初の局を打つ。最初の境界の後に 4 席の仮の枠を作り、GameRunner と同じ順に observe（境界ごと）・policy（強制でない判断ごと）・handEnd
   * （局末）を呼ぶ。手は安い打ち方の one-hot に床 0.1 の b を混ぜて {@link CrnKeys#select} で引く。
   */
  private static Hand playHand(long seed, LedgerGames.Style style, HandShapeAnalyzer shapes, StageTimer timer) {
    var engine = new GameEngine(SeedMixer.mix64(seed * 2_147_483_647L));
    engine.getState().enableSeatStreamDraws();
    var tracker = new EventLedger.Tracker();
    var random = new SplittableRandom(seed);
    var behaviour = new Behaviour(0.1, 0, CrnScheme.V1);
    var cursor = new CrnKeys.Cursor(CrnKeys.KIND_CLASSES);
    long handKey = CrnKeys.handKey(7L, seed, 0);
    List<Slot> slots = null;
    int draws = 0;
    GameStepResult step = engine.stepHanchan();
    while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
      List<EngineDecisionPoint> points = awaiting.decisions();
      GameState state = engine.getState();
      tracker.observe(state, points, slots == null ? List.of() : slots, null);
      if (slots == null) slots = LedgerGames.slots();
      else if (state.turnEvent() instanceof TurnEvent.Draw && points.size() == 1) draws++;
      var selections = new EngineSelectionBuffer();
      for (EngineDecisionPoint point : points) {
        int seat = point.player(), kindClass = CrnKeys.kindClass(point.kind());
        int ordinal = cursor.next(seat, kindClass);
        List<Action> legal = point.legalActions();
        int n = legal.size();
        int chosen = 0;
        if (n > 1) {
          float[] sigma = new float[n];
          sigma[LedgerGames.choose(state, point, style, random, shapes)] = 1f;
          float[] p = behaviour.policy(sigma, n);
          float[] scores = new float[n];
          for (int a = 0; a < n; a++) scores[a] = score(legal.get(a), seat);
          chosen = CrnKeys.select(CrnKeys.decisionKey(handKey, seat, kindClass, ordinal), legal, p, 0);
          tracker.policy(seat, scores, p, chosen, n, slots);
        }
        selections.add(point.id(), legal.get(chosen));
      }
      step = engine.commitDecisions(selections);
    }
    RoundSettlement settlement =
        switch (step) {
          case GameStepResult.RoundSettled settled -> settled.settlement();
          case GameStepResult.HanchanEnded ended -> ended.settlement();
          default -> throw new IllegalStateException("Unexpected boundary " + step);
        };
    tracker.handEnd(engine.getState(), settlement, slots);
    if (timer != null) tracker.flush(timer);
    return new Hand(slots, draws);
  }

  /**
   * 20000 局（各種の最初の局）で、席ごとの枠の帳簿 L_{h,i} の平均の局で束ねた z が全ての席・欄で |z| &lt; 4.5（0 でない局が 30 未満の席・欄は
   * 飛ばす）。z は席ごとに取る（局で 4 席を足すと、裏ドラの項は重み ω の和が 4 席で 0 なので和が恒等的に 0 になり、試験にならない）。主な欄
   * （受け入れ・ツモ・表示・裏・方策）は全ての席で十分に起きる。
   */
  @Test(timeOut = 600_000)
  public void everyTermIsCentredOverManyHands() {
    var shapes = new HandShapeAnalyzer();
    int features = EventLedger.FEATURES, seats = GameState.NUM_PLAYERS, hands = 20_000;
    double[][] sum = new double[seats][features], square = new double[seats][features];
    int[][] nonzero = new int[seats][features];
    for (long seed = 1; seed <= hands; seed++) {
      Hand hand = playHand(seed, seed % 3 == 0 ? LedgerGames.KANS : LedgerGames.PLAIN, shapes, null);
      for (Slot slot : hand.slots()) {
        int seat = slot.target().seat();
        for (int f = 0; f < features; f++) {
          double value = slot.ledger()[f];
          Assert.assertTrue(Double.isFinite(value), "seat " + seat + " " + EventLedger.NAMES.get(f));
          sum[seat][f] += value;
          square[seat][f] += value * value;
          if (value != 0) nonzero[seat][f]++;
        }
      }
    }
    int tested = 0;
    for (int seat = 0; seat < seats; seat++)
      for (int f = 0; f < features; f++) {
        if (nonzero[seat][f] < 30) continue;
        tested++;
        double mean = sum[seat][f] / hands, variance = square[seat][f] / hands - mean * mean;
        double z = mean / Math.sqrt(variance / hands);
        Assert.assertTrue(
            Math.abs(z) < 4.5,
            "seat " + seat + " " + EventLedger.NAMES.get(f) + " mean=" + mean + " z=" + z + " nonzero=" + nonzero[seat][f]);
      }
    for (int seat = 0; seat < seats; seat++)
      for (int f :
          new int[] {
            EventLedger.OWN_IMPROVE,
            EventLedger.OWN_TSUMO,
            EventLedger.OWN_DRAWN_DANGER,
            EventLedger.OPP_TSUMO,
            EventLedger.DORA_SELF,
            EventLedger.URA,
            EventLedger.POLICY_SELF,
            EventLedger.POLICY_OPPONENT
          })
        Assert.assertTrue(
            nonzero[seat][f] >= 30,
            "seat " + seat + " " + EventLedger.NAMES.get(f) + " happened in " + nonzero[seat][f] + " hands");
    Assert.assertTrue(tested >= 48, "tested seat features " + tested);
  }

  /** 方策の項: 乱数の p（床あり・0・NaN・負を含む）と s で、10^5 個の鍵の {@link CrnKeys#select} の δ の平均の z &lt; 4.5、選んだ頻度が b に合う。 */
  @Test
  public void policyTermIsCentred() {
    List<Action> legal = List.of(Action.dahai(0), Action.dahai(9), Action.dahai(18), Action.dahai(27), Action.dahai(31));
    int n = legal.size(), keys = 100_000;
    var random = new SplittableRandom(5);
    for (int config = 0; config < 4; config++) {
      float[] p = new float[n], scores = new float[n];
      for (int a = 0; a < n; a++) {
        p[a] = config == 0 ? (a == 2 ? 0.96f : 0.01f) : (float) random.nextDouble();
        scores[a] = (float) (6 * random.nextDouble() - 3);
      }
      if (config == 2) {
        p[0] = 0;
        p[1] = Float.NaN;
        scores[4] = 20f;
      }
      if (config == 3) {
        p[3] = -0.5f;
        scores[1] = -15f;
      }
      double weight = 0;
      for (float probability : p) weight += probability > 0 ? probability : 0;
      long[] counts = new long[n];
      double sum = 0, square = 0;
      for (int k = 0; k < keys; k++) {
        long key = SeedMixer.mix64(config * 1_000_000_007L + k);
        int chosen = CrnKeys.select(key, legal, p, 0);
        counts[chosen]++;
        Assert.assertTrue(EventLedger.policyCounts(scores, p, n));
        double delta = EventLedger.policyDelta(scores, p, chosen, n);
        Assert.assertTrue(Math.abs(delta) <= 2 * EventLedger.POLICY_CLIP);
        sum += delta;
        square += delta * delta;
      }
      double mean = sum / keys, variance = square / keys - mean * mean;
      Assert.assertTrue(Math.abs(mean / Math.sqrt(variance / keys)) < 4.5, "config " + config + " mean " + mean);
      double[] expected = new double[n];
      for (int a = 0; a < n; a++) expected[a] = keys * (p[a] > 0 ? p[a] : 0) / weight;
      double p0 = LedgerGames.chiSquareUpper(LedgerGames.chiSquare(counts, expected), LedgerGames.degrees(expected));
      Assert.assertTrue(p0 > 1e-4, "config " + config + " frequencies " + Arrays.toString(counts));
    }
  }

  /** 確率的丸め: 決まった x（正負・正規・非正規・2 の冪の際・fp16 で厳密な値）を 10^5 個の硬貨で丸め、平均の z &lt; 4.5、結果は fp16 で厳密。 */
  @Test
  public void stochasticRoundingIsUnbiased() {
    double[] xs = {0.3, -0.3, 1e-6, -2.5e-5, 6.1e-5, Math.scalb(1.0, -14), 1 + Math.scalb(1.0, -11), 3.0, 1024.3, -777.77, 65000.7, -65000.7};
    int coins = 100_000;
    for (int i = 0; i < xs.length; i++) {
      double x = xs[i], sum = 0, square = 0;
      for (int k = 0; k < coins; k++) {
        float rounded = EventLedger.half(x, EventLedger.coin(11L, i, k, k % 7, k % EventLedger.FEATURES));
        Assert.assertEquals(Float.float16ToFloat(Float.floatToFloat16(rounded)), rounded, 0f, "x " + x);
        sum += rounded;
        square += (double) rounded * rounded;
      }
      double mean = sum / coins, variance = Math.max(0, square / coins - mean * mean);
      if (variance < 1e-30) {
        Assert.assertEquals(mean, x, Math.abs(x) * 1e-12, "x " + x);
        continue;
      }
      double z = (mean - x) / Math.sqrt(variance / coins);
      Assert.assertTrue(Math.abs(z) < 4.5, "x " + x + " mean " + mean + " z " + z);
    }
    Assert.assertEquals(EventLedger.half(0, 0.5), 0f);
    Assert.assertTrue(Float.isNaN(EventLedger.half(Double.NaN, 0.5)));
    Assert.assertEquals(EventLedger.half(3.0, 0.999), 3f);
  }

  /**
   * 方策の項の切りと飛ばし: |s| &gt; {@value EventLedger#POLICY_CLIP} の手は切った s̄ で計算する、b &gt; 0 の手の s が NaN・±∞ なら足さない（b = 0 の
   * 手なら足す）、p の NaN・0・負は重み 0（{@link CrnKeys#select} と同じ判定）、s のない判断は足さない。合法手の toIndex は重ならない。
   */
  @Test
  public void policyTermClipsAndSkips() {
    float[] scores = {20f, 0f, -30f}, p = {0.5f, 0.3f, 0.2f};
    Assert.assertTrue(EventLedger.policyCounts(scores, p, 3));
    Assert.assertEquals(EventLedger.policyDelta(scores, p, 0, 3), 8 - (0.5 * 8 + 0.2 * -8), 1e-6);
    Assert.assertEquals(EventLedger.policyDelta(scores, p, 2, 3), -8 - (0.5 * 8 + 0.2 * -8), 1e-6);
    Assert.assertFalse(EventLedger.policyCounts(new float[] {Float.NaN, 0f, 1f}, new float[] {0.5f, 0.5f, 0f}, 3));
    Assert.assertFalse(EventLedger.policyCounts(new float[] {Float.POSITIVE_INFINITY, 0f}, new float[] {0.5f, 0.5f}, 2));
    Assert.assertTrue(EventLedger.policyCounts(new float[] {Float.NaN, 0f, 1f}, new float[] {0f, 0.5f, 0.5f}, 3));
    Assert.assertEquals(EventLedger.policyDelta(new float[] {Float.NaN, 0f, 1f}, new float[] {0f, 0.5f, 0.5f}, 2, 3), 0.5, 1e-6);
    float[] odd = {Float.NaN, -1f, 0f, 0.7f, 0.3f};
    Assert.assertEquals(EventLedger.policyDelta(new float[] {9f, 9f, 9f, 1f, 0f}, odd, 3, 5), 1 - 0.7, 1e-6);
    List<Action> legal = List.of(Action.dahai(1), Action.dahai(2), Action.dahai(3), Action.dahai(4), Action.dahai(5));
    for (int k = 0; k < 10_000; k++) Assert.assertTrue(CrnKeys.select(SeedMixer.mix64(k), legal, odd, 0) >= 3);
    // Tracker: s のない判断と、b > 0 の手の s が有限でない判断は帳簿を変えない
    var tracker = new EventLedger.Tracker();
    List<Slot> slots = LedgerGames.slots();
    tracker.policy(0, null, p, 0, 3, slots);
    tracker.policy(0, new float[] {Float.NaN, 0f, 1f}, p, 1, 3, slots);
    for (Slot slot : slots) for (double value : slot.ledger()) Assert.assertEquals(value, 0.0);
    tracker.policy(1, scores, p, 0, 3, slots);
    for (Slot slot : slots) {
      int seat = slot.target().seat();
      Assert.assertEquals(slot.ledger()[EventLedger.POLICY_SELF], seat == 1 ? 5.6 : 0, 1e-6);
      Assert.assertEquals(slot.ledger()[EventLedger.POLICY_OPPONENT], seat == 1 ? 0 : 5.6, 1e-6);
    }
    // 実際の境界の合法手の toIndex は重ならない（b の正規化は手の ID の重なりがない前提）
    var shapes = new HandShapeAnalyzer();
    for (long seed = 1; seed <= 20; seed++) {
      var engine = new GameEngine(seed);
      var random = new SplittableRandom(seed);
      GameStepResult step = engine.stepHanchan();
      while (step instanceof GameStepResult.AwaitingDecisions awaiting) {
        for (EngineDecisionPoint point : awaiting.decisions()) {
          Set<Integer> ids = new HashSet<>();
          for (Action action : point.legalActions()) Assert.assertTrue(ids.add(action.toIndex()), point.toString());
        }
        step = engine.commitDecisions(LedgerGames.select(engine.getState(), awaiting.decisions(), LedgerGames.CALLS, random, shapes, null));
      }
    }
  }

  /** 有限でない増分は帳簿を付けずに例外にする（壊れた c。重みの和が ∞ の分布で方策の項が NaN になる場合も同じ）。fp16 の範囲の外も例外。 */
  @Test
  public void nonFiniteDeltaFails() {
    Assert.assertThrows(IllegalStateException.class, () -> EventLedger.add(new double[EventLedger.FEATURES], 3, Double.NaN));
    Assert.assertThrows(
        IllegalStateException.class, () -> EventLedger.add(new double[EventLedger.FEATURES], 3, Double.NEGATIVE_INFINITY));
    var tracker = new EventLedger.Tracker();
    List<Slot> slots = LedgerGames.slots();
    Assert.assertThrows(
        IllegalStateException.class,
        () -> tracker.policy(0, new float[] {0f, 1f}, new float[] {Float.POSITIVE_INFINITY, 0.5f}, 0, 2, slots));
    Assert.assertThrows(IllegalStateException.class, () -> EventLedger.half(70_000, 0.5));
    Assert.assertThrows(IllegalStateException.class, () -> EventLedger.half(Double.POSITIVE_INFINITY, 0.5));
  }

  /** 点の換算は天鳳の支払い（切り上げ満貫なし）: 子の満貫ツモは 2000/4000、親の満貫ツモは 4000 オール、栄和は子 8000・親 12000、本場と供託を足す。 */
  @Test
  public void paymentsFollowTheScoreTable() {
    Assert.assertEquals(EventLedger.tsumoGain(2000, 1, 0, 0, 0), 8000);
    Assert.assertEquals(EventLedger.tsumoGain(2000, 0, 0, 1, 2), 12000 + 300 + 2000);
    Assert.assertEquals(EventLedger.tsumoPayment(2000, 1, 0, 0, 1), 4000 + 100);
    Assert.assertEquals(EventLedger.tsumoPayment(2000, 1, 2, 0, 1), 2000 + 100);
    Assert.assertEquals(EventLedger.tsumoPayment(2000, 0, 2, 0, 0), 4000);
    Assert.assertEquals(EventLedger.ronPayment(2000, 1, 0, 0), 8000);
    Assert.assertEquals(EventLedger.ronPayment(2000, 0, 0, 2), 12000 + 600);
    Assert.assertEquals(EventLedger.kappa(100_000), EventLedger.CLIP);
    Assert.assertEquals(EventLedger.kappa(-3900), -3.9, 1e-12);
  }

  /**
   * -Dprism.ledger の TSV（{@link EventLedger#dumpLines}）: 測った手ごとに 1 行、欄は見出しと同じ数（半荘・判断・席・類・種別・行の種類・打った手・
   * 手・Y・帳簿の {@value EventLedger#FEATURES} 欄）で、値は渡した（行に書くのと同じ丸めた）値そのもの。測っていない手は書かない。
   */
  @Test
  public void dumpWritesOneLinePerMeasuredAction() {
    int features = EventLedger.FEATURES;
    var draft =
        new RowDraft(
            42,
            7,
            2,
            Kind.DISCARD,
            DecisionClass.DISCARD_MID,
            Row.Purpose.TRAIN,
            Row.Group.A,
            Row.Source.MAIN,
            1,
            3,
            new float[] {0.5f, 0.3f, 0.2f},
            new byte[] {0, 2},
            -1,
            Float.NaN,
            Float.NaN,
            -1,
            null);
    float[] values = {0.25f, Float.NaN, -0.5f};
    float[] ledger = new float[3 * features];
    Arrays.fill(ledger, Float.NaN);
    for (int a : new int[] {0, 2})
      for (int f = 0; f < features; f++) ledger[a * features + f] = (a * features + f) * 0.125f - 1;
    String[] header = EventLedger.dumpHeader().split("\n", -1);
    Assert.assertEquals(header.length, 2);
    Assert.assertEquals(header[1], "");
    String[] names = header[0].split("\t");
    Assert.assertEquals(names.length, 9 + features);
    Assert.assertEquals(List.of(names).subList(9, 9 + features), EventLedger.NAMES);
    String text = EventLedger.dumpLines(draft, values, ledger);
    Assert.assertTrue(text.endsWith("\n"));
    String[] lines = text.split("\n");
    Assert.assertEquals(lines.length, 2);
    int[] actions = {0, 2};
    for (int k = 0; k < lines.length; k++) {
      int a = actions[k];
      String[] fields = lines[k].split("\t");
      Assert.assertEquals(fields.length, names.length);
      Assert.assertEquals(
          List.of(fields).subList(0, 9),
          List.of("42", "7", "2", "DISCARD_MID", "DISCARD", "MAIN", "1", Integer.toString(a), Float.toString(values[a])));
      for (int f = 0; f < features; f++)
        Assert.assertEquals(Float.parseFloat(fields[9 + f]), ledger[a * features + f], 0f, "action " + a + " " + names[9 + f]);
    }
  }

  /**
   * 帳簿の CPU の量（{@link StageTimer.Stage#LEDGER} と同じ計り方）: 4 席に枠を持つ局を打ち、ツモ 1 回あたりの帳簿の時間をログに出す（行
   * "LEDGER_COST"）。値の上限は緩い目安（1 回 0.5 ms）。
   */
  @Test(timeOut = 300_000)
  public void ledgerCostPerDraw() {
    var shapes = new HandShapeAnalyzer();
    for (long seed = 1; seed <= 50; seed++) playHand(seed, LedgerGames.PLAIN, shapes, null);
    var timer = new StageTimer();
    long draws = 0;
    for (long seed = 1001; seed <= 1400; seed++) draws += playHand(seed, LedgerGames.PLAIN, shapes, timer).draws();
    StageTimer.Entry entry = timer.snapshot().get(StageTimer.Stage.LEDGER);
    Assert.assertNotNull(entry);
    double perCall = entry.seconds() / entry.count() * 1e9, perDraw = entry.seconds() / draws * 1e9;
    log.info(
        "LEDGER_COST calls={} draws={} ns/call={} ns/draw(all ledger work per draw)={}",
        entry.count(),
        draws,
        String.format(Locale.ROOT, "%.0f", perCall),
        String.format(Locale.ROOT, "%.0f", perDraw));
    Assert.assertTrue(perDraw < 500_000, "ns per draw " + perDraw);
  }
}
