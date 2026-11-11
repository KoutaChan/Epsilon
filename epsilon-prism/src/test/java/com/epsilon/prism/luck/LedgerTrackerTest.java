package com.epsilon.prism.luck;

import com.epsilon.agent.learn.StageTimer;
import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnKeys;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.ChanceKind;
import com.epsilon.mahjong.engine.ChanceLog;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RandomGames;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.core.CrnSeeds;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.SplittableRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 運の精算のトラッカー（{@link LedgerTracker}）と確率的丸め（{@link LedgerRounding}）の平均が 0 になることを確かめる。多くの局を打つと、
 * 席・列ごとに局で束ねた平均が 0 に近い。方策の項は b の上で平均 0 で、選ばれる頻度が b に合う。確率的丸めは偏らず、結果は fp16 で厳密に
 * 表せる。ほかに、s のクリップと足さない判断、有限でない増分の例外、点数表どおりの払い、分岐が親に触らないこと、裏ドラのプールも確かめる。
 */
public class LedgerTrackerTest {
  private static final Logger log = LoggerFactory.getLogger(LedgerTrackerTest.class);

  private static final LedgerColumn[] COLUMNS = LedgerColumn.values();

  /** 方策の列で s をクリップする幅（{@link LedgerGames#RECORDING} の値）。 */
  private static final double POLICY_CLIP = LedgerGames.RECORDING.policyClip();

  /** 帳簿の計算にかかる時間をまとめる段。収集の段と同じ測り方をする。 */
  private enum Stage {
    LEDGER
  }

  @DataProvider
  public Object[][] variants() {
    return new Object[][] {{Variant.TENHOU_4P}, {Variant.TENHOU_3P}};
  }

  /** テストで使う s。手の番号と席から作るハッシュで、値は [-1, 1)。方策の項は、どの s でも平均 0 になる。 */
  private static float score(int code, int seat) {
    return (float) (2 * CrnSeeds.unit(SeedMixer.mix64(code * 131L + seat)) - 1);
  }

  /** 種類ごとに、赤でない手出しの打牌の番号を返す。 */
  private static int[] discards(int... kinds) {
    int[] codes = new int[kinds.length];
    for (int i = 0; i < kinds.length; i++) codes[i] = Action.discard(kinds[i], false, false, false);
    return codes;
  }

  /** p に比例して手を 1 つ引く。重みを持つのは p &gt; 0 の手だけ。 */
  private static int sample(float[] p, int n, Random random) {
    double sum = 0;
    for (int a = 0; a < n; a++) sum += p[a] > 0 ? p[a] : 0;
    double u = random.nextDouble() * sum;
    int last = -1;
    for (int a = 0; a < n; a++) {
      if (!(p[a] > 0)) continue;
      last = a;
      u -= p[a];
      if (u < 0) return a;
    }
    return last;
  }

  /** 1 局を打って帳簿を付けた結果。 */
  private record Hand(List<LedgerGames.Seat> seats, int draws) {}

  /**
   * 半荘の最初の局を、席ごとのツモの列で打つ。最初の判断の時点の後に、席ごとにテスト用の受け取り先を作る。収集と同じ順に、判断の時点ごとに
   * {@code observe}、合法手が 2 つ以上ある判断ごとに {@code policy}、局末に {@code handEnd} を呼ぶ。手は、安い打ち方の one-hot に下限 0.1 を
   * 混ぜた b から引く。
   */
  private static Hand playHand(Variant variant, long seed, LedgerGames.Style style, StageTimer<Stage> timer) {
    var engine = new Engine(variant);
    engine.startGame(SeedMixer.mix64(seed * 2_147_483_647L), DrawOrder.SEAT_STREAMS);
    var ws = new RuleWorkspace(variant);
    var tracker = new LedgerTracker(LedgerGames.RECORDING);
    var random = new Random(seed);
    var chooser = LedgerGames.chooser(variant, style);
    var behaviour = new Behaviour(0.1, 0, CrnScheme.SINGLE_RACE);
    int[] choices = new int[variant.seats()];
    List<LedgerGames.Seat> seats = null;
    int firstDraws = 0;
    Boundary boundary = engine.advance(ws);
    while (boundary == Boundary.DECISIONS) {
      FullView full = engine.full();
      tracker.observe(full, seats == null ? List.of() : seats, ws);
      if (seats == null) {
        seats = LedgerGames.seats(variant);
        firstDraws = draws(full.chance());
      }
      Decisions decisions = engine.decisions();
      for (int slot = 0; slot < decisions.size(); slot++) {
        int n = decisions.legalCount(slot), seat = decisions.seat(slot);
        int chosen = chooser.choose(engine, slot, random);
        if (n > 1) {
          float[] sigma = new float[n];
          sigma[chosen] = 1f;
          float[] p = behaviour.policy(sigma, n);
          float[] scores = new float[n];
          for (int a = 0; a < n; a++) scores[a] = score(decisions.legal(slot, a), seat);
          chosen = sample(p, n, random);
          tracker.policy(seat, scores, p, chosen, n, seats);
        }
        choices[slot] = chosen;
      }
      boundary = engine.commit(choices, ws);
    }
    tracker.handEnd(engine.full(), engine.roundEnd(), seats);
    if (timer != null) tracker.flush(timer, Stage.LEDGER);
    return new Hand(seats, draws(engine.full().chance()) - firstDraws);
  }

  private static int draws(ChanceLog log) {
    int draws = 0;
    for (int i = 0; i < log.size(); i++) if (LedgerGames.draw(log.kind(i))) draws++;
    return draws;
  }

  /**
   * 20000 局（それぞれ半荘の最初の局）を打ち、席ごとの受け取り先の帳簿を局で束ねて z を取る。全ての席・列で |z| &lt; 4.5 を求める
   * （0 でない局が 30 未満の席・列は飛ばす）。局で全ての席を足すと、裏ドラの項は重みの和が 0 なので和が恒等的に 0 になる。そのため z は
   * 席ごとに取る。主な列（受け入れ・ツモ・表示・裏ドラ・方策）は、全ての席で十分な回数起きる。
   */
  @Test(dataProvider = "variants", timeOut = 900_000)
  public void everyTermIsCentredOverManyHands(Variant variant) {
    int columns = LedgerColumn.COUNT, seats = variant.seats(), hands = 20_000;
    double[][] sum = new double[seats][columns], square = new double[seats][columns];
    int[][] nonzero = new int[seats][columns];
    for (long seed = 1; seed <= hands; seed++) {
      Hand hand = playHand(variant, seed, seed % 3 == 0 ? LedgerGames.KANS : LedgerGames.PLAIN, null);
      for (LedgerGames.Seat target : hand.seats()) {
        int seat = target.seat();
        for (int f = 0; f < columns; f++) {
          double value = target.ledger()[f];
          Assert.assertTrue(Double.isFinite(value), "seat " + seat + " " + COLUMNS[f].reportName());
          sum[seat][f] += value;
          square[seat][f] += value * value;
          if (value != 0) nonzero[seat][f]++;
        }
      }
    }
    int tested = 0;
    for (int seat = 0; seat < seats; seat++)
      for (int f = 0; f < columns; f++) {
        if (nonzero[seat][f] < 30) continue;
        tested++;
        double mean = sum[seat][f] / hands, variance = square[seat][f] / hands - mean * mean;
        double z = mean / Math.sqrt(variance / hands);
        Assert.assertTrue(
            Math.abs(z) < 4.5,
            variant + " seat " + seat + " " + COLUMNS[f].reportName() + " mean=" + mean + " z=" + z + " nonzero="
                + nonzero[seat][f]);
      }
    for (int seat = 0; seat < seats; seat++)
      for (LedgerColumn column :
          new LedgerColumn[] {
            LedgerColumn.OWN_IMPROVE_1,
            LedgerColumn.OWN_TSUMO,
            LedgerColumn.OWN_DRAWN_DANGER,
            LedgerColumn.OPP_TSUMO,
            LedgerColumn.DORA_SELF,
            LedgerColumn.URA,
            LedgerColumn.POLICY_SELF,
            LedgerColumn.POLICY_OPPONENT
          })
        Assert.assertTrue(
            nonzero[seat][column.ordinal()] >= 30,
            variant + " seat " + seat + " " + column.reportName() + " happened in " + nonzero[seat][column.ordinal()]
                + " hands");
    Assert.assertTrue(tested >= 12 * seats, variant + " tested seat columns " + tested);
  }

  /**
   * 方策の項を確かめる。乱数の p（下限のあるもの、0・NaN・負を含むもの）と s で、10^5 個のキーで {@link CrnKeys#select} が選んだ手の δ を
   * 平均し、z &lt; 4.5 を求める。選ばれた頻度は b に合う。
   */
  @Test
  public void policyTermIsCentred() {
    int[] legal = discards(0, 9, 18, 27, 31);
    int n = legal.length, keys = 100_000;
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
        Assert.assertTrue(LedgerTracker.policyCounts(scores, p, n));
        double delta = LedgerTracker.policyDelta(scores, p, chosen, n, POLICY_CLIP);
        Assert.assertTrue(Math.abs(delta) <= 2 * POLICY_CLIP);
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

  /**
   * 確率的丸めを確かめる。決まった x（正負・正規数・非正規数・2 の冪の際・fp16 で厳密に表せる値）を 10^5 個の乱数で丸め、平均の
   * z &lt; 4.5 を求める。丸めた結果は fp16 で厳密に表せる。
   */
  @Test
  public void stochasticRoundingIsUnbiased() {
    double[] xs = {0.3, -0.3, 1e-6, -2.5e-5, 6.1e-5, Math.scalb(1.0, -14), 1 + Math.scalb(1.0, -11), 3.0, 1024.3, -777.77, 65000.7, -65000.7};
    int coins = 100_000;
    for (int i = 0; i < xs.length; i++) {
      double x = xs[i], sum = 0, square = 0;
      for (int k = 0; k < coins; k++) {
        float rounded = LedgerRounding.round(x, LedgerRounding.coin(LedgerRounding.key(11L, i, k), k % 7, k % LedgerColumn.COUNT));
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
    Assert.assertEquals(LedgerRounding.round(0, 0.5), 0f);
    Assert.assertTrue(Float.isNaN(LedgerRounding.round(Double.NaN, 0.5)));
    Assert.assertEquals(LedgerRounding.round(3.0, 0.999), 3f);
  }

  /**
   * 方策の項のクリップと飛ばしを確かめる。|s| がクリップの幅（8）を越える手は、クリップした s̄ で計算する。b &gt; 0 の手の s が NaN・±∞
   * なら足さない（b = 0 の手なら足す）。p の NaN・0・負は重み 0 にする（{@link CrnKeys#select} と同じ判定）。s のない判断は足さない。
   * エンジンの判断では、合法手の番号は重ならない。
   */
  @Test
  public void policyTermClipsAndSkips() {
    float[] scores = {20f, 0f, -30f}, p = {0.5f, 0.3f, 0.2f};
    Assert.assertTrue(LedgerTracker.policyCounts(scores, p, 3));
    Assert.assertEquals(LedgerTracker.policyDelta(scores, p, 0, 3, POLICY_CLIP), 8 - (0.5 * 8 + 0.2 * -8), 1e-6);
    Assert.assertEquals(LedgerTracker.policyDelta(scores, p, 2, 3, POLICY_CLIP), -8 - (0.5 * 8 + 0.2 * -8), 1e-6);
    Assert.assertFalse(LedgerTracker.policyCounts(new float[] {Float.NaN, 0f, 1f}, new float[] {0.5f, 0.5f, 0f}, 3));
    Assert.assertFalse(LedgerTracker.policyCounts(new float[] {Float.POSITIVE_INFINITY, 0f}, new float[] {0.5f, 0.5f}, 2));
    Assert.assertTrue(LedgerTracker.policyCounts(new float[] {Float.NaN, 0f, 1f}, new float[] {0f, 0.5f, 0.5f}, 3));
    Assert.assertEquals(
        LedgerTracker.policyDelta(new float[] {Float.NaN, 0f, 1f}, new float[] {0f, 0.5f, 0.5f}, 2, 3, POLICY_CLIP), 0.5, 1e-6);
    float[] odd = {Float.NaN, -1f, 0f, 0.7f, 0.3f};
    Assert.assertEquals(LedgerTracker.policyDelta(new float[] {9f, 9f, 9f, 1f, 0f}, odd, 3, 5, POLICY_CLIP), 1 - 0.7, 1e-6);
    int[] legal = discards(1, 2, 3, 4, 5);
    for (int k = 0; k < 10_000; k++) Assert.assertTrue(CrnKeys.select(SeedMixer.mix64(k), legal, odd, 0) >= 3);
    // トラッカー: s のない判断と、b > 0 の手の s が有限でない判断は帳簿を変えない
    var tracker = new LedgerTracker(LedgerGames.RECORDING);
    List<LedgerGames.Seat> seats = LedgerGames.seats(Variant.TENHOU_4P);
    tracker.policy(0, null, p, 0, 3, seats);
    tracker.policy(0, new float[] {Float.NaN, 0f, 1f}, p, 1, 3, seats);
    for (LedgerGames.Seat target : seats) for (double value : target.ledger()) Assert.assertEquals(value, 0.0);
    tracker.policy(1, scores, p, 0, 3, seats);
    for (LedgerGames.Seat target : seats) {
      int seat = target.seat();
      Assert.assertEquals(target.ledger()[LedgerColumn.POLICY_SELF.ordinal()], seat == 1 ? 5.6 : 0, 1e-6);
      Assert.assertEquals(target.ledger()[LedgerColumn.POLICY_OPPONENT.ordinal()], seat == 1 ? 0 : 5.6, 1e-6);
    }
    // 判断の合法手の番号は重ならない（b の正規化は、手の番号が重ならないことを前提にする）
    for (Variant variant : Variant.values()) {
      RandomGames games = RandomGames.start(variant, 77L + variant.ordinal(), LedgerGames.chooser(variant, LedgerGames.CALLS));
      for (int step = 0; step < 20_000; step++) {
        if (games.boundary() == Boundary.DECISIONS) {
          Decisions decisions = games.engine.decisions();
          for (int slot = 0; slot < decisions.size(); slot++) {
            Set<Integer> codes = new HashSet<>();
            for (int a = 0; a < decisions.legalCount(slot); a++)
              Assert.assertTrue(codes.add(decisions.legal(slot, a)), variant + " slot " + slot);
          }
        }
        games.step();
      }
    }
  }

  /**
   * 有限でない増分は、帳簿を付けずに例外にする。c が壊れている場合も、重みの和が ∞ の分布で方策の項が NaN になる場合も同じ。fp16 の範囲の
   * 外も例外にする。
   */
  @Test
  public void nonFiniteDeltaFails() {
    Assert.assertThrows(IllegalStateException.class, () -> LedgerTracker.add(new double[LedgerColumn.COUNT], 3, Double.NaN));
    Assert.assertThrows(
        IllegalStateException.class,
        () -> LedgerTracker.add(new double[LedgerColumn.COUNT], 3, Double.NEGATIVE_INFINITY));
    var tracker = new LedgerTracker(LedgerGames.RECORDING);
    List<LedgerGames.Seat> seats = LedgerGames.seats(Variant.TENHOU_4P);
    Assert.assertThrows(
        IllegalStateException.class,
        () -> tracker.policy(0, new float[] {0f, 1f}, new float[] {Float.POSITIVE_INFINITY, 0.5f}, 0, 2, seats));
    Assert.assertThrows(IllegalStateException.class, () -> LedgerRounding.round(70_000, 0.5));
    Assert.assertThrows(IllegalStateException.class, () -> LedgerRounding.round(Double.POSITIVE_INFINITY, 0.5));
  }

  /**
   * 点数の換算を確かめる。四人打ちの決まった値（子の満貫ツモは 2000/4000、親は 4000 オール、栄和は子 8000・親 12000、本場と供託を足す）と
   * 一致し、全てのルールで {@link Settlement} の払いと同じ値になる。裏ドラの重みは払いの割合の負で、四人打ちでは −1/2・−1/4・−1/3 と
   * 厳密に同じ。
   */
  @Test
  public void paymentsFollowTheScoreTable() {
    Variant four = Variant.TENHOU_4P;
    Assert.assertEquals(LedgerTracker.tsumoGain(four, 2000, 1, 0, 0, 0), 8000);
    Assert.assertEquals(LedgerTracker.tsumoGain(four, 2000, 0, 0, 1, 2), 12000 + 300 + 2000);
    Assert.assertEquals(LedgerTracker.tsumoPayment(2000, 1, 0, 0, 1), 4000 + 100);
    Assert.assertEquals(LedgerTracker.tsumoPayment(2000, 1, 2, 0, 1), 2000 + 100);
    Assert.assertEquals(LedgerTracker.tsumoPayment(2000, 0, 2, 0, 0), 4000);
    Assert.assertEquals(LedgerTracker.ronPayment(four, 2000, 1, 0, 0), 8000);
    Assert.assertEquals(LedgerTracker.ronPayment(four, 2000, 0, 0, 2), 12000 + 600);
    double clip = LedgerGames.RECORDING.pointClip();
    Assert.assertEquals(LedgerTracker.thousands(100_000, clip), clip);
    Assert.assertEquals(LedgerTracker.thousands(-3900, clip), -3.9, 1e-12);
    var random = new Random(3);
    int[] bases = {240, 320, 480, 640, 960, 1280, 1920, 2000, 3000, 4000, 6000, 8000};
    for (Variant variant : Variant.values()) {
      int seats = variant.seats();
      RoundOutcome outcome = new RoundOutcome(variant);
      for (int trial = 0; trial < 400; trial++) {
        int base = bases[random.nextInt(bases.length)];
        int winner = random.nextInt(seats), dealer = random.nextInt(seats), honba = random.nextInt(4), sticks = random.nextInt(3);
        int[] delta = new int[seats];
        Settlement.tsumo(variant, base, winner, dealer, honba, RoundOutcome.NO_SEAT, delta);
        Assert.assertEquals(LedgerTracker.tsumoGain(variant, base, winner, dealer, honba, sticks), delta[winner] + Variant.DEPOSIT * sticks);
        for (int payer = 0; payer < seats; payer++)
          if (payer != winner) Assert.assertEquals(LedgerTracker.tsumoPayment(base, winner, payer, dealer, honba), -delta[payer]);
        int loser = (winner + 1 + random.nextInt(seats - 1)) % seats;
        delta = new int[seats];
        Settlement.ron(variant, base, winner, loser, dealer, honba, RoundOutcome.NO_SEAT, delta);
        Assert.assertEquals(LedgerTracker.ronPayment(variant, base, winner, dealer, honba), -delta[loser]);
        Assert.assertEquals(LedgerTracker.ronPayment(variant, base, winner, dealer, honba), delta[winner]);
        // 裏ドラの重み: 和了者は +1、栄和なら放銃者は −1、ツモなら払いの割合の負（合わせて −1）
        outcome.setRon(base, winner, loser, RoundOutcome.NO_SEAT, 1 << winner);
        for (int seat = 0; seat < seats; seat++)
          Assert.assertEquals(
              LedgerTracker.uraWeight(variant, outcome, winner, seat, dealer), seat == winner ? 1.0 : seat == loser ? -1.0 : 0.0);
        outcome.setTsumo(base, winner, RoundOutcome.NO_SEAT, 1 << winner);
        Assert.assertEquals(outcome.kind(), EndKind.TSUMO);
        double total = 0;
        for (int seat = 0; seat < seats; seat++) {
          double weight = LedgerTracker.uraWeight(variant, outcome, winner, seat, dealer);
          if (seat == winner) Assert.assertEquals(weight, 1.0);
          else {
            // 割合は、切り上げが効かない基本点 100 の払いで決まり、和了の基本点に依らない
            double share = ScoreMath.tsumo(100, winner == dealer, seat == dealer);
            double all = LedgerTracker.tsumoGain(variant, 100, winner, dealer, 0, 0);
            Assert.assertEquals(weight, -share / all, 1e-12, variant + " base " + base);
            total += weight;
          }
        }
        Assert.assertEquals(total, -1.0, 1e-12);
        if (seats == 4)
          for (int seat = 0; seat < seats; seat++) {
            if (seat == winner) continue;
            double expected = winner == dealer ? -1.0 / 3 : seat == dealer ? -0.5 : -0.25;
            Assert.assertEquals(
                Double.doubleToRawLongBits(LedgerTracker.uraWeight(variant, outcome, winner, seat, dealer)),
                Double.doubleToRawLongBits(expected));
          }
      }
    }
  }

  /**
   * 帳簿にかかる CPU の時間を、収集の帳簿の段と同じ測り方で測る。全ての席に受け取り先を持たせて局を打ち、ツモ 1 回あたりの帳簿の時間を
   * ログの "LEDGER_COST" の行に出す。上限は緩い目安（1 回 0.5 ms）。
   */
  @Test(dataProvider = "variants", timeOut = 300_000)
  public void ledgerCostPerDraw(Variant variant) {
    for (long seed = 1; seed <= 50; seed++) playHand(variant, seed, LedgerGames.PLAIN, null);
    var timer = new StageTimer<>(Stage.class);
    long draws = 0;
    for (long seed = 1001; seed <= 1400; seed++) draws += playHand(variant, seed, LedgerGames.PLAIN, timer).draws();
    StageTimer.Entry entry = timer.snapshot().get(Stage.LEDGER);
    Assert.assertNotNull(entry);
    double perCall = entry.seconds() / entry.count() * 1e9, perDraw = entry.seconds() / draws * 1e9;
    log.info(
        "LEDGER_COST variant={} calls={} draws={} ns/call={} ns/draw(all ledger work per draw)={}",
        variant,
        entry.count(),
        draws,
        String.format(Locale.ROOT, "%.0f", perCall),
        String.format(Locale.ROOT, "%.0f", perDraw));
    Assert.assertTrue(perDraw < 500_000, "ns per draw " + perDraw);
  }

  /**
   * 1 局を打ち、主経路の帳簿を返す。{@code fork} 番目の判断の時点（−1 なら分岐を作らない）で、(a) 親のトラッカーを写した分岐を別の手で
   * 局末まで進め、(b) {@code follower} が null でなければ、同じ手で並べて進める写しの帳簿を {@code follower} に付ける。{@code at} が null
   * でなければ、分岐の時点の主経路の帳簿をそこに写す。
   */
  private static List<LedgerGames.Seat> mainLedger(
      Variant variant, long seed, int fork, List<LedgerGames.Seat> follower, double[][] at) {
    var engine = new Engine(variant);
    engine.startGame(SeedMixer.mix64(seed), DrawOrder.SEAT_STREAMS);
    var ws = new RuleWorkspace(variant);
    var copyWs = new RuleWorkspace(variant);
    var random = new Random(seed);
    var chooser = LedgerGames.chooser(variant, LedgerGames.KANS);
    var tracker = new LedgerTracker(LedgerGames.RECORDING);
    List<LedgerGames.Seat> seats = LedgerGames.seats(variant);
    int[] choices = new int[variant.seats()];
    Engine copy = null;
    LedgerTracker copyTracker = null;
    int decision = 0;
    Boundary boundary = engine.advance(ws);
    while (boundary == Boundary.DECISIONS) {
      tracker.observe(engine.full(), seats, ws);
      if (copy != null) copyTracker.observe(copy.full(), follower, copyWs);
      Decisions decisions = engine.decisions();
      for (int slot = 0; slot < decisions.size(); slot++) choices[slot] = chooser.choose(engine, slot, random);
      if (decision++ == fork) {
        branch(variant, engine.copy(), new LedgerTracker(tracker), seed);
        for (int seat = 0; seat < seats.size() && at != null; seat++) at[seat] = seats.get(seat).ledger().clone();
        if (follower != null) {
          copy = engine.copy();
          copyTracker = new LedgerTracker(tracker);
        }
      }
      boundary = engine.commit(choices, ws);
      if (copy != null) Assert.assertEquals(copy.commit(choices, copyWs), boundary);
    }
    tracker.handEnd(engine.full(), engine.roundEnd(), seats);
    if (copy != null) copyTracker.handEnd(copy.full(), copy.roundEnd(), follower);
    return seats;
  }

  /** 分岐を別の乱数の手で局末まで進め、分岐自身の受け取り先に帳簿を付ける。 */
  private static void branch(Variant variant, Engine branch, LedgerTracker tracker, long seed) {
    var ws = new RuleWorkspace(variant);
    var random = new Random(seed + 1);
    var chooser = LedgerGames.chooser(variant, LedgerGames.CALLS);
    List<LedgerGames.Seat> seats = LedgerGames.seats(variant);
    int[] choices = new int[variant.seats()];
    while (true) {
      Decisions decisions = branch.decisions();
      for (int slot = 0; slot < decisions.size(); slot++) choices[slot] = chooser.choose(branch, slot, random);
      if (branch.commit(choices, ws) != Boundary.DECISIONS) break;
      tracker.observe(branch.full(), seats, ws);
    }
    tracker.handEnd(branch.full(), branch.roundEnd(), seats);
  }

  /**
   * 分岐のトラッカーが親に触らないことを確かめる。主経路のトラッカーを写した分岐を局末まで進めてから主経路を進めた帳簿は、分岐を作らずに
   * 進めた帳簿とビットまで一致する（席の表は作った後に変えない）。同じ手で並べて進めた写しの帳簿は、写した時点からの主経路の増分と同じに
   * なる（写しは親が読んだ位置から読む）。
   */
  @Test(dataProvider = "variants", timeOut = 300_000)
  public void forkTrackerDoesNotTouchItsParent(Variant variant) {
    int moved = 0;
    for (long seed = 1; seed <= 40; seed++) {
      int fork = 8 + (int) (seed % 5);
      List<LedgerGames.Seat> alone = mainLedger(variant, seed, -1, null, null);
      List<LedgerGames.Seat> follower = LedgerGames.seats(variant);
      double[][] at = new double[variant.seats()][];
      List<LedgerGames.Seat> forked = mainLedger(variant, seed, fork, follower, at);
      for (int seat = 0; seat < alone.size(); seat++)
        for (int f = 0; f < LedgerColumn.COUNT; f++) {
          String where = variant + " seed " + seed + " seat " + seat + " " + COLUMNS[f].reportName();
          double parent = forked.get(seat).ledger()[f];
          Assert.assertEquals(Double.doubleToRawLongBits(parent), Double.doubleToRawLongBits(alone.get(seat).ledger()[f]), where);
          if (at[seat] == null) continue;
          double copied = follower.get(seat).ledger()[f];
          Assert.assertEquals(copied, parent - at[seat][f], 1e-9, where + " (copy)");
          if (copied != 0) moved++;
        }
    }
    Assert.assertTrue(moved > 100, "the copies booked " + moved + " nonzero terms");
  }

  /**
   * 裏ドラの期待値に使うプールを確かめる。判断の時点では、{@link LedgerTables#pool} は種類ごとに「枚数 − 見えている枚数 − 全ての席の
   * 純手牌」と同じになる。局末では、エンジンがその前に裏ドラを誰にも見えない牌からめくっているので、プールに裏ドラの表示牌の枚数を
   * 足した値が同じになる。
   */
  @Test(dataProvider = "variants", timeOut = 300_000)
  public void uraPoolIsUnseenPlusUra(Variant variant) {
    int[] pool = new int[Tiles.KINDS];
    int[] withUra = {0};
    HandShape hand = new HandShape();
    TileSet set = variant.tiles();
    LedgerTables tables = LedgerTables.of(variant);
    LedgerGames.walk(
        variant,
        11L,
        LedgerGames.PLAIN,
        3000,
        new LedgerGames.Visitor() {
          @Override
          public void decision(Engine engine, RuleWorkspace ws) {
            check(engine.full(), 0);
          }

          @Override
          public void roundEnd(Engine engine) {
            ChanceLog log = engine.full().chance();
            int uras = 0;
            for (int i = 0; i < log.size(); i++) if (log.kind(i) == ChanceKind.URA) uras++;
            if (uras > 0) withUra[0]++;
            check(engine.full(), uras);
          }

          private void check(FullView full, int uras) {
            int total = tables.pool(full, pool);
            ChanceLog log = full.chance();
            for (int i = log.size() - uras; i < log.size(); i++) {
              Assert.assertEquals(log.kind(i), ChanceKind.URA);
              pool[Tiles.kind(log.tile(i))]++;
            }
            SeatView table = full.seat(0);
            int left = 0;
            for (long bits = set.kinds(); bits != 0; bits &= bits - 1) {
              int kind = Long.numberOfTrailingZeros(bits);
              int expected = set.copies(kind) - table.visible(kind);
              for (int seat = 0; seat < variant.seats(); seat++) {
                full.copyHand(seat, hand);
                expected -= hand.count(kind);
              }
              Assert.assertEquals(pool[kind], expected, variant + " kind " + kind + " uras " + uras);
              left += expected;
            }
            Assert.assertEquals(total + uras, left);
          }
        });
    Assert.assertTrue(withUra[0] >= 30, variant + " round ends with ura " + withUra[0]);
  }
}
