package com.epsilon.classic.model;

import static org.testng.Assert.assertEquals;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.ClassicFields.RoundInt;
import com.epsilon.classic.input.ClassicFields.WinFact;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicSchema.FieldLayout;
import com.epsilon.classic.input.Tensor;
import com.epsilon.classic.model.PointProjection.Projection;
import com.epsilon.mahjong.rules.DealerOutcome;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Variant;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * Major の点数の計算が出す終局・連荘・次局・西入・終局の効用が、エンジンの終局の判定（{@link Progression#finishes}）と合うことを確かめる。
 * 和了は 1 翻 30 符で、待ちの 8 通りのうち下家からのロンを読む。
 */
public class PointProjectionTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int SEATS = ClassicSchema.SEATS;
  private static final int FU_30 = 3;

  /** 26 個の値の中の位置。 */
  private static final int END = 19;

  private static final int DEALER_REPEAT = 20;
  private static final int NEXT_ROUND = 21;
  private static final int WEST_EXTENSION = 22;
  private static final int UTILITY = 23;

  private NDManager manager;

  @BeforeClass(groups = "native")
  public void open() {
    manager = NDManager.newBaseManager(Device.cpu(), "PyTorch");
  }

  @AfterClass(groups = "native", alwaysRun = true)
  public void close() {
    if (manager != null) {
      manager.close();
    }
  }

  @Test(groups = "native")
  public void aWinIsJudgedOnTheScoresAfterTheWinnerTakesTheDeposits() {
    int south4 = VARIANT.lastRound();
    int dealer = VARIANT.dealer(south4);
    int child = (dealer + 1) % SEATS;
    assertMatchesEngine(child, south4, 2, false, around(child, 28_600, 26_000, 25_400, 18_000));
    assertMatchesEngine(child, south4, 0, true, around(child, 29_000, 26_000, 25_000, 20_000));
    assertMatchesEngine(dealer, south4, 2, false, around(dealer, 27_600, 26_000, 25_400, 19_000));
    assertMatchesEngine(dealer, south4, 2, false, around(dealer, 29_500, 20_000, 31_500, 17_000));
  }

  @Test(groups = "native")
  public void theDealerRepeatsTheFinalRoundUntilTheGameCanEnd() {
    int west4 = VARIANT.finalRound();
    int dealer = VARIANT.dealer(west4);
    assertMatchesEngine(dealer, west4, 0, false, around(dealer, 20_000, 27_000, 27_000, 26_000));
    assertMatchesEngine(dealer, west4, 0, false, around(dealer, 28_000, 22_000, 29_000, 21_000));
    assertMatchesEngine(dealer, west4, 0, false, around(dealer, 29_000, 27_000, 23_000, 21_000));
    int child = (dealer + 1) % SEATS;
    assertMatchesEngine(child, west4, 0, false, around(child, 20_000, 27_000, 27_000, 26_000));
  }

  /**
   * 下家からロンした結果を、エンジンの判定と比べる。
   *
   * @param seat 和了する席
   * @param round 局の番号
   * @param kyotaku 場の供託の本数
   * @param riichi 和了の前に立直の供託を出すなら true
   * @param scores 絶対席の順に並べた、和了の前の点
   */
  private void assertMatchesEngine(int seat, int round, int kyotaku, boolean riichi, int[] scores) {
    boolean dealerWins = VARIANT.dealer(round) == seat;
    int[] after = scores.clone();
    int payment = ScoreMath.ron(ScoreMath.base(1, 30, 0), dealerWins);
    after[seat] += payment + Variant.DEPOSIT * kyotaku;
    after[(seat + 1) % SEATS] -= payment;
    boolean finishes =
        Progression.finishes(VARIANT, round, after, dealerWins ? DealerOutcome.CONTINUES : DealerOutcome.PASSES);
    int rank = 0;
    for (int other = 0; other < SEATS; other++) {
      if (after[other] > after[seat] || (after[other] == after[seat] && other < seat)) {
        rank++;
      }
    }
    boolean southOrLater = round >= VARIANT.lastRound();
    float[] expected = new float[PointProjection.FEATURES];
    expected[END] = finishes ? 1 : 0;
    expected[DEALER_REPEAT] = !finishes && dealerWins ? 1 : 0;
    expected[NEXT_ROUND] = !finishes && !dealerWins && !southOrLater ? 1 : 0;
    expected[WEST_EXTENSION] = !finishes && !dealerWins && southOrLater ? 1 : 0;
    expected[UTILITY] = finishes ? (float) RankUtility.AVERAGE_RANK.value(VARIANT, rank) : 0;

    String scene = "seat " + seat + " round " + round + " kyotaku " + kyotaku + " riichi " + riichi;
    try (NDManager working = manager.newSubManager()) {
      Projection projection = ronFromShimocha(working, seat, round, kyotaku, riichi, scores);
      float[] features = projection.features().get("0, 0").toFloatArray();
      float[] gates = projection.gates().get("0, 0").toFloatArray();
      for (int index : new int[] {END, DEALER_REPEAT, NEXT_ROUND, WEST_EXTENSION, UTILITY}) {
        assertEquals(features[index], expected[index], 1e-6f, scene + " feature " + index);
      }
      assertEquals(gates[0], expected[END], 1e-6f, scene + " gate end");
      assertEquals(gates[1], expected[UTILITY], 1e-6f, scene + " gate utility");
    }
  }

  /** 待ちの 8 通りの結果を求める。先頭が下家からのロン。 */
  private static Projection ronFromShimocha(
      NDManager working, int seat, int round, int kyotaku, boolean riichi, int[] scores) {
    ClassicSchema schema = ClassicSchema.of(Architecture.MAJOR);
    FieldLayout<RoundInt> roundLayout = schema.layout(RoundInt.class);
    int[] state = new int[schema.width(Tensor.STATE_CATEGORIES)];
    state[roundLayout.offset(RoundInt.PLAYER_SEAT)] = seat + 1;
    state[roundLayout.offset(RoundInt.DEALER_RELATIVE_SEAT)] = (VARIANT.dealer(round) - seat + SEATS) % SEATS + 1;
    state[roundLayout.offset(RoundInt.KYOKU_INDEX)] = round + 1;
    state[roundLayout.offset(RoundInt.KYOTAKU)] = kyotaku;
    FieldLayout<WinFact> factLayout = schema.layout(WinFact.class);
    int[] facts = new int[2 * factLayout.stride()];
    for (int win = 0; win < 2; win++) {
      int base = win * factLayout.stride();
      facts[base + factLayout.offset(WinFact.VALID)] = 1;
      facts[base + factLayout.offset(WinFact.HAN_WITHOUT_URA)] = 1;
      facts[base + factLayout.offset(WinFact.FU_CODE)] = FU_30;
    }
    int[] ledger = new int[SEATS];
    for (int player = 0; player < SEATS; player++) {
      ledger[player] = scores[player] / 100;
    }
    PointProjection projection = new PointProjection(VARIANT, RankUtility.AVERAGE_RANK);
    projection.initialize(working, DataType.FLOAT32, new Shape(-1, PointProjection.FEATURES));
    Pass pass = new Pass(new ParameterStore(working, false), working, Precision.FLOAT32, false);
    NDArray aka = working.zeros(new Shape(1), DataType.INT32);
    return projection.waits(
        pass,
        working.create(ledger, new Shape(1, SEATS)),
        working.create(state, new Shape(1, state.length)),
        working.create(facts, new Shape(1, 2, factLayout.stride())),
        aka,
        working.create(new int[] {riichi ? 1 : 0}));
  }

  /** 和了する席から相対席の順に並べた点を、絶対席の順に並べ直す。 */
  private static int[] around(int seat, int... relative) {
    int[] scores = new int[SEATS];
    for (int offset = 0; offset < SEATS; offset++) {
      scores[(seat + offset) % SEATS] = relative[offset];
    }
    return scores;
  }
}
