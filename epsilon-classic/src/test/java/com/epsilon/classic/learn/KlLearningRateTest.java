package com.epsilon.classic.learn;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;

import com.epsilon.classic.learn.KlLearningRate.Action;
import com.epsilon.classic.learn.KlLearningRate.Adjustment;
import com.epsilon.classic.learn.KlLearningRate.Schedule;
import com.epsilon.classic.learn.KlLearningRate.Settings;
import com.epsilon.classic.learn.KlLearningRate.State;
import com.google.gson.Gson;
import java.util.List;
import org.testng.annotations.Test;

/**
 * KL で方策の学習率を決める制御を確かめる。ウォームアップの中央値で目標を決めること、直近の中央値と不感帯、√(目標/中央値) の倍率と
 * 1 回の上限、学習率の範囲、目標の予定（保つ・cos で下げる・保つ）を見る。受け入れなかった macro は状態を変えず、書き出した状態から
 * 続けても動きは同じになる。
 */
public class KlLearningRateTest {

  private static final double EPSILON = 1e-15;

  /** 初め 1e-4、範囲 1e-6〜1e-3、ウォームアップ 3 回、中央値 3 回、目標 1.25 倍、不感帯 1.25、1 回 1.02 倍、目標を下げない。 */
  private static Settings flat() {
    return new Settings(1e-4, 1e-6, 1e-3, 3, 3, 1.25, 1.25, 1.02, 0, 0.5);
  }

  @Test
  public void warmupSetsTheTargetFromTheMedian() {
    KlLearningRate rate = new KlLearningRate(flat());
    Adjustment first = rate.accept(0.02, 4);
    assertEquals(first.action(), Action.WARMUP);
    assertEquals(first.applied(), 1e-4);
    assertEquals(first.next(), 1e-4);
    assertTrue(Double.isNaN(first.target()));
    assertNull(rate.state().schedule());
    assertEquals(rate.accept(0.01, 4).action(), Action.WARMUP);
    Adjustment last = rate.accept(0.03, 4);
    assertEquals(last.action(), Action.WARMUP);
    assertEquals(last.next(), 1e-4);
    assertEquals(last.median(), 0.02, EPSILON);
    State state = rate.state();
    assertEquals(state.macros(), 3);
    assertEquals(state.steps(), 12);
    assertEquals(state.warmup(), List.of(0.02, 0.01, 0.03));
    assertEquals(state.baseline(), 0.02, EPSILON);
    assertNotNull(state.schedule());
    assertEquals(state.schedule().anchorStep(), 12);
    assertEquals(state.schedule().anchorTarget(), 0.025, EPSILON);
    assertEquals(last.target(), 0.025, EPSILON);
    assertEquals(rate.rate(), 1e-4);
  }

  /** 目標 0.025、不感帯 [0.02, 0.03125]。中央値は直近 3 回で取るので、1 回だけ外れても動かない。 */
  @Test
  public void theMedianOfTheRecentMacrosDrivesTheRate() {
    KlLearningRate rate = warmed(flat());
    assertEquals(rate.accept(0.025, 1).action(), Action.HOLD);
    Adjustment outlier = rate.accept(0.001, 1);
    assertEquals(outlier.action(), Action.HOLD);
    assertEquals(outlier.median(), 0.025, EPSILON);
    assertEquals(outlier.next(), 1e-4);
    Adjustment low = rate.accept(0.001, 1);
    assertEquals(low.median(), 0.001, EPSILON);
    assertEquals(low.action(), Action.INCREASE);
    assertEquals(low.next(), 1e-4 * 1.02, 1e-18);
    for (int macro = 0; macro < 3; macro++) {
      rate.accept(0.5, 1);
    }
    Adjustment high = rate.accept(0.5, 1);
    assertEquals(high.action(), Action.DECREASE);
    assertEquals(high.next(), high.applied() / 1.02, 1e-18);
    assertEquals(rate.state().recent(), List.of(0.5, 0.5, 0.5));
  }

  /** 1 回の上限が大きければ、学習率に √(目標/中央値) をそのまま掛ける。中央値 0 は割る前に小さな正の数にする。 */
  @Test
  public void theStepIsTheSquareRootOfTheTargetOverTheMedian() {
    Settings wide = new Settings(1e-4, 1e-8, 1, 1, 1, 1, 1.25, 100, 0, 0.5);
    KlLearningRate rate = new KlLearningRate(wide);
    rate.accept(0.04, 1);
    Adjustment quarter = rate.accept(0.01, 1);
    assertEquals(quarter.action(), Action.INCREASE);
    assertEquals(quarter.next(), 2e-4, 1e-16);
    Adjustment fourfold = rate.accept(0.16, 1);
    assertEquals(fourfold.action(), Action.DECREASE);
    assertEquals(fourfold.next(), 1e-4, 1e-16);
    Adjustment zero = rate.accept(0, 1);
    assertEquals(zero.action(), Action.INCREASE);
    assertEquals(zero.next(), 1e-2, 1e-16, "the factor stops at 100");
  }

  @Test
  public void theRateStaysInsideItsBounds() {
    Settings narrow = new Settings(1e-4, 9.9e-5, 1.01e-4, 1, 1, 1, 1.25, 1.02, 0, 0.5);
    KlLearningRate rate = new KlLearningRate(narrow);
    rate.accept(0.04, 1);
    Adjustment up = rate.accept(0.001, 1);
    assertEquals(up.action(), Action.AT_MAXIMUM);
    assertEquals(up.next(), 1.01e-4);
    Adjustment hold = rate.accept(0.001, 1);
    assertEquals(hold.action(), Action.AT_MAXIMUM);
    assertEquals(rate.rate(), 1.01e-4);
    rate.accept(1, 1);
    Adjustment down = rate.accept(1, 1);
    assertEquals(down.action(), Action.AT_MINIMUM);
    assertEquals(down.next(), 9.9e-5);
  }

  /**
   * 予定 1,000 ステップ・0.5 倍: 100 ステップまでは保ち、1,000 ステップまで cos で半分に下げ、その後は保つ。ウォームアップが 1 割より
   * 長ければ、ウォームアップの終わりから下げる。
   */
  @Test
  public void theTargetHoldsThenDecaysOnACosine() {
    Settings planned = new Settings(1e-4, 1e-6, 1e-3, 3, 3, 1.25, 1.25, 1.02, 1_000, 0.5);
    Schedule early = Schedule.of(12, 0.02, planned);
    assertEquals(early.decayStep(), 100);
    assertEquals(early.endStep(), 1_000);
    assertEquals(early.endTarget(), 0.01, EPSILON);
    assertEquals(early.target(12), 0.02, EPSILON);
    assertEquals(early.target(100), 0.02, EPSILON);
    assertEquals(early.target(550), 0.015, 1e-12);
    assertEquals(early.target(325), 0.01 + 0.01 * 0.5 * (1 + Math.cos(Math.PI / 4)), 1e-12);
    assertEquals(early.target(1_000), 0.01, EPSILON);
    assertEquals(early.target(5_000), 0.01, EPSILON);
    Schedule late = Schedule.of(300, 0.02, planned);
    assertEquals(late.decayStep(), 300);
    assertEquals(late.target(650), 0.015, 1e-12);
    Schedule constant = Schedule.of(300, 0.02, flat());
    assertEquals(constant.target(1_000_000), 0.02);
    assertThrows(IllegalStateException.class, () -> Schedule.of(1_000, 0.02, planned));

    KlLearningRate rate = new KlLearningRate(planned);
    rate.accept(0.016, 100);
    rate.accept(0.016, 100);
    rate.accept(0.016, 100);
    Adjustment held = rate.accept(0.02, 250);
    assertEquals(rate.state().schedule().decayStep(), 300);
    assertEquals(held.target(), 0.01 + 0.01 * 0.5 * (1 + Math.cos(Math.PI * 250 / 700)), 1e-12);
    assertEquals(held.action(), Action.HOLD);
  }

  @Test
  public void aRejectedMacroChangesNothing() {
    KlLearningRate rate = warmed(flat());
    State before = rate.state();
    Adjustment rejected = rate.reject(0.9);
    assertEquals(rejected.action(), Action.REJECTED);
    assertEquals(rejected.applied(), before.rate());
    assertEquals(rejected.next(), before.rate());
    assertEquals(rejected.observed(), 0.9);
    assertEquals(rejected.median(), 0.02, EPSILON);
    assertEquals(rate.state(), before);
    KlLearningRate fresh = new KlLearningRate(flat());
    Adjustment early = fresh.reject(0.1);
    assertTrue(Double.isNaN(early.median()) && Double.isNaN(early.target()));
    assertEquals(fresh.state().macros(), 0);
  }

  /** 書き出した状態（Gson で往復した JSON）から続けると、途切れずに続けたのと同じ動きになる。 */
  @Test
  public void aSavedStateContinuesTheSameWay() {
    Settings planned = new Settings(1e-4, 1e-6, 1e-3, 3, 3, 1.25, 1.25, 1.02, 400, 0.5);
    double[] kls = {0.02, 0.01, 0.03, 0.002, 0.001, 0.0005, 0.04, 0.05, 0.06, 0.02, 0.025, 0.01};
    for (int split = 0; split <= kls.length; split++) {
      KlLearningRate unbroken = new KlLearningRate(planned);
      KlLearningRate first = new KlLearningRate(planned);
      for (int macro = 0; macro < split; macro++) {
        assertEquals(first.accept(kls[macro], 30), unbroken.accept(kls[macro], 30));
      }
      Gson gson = new Gson();
      State saved = gson.fromJson(gson.toJson(first.state()), State.class);
      assertEquals(saved, first.state());
      KlLearningRate resumed = new KlLearningRate(planned, saved);
      for (int macro = split; macro < kls.length; macro++) {
        assertEquals(resumed.accept(kls[macro], 30), unbroken.accept(kls[macro], 30), "macro " + macro);
      }
      assertEquals(resumed.state(), unbroken.state());
    }
  }

  @Test
  public void badInputsAreRejected() {
    KlLearningRate rate = new KlLearningRate(flat());
    assertThrows(IllegalArgumentException.class, () -> rate.accept(-1e-3, 1));
    assertThrows(IllegalArgumentException.class, () -> rate.accept(Double.NaN, 1));
    assertThrows(IllegalArgumentException.class, () -> rate.accept(Double.POSITIVE_INFINITY, 1));
    assertThrows(IllegalArgumentException.class, () -> rate.accept(0.01, 0));
    assertThrows(IllegalArgumentException.class, () -> rate.reject(-1));
    assertThrows(IllegalArgumentException.class, () -> new Settings(1e-4, 2e-4, 1e-3, 3, 3, 1.25, 1.25, 1.02, 0, 0.5));
    assertThrows(IllegalArgumentException.class, () -> new Settings(1e-4, 1e-6, 1e-3, 0, 3, 1.25, 1.25, 1.02, 0, 0.5));
    assertThrows(IllegalArgumentException.class, () -> new Settings(1e-4, 1e-6, 1e-3, 3, 3, 1.25, 1.0, 1.02, 0, 0.5));
    assertThrows(IllegalArgumentException.class, () -> new Settings(1e-4, 1e-6, 1e-3, 3, 3, 1.25, 1.25, 1.02, 0, 0));

    KlLearningRate warmed = warmed(flat());
    State state = warmed.state();
    assertThrows(
        IllegalArgumentException.class,
        () -> new KlLearningRate(flat(), new State(3, 3, 2e-3, state.warmup(), state.recent(), state.baseline(), state.schedule())));
    assertThrows(
        IllegalArgumentException.class,
        () -> new KlLearningRate(flat(), new State(2, 3, 1e-4, state.warmup(), state.recent(), state.baseline(), state.schedule())));
    assertThrows(
        IllegalArgumentException.class,
        () -> new KlLearningRate(flat(), new State(2, 3, 1e-4, List.of(0.1, 0.2), List.of(0.1, 0.2), state.baseline(), state.schedule())));
    assertThrows(
        IllegalArgumentException.class, () -> new State(0, 0, 1e-4, List.of(), List.of(), 0.02, null));
  }

  /** ウォームアップの KL 0.02・0.01・0.03（どれも 1 ステップ）を受け入れた制御。目標は 0.025。 */
  private static KlLearningRate warmed(Settings settings) {
    KlLearningRate rate = new KlLearningRate(settings);
    rate.accept(0.02, 1);
    rate.accept(0.01, 1);
    rate.accept(0.03, 1);
    return rate;
  }
}
