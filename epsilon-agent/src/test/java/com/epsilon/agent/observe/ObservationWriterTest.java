package com.epsilon.agent.observe;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.observe.ObservationSchema.HeaderBits;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import org.testng.annotations.Test;

/** 観測を書くクラスのテスト。ルールや席の食い違いと、観測のフィールドの範囲を外れる値を拒む。 */
public class ObservationWriterTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  @Test
  public void rejectsAnotherVariant() {
    Engine engine = firstDecision(RoundStart.first(VARIANT));
    Decisions decisions = engine.decisions();
    SeatView view = engine.view(decisions.seat(0));
    ObservationWriter writer = new ObservationWriter(VARIANT);
    ObservationBatch three = new ObservationBatch(Variant.TENHOU_3P, 64);
    expectThrows(IllegalArgumentException.class, () -> writer.capture(view, decisions, 0, three));

    ObservationBatch four = new ObservationBatch(VARIANT, 64);
    int index = writer.capture(view, decisions, 0, four);
    expectThrows(IllegalArgumentException.class, () -> new Observation(Variant.TENHOU_3P).bind(four, index));
    Observation read = new Observation(VARIANT).bind(four, index);
    expectThrows(IllegalArgumentException.class, () -> three.add(read));
    assertEquals(four.add(read), index + 1);
  }

  @Test
  public void rejectsAViewOfAnotherSeat() {
    Engine engine = firstDecision(RoundStart.first(VARIANT));
    Decisions decisions = engine.decisions();
    SeatView other = engine.view((decisions.seat(0) + 1) % VARIANT.seats());
    ObservationBatch batch = new ObservationBatch(VARIANT, 64);
    expectThrows(
        IllegalArgumentException.class, () -> new ObservationWriter(VARIANT).capture(other, decisions, 0, batch));
  }

  /**
   * 点数が 100 点単位でない卓と、本場が観測の上限を超える卓は書けず、バッチに何も残さない。観測が配列に収まっていなければ読めない。
   * フィールドの範囲を外れる値は書けない。
   */
  @Test
  public void rejectsValuesOutsideTheFields() {
    ObservationWriter writer = new ObservationWriter(VARIANT);
    ObservationBatch batch = new ObservationBatch(VARIANT, 64);
    expectThrows(
        IllegalArgumentException.class,
        () -> capture(writer, new RoundStart(VARIANT, 0, 0, 0, new int[] {25_050, 24_950, 25_000, 25_000}), batch));
    int[] even = {25_000, 25_000, 25_000, 25_000};
    expectThrows(
        IllegalArgumentException.class,
        () -> capture(writer, new RoundStart(VARIANT, 0, ObservationSchema.MAX_HONBA + 1, 0, even), batch));
    assertEquals(batch.size(), 0);

    int index = capture(writer, new RoundStart(VARIANT, 0, ObservationSchema.MAX_HONBA, 0, even), batch);
    long[] truncated = Arrays.copyOfRange(batch.words(), batch.start(index), batch.start(index) + batch.longs(index) - 1);
    expectThrows(IndexOutOfBoundsException.class, () -> new Observation(VARIANT).bind(truncated, 0));

    ObservationSchema schema = ObservationSchema.of(VARIANT);
    long[] header = new long[schema.header.longs(1)];
    schema.header.put(header, 0, 0, HeaderBits.WALL, VARIANT.liveWall());
    expectThrows(
        IllegalArgumentException.class, () -> schema.header.put(header, 0, 0, HeaderBits.WALL, VARIANT.liveWall() + 1));
    expectThrows(IllegalArgumentException.class, () -> schema.header.put(header, 0, 0, HeaderBits.INDICATORS, 0));
    expectThrows(IllegalArgumentException.class, () -> schema.header.get(header, 0, 0, HeaderBits.INDICATOR));
  }

  /** 卓の最初の判断で、判断する 1 番目の席の観測を書き、バッチの中の番号を返す。 */
  private static int capture(ObservationWriter writer, RoundStart start, ObservationBatch batch) {
    Engine engine = firstDecision(start);
    Decisions decisions = engine.decisions();
    return writer.capture(engine.view(decisions.seat(0)), decisions, 0, batch);
  }

  private static Engine firstDecision(RoundStart start) {
    Engine engine = new Engine(VARIANT);
    engine.startAt(start, 7, 1, DrawOrder.WALL_ORDER);
    assertEquals(engine.advance(new RuleWorkspace(VARIANT)), Boundary.DECISIONS);
    return engine;
  }
}
