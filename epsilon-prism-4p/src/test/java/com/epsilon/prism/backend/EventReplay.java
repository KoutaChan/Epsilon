package com.epsilon.prism.backend;

import com.epsilon.agent.observe.EventRecords.Prefix;
import com.epsilon.agent.observe.Observation;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.InputSchema;
import com.epsilon.prism.model.ShapeArm;
import java.nio.ByteBuffer;
import org.testng.Assert;

/**
 * イベントのページのテストで、同じ判断を 2 行に並べたバッチを作り、席ごとの公開の履歴をページに渡す。局が変われば履歴を捨てる。2 行目は
 * 1 行目と同じ履歴を共有する。
 */
final class EventReplay {
  static final int ROWS = 2;

  final InputFormat format;
  final BatchPlan.Capacity capacity = GpuCase.capacity(ROWS);
  final Prefix[] history;
  final HostInputs full;
  final EventPages pages;
  final InputFormat.RowWriter writer;
  final Observation view;
  private HostInputs other;
  private int kyoku = -1, honba = -1;

  EventReplay(ShapeArm arm) {
    this(arm.shape().inputFormat(InputSchema.FOUR_PLAYER));
  }

  EventReplay(InputFormat format) {
    this.format = format;
    history = new Prefix[format.seats()];
    full = HostInputs.allocate(format, capacity, ByteBuffer::allocateDirect);
    pages = new EventPages(format, capacity, ByteBuffer::allocateDirect);
    writer = format.writer();
    view = new Observation(format.variant());
  }

  /** 判断を 2 行に並べる。席の履歴に判断までのイベントを足し、両方の行にその履歴を渡す。 */
  void encode(long[] words) {
    view.bind(words, 0);
    if (kyoku != view.kyoku() || honba != view.honba()) {
      for (Prefix previous : history) if (previous != null) previous.release();
      java.util.Arrays.fill(history, null);
    }
    kyoku = view.kyoku();
    honba = view.honba();
    int seat = view.observer();
    full.plan().clear();
    for (int r = 0; r < ROWS; r++) Assert.assertEquals(full.plan().add(view), r);
    full.plan().seal();
    // 2 行目は親がこのバッチで追記したイベントも共有する。どちらの行もまだ GPU に送っていない。
    Prefix previous = history[seat];
    history[seat] = Prefix.append(previous, view.events(), view.kanDoraIndicators());
    if (previous != null && previous != history[seat]) previous.release();
    for (int r = 0; r < ROWS; r++) {
      writer.write(view, full, r);
      pages.observe(r, history[seat]);
    }
  }

  /** 2 つの判断を 1 行ずつ並べ、それぞれの履歴をページに渡す。 */
  void encodePair(long[] left, long[] right, Prefix first, Prefix second) {
    full.plan().clear();
    Assert.assertEquals(full.plan().add(view.bind(left, 0)), 0);
    Assert.assertEquals(full.plan().add(view.bind(right, 0)), 1);
    full.plan().seal();
    writer.write(view.bind(left, 0), full, 0);
    writer.write(view.bind(right, 0), full, 1);
    pages.observe(0, first);
    pages.observe(1, second);
  }

  /**
   * ページを使う提出の前に全量で流す、別の判断のバッチ。判断を 2 行に並べ、ページの履歴には渡さない。
   *
   * @param words 公開のイベントが多い判断（ページを使う提出の入力より長いこと）
   */
  void other(long[] words) {
    other = HostInputs.allocate(format, capacity, ByteBuffer::allocateDirect);
    var row = new Observation(format.variant()).bind(words, 0);
    other.plan().clear();
    for (int r = 0; r < ROWS; r++) Assert.assertEquals(other.plan().add(row), r);
    other.plan().seal();
    for (int r = 0; r < ROWS; r++) writer.write(row, other, r);
  }

  /** {@link #other} で作ったバッチ。 */
  HostInputs other() {
    Assert.assertNotNull(other, "the replay has no other batch");
    return other;
  }
}
