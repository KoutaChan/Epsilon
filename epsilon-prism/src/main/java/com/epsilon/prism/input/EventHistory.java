package com.epsilon.prism.input;

import com.epsilon.agent.observe.EventRecords.Prefix;
import com.epsilon.agent.observe.Observation;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * 観測席の公開イベントについて、前の判断までに符号化した列を保持する。
 *
 * <p>経路の公開順があり、公開した時点でコードが確定する入力形式では、{@link Bytes} がカテゴリ・スカラー・関係の各列を
 * {@link Prefix} の区間ごとに持つ。分岐は共通の区間を共有し、ライターはその後のイベントだけを符号化する。
 * GPU の提出では {@link HostBatch#copyEvents} が必要な区間のバイトを取り出す。
 *
 * <p>公開順を持たない記録 O に履歴を渡すと、ライターは保存した記録のビット列と槓ドラ表示を O と照合する。
 * 一致した確定区間のイベントと関係の下書きを使い回す。最後の打牌は鳴いた席が後で決まるため、照合の対象は最後の記録の直前まで。
 * この履歴を使う経路は {@link #copy} で分ける。同じ履歴に重なった書き込みは、使用中の印を取れた方だけが履歴を使う。
 */
public final class EventHistory {

  /** 同じ経路・席の追加分をプリミティブのページ番号で持つ。親は分岐時に確定した位置まで共有する。 */
  static final class Bytes implements Prefix.Encoding {
    final InputFormat format;
    final Bytes parent, other;
    private final PublishedEventStore store;
    private final int first;
    private int[] pages = new int[2];
    private int pageCount;
    private volatile long extent;

    int end() {
      return (int) extent;
    }

    long extent() {
      return extent;
    }

    private Bytes(InputFormat format, Bytes parent, Bytes other) {
      this.format = format;
      this.parent = parent;
      this.other = other;
      store = parent == null ? PublishedEventStore.of(format) : parent.store;
      first = parent == null ? 0 : parent.end();
      extent = parent == null ? 0 : parent.extent;
    }

    @Override
    public void unused() {
      for (Bytes bytes = this; bytes != null; bytes = bytes.other)
        bytes.store.give(bytes.pages, bytes.pageCount);
    }

    private static Bytes local(InputFormat format, Prefix tip) {
      for (Bytes b = (Bytes) tip.encoding(); b != null; b = b.other)
        if (b.format == format) return b;
      return null;
    }

    static Bytes find(InputFormat format, Prefix tip) {
      for (Prefix p = tip; p != null; p = p.parent) {
        Bytes bytes = local(format, p);
        if (bytes != null) return bytes;
      }
      return null;
    }

    static Bytes publish(
        InputFormat format,
        Bytes previous,
        Prefix tip,
        byte[] categories,
        short[] scalars,
        byte[] relations) {
      if (previous != null && previous.end() == tip.end) return previous;
      Bytes bytes = local(format, tip);
      Bytes parent =
          bytes != null || tip.parent == null
              ? null
              : publish(format, previous, tip.parent, categories, scalars, relations);
      if (bytes == null && tip.parent != null && (parent == null || parent.end() < tip.first()))
        return parent;
      synchronized (tip) {
        bytes = local(format, tip);
        if (bytes == null) {
          bytes = new Bytes(format, parent, (Bytes) tip.encoding());
          if (!bytes.append(tip, categories, scalars, relations)) return parent;
          tip.encoding(bytes);
        } else {
          bytes.append(tip, categories, scalars, relations);
        }
        return bytes;
      }
    }

    private boolean append(Prefix tip, byte[] categories, short[] scalars, byte[] relations) {
      int required = (tip.end - first + PublishedEventStore.ROWS - 1) / PublishedEventStore.ROWS;
      if (required > pageCount) {
        if (required > pages.length)
          pages = Arrays.copyOf(pages, Math.max(required, pages.length * 2));
        if (!store.take(pages, pageCount, required)) return false;
        pageCount = required;
      }
      for (int e = end(); e < tip.end; e++) {
        int index = (e - first) / PublishedEventStore.ROWS;
        store.store(
            pages[index],
            (e - first) % PublishedEventStore.ROWS,
            e,
            format.fixedTokens(),
            categories,
            scalars,
            relations);
      }
      extent = ((long) tip.records << Integer.SIZE) | Integer.toUnsignedLong(tip.end);
      return true;
    }

    void copy(int from, int count, ByteBuffer out, int categories, int scalars, int relations) {
      int c = format.event().width(),
          s = store.scalars * Short.BYTES,
          r = format.relation().width();
      int until = from + count;
      for (Bytes range = this; until > from; range = range.parent) {
        int start = Math.max(range.first, from);
        for (int e = start; e < until; e++) {
          int i = e - range.first, at = e - from;
          range.store.copy(
              range.pages[i / PublishedEventStore.ROWS],
              i % PublishedEventStore.ROWS,
              out,
              categories + at * c,
              scalars + at * s,
              relations + at * r);
        }
        until = Math.min(until, start);
      }
    }
  }

  private static final VarHandle BUSY;

  static {
    try {
      BUSY = MethodHandles.lookup().findVarHandle(EventHistory.class, "busy", int.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  /** 1 つの槓ドラの表示を詰めるビットの数（前の記録の番号 10 ビットと牌の種類 6 ビット）。 */
  private static final int INDICATOR_BITS = 16;

  private volatile int busy;

  /** 中身を作った入力の形式（最初に使うまで null）。 */
  private InputFormat format;

  /** 中身が最後まで書けているか。途中で失敗した書き込みは使わない。 */
  private boolean valid;

  /** 確定した記録の数と、その生のビット列。 */
  private int records;

  private long[] words = new long[8];

  /** 確定した区間に入る槓ドラの表示の数と、表示ごとの前の記録の番号と牌の種類（{@link #INDICATOR_BITS} ずつ詰める）。 */
  private int indicators;

  private long indicatorKeys;

  /** 確定した公開の位置の数と、その位置のイベントの下書きの行と、関係の下書きのイベントの行。 */
  private int positions;

  private byte[] eventRows = new byte[0];
  private byte[] relationRows = new byte[0];

  /** 空の履歴を作る（経路の最初の判断の前）。 */
  public EventHistory() {}

  private EventHistory(EventHistory from) {
    format = from.format;
    valid = true;
    records = from.records;
    words = Arrays.copyOf(from.words, from.words.length);
    indicators = from.indicators;
    indicatorKeys = from.indicatorKeys;
    positions = from.positions;
    eventRows = Arrays.copyOf(from.eventRows, positions * format.event().width());
    relationRows = Arrays.copyOf(from.relationRows, positions * format.relation().width());
  }

  /**
   * 分岐した経路の履歴を、この履歴の写しで作る。経路の判断の答えが出た後（ライターが使っていないとき）に呼ぶ。使用中なら空の履歴を
   * 返す（出力は変わらない）。
   */
  public EventHistory copy() {
    if (!BUSY.compareAndSet(this, 0, 1)) return new EventHistory();
    try {
      return format == null || !valid ? new EventHistory() : new EventHistory(this);
    } finally {
      busy = 0;
    }
  }

  /** 経路が次の局に入った。中身を捨てる（捨てなくても、次の判断の確かめで捨てる）。 */
  public void clear() {
    if (!BUSY.compareAndSet(this, 0, 1)) return;
    try {
      forget();
    } finally {
      busy = 0;
    }
  }

  /**
   * 使用中の印を取る。中身が別の形式で作られていれば捨てる。
   *
   * @return 印を取れたら true（{@link #release} で返す）
   */
  boolean acquire(InputFormat format) {
    if (!BUSY.compareAndSet(this, 0, 1)) return false;
    if (this.format != format) {
      this.format = format;
      forget();
    }
    return true;
  }

  void release() {
    busy = 0;
  }

  /**
   * 確定した記録と槓ドラの表示が O の先頭と同じなら、その記録の数を返す。違えば中身を捨てて 0 を返す。確定した記録は O の最後の記録より
   * 前にあること（同じ経路・観測席の後の判断では、記録は増えるだけ）。
   */
  int match(Observation o) {
    if (valid
        && records > 0
        && records < o.events()
        && o.sameEvents(words, records)
        && sameIndicators(o)) {
      return records;
    }
    forget();
    return 0;
  }

  /**
   * この判断で確定した記録を覚える。下書きの行は {@link #saveRows} で写し、そこで中身が書き終わる。
   *
   * @param to 確定した記録の数
   */
  void settle(Observation o, int to) {
    valid = false;
    words = o.copyEvents(to, words);
    records = to;
    indicators = 0;
    indicatorKeys = 0L;
    for (int k = 0, count = o.kanDoraIndicators(); k < count && o.kanDoraEvent(k) < to; k++) {
      indicatorKeys |= indicatorKey(o, k) << (k * INDICATOR_BITS);
      indicators++;
    }
  }

  /** 確定した公開の位置の数。 */
  int positions() {
    return positions;
  }

  byte[] eventRows() {
    return eventRows;
  }

  byte[] relationRows() {
    return relationRows;
  }

  /**
   * 下書きの、確定した公開の位置 {@code [positions, to)} の行を写し、中身を書き終える。{@code [0, positions)} の行は、この判断で
   * 下書きに写した行なので写し直さない。
   *
   * @param to この判断で確定した公開の位置の数
   */
  void saveRows(
      byte[] eventDraft,
      int eventWidth,
      byte[] relationDraft,
      int relationAt,
      int relationWidth,
      int to) {
    int from = positions;
    if (to < from)
      throw new IllegalStateException("settled positions shrank from " + from + " to " + to);
    if (eventRows.length < to * eventWidth)
      eventRows = Arrays.copyOf(eventRows, Math.max(to, 2 * from) * eventWidth);
    if (relationRows.length < to * relationWidth) {
      relationRows = Arrays.copyOf(relationRows, Math.max(to, 2 * from) * relationWidth);
    }
    System.arraycopy(
        eventDraft, from * eventWidth, eventRows, from * eventWidth, (to - from) * eventWidth);
    System.arraycopy(
        relationDraft,
        relationAt + from * relationWidth,
        relationRows,
        from * relationWidth,
        (to - from) * relationWidth);
    positions = to;
    valid = true;
  }

  private boolean sameIndicators(Observation o) {
    int count = o.kanDoraIndicators();
    if (count < indicators) return false;
    for (int k = 0; k < indicators; k++) {
      if (indicatorKey(o, k) != (indicatorKeys >>> (k * INDICATOR_BITS) & 0xFFFF)) return false;
    }
    return count == indicators || o.kanDoraEvent(indicators) >= records;
  }

  /** k 番目の槓ドラの表示の、前の記録の番号と牌の種類を詰めた値。 */
  private static long indicatorKey(Observation o, int k) {
    return (long) o.kanDoraEvent(k) << 6 | o.indicatorTile(k + 1);
  }

  private void forget() {
    valid = false;
    records = 0;
    positions = 0;
    indicators = 0;
    indicatorKeys = 0L;
  }
}
