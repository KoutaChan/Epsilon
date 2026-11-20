package com.epsilon.prism.backend;

import com.epsilon.agent.observe.EventRecords.Prefix;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFeature;
import com.epsilon.prism.input.InputFormat;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.function.IntFunction;

/** 公開イベントの GPU 上の保存先を決め、未収載の区間の入力を配置表の後ろに詰める。 */
public final class EventPages {

  /** 経路が分岐点から独自に追加した位置の配置表。祖先の列はそのまま参照する。 */
  private static final class Binding implements Prefix.Placement {
    volatile Resident pool;
    final Object epoch;
    Binding other, older, newer, releasedNext;
    EventSlotPool.Bin bin;
    int index, used;
    long scan, lastUse;
    volatile boolean assigned = true;

    Binding(Resident pool, Binding other) {
      this.pool = pool;
      this.epoch = pool.epoch;
      this.other = other;
    }

    @Override
    public void unused() {
      for (Binding b = this; b != null; b = b.other) {
        Resident owner = b.pool;
        if (owner != null) owner.release(b);
      }
    }
  }

  /** デバイスのイベント数を上限にスロットを貸す。空き番号はプリミティブの列にまとめて再利用する。 */
  public static final class Resident {
    private final int capacity;
    private final EventSlotPool slots;
    private volatile Object epoch = new Object();
    private Binding oldest, newest, released;
    private final Object order = new Object();
    private long prepared, submitted, scan;
    private volatile long completed;

    public Resident(int slots) {
      if (slots <= 0) throw new IllegalArgumentException("event cache must have slots");
      capacity = slots;
      this.slots = new EventSlotPool(slots);
    }

    int slots() {
      return capacity;
    }

    public void clear() {
      synchronized (order) {
        reset();
      }
    }

    void synchronize(Runnable wait) {
      synchronized (order) {
        wait.run();
        completed = submitted;
      }
    }

    private synchronized void reset() {
      epoch = new Object();
      prepared = submitted = completed = scan = 0;
      for (Binding b = oldest; b != null; ) {
        Binding next = b.newer;
        b.assigned = false;
        b.pool = null;
        b.newer = b.older = b.releasedNext = null;
        b.bin = null;
        b = next;
      }
      oldest = newest = released = null;
      slots.clear();
    }

    private synchronized void release(Binding b) {
      if (b.assigned && b.epoch == epoch) {
        b.releasedNext = released;
        released = b;
      }
    }

    private synchronized Binding drain() {
      Binding head = released;
      released = null;
      return head;
    }

    private void reclaim(Binding b) {
      if (!b.assigned) return;
      b.assigned = false;
      b.pool = null;
      if (b.older == null) oldest = b.newer;
      else b.older.newer = b.newer;
      if (b.newer == null) newest = b.older;
      else b.newer.older = b.older;
      b.older = b.newer = null;
      if (b.bin != null) {
        slots.give(b.bin.slab(b.index), b.bin.offset(b.index), b.used);
        b.bin.give(b.index);
        b.bin = null;
      }
    }

    /** 同一ストリームの後続の受付で使う場所を用意する。終了した経路を先に返す。 */
    private void reclaimReleased(int events) {
      if (events > capacity)
        throw new IllegalArgumentException("event batch exceeds cache capacity");
      for (Binding b = drain(); b != null; ) {
        Binding next = b.releasedNext;
        b.releasedNext = null;
        if (b.lastUse <= completed) reclaim(b);
        else release(b);
        b = next;
      }
    }

    private Binding binding(Prefix tip) {
      Binding first = (Binding) tip.placement();
      if (first != null && first.epoch == epoch && first.assigned) return first;
      synchronized (tip) {
        Binding others = null;
        for (Binding b = (Binding) tip.placement(); b != null; b = b.other) {
          if (b.epoch == epoch && b.assigned) return b;
        }
        for (Binding b = (Binding) tip.placement(); b != null; ) {
          Binding next = b.other;
          if (b.assigned) {
            b.other = others;
            others = b;
          }
          b = next;
        }
        Binding b = new Binding(this, others);
        b.older = newest;
        if (newest == null) oldest = b;
        else newest.newer = b;
        newest = b;
        tip.place(b);
        return b;
      }
    }
  }

  /** 同じレーンの先行提出が成功し、重みの世代が変わっていなければ配置表を使う。 */
  public static final class Reservation {
    private final Resident resident;
    private final Object epoch;
    private final long sequence;
    private final int misses;

    private Reservation(Resident resident, long sequence, int misses) {
      this.resident = resident;
      this.epoch = resident.epoch;
      this.sequence = sequence;
      this.misses = misses;
    }

    public int misses() {
      return misses;
    }

    /**
     * 予約順で同じストリームに積む。予約の後にページ表を空にした（失敗や取り消しで世代が進んだ）なら、何も積まずに false を返す。その時、
     * 呼び出し側はページを使わずに全てを送る。途中まで書いた提出に続く予約はすべて失効する。
     *
     * @return 積んだら true、予約が古い世代のものなら false
     * @throws IllegalStateException 同じ世代の前の予約がまだ積まれていないとき
     */
    public boolean submit(Runnable enqueue) {
      synchronized (resident.order) {
        if (resident.epoch != epoch) return false;
        if (resident.submitted + 1 != sequence)
          throw new IllegalStateException("out-of-order event page reservation");
        try {
          enqueue.run();
          resident.submitted = sequence;
          return true;
        } catch (RuntimeException | Error failure) {
          resident.clear();
          throw failure;
        }
      }
    }

    /** この受付の完了通知までに同じストリームで使い終えたスロットを再利用できるようにする。 */
    public void complete() {
      synchronized (resident.order) {
        if (resident.epoch != epoch) return;
        if (sequence > resident.submitted)
          throw new IllegalStateException("event page reservation has not been submitted");
        resident.completed = Math.max(resident.completed, sequence);
      }
    }
  }

  private final Prefix[] tips;
  private final int[] indices;
  private final ByteBuffer map;
  private final IntBuffer wire;
  private final ByteBuffer basicCodes;
  private HostBatch encoded;
  private Reservation pending;
  private int missed, mask;

  public EventPages(
      InputFormat format, BatchPlan.Capacity capacity, IntFunction<ByteBuffer> memory) {
    tips = new Prefix[capacity.rows()];
    indices = new int[capacity.events()];
    int stride =
        format.width(Tensor.EVENT_CATEGORY)
            + Short.BYTES * format.width(Tensor.EVENT_SCALAR)
            + format.width(Tensor.RELATION);
    int entries = Math.addExact(capacity.rows(), Math.multiplyExact(2, capacity.events()));
    int payload =
        Math.addExact(
            Math.multiplyExact(entries, Integer.BYTES),
            Math.multiplyExact(stride, capacity.events()));
    int basic =
        format.has(InputFeature.CANDIDATE_SEGMENTS)
            ? Math.multiplyExact(capacity.transitions(), format.fixedTokens())
            : 0;
    map = memory.apply(Math.addExact(payload, basic)).order(ByteOrder.LITTLE_ENDIAN);
    wire = map.asIntBuffer();
    basicCodes = basic == 0 ? null : map.slice(payload, basic);
  }

  /** 入力ライターに遷移ごとの書き先を渡す。経路の参照はバッチから借りる。 */
  public void bind(HostBatch batch) {
    encoded = batch;
    batch.basicCodes(basicCodes);
  }

  /** この行の観測席の区間を、受付が終わるまで保つ。 */
  public void observe(int row, Prefix prefix) {
    if (prefix != null) prefix.retain();
    replace(row, prefix);
  }

  /** 経路から独立した O の再生では、この行だけが区間を所有する。 */
  public void observeSnapshot(int row, int records, int indicators) {
    replace(row, Prefix.append(null, records, indicators));
  }

  private void replace(int row, Prefix prefix) {
    if (tips[row] != null) tips[row].release();
    tips[row] = prefix;
  }

  public void clear(int rows) {
    for (int row = 0; row < rows; row++) if (tips[row] != null) replace(row, null);
  }

  public ByteBuffer map() {
    return map;
  }

  /** 配置表の後に、GPU に無い区間のカテゴリ・スカラー・関係属性を列ごとに詰め、転送する総バイト数を返す。 */
  public int stage(HostBatch batch, int misses) {
    BatchPlan plan = batch.plan();
    int first = plan.rows() + plan.count(Axis.EVENT);
    int categories = (first + misses) * Integer.BYTES;
    int catWidth = plan.format().width(Tensor.EVENT_CATEGORY);
    int scalarWidth = Short.BYTES * plan.format().width(Tensor.EVENT_SCALAR);
    int relWidth = plan.format().width(Tensor.RELATION);
    int scalars = categories + misses * catWidth;
    int relations = scalars + misses * scalarWidth;
    int row = 0;
    for (int i = 0; i < misses; ) {
      int event = indices[i];
      while (event >= plan.firstEvent(row) + plan.events(row)) row++;
      int local = event - plan.firstEvent(row);
      int limit = Math.min(misses - i, plan.events(row) - local);
      int count = 1;
      while (count < limit && indices[i + count] == event + count) count++;
      batch.copyEvents(
          row,
          local,
          count,
          map,
          categories + i * catWidth,
          scalars + i * scalarWidth,
          relations + i * relWidth);
      i += count;
    }
    int end = relations + misses * relWidth;
    if (plan.format().has(InputFeature.CANDIDATE_SEGMENTS)) {
      int fixed = plan.format().fixedTokens(), width = plan.format().candidateCodes().codeWidth();
      if (encoded == batch) {
        int length = plan.count(Axis.TRANSITION) * fixed;
        map.put(end, basicCodes, 0, length);
        return end + length;
      }
      int source = plan.offset(Tensor.TRANSITION_CODE);
      var codes = batch.storage();
      for (int i = 0, count = plan.count(Axis.TRANSITION); i < count; i++) {
        map.put(end, codes, source, fixed);
        end += fixed;
        source += width;
      }
    }
    return end;
  }

  /** 元の行順で配置表を作る。先行受付がスロットを使っていて収まらないときは null を返す。 */
  public synchronized Reservation reserve(BatchPlan plan, Resident resident) {
    synchronized (resident.order) {
      return reserveInOrder(plan, resident);
    }
  }

  /** この配置表を使った GPU の処理が完了したことを通知する。 */
  public synchronized void complete() {
    if (pending != null) {
      pending.complete();
      pending = null;
    }
  }

  private Reservation reserveInOrder(BatchPlan plan, Resident resident) {
    if (pending != null
        && pending.epoch == pending.resident.epoch
        && pending.sequence > pending.resident.completed)
      throw new IllegalStateException("event page map is in use");
    pending = null;
    int rows = plan.rows(), events = plan.count(Axis.EVENT);
    for (int row = 0; row < rows; row++) {
      Prefix tip = tip(row);
      if ((tip == null ? 0 : tip.end) != plan.events(row))
        throw new IllegalStateException("incomplete event history");
    }
    resident.reclaimReleased(events);
    long sequence = Math.incrementExact(resident.prepared);
    try {
      if (resident.slots.available() < events) {
        long scan = ++resident.scan;
        int needed = 0;
        for (int row = 0; row < rows; row++) {
          for (Prefix p = tip(row); p != null; p = p.parent) {
            Binding b = resident.binding(p);
            if (b.scan == scan) break;
            b.scan = scan;
            needed += p.end - p.first() - b.used;
          }
        }
        for (Binding b = resident.oldest; resident.slots.available() < needed; ) {
          if (b == null) return null;
          Binding next = b.newer;
          if (b.scan != scan && b.lastUse <= resident.completed) resident.reclaim(b);
          b = next;
        }
      }
      int misses = pack(plan, resident, rows, events, sequence);
      resident.prepared = sequence;
      pending = new Reservation(resident, sequence, misses);
      return pending;
    } catch (RuntimeException | Error failure) {
      resident.clear();
      throw failure;
    }
  }

  private Prefix tip(int row) {
    Prefix tip = encoded == null ? null : encoded.publication(row);
    return tip == null ? tips[row] : tip;
  }

  private int pack(BatchPlan plan, Resident resident, int rows, int events, long sequence) {
    missed = 0;
    int fixed = plan.format().fixedTokens();
    for (int row = 0; row < rows; row++) {
      Prefix tip = tip(row);
      int start = plan.firstEvent(row);
      mask = 0;
      if (tip != null) {
        fill(tip, resident, start, fixed, rows, sequence);
      }
      wire.put(row, mask);
    }
    wire.put(rows + events, indices, 0, missed);
    return missed;
  }

  /** 親のスロットを共有し、各経路は自分で追記した範囲だけをバッチの対応する位置に写す。 */
  private void fill(Prefix tip, Resident resident, int start, int fixed, int rows, long sequence) {
    for (Prefix p = tip; p != null; p = p.parent) {
      Binding b = resident.binding(p);
      b.lastUse = sequence;
      int first = p.first(), count = p.end - first;
      if (b.bin == null || count > b.bin.width) {
        EventSlotPool.Bin next = resident.slots.bin(count);
        int index = next.take();
        if (b.bin != null) {
          next.slab(index)
              .put(next.offset(index), b.bin.slab(b.index), b.bin.offset(b.index), b.used);
          b.bin.give(b.index);
        }
        b.bin = next;
        b.index = index;
      }
      IntBuffer source = b.bin.slab(b.index);
      int offset = b.bin.offset(b.index);
      if (b.used < count) {
        int from = first + b.used;
        int firstTile = (fixed + from) / 16, lastTile = (fixed + p.end - 1) / 16;
        mask |= ((2 << lastTile) - 1) & (-1 << firstTile);
        resident.slots.append(source, offset + b.used, count - b.used);
        for (int e = from; e < p.end; e++) indices[missed++] = start + e;
        b.used = count;
      }
      wire.put(rows + start + first, source, offset, count);
    }
  }
}
