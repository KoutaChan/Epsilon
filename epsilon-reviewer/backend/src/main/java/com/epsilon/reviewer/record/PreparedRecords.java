package com.epsilon.reviewer.record;

import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.store.OwnerId;
import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

/**
 * 読み込んだ牌譜を、解析が終わるまで預かる。解析を待つ牌譜（WAITING）は 30 分で消し、解析のキューに入ったか解析中の牌譜（QUEUED）は
 * 解析が終わるまで残す。件数と大きさの上限は、2 つの状態を合わせて数える。
 */
public final class PreparedRecords {
  /** オーナーごとの件数の上限。 */
  static final int PER_OWNER = 4;

  /** 全体の件数の上限。 */
  static final int TOTAL = 64;

  /** 元のバイト列の合計の上限（バイト）。 */
  static final long BYTES = 128L * 1024 * 1024;

  /** WAITING の牌譜を預かる時間。 */
  static final Duration LIFETIME = Duration.ofMinutes(30);

  private final Clock clock;
  private final Map<RecordId, Entry> entries = new HashMap<>();
  private long bytes;

  public PreparedRecords() {
    this(Clock.systemUTC());
  }

  PreparedRecords(Clock clock) {
    this.clock = clock;
  }

  /** @throws ReviewerException too_many_records（オーナーごとに 4 件、全体で 64 件、合計 128 MiB を超える） */
  public synchronized PreparedRecord add(OwnerId owner, LoadedRecord record, byte[] source) {
    expire();
    long owned =
        entries.values().stream().filter(entry -> entry.record.owner().equals(owner)).count();
    if (owned >= PER_OWNER || entries.size() >= TOTAL || bytes + source.length > BYTES)
      throw new ReviewerException(ErrorCode.TOO_MANY_RECORDS);
    var prepared = new PreparedRecord(RecordId.random(), owner, record, source);
    entries.put(prepared.id(), new Entry(prepared, clock.instant()));
    bytes += source.length;
    return prepared;
  }

  /**
   * WAITING の牌譜を解析に渡し、QUEUED にする。
   *
   * @throws ReviewerException record_expired（牌譜が無い、期限が切れた、オーナーが違う、既に渡した）
   */
  public synchronized PreparedRecord take(OwnerId owner, RecordId id) {
    expire();
    Entry entry = entries.get(id);
    if (entry == null || entry.queued || !entry.record.owner().equals(owner))
      throw new ReviewerException(ErrorCode.RECORD_EXPIRED);
    entry.queued = true;
    return entry.record;
  }

  /** 解析が牌譜を受け付けられなかったとき、WAITING に戻す。 */
  public synchronized void restore(PreparedRecord record) {
    Entry entry = entries.get(record.id());
    if (entry != null) entry.queued = false;
  }

  /** 解析が終わった牌譜を外す。 */
  public synchronized void release(PreparedRecord record) {
    if (entries.remove(record.id()) != null) bytes -= record.source().length;
  }

  /** 期限を過ぎた WAITING の牌譜を外す。 */
  public synchronized void sweep() {
    expire();
  }

  private void expire() {
    Instant limit = clock.instant().minus(LIFETIME);
    entries
        .values()
        .removeIf(
            entry -> {
              boolean expired = !entry.queued && entry.created.isBefore(limit);
              if (expired) bytes -= entry.record.source().length;
              return expired;
            });
  }

  private static final class Entry {
    final PreparedRecord record;
    final Instant created;
    boolean queued;

    Entry(PreparedRecord record, Instant created) {
      this.record = record;
      this.created = created;
    }
  }
}
