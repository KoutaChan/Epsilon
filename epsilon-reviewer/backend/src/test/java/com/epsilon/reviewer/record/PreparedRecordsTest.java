package com.epsilon.reviewer.record;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.store.OwnerId;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneId;
import java.time.ZoneOffset;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 読み込んだ牌譜の上限と期限を確かめる。上限はオーナーごとに 4 件、全体で 64 件・128 MiB で、解析を待っている牌譜も数える。期限が
 * あるのは WAITING の牌譜だけで、30 分。
 */
public class PreparedRecordsTest {
  private static final OwnerId OWNER = OwnerId.fromCookieKey("A".repeat(43));
  private static final OwnerId OTHER = OwnerId.fromCookieKey("B".repeat(43));
  private static final byte[] BYTES = TestFiles.played(Variant.TENHOU_4P, 0);
  private static final LoadedRecord RECORD = RecordLoader.load(BYTES);

  @Test
  public void ownersAndTheWholeStoreHaveLimits() {
    var records = new PreparedRecords();
    for (int i = 0; i < PreparedRecords.PER_OWNER; i++) records.add(OWNER, RECORD, BYTES);
    refused(() -> records.add(OWNER, RECORD, BYTES), ErrorCode.TOO_MANY_RECORDS);
    records.add(OTHER, RECORD, BYTES);
    var full = new PreparedRecords();
    for (int i = 0; i < PreparedRecords.TOTAL; i++)
      full.add(OwnerId.fromCookieKey(String.format("%043d", i / 4)), RECORD, BYTES);
    refused(() -> full.add(OWNER, RECORD, BYTES), ErrorCode.TOO_MANY_RECORDS);
    var bytes = new PreparedRecords();
    bytes.add(OWNER, RECORD, new byte[100 * 1024 * 1024]);
    refused(() -> bytes.add(OTHER, RECORD, new byte[29 * 1024 * 1024]), ErrorCode.TOO_MANY_RECORDS);
    bytes.add(OTHER, RECORD, new byte[28 * 1024 * 1024]);
  }

  @Test
  public void takenRecordsCountUntilReleasedAndCanBeRestored() {
    var records = new PreparedRecords();
    PreparedRecord record = records.add(OWNER, RECORD, BYTES);
    refused(() -> records.take(OTHER, record.id()), ErrorCode.RECORD_EXPIRED);
    Assert.assertSame(records.take(OWNER, record.id()), record);
    refused(() -> records.take(OWNER, record.id()), ErrorCode.RECORD_EXPIRED);
    records.restore(record);
    records.take(OWNER, record.id());
    for (int i = 1; i < PreparedRecords.PER_OWNER; i++) records.add(OWNER, RECORD, BYTES);
    refused(() -> records.add(OWNER, RECORD, BYTES), ErrorCode.TOO_MANY_RECORDS);
    records.release(record);
    records.add(OWNER, RECORD, BYTES);
    refused(() -> records.take(OWNER, record.id()), ErrorCode.RECORD_EXPIRED);
  }

  @Test
  public void waitingRecordsExpireAfterThirtyMinutesButQueuedOnesStay() {
    var clock = new MovingClock();
    var records = new PreparedRecords(clock);
    PreparedRecord waiting = records.add(OWNER, RECORD, BYTES);
    PreparedRecord queued = records.add(OWNER, RECORD, BYTES);
    records.take(OWNER, queued.id());
    clock.now = clock.now.plus(PreparedRecords.LIFETIME).plusSeconds(1);
    records.sweep();
    refused(() -> records.take(OWNER, waiting.id()), ErrorCode.RECORD_EXPIRED);
    for (int i = 1; i < PreparedRecords.PER_OWNER; i++) records.add(OWNER, RECORD, BYTES);
    refused(() -> records.add(OWNER, RECORD, BYTES), ErrorCode.TOO_MANY_RECORDS);
    records.release(queued);
    records.add(OWNER, RECORD, BYTES);
  }

  private static void refused(Assert.ThrowingRunnable action, ErrorCode code) {
    Assert.assertEquals(Assert.expectThrows(ReviewerException.class, action).code(), code);
  }

  private static final class MovingClock extends Clock {
    Instant now = Instant.parse("2026-11-04T00:00:00Z");

    @Override
    public ZoneId getZone() {
      return ZoneOffset.UTC;
    }

    @Override
    public Clock withZone(ZoneId zone) {
      return this;
    }

    @Override
    public Instant instant() {
      return now;
    }
  }
}
