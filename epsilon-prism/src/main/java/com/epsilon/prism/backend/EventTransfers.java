package com.epsilon.prism.backend;

import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.input.PublishedEventStore;

/**
 * レーンに残る公開履歴を照合し、欠けたイベントの転送を提出順に準備する。失敗や取り消しがあるとページ表を空にして世代を進め、それより前の
 * 予約は全てのイベントを送る。次の準備からは空のページ表で始め直す（最初の提出は全てのイベントを送り、その後の同じ履歴はまた命中する）。
 * ページを使う提出そのものが失敗したときは、native の計画がページを使えなくなるので、重みを公開し直すまでページを使わない。
 */
public final class EventTransfers implements AutoCloseable {
  public interface Paged {
    void submit(int misses, int slots, int bytes);
  }

  private final InputFormat format;
  private final int maxEvents;
  private final boolean reusable;
  private volatile EventPages.Resident resident;
  private volatile boolean unusable;
  private boolean disabled;
  private Runnable release;

  public EventTransfers(InputFormat format, int maxEvents, boolean reusable) {
    this.format = format;
    this.maxEvents = maxEvents;
    this.reusable = reusable;
  }

  public void configure(int slots) {
    close();
    release = PublishedEventStore.register(format, slots);
    disabled = slots == 0;
    resident = disabled ? null : new EventPages.Resident(slots);
  }

  /** ページ表を空にして世代を進める。それより前の予約は全てのイベントを送る。 */
  public void invalidate() {
    var current = resident;
    if (current != null) current.clear();
  }

  /** 重みを公開し直した後に呼ぶ。保存したイベントは前の重みで作ったので捨て、ページをまた使う。 */
  public void published() {
    unusable = false;
    invalidate();
  }

  public Runnable prepare(HostBatch batch, EventPages pages, Runnable submit, Paged paged) {
    Runnable full =
        () -> {
          batch.materializeEvents();
          submit.run();
        };
    if (pages == null || unusable || disabled) return full;
    if (!reusable) throw new IllegalArgumentException("Event prefix is not reusable");
    var current = resident;
    if (current == null) resident = current = new EventPages.Resident(maxEvents);
    int slots = current.slots();
    if (batch.plan().count(Axis.EVENT) > slots) return full;
    try {
      var reservation = pages.reserve(batch.plan(), current);
      if (reservation == null) return full;
      int misses = reservation.misses();
      int bytes = pages.stage(batch, misses);
      return () -> {
        boolean submitted;
        try {
          submitted =
              !unusable
                  && reservation.submit(
                      () -> {
                        try {
                          paged.submit(misses, slots, bytes);
                        } catch (RuntimeException | Error e) {
                          unusable = true;
                          throw e;
                        }
                      });
        } catch (RuntimeException | Error e) {
          invalidate();
          throw e;
        }
        if (!submitted) full.run();
      };
    } catch (RuntimeException | Error e) {
      invalidate();
      throw e;
    }
  }

  public void synchronize(Runnable wait) {
    try {
      if (resident == null) wait.run();
      else resident.synchronize(wait);
    } catch (RuntimeException | Error failure) {
      invalidate();
      throw failure;
    }
  }

  @Override
  public void close() {
    if (resident != null) {
      resident.clear();
      resident = null;
    }
    if (release != null) {
      release.run();
      release = null;
    }
  }
}
