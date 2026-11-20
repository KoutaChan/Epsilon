package com.epsilon.prism.backend;

import static com.epsilon.prism.input.InputSchema.FOUR_PLAYER;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.observe.EventRecords.Prefix;
import com.epsilon.agent.observe.Observation;
import com.epsilon.prism.Limits;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.policy.PolicyRequests;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import org.testng.annotations.Test;

/**
 * レーンのイベントの転送の世代を、CPU のページ表で確かめる。取り消しや失敗で世代が進むと、それより前の予約はページを使わずに全てを送る。
 * 次の準備は空のページ表から始め直して全てのイベントを送り、その次の同じ履歴はまた命中する。ページを使う提出そのものが失敗すると、
 * 重みを公開し直すまで同じ履歴でもページを使わない。
 */
public class EventTransfersTest {
  private static final int ROWS = 2;

  /** 同じ観測席の履歴を持つ行のバッチと配置表、提出の記録。 */
  private static final class Lane {
    final InputFormat format = ShapeArm.V4_S.shape().inputFormat(FOUR_PLAYER);
    final BatchPlan.Capacity capacity =
        new BatchPlan.Capacity(
            ROWS,
            ROWS * Limits.MAX_EVENT_TOKENS,
            ROWS * Limits.MAX_TRANSITIONS,
            ROWS * Limits.MAX_LEGAL_ACTIONS);
    final HostInputs host = HostInputs.allocate(format, capacity, ByteBuffer::allocateDirect);
    final EventPages pages = new EventPages(format, capacity, ByteBuffer::allocateDirect);
    final EventTransfers transfers = new EventTransfers(format, capacity.events(), true);
    final List<Integer> paged = new ArrayList<>();
    final Prefix prefix;
    int full;

    Lane() {
      var view = new Observation(format.variant());
      for (long[] words : PolicyRequests.observations(31, TableWalk.Policy.CALLS, 200)) {
        view.bind(words, 0);
        if (view.events() > 0) break;
      }
      assertTrue(view.events() > 0, "a decision with public events");
      host.plan().clear();
      for (int row = 0; row < ROWS; row++) assertEquals(host.plan().add(view), row);
      host.plan().seal();
      prefix = Prefix.append(null, view.events(), view.kanDoraIndicators());
      var writer = format.writer();
      for (int row = 0; row < ROWS; row++) {
        writer.write(view, host, row);
        pages.observe(row, prefix);
      }
      transfers.configure(capacity.events());
    }

    int events() {
      return host.plan().events(0);
    }

    Runnable prepare() {
      return prepare((misses, slots, bytes) -> paged.add(misses));
    }

    Runnable prepare(EventTransfers.Paged submit) {
      Runnable step = transfers.prepare(host.input(), pages, () -> full++, submit);
      return () -> {
        try {
          step.run();
        } finally {
          pages.complete();
        }
      };
    }

    void close() {
      pages.clear(ROWS);
      prefix.release();
      transfers.close();
    }
  }

  @Test
  public void aCancelledGenerationRestartsFromAnEmptyPageMap() {
    var lane = new Lane();
    int events = lane.events();
    lane.prepare().run();
    lane.prepare().run();
    assertEquals(lane.paged, List.of(events, 0), "the first transfer sends every event");
    var stale = lane.prepare();
    lane.transfers.invalidate();
    stale.run();
    assertEquals(lane.full, 1, "a reservation from the cancelled generation sends every event");
    assertEquals(lane.paged.size(), 2);
    lane.prepare().run();
    lane.prepare().run();
    assertEquals(
        lane.paged,
        List.of(events, 0, events, 0),
        "the next generation starts from an empty page map and hits again");
    assertEquals(lane.full, 1);
    lane.close();
  }

  @Test
  public void aFailedPagedSubmissionStopsPagingUntilTheWeightsAreRepublished() {
    var lane = new Lane();
    int events = lane.events();
    lane.prepare().run();
    var failure =
        expectThrows(
            IllegalStateException.class,
            () ->
                lane.prepare(
                        (misses, slots, bytes) -> {
                          throw new IllegalStateException("paged submission failed");
                        })
                    .run());
    assertEquals(failure.getMessage(), "paged submission failed");
    lane.prepare().run();
    lane.prepare().run();
    assertEquals(lane.full, 2, "the lane stops paging after the plan failed a paged submission");
    lane.transfers.published();
    lane.prepare().run();
    lane.prepare().run();
    assertEquals(lane.paged, List.of(events, events, 0));
    assertEquals(lane.full, 2);
    lane.close();
  }
}
