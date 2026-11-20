package com.epsilon.prism.policy;

import ai.djl.pytorch.engine.PtEvent;
import com.epsilon.prism.input.HostBatch;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 全スロットの重みがそろう前に失敗したときの世代の寿命を確かめる。 */
public class LanePublicationTest {
  private static final class Lane implements NativeInference.Lane {
    final long generation;
    boolean closed;
    boolean failRelease;

    Lane(long generation) {
      this.generation = generation;
    }

    public void close() {
      closed = true;
      if (failRelease) throw new IllegalStateException("Release failed");
    }

    public void submit(HostBatch batch, ByteBuffer scores, long version) {
      if (closed || version != generation)
        throw new IllegalStateException("Generation unavailable");
    }

    public void record(PtEvent event) {
      throw new AssertionError();
    }

    public void synchronize() {
      throw new AssertionError();
    }
  }

  @Test
  public void retainsTheCompleteGenerationWhenAnySlotFails() {
    var active = NativeInference.openLanes(4, i -> new Lane(7));
    for (int slot = 0; slot < active.length; ++slot) {
      int failed = slot;
      var staged = new ArrayList<Lane>();
      var error =
          Assert.expectThrows(
              IllegalStateException.class,
              () ->
                  NativeInference.openLanes(
                      4,
                      i -> {
                        if (i == failed) throw new IllegalStateException("Weight transfer failed");
                        var lane = new Lane(8);
                        lane.failRelease = i == 0;
                        staged.add(lane);
                        return lane;
                      }));
      Assert.assertEquals(error.getMessage(), "Weight transfer failed");
      Assert.assertEquals(error.getSuppressed().length, failed > 0 ? 1 : 0);
      Assert.assertEquals(staged.size(), failed);
      for (var lane : staged) {
        Assert.assertTrue(lane.closed);
        Assert.expectThrows(IllegalStateException.class, () -> lane.submit(null, null, 8));
      }
      for (var lane : active) {
        lane.submit(null, null, 7);
        Assert.expectThrows(IllegalStateException.class, () -> lane.submit(null, null, 8));
      }
    }
    for (var lane : NativeInference.openLanes(4, i -> new Lane(8))) {
      lane.submit(null, null, 8);
      lane.close();
    }
    for (var lane : active) lane.close();
  }
}
