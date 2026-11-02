package com.epsilon.prism.policy;

import ai.djl.pytorch.engine.PtEvent;
import com.epsilon.prism.backend.HipReference;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.input.HostBatch;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * HIP の実行枠の道具の代わり（native なしの試験）。カーネルと同じ式・同じ丸めの CPU の参照（{@link HipReference}）で s を書き、版を計画と同じく
 * 写しの版と照らす。ストリームを持たないので、呼び出しスレッドの口（{@link HipInference#score}）だけで使う。
 */
final class ReferenceLane implements HipInference.Lane {

  private final HipReference reference;
  private final long version;

  /**
   * @param pack 写す重みの束
   */
  ReferenceLane(WeightPack pack) {
    this(pack, pack.version());
  }

  /**
   * 写しの版を指定する（古い写しの試験）。
   *
   * @param pack 写す重みの束
   * @param version 写しの版の番号
   */
  ReferenceLane(WeightPack pack, long version) {
    reference = new HipReference(pack);
    this.version = version;
  }

  @Override
  public void submit(HostBatch batch, ByteBuffer out, long expected) {
    if (expected != version) {
      throw new IllegalStateException(
          "weight pack version " + version + " is stale; the network is at version " + expected);
    }
    float[] scores = reference.forward(batch);
    ByteBuffer target = out.duplicate().order(ByteOrder.LITTLE_ENDIAN);
    for (int action = 0; action < scores.length; action++) {
      target.putFloat(action * Float.BYTES, scores[action]);
    }
  }

  @Override
  public void record(PtEvent event) {
    throw new UnsupportedOperationException("the CPU reference has no device stream");
  }

  @Override
  public void synchronize() {}

  @Override
  public void close() {}
}
