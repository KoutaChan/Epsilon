package com.epsilon.prism.policy;

import ai.djl.pytorch.engine.PtEvent;
import com.epsilon.prism.backend.HipReference;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.input.HostBatch;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * HIP のスロットのレーンの代わりに使う、native なしのテスト用のレーン。カーネルと同じ式・同じ丸めの CPU の参照（{@link HipReference}）で
 * s を書く。本物と同じく、写した重みのバージョンを要求されたバージョンと照合し、違えば例外を投げる。ストリームを持たないので、呼び出した
 * スレッドで評価する入口（{@link HipInference#score}）でだけ使う。
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
   * 写した重みのバージョンを指定して作る。古い重みを写したレーンのテストに使う。
   *
   * @param pack 写す重みの束
   * @param version 写した重みとして名乗るバージョン番号
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
