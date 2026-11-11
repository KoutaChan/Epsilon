package com.epsilon.prism.policy;

import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.Staging;

/**
 * eager の 1 回の受付で行うデバイスの処理。バッチごとに入力を転送し（{@link #upload}）、順伝播する（{@link #forward}）。GPU では、転送を
 * H2D ストリームの中で、順伝播を計算ストリームの中で呼ぶ。
 *
 * <p>出力は合法手の s（FP32）を、バッチの順、バッチの中では手の順に詰めてつないだもの（HIP の推論の出力と同じ並び）。ネットワークの s
 * [行, 手の容量] から、合法手の位置の値だけをデバイスの上で集める（位置は入力と一緒に送る）。
 *
 * <p>入力の配列・合法手の位置・出力はジョブのマネージャーに付け、{@link #close} まで持つ。これらは H2D ストリームで確保した領域なので、
 * 計算が終わるまで解放しない（CPU では入力の配列がバッチの領域を指す）。バッチごとの順伝播の途中の配列は、そのバッチの順伝播を出した
 * 直後に解放する。計算ストリームで確保した領域は、同じストリームの後の確保だけが使い回すので、計算が終わる前に解放してよい。
 * インデックスと合法手の位置は、ネットワークから借りたバッファ（{@link PinnedUploads}）から送り、{@link #close} で返す。
 */
final class EagerPass implements AutoCloseable {

  private final EagerInference network;
  private final Encoding encoding;
  private final NDManager work;
  private final PinnedUploads uploads;
  private final NDManager[] scopes;
  private final NetworkInputs[] inputs;
  private final NDArray[] slots;
  private final int[] widths;

  /**
   * @param network eager の推論のバックエンド
   * @param encoding 符号化を終えた作業領域
   */
  EagerPass(EagerInference network, Encoding encoding) {
    this.network = network;
    this.encoding = encoding;
    int chunks = encoding.chunks().size();
    work = network.newScope();
    uploads = network.acquireUploads();
    scopes = new NDManager[chunks];
    inputs = new NetworkInputs[chunks];
    slots = new NDArray[chunks];
    widths = new int[chunks];
  }

  /** バッチごとの入力と合法手の位置を、ネットワークのデバイスに作る。 */
  void upload() {
    for (int chunk = 0; chunk < inputs.length; chunk++) {
      BatchPlan plan = encoding.chunks().get(chunk).batch().plan();
      scopes[chunk] = work.newSubManager();
      HostInputs batch = encoding.chunks().get(chunk).batch();
      inputs[chunk] = NetworkInputs.of(scopes[chunk], uploads, batch.input());
      int width = 0;
      for (int row = 0; row < plan.rows(); row++) {
        width = Math.max(width, plan.actions(row));
      }
      widths[chunk] = width;
      slots[chunk] = slots(plan, width);
    }
  }

  /**
   * バッチごとに順伝播し、合法手の s をバッチの順につなぐ。
   *
   * @return FP32 の 1 次元の配列。バッチ c の行 j の手 k は、バッチ c の部分の先頭から firstAction(j) + k 番目
   * @throws IllegalStateException ネットワークの手の容量が、バッチの行の手の数の最大と違うとき
   */
  NDArray forward() {
    NDList packed = new NDList(inputs.length);
    try (InferenceMode ignored = network.inference()) {
      for (int chunk = 0; chunk < inputs.length; chunk++) {
        NDArray scores = network.scores(inputs[chunk]).toType(DataType.FLOAT32, false);
        if (scores.getShape().get(1) != widths[chunk]) {
          throw new IllegalStateException(
              "scores " + scores.getShape() + " do not have " + widths[chunk] + " action columns");
        }
        NDArray values = NDArrays.gatherRows(scores.reshape(-1, 1), slots[chunk]).reshape(-1);
        values.attach(work);
        for (NDArray input : inputs[chunk].tensors().values()) {
          input.attach(work);
        }
        scopes[chunk].close();
        scopes[chunk] = null;
        packed.add(values);
      }
      return packed.size() == 1 ? packed.singletonOrThrow() : NDArrays.concat(packed);
    }
  }

  /** バッチの手ごとに、s [行, 幅] を平らにした配列の中の位置（INT64）を作る。 */
  private NDArray slots(BatchPlan plan, int width) {
    int actions = plan.count(Axis.ACTION);
    long[] positions = new long[actions];
    for (int row = 0; row < plan.rows(); row++) {
      int first = plan.firstAction(row);
      for (int action = 0; action < plan.actions(row); action++) {
        positions[first + action] = (long) row * width + action;
      }
    }
    Staging staging = new Staging();
    Staging.Section section = staging.add(positions, new Shape(actions));
    staging.send(uploads, work);
    return section.array();
  }

  @Override
  public void close() {
    try {
      work.close();
    } finally {
      network.releaseUploads(uploads);
    }
  }
}
