package com.epsilon.prism.policy;

import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.model.NetworkInputs;
import java.nio.LongBuffer;

/**
 * eager の 1 回の受け付けのデバイスの仕事: 塊ごとの入力の転送（{@link #upload}）と順伝播（{@link #forward}）。GPU では転送を H2D のストリーム
 * の中で、順伝播を計算のストリームの中で呼ぶ。
 *
 * <p>出力は塊の順に、塊の手の順の s（FP32）を詰めてつないだもの（HIP の推論の出力と同じ並び）。網の s [行, 手の容量] から合法手の枠だけを
 * デバイスの上で集める（枠の位置は入力と一緒に送る）。
 *
 * <p>入力の配列・枠の位置・出力は仕事の管理元に付けて、{@link #close} まで持つ（H2D のストリームで取った領域なので、計算が終わるまで捨てない。
 * CPU では入力の配列が塊の領域を指す）。塊ごとの順伝播の途中の配列は、その塊の順伝播を出した直後に捨てる。計算のストリームで取った領域は、同じ
 * ストリームの後の確保だけが使い回すので、計算が終わる前に捨ててよい。
 */
final class EagerPass implements AutoCloseable {

  private final EagerInference network;
  private final Encoding encoding;
  private final NDManager work;
  private final NDManager[] scopes;
  private final NetworkInputs[] inputs;
  private final NDArray[] slots;
  private final int[] widths;

  /**
   * @param network 網
   * @param encoding 符号化を終えた作業域
   */
  EagerPass(EagerInference network, Encoding encoding) {
    this.network = network;
    this.encoding = encoding;
    int chunks = encoding.chunks().size();
    work = network.newScope();
    scopes = new NDManager[chunks];
    inputs = new NetworkInputs[chunks];
    slots = new NDArray[chunks];
    widths = new int[chunks];
  }

  /** 塊ごとの入力と、合法手の枠の位置を網のデバイスに作る。 */
  void upload() {
    for (int chunk = 0; chunk < inputs.length; chunk++) {
      BatchPlan plan = encoding.chunks().get(chunk).batch().plan();
      scopes[chunk] = work.newSubManager();
      HostInputs batch = encoding.chunks().get(chunk).batch();
      inputs[chunk] = NetworkInputs.of(scopes[chunk], batch.input(), batch.frontier());
      int width = 0;
      for (int row = 0; row < plan.rows(); row++) {
        width = Math.max(width, plan.actions(row));
      }
      widths[chunk] = width;
      slots[chunk] = slots(plan, width);
    }
  }

  /**
   * 塊ごとに順伝播し、合法手の s を塊の順につなぐ。
   *
   * @return FP32 の 1 次元の配列（塊 c の行 j の手 k は、塊 c の先頭から firstAction(j) + k）
   * @throws IllegalStateException 網の手の容量が塊の行の手の数の最大と違うとき
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

  /** 塊の手ごとの、s [行, 幅] を平らにした中の位置（INT64）。 */
  private NDArray slots(BatchPlan plan, int width) {
    int actions = plan.count(Axis.ACTION);
    long[] positions = new long[actions];
    for (int row = 0; row < plan.rows(); row++) {
      int first = plan.firstAction(row);
      for (int action = 0; action < plan.actions(row); action++) {
        positions[first + action] = (long) row * width + action;
      }
    }
    return work.create(LongBuffer.wrap(positions), new Shape(actions));
  }

  @Override
  public void close() {
    work.close();
  }
}
