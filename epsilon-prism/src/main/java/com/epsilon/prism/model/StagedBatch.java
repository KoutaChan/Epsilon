package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.FrontierBatch;
import com.epsilon.prism.input.FrontierField;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.input.InputSchema.Tensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * 学習の微小バッチのデバイスへの入力を、ホストで前もって並べたもの: 網の入力の区画（{@link Tensor} の順と、前線の腕なら前線の区画）・索引
 * （{@link BatchIndex}）・損失の入力（{@link LossInputs}）。学習の歩の作り手が作業のスレッドで {@link UploadRegion} に書き
 * （{@link #stage}）、レーンが {@link #send} で型ごとに 1 回の非同期の写しを積む。レーンのホストは写しも索引の組み立ても待たない。
 *
 * <p>並び: 型ごとの 1 本の配列に、網の入力の区画（区画ごとに {@link Staging#ALIGNMENT} の境目から）、索引の組、損失の入力の組（組ごとに
 * 境目から、組の中は同期で送るときと同じ並び）を続ける。どの区画も、先頭の 512 の剰余と値が区画（網の入力）・組（索引と損失の入力）ごとに
 * 別の配列で送るときと同じなので、網と損失の演算の結果はビットで同じ。
 */
public final class StagedBatch {

  /**
   * 送った網の入力（前もって送った索引を持つ）と損失の入力。
   *
   * @param inputs 網の入力
   * @param loss 損失の入力
   */
  public record Sent(NetworkInputs inputs, LossInputs loss) {}

  /** 塊の数と索引のホストの写し。 */
  private record Columns(
      int rows, int events, int transitions, int actions, int[] rowColumns, int[] actionColumns)
      implements HostColumns {

    @Override
    public int row(int row, RowColumn column) {
      return rowColumns[row * NetworkInputs.ROW_WIDTH + column.ordinal()];
    }

    @Override
    public int action(int action, ActionColumn column) {
      return actionColumns[action * NetworkInputs.ACTION_WIDTH + column.ordinal()];
    }
  }

  private final UploadRegion region;
  private final Staging staging = new Staging();

  /** 網の入力の区画（要素が 0 個の区画は null で、送るときに空の配列を作る）。 */
  private final Map<Tensor, Staging.Section> tensors = new EnumMap<>(Tensor.class);

  private final Staging.Section frontier;
  private final Columns columns;
  private final BatchIndex index;
  private final LossInputs loss;

  private StagedBatch(
      UploadRegion region,
      HostBatch batch,
      FrontierBatch frontierBatch,
      boolean[] measured,
      AuxiliaryLabels auxiliary,
      List<ValueLabel> labels,
      AuxiliaryLoss.Denominators denominators) {
    this.region = region;
    BatchPlan plan = batch.plan();
    int rows = plan.rows();
    int events = plan.count(Axis.EVENT);
    int transitions = plan.count(Axis.TRANSITION);
    int actions = plan.count(Axis.ACTION);
    columns =
        new Columns(
            rows,
            events,
            transitions,
            actions,
            NetworkInputs.ints(section(batch, Tensor.ROW_COLUMN), rows * NetworkInputs.ROW_WIDTH),
            NetworkInputs.ints(
                section(batch, Tensor.ACTION_COLUMN), actions * NetworkInputs.ACTION_WIDTH));
    for (Tensor tensor : Tensor.values()) {
      ByteBuffer bytes = section(batch, tensor);
      staging.align();
      tensors.put(
          tensor,
          bytes.remaining() == 0
              ? null
              : staging.add(bytes, Staging.Kind.of(tensor.element()), shape(tensor, columns)));
    }
    if (frontierBatch == null) {
      frontier = null;
    } else {
      staging.align();
      frontier =
          staging.add(
              frontierBatch.blob().slice(0, frontierBatch.bytes()),
              Staging.Kind.of(FrontierField.ELEMENT),
              new Shape(transitions, FrontierField.WIDTH));
    }
    staging.align();
    index = new BatchIndex(columns, measured, auxiliary, staging);
    staging.align();
    loss = LossInputs.stage(columns, labels, auxiliary, denominators, staging);
    staging.write(region);
  }

  /**
   * Tokenizer が書いた塊（と前線の腕なら前線の書き手が書いた塊）と学習の教師から、網の入力・索引・損失の入力を並べて領域に書く（ホストの
   * スレッド。領域の前の写しの終わりを待ってから書く）。塊と教師は送り終えるまで変えない。
   *
   * @param region 領域（送り終えるまでこの微小バッチだけが使う）
   * @param batch 計画を封じた塊
   * @param frontier 前線の塊（前線のない入力は null）
   * @param measured 学習の C の印（塊の手の順。順伝播の {@link Pass#training} に渡すのと同じ配列）
   * @param auxiliary 補助の頭の教師（補助の頭を計算しないなら null。順伝播に渡すのと同じもの）
   * @param labels 行の順の教師（錨を混ぜた後）
   * @param denominators 補助の頭の平均の分母（教師があるときだけ）
   * @return 並べた微小バッチ
   * @throws IllegalArgumentException 索引が並びの約束を外れるか、教師が塊と合わないとき
   */
  public static StagedBatch stage(
      UploadRegion region,
      HostBatch batch,
      FrontierBatch frontier,
      boolean[] measured,
      AuxiliaryLabels auxiliary,
      List<ValueLabel> labels,
      AuxiliaryLoss.Denominators denominators) {
    return new StagedBatch(region, batch, frontier, measured, auxiliary, labels, denominators);
  }

  /**
   * 型ごとに 1 回の非同期の写しを呼び手のスレッドの今のストリームに積み、写しの後に領域の event を記録して、網の入力と損失の入力を返す
   * （ホストは写しを待たない）。送り直すときは領域の前の写しの終わりを待ってから書き直す。
   *
   * @param manager 配列を作る管理元
   * @param stream 呼び手のスレッドの今のストリーム（CPU の管理元なら null）
   * @return 網の入力と損失の入力
   */
  public Sent send(NDManager manager, PtStream stream) {
    if (!staging.written()) {
      staging.write(region);
    }
    staging.enqueue(manager, stream);
    Map<Tensor, NDArray> arrays = new EnumMap<>(Tensor.class);
    for (Tensor tensor : Tensor.values()) {
      Staging.Section section = tensors.get(tensor);
      arrays.put(
          tensor,
          section == null
              ? manager.create(shape(tensor, columns), Staging.Kind.of(tensor.element()).type())
              : section.array());
    }
    NetworkInputs inputs =
        new NetworkInputs(
            arrays,
            frontier == null ? null : frontier.array(),
            columns.rows(),
            columns.events(),
            columns.transitions(),
            columns.actions(),
            columns.rowColumns(),
            columns.actionColumns(),
            null,
            index);
    return new Sent(inputs, loss);
  }

  /**
   * 書いた領域を返す。
   *
   * @return 領域
   */
  public UploadRegion region() {
    return region;
  }

  private static Shape shape(Tensor tensor, Columns columns) {
    return new Shape(
        NetworkInputs.count(
            tensor.axis(),
            columns.rows(),
            columns.events(),
            columns.transitions(),
            columns.actions()),
        tensor.width());
  }

  /** 塊の blob の区画（転送の符号のまま）。 */
  private static ByteBuffer section(HostBatch batch, Tensor tensor) {
    BatchPlan plan = batch.plan();
    int bytes = plan.count(tensor.axis()) * tensor.width() * tensor.element().bytes();
    return batch.blob().slice(plan.offset(tensor), bytes).order(ByteOrder.LITTLE_ENDIAN);
  }
}
