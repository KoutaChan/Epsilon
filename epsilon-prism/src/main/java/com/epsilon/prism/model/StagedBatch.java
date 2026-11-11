package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * 学習のマイクロバッチでデバイスに送る入力を、ホストで前もって並べたもの。中身は、ネットワークの入力の領域（{@link Tensor} の順）、
 * インデックス（{@link BatchIndex}）、損失の入力（{@link LossInputs}）。
 * 学習のステップを組み立てる側が、作業スレッドでページ固定のバッファ（{@link PinnedUploads}）に書く（{@link #stage}）。レーンは
 * {@link #send} で、要素の型ごとに 1 回の非同期の写しを写し用のストリームに積む。計算の直前には {@link Upload#receive} で、写しの
 * 終わりを待つ処理を計算ストリームに積む。レーンのホスト側は、写しもインデックスの組み立ても待たない。
 *
 * <p>並び: 要素の型ごとに 1 本の配列へ、ネットワークの入力の領域、インデックスの組、損失の入力の組を続けて置く。領域と組はそれぞれ
 * {@link Staging#ALIGNMENT} の境界から始め、組の中は同期で送るときと同じ並びにする。どの領域でも、先頭のアドレスを 512 で割った余りと
 * 値が、領域（ネットワークの入力）や組（インデックスと損失の入力）ごとに別の配列で送るときと同じになる。そのため、ネットワークと損失の
 * 演算の結果はビット単位で一致する。
 */
public final class StagedBatch {

  /**
   * 送ったネットワークの入力（前もって送ったインデックスを持つ）と損失の入力。
   *
   * @param inputs ネットワークの入力
   * @param loss 損失の入力
   */
  public record Sent(NetworkInputs inputs, LossInputs loss) {}

  /** 積んだ写し（{@link #send}）。送った配列は、{@link #receive} を呼んだストリームの上で、呼んだ後にだけ使える。 */
  public static final class Upload {

    private final PinnedUploads.Transfer transfer;
    private final Sent sent;

    private Upload(PinnedUploads.Transfer transfer, Sent sent) {
      this.transfer = transfer;
      this.sent = sent;
    }

    /**
     * 呼び出し元のスレッドの今のストリームに、写しの終わりを待つ処理を積む（ホストは待たない）。写し先の配列をそのストリームで使うものとして
     * 記録し、ネットワークの入力と損失の入力を返す。CPU では写しが同期で終わっているので、何も積まない。
     *
     * @return ネットワークの入力と損失の入力
     */
    public Sent receive() {
      transfer.waitOnStream();
      return sent;
    }
  }

  /** バッチの行・イベント・遷移・手の数と、インデックスの列のホスト側の写し。 */
  private record Columns(
      InputFormat format,
      int rows,
      int events,
      int transitions,
      int actions,
      int[] rowColumns,
      int[] actionColumns)
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

  private final PinnedUploads region;
  private final Staging staging = new Staging();

  /** ネットワークの入力の領域（要素が 0 個の領域は null で、送るときに空の配列を作る）。 */
  private final Map<Tensor, Staging.Section> tensors = new EnumMap<>(Tensor.class);

  private final Columns columns;
  private final BatchIndex index;
  private final LossInputs loss;

  private StagedBatch(
      PinnedUploads region,
      HostBatch batch,
      boolean[] measured,
      AuxiliaryLabels auxiliary,
      List<ValueLabel> labels,
      AuxiliaryLoss.Denominators denominators) {
    this.region = region;
    BatchPlan plan = batch.plan();
    InputFormat format = plan.format();
    int rows = plan.rows();
    int events = plan.count(Axis.EVENT);
    int transitions = plan.count(Axis.TRANSITION);
    int actions = plan.count(Axis.ACTION);
    columns =
        new Columns(
            format,
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
    staging.align();
    index = new BatchIndex(columns, measured, auxiliary, staging);
    staging.align();
    loss = LossInputs.stage(columns, labels, auxiliary, denominators, staging);
    staging.write(region);
  }

  /**
   * 入力のライターが書いたバッチと学習の教師から、ネットワークの入力・インデックス・損失の入力を並べてバッファに書く。ホストのスレッドで
   * 呼び、バッファの前の写しが終わるのを待ってから書く。バッチと教師は、送り終えるまで変えてはならない。
   *
   * @param region バッファ（送り終えるまで、このマイクロバッチだけが使う）
   * @param batch 計画を確定したバッチ
   * @param measured ラベルの行で値を測った手（C）の印（バッチの手の順。順伝播の {@link Pass#training} に渡すのと同じ配列）
   * @param auxiliary 補助のヘッドの教師（補助のヘッドを計算しないなら null。順伝播に渡すのと同じもの）
   * @param labels 行の順に並べた教師（アンカーを混ぜた後）
   * @param denominators 補助のヘッドの平均の分母（教師があるときだけ）
   * @return 並べたマイクロバッチ
   * @throws IllegalArgumentException インデックスが並びの契約を外れるとき、または教師がバッチと合わないとき
   */
  public static StagedBatch stage(
      PinnedUploads region,
      HostBatch batch,
      boolean[] measured,
      AuxiliaryLabels auxiliary,
      List<ValueLabel> labels,
      AuxiliaryLoss.Denominators denominators) {
    return new StagedBatch(region, batch, measured, auxiliary, labels, denominators);
  }

  /**
   * 要素の型ごとに 1 回の非同期の写しをストリームに積み、写しのチケットをバッファに預けて、積んだ写しを返す（ホストは写しを待たない）。
   * 送った配列を計算に使うストリームは、使う前に {@link Upload#receive} を呼ぶ。送り直すときは、バッファの前の写しが終わるのを待ってから
   * 書き直す。
   *
   * @param manager 写し先の配列を作るマネージャー（計算を終えるまで閉じない）
   * @param stream 写しを積むストリーム（CPU のマネージャーなら null）
   * @return 積んだ写し
   */
  public Upload send(NDManager manager, PtStream stream) {
    if (!staging.written()) {
      staging.write(region);
    }
    PinnedUploads.Transfer transfer = staging.enqueue(manager, stream);
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
            columns.format(),
            arrays,
            columns.rows(),
            columns.events(),
            columns.transitions(),
            columns.actions(),
            columns.rowColumns(),
            columns.actionColumns(),
            null,
            index);
    return new Upload(transfer, new Sent(inputs, loss));
  }

  /**
   * 書いたバッファを返す。
   *
   * @return バッファ
   */
  public PinnedUploads region() {
    return region;
  }

  private static Shape shape(Tensor tensor, Columns columns) {
    return new Shape(
        NetworkInputs.count(
            columns.format(),
            tensor.axis(),
            columns.rows(),
            columns.events(),
            columns.transitions(),
            columns.actions()),
        columns.format().width(tensor));
  }

  /** バッチの blob の中の、1 つのテンソルの領域（転送用のエンコードのまま）。 */
  private static ByteBuffer section(HostBatch batch, Tensor tensor) {
    BatchPlan plan = batch.plan();
    int bytes = plan.count(tensor.axis()) * plan.format().width(tensor) * tensor.element().bytes();
    return batch.blob().slice(plan.offset(tensor), bytes).order(ByteOrder.LITTLE_ENDIAN);
  }
}
