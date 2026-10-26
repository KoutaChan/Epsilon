package com.epsilon.prism.model;

import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.ParameterStore;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionGroupedInferenceBatch;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.nano.ai.model.DecisionInferenceExecution;
import com.epsilon.nano.ai.model.EpsilonDecisionFrozenFeatures;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * 1つの実行枠が専有する入力バッファ。入力は1回だけデバイスへ送り、同じデバイス上のネットに続けて通す。
 *
 * <p>使い方は旧 LCFR の推論ワークスペースと同じ: {@link #stage} → {@link Batch#upload()}（H2D ストリーム）→ {@link
 * Batch#forward()}（計算ストリーム）→ 完了イベントの後に {@link Batch#complete} と {@link Batch#close()}。
 */
public final class InferenceWorkspace implements AutoCloseable {
  private final Net[] nets;
  private final int batchLimit;
  private final com.epsilon.config.settings.DecisionTensorTransfer tensorTransfer;
  private final com.epsilon.config.settings.DecisionComputePrecision precision;
  private final NDManager storage;
  private final List<DecisionBatchTransfer.Workspace> transfers = new ArrayList<>();
  private boolean closed;

  /** @param models 同じデバイス上の凍結したネット。出力は {@link Scores#nets()} に同じ順で並ぶ */
  public InferenceWorkspace(List<PrismModel> models) {
    nets = new Net[models.size()];
    for (int k = 0; k < nets.length; k++) nets[k] = new Net(models.get(k));
    PrismModel first = models.getFirst();
    batchLimit = first.config().inferenceBatchSize();
    tensorTransfer = first.config().transfer(first.device());
    precision = first.config().precision(first.device());
    storage = first.manager().newSubManager();
  }

  private static final class Net {
    final PrismNetwork network;
    final EpsilonDecisionFrozenFeatures frozen;
    final List<ExecutionSlot> executions = new ArrayList<>();

    Net(PrismModel model) {
      network = model.network();
      frozen = model.frozenFeatures();
    }

    DecisionInferenceExecution execution(int index, int rows, int limit) {
      if (frozen == null) return null;
      while (executions.size() <= index) executions.add(new ExecutionSlot());
      ExecutionSlot slot = executions.get(index);
      if (slot.execution == null || slot.capacity < rows) {
        if (slot.execution != null) {
          slot.execution.close();
          slot.execution = null;
        }
        int capacity = Math.min(limit, Integer.highestOneBit(Math.max(1, rows - 1)) << 1);
        slot.capacity = Math.max(rows, capacity);
        slot.execution = frozen.newExecution(slot.capacity, 1);
      }
      return slot.execution;
    }
  }

  private static final class ExecutionSlot {
    DecisionInferenceExecution execution;
    int capacity;
  }

  /** 行を上限ごとの塊に分け、ページ固定バッファへ一度だけコピーする。 */
  public Batch stage(EncodedRows encoded) {
    List<Chunk> chunks = new ArrayList<>();
    List<Group> pending = new ArrayList<>();
    int pendingRows = 0;
    for (var input : encoded.groups()) {
      for (int from = 0; from < input.host().size(); ) {
        int count = Math.min(input.host().size() - from, batchLimit - pendingRows);
        pending.add(new Group(input, input.host().sliceRows(from, count)));
        from += count;
        pendingRows += count;
        if (pendingRows == batchLimit) {
          chunks.add(stageChunk(chunks.size(), pending));
          pending = new ArrayList<>();
          pendingRows = 0;
        }
      }
    }
    if (!pending.isEmpty()) chunks.add(stageChunk(chunks.size(), pending));
    return new Batch(encoded.size(), chunks);
  }

  private Chunk stageChunk(int index, List<Group> groups) {
    if (index == transfers.size()) transfers.add(new DecisionBatchTransfer.Workspace(storage));
    var transfer = transfers.get(index);
    var staged =
        transfer.stageGroupedInference(
            groups.stream().map(group -> group.rows).toList(), tensorTransfer);
    return new Chunk(groups, transfer, staged, nets.length);
  }

  private record Group(EncodedRows.Group input, DecisionHostBatch.RowSlice rows) {}

  private static final class Chunk {
    final List<Group> groups;
    final DecisionBatchTransfer.Workspace transfer;
    final DecisionBatchTransfer.StagedGroupedInference staged;
    final DecisionInferenceExecution[] executions;
    final AutoCloseable[] dependencies;
    final boolean[] failedForward;
    final int[] outputOffsets;

    DecisionGroupedInferenceBatch device;

    Chunk(
        List<Group> groups,
        DecisionBatchTransfer.Workspace transfer,
        DecisionBatchTransfer.StagedGroupedInference staged,
        int nets) {
      this.groups = groups;
      this.transfer = transfer;
      this.staged = staged;
      executions = new DecisionInferenceExecution[nets];
      dependencies = new AutoCloseable[nets];
      failedForward = new boolean[nets];
      outputOffsets = new int[nets];
    }
  }

  /** ネットのスコア。nets[k][i] はネット k の、要求 i の合法手ごとの値。 */
  public record Scores(float[][][] nets) {}

  public final class Batch implements AutoCloseable {
    private final int size;
    private final List<Chunk> chunks;
    private NDManager manager;
    private int outputElements;
    private boolean released;

    private Batch(int size, List<Chunk> chunks) {
      this.size = size;
      this.chunks = chunks;
    }

    public void upload() {
      manager = storage.newSubManager();
      DataType inputType =
          nets[0].frozen == null ? DataType.FLOAT32 : nets[0].frozen.inputNumericType();
      for (Chunk chunk : chunks)
        chunk.device = chunk.transfer.enqueueGroupedInference(manager, chunk.staged, inputType);
    }

    /** 全ネットの合法手スコアを一つの FP32 出力へまとめる。ホスト側で完了を待たない。 */
    public NDArray forward() {
      NDList outputs = new NDList();
      try (var inference = manager.getEngine().newInferenceMode();
          Autocast ignored = EpsilonDecisionAutocast.open(manager, precision)) {
        for (int net = 0; net < nets.length; net++) {
          Net target = nets[net];
          var parameters =
              target.frozen == null
                  ? new ParameterStore(manager, false)
                  : target.frozen.parameters(manager);
          for (int index = 0; index < chunks.size(); index++) {
            Chunk chunk = chunks.get(index);
            chunk.executions[net] = target.execution(index, chunk.device.rowCount(), batchLimit);
            DecisionInferenceExecution.Forward forward = null;
            NDArray scores;
            try {
              forward =
                  chunk.executions[net] == null
                      ? null
                      : chunk.executions[net].beginForward(manager);
              scores = target.network.inference(parameters, chunk.device, target.frozen, forward);
              if (forward != null) chunk.dependencies[net] = forward.seal();
            } catch (RuntimeException | Error failure) {
              if (chunk.executions[net] != null) {
                chunk.failedForward[net] = true;
                if (forward != null) forward.poison(failure);
              }
              throw failure;
            }
            NDArray packed = scores.toType(DataType.FLOAT32, false);
            chunk.outputOffsets[net] = outputElements;
            outputElements += Math.toIntExact(packed.size());
            outputs.add(packed.reshape(-1));
          }
        }
        NDArray result =
            outputs.isEmpty()
                ? null
                : outputs.size() == 1 ? outputs.getFirst() : NDArrays.concat(outputs);
        return result;
      }
    }

    public int outputElements() {
      return outputElements;
    }

    /** 各行のスコアを、再利用される出力バッファから独立した配列へ移す。 */
    public Scores complete(FloatBuffer output) {
      float[][][] result = new float[nets.length][size][];
      for (int net = 0; net < nets.length; net++) {
        for (Chunk chunk : chunks) {
          int offset = output.position() + chunk.outputOffsets[net];
          for (Group group : chunk.groups) {
            for (int row = 0; row < group.rows.size(); row++) {
              int sourceRow = group.rows.fromInclusive() + row;
              float[] values = new float[group.input.host().legalActionCount(sourceRow)];
              output.get(offset, values);
              result[net][group.input.indices()[sourceRow]] = values;
              offset += values.length;
            }
          }
        }
      }
      return new Scores(result);
    }

    @Override
    public void close() {
      if (released) return;
      released = true;
      Throwable failure = null;
      for (int index = chunks.size() - 1; index >= 0; index--) {
        Chunk chunk = chunks.get(index);
        for (int net = 0; net < nets.length; net++) {
          try {
            if (chunk.dependencies[net] != null) chunk.dependencies[net].close();
            if (chunk.failedForward[net]) {
              chunk.executions[net].releaseFailedForwardAfterCompletion();
              chunk.executions[net].close();
              nets[net].executions.get(index).execution = null;
            }
          } catch (Throwable problem) {
            failure = addFailure(failure, problem);
          }
        }
      }
      try {
        if (manager != null) manager.close();
      } catch (RuntimeException | Error problem) {
        failure = addFailure(failure, problem);
      }
      rethrow(failure);
    }
  }

  @Override
  public void close() {
    if (closed) return;
    Throwable failure = null;
    for (Net net : nets)
      for (ExecutionSlot slot : net.executions) {
        try {
          if (slot.execution != null) slot.execution.close();
          slot.execution = null;
        } catch (Throwable problem) {
          failure = addFailure(failure, problem);
        }
      }
    for (var transfer : transfers) {
      try {
        transfer.close();
      } catch (Throwable problem) {
        failure = addFailure(failure, problem);
      }
    }
    try {
      storage.close();
    } catch (Throwable problem) {
      failure = addFailure(failure, problem);
    }
    if (failure == null) closed = true;
    rethrow(failure);
  }

  private static Throwable addFailure(Throwable failure, Throwable additional) {
    if (failure == null) return additional;
    if (failure != additional) failure.addSuppressed(additional);
    return failure;
  }

  private static void rethrow(Throwable failure) {
    if (failure instanceof RuntimeException problem) throw problem;
    if (failure instanceof Error problem) throw problem;
    if (failure != null)
      throw new IllegalStateException("Cannot release inference execution", failure);
  }
}
