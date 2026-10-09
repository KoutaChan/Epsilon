package com.epsilon.cfr.model;

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

/** 一つの実行枠が専有する入力バッファ。デバイスからスコアを受け取り、入力順の予測を返す。 */
public final class CfrInferenceWorkspace implements AutoCloseable {
  private final CfrModel model;
  private final CfrNetwork network;
  private final EpsilonDecisionFrozenFeatures frozenFeatures;
  private final boolean stateFusions;
  private final NDManager storage;
  private final List<DecisionBatchTransfer.Workspace> transfers = new ArrayList<>();
  private final List<ExecutionSlot> executions = new ArrayList<>();
  private Batch active;
  private boolean closed;

  CfrInferenceWorkspace(
      CfrModel model,
      NDManager owner,
      CfrNetwork network,
      EpsilonDecisionFrozenFeatures frozenFeatures,
      boolean stateFusions) {
    this.model = model;
    this.network = network;
    this.frozenFeatures = frozenFeatures;
    this.stateFusions = stateFusions;
    storage = owner.newSubManager();
  }

  private DecisionInferenceExecution execution(int index, int rows) {
    if (frozenFeatures == null || !stateFusions) return null;
    while (executions.size() <= index) executions.add(new ExecutionSlot());
    ExecutionSlot slot = executions.get(index);
    if (slot.execution == null || slot.capacity < rows) {
      if (slot.execution != null) {
        slot.execution.close();
        slot.execution = null;
      }
      int capacity =
          Math.min(model.config().inferenceBatchSize(), Integer.highestOneBit(rows - 1) << 1);
      slot.capacity = Math.max(rows, capacity);
      slot.execution = frozenFeatures.newExecution(slot.capacity, 1);
    }
    return slot.execution;
  }

  private static final class ExecutionSlot {
    private DecisionInferenceExecution execution;
    private int capacity;
  }

  /** 実在する遷移だけを集め、Nanoと共通のページ固定バッファへ一度だけコピーする。 */
  public Batch stage(CfrModel.EncodedBatch encoded) {
    if (closed || active != null)
      throw new IllegalStateException("Inference workspace is in use or closed");
    double[][] policies = encoded.policies();
    List<Chunk> chunks = new ArrayList<>();
    List<Group> pending = new ArrayList<>();
    int pendingRows = 0;
    int limit = model.config().inferenceBatchSize();
    for (var input : encoded.groups) {
      if (input.forced) continue;
      for (int from = 0; from < input.host.size(); ) {
        int count = Math.min(input.host.size() - from, limit - pendingRows);
        pending.add(new Group(input, input.host.sliceRows(from, count)));
        from += count;
        pendingRows += count;
        if (pendingRows == limit) {
          chunks.add(stageChunk(chunks.size(), pending));
          pending = new ArrayList<>();
          pendingRows = 0;
        }
      }
    }
    if (!pending.isEmpty()) chunks.add(stageChunk(chunks.size(), pending));
    active = new Batch(policies, chunks);
    return active;
  }

  private Chunk stageChunk(int index, List<Group> groups) {
    if (index == transfers.size()) transfers.add(new DecisionBatchTransfer.Workspace(storage));
    var transfer = transfers.get(index);
    var staged =
        transfer.stageGroupedInference(
            groups.stream().map(group -> group.rows).toList(), model.config().tensorTransfer());
    return new Chunk(List.copyOf(groups), transfer, staged);
  }

  private static final class Group {
    final CfrModel.EncodedGroup input;
    final DecisionHostBatch.RowSlice rows;

    Group(CfrModel.EncodedGroup input, DecisionHostBatch.RowSlice rows) {
      this.input = input;
      this.rows = rows;
    }
  }

  private static final class Chunk {
    final List<Group> groups;
    final DecisionBatchTransfer.Workspace transfer;
    final DecisionBatchTransfer.StagedGroupedInference staged;
    DecisionGroupedInferenceBatch device;
    DecisionInferenceExecution execution;
    AutoCloseable dependency;
    boolean failedForward;
    int outputOffset;

    Chunk(
        List<Group> groups,
        DecisionBatchTransfer.Workspace transfer,
        DecisionBatchTransfer.StagedGroupedInference staged) {
      this.groups = groups;
      this.transfer = transfer;
      this.staged = staged;
    }
  }

  /**
   * 呼び出し側で {@link #upload()} と {@link #forward()} を各ストリーム上で実行し、完了イベント後に {@link
   * #complete(FloatBuffer)} と {@link #close()} を呼ぶ。
   */
  public final class Batch implements AutoCloseable {
    private final double[][] policies;
    private final List<Chunk> chunks;
    private NDManager manager;
    private int outputElements;
    private Phase phase = Phase.STAGED;

    private Batch(double[][] policies, List<Chunk> chunks) {
      this.policies = policies;
      this.chunks = chunks;
    }

    public void upload() {
      require(Phase.STAGED);
      manager = storage.newSubManager();
      DataType inputType =
          frozenFeatures == null ? DataType.FLOAT32 : frozenFeatures.inputNumericType();
      for (Chunk chunk : chunks)
        chunk.device = chunk.transfer.enqueueGroupedInference(manager, chunk.staged, inputType);
      phase = Phase.UPLOADED;
    }

    /** 合法行動のスコアを一つのFP32出力へまとめる。ホスト側で完了を待たない。 */
    public NDArray forward() {
      require(Phase.UPLOADED);
      NDList outputs = new NDList();
      try (var inference = manager.getEngine().newInferenceMode();
          Autocast ignored =
              EpsilonDecisionAutocast.open(manager, model.config().precision(model.device()))) {
        var parameters =
            frozenFeatures == null
                ? new ParameterStore(manager, false)
                : frozenFeatures.parameters(manager);
        for (int index = 0; index < chunks.size(); index++) {
          Chunk chunk = chunks.get(index);
          chunk.execution = execution(index, chunk.device.rowCount());
          DecisionInferenceExecution.Forward forward = null;
          NDArray scores;
          try {
            forward = chunk.execution == null ? null : chunk.execution.beginForward(manager);
            scores = network.inference(parameters, chunk.device, frozenFeatures, forward);
            if (forward != null) chunk.dependency = forward.seal();
          } catch (RuntimeException | Error failure) {
            if (chunk.execution != null) {
              chunk.failedForward = true;
              if (forward != null) forward.poison(failure);
            }
            throw failure;
          }
          NDArray packed = scores.toType(DataType.FLOAT32, false);
          chunk.outputOffset = outputElements;
          outputElements = Math.addExact(outputElements, Math.toIntExact(packed.size()));
          outputs.add(packed.reshape(-1));
        }
        NDArray result =
            outputs.isEmpty()
                ? null
                : outputs.size() == 1 ? outputs.getFirst() : NDArrays.concat(outputs);
        phase = Phase.FORWARDED;
        return result;
      }
    }

    public int outputElements() {
      require(Phase.FORWARDED);
      return outputElements;
    }

    /** 各行の方策を検証し、再利用される出力バッファから独立した配列へ移す。 */
    public CfrModel.Policies complete(FloatBuffer output) {
      require(Phase.FORWARDED);
      if (output.remaining() < outputElements)
        throw new IllegalArgumentException("Truncated inference output");
      for (Chunk chunk : chunks) {
        int offset = output.position() + chunk.outputOffset;
        for (Group group : chunk.groups) {
          for (int row = 0; row < group.rows.size(); row++) {
            int sourceRow = group.rows.fromInclusive() + row;
            int index = group.input.indices[sourceRow];
            int count = group.input.host.legalActionCount(sourceRow);
            float[] scores = copy(output, offset, count);
            policies[index] = CfrModel.policy(scores, count, model.role());
            offset += count;
          }
        }
      }
      phase = Phase.COMPLETED;
      return new CfrModel.Policies(policies);
    }

    private void require(Phase expected) {
      if (phase != expected)
        throw new IllegalStateException("Inference batch is " + phase + ", expected " + expected);
    }

    @Override
    public void close() {
      if (phase == Phase.CLOSED) return;
      Throwable failure = null;
      try {
        for (int index = chunks.size() - 1; index >= 0; index--) {
          Chunk chunk = chunks.get(index);
          try {
            if (chunk.dependency != null) chunk.dependency.close();
            if (chunk.failedForward) {
              chunk.execution.releaseFailedForwardAfterCompletion();
              chunk.execution.close();
              executions.get(index).execution = null;
            }
          } catch (Throwable problem) {
            if (failure == null) failure = problem;
            else failure.addSuppressed(problem);
          }
        }
        try {
          if (manager != null) manager.close();
        } catch (RuntimeException | Error problem) {
          if (failure == null) failure = problem;
          else failure.addSuppressed(problem);
        }
      } finally {
        phase = Phase.CLOSED;
        active = null;
      }
      rethrow(failure);
    }
  }

  private enum Phase {
    STAGED,
    UPLOADED,
    FORWARDED,
    COMPLETED,
    CLOSED
  }

  private static float[] copy(FloatBuffer source, int offset, int length) {
    float[] values = new float[length];
    source.get(offset, values);
    for (float value : values)
      if (!Float.isFinite(value)) throw new IllegalStateException("Nonfinite inference output");
    return values;
  }

  @Override
  public void close() {
    if (closed) return;
    if (active != null) throw new IllegalStateException("Inference batch has not drained");
    Throwable failure = null;
    for (ExecutionSlot slot : executions) {
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
