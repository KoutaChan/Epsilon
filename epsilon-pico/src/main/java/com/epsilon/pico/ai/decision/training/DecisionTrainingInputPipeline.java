package com.epsilon.pico.ai.decision.training;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtTransferBatch;
import ai.djl.pytorch.engine.PtTransferTicket;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.DecisionTrainingDeviceTransferSchedule;
import com.epsilon.core.GameState;
import com.epsilon.pico.ai.decision.EpsilonUtilityTargets;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionDataException;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionTrajectoryPayloadStore;
import com.epsilon.pico.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.pico.ai.decision.input.DecisionInputLayout;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import com.epsilon.pico.ai.decision.input.DecisionTrainingSlabWriter;
import com.epsilon.pico.ai.decision.input.DecisionTrainingTargetLayout;
import com.epsilon.pico.ai.decision.training.DecisionTrainingMetrics.AdvantageSigns;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 1つの学習ワーカーに属し、入力の復号とホストからデバイスへの転送を進める。
 *
 * <p>保持する容量は小バッチ2個分に固定する。専用のワーカーが外部ファイルを読み、最終的なホスト側バッファへ直接復号する。ON_DEMAND
 * では学習ワーカーが取得した時点で転送し、OVERLAPPED では復号後すぐに専用ストリームで転送を開始する。
 *
 * <p>ページロックしたホスト側バッファは転送完了時に回収する。デバイス側バッチは別のリソース管理オブジェクトで保持し、計算中もホスト側バッファを次の復号へ再利用できる。
 */
final class DecisionTrainingInputPipeline implements AutoCloseable {

  /** 現在のバッチとlookahead バッチを同時に保持する固定容量。 */
  static final int CAPACITY = 2;

  private static final AtomicInteger WORKER_IDS = new AtomicInteger();

  private final com.epsilon.ai.decision.EpsilonUtilityProfile utilityProfile;
  private final int laneIndex;
  private final NDManager owner;
  private final Device device;
  private final DecisionTensorTransfer transferMode;
  private final DecisionTrainingDeviceTransferSchedule schedule;
  private final PtNDManager pytorchOwner;
  private final PtStream transferStream;
  private final HostSlot[] slots = new HostSlot[CAPACITY];
  private final Semaphore submissions = new Semaphore(CAPACITY);
  private final ThreadPoolExecutor worker;
  private final ArrayDeque<PendingTransfer> retirements = new ArrayDeque<>();
  private final Set<DecisionTrainingReadyBatch> readyBatches =
      Collections.newSetFromMap(new IdentityHashMap<>());

  private EpsilonDecisionTrajectoryPayloadStore.ExternalFileSession payloadSession;
  private Throwable failure;
  private int nextSlot;
  private int activeDeviceLeases;
  private boolean accepting = true;
  private boolean closed;

  /**
   * 学習ワーカー-局所的なパイプラインを作る。
   *
   * @param laneIndex データ並列学習ワーカーの固定インデックス
   * @param owner 学習ワーカーのモデルデバイスを所有する管理元
   * @param transferMode ホストバッファの媒体
   * @param schedule デバイス転送を開始するタイミング
   */
  DecisionTrainingInputPipeline(
      int laneIndex,
      NDManager owner,
      DecisionTensorTransfer transferMode,
      DecisionTrainingDeviceTransferSchedule schedule,
      DecisionExecutionContext executionContext) {
    this(
        laneIndex,
        owner,
        transferMode,
        schedule,
        executionContext,
        EpsilonUtilityTargets.configuredTrainingProfile());
  }

  DecisionTrainingInputPipeline(
      int laneIndex,
      NDManager owner,
      DecisionTensorTransfer transferMode,
      DecisionTrainingDeviceTransferSchedule schedule,
      DecisionExecutionContext executionContext,
      com.epsilon.ai.decision.EpsilonUtilityProfile utilityProfile) {
    this.utilityProfile = utilityProfile;
    if (laneIndex < 0) {
      throw new IllegalArgumentException("laneIndex must be non-negative: " + laneIndex);
    }
    this.laneIndex = laneIndex;
    this.owner = Objects.requireNonNull(owner, "owner");
    device = owner.getDevice();
    this.transferMode = Objects.requireNonNull(transferMode, "transferMode");
    this.schedule = Objects.requireNonNull(schedule, "schedule");
    pytorchOwner = owner instanceof PtNDManager manager ? manager : null;

    if (transferMode == DecisionTensorTransfer.PINNED_BUFFER && pytorchOwner == null) {
      throw new IllegalArgumentException("PINNED_BUFFER requires a PyTorch manager");
    }
    if (schedule == DecisionTrainingDeviceTransferSchedule.OVERLAPPED
        && (transferMode != DecisionTensorTransfer.PINNED_BUFFER || !device.isGpu())) {
      throw new IllegalArgumentException("OVERLAPPED requires PINNED_BUFFER on a GPU lane");
    }
    transferStream =
        transferMode == DecisionTensorTransfer.PINNED_BUFFER
            ? executionContext.streams(device).h2d()
            : null;
    for (int index = 0; index < slots.length; index++) {
      slots[index] = new HostSlot(index);
    }
    worker =
        new ThreadPoolExecutor(
            1,
            1,
            0L,
            TimeUnit.MILLISECONDS,
            // データタスクはSemaphoreで二件に制限し、残り一枠はキュー末尾のセッション-解放用に予約する。
            new ArrayBlockingQueue<>(CAPACITY + 1),
            task -> {
              Thread thread =
                  new Thread(
                      task,
                      "decision-training-input-lane-"
                          + laneIndex
                          + "-"
                          + WORKER_IDS.incrementAndGet());
              thread.setDaemon(true);
              return thread;
            },
            new ThreadPoolExecutor.AbortPolicy());
  }

  /** 現在新たに投入できる小バッチ数を返す。 */
  int availableSubmissionSlots() {
    return submissions.availablePermits();
  }

  /**
   * H2D完了済み処理の識別情報を回収し、現在復号へ再利用できるホスト枠数を返す。
   *
   * <p>デバイス利用権の寿命とは独立したホスト retirementをテストと診断用の集計から確認するための学習ワーカー-局所的な値である。
   */
  synchronized int availableHostSlots() {
    retireCompleted();
    int available = 0;
    for (HostSlot slot : slots) {
      if (slot.state == SlotState.FREE) {
        available++;
      }
    }
    return available;
  }

  /** 現在順伝播/逆伝播側が所有するデバイス側バッチ数を返す。 */
  synchronized int activeDeviceLeases() {
    return activeDeviceLeases;
  }

  /**
   * サンプルの参照情報小バッチを復号キューへ一度だけ投入する。
   *
   * <p>二件が未完了（キュー、準備済み、デバイス利用権のいずれか）なら三件目を受理せず不整合を検出して直ちに例外を送出する。これにより実行処理キューやホスト・デバイス
   * メモリが学習速度とは無関係に増えない。
   *
   * @param plan コピーせず所有権を借りる作業計画
   * @return 復号と任意の先行H2Dを経て準備済みバッチで完了するfuture
   */
  CompletableFuture<DecisionTrainingReadyBatch> submit(DecisionTrainingBatchPlan plan) {
    Objects.requireNonNull(plan, "plan");
    CompletableFuture<DecisionTrainingReadyBatch> future = new CompletableFuture<>();
    synchronized (this) {
      requireAccepting();
      if (!submissions.tryAcquire()) {
        throw new IllegalStateException("training input lane already has two outstanding batches");
      }
      try {
        worker.execute(() -> prepare(plan, future));
      } catch (RejectedExecutionException rejected) {
        submissions.release();
        throw new IllegalStateException("training input pipeline is not accepting work", rejected);
      }
    }
    return future;
  }

  /**
   * 準備済みバッチを現在の演算ストリームへ一度だけ取得する。
   *
   * @param ready このパイプラインが返した準備済みバッチ
   * @return FLOAT32入力を持つデバイス側バッチ利用権
   */
  DecisionTrainingDeviceBatchLease acquire(DecisionTrainingReadyBatch ready) {
    return acquire(ready, DataType.FLOAT32);
  }

  /**
   * 準備済みバッチを現在の演算ストリームへ一度だけ取得する。
   *
   * <p>OVERLAPPEDではtransfer イベントのGPU-side 待ちだけを現在のストリームへキューへの追加する。ホストスレッドは複製完了を待たない。
   *
   * @param ready このパイプラインが返した準備済みバッチ
   * @param inputNumericType モデルへ渡す数値入力型
   * @return デバイス管理元と論理的なバッチの唯一の利用権
   */
  DecisionTrainingDeviceBatchLease acquire(
      DecisionTrainingReadyBatch ready, DataType inputNumericType) {
    Objects.requireNonNull(ready, "ready");
    Objects.requireNonNull(inputNumericType, "inputNumericType");
    DecisionTrainingReadyBatch.Payload payload = takeReady(ready);
    try {
      PendingTransfer transfer = payload.pendingTransfer();
      if (transfer == null) {
        DecisionTrainingSealedHostBatch hostBatch = payload.hostBatch();
        transfer =
            transferMode == DecisionTensorTransfer.PINNED_BUFFER ? enqueuePinned(hostBatch) : null;
        if (transfer == null) {
          return acquireDirect(hostBatch, inputNumericType);
        }
      }
      return handoff(transfer, ready.summary(), inputNumericType);
    } catch (RuntimeException | Error exception) {
      submissions.release();
      markFailed(exception);
      throw exception;
    }
  }

  /**
   * ワーカーキューを処理の回収した後、同じワーカー上でキャッシュした入力と確率分布の本体入出力経路を閉じる。
   *
   * <p>学習サイクル境界で呼ぶと、削除済み入力と確率分布の本体ファイルのinodeを次の学習サイクルまで保持しない。
   */
  void finishInputPass() throws IOException {
    synchronized (this) {
      if (closed) {
        return;
      }
    }
    CompletableFuture<Void> completion = new CompletableFuture<>();
    try {
      worker.execute(
          () -> {
            try {
              closePayloadSession();
              completion.complete(null);
            } catch (Throwable exception) {
              completion.completeExceptionally(exception);
            }
          });
    } catch (RejectedExecutionException rejected) {
      throw new IOException("Decision training input worker is closed", rejected);
    }
    await(completion, "finish Decision training input pass");
  }

  /** 未取得準備済みバッチを破棄する。通常は{@link DecisionTrainingReadyBatch#close()}から呼ぶ。 */
  void discard(DecisionTrainingReadyBatch ready) {
    DecisionTrainingReadyBatch.Payload payload;
    synchronized (this) {
      if (!readyBatches.remove(ready)) {
        return;
      }
      payload = ready.claim(this);
    }
    try {
      if (payload.hostBatch() != null) {
        payload.hostBatch().close();
      } else {
        retire(payload.pendingTransfer(), true);
      }
    } finally {
      submissions.release();
    }
  }

  /** ホスト利用権を書き込みが完了した状態へ遷移させる。 */
  synchronized void sealHostSlot(
      HostSlot slot, long generation, int playerMemoryCount, int transitionCount) {
    requireSlot(slot, generation, SlotState.FILLING);
    slot.playerMemoryCount = playerMemoryCount;
    slot.transitionCount = transitionCount;
    slot.state = SlotState.SEALED;
  }

  /** 未完了ホスト利用権をFREEへ戻す。 */
  synchronized void abortHostSlot(HostSlot slot, long generation) {
    requireSlot(slot, generation, SlotState.FILLING);
    slot.release();
  }

  /** 書き込みが完了したバッチの学習ワーカー・世代を確認する。 */
  synchronized void requireSealedHostSlot(HostSlot slot, long generation) {
    requireSlot(slot, generation, SlotState.SEALED);
  }

  /** 未転送書き込みが完了したバッチをFREEへ戻す。 */
  synchronized void releaseSealedHostSlot(HostSlot slot, long generation) {
    requireSlot(slot, generation, SlotState.SEALED);
    slot.release();
  }

  /** デバイス利用権終了を記録する。 */
  synchronized void releaseDeviceLease() {
    activeDeviceLeases--;
    submissions.release();
  }

  /**
   * ワーカー、準備済みバッチ、H2D 処理の識別情報、ホスト記憶領域の順で学習ワーカー資源を閉じる。
   *
   * <p>デバイス利用権は独立した専用のリソース管理オブジェクトを所有するため、このパイプラインより後に閉じてもよい。
   */
  @Override
  public void close() {
    synchronized (this) {
      if (closed) {
        return;
      }
      accepting = false;
    }

    Throwable closeFailure = null;
    try {
      finishInputPass();
    } catch (Throwable exception) {
      closeFailure = exception;
    }
    worker.shutdown();
    try {
      worker.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      closeFailure = addSuppressed(closeFailure, interrupted);
    }

    DecisionTrainingReadyBatch[] outstanding;
    synchronized (this) {
      outstanding = readyBatches.toArray(DecisionTrainingReadyBatch[]::new);
    }
    for (DecisionTrainingReadyBatch ready : outstanding) {
      try {
        ready.close();
      } catch (Throwable exception) {
        closeFailure = addSuppressed(closeFailure, exception);
      }
    }

    try {
      drainRetirements();
    } catch (Throwable exception) {
      closeFailure = addSuppressed(closeFailure, exception);
    }
    for (HostSlot slot : slots) {
      try {
        slot.close();
      } catch (Throwable exception) {
        closeFailure = addSuppressed(closeFailure, exception);
      }
    }
    synchronized (this) {
      closed = true;
    }
    rethrow(closeFailure);
  }

  private void prepare(
      DecisionTrainingBatchPlan plan, CompletableFuture<DecisionTrainingReadyBatch> completion) {
    DecisionTrainingHostBatchLease hostLease = null;
    DecisionTrainingSealedHostBatch sealed = null;
    try {
      synchronized (this) {
        if (failure != null) {
          throw new IllegalStateException("training input pipeline is terminal", failure);
        }
      }
      DecisionBucket bucket = plan.samples().getFirst().bucket();
      hostLease = acquireHostBatch(plan.rows(), bucket);
      DecisionTrainingSlabWriter writer = hostLease.writer();
      SummaryBuilder summary = new SummaryBuilder(bucket);
      for (int row = 0; row < plan.rows(); row++) {
        EpsilonDecisionTrainingSampleDescriptor sample = plan.samples().get(row);
        float actorWeight = actorWeight(sample, plan.policySignalMultipliers(), bucket);
        payloadSession().readTrainingRow(sample, writer, row, actorWeight, 1.0f);
        summary.add(sample, actorWeight);
      }
      DecisionTrainingBatchSummary batchSummary = summary.finish(plan.rows());
      sealed = hostLease.seal(batchSummary);
      hostLease = null;
      PendingTransfer transfer =
          schedule == DecisionTrainingDeviceTransferSchedule.OVERLAPPED
              ? enqueuePinned(sealed)
              : null;
      if (transfer != null) {
        sealed = null;
      }
      DecisionTrainingReadyBatch ready =
          new DecisionTrainingReadyBatch(this, batchSummary, sealed, transfer);
      sealed = null;
      synchronized (this) {
        readyBatches.add(ready);
      }
      if (!completion.complete(ready)) {
        ready.close();
      }
    } catch (Throwable exception) {
      if (hostLease != null) {
        closeSuppressing(hostLease, exception);
      }
      if (sealed != null) {
        closeSuppressing(sealed, exception);
      }
      submissions.release();
      markFailed(exception);
      completion.completeExceptionally(exception);
    }
  }

  private DecisionTrainingHostBatchLease acquireHostBatch(int rows, DecisionBucket bucket) {
    while (true) {
      PendingTransfer oldest = null;
      synchronized (this) {
        retireCompleted();
        for (int offset = 0; offset < slots.length; offset++) {
          int index = Math.floorMod(nextSlot + offset, slots.length);
          HostSlot slot = slots[index];
          if (slot.state == SlotState.FREE) {
            nextSlot = Math.floorMod(index + 1, slots.length);
            return slot.begin(rows, bucket);
          }
        }
        if (!retirements.isEmpty()) {
          oldest = retirements.removeFirst();
        }
      }
      if (oldest == null) {
        throw new IllegalStateException(
            "two host batches are still waiting for device acquisition");
      }
      finishRetirement(oldest, true);
    }
  }

  private PendingTransfer enqueuePinned(DecisionTrainingSealedHostBatch sealed) {
    HostSlot slot = sealed.claim(this);
    long generation = sealed.generation();
    PtNDManager destination = null;
    try {
      synchronized (this) {
        requireSlot(slot, generation, SlotState.SEALED);
        slot.state = SlotState.IN_FLIGHT;
      }
      NDManager candidate = owner.newSubManager();
      if (!(candidate instanceof PtNDManager pytorchDestination)) {
        candidate.close();
        throw new IllegalStateException("PyTorch lane returned a non-PyTorch sub-manager");
      }
      destination = pytorchDestination;
      PendingTransfer transfer;
      synchronized (transferStream) {
        try (PtTransferBatch batch = transferStream.newTransferBatch(destination)) {
          NDArray inputCategories =
              slot.inputCategories.copyPinned(batch, slot.inputCategoryElements);
          NDArray inputNumerics = slot.inputNumerics.copyPinned(batch, slot.inputNumericElements);
          NDArray playerMemoryIndices =
              copyPinnedOrEmpty(
                  batch, destination, slot.playerMemoryIndices, slot.playerMemoryCount);
          NDArray transitionIndices =
              copyPinnedOrEmpty(batch, destination, slot.transitionIndices, slot.transitionCount);
          NDArray targetCategories =
              slot.targetCategories.copyPinned(batch, slot.targetCategoryElements);
          NDArray targetNumerics =
              slot.targetNumerics.copyPinned(batch, slot.targetNumericElements);
          PtTransferTicket ticket = batch.commit();
          transfer =
              new PendingTransfer(
                  slot,
                  destination,
                  ticket,
                  inputCategories,
                  inputNumerics,
                  playerMemoryIndices,
                  transitionIndices,
                  targetCategories,
                  targetNumerics);
        }
      }
      destination = null;
      return transfer;
    } catch (RuntimeException | Error exception) {
      if (destination != null) {
        closeSuppressing(destination, exception);
      }
      synchronized (this) {
        if (slot.generation == generation && slot.state == SlotState.IN_FLIGHT) {
          slot.state = SlotState.BROKEN;
        }
      }
      throw exception;
    }
  }

  private DecisionTrainingDeviceBatchLease acquireDirect(
      DecisionTrainingSealedHostBatch sealed, DataType inputNumericType) {
    HostSlot slot = sealed.claim(this);
    long generation = sealed.generation();
    NDManager destination = owner.newSubManager();
    try {
      synchronized (this) {
        requireSlot(slot, generation, SlotState.SEALED);
        slot.state = SlotState.IN_FLIGHT;
      }
      NDArray inputCategories =
          slot.inputCategories.copyDirect(destination, slot.inputCategoryElements);
      NDArray inputNumerics = slot.inputNumerics.copyDirect(destination, slot.inputNumericElements);
      NDArray playerMemoryIndices =
          slot.playerMemoryIndices.copyDirectOrEmpty(destination, slot.playerMemoryCount);
      NDArray transitionIndices =
          slot.transitionIndices.copyDirectOrEmpty(destination, slot.transitionCount);
      NDArray targetCategories =
          slot.targetCategories.copyDirect(destination, slot.targetCategoryElements);
      NDArray targetNumerics =
          slot.targetNumerics.copyDirect(destination, slot.targetNumericElements);
      DecisionDeviceBatch batch =
          DecisionBatchTransfer.bindTrainingSlabs(
              slot.rows,
              slot.bucket,
              inputCategories,
              inputNumerics,
              playerMemoryIndices,
              transitionIndices,
              targetCategories,
              targetNumerics,
              true,
              inputNumericType);
      synchronized (this) {
        slot.release();
        activeDeviceLeases++;
      }
      DecisionTrainingDeviceBatchLease lease =
          new DecisionTrainingDeviceBatchLease(this, destination, batch, sealed.summary());
      destination = null;
      return lease;
    } finally {
      if (destination != null) {
        destination.close();
        synchronized (this) {
          if (slot.generation == generation && slot.state == SlotState.IN_FLIGHT) {
            slot.release();
          }
        }
      }
    }
  }

  private DecisionTrainingDeviceBatchLease handoff(
      PendingTransfer transfer, DecisionTrainingBatchSummary summary, DataType inputNumericType) {
    if (!device.equals(transfer.ticket.getDevice())) {
      throw new IllegalArgumentException("transfer ticket belongs to another device");
    }
    try {
      transfer.ticket.waitOnStream();
      DecisionDeviceBatch batch =
          DecisionBatchTransfer.bindTrainingSlabs(
              transfer.slot.rows,
              transfer.slot.bucket,
              transfer.inputCategories,
              transfer.inputNumerics,
              transfer.playerMemoryIndices,
              transfer.transitionIndices,
              transfer.targetCategories,
              transfer.targetNumerics,
              true,
              inputNumericType);
      NDManager destination = transfer.takeDestination();
      synchronized (this) {
        activeDeviceLeases++;
      }
      DecisionTrainingDeviceBatchLease lease =
          new DecisionTrainingDeviceBatchLease(this, destination, batch, summary);
      retire(transfer, false);
      return lease;
    } catch (RuntimeException | Error exception) {
      retire(transfer, true);
      throw exception;
    }
  }

  private DecisionTrainingReadyBatch.Payload takeReady(DecisionTrainingReadyBatch ready) {
    synchronized (this) {
      if (!readyBatches.remove(ready)) {
        throw new IllegalStateException("ready batch is not owned by this training lane");
      }
      DecisionTrainingReadyBatch.Payload payload = ready.claim(this);
      return payload;
    }
  }

  private void retire(PendingTransfer transfer, boolean closeDestination) {
    synchronized (this) {
      transfer.closeDestination = closeDestination;
      retirements.addLast(transfer);
    }
  }

  private void retireCompleted() {
    while (!retirements.isEmpty()) {
      PendingTransfer transfer = retirements.getFirst();
      if (!transfer.ticket.isComplete()) {
        return;
      }
      retirements.removeFirst();
      finishRetirement(transfer, false);
    }
  }

  private void drainRetirements() {
    while (true) {
      PendingTransfer transfer;
      synchronized (this) {
        transfer = retirements.pollFirst();
      }
      if (transfer == null) {
        return;
      }
      finishRetirement(transfer, true);
    }
  }

  private void finishRetirement(PendingTransfer transfer, boolean synchronize) {
    Throwable retirementFailure = null;
    try {
      if (synchronize) {
        transfer.ticket.synchronize();
      }
      transfer.ticket.close();
    } catch (Throwable exception) {
      retirementFailure = exception;
    }
    NDManager destination = transfer.destination;
    if (transfer.closeDestination && destination != null) {
      try {
        destination.close();
      } catch (Throwable exception) {
        retirementFailure = addSuppressed(retirementFailure, exception);
      }
      transfer.destination = null;
    }
    synchronized (this) {
      if (retirementFailure == null) {
        transfer.slot.release();
      } else {
        transfer.slot.state = SlotState.BROKEN;
      }
    }
    rethrow(retirementFailure);
  }

  private EpsilonDecisionTrajectoryPayloadStore.ExternalFileSession payloadSession() {
    if (payloadSession == null) {
      payloadSession = EpsilonDecisionTrajectoryPayloadStore.openExternalFileSession();
    }
    return payloadSession;
  }

  private void closePayloadSession() throws IOException {
    if (payloadSession != null) {
      payloadSession.close();
      payloadSession = null;
    }
  }

  private float actorWeight(
      EpsilonDecisionTrainingSampleDescriptor sample,
      DecisionPolicySignalMultipliers multipliers,
      DecisionBucket bucket) {
    if (!sample.bucket().equals(bucket)) {
      throw new EpsilonDecisionDataException(
          "training microbatch mixes Decision buckets: expected="
              + bucket
              + " actual="
              + sample.bucket());
    }
    if (sample.actorSnapshotId() <= 0L) {
      throw new EpsilonDecisionDataException("online fragment sample has no actor identity");
    }
    EpsilonUtilityTargets.requireTrainingProfile(sample.ruleProfile(), utilityProfile);
    if (!sample.learningRole().advancesActorClock() || sample.legalActionCount() <= 1) {
      return 0.0f;
    }
    EpsilonDecisionPointKind kind =
        EpsilonDecisionPointKind.ofLegalActions(
            sample.legalActionIdsView(), sample.legalActionCount());
    return multipliers.forKind(kind);
  }

  private synchronized void markFailed(Throwable exception) {
    if (failure == null) {
      failure = exception;
      accepting = false;
    }
  }

  private void requireAccepting() {
    if (!accepting || closed) {
      throw new IllegalStateException("training input pipeline is not accepting work", failure);
    }
  }

  private void requireSlot(HostSlot slot, long generation, SlotState expected) {
    if (slot.owner != this || slot.generation != generation || slot.state != expected) {
      throw new IllegalStateException(
          "training host slot state mismatch: expected=" + expected + " actual=" + slot.state);
    }
  }

  private static NDArray copyPinnedOrEmpty(
      PtTransferBatch batch, PtNDManager destination, HostSlab source, int elements) {
    return elements == 0
        ? destination.zeros(new Shape(0), DataType.INT32)
        : source.copyPinned(batch, elements);
  }

  private static void await(CompletableFuture<Void> future, String operation) throws IOException {
    try {
      future.get();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while attempting to " + operation, interrupted);
    } catch (ExecutionException failed) {
      Throwable cause = failed.getCause();
      if (cause instanceof IOException io) {
        throw io;
      }
      if (cause instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw new IOException("Failed to " + operation, cause);
    }
  }

  private static Throwable addSuppressed(Throwable primary, Throwable secondary) {
    if (primary == null) {
      return secondary;
    }
    primary.addSuppressed(secondary);
    return primary;
  }

  private static void closeSuppressing(AutoCloseable closeable, Throwable primary) {
    try {
      closeable.close();
    } catch (Throwable closeFailure) {
      primary.addSuppressed(closeFailure);
    }
  }

  private static void rethrow(Throwable throwable) {
    if (throwable instanceof RuntimeException runtime) {
      throw runtime;
    }
    if (throwable instanceof Error error) {
      throw error;
    }
    if (throwable != null) {
      throw new IllegalStateException(
          "Failed to close Decision training input pipeline", throwable);
    }
  }

  /** 二つのホスト側の実行枠の一つ。状態遷移は外側パイプラインが直列化する。 */
  final class HostSlot implements AutoCloseable {
    private final DecisionTrainingInputPipeline owner = DecisionTrainingInputPipeline.this;
    private final int index;
    private final HostSlab inputCategories = new HostSlab(DataType.INT16);
    private final HostSlab inputNumerics = new HostSlab(DataType.FLOAT32);
    private final HostSlab targetCategories = new HostSlab(DataType.INT32);
    private final HostSlab targetNumerics = new HostSlab(DataType.FLOAT32);
    private final HostSlab playerMemoryIndices = new HostSlab(DataType.INT32);
    private final HostSlab transitionIndices = new HostSlab(DataType.INT32);
    private SlotState state = SlotState.FREE;
    private long generation;
    private int rows;
    private DecisionBucket bucket;
    private int inputCategoryElements;
    private int inputNumericElements;
    private int targetCategoryElements;
    private int targetNumericElements;
    private int playerMemoryCount;
    private int transitionCount;

    private HostSlot(int index) {
      this.index = index;
    }

    private DecisionTrainingHostBatchLease begin(int rows, DecisionBucket bucket) {
      generation++;
      this.rows = rows;
      this.bucket = bucket;
      DecisionInputLayout inputLayout = new DecisionInputLayout(rows, bucket);
      DecisionTrainingTargetLayout targetLayout = new DecisionTrainingTargetLayout(rows, bucket);
      inputCategoryElements = inputLayout.categoricalElementCount();
      inputNumericElements = inputLayout.numericElementCount();
      targetCategoryElements = targetLayout.categoricalElementCount();
      targetNumericElements = targetLayout.numericElementCount();
      inputCategories.ensureAtLeast(inputCategoryElements);
      inputNumerics.ensureAtLeast(inputNumericElements);
      targetCategories.ensureAtLeast(targetCategoryElements);
      targetNumerics.ensureAtLeast(targetNumericElements);
      int playerMemoryCapacity = maximumPlayerMemoryIndices(rows);
      int transitionCapacity =
          Math.multiplyExact(
              rows,
              Math.multiplyExact(bucket.legalActionCapacity(), bucket.actionTransitionCapacity()));
      playerMemoryIndices.ensureAtLeast(playerMemoryCapacity);
      transitionIndices.ensureAtLeast(transitionCapacity);
      playerMemoryCount = 0;
      transitionCount = 0;
      state = SlotState.FILLING;
      DecisionTrainingSlabWriter writer =
          new DecisionTrainingSlabWriter(
              rows,
              bucket,
              inputCategories.shortView(inputCategoryElements),
              inputNumerics.floatView(inputNumericElements),
              targetCategories.intView(targetCategoryElements),
              targetNumerics.floatView(targetNumericElements),
              playerMemoryIndices.intView(playerMemoryCapacity),
              transitionIndices.intView(transitionCapacity));
      return new DecisionTrainingHostBatchLease(owner, this, generation, writer);
    }

    private void release() {
      state = SlotState.FREE;
      rows = 0;
      bucket = null;
      playerMemoryCount = 0;
      transitionCount = 0;
    }

    @Override
    public void close() {
      inputCategories.close();
      inputNumerics.close();
      targetCategories.close();
      targetNumerics.close();
      playerMemoryIndices.close();
      transitionIndices.close();
      state = SlotState.CLOSED;
    }

    @Override
    public String toString() {
      return "DecisionTrainingHostSlot[lane=" + laneIndex + ",index=" + index + "]";
    }
  }

  /** 一種類の再利用可能なページロックしたまたは直接参照のホスト側の連続バッファ。 */
  private final class HostSlab implements AutoCloseable {
    private final DataType dataType;
    private PtPinnedBuffer pinnedBuffer;
    private ByteBuffer directBuffer;
    private int capacity;

    private HostSlab(DataType dataType) {
      this.dataType = dataType;
    }

    private void ensureAtLeast(int elements) {
      if (capacity >= elements) {
        return;
      }
      close();
      if (transferMode == DecisionTensorTransfer.PINNED_BUFFER) {
        pinnedBuffer = pytorchOwner.allocatePinned(elements, dataType);
      } else {
        directBuffer = owner.allocateDirect(Math.multiplyExact(elements, dataType.getNumOfBytes()));
      }
      capacity = elements;
    }

    private ShortBuffer shortView(int elements) {
      return byteView(elements).asShortBuffer();
    }

    private FloatBuffer floatView(int elements) {
      return byteView(elements).asFloatBuffer();
    }

    private IntBuffer intView(int elements) {
      return byteView(elements).asIntBuffer();
    }

    private ByteBuffer byteView(int elements) {
      ByteBuffer view =
          (pinnedBuffer == null ? directBuffer : pinnedBuffer.getByteBuffer())
              .duplicate()
              .order(ByteOrder.nativeOrder());
      view.clear();
      view.limit(Math.multiplyExact(elements, dataType.getNumOfBytes()));
      return view.slice().order(ByteOrder.nativeOrder());
    }

    private NDArray copyPinned(PtTransferBatch batch, int elements) {
      return batch.copy(pinnedBuffer, new Shape(elements));
    }

    private NDArray copyDirect(NDManager destination, int elements) {
      ByteBuffer source = byteView(elements);
      Shape shape = new Shape(elements);
      if (destination instanceof PtNDManager pytorchDestination) {
        PtNDArray array = pytorchDestination.create(shape, dataType);
        array.copyFromDirectBuffer(source);
        return array;
      }
      Buffer typed =
          switch (dataType) {
            case INT16 -> source.asShortBuffer();
            case INT32 -> source.asIntBuffer();
            case FLOAT32 -> source.asFloatBuffer();
            default ->
                throw new IllegalStateException("Unsupported training slab type: " + dataType);
          };
      return destination.create(typed, shape, dataType);
    }

    private NDArray copyDirectOrEmpty(NDManager destination, int elements) {
      return elements == 0
          ? destination.zeros(new Shape(0), dataType)
          : copyDirect(destination, elements);
    }

    @Override
    public void close() {
      if (pinnedBuffer != null) {
        pinnedBuffer.close();
        pinnedBuffer = null;
      }
      directBuffer = null;
      capacity = 0;
    }
  }

  /** H2D 処理の識別情報、未加工のデバイス連続バッファ、ホスト側の実行枠 retirementをまとめる内部所有権。 */
  static final class PendingTransfer {
    private final HostSlot slot;
    private NDManager destination;
    private final PtTransferTicket ticket;
    private final NDArray inputCategories;
    private final NDArray inputNumerics;
    private final NDArray playerMemoryIndices;
    private final NDArray transitionIndices;
    private final NDArray targetCategories;
    private final NDArray targetNumerics;
    private boolean closeDestination;

    private PendingTransfer(
        HostSlot slot,
        NDManager destination,
        PtTransferTicket ticket,
        NDArray inputCategories,
        NDArray inputNumerics,
        NDArray playerMemoryIndices,
        NDArray transitionIndices,
        NDArray targetCategories,
        NDArray targetNumerics) {
      this.slot = slot;
      this.destination = destination;
      this.ticket = ticket;
      this.inputCategories = inputCategories;
      this.inputNumerics = inputNumerics;
      this.playerMemoryIndices = playerMemoryIndices;
      this.transitionIndices = transitionIndices;
      this.targetCategories = targetCategories;
      this.targetNumerics = targetNumerics;
    }

    private NDManager takeDestination() {
      NDManager manager = destination;
      if (manager == null) {
        throw new IllegalStateException("transfer destination is already consumed");
      }
      destination = null;
      return manager;
    }
  }

  /** 復号反復処理で追加配列を作らずに集計値を構築する。 */
  private static final class SummaryBuilder {
    private final DecisionBucket bucket;
    private double actorMass;
    private double valueMass;
    private long positive;
    private long negative;
    private long zero;

    private SummaryBuilder(DecisionBucket bucket) {
      this.bucket = bucket;
    }

    private void add(EpsilonDecisionTrainingSampleDescriptor sample, float actorWeight) {
      actorMass += actorWeight;
      valueMass += 1.0;
      if (!(actorWeight > 0.0f)) {
        return;
      }
      float scalar = sample.advantage();
      if (!Float.isFinite(scalar)) {
        throw new EpsilonDecisionDataException("scalar advantage must be finite");
      }
      if (scalar > 0.0f) {
        positive++;
      } else if (scalar < 0.0f) {
        negative++;
      } else {
        zero++;
      }
    }

    private DecisionTrainingBatchSummary finish(int rows) {
      return new DecisionTrainingBatchSummary(
          rows, bucket, actorMass, valueMass, new AdvantageSigns(positive, negative, zero));
    }
  }

  private enum SlotState {
    FREE,
    FILLING,
    SEALED,
    IN_FLIGHT,
    BROKEN,
    CLOSED
  }

  private static int maximumPlayerMemoryIndices(int rows) {
    int tokensPerPlayer =
        1
            + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER
            + DecisionInputSchema.MAX_MELDS_PER_PLAYER;
    return Math.multiplyExact(rows, Math.multiplyExact(GameState.NUM_PLAYERS, tokensPerPlayer));
  }
}
