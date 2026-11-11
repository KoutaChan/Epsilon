package com.epsilon.agent.learn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.pytorch.engine.PtCopyEvent;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtFlatGradientPacker;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.pytorch.engine.PtTensorCopyPlan;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.GradientCollector;
import ai.djl.training.optimizer.Optimizer;
import com.epsilon.agent.runtime.NetworkRuntime;
import java.io.IOException;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * 複数のレーン（デバイスごとのネットワーク）で、データ並列に学習する。レーン 0 が主のネットワークで、ほかのレーンはそのレプリカ。
 *
 * <p>1 ステップは次のように進む。まず呼び出し側が、各レーンで順伝播と逆伝播を積む（{@link #execute}）。続く更新
 * （{@link #step}）では、パラメーターの組（{@link ParameterGroup}）ごとに、勾配を 1 本の連続した FP32 の配列へ詰め、パラメーターの
 * 勾配を 0 にする。詰めた勾配を主のレーンで足し合わせ、全ての組の勾配を確かめる。確かめるのは、有限であること、0 でない値を含むこと
 * （全て 0 でも更新する組を除く）、倍率を掛けても溢れないこと。全て通れば、組ごとのオプティマイザで連続した配列を 1 回だけ更新し、
 * 更新した重みを全てのレーンのネットワークへ配る。
 *
 * <p>確かめに通らない組が 1 つでもあれば、どの組も更新せず、そのステップの勾配を捨てる。更新を始めた後に失敗したときは、レーンの
 * 重みがそろっている保証が無いので、以後は使えなくする。
 *
 * <p>GPU のレーンの計算は、そのデバイスの計算用のストリーム（実行時のリソースから借りる）に積む。デバイスの間の順序はイベントで
 * 伝え、ホストは待たない。
 *
 * @param <L> レーンの型（モデルの系列ごとに、ネットワークと計算用のヘルパーを持たせる）
 */
public class DataParallelGroup<L extends DataParallelGroup.Lane> implements AutoCloseable {

  private final List<L> lanes;
  private final Map<ParameterGroup, GroupState> groups = new LinkedHashMap<>();
  private final ExecutorService executor;
  private Throwable terminalFailure;
  private boolean closed;

  /** 0 番のレーンが主になる。レプリカのネットワークは、主のネットワークと同じ名前・形・値のパラメーターを持っていなければならない。 */
  public DataParallelGroup(List<L> lanes) {
    if (lanes.isEmpty()) {
      throw new IllegalArgumentException("A data-parallel group needs at least one lane");
    }
    this.lanes = List.copyOf(lanes);
    executor =
        lanes.size() == 1
            ? null
            : Executors.newFixedThreadPool(
                lanes.size() - 1, Thread.ofPlatform().daemon().name("epsilon-data-parallel-", 0).factory());
    // 最初の順伝播が、ネットワークの用意（レプリカの初期化と値のコピー）を追い越さないように、レーンのストリームが待つイベントを記録する
    for (L lane : this.lanes) {
      lane.recordParametersReady();
    }
  }

  public int laneCount() {
    return lanes.size();
  }

  /** i 番目のジョブをレーン i で並行して走らせ、結果をジョブと同じ順に返す。ジョブの数はレーンの数以下にする。 */
  public <T> List<T> execute(List<? extends LaneWork<L, T>> work) throws IOException {
    int[] assigned = new int[work.size()];
    for (int index = 0; index < assigned.length; index++) {
      assigned[index] = index;
    }
    return execute(assigned, work);
  }

  /**
   * i 番目のジョブを、i 番目に指定したレーンで並行して走らせ、結果をジョブと同じ順に返す。同じレーンは 2 度指定できない。レーン 0
   * のジョブは呼び出したスレッドで走り、ほかは並列用のスレッドで走る。
   */
  public <T> List<T> execute(int[] laneIndices, List<? extends LaneWork<L, T>> work) throws IOException {
    requireUsable();
    if (work.isEmpty() || work.size() > lanes.size() || laneIndices.length != work.size()) {
      throw new IllegalArgumentException(
          "Lane work must be 1.." + lanes.size() + " items, one per lane: " + work.size() + " items for "
              + laneIndices.length + " lanes");
    }
    requireDistinctLanes(laneIndices);
    List<Future<T>> futures = new ArrayList<>(work.size());
    int[] futureIndices = new int[work.size()];
    int canonical = -1;
    for (int index = 0; index < work.size(); index++) {
      if (laneIndices[index] == 0) {
        canonical = index;
        continue;
      }
      L lane = lanes.get(laneIndices[index]);
      LaneWork<L, T> task = work.get(index);
      futureIndices[futures.size()] = index;
      futures.add(executor.submit(() -> run(lane, task)));
    }
    List<T> results = new ArrayList<>(Collections.<T>nCopies(work.size(), null));
    Throwable failure = null;
    if (canonical >= 0) {
      try {
        results.set(canonical, run(lanes.getFirst(), work.get(canonical)));
      } catch (Throwable laneFailure) {
        failure = laneFailure;
      }
    }
    boolean interrupted = false;
    for (int index = 0; index < futures.size(); index++) {
      while (true) {
        try {
          results.set(futureIndices[index], futures.get(index).get());
          break;
        } catch (InterruptedException waitFailure) {
          interrupted = true;
        } catch (ExecutionException laneFailure) {
          failure = suppress(failure, laneFailure.getCause());
          break;
        }
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
    if (failure == null) {
      return results;
    }
    throw propagate(failure);
  }

  /** 全てのレーンで、ネットワークのパラメーターの勾配を 0 にする。失敗したステップの勾配を捨てるときに使う。 */
  public void clearGradients() {
    requireOpen();
    try (ComputeScopes ignored = new ComputeScopes()) {
      for (L lane : lanes) {
        for (Parameter parameter : lane.model().getBlock().getParameters().values()) {
          NDArray array = parameter.getArray();
          if (array.hasGradient()) {
            try (NDArray gradient = array.getGradient()) {
              gradient.fillI(0f);
            }
          }
        }
      }
    }
  }

  /**
   * 指定したレーンの勾配を足し合わせ、各組を 1 回ずつ更新する。組どうしでパラメーターが重なってはならない。同じ組を 1 つの
   * ステップに 2 回入れたときや、レーンか更新が 1 つも無いときは IllegalArgumentException を投げる。
   *
   * @param contributors 勾配を溜めたレーン（ほかのレーンの勾配は足さず、0 にもしない）
   * @param updates 組ごとの更新
   * @throws IllegalStateException どれかの組の勾配が確かめに通らないとき（このときはどの組も更新しない）
   */
  public void step(int[] contributors, List<Update> updates) throws IOException {
    requireUsable();
    if (contributors.length == 0 || updates.isEmpty()) {
      throw new IllegalArgumentException("A step needs contributing lanes and updates");
    }
    requireDistinctLanes(contributors);
    GroupState[] states = new GroupState[updates.size()];
    Set<ParameterGroup> seen = Collections.newSetFromMap(new IdentityHashMap<>());
    for (int index = 0; index < states.length; index++) {
      ParameterGroup group = updates.get(index).group();
      if (!seen.add(group)) {
        throw new IllegalArgumentException("A step updates " + group.name() + " twice");
      }
      states[index] = groups.computeIfAbsent(group, key -> new GroupState(key, Layout.of(lanes, key)));
    }
    try (ComputeScopes ignored = new ComputeScopes()) {
      for (GroupState state : states) {
        state.initialize(lanes);
      }
    }
    LaneWork<L, Boolean> packing =
        lane -> {
          for (GroupState state : states) {
            lane.buffers(state).packAndClear(state.group.allowsMissingGradients());
          }
          return Boolean.TRUE;
        };
    execute(contributors, Collections.nCopies(contributors.length, packing));

    boolean committing = false;
    try {
      try (ComputeScopes ignored = new ComputeScopes()) {
        TransferBuffers[] canonical = new TransferBuffers[states.length];
        for (int index = 0; index < states.length; index++) {
          canonical[index] = reduce(states[index], contributors);
        }
        checkAndScale(updates, canonical);
        for (int index = 0; index < states.length; index++) {
          prepareGradient(states[index].group, canonical[index].gradient);
        }
        committing = true;
        for (int index = 0; index < states.length; index++) {
          TransferBuffers updated = canonical[index];
          updates.get(index).optimizer().update(states[index].group.name(), updated.parameter, updated.gradient);
          for (int lane = 1; lane < lanes.size(); lane++) {
            updated.parameter.copyTo(lanes.get(lane).buffers(states[index]).parameter);
          }
        }
      }
      LaneWork<L, Boolean> distributing =
          lane -> {
            for (GroupState state : states) {
              lane.buffers(state).copyToModel();
            }
            return Boolean.TRUE;
          };
      execute(Collections.nCopies(lanes.size(), distributing));
    } catch (IOException | RuntimeException | Error failure) {
      if (committing && terminalFailure == null) {
        terminalFailure = failure;
      }
      throw failure;
    }
  }

  /** 全てのレーンで、重みの更新が終わるまで待つ。ネットワークをほかのストリームで使う前に呼ぶ。 */
  public void awaitParameterUpdates() {
    requireUsable();
    for (L lane : lanes) {
      if (lane.stepCompletion != null) {
        lane.stepCompletion.synchronize();
      }
    }
  }

  /** レプリカの勾配を受ける配列を捨て、レーンを後ろから順に閉じる（レプリカのネットワークも閉じる）。 */
  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    if (executor != null) {
      executor.close();
    }
    for (GroupState state : groups.values()) {
      if (state.reduction != null) {
        state.reduction.close();
      }
    }
    RuntimeException failure = null;
    for (int index = lanes.size() - 1; index >= 0; index--) {
      try {
        lanes.get(index).close();
      } catch (RuntimeException closeFailure) {
        if (failure == null) {
          failure = closeFailure;
        } else {
          failure.addSuppressed(closeFailure);
        }
      }
    }
    if (failure != null) {
      throw failure;
    }
  }

  /** 元のモデルの全てのパラメーターの値を、コピー先のモデルで同じ名前のパラメーターへコピーする。レプリカを作るときに使う。 */
  public static void copyParameters(Model source, Model target) {
    ParameterList targetParameters = target.getBlock().getParameters();
    for (var entry : source.getBlock().getParameters()) {
      Parameter parameter = targetParameters.get(entry.getKey());
      if (parameter == null) {
        throw new IllegalArgumentException("The replica has no parameter " + entry.getKey());
      }
      entry.getValue().getArray().copyTo(parameter.getArray());
    }
  }

  /**
   * 足し合わせた組の勾配に、モデルの系列ごとの処理（ノルムで勾配を縮めるなど）をする。勾配を確かめて倍率を掛けた後、更新の前に
   * 呼ばれる。実装は勾配の配列を保持せず、値を有限のまま保つこと。
   */
  protected void prepareGradient(ParameterGroup group, NDArray gradient) {}

  /** 組の勾配を主のレーンで足し合わせ、主のレーンにあるその組の配列を返す。 */
  private TransferBuffers reduce(GroupState state, int[] contributors) {
    TransferBuffers canonical = lanes.getFirst().buffers(state);
    boolean canonicalContributes = false;
    for (int lane : contributors) {
      canonicalContributes |= lane == 0;
    }
    if (!canonicalContributes) {
      canonical.gradient.fillI(0f);
    }
    for (int lane : contributors) {
      if (lane != 0) {
        lanes.get(lane).buffers(state).gradient.copyTo(state.reduction);
        canonical.gradient.addi(state.reduction);
      }
    }
    return canonical;
  }

  /** 組ごとに勾配の絶対値の最大を求め、ホストとの 1 回の同期でまとめて読んで確かめる。全ての組が通ってから倍率を掛ける。 */
  private static void checkAndScale(List<Update> updates, TransferBuffers[] canonical) {
    float[] maximums;
    try (NDManager scope = canonical[0].gradient.getManager().newSubManager()) {
      NDList peaks = new NDList(canonical.length);
      for (TransferBuffers buffers : canonical) {
        NDArray absolute = buffers.gradient.abs();
        absolute.attach(scope);
        NDArray peak = absolute.max();
        peak.attach(scope);
        peaks.add(peak);
      }
      NDArray stacked = NDArrays.stack(peaks);
      stacked.attach(scope);
      maximums = stacked.toFloatArray();
    }
    for (int index = 0; index < canonical.length; index++) {
      Update update = updates.get(index);
      String name = update.group().name();
      float maximum = maximums[index];
      if (!Float.isFinite(maximum)) {
        throw new IllegalStateException("The " + name + " gradient holds a non-finite value");
      }
      if (maximum == 0f && !update.group().allowsZeroGradient()) {
        throw new IllegalStateException("The " + name + " gradient is zero");
      }
      if ((double) maximum * update.gradientScale() > Float.MAX_VALUE) {
        throw new IllegalStateException(
            "Scaling the " + name + " gradient overflows: maxAbs=" + maximum + " scale=" + update.gradientScale());
      }
    }
    for (int index = 0; index < canonical.length; index++) {
      float scale = updates.get(index).gradientScale();
      if (scale != 1f) {
        canonical[index].gradient.muli(scale);
      }
    }
  }

  private void requireDistinctLanes(int[] laneIndices) {
    boolean[] used = new boolean[lanes.size()];
    for (int lane : laneIndices) {
      if (lane < 0 || lane >= lanes.size() || used[lane]) {
        throw new IllegalArgumentException("Invalid or repeated lane " + lane + " of " + lanes.size());
      }
      used[lane] = true;
    }
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("The data-parallel group is closed");
    }
  }

  private void requireUsable() {
    requireOpen();
    if (terminalFailure != null) {
      throw new IllegalStateException("The data-parallel group failed during an update", terminalFailure);
    }
  }

  private static <L extends Lane, T> T run(L lane, LaneWork<L, T> work) throws Exception {
    try (LaneExecution ignored = lane.openExecution()) {
      return work.run(lane);
    }
  }

  private static Throwable suppress(Throwable failure, Throwable next) {
    if (failure == null) {
      return next;
    }
    failure.addSuppressed(next);
    return failure;
  }

  private static IOException propagate(Throwable failure) {
    if (failure instanceof IOException io) {
      return io;
    }
    if (failure instanceof RuntimeException runtime) {
      throw runtime;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    return new IOException("A data-parallel lane failed", failure);
  }

  /** 1 つのオプティマイザで一緒に更新するパラメーターの組。モデルの系列ごとに enum で宣言する。 */
  public interface ParameterGroup {
    /** 組の名前。オプティマイザの状態を引くのに使い、例外のメッセージにも出す。 */
    String name();

    boolean includes(String parameter);

    /** レーンで勾配を持たないパラメーター（そのレーンの損失に関わらなかったもの）を、勾配 0 とみなすか。false なら例外にする。 */
    boolean allowsMissingGradients();

    /** 足し合わせた勾配が全て 0 のステップでも更新するか。false なら例外にする。 */
    boolean allowsZeroGradient();
  }

  /**
   * 1 つの組の更新。
   *
   * @param gradientScale 足し合わせた勾配に掛ける倍率（有限の正の数）
   */
  public record Update(ParameterGroup group, Optimizer optimizer, float gradientScale) {
    public Update {
      Objects.requireNonNull(group, "group");
      Objects.requireNonNull(optimizer, "optimizer");
      if (!Float.isFinite(gradientScale) || gradientScale <= 0f) {
        throw new IllegalArgumentException("The gradient scale must be finite and positive: " + gradientScale);
      }
    }
  }

  @FunctionalInterface
  public interface LaneWork<L extends Lane, T> {
    T run(L lane) throws Exception;
  }

  /**
   * 1 つのデバイスのネットワーク。GPU のレーンは、そのデバイスの計算用のストリーム（実行時のリソースから借りる）で計算し、重みの
   * 用意と更新が終わったことをイベントで次のジョブへ伝える。CPU のレーンはストリームもイベントも持たない。
   */
  public static class Lane implements AutoCloseable {
    private final Model model;
    private final boolean ownsModel;
    private final Map<ParameterGroup, TransferBuffers> transfers = new LinkedHashMap<>();
    final PtStream computeStream;
    final PtEvent stepCompletion;
    final PtEvent parametersReady;
    private PtPinnedBuffer metricBuffer;
    private PtCopyEvent metricEvent;
    private int metricCount;

    /**
     * @param ownsModel レーンを閉じるときにネットワークも閉じるか（レプリカは true、主のレーンは false）
     * @param runtime 実行時のリソース。GPU のレーンはここから計算用のストリームを借りる
     */
    public Lane(Model model, boolean ownsModel, NetworkRuntime runtime) {
      this.model = Objects.requireNonNull(model, "model");
      this.ownsModel = ownsModel;
      Device device = model.getNDManager().getDevice();
      if (!device.isGpu()) {
        computeStream = null;
        stepCompletion = null;
        parametersReady = null;
        return;
      }
      computeStream = Objects.requireNonNull(runtime, "runtime").streams(device).compute();
      PtEvent completion = computeStream.newEvent();
      try {
        parametersReady = computeStream.newEvent();
      } catch (RuntimeException | Error failure) {
        completion.close();
        throw failure;
      }
      stepCompletion = completion;
    }

    public Model model() {
      return model;
    }

    public NDManager manager() {
      return model.getNDManager();
    }

    /** 計算用のストリーム（CPU のレーンは null）。自前の演算を、DJL の演算と同じストリームに積むために使う。 */
    public final PtStream computeStream() {
      return computeStream;
    }

    public GradientCollector newGradientCollector() {
      return manager().getEngine().newGradientCollector();
    }

    /**
     * 1 次元の FP32 の指標を、使い回すページ固定のバッファへ非同期に転送する（GPU のレーンで使う）。前の転送を読み終える前に
     * 呼ぶと IllegalStateException を投げる。PyTorch の配列でなければ IllegalArgumentException を投げる。
     */
    public void enqueueMetricReadback(NDArray metrics) {
      if (metricEvent != null) {
        throw new IllegalStateException("A metric readback is already pending");
      }
      if (metrics.getShape().dimension() != 1 || metrics.getDataType() != DataType.FLOAT32) {
        throw new IllegalArgumentException(
            "Metrics must be one-dimensional FLOAT32: " + metrics.getShape() + " " + metrics.getDataType());
      }
      if (!(metrics instanceof PtNDArray pytorch) || !(manager() instanceof PtNDManager pinned)) {
        throw new IllegalArgumentException("An asynchronous metric readback needs PyTorch arrays");
      }
      int count = Math.toIntExact(metrics.size());
      if (metricBuffer == null || metricBuffer.size() < count) {
        if (metricBuffer != null) {
          metricBuffer.close();
        }
        metricBuffer = pinned.allocatePinned(count, DataType.FLOAT32);
      }
      metricEvent = pytorch.copyToPinnedBufferAsync(metricBuffer);
      metricCount = count;
    }

    /** 出した指標の転送が終わるのを待って、値を読む。 */
    public float[] awaitMetricReadback() {
      if (metricEvent == null) {
        throw new IllegalStateException("No metric readback is pending");
      }
      try {
        metricEvent.synchronize();
        float[] values = new float[metricCount];
        metricBuffer.getByteBuffer().duplicate().order(ByteOrder.nativeOrder()).asFloatBuffer().get(values);
        return values;
      } finally {
        closeMetricReadback();
      }
    }

    /** 失敗したときに、まだ読んでいない指標の転送を捨てる。 */
    public void closeMetricReadback() {
      if (metricEvent == null) {
        return;
      }
      try {
        metricEvent.close();
      } finally {
        metricEvent = null;
        metricCount = 0;
      }
    }

    /** 計算が終わったことを確かめた後に呼ばれ、モデルの系列ごとに持つデバイスのリソースを解放する。 */
    protected void closeRuntime() {}

    @Override
    public void close() {
      Throwable failure = closeAfter(this::closeMetricReadback, null);
      if (stepCompletion != null) {
        try {
          stepCompletion.synchronize();
          parametersReady.synchronize();
        } catch (RuntimeException | Error problem) {
          failure = suppress(failure, problem);
        }
      }
      for (TransferBuffers transfer : transfers.values()) {
        failure = closeAfter(transfer, failure);
      }
      failure = closeAfter(this::closeRuntime, failure);
      if (metricBuffer != null) {
        failure = closeAfter(metricBuffer, failure);
      }
      if (stepCompletion != null) {
        failure = closeAfter(stepCompletion, failure);
        failure = closeAfter(parametersReady, failure);
      }
      if (ownsModel) {
        failure = closeAfter(model, failure);
      }
      if (failure instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (failure instanceof Error error) {
        throw error;
      }
      if (failure != null) {
        throw new IllegalStateException("Cannot close a data-parallel lane", failure);
      }
    }

    /** 重みの用意ができたこと（または配り終えたこと）をイベントに記録する。このレーンで次に計算するジョブは、デバイスの上でその後に走る。 */
    void recordParametersReady() {
      if (parametersReady != null) {
        parametersReady.record();
        stepCompletion.record();
      }
    }

    /** レーンの計算用のストリームに切り替え、重みの用意を待ってから計算する。閉じると、元のストリームもこの計算の終わりを待つ。 */
    LaneExecution openExecution() {
      if (computeStream == null) {
        return LaneExecution.DIRECT;
      }
      PtStreamScope scope = computeStream.openScope();
      try {
        parametersReady.waitOnStream();
        return new LaneExecution(scope, stepCompletion);
      } catch (RuntimeException | Error failure) {
        scope.close();
        throw failure;
      }
    }

    TransferBuffers buffers(GroupState state) {
      return transfers.computeIfAbsent(state.group, ignored -> new TransferBuffers(model, state.layout));
    }

    private static Throwable closeAfter(AutoCloseable resource, Throwable failure) {
      try {
        resource.close();
        return failure;
      } catch (Throwable problem) {
        return suppress(failure, problem);
      }
    }
  }

  /** レーンの計算が終わったことを、イベントで元のストリームへ伝える（ホストは待たない）。 */
  private record LaneExecution(PtStreamScope scope, PtEvent completion) implements AutoCloseable {
    private static final LaneExecution DIRECT = new LaneExecution(null, null);

    @Override
    public void close() {
      if (completion == null) {
        return;
      }
      try {
        completion.record();
      } finally {
        scope.close();
      }
      completion.waitOnStream();
    }
  }

  /**
   * 主のレーンで組の勾配を足し合わせ、更新し、重みを配る間に使う。GPU ごとにレーンの計算用のストリームへ切り替え、前のステップの
   * 更新と重みの用意が終わった後に積む。閉じるときに、重みの用意のイベントを記録し直す。
   */
  private final class ComputeScopes implements AutoCloseable {
    private final List<PtStreamScope> scopes = new ArrayList<>();

    ComputeScopes() {
      Set<PtStream> opened = Collections.newSetFromMap(new IdentityHashMap<>());
      try {
        for (L lane : lanes) {
          if (lane.computeStream != null && opened.add(lane.computeStream)) {
            scopes.add(lane.computeStream.openScope());
          }
        }
        for (L lane : lanes) {
          if (lane.stepCompletion != null) {
            lane.stepCompletion.waitOnStream();
            lane.parametersReady.waitOnStream();
          }
        }
      } catch (RuntimeException | Error failure) {
        closeScopes();
        throw failure;
      }
    }

    @Override
    public void close() {
      try {
        for (L lane : lanes) {
          if (lane.parametersReady != null) {
            lane.parametersReady.record();
          }
        }
      } finally {
        closeScopes();
      }
      for (L lane : lanes) {
        if (lane.parametersReady != null) {
          lane.parametersReady.waitOnStream();
        }
      }
    }

    private void closeScopes() {
      for (int index = scopes.size() - 1; index >= 0; index--) {
        scopes.get(index).close();
      }
    }
  }

  /** 組のパラメーターの並びと、レプリカの勾配を主のレーンで受ける配列。 */
  private static final class GroupState {
    private final ParameterGroup group;
    private final Layout layout;
    private NDArray reduction;
    private boolean initialized;

    GroupState(ParameterGroup group, Layout layout) {
      this.group = group;
      this.layout = layout;
    }

    /** 最初のステップの前に、全てのレーンに組の配列を作り、主のレーンの重みをそのレーンの連続した配列へ詰める。 */
    void initialize(List<? extends Lane> lanes) {
      if (initialized) {
        return;
      }
      for (Lane lane : lanes) {
        lane.buffers(this);
      }
      Lane canonical = lanes.getFirst();
      if (lanes.size() > 1) {
        reduction = canonical.manager().zeros(new Shape(layout.elements()), DataType.FLOAT32);
      }
      NDArray packed = canonical.buffers(this).parameter;
      ParameterList parameters = canonical.model().getBlock().getParameters();
      for (Slice slice : layout.slices()) {
        try (NDArray flat = parameters.get(slice.name()).getArray().reshape(slice.elements())) {
          packed.set(slice.index(), flat);
        }
      }
      initialized = true;
    }
  }

  /**
   * 組のパラメーターを連続した配列に並べたときの、それぞれの位置。並べる順は、主のネットワークでのパラメーターの順。
   *
   * @param elements 全ての要素の数
   */
  private record Layout(List<Slice> slices, long elements) {

    /** 組の並びを作る。組の全てのパラメーターが凍結されていない FP32 であることと、レプリカも同じ名前と形を持つことを確かめる。 */
    static Layout of(List<? extends Lane> lanes, ParameterGroup group) {
      List<Slice> slices = new ArrayList<>();
      long offset = 0;
      for (var entry : lanes.getFirst().model().getBlock().getParameters()) {
        if (!group.includes(entry.getKey())) {
          continue;
        }
        Parameter parameter = entry.getValue();
        if (!parameter.requiresGradient()) {
          throw new IllegalStateException("Parameter " + entry.getKey() + " of " + group.name() + " is frozen");
        }
        NDArray array = parameter.getArray();
        if (array.getDataType() != DataType.FLOAT32) {
          throw new IllegalArgumentException(
              "Parameter " + entry.getKey() + " of " + group.name() + " is " + array.getDataType() + ", not FLOAT32");
        }
        long end = Math.addExact(offset, array.getShape().size());
        slices.add(new Slice(entry.getKey(), array.getShape(), offset, end));
        offset = end;
      }
      if (slices.isEmpty()) {
        throw new IllegalArgumentException("Group " + group.name() + " holds no parameter");
      }
      for (int index = 1; index < lanes.size(); index++) {
        ParameterList replica = lanes.get(index).model().getBlock().getParameters();
        for (Slice slice : slices) {
          Parameter parameter = replica.get(slice.name());
          if (parameter == null
              || !parameter.getArray().getShape().equals(slice.shape())
              || parameter.getArray().getDataType() != DataType.FLOAT32) {
            throw new IllegalArgumentException("Lane " + index + " differs at parameter " + slice.name());
          }
        }
      }
      return new Layout(List.copyOf(slices), offset);
    }
  }

  /** 1 つのパラメーターが連続した配列で占める範囲（始まりを含み、終わりを含まない）。 */
  private record Slice(String name, Shape shape, long offset, long end) {
    Shape elements() {
      return new Shape(end - offset);
    }

    NDIndex index() {
      return new NDIndex().addSliceDim(offset, end);
    }
  }

  /**
   * レーンと組ごとに、連続した勾配・重みの配列と、ネットワークのパラメーターとの間で値をコピーする。GPU では、勾配を詰めて 0 に
   * するのも、重みを配るのも、それぞれ 1 回のカーネルで行う。CPU では、パラメーターの形に戻したビューを通して、パラメーターごとに
   * コピーする。
   */
  private static final class TransferBuffers implements AutoCloseable {
    private final Layout layout;
    private final NDArray gradient;
    private final NDArray parameter;
    private final NDList modelParameters = new NDList();
    private final List<NDArray> parameterViews = new ArrayList<>();
    private final List<NDArray> gradientViews = new ArrayList<>();
    private final PtFlatGradientPacker packer;
    private final PtTensorCopyPlan copyPlan;

    TransferBuffers(Model model, Layout layout) {
      this.layout = layout;
      NDManager manager = model.getNDManager();
      boolean gpu = manager.getDevice().isGpu();
      gradient = manager.zeros(new Shape(layout.elements()), DataType.FLOAT32);
      parameter = manager.zeros(new Shape(layout.elements()), DataType.FLOAT32);
      PtTensorCopyPlan createdPlan = null;
      try {
        ParameterList parameters = model.getBlock().getParameters();
        for (Slice slice : layout.slices()) {
          modelParameters.add(parameters.get(slice.name()).getArray());
          parameterViews.add(view(parameter, slice));
          if (!gpu) {
            gradientViews.add(view(gradient, slice));
          }
        }
        if (gpu) {
          PtEngine engine = (PtEngine) manager.getEngine();
          createdPlan = engine.newTensorCopyPlan(new NDList(parameterViews), modelParameters);
          packer = engine.newFlatGradientPacker(modelParameters, gradient);
        } else {
          packer = null;
        }
        copyPlan = createdPlan;
      } catch (RuntimeException | Error failure) {
        if (createdPlan != null) {
          createdPlan.close();
        }
        closeArrays(failure);
        throw failure;
      }
    }

    /** レーンのパラメーターの勾配を連続した配列へ詰め、パラメーターの勾配を 0 にする。 */
    void packAndClear(boolean allowMissing) {
      if (packer != null) {
        packer.packAndClear(
            allowMissing
                ? PtFlatGradientPacker.MissingGradientPolicy.ZERO
                : PtFlatGradientPacker.MissingGradientPolicy.ERROR);
        return;
      }
      for (int index = 0; index < modelParameters.size(); index++) {
        // NDArray#getGradient は勾配の無いパラメーターにも新しい 0 の配列を返すので、勾配が無いことを見分けるために JNI から直接読む
        PtNDArray sourceGradient = JniUtils.getGradient((PtNDArray) modelParameters.get(index));
        if (sourceGradient == null) {
          if (!allowMissing) {
            throw new IllegalStateException("Parameter " + layout.slices().get(index).name() + " has no gradient");
          }
          gradientViews.get(index).fillI(0f);
          continue;
        }
        try (sourceGradient) {
          sourceGradient.copyTo(gradientViews.get(index));
          sourceGradient.fillI(0f);
        }
      }
    }

    /** 連続した重みの配列を、ネットワークのパラメーターへコピーする。 */
    void copyToModel() {
      if (copyPlan != null) {
        copyPlan.copy();
        return;
      }
      for (int index = 0; index < modelParameters.size(); index++) {
        parameterViews.get(index).copyTo(modelParameters.get(index));
      }
    }

    @Override
    public void close() {
      if (copyPlan != null) {
        copyPlan.close();
      }
      if (packer != null) {
        packer.close();
      }
      closeArrays(null);
    }

    /** 持っている配列を全て閉じる。閉じるときの例外は、先に起きた例外を渡されていればその suppressed に足し、無ければ最初の例外を投げる。 */
    private void closeArrays(Throwable failure) {
      List<NDArray> owned = new ArrayList<>(gradientViews);
      owned.addAll(parameterViews);
      owned.add(parameter);
      owned.add(gradient);
      RuntimeException first = null;
      for (NDArray array : owned) {
        try {
          array.close();
        } catch (RuntimeException problem) {
          if (failure != null) {
            failure.addSuppressed(problem);
          } else if (first == null) {
            first = problem;
          } else {
            first.addSuppressed(problem);
          }
        }
      }
      if (first != null) {
        throw first;
      }
    }

    /** 連続した配列から 1 つのパラメーターの範囲を取り出し、パラメーターの形に戻したビューを返す。 */
    private static NDArray view(NDArray flat, Slice slice) {
      try (NDArray range = flat.get(slice.index())) {
        return range.reshape(slice.shape());
      }
    }
  }
}
