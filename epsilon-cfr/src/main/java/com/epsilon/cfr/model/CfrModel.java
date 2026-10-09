package com.epsilon.cfr.model;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.training.ParameterStore;
import com.epsilon.cfr.collection.CfrPolicy;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInferenceStoragePool;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.nano.ai.model.EpsilonDecisionFrozenFeatures;
import com.epsilon.network.NetworkDevices;
import com.epsilon.network.NetworkRuntime;
import com.epsilon.spi.PolicyExecutionContext;
import com.epsilon.training.DataParallelGroup;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

/**
 * 符号付き後悔または平均方策を独立に学習するモデル。バッチ単位で勾配を計算し、AdamWの状態を保存する。
 *
 * <p>各行は目的関数の重みに比例して抽出済みのため、重要度重みを重ねて掛けない。
 */
public final class CfrModel implements CfrPolicy, AutoCloseable {
  public enum Role {
    REGRET,
    AVERAGE
  }

  public record TrainingMetrics(int samples, double loss, double gradientNorm, long updates) {}

  /** Nanoから引き継いだ表現パラメーター数。方策出力、価値、最適化履歴は含まない。 */
  public record NanoInitializationReport(
      int copiedParameters, long copiedScalars, long totalScalars) {}

  private record Metadata(
      int version,
      String architecture,
      String inputSchema,
      Role role,
      CfrModelConfig config,
      long updates,
      boolean trainingState) {}

  /** 一つの容量区分の入力と、バッチ内の各行に対応する元の入力位置。 */
  static final class EncodedGroup {
    final DecisionHostBatch host;
    final int[] indices;
    final boolean forced;

    EncodedGroup(DecisionHostBatch host, int[] indices, boolean forced) {
      this.host = host;
      this.indices = indices;
      this.forced = forced;
    }
  }

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private final CfrModelConfig config;
  private final Role role;
  private final Model model;
  private final CfrNetwork network;
  private final NetworkDevices devices;
  private final DecisionBatchTransfer.Workspace inferenceTransfer;
  private final CfrInputEncoder inputEncoder = new CfrInputEncoder();
  private CfrLearner learner;
  private EpsilonDecisionFrozenFeatures frozenFeatures;
  private boolean trainable = true;
  private boolean closed;
  private long updates;

  public CfrModel(CfrModelConfig config, Role role, Device device) {
    this(config, role, NetworkDevices.of(device));
  }

  public CfrModel(CfrModelConfig config, Role role, NetworkDevices devices) {
    this.config = Objects.requireNonNull(config);
    this.role = Objects.requireNonNull(role);
    Device primary = first(devices);
    this.devices = devices;
    NetworkRuntime.initialize();
    model =
        Model.newInstance(
            "epsilon-cfr-" + role.name().toLowerCase(Locale.ROOT), primary, "PyTorch");
    network = new CfrNetwork(config);
    model.setBlock(network);
    synchronized (CfrModel.class) {
      model.getNDManager().getEngine().setRandomSeed(Long.hashCode(config.seed()));
      network.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, 1));
    }
    inferenceTransfer = new DecisionBatchTransfer.Workspace(model.getNDManager());
    completeParameterPublication();
  }

  private static Device first(NetworkDevices devices) {
    Objects.requireNonNull(devices, "devices");
    Set<Device> unique = new HashSet<>();
    for (Device device : devices)
      if (!unique.add(Objects.requireNonNull(device)))
        throw new IllegalArgumentException("Duplicate learner device: " + device);
    return devices.primary();
  }

  public CfrModelConfig config() {
    return config;
  }

  public Role role() {
    return role;
  }

  public long updates() {
    return updates;
  }

  public Device device() {
    return model.getNDManager().getDevice();
  }

  List<String> parameterNames() {
    return List.copyOf(network.getParameters().keys());
  }

  /** CPU上で構築する、デバイスに依存しない入力。同じモデルの別の凍結コピーでも評価できる。 */
  public static final class EncodedBatch {
    final List<EncodedGroup> groups;
    private final int size;

    EncodedBatch(List<EncodedGroup> groups, int size) {
      this.groups = List.copyOf(groups);
      this.size = size;
    }

    public int size() {
      return size;
    }

    /** 保存用観測はCPUプールで構築し、同じ確定済み入力による推論と重ねる。 */
    public CompletableFuture<List<ObservationPayload>> encodePayloadsAsync(
        PolicyExecutionContext execution) {
      return CfrInputEncoder.encodePayloads(this, Objects.requireNonNull(execution));
    }

    double[][] policies() {
      double[][] policies = new double[size()][];
      for (EncodedGroup group : groups) {
        if (!group.forced) continue;
        for (int index : group.indices) policies[index] = new double[] {1};
      }
      return policies;
    }
  }

  /** デバイスの出力バッファから独立した方策。保存用観測が揃ってから予測へまとめる。 */
  public static final class Policies {
    private final double[][] values;

    Policies(double[][] values) {
      this.values = values;
    }

    public List<Prediction> withObservations(List<ObservationPayload> observations) {
      if (observations.size() != values.length)
        throw new IllegalArgumentException("Observation count does not match policy count");
      Prediction[] predictions = new Prediction[values.length];
      for (int row = 0; row < values.length; row++)
        predictions[row] = new Prediction(values[row], observations.get(row));
      return List.of(predictions);
    }
  }

  /** 共通GPUパイプラインの一つの実行枠が専有する、再利用可能な入力バッファを作る。 */
  public synchronized CfrInferenceWorkspace newInferenceWorkspace() {
    return newInferenceWorkspace(true);
  }

  synchronized CfrInferenceWorkspace newInferenceWorkspace(boolean stateFusions) {
    ensureOpen();
    if (trainable) throw new IllegalStateException("Staged inference requires a frozen model");
    return new CfrInferenceWorkspace(
        this, model.getNDManager(), network, frozenFeatures, stateFusions);
  }

  /** CPUでの特徴構築はGPUとの同期を行わないため、前バッチのGPU実行と重ねられる。 */
  public EncodedBatch encode(List<DecisionContext> contexts) {
    return encode(contexts, null);
  }

  /** Nanoと同じCPUエンコーダーを使い、借用した観測の全行を処理してから返す。 */
  public EncodedBatch encode(List<DecisionContext> contexts, PolicyExecutionContext execution) {
    return inputEncoder.encode(contexts, execution);
  }

  /** 共通CPUプールで容量区分ごとの入力を直接構築し、全行が確定してから完了する。 */
  public CompletableFuture<EncodedBatch> encodeAsync(
      List<DecisionContext> contexts, PolicyExecutionContext execution) {
    return inputEncoder.encodeAsync(contexts, execution);
  }

  /** 収集側が借用した入力領域を使う。呼び出し側は符号化・推論・保存の全処理が終了してから領域を返却する。 返された入力を借用期間の外へ持ち出してはならない。 */
  public CompletableFuture<EncodedBatch> encodeAsync(
      List<DecisionContext> contexts,
      PolicyExecutionContext execution,
      DecisionInferenceStoragePool.Lease storage) {
    return inputEncoder.encodeAsync(contexts, execution, storage);
  }

  @Override
  public List<Prediction> evaluate(List<DecisionContext> contexts) {
    var encoded = encode(contexts);
    return evaluateEncoded(encoded)
        .withObservations(CfrInputEncoder.encodePayloads(encoded, null).join());
  }

  /** 保存済みの観測を同じ容量区分ごとにまとめて推論し、入力順の方策を返す。 */
  public List<Prediction> evaluateObservations(List<ObservationPayload> observations) {
    return evaluateEncoded(inputEncoder.decode(observations)).withObservations(observations);
  }

  public synchronized Policies evaluateEncoded(EncodedBatch encoded) {
    ensureOpen();
    double[][] policies = encoded.policies();
    for (EncodedGroup group : encoded.groups) {
      if (group.forced) continue;
      for (int from = 0; from < group.host.size(); from += config.inferenceBatchSize()) {
        var rows =
            group.host.sliceRows(
                from, Math.min(group.host.size() - from, config.inferenceBatchSize()));
        float[][] scores = predictScores(rows);
        for (int row = 0; row < rows.size(); row++) {
          int index = group.indices[from + row];
          int count = group.host.legalActionCount(from + row);
          policies[index] = policy(scores[row], count, role);
        }
      }
    }
    return new Policies(policies);
  }

  /** 符号付きスコアまたはロジットをそのまま返す。検証データの回帰評価や、保存前後のモデルの一致確認に使う。 */
  public synchronized float[] rawScores(ObservationPayload encodedObservation) {
    ensureOpen();
    var row = CfrObservationCodec.decode(encodedObservation);
    float[] scores = predictScores(row.host().sliceRows(0, 1))[0];
    return Arrays.copyOf(scores, row.host().legalActionCount(0));
  }

  private float[][] predictScores(DecisionHostBatch.RowSlice rows) {
    awaitModelReady();
    try (NDManager scope = model.getNDManager().newSubManager()) {
      var batch =
          DecisionBatchTransfer.transferTrainingToDevice(
              scope, rows, inferenceTransfer, config.tensorTransfer());
      NDArray scores;
      try (Autocast ignored = EpsilonDecisionAutocast.open(scope, config.precision(device()))) {
        var parameters = new ParameterStore(scope, false);
        scores = network.scores(parameters, batch, false);
      }
      float[] flat = scores.toType(DataType.FLOAT32, false).toFloatArray();
      int capacity = rows.bucket().legalActionCapacity();
      float[][] result = new float[rows.size()][capacity];
      for (int row = 0; row < rows.size(); row++)
        System.arraycopy(flat, row * capacity, result[row], 0, capacity);
      return result;
    }
  }

  static double[] policy(float[] scores, int count, Role role) {
    double[] result = new double[count];
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < count; i++) {
      if (!Float.isFinite(scores[i])) throw new IllegalStateException("Nonfinite model score");
      max = Math.max(max, scores[i]);
    }
    double sum = 0;
    for (int i = 0; i < count; i++) {
      result[i] = role == Role.REGRET ? Math.max(0, scores[i]) : Math.exp(scores[i] - max);
      sum += result[i];
    }
    if (sum == 0) Arrays.fill(result, 1d / count);
    else for (int i = 0; i < count; i++) result[i] /= sum;
    return result;
  }

  /** 重み付きサンプラーが選んだ行を使い、共通の {@link DataParallelGroup} でモデルを一回更新する。 */
  public synchronized TrainingMetrics train(List<CfrTrainingRow> rows) {
    ensureOpen();
    if (!trainable) throw new IllegalStateException("Frozen deployment model cannot be trained");
    TrainingMetrics metrics = learner().train(rows, Math.addExact(updates, 1));
    updates = metrics.updates();
    return metrics;
  }

  private CfrLearner learner() {
    if (learner == null) learner = new CfrLearner(config, role, model, network, devices);
    return learner;
  }

  public synchronized void saveModel(Path directory) throws IOException {
    save(directory, false);
  }

  /**
   * 学習前のモデルへNanoの状態と合法候補の表現を移す。呼び出し側は変換中に入力モデルを更新しないこと。
   *
   * <p>Nanoの条件付き方策グラフはCFRの平坦な出力と異なるため、出力はゼロのまま残す。{@code REGRET} は累積後悔ゼロから開始する。 {@code AVERAGE}
   * へ元の方策も引き継ぐ場合は、観測に対する全合法手の分布で別途蒸留する。CFR固有の出力層とAdamWの状態は初期値を保つ。
   */
  public synchronized NanoInitializationReport initializeRepresentationFromNano(Model source) {
    ensureOpen();
    Objects.requireNonNull(source);
    if (!trainable || updates != 0 || learner != null)
      throw new IllegalStateException("Nano initialization requires a fresh CFR model");
    NanoInitializationReport report = network.initializeRepresentationFromNano(source.getBlock());
    completeParameterPublication();
    return report;
  }

  /** 新しいGRPを使う学習区間で平均方策の表現を再利用する。出力層と最適化履歴は引き継がない。 */
  public void initializeRepresentationFrom(CfrModel source) {
    Objects.requireNonNull(source);
    if (source == this) throw new IllegalArgumentException("Cannot initialize a model from itself");
    synchronized (CfrModel.class) {
      synchronized (source) {
        synchronized (this) {
          ensureOpen();
          source.ensureOpen();
          awaitModelReady();
          source.awaitModelReady();
          if (!trainable || updates != 0 || role != Role.REGRET || source.role != Role.AVERAGE)
            throw new IllegalStateException(
                "Representation initialization requires a fresh regret model and an"
                    + " average model");
          if (config.hiddenSize() != source.config.hiddenSize())
            throw new IllegalArgumentException("CFR representation dimensions differ");
          DataParallelGroup.copyParameters(
              source.model,
              model,
              name -> !network.isOutputParameter(network.getParameters().get(name)));
          completeParameterPublication();
          if (learner != null) learner.synchronizeParameters();
        }
      }
    }
  }

  public synchronized void saveTraining(Path directory) throws IOException {
    if (!trainable) throw new IllegalStateException("Frozen model has no training state");
    save(directory, true);
  }

  private void save(Path directory, boolean trainingState) throws IOException {
    ensureOpen();
    awaitModelReady();
    Files.createDirectories(directory);
    try (var out =
        new DataOutputStream(
            new BufferedOutputStream(Files.newOutputStream(directory.resolve("weights.bin"))))) {
      network.saveParameters(out);
    }
    if (trainingState) learner().saveOptimizer(directory.resolve("optimizer.bin"));
    Files.writeString(
        directory.resolve("model.json"),
        GSON.toJson(
            new Metadata(
                3,
                CfrNetwork.ARCHITECTURE,
                DecisionInputSchema.fingerprint(),
                role,
                config,
                updates,
                trainingState)));
  }

  public static CfrModel loadModel(Path directory, Device device) throws IOException {
    return load(directory, NetworkDevices.of(device), false);
  }

  public static CfrModel loadTraining(Path directory, Device device) throws IOException {
    return load(directory, NetworkDevices.of(device), true);
  }

  public static CfrModel loadTraining(Path directory, NetworkDevices devices) throws IOException {
    return load(directory, devices, true);
  }

  private static CfrModel load(Path directory, NetworkDevices devices, boolean training)
      throws IOException {
    Metadata metadata;
    try {
      metadata = GSON.fromJson(Files.readString(directory.resolve("model.json")), Metadata.class);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid CFR model metadata", invalid);
    }
    if (metadata == null
        || metadata.version() != 3
        || !CfrNetwork.ARCHITECTURE.equals(metadata.architecture())
        || !DecisionInputSchema.fingerprint().equals(metadata.inputSchema())
        || metadata.role() == null
        || metadata.config() == null
        || metadata.updates() < 0) throw new IOException("CFR model architecture/schema mismatch");
    if (training && !metadata.trainingState())
      throw new IOException("Deployment artifact has no optimizer state");
    CfrModel result = new CfrModel(metadata.config(), metadata.role(), devices);
    try {
      try (var in =
          new DataInputStream(
              new BufferedInputStream(Files.newInputStream(directory.resolve("weights.bin"))))) {
        result.network.loadParameters(result.model.getNDManager(), in);
        if (in.read() != -1) throw new IOException("Trailing CFR weight data");
      }
      if (training) result.learner().loadOptimizer(directory.resolve("optimizer.bin"));
      result.updates = metadata.updates();
      result.trainable = training;
      result.setTrainableParameters(training);
      if (!training) result.initializeFrozenFeatures();
      result.completeParameterPublication();
      return result;
    } catch (MalformedModelException failure) {
      result.close();
      throw new IOException("Invalid CFR model weights", failure);
    } catch (IOException | RuntimeException | Error failure) {
      result.close();
      throw failure;
    }
  }

  /** 並行する収集ワーカーで共有できる、独立した変更不可のパラメータースナップショットを作る。 */
  public synchronized CfrModel copyForInference(Device destination) {
    ensureOpen();
    awaitModelReady();
    CfrModel result = new CfrModel(config, role, destination);
    try {
      DataParallelGroup.copyParameters(model, result.model);
      result.updates = updates;
      result.trainable = false;
      result.setTrainableParameters(false);
      result.initializeFrozenFeatures();
      result.completeParameterPublication();
      return result;
    } catch (RuntimeException | Error failure) {
      result.close();
      throw new IllegalStateException("Cannot create frozen CFR snapshot", failure);
    }
  }

  private void awaitModelReady() {
    if (learner != null) learner.awaitModelReady();
  }

  private void initializeFrozenFeatures() {
    if (!device().isGpu()) return;
    DataType type =
        switch (config.precision(device())) {
          case FLOAT32 -> DataType.FLOAT32;
          case FLOAT16 -> DataType.FLOAT16;
          case BFLOAT16 -> DataType.BFLOAT16;
        };
    if (type != DataType.FLOAT32)
      frozenFeatures = network.frozenFeatures(model.getNDManager(), type);
  }

  /** 初期化・復元・コピーを終えてから、凍結モデルを任意の推論ストリームへ公開する。 */
  private void completeParameterPublication() {
    if (device().isGpu()) {
      try (var completion = ((PtEngine) model.getNDManager().getEngine()).newEvent(device())) {
        completion.record();
        completion.synchronize();
      }
    }
  }

  private void ensureOpen() {
    if (closed) throw new IllegalStateException("CFR model is closed");
  }

  private void setTrainableParameters(boolean enabled) {
    for (Parameter parameter : network.getParameters().values()) {
      parameter.freeze(!enabled);
      parameter.getArray().setRequiresGradient(enabled);
    }
  }

  @Override
  public synchronized void close() {
    if (closed) return;
    closed = true;
    try {
      if (learner != null) learner.close();
    } finally {
      try {
        inferenceTransfer.close();
      } finally {
        try {
          if (frozenFeatures != null) frozenFeatures.close();
        } finally {
          model.close();
        }
      }
    }
  }
}
