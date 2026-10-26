package com.epsilon.prism.model;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.nano.ai.model.EpsilonDecisionFrozenFeatures;
import com.epsilon.network.NetworkDevices;
import com.epsilon.network.NetworkRuntime;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * 生徒 π_θ の網（{@link PrismNetwork}。s(o, a) を出し、打つときは argmax）。学習用は {@link #fit} で当てはめ（差分整合 {@link PrismLoss} か蒸留
 * {@link Distill}）、推論用は凍結して読み（{@link #load}）、収集・評価で共有する。
 *
 * <p>保存するのは重み（weights.bin）と model.json だけで、最適化器の状態は持たない。
 */
public final class PrismModel implements AutoCloseable {
  private record Metadata(
      int version, String architecture, String inputSchema, ModelConfig config) {}

  private static final int FORMAT = 1;
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private final ModelConfig config;
  private final Model model;
  private final PrismNetwork network;
  private final NetworkDevices devices;
  private final DecisionBatchTransfer.Workspace inferenceTransfer;
  private EpsilonDecisionFrozenFeatures frozenFeatures;
  private boolean closed;

  /** 新しい網。初期値は config の種で決まる（出力層はゼロ）。 */
  public PrismModel(ModelConfig config, String name, NetworkDevices devices) {
    this.config = config;
    this.devices = devices;
    // エンジンを初めて取る前に、全モデル共通の実行既定値（PyTorch の CPU スレッド数）を決める
    NetworkRuntime.initialize();
    model = Model.newInstance("epsilon-prism-" + name, devices.primary(), "PyTorch");
    network = new PrismNetwork(config);
    model.setBlock(network);
    synchronized (PrismModel.class) {
      model.getNDManager().getEngine().setRandomSeed(Long.hashCode(config.seed()));
      network.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, 1));
    }
    inferenceTransfer = new DecisionBatchTransfer.Workspace(model.getNDManager());
    completeParameterPublication();
  }

  /** 微小バッチ: 結合した入力と、その行（{@link Fit#pack}）。 */
  public record Part<E extends Example>(DecisionHostBatch host, List<E> rows) {}

  /**
   * 結合済みの 1 歩分（レーンごとの微小バッチ）。結合はホスト側だけなので、学習とは別のスレッドで先に作れる。
   *
   * @param rows 損失の分母の行数（{@link Objective#contributes} が真の行の数。なければ 1）
   */
  public record Packed<E extends Example>(List<List<Part<E>>> lanes, int rows) {}

  public ModelConfig config() {
    return config;
  }

  public Device device() {
    return model.getNDManager().getDevice();
  }

  PrismNetwork network() {
    return network;
  }

  NDManager manager() {
    return model.getNDManager();
  }

  /** BF16 などの凍結推論用の特徴。FP32 と CPU では null。 */
  EpsilonDecisionFrozenFeatures frozenFeatures() {
    return frozenFeatures;
  }

  /**
   * 1 回の当てはめを始める。AdamW はここで新しく作り、学習率は schedule（当てはめの最初の更新を 1 とする番号）で決まる。
   *
   * @param gradientClip 1 歩の勾配のノルムの上限（0 なら切らない。model.json の gradientClip は使わない）
   * @param objective 損失（{@link PrismLoss} か {@link Distill}）
   */
  public synchronized <E extends Example> Fit<E> fit(
      LearningRateSchedule schedule, float gradientClip, Objective<E> objective) {
    return new Fit<>(
        new PrismLearner<>(config, model, network, devices, schedule, gradientClip, objective));
  }

  /** 当てはめ。閉じると更新の完了を待って、学習器（AdamW とデータ並列の複製）を捨てる。重みはこの網に残る。 */
  public final class Fit<E extends Example> implements AutoCloseable {
    private final PrismLearner<E> learner;

    private Fit(PrismLearner<E> learner) {
      this.learner = learner;
    }

    /** 1 歩分の行をレーンごとに割り当てて結合する（ホスト側だけ。先読みのスレッドから呼んでよい）。 */
    public Packed<E> pack(List<E> rows) {
      return learner.pack(rows);
    }

    /** 結合済みの 1 歩分で 1 回の更新。返り値はその更新の前の損失（回収しない歩は NaN）。 */
    public double train(Packed<E> packed) {
      synchronized (PrismModel.this) {
        return learner.train(packed);
      }
    }

    /** 1 回の更新（結合を含む）。 */
    public double train(List<E> rows) {
      return train(pack(rows));
    }

    /** 段階ごとの秒数 {行の結合, 転送, 順伝播と逆伝播, 最適化}（報告だけ）。 */
    public double[] stageSeconds() {
      synchronized (PrismModel.this) {
        return learner.stageSeconds();
      }
    }

    /** これまでの更新のうち、勾配のノルムを上限で切った歩数の見積もり（報告だけ）。 */
    public int clippedSteps() {
      synchronized (PrismModel.this) {
        return learner.clippedSteps();
      }
    }

    @Override
    public void close() {
      synchronized (PrismModel.this) {
        learner.close();
      }
    }
  }

  /**
   * 1 バッチ分の s。行ごとに合法手の数だけ返す。学習と同じ計算経路を推論モードで通す（勾配の記録はスレッドごとの設定なので、呼ぶスレッドに依らない
   * ようにする）。
   */
  public synchronized float[][] scores(DecisionHostBatch host) {
    float[][] result = new float[host.size()][];
    for (int from = 0; from < host.size(); from += config.inferenceBatchSize()) {
      int count = Math.min(host.size() - from, config.inferenceBatchSize());
      var rows = host.sliceRows(from, count);
      try (NDManager scope = model.getNDManager().newSubManager()) {
        var batch =
            DecisionBatchTransfer.transferTrainingToDevice(
                scope, rows, inferenceTransfer, config.transfer(device()));
        float[] flat;
        try (var inference = scope.getEngine().newInferenceMode();
            Autocast ignored = EpsilonDecisionAutocast.open(scope, config.precision(device()))) {
          flat =
              network
                  .scores(new ParameterStore(scope, false), batch, false, new PairList<>())
                  .toType(DataType.FLOAT32, false)
                  .toFloatArray();
        }
        int capacity = rows.bucket().legalActionCapacity();
        for (int row = 0; row < rows.size(); row++) {
          float[] values = new float[host.legalActionCount(from + row)];
          System.arraycopy(flat, row * capacity, values, 0, values.length);
          result[from + row] = values;
        }
      }
    }
    return result;
  }

  /** まとめた入力の s。返り値の i 番目は元の並びの i 番目の行。 */
  public float[][] scores(EncodedRows rows) {
    float[][] result = new float[rows.size()][];
    for (EncodedRows.Group group : rows.groups()) {
      float[][] values = scores(group.host());
      for (int row = 0; row < values.length; row++) result[group.indices()[row]] = values[row];
    }
    return result;
  }

  /** 1 行ずつの入力（復号した payload）の s。容量区分ごとにまとめて inferenceBatchSize ずつ計算する。返り値の i 番目は hosts の i 番目の行。 */
  public float[][] scores(List<DecisionHostBatch> hosts) {
    return scores(EncodedRows.of(hosts));
  }

  /** 重み（weights.bin）と model.json を書く。 */
  public synchronized void save(Path directory) throws IOException {
    Files.createDirectories(directory);
    try (var out =
        new DataOutputStream(
            new BufferedOutputStream(Files.newOutputStream(directory.resolve("weights.bin"))))) {
      network.saveParameters(out);
    }
    Files.writeString(
        directory.resolve("model.json"),
        GSON.toJson(
            new Metadata(
                FORMAT, PrismNetwork.ARCHITECTURE, DecisionInputSchema.fingerprint(), config)));
  }

  /**
   * 保存した網を読む（構成は保存した model.json のもの）。
   *
   * @param training true なら学習用（devices の先頭が正本で、残りはデータ並列の複製）、false なら推論用の凍結モデル
   */
  public static PrismModel load(Path directory, NetworkDevices devices, boolean training)
      throws IOException {
    return load(directory, readConfig(directory), devices, training);
  }

  /** 保存した網の構成（model.json の config。形式・網の形式・入力の版を確かめる）。 */
  public static ModelConfig readConfig(Path directory) throws IOException {
    Metadata metadata;
    try {
      metadata = GSON.fromJson(Files.readString(directory.resolve("model.json")), Metadata.class);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid Prism model metadata " + directory, invalid);
    }
    if (metadata == null
        || metadata.version() != FORMAT
        || metadata.config() == null
        || !PrismNetwork.ARCHITECTURE.equals(metadata.architecture())
        || !DecisionInputSchema.fingerprint().equals(metadata.inputSchema()))
      throw new IOException("Prism model format/architecture/schema mismatch " + directory);
    return metadata.config();
  }

  /**
   * 保存した網の重みを、構成 config の網に読む（run の設定で版 1 を作り直すとき。バッチ・精度・転送・種などは config のものになる）。網の幅は保存した
   * 網と同じでなければならない。
   *
   * @param training true なら学習用、false なら推論用の凍結モデル
   */
  public static PrismModel load(
      Path directory, ModelConfig config, NetworkDevices devices, boolean training)
      throws IOException {
    int saved = readConfig(directory).hiddenSize();
    if (config.hiddenSize() != saved)
      throw new IOException(
          "Prism model width " + saved + " differs from " + config.hiddenSize() + ": " + directory);
    var result = new PrismModel(config, directory.getFileName().toString(), devices);
    try {
      try (var in =
          new DataInputStream(
              new BufferedInputStream(Files.newInputStream(directory.resolve("weights.bin"))))) {
        result.readParameters(in);
        if (in.read() != -1) throw new IOException("Trailing Prism weight data " + directory);
      }
      result.prepare(training);
      return result;
    } catch (IOException | RuntimeException | Error failure) {
      result.close();
      throw failure;
    }
  }

  /**
   * この網の重みを、同じ形の網 other の重みと混ぜる: w ← (1 − eta)·w + eta·w_other（要素ごと、FP32）。eta = 1 なら other の写し、eta = 0 なら
   * 何もしない。方策の記憶（{@link com.epsilon.prism.run.Averaging}）と版の重みの平均に使う。
   *
   * @throws IllegalArgumentException eta が [0, 1] の外か、パラメーターの名前か形が違う
   */
  public synchronized void mixIn(PrismModel other, double eta) {
    if (!(eta >= 0 && eta <= 1)) throw new IllegalArgumentException("eta must be in [0, 1]: " + eta);
    if (eta == 0) return;
    PairList<String, Parameter> target = network.getParameters();
    PairList<String, Parameter> source = other.network.getParameters();
    if (source.size() != target.size())
      throw new IllegalArgumentException("Models have different parameter lists");
    for (int i = 0; i < target.size(); i++) {
      String name = target.keyAt(i);
      var destination = target.valueAt(i).getArray();
      Parameter parameter = source.get(name);
      if (parameter == null || !parameter.getArray().getShape().equals(destination.getShape()))
        throw new IllegalArgumentException("Parameter " + name + " differs between models");
      float[] values = parameter.getArray().toType(DataType.FLOAT32, false).toFloatArray();
      if (eta < 1) {
        float[] own = destination.toType(DataType.FLOAT32, false).toFloatArray();
        for (int k = 0; k < values.length; k++)
          values[k] = (float) ((1 - eta) * own[k] + eta * values[k]);
      }
      destination.set(values);
    }
    completeParameterPublication();
  }

  /**
   * この網の重みを、同じ形の網 models の重みの要素ごとの平均で置き換える（{@link #mixIn} を η = 1/k で順に重ねる）。
   *
   * @throws IllegalArgumentException models が空か、パラメーターの名前か形が違う
   */
  public synchronized void setToMean(List<PrismModel> models) {
    if (models.isEmpty()) throw new IllegalArgumentException("No models to average");
    for (int k = 0; k < models.size(); k++) mixIn(models.get(k), 1.0 / (k + 1));
  }

  /** 重みの列 in の先頭から、この網のパラメーターを読む（列の続きは読まない）。 */
  public synchronized void readParameters(DataInputStream in) throws IOException {
    try {
      network.loadParameters(model.getNDManager(), in);
    } catch (MalformedModelException invalid) {
      throw new IOException("Invalid Prism model weights", invalid);
    }
    completeParameterPublication();
  }

  /** 読み込んだ（か写した）重みを学習用（勾配を要求する）か推論用（凍結）にする。 */
  synchronized void prepare(boolean training) {
    if (training) requireGradients(true);
    else freeze();
    completeParameterPublication();
  }

  private void freeze() {
    requireGradients(false);
    initializeFrozenFeatures();
  }

  /** 読み込んだ重みは勾配を要求しないので、学習用なら要求させる。 */
  private void requireGradients(boolean enabled) {
    for (Parameter parameter : network.getParameters().values()) {
      parameter.freeze(!enabled);
      parameter.getArray().setRequiresGradient(enabled);
    }
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

  /** 初期化・復元・書き換えを終えてから、網を任意の推論ストリームへ公開する。 */
  void completeParameterPublication() {
    if (device().isGpu()) {
      try (var completion = ((PtEngine) model.getNDManager().getEngine()).newEvent(device())) {
        completion.record();
        completion.synchronize();
      }
    }
  }

  @Override
  public synchronized void close() {
    if (closed) return;
    closed = true;
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
