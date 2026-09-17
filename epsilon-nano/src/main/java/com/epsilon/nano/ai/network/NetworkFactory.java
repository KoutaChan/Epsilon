package com.epsilon.nano.ai.network;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpNetwork;
import com.epsilon.config.settings.BeliefSettings;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.GrpSettings;
import com.epsilon.nano.ai.belief.EpsilonBeliefNetwork;
import com.epsilon.nano.ai.model.EpsilonDecisionNetwork;
import com.epsilon.nano.config.settings.EpsilonSettings;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Decision 経路で使うモデルとデバイスのファクトリ。 */
public final class NetworkFactory {

  private static final Logger log = LoggerFactory.getLogger(NetworkFactory.class);

  static {
    // 小さいGRPのCPU演算でスレッド過多を避け、明示されたDJL/OpenMP/MKL設定は優先する。
    String ompThreads = System.getenv("OMP_NUM_THREADS");
    String mklThreads = System.getenv("MKL_NUM_THREADS");
    if (System.getProperty("ai.djl.pytorch.num_threads") == null
        && (ompThreads == null || ompThreads.isBlank())
        && (mklThreads == null || mklThreads.isBlank())) {
      System.setProperty("ai.djl.pytorch.num_threads", "1");
    }
  }

  private NetworkFactory() {}

  public static NetworkDevices getLearnerDevices(DeviceSettings settings) {
    return resolveDevices(
        new DeviceSelection("learner", settings.learner(), settings.validateIndices()),
        () -> getPreferredDevices(0));
  }

  public static NetworkDevices getInferenceDevices(DeviceSettings settings) {
    return resolveDevices(
        new DeviceSelection("inference", settings.inference(), settings.validateIndices()),
        () -> getPreferredDevices(0));
  }

  public static NetworkDevices getGrpDevices(DeviceSettings settings) {
    return resolveDevices(
        new DeviceSelection("grp", settings.grp(), settings.validateIndices()),
        () -> getPreferredDevices(0));
  }

  public static Device getLearnerDevice(DeviceSettings settings) {
    return getLearnerDevices(settings).primary();
  }

  /**
   * 利用可能な先頭 GPU、なければ CPU を返す。
   *
   * @return 単一モデルの既定デバイス
   */
  public static Device getPreferredDevice() {
    return getPreferredDevices(1).primary();
  }

  /**
   * 学習器設定で解決した先頭デバイスを返す。
   *
   * @return 主となる学習器デバイス
   */
  public static Device getLearnerDevice() {
    return getLearnerDevices().primary();
  }

  /**
   * 学習器用に設定された全デバイスを返す。
   *
   * @return データ並列学習器デバイス集合
   */
  public static NetworkDevices getLearnerDevices() {
    return resolveDevices(DeviceSelection.learner(), () -> getPreferredDevices(0));
  }

  /**
   * Decision 推論モデルの複製用デバイスを返す。
   *
   * @return 推論デバイス集合
   */
  public static NetworkDevices getInferenceDevices() {
    return resolveDevices(DeviceSelection.inference(), () -> getPreferredDevices(0));
  }

  /**
   * GRP 推論・学習用デバイスを返す。
   *
   * @return GRP デバイス集合
   */
  public static NetworkDevices getGrpDevices() {
    return resolveDevices(DeviceSelection.grp(), () -> getPreferredDevices(0));
  }

  /**
   * バックエンドが認識する GPU をインデックス順に返し、GPU がなければ CPU へ代替処理する。
   *
   * @param maxGpus GPU 数の上限。0 以下なら全 GPU
   * @return 1要素以上のデバイス集合
   */
  public static NetworkDevices getPreferredDevices(int maxGpus) {
    {
      Engine engine = Engine.getInstance();
      int gpuCount = engine.getGpuCount();
      if (gpuCount > 0) {
        int limit = maxGpus > 0 ? Math.min(gpuCount, maxGpus) : gpuCount;
        ArrayList<Device> devices = new ArrayList<>(limit);
        for (int i = 0; i < limit; i++) {
          devices.add(Device.gpu(i));
        }
        return NetworkDevices.copyOf(devices);
      }
      log.warn("GPU count is {}", gpuCount);
    }
    return NetworkDevices.of(Device.cpu());
  }

  /**
   * 既定デバイスと標準構造で初期化済み Decision モデルを作る。
   *
   * @return 呼び出し側が解放する Decision モデル
   */
  public static Model createDecisionModel() {
    return createDecisionModel(getPreferredDevice());
  }

  /**
   * 指定デバイスと標準構造で初期化済み Decision モデルを作る。
   *
   * @param device モデルパラメーターを配置するデバイス
   * @return 呼び出し側が解放する Decision モデル
   */
  public static Model createDecisionModel(Device device) {
    return createDecisionModel(device, true);
  }

  /**
   * 指定デバイスに標準 Decision モデルを作り、必要なら構造を記録する。
   *
   * @param device モデルパラメーターを配置するデバイス
   * @param logArchitecture 構造識別子をログするなら {@code true}
   * @return 初期化済み Decision モデル
   */
  public static Model createDecisionModel(Device device, boolean logArchitecture) {
    return createDecisionModel(device, logArchitecture, new EpsilonDecisionNetwork());
  }

  /**
   * 指定した隠れ層幅の Decision モデルを作る。主に複製用モデル構築で使用する。
   *
   * @param device モデルパラメーターを配置するデバイス
   * @param logArchitecture 構造識別子をログするなら {@code true}
   * @param hiddenSize 状態・候補埋め込みの幅
   * @return 初期化済み Decision モデル
   */
  public static Model createDecisionModel(Device device, boolean logArchitecture, int hiddenSize) {
    return createDecisionModel(device, logArchitecture, new EpsilonDecisionNetwork(hiddenSize));
  }

  /** 入力元モデルの効用定義を保持するモデルの複製用の明示的な生成境界。 */
  public static Model createDecisionModel(
      Device device,
      boolean logArchitecture,
      int hiddenSize,
      EpsilonUtilityProfile utilityProfile) {
    return createDecisionModel(
        device, logArchitecture, new EpsilonDecisionNetwork(hiddenSize, utilityProfile));
  }

  private static Model createDecisionModel(
      Device device, boolean logArchitecture, EpsilonDecisionNetwork block) {
    Model model = Model.newInstance("epsilon_decision", device, "PyTorch");
    model.setBlock(block);
    if (logArchitecture) {
      log.info(
          "Decision architecture: {} valueDefinition={} device={}",
          EpsilonDecisionNetwork.architectureSummary(block.hiddenSize()),
          EpsilonDecisionHlGauss.fingerprint(block.utilityProfile()),
          device);
    }
    NDManager manager = model.getNDManager();
    block.initialize(
        manager,
        DataType.FLOAT32,
        new Shape(-1, com.epsilon.nano.ai.decision.input.DecisionInputSchema.STATE_INT_COUNT),
        new Shape(-1, -1, com.epsilon.nano.ai.decision.input.DecisionInputSchema.ACTION_INT_STRIDE),
        new Shape(
            -1,
            -1,
            -1,
            com.epsilon.nano.ai.decision.input.DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE),
        new Shape(
            -1,
            -1,
            -1,
            com.epsilon.nano.ai.decision.input.DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT),
        new Shape(
            -1,
            -1,
            -1,
            com.epsilon.nano.ai.decision.input.DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT,
            com.epsilon.nano.ai.decision.input.DecisionInputSchema
                .ACTION_TRANSITION_WAIT_YAKU_STRIDE),
        new Shape(-1, com.epsilon.nano.ai.decision.input.DecisionInputSchema.STATE_FLOAT_COUNT),
        new Shape(
            -1, -1, com.epsilon.nano.ai.decision.input.DecisionInputSchema.ACTION_FLOAT_STRIDE),
        new Shape(
            -1,
            -1,
            -1,
            com.epsilon.nano.ai.decision.input.DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE),
        new Shape(
            -1,
            -1,
            -1,
            com.epsilon.nano.ai.decision.input.DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT,
            com.epsilon.nano.ai.decision.input.DecisionInputSchema
                .ACTION_TRANSITION_WAIT_FLOAT_STRIDE));
    return model;
  }

  /**
   * 指定デバイスに初期化済み GRP モデルを作る。
   *
   * @param device モデルパラメーターを配置するデバイス
   * @return 呼び出し側が解放する GRP モデル
   */
  public static Model createGrpModel(Device device) {
    var config = EpsilonSettings.defaults().bind(GrpSettings.class);
    return createGrpModel(device, config.hidden(), config.layers());
  }

  public static Model createGrpModel(Device device, int hidden, int layers) {
    Model model = Model.newInstance("epsilon_grp", device, "PyTorch");
    EpsilonGrpNetwork block = new EpsilonGrpNetwork(hidden, layers);
    model.setBlock(block);
    log.info(
        "GRP architecture: {} device={}",
        EpsilonGrpNetwork.architectureSummary(hidden, layers),
        device);
    NDManager manager = model.getNDManager();
    block.initialize(
        manager,
        DataType.FLOAT32,
        new Shape(-1, -1, EpsilonGrpFeature.FEATURE_SIZE),
        new Shape(-1));
    return model;
  }

  /**
   * 既定デバイスに初期化済み Belief モデルを作る。
   *
   * @return 呼び出し側が解放する Belief モデル
   */
  public static Model createBeliefModel() {
    return createBeliefModel(getPreferredDevice());
  }

  /**
   * 指定デバイスに初期化済み Belief モデルを作る。
   *
   * @param device モデルパラメーターを配置するデバイス
   * @return 呼び出し側が解放する Belief モデル
   */
  public static Model createBeliefModel(Device device) {
    return createBeliefModel(
        device, EpsilonSettings.defaults().bind(BeliefSettings.class).hidden());
  }

  public static Model createBeliefModel(Device device, int hidden) {
    Model model = Model.newInstance("epsilon_belief", device, "PyTorch");
    EpsilonBeliefNetwork block = new EpsilonBeliefNetwork(hidden);
    model.setBlock(block);
    log.info(
        "Belief architecture: {} device={}",
        EpsilonBeliefNetwork.architectureSummary(hidden),
        device);
    NDManager manager = model.getNDManager();
    block.initialize(
        manager,
        DataType.FLOAT32,
        new Shape(-1, com.epsilon.nano.ai.decision.input.DecisionInputSchema.STATE_INT_COUNT),
        new Shape(-1, com.epsilon.nano.ai.decision.input.DecisionInputSchema.STATE_FLOAT_COUNT));
    return model;
  }

  private static NetworkDevices resolveDevices(
      DeviceSelection selection, Supplier<NetworkDevices> fallbackSupplier) {
    String deviceSpec = selection.deviceSpec().trim();
    if (deviceSpec.isEmpty()) {
      return fallbackSupplier.get();
    }
    String normalized = deviceSpec.toLowerCase(Locale.ROOT);
    if ("auto".equals(normalized) || "all".equals(normalized)) {
      return getPreferredDevices(0);
    }

    int gpuCount = selection.validateGpuIndices() ? detectedGpuCount() : -1;
    List<Device> devices = new ArrayList<>();
    for (String rawToken : deviceSpec.split(",", -1)) {
      String token = rawToken.trim();
      devices.add(parseDeviceToken(selection.role(), token, gpuCount));
    }
    return NetworkDevices.copyOf(devices);
  }

  private static Device parseDeviceToken(String role, String token, int gpuCount) {
    String normalized = token.toLowerCase(Locale.ROOT);
    if ("cpu".equals(normalized)) {
      return Device.cpu();
    }
    if (normalized.startsWith("gpu:")) {
      normalized = normalized.substring("gpu:".length());
    } else if (normalized.startsWith("gpu")) {
      normalized = normalized.substring("gpu".length());
    } else if (normalized.startsWith("cuda:")) {
      normalized = normalized.substring("cuda:".length());
    }
    int gpuIndex;
    try {
      gpuIndex = Integer.parseInt(normalized);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException("Invalid " + role + " device: " + token, e);
    }
    if (gpuIndex < 0) {
      throw new IllegalArgumentException("Negative GPU index in " + role + " devices: " + token);
    }
    if (gpuCount == 0) {
      throw new IllegalArgumentException(
          "GPU configured for " + role + " but no GPU is visible: " + token);
    }
    if (gpuCount > 0 && gpuIndex >= gpuCount) {
      throw new IllegalArgumentException(
          "GPU index outside visible range in "
              + role
              + " devices: token="
              + token
              + " visibleGpus="
              + gpuCount);
    }
    return Device.gpu(gpuIndex);
  }

  private static int detectedGpuCount() {
    return Engine.getInstance().getGpuCount();
  }

  private record DeviceSelection(String role, String deviceSpec, boolean validateGpuIndices) {

    private static DeviceSelection learner() {
      return configured("learner", DeviceSettings::learner);
    }

    private static DeviceSelection inference() {
      return configured("inference", DeviceSettings::inference);
    }

    private static DeviceSelection grp() {
      return configured("grp", DeviceSettings::grp);
    }

    private static DeviceSelection configured(
        String role, Function<DeviceSettings, String> deviceSpecReader) {
      DeviceSettings settings = EpsilonSettings.defaults().bind(DeviceSettings.class);
      return new DeviceSelection(
          role, deviceSpecReader.apply(settings), settings.validateIndices());
    }
  }
}
