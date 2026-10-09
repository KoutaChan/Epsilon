package com.epsilon.major.ai.network;

import ai.djl.Device;
import ai.djl.Model;
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
import com.epsilon.major.ai.belief.EpsilonBeliefNetwork;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.epsilon.major.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.major.ai.model.EpsilonDecisionNetwork;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.epsilon.network.NetworkDeviceResolver;
import com.epsilon.network.NetworkDevices;
import com.epsilon.network.NetworkRuntime;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Decision 経路で使うモデルとデバイスのファクトリ。 */
public final class NetworkFactory {

  private static final Logger log = LoggerFactory.getLogger(NetworkFactory.class);

  static {
    NetworkRuntime.initialize();
  }

  private NetworkFactory() {}

  public static NetworkDevices getLearnerDevices(DeviceSettings settings) {
    return NetworkDeviceResolver.learner(settings);
  }

  public static NetworkDevices getInferenceDevices(DeviceSettings settings) {
    return NetworkDeviceResolver.inference(settings);
  }

  public static NetworkDevices getGrpDevices(DeviceSettings settings) {
    return NetworkDeviceResolver.grp(settings);
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
   * @return 主デバイス学習器デバイス
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
    return getLearnerDevices(EpsilonSettings.defaults().bind(DeviceSettings.class));
  }

  /**
   * Decision 推論モデルの複製用デバイスを返す。
   *
   * @return 推論デバイス集合
   */
  public static NetworkDevices getInferenceDevices() {
    return getInferenceDevices(EpsilonSettings.defaults().bind(DeviceSettings.class));
  }

  /**
   * GRP 推論・学習用デバイスを返す。
   *
   * @return GRP デバイス集合
   */
  public static NetworkDevices getGrpDevices() {
    return getGrpDevices(EpsilonSettings.defaults().bind(DeviceSettings.class));
  }

  /**
   * 実行基盤が認識する GPU をインデックス順に返し、GPU がなければ CPU へ代替処理する。
   *
   * @param maxGpus GPU 数の上限。0 以下なら全 GPU
   * @return 1要素以上のデバイス集合
   */
  public static NetworkDevices getPreferredDevices(int maxGpus) {
    return NetworkDeviceResolver.preferred(maxGpus);
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
   * 指定した隠れ層幅の Decision モデルを作る。主に複製モデル構築で使用する。
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
    block.initialize(manager, DataType.FLOAT32, DecisionNetworkInputs.initializationShapes());
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
        new Shape(-1, DecisionInputSchema.STATE_INT_COUNT),
        new Shape(-1, DecisionInputSchema.STATE_FLOAT_COUNT));
    return model;
  }
}
