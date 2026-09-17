package com.epsilon.major.ai.decision.runtime;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.major.ai.model.EpsilonDecisionNetwork;
import com.epsilon.major.ai.network.NetworkFactory;
import com.epsilon.major.config.settings.DecisionInferenceSettings;

/** Decision モデルの独立した推論用複製を作る。 */
public final class EpsilonDecisionModelCopies {

  private EpsilonDecisionModelCopies() {}

  /** 所有モデルを複製せず推論用に準備する。CPUはFP32を保持し、準備失敗時はモデルを閉じる。 */
  public static Model prepareOwnedInferenceModel(Model model, DecisionInferenceSettings settings) {
    try {
      DecisionComputePrecision precision = settings.computePrecision();
      if (model.getNDManager().getDevice().isGpu()
          && settings.frozenBfloat16Parameters()
          && precision != DecisionComputePrecision.FLOAT32) {
        castFrozenParameters(
            model,
            precision == DecisionComputePrecision.FLOAT16 ? DataType.FLOAT16 : DataType.BFLOAT16);
      } else {
        freezeParameters(model);
      }
      return model;
    } catch (RuntimeException | Error failure) {
      try {
        model.close();
      } catch (RuntimeException | Error cleanupFailure) {
        failure.addSuppressed(cleanupFailure);
      }
      throw failure;
    }
  }

  /**
   * 入力元と独立した固定したモデルを転送先デバイス上に作る。
   *
   * <p>入力元は複製中に更新されないこと。転送先モデルが所有する初期化済み配列へ直接複製するため、入力元の管理元や寿命には依存しない。
   */
  static Model copyFrozenToDevice(Model sourceModel, Device targetDevice) {
    Model source = requireDecisionModel(sourceModel, "source");
    EpsilonDecisionNetwork sourceBlock = decisionBlock(source, "source");
    Model target =
        NetworkFactory.createDecisionModel(
            targetDevice, false, sourceBlock.hiddenSize(), sourceBlock.utilityProfile());
    try {
      validateStructure(source, target);
      copyParameters(source, target);
      freezeParameters(target);
      return target;
    } catch (RuntimeException | Error failure) {
      try {
        target.close();
      } catch (RuntimeException | Error cleanupFailure) {
        failure.addSuppressed(cleanupFailure);
      }
      throw failure;
    }
  }

  /** 凍結した推論モデルの複製のパラメーターを指定精度へ一度だけ変換し、元配列を解放する。 */
  static void castFrozenParameters(Model model, DataType dataType) {
    Model replica = requireDecisionModel(model, "inferenceReplica");
    freezeParameters(replica);
    for (Parameter parameter : replica.getBlock().getParameters().values()) {
      parameter.castArray(dataType);
    }
  }

  static void validateStructure(Model sourceModel, Model targetModel) {
    Model source = requireDecisionModel(sourceModel, "source");
    Model target = requireDecisionModel(targetModel, "target");
    EpsilonDecisionNetwork sourceBlock = decisionBlock(source, "source");
    EpsilonDecisionNetwork targetBlock = decisionBlock(target, "target");
    if (sourceBlock.hiddenSize() != targetBlock.hiddenSize()) {
      throw new IllegalArgumentException(
          "Decision model block mismatch: source hidden="
              + sourceBlock.hiddenSize()
              + " target hidden="
              + targetBlock.hiddenSize());
    }
    if (sourceBlock.utilityProfile() != targetBlock.utilityProfile()) {
      throw new IllegalArgumentException(
          "Decision model utility profile mismatch: source="
              + sourceBlock.utilityProfile()
              + " target="
              + targetBlock.utilityProfile());
    }
    validateParameters(source.getBlock().getParameters(), target.getBlock().getParameters());
  }

  private static void validateParameters(ParameterList source, ParameterList target) {
    if (source.size() != target.size()) {
      throw new IllegalArgumentException(
          "Decision model parameter count mismatch: source="
              + source.size()
              + " target="
              + target.size());
    }
    for (String name : source.keys()) {
      Parameter sourceParameter = source.get(name);
      Parameter targetParameter = target.get(name);
      if (targetParameter == null) {
        throw new IllegalArgumentException("Decision target parameter is missing: " + name);
      }
      if (!sourceParameter.isInitialized() || !targetParameter.isInitialized()) {
        throw new IllegalArgumentException("Decision model parameter is not initialized: " + name);
      }
      if (!sourceParameter.getShape().equals(targetParameter.getShape())) {
        throw new IllegalArgumentException(
            "Decision model parameter shape mismatch: name="
                + name
                + " source="
                + sourceParameter.getShape()
                + " target="
                + targetParameter.getShape());
      }
    }
  }

  private static void copyParameters(Model sourceModel, Model targetModel) {
    ParameterList source = sourceModel.getBlock().getParameters();
    ParameterList target = targetModel.getBlock().getParameters();
    for (String name : source.keys()) {
      NDArray sourceArray = source.get(name).getArray();
      NDArray targetArray = target.get(name).getArray();
      if (sourceArray.getDataType() != targetArray.getDataType()) {
        throw new IllegalArgumentException(
            "Decision model parameter data type mismatch: name="
                + name
                + " source="
                + sourceArray.getDataType()
                + " target="
                + targetArray.getDataType());
      }
      if (sourceArray.isSparse() || targetArray.isSparse()) {
        throw new IllegalArgumentException("Decision model parameters must be dense: " + name);
      }
      sourceArray.copyTo(targetArray);
    }
  }

  /** 全精度の推論モデルの複製を明示的に凍結し、勾配グラフから外す。 */
  static void freezeParameters(Model model) {
    Model replica = requireDecisionModel(model, "inferenceReplica");
    replica.getBlock().freezeParameters(true);
    for (Parameter parameter : replica.getBlock().getParameters().values()) {
      if (parameter.isInitialized()) {
        parameter.getArray().setRequiresGradient(false);
      }
    }
  }

  public static Model requireDecisionModel(Model model, String label) {
    decisionBlock(model, label);
    return model;
  }

  private static EpsilonDecisionNetwork decisionBlock(Model model, String label) {
    if (!(model.getBlock() instanceof EpsilonDecisionNetwork decisionNetwork)) {
      throw new IllegalArgumentException(
          "Decision " + label + " model has unexpected block: " + model.getBlock());
    }
    return decisionNetwork;
  }
}
