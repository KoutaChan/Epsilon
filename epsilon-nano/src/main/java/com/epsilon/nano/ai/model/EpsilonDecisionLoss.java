package com.epsilon.nano.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.ai.decision.DecisionBranchLoss;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.nano.ai.decision.EpsilonUtilityTargets;
import com.epsilon.nano.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.decision.policy.DecisionAlternative;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyScores;
import com.epsilon.nano.ai.decision.policy.EpsilonDecisionPolicyGraph;
import com.epsilon.nano.ai.decision.training.DecisionOnlineLossConfig;

/**
 * 牌譜による方策の模倣学習、スカラー価値の学習、選択行動の方策勾配学習の損失を計算する。
 *
 * <p>PPO教師信号は実行資格を持つ選択した行動に対してだけ与える。探索学習への寄与は {@code alpha + (1-alpha) * piRollout/muBehavior}
 * とし、方策更新比率 {@code piCurrent/piRollout} だけをPPO クリップした代理目的関数へ入れる。エントロピー
 * 項は反実仮想ラベルを使わず、現在の合法手分布全体へ勾配を流す。対局生成方策との {@code KL(piRollout || piCurrent)}
 * は損失に加算せず、更新前の診断と棄却判定にだけ使う。
 *
 * <p>限定二分岐が有効な自己対局では、完成した比較を対象gateだけの二択目的へ渡す。そのgateは通常PPOと
 * エントロピー項からdetachする。KYUSHUは比較未完成でもdetachし、比較結果を価値教師へ混ぜない。
 *
 * <p>Decision 価値は期待効用をガウス分布のヒストグラムへ変換した HL-Gauss 交差エントロピーで学習する。GRP 境界分布の期待
 * 効用は固定事前予測としてネットワークへ入り、Decision 側は区間ロジットの残差を学習する。
 */
public final class EpsilonDecisionLoss {

  private static final float PROBABILITY_EPSILON = 1.0e-8f;
  private static final DataType LOSS_DATA_TYPE = DataType.FLOAT32;

  private EpsilonDecisionLoss() {}

  /**
   * 牌譜などの固定データによる方策ネットワーク専用のサンプル重み付き模倣学習損失を計算する。
   *
   * @param output ネットワークが出力した候補・分岐候補スコア
   * @param input 合法候補、参照経路、教師枠、サンプル重みを持つデバイス側バッチ
   * @return 方策損失、エントロピー、教師行動確率
   */
  public static PolicyPretrainLoss computePolicyPretrainingLoss(
      DecisionPolicyScores output, DecisionDeviceBatch input) {
    if (!input.hasTrainingTargets()) {
      throw new IllegalArgumentException("Decision policy pretrain requires training targets");
    }
    if (input.bucket().legalActionCapacity() == 1) {
      throw new IllegalArgumentException(
          "Decision policy pretrain requires at least two legal actions");
    }
    long batch = input.rowCount();
    DecisionDeviceBatch.TrainingTargets targets = input.trainingTargets();
    NDArray sampleWeight = targets.sampleWeight().reshape(batch).stopGradient();
    NDArray chosenIndices = chosenIndices(targets.chosenSlot(), batch);
    EpsilonDecisionPolicyGraph.Distribution distribution =
        composePolicyDistributionForLoss(output, input);
    NDArray selectedLogProbability =
        selectedByIndices(distribution.logProbabilities(), chosenIndices);
    NDArray loss = weightedMean(selectedLogProbability.neg(), sampleWeight);
    NDArray entropy =
        weightedMean(
            distribution
                .probabilities()
                .mul(distribution.logProbabilities())
                .sum(new int[] {1})
                .neg(),
            sampleWeight);
    NDArray chosenProbability =
        weightedMean(selectedByIndices(distribution.probabilities(), chosenIndices), sampleWeight);
    return new PolicyPretrainLoss(loss, entropy, chosenProbability);
  }

  /**
   * 完全な入力検査を有効にして自己対局学習損失を計算する診断向けオーバーロード。
   *
   * <p>本番頻繁に実行する処理ではモデルに固定した効用の定義を渡し、検査済み学習データファイルを CPU へ戻さない。
   *
   * @param output 方策と価値のネットワーク出力
   * @param input 自己対局学習教師値を持つデバイス側バッチ
   * @param config PPO クリップ・探索学習への寄与混合率・エントロピー係数の設定
   * @param hasPositiveActorWeight バッチに方策モデル対象サンプルが存在するか
   * @return 逆伝播対象損失と全診断量
   */
  public static TrainingLossResult computeOnlineTrainingLoss(
      EpsilonDecisionOutput output,
      DecisionDeviceBatch input,
      DecisionOnlineLossConfig config,
      boolean hasPositiveActorWeight) {
    return computeOnlineTrainingLoss(output, input, config, hasPositiveActorWeight, true);
  }

  /** ネットワークの方策スコアを損失と同じFLOAT32境界で合法手分布へ合成する。 */
  public static EpsilonDecisionPolicyGraph.Distribution composePolicyDistributionForLoss(
      DecisionPolicyScores scores, DecisionDeviceBatch input) {
    return applyLossPrecision(scores)
        .distribution(input.inputs().actionCategories(), input.inputs().actionRoutes());
  }

  /**
   * 方策と価値の損失および診断値を計算する。
   *
   * <p>{@code debugValidatePolicyGradientInputs} は収集済みテンソルをCPUへ戻す完全検査を有効にする。比率診断は
   * 本番実行時もデバッグ時もデバイス上で同じように集約する。
   *
   * @param output 方策と価値のネットワーク出力
   * @param input 自己対局学習教師値を持つデバイス側バッチ
   * @param config PPO クリップ・探索学習への寄与混合率・エントロピー係数の設定
   * @param hasPositiveActorWeight バッチに方策モデル対象サンプルが存在するか
   * @param debugValidatePolicyGradientInputs ホスト検査と詳細比率分位点を有効にするか
   * @return 逆伝播対象損失と全診断量
   */
  public static TrainingLossResult computeOnlineTrainingLoss(
      EpsilonDecisionOutput output,
      DecisionDeviceBatch input,
      DecisionOnlineLossConfig config,
      boolean hasPositiveActorWeight,
      boolean debugValidatePolicyGradientInputs) {
    if (!input.hasTrainingTargets()) {
      throw new IllegalArgumentException("Decision loss requires training targets");
    }
    return computeOnlineTrainingLoss(
        output,
        input,
        config,
        hasPositiveActorWeight,
        debugValidatePolicyGradientInputs,
        EpsilonUtilityTargets.configuredTrainingProfile());
  }

  /** モデル生成時の効用の定義を使い、CPU 転送なしで自己対局学習損失を計算する。 */
  public static TrainingLossResult computeOnlineTrainingLoss(
      EpsilonDecisionOutput output,
      DecisionDeviceBatch input,
      DecisionOnlineLossConfig config,
      boolean hasPositiveActorWeight,
      boolean debugValidatePolicyGradientInputs,
      EpsilonUtilityProfile utilityProfile) {
    return computeOnlineTrainingLoss(
        output,
        input,
        config,
        hasPositiveActorWeight,
        debugValidatePolicyGradientInputs,
        utilityProfile,
        null);
  }

  /** モデルと同じ効用の定義を持つ実行デバイス所有の定数を使い、教師ラベルの定数転送を省く。 */
  public static TrainingLossResult computeOnlineTrainingLoss(
      EpsilonDecisionOutput output,
      DecisionDeviceBatch input,
      DecisionOnlineLossConfig config,
      boolean hasPositiveActorWeight,
      boolean debugValidatePolicyGradientInputs,
      EpsilonUtilityProfile utilityProfile,
      EpsilonDecisionHlGauss.DeviceConstants valueConstants) {
    if (!input.hasTrainingTargets()) {
      throw new IllegalArgumentException("Decision loss requires training targets");
    }
    DecisionDeviceBatch.TrainingTargets targets = input.trainingTargets();
    long batch = input.rowCount();
    long legalActionCapacity = input.bucket().legalActionCapacity();
    requireShapes(output, input, targets, batch, legalActionCapacity);

    NDArray sampleWeight = targets.sampleWeight().reshape(batch).stopGradient();
    NDArray actorWeight = targets.actorWeight().reshape(batch).stopGradient();
    NDArray actorWeights =
        hasPositiveActorWeight
            ? actorWeight.mul(sampleWeight).stopGradient()
            : actorWeight.mul(0.0f).stopGradient();
    NDArray sampleWeightSum = sampleWeight.sum();
    NDArray sampleWeightNormalization = sampleWeightSum.maximum(PROBABILITY_EPSILON);
    NDArray chosenIndices = chosenIndices(targets.chosenSlot(), batch);
    NDArray valueLoss =
        computeValueLossWithLabels(
            output.valueLogits(),
            valueConstants == null
                ? EpsilonDecisionHlGauss.targetProbabilities(targets.valueTarget(), utilityProfile)
                : valueConstants.targetProbabilities(targets.valueTarget()),
            sampleWeight,
            sampleWeightNormalization);
    DecisionPolicyScores onlineScores = applyLossPrecision(output.policyScores());
    if (config.branchComparisonEnabled()) {
      onlineScores =
          new DecisionPolicyScores(
              DecisionBranchLoss.detachComparedGates(
                  onlineScores.alternativeScores(), targets.branchTargets()),
              onlineScores.actionCandidateScores(),
              onlineScores.riichiGateScores());
    }
    EpsilonDecisionPolicyGraph.Distribution currentPolicy =
        composePolicyDistributionForLoss(onlineScores, input);
    NDArray selectedCurrentLogProbability =
        selectedByIndices(currentPolicy.logProbabilities(), chosenIndices);
    NDArray behaviorCloningLoss =
        weightedMean(selectedCurrentLogProbability.neg(), sampleWeight, sampleWeightNormalization);

    if (!hasPositiveActorWeight) {
      NDArray zero = sampleWeightSum.mul(0.0f);
      return new TrainingLossResult(
          zero,
          zero,
          zero,
          behaviorCloningLoss,
          valueLoss,
          zero,
          zero,
          zero,
          zero,
          zero,
          zero,
          zero,
          zero,
          zero,
          zeroPolicyRatioDiagnostics(zero));
    }

    if (debugValidatePolicyGradientInputs) {
      NDArray legalMask =
          input
              .legalActionMask()
              .reshape(batch, legalActionCapacity)
              .toType(DataType.FLOAT32, false)
              .stopGradient();
      requirePolicyGradientInputs(
          targets,
          legalMask,
          chosenMask(targets.chosenSlot(), batch, legalActionCapacity),
          batch,
          legalActionCapacity);
    }
    NDArray actorWeightSum = actorWeights.sum();
    NDArray actorWeightNormalization = actorWeightSum.maximum(PROBABILITY_EPSILON);
    NDArray rolloutPolicy =
        targets.rolloutPolicy().reshape(batch, legalActionCapacity).stopGradient();
    NDArray behaviorPolicy =
        targets.behaviorPolicy().reshape(batch, legalActionCapacity).stopGradient();
    NDArray rolloutLogProbability = rolloutPolicy.add(PROBABILITY_EPSILON).log().stopGradient();
    NDArray behaviorLogProbability = behaviorPolicy.add(PROBABILITY_EPSILON).log().stopGradient();
    NDArray selectedBehaviorProbability =
        selectedByIndices(behaviorPolicy, chosenIndices).stopGradient();
    NDArray selectedRolloutProbability =
        selectedByIndices(rolloutPolicy, chosenIndices).stopGradient();
    NDArray selectedCurrentProbability =
        selectedByIndices(currentPolicy.probabilities(), chosenIndices);

    // 方策モデル対象外の行は選択確率ゼロを許す。重みを掛ける前に0除算を除き、対象行の比率は変えない。
    NDArray inactiveActor = actorWeights.lte(0.0f).toType(LOSS_DATA_TYPE, false);
    NDArray rawExplorationRatio =
        selectedRolloutProbability
            .div(selectedBehaviorProbability.add(inactiveActor))
            .stopGradient();
    NDArray explorationCreditWeight =
        rawExplorationRatio
            .mul(1.0f - config.explorationCreditMix())
            .add(config.explorationCreditMix())
            .stopGradient();
    NDArray policyUpdateRatio =
        selectedCurrentProbability.div(selectedRolloutProbability.add(inactiveActor));
    NDArray clippedPolicyUpdateRatio =
        clipPolicyUpdateRatio(policyUpdateRatio, config.policyUpdateClipRange());
    NDArray scalarAdvantage = targets.advantage().reshape(batch).stopGradient();
    NDArray unclippedPolicyObjective = policyUpdateRatio.mul(scalarAdvantage);
    NDArray clippedPolicyObjective = clippedPolicyUpdateRatio.mul(scalarAdvantage);
    NDArray policyGradientLoss =
        weightedMean(
            explorationCreditWeight
                .mul(unclippedPolicyObjective.minimum(clippedPolicyObjective))
                .neg(),
            actorWeights,
            actorWeightNormalization);
    NDArray detachedPolicyUpdateRatio = policyUpdateRatio.stopGradient();
    NDArray effectiveActorRatio =
        explorationCreditWeight.mul(detachedPolicyUpdateRatio).stopGradient();
    NDArray policyUpdateClipIndicator =
        detachedPolicyUpdateRatio
            .sub(clipPolicyUpdateRatio(detachedPolicyUpdateRatio, config.policyUpdateClipRange()))
            .abs()
            .gt(0.0f)
            .toType(DataType.FLOAT32, false)
            .stopGradient();
    NDArray rolloutPolicyKlPerSample =
        rolloutPolicy
            .mul(rolloutLogProbability.sub(currentPolicy.logProbabilities()))
            .sum(new int[] {1});
    NDArray nonNegativeRolloutPolicyKl = rolloutPolicyKlPerSample.maximum(0.0f).stopGradient();

    NDArray currentEntropyPerSample =
        currentPolicy
            .probabilities()
            .mul(currentPolicy.logProbabilities())
            .sum(new int[] {1})
            .neg();
    NDArray currentEntropy =
        weightedMean(currentEntropyPerSample, actorWeights, actorWeightNormalization);
    NDArray entropyBonusLoss = currentEntropy.mul(-config.entropyCoefficient());
    NDArray rolloutEntropyPerSample =
        rolloutPolicy.mul(rolloutLogProbability).sum(new int[] {1}).neg().stopGradient();
    NDArray behaviorToRolloutPolicyKlPerSample =
        behaviorPolicy
            .mul(behaviorLogProbability.sub(rolloutLogProbability))
            .sum(new int[] {1})
            .stopGradient();
    OnlineDiagnostics diagnostics =
        summarizeOnlineDiagnostics(
            nonNegativeRolloutPolicyKl,
            rolloutEntropyPerSample,
            behaviorToRolloutPolicyKlPerSample,
            selectedCurrentProbability.stopGradient(),
            selectedBehaviorProbability,
            rawExplorationRatio,
            explorationCreditWeight,
            detachedPolicyUpdateRatio,
            effectiveActorRatio,
            policyUpdateClipIndicator,
            actorWeights);
    if (config.branchComparisonEnabled()) {
      policyGradientLoss =
          policyGradientLoss.add(
              DecisionBranchLoss.loss(
                  output.policyScores().alternativeScores().toType(DataType.FLOAT32, false),
                  targets.branchTargets(),
                  actorWeights,
                  config.policyUpdateClipRange()));
    }
    NDArray total = policyGradientLoss.add(entropyBonusLoss);
    return new TrainingLossResult(
        total,
        policyGradientLoss,
        entropyBonusLoss,
        behaviorCloningLoss,
        valueLoss,
        diagnostics.meanRolloutPolicyKl(),
        diagnostics.maximumRolloutPolicyKl(),
        currentEntropy,
        diagnostics.rolloutEntropy(),
        diagnostics.behaviorToRolloutPolicyKl(),
        diagnostics.chosenCurrentProbability(),
        diagnostics.behaviorProbabilityMean(),
        diagnostics.behaviorProbabilityMinimum(),
        actorWeightSum.div((float) batch),
        diagnostics.policyRatios());
  }

  private static NDArray clipPolicyUpdateRatio(NDArray ratio, float clipRange) {
    return ratio.maximum(1.0f - clipRange).minimum(1.0f + clipRange);
  }

  /** 無効な方策は 微小値の加算やクリップで補正せず、収集契約違反として処理を中断する。 */
  private static void requirePolicyGradientInputs(
      DecisionDeviceBatch.TrainingTargets targets,
      NDArray legalMask,
      NDArray chosenMask,
      long batch,
      long actions) {
    float[] legal = legalMask.toFloatArray();
    float[] chosen = chosenMask.toFloatArray();
    float[] behavior = targets.behaviorPolicy().toFloatArray();
    float[] rollout = targets.rolloutPolicy().toFloatArray();
    float[] advantage = targets.advantage().toFloatArray();
    float[] actorWeight = targets.actorWeight().toFloatArray();
    float[] sampleWeight = targets.sampleWeight().toFloatArray();
    int width = Math.toIntExact(actions);
    for (int row = 0; row < batch; row++) {
      if (!(actorWeight[row] > 0.0f) || !(sampleWeight[row] > 0.0f)) {
        continue;
      }
      int selectedCount = 0;
      double behaviorSum = 0.0;
      double rolloutSum = 0.0;
      for (int action = 0; action < width; action++) {
        int index = row * width + action;
        float legalValue = legal[index];
        float chosenValue = chosen[index];
        float behaviorValue = behavior[index];
        float rolloutValue = rollout[index];
        if (!Float.isFinite(legalValue)
            || !Float.isFinite(chosenValue)
            || !Float.isFinite(behaviorValue)
            || !Float.isFinite(rolloutValue)) {
          throw new IllegalArgumentException("policy-gradient policy inputs must be finite");
        }
        if (legalValue > 0.5f) {
          if (!(behaviorValue > 0.0f) || !(rolloutValue > 0.0f)) {
            throw new IllegalArgumentException(
                "policy-gradient legal actions require strictly positive behavior/rollout");
          }
          behaviorSum += behaviorValue;
          rolloutSum += rolloutValue;
          if (chosenValue > 0.5f) {
            selectedCount++;
          }
        } else if (Math.abs(behaviorValue) > 1.0e-7f
            || Math.abs(rolloutValue) > 1.0e-7f
            || Math.abs(chosenValue) > 1.0e-7f) {
          throw new IllegalArgumentException("policy-gradient illegal action entries must be zero");
        }
      }
      if (selectedCount != 1
          || Math.abs(behaviorSum - 1.0) > 1.0e-4
          || Math.abs(rolloutSum - 1.0) > 1.0e-4) {
        throw new IllegalArgumentException(
            "policy-gradient rows require one selected action and normalized policies");
      }
      if (!Float.isFinite(advantage[row])) {
        throw new IllegalArgumentException("policy-gradient advantage must be finite");
      }
    }
  }

  /**
   * 方策モデル診断ベクトルを一つの行列へまとめ、同じ重みによる総和/最小値/最大値を各一回だけ実行する。
   *
   * <p>診断値はすべて勾配から切り離されており、PPO 損失の計算計算グラフには接続しない。
   */
  private static OnlineDiagnostics summarizeOnlineDiagnostics(
      NDArray nonNegativeRolloutPolicyKl,
      NDArray rolloutEntropy,
      NDArray behaviorToRolloutPolicyKl,
      NDArray chosenCurrentProbability,
      NDArray behaviorProbability,
      NDArray rawExplorationRatio,
      NDArray explorationCreditWeight,
      NDArray policyUpdateRatio,
      NDArray effectiveActorRatio,
      NDArray policyUpdateClipIndicator,
      NDArray actorWeights) {
    final int meanKlRow = 0;
    final int rolloutEntropyRow = 1;
    final int behaviorKlRow = 2;
    final int chosenCurrentProbabilityRow = 3;
    final int behaviorProbabilityRow = 4;
    final int rawExplorationRatioRow = 5;
    final int explorationCreditWeightRow = 6;
    final int policyUpdateRatioRow = 7;
    final int effectiveActorRatioRow = 8;
    final int policyUpdateClipIndicatorRow = 9;
    final int effectiveActorRatioMeanSquareRow = 10;

    NDArray values =
        NDArrays.stack(
                new NDList(
                    nonNegativeRolloutPolicyKl,
                    rolloutEntropy,
                    behaviorToRolloutPolicyKl,
                    chosenCurrentProbability,
                    behaviorProbability,
                    rawExplorationRatio,
                    explorationCreditWeight,
                    policyUpdateRatio,
                    effectiveActorRatio,
                    policyUpdateClipIndicator,
                    effectiveActorRatio.square()))
            .stopGradient();
    NDArray statistics = NDArrays.weightedRowStatistics(values, actorWeights);
    NDArray means = statistics.get(0);
    NDArray minima = statistics.get(1);
    NDArray maxima = statistics.get(2);

    return new OnlineDiagnostics(
        means.get(meanKlRow),
        maxima.get(meanKlRow),
        means.get(rolloutEntropyRow),
        means.get(behaviorKlRow),
        means.get(chosenCurrentProbabilityRow),
        means.get(behaviorProbabilityRow),
        minima.get(behaviorProbabilityRow),
        new PolicyRatioDiagnostics(
            ratioSummary(means, minima, maxima, rawExplorationRatioRow),
            ratioSummary(means, minima, maxima, explorationCreditWeightRow),
            ratioSummary(means, minima, maxima, policyUpdateRatioRow),
            ratioSummary(means, minima, maxima, effectiveActorRatioRow),
            means.get(policyUpdateClipIndicatorRow),
            means.get(effectiveActorRatioMeanSquareRow)));
  }

  private static RatioSummary ratioSummary(NDArray means, NDArray minima, NDArray maxima, int row) {
    return new RatioSummary(means.get(row), minima.get(row), maxima.get(row));
  }

  private static PolicyRatioDiagnostics zeroPolicyRatioDiagnostics(NDArray reference) {
    NDArray zero = reference.sum().mul(0.0f);
    RatioSummary summary = new RatioSummary(zero, zero, zero);
    return new PolicyRatioDiagnostics(summary, summary, summary, summary, zero, zero);
  }

  /** 効用-区間ロジット [バッチ,101] とスカラー教師 [バッチ] の HL-Gauss 交差エントロピー。 */
  public static NDArray computeValueLoss(
      NDArray valueLogits, NDArray valueTarget, NDArray sampleWeight) {
    return computeValueLoss(
        valueLogits, valueTarget, sampleWeight, EpsilonUtilityTargets.configuredTrainingProfile());
  }

  /** モデル固有の値域を使う、サンプル重みで正規化した HL-Gauss 目的関数。 */
  public static NDArray computeValueLoss(
      NDArray valueLogits,
      NDArray valueTarget,
      NDArray sampleWeight,
      EpsilonUtilityProfile utilityProfile) {
    return computeValueLoss(
        valueLogits,
        valueTarget,
        sampleWeight,
        sampleWeight.sum().maximum(PROBABILITY_EPSILON),
        utilityProfile);
  }

  /** 実行単位所有の固定区間境界を共有し、教師ラベルの定数転送を省くHL-Gauss 目的関数。 */
  public static NDArray computeValueLoss(
      NDArray valueLogits,
      NDArray valueTarget,
      NDArray sampleWeight,
      EpsilonDecisionHlGauss.DeviceConstants constants) {
    return computeValueLossWithLabels(
        valueLogits,
        constants.targetProbabilities(valueTarget),
        sampleWeight,
        sampleWeight.sum().maximum(PROBABILITY_EPSILON));
  }

  private static NDArray computeValueLoss(
      NDArray valueLogits,
      NDArray valueTarget,
      NDArray lossWeight,
      NDArray normalization,
      EpsilonUtilityProfile utilityProfile) {
    return computeValueLossWithLabels(
        valueLogits,
        EpsilonDecisionHlGauss.targetProbabilities(valueTarget, utilityProfile),
        lossWeight,
        normalization);
  }

  private static NDArray computeValueLossWithLabels(
      NDArray valueLogits, NDArray labels, NDArray lossWeight, NDArray normalization) {
    long batch = lossWeight.getShape().get(0);
    requireShape(valueLogits, "valueLogits", batch, EpsilonDecisionHlGauss.BIN_COUNT);
    requireShape(labels, "valueLabels", batch, EpsilonDecisionHlGauss.BIN_COUNT);
    requireShape(lossWeight, "lossWeight", batch);
    requireShape(normalization, "normalization");
    NDArray logits = applyLossPrecision(valueLogits);
    NDArray perRow = logits.logSoftmax(1).mul(labels).sum(new int[] {1}).neg();
    return perRow.mul(lossWeight).sum().div(normalization);
  }

  /** 方策グラフの全分岐と候補正規化を損失精度で評価できるスコアへ変換する。 */
  private static DecisionPolicyScores applyLossPrecision(DecisionPolicyScores scores) {
    return new DecisionPolicyScores(
        applyLossPrecision(scores.alternativeScores()),
        applyLossPrecision(scores.actionCandidateScores()),
        applyLossPrecision(scores.riichiGateScores()));
  }

  /**
   * Autocastされたネットワーク出力を、勾配を保ったままFLOAT32 損失へ接続する。
   *
   * <p>DJL PyTorchの{@code toType}はテンソル移送用の非微分演算であるため、長さ1のFLOAT32 ゼロを加算して
   * PyTorchの型昇格を使う。ゼロは全要素分確保せずブロードキャストされ、結果側だけが損失に必要なFLOAT32 テンソルとなる。
   */
  private static NDArray applyLossPrecision(NDArray values) {
    return values.getDataType() == LOSS_DATA_TYPE
        ? values
        : values.add(values.getManager().zeros(new Shape(1), LOSS_DATA_TYPE));
  }

  private static NDArray selectedByIndices(NDArray values, NDArray chosenIndices) {
    return values.gather(chosenIndices, 1).reshape(chosenIndices.size());
  }

  private static NDArray chosenIndices(NDArray chosenSlot, long batch) {
    return chosenSlot.reshape(batch, 1).toType(DataType.INT64, false).stopGradient();
  }

  private static NDArray chosenMask(NDArray chosenSlot, long batch, long actions) {
    NDArray slots = chosenSlot.getManager().arange(actions).reshape(1, actions);
    return slots.eq(chosenSlot.reshape(batch, 1)).toType(DataType.FLOAT32, false).stopGradient();
  }

  private static NDArray weightedMean(NDArray values, NDArray weights, NDArray normalization) {
    return values.mul(weights).sum().div(normalization);
  }

  private static NDArray weightedMean(NDArray values, NDArray weights) {
    return weightedMean(values, weights, weights.sum().maximum(PROBABILITY_EPSILON));
  }

  private static void requireShapes(
      EpsilonDecisionOutput output,
      DecisionDeviceBatch input,
      DecisionDeviceBatch.TrainingTargets targets,
      long batch,
      long actions) {
    requireShape(
        output.policyScores().alternativeScores(),
        "alternativeScores",
        batch,
        DecisionAlternative.NETWORK_SIZE);
    requireShape(
        output.policyScores().actionCandidateScores(), "actionCandidateScores", batch, actions);
    requireShape(output.policyScores().riichiGateScores(), "riichiGateScores", batch, actions);
    requireShape(output.valueLogits(), "valueLogits", batch, EpsilonDecisionHlGauss.BIN_COUNT);
    requireShape(
        input.inputs().actionCategories(),
        "actionCategories",
        batch,
        actions,
        DecisionInputSchema.ACTION_INT_STRIDE);
    requireShape(targets.chosenSlot(), "chosenSlot", batch);
    requireShape(targets.behaviorPolicy(), "behaviorPolicy", batch, actions);
    requireShape(targets.rolloutPolicy(), "rolloutPolicy", batch, actions);
    requireShape(targets.valueTarget(), "valueTarget", batch);
    requireShape(targets.advantage(), "advantage", batch);
    requireShape(targets.actorWeight(), "actorWeight", batch);
    requireShape(targets.sampleWeight(), "sampleWeight", batch);
  }

  private static void requireShape(NDArray array, String label, long... dimensions) {
    Shape expected = new Shape(dimensions);
    if (!array.getShape().equals(expected)) {
      throw new IllegalArgumentException(
          label + " shape must be " + expected + ", got " + array.getShape());
    }
  }

  /**
   * 牌譜などの固定データによる方策事前学習の集約値。
   *
   * @param total 選択した行動の 重み付き負の対数尤度
   * @param entropy 最終的な合法手の確率分布の 重み付きエントロピー
   * @param chosenProbability 教師行動に割り当てた 重み付き平均確率
   */
  public record PolicyPretrainLoss(NDArray total, NDArray entropy, NDArray chosenProbability) {}

  /** 一種類の比率を方策学習の重みで集約した診断値。 */
  public record RatioSummary(NDArray mean, NDArray min, NDArray max) {}

  private record OnlineDiagnostics(
      NDArray meanRolloutPolicyKl,
      NDArray maximumRolloutPolicyKl,
      NDArray rolloutEntropy,
      NDArray behaviorToRolloutPolicyKl,
      NDArray chosenCurrentProbability,
      NDArray behaviorProbabilityMean,
      NDArray behaviorProbabilityMinimum,
      PolicyRatioDiagnostics policyRatios) {}

  /**
   * 元の探索比、探索学習への寄与、方策更新、実効方策モデル比を混同しない選択行動診断。
   *
   * @param rawExplorationRatio {@code piRollout/muBehavior}。診断専用
   * @param explorationCreditWeight {@code alpha + (1-alpha) * piRollout/muBehavior}
   * @param policyUpdate {@code piCurrent/piRollout}
   * @param effectiveActorRatio {@code explorationCreditWeight * policyUpdate}
   * @param policyUpdateClipFraction クリップ範囲外だった方策学習の重みの比率
   * @param effectiveActorRatioMeanSquare 小バッチを跨いで厳密なESSを集約するための加重二次モーメント
   */
  public record PolicyRatioDiagnostics(
      RatioSummary rawExplorationRatio,
      RatioSummary explorationCreditWeight,
      RatioSummary policyUpdate,
      RatioSummary effectiveActorRatio,
      NDArray policyUpdateClipFraction,
      NDArray effectiveActorRatioMeanSquare) {}

  /**
   * 方策モデル、エントロピー、価値と監査指標を一つのデバイス上の結果にまとめる。
   *
   * @param total 選択段階の合計損失
   * @param policyGradientLoss 選択行動の方策勾配による学習損失
   * @param entropyBonusLoss 最終的な行動の確率分布エントロピーを支える {@code -coefficient * entropy}
   * @param behaviorCloningLoss 模倣学習損失
   * @param valueLoss Decision スカラー価値の HL-Gauss 交差エントロピー
   * @param meanRolloutPolicyKl 方策学習の重み付き {@code KL(piRollout || piCurrent)} 平均
   * @param maximumRolloutPolicyKl サンプル単位 {@code KL(piRollout || piCurrent)} 最大値
   * @param currentEntropy 現在最終的な行動の確率分布の加重エントロピー
   * @param rolloutEntropy 対局生成最終的な行動の確率分布の加重エントロピー
   * @param behaviorToRolloutPolicyKl {@code KL(muBehavior || piRollout)}
   * @param chosenCurrentProb 選択行動の現在方策平均確率
   * @param behaviorProbMean 選択行動の探索後の平均確率
   * @param behaviorProbMin 選択行動の探索後の最小確率
   * @param actorWeight 方策損失に寄与した重み総量
   * @param policyRatioDiagnostics 元の探索比と実効探索学習への寄与を分離した比率診断
   */
  public record TrainingLossResult(
      NDArray total,
      NDArray policyGradientLoss,
      NDArray entropyBonusLoss,
      NDArray behaviorCloningLoss,
      NDArray valueLoss,
      NDArray meanRolloutPolicyKl,
      NDArray maximumRolloutPolicyKl,
      NDArray currentEntropy,
      NDArray rolloutEntropy,
      NDArray behaviorToRolloutPolicyKl,
      NDArray chosenCurrentProb,
      NDArray behaviorProbMean,
      NDArray behaviorProbMin,
      NDArray actorWeight,
      PolicyRatioDiagnostics policyRatioDiagnostics) {}
}
