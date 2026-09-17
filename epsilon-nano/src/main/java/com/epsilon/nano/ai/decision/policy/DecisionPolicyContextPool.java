package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import com.epsilon.core.Action;

/**
 * 条件付きの行動確率で下位の候補表現を重み付けし、上位の分岐を評価する文脈を作る。
 *
 * <p>方策の確率分布と同じ重みを使い、上位の損失から子のロジットへの勾配だけを切り離す。候補表現への勾配は維持する。
 */
public final class DecisionPolicyContextPool {

  private static final int[] ALTERNATIVE_TYPE_MAPPING = createAlternativeTypeMapping();
  private static final int[] PASS_TYPE_MAPPING = {Action.Type.PASS.ordinal()};

  private DecisionPolicyContextPool() {}

  static NDArray aggregateAlternativeContexts(
      NDArray alternativeScores,
      NDArray alternativeContexts,
      NDArray alternativePresence,
      DecisionAlternative... alternatives) {
    int[] choiceIndices = new int[alternatives.length];
    for (int index = 0; index < alternatives.length; ++index) {
      choiceIndices[index] = alternatives[index].networkIndex();
    }
    return NDArrays.indexedMaskedSoftmaxPool(
        alternativeScores.stopGradient(), alternativePresence, alternativeContexts, choiceIndices);
  }

  static NDArray presentAny(NDArray[] legalByIndex, DecisionAlternative... alternatives) {
    NDArray result = null;
    for (DecisionAlternative alternative : alternatives) {
      NDArray legal = legalByIndex[alternative.networkIndex()];
      result = result == null ? legal : result.add(legal);
    }
    return result.minimum(1.0f).stopGradient();
  }

  /** 二値分岐の条件付き分布で二分岐を集約し、片側だけ合法ならそのコンテキストを維持する。 */
  static NDArray mixBinaryBranchContexts(
      NDArray baselineContext,
      NDArray selectedContext,
      NDArray selectedLogit,
      NDArray baselinePresence,
      NDArray selectedPresence) {
    NDArray selectedProbability = Activation.sigmoid(selectedLogit).stopGradient();
    BinaryBranchWeights weights =
        binaryBranchWeights(baselinePresence, selectedPresence, selectedProbability);
    return baselineContext
        .mul(weights.baselineWeight())
        .add(selectedContext.mul(weights.selectedWeight()));
  }

  static NDArray presentEither(NDArray first, NDArray second) {
    return first.add(second).minimum(1.0f).stopGradient();
  }

  static NDArray poolCandidates(
      NDArray candidateEmbeddings, NDArray candidateScores, NDArray candidateMask) {
    NDArray weights = conditionalCandidateWeights(candidateScores, candidateMask);
    return candidateEmbeddings.mul(weights.expandDims(2)).sum(new int[] {1});
  }

  /**
   * 全行動種類の条件付き候補文脈と存在マスクを一括して構成する。
   *
   * <p>学習時は候補表現への勾配だけを維持し、方策スコアは従来どおり勾配を切り離す。ROCm推論時は種類ごとの マスク付きソフトマックス・積・集約を単一カーネルへまとめる。
   */
  static CandidateTypeContexts poolCandidateTypes(
      NDArray candidateEmbeddings,
      NDArray candidateScores,
      DecisionPolicyCandidates.CandidateMasks candidateMasks) {
    NDArray contexts =
        NDArrays.groupedMaskedSoftmaxPool(
            candidateScores.stopGradient(), candidateMasks.typeMasks(), candidateEmbeddings);
    NDArray presence =
        candidateMasks
            .typeMasks()
            .transpose(2, 0, 1)
            .sum(new int[] {2})
            .minimum(1.0f)
            .stopGradient();
    return new CandidateTypeContexts(contexts, presence);
  }

  /** 種類一括集約をネットワーク列順へ並べ替え、PASSを独立した連続集合として返す。 */
  static MappedCandidateContexts poolMappedCandidateTypes(
      NDArray candidateEmbeddings,
      NDArray candidateScores,
      DecisionPolicyCandidates.CandidateMasks candidateMasks) {
    CandidateTypeContexts byType =
        poolCandidateTypes(candidateEmbeddings, candidateScores, candidateMasks);
    return mapCandidateTypes(byType);
  }

  /** Fusion定数へ関連付ける分岐候補列から行動種類への写像を返す。 */
  public static int[] alternativeTypeMapping() {
    return ALTERNATIVE_TYPE_MAPPING.clone();
  }

  /** Fusion定数へ関連付けるPASS 集合の行動種類写像を返す。 */
  public static int[] passTypeMapping() {
    return PASS_TYPE_MAPPING.clone();
  }

  /** 後段Fusionへ連続な単一列として渡す分岐候補の行動種類写像を返す。 */
  public static int[] directTypeMapping(DecisionAlternative alternative) {
    if (!alternative.hasActionType()) {
      throw new IllegalArgumentException("direct candidate context requires an action type");
    }
    return new int[] {alternative.actionType().ordinal()};
  }

  /** 方策グラフの種類内候補分布と同じ重みを返す。 */
  static NDArray conditionalCandidateWeights(NDArray candidateScores, NDArray candidateMask) {
    return NDArrays.maskedSoftmax(candidateScores, candidateMask, 1).stopGradient();
  }

  /** CONTINUE後の打牌識別情報と、その識別情報に条件付いたDAMA/RIICHIを方策と同じ順序で集約する。 */
  static NDArray poolDiscardContinuation(
      NDArray candidateEmbeddings,
      NDArray actionCandidateScores,
      NDArray riichiGateScores,
      DecisionPolicyCandidates.DiscardChoices discardChoices) {
    return poolDiscardContinuation(
        gatherDiscardBranches(candidateEmbeddings, discardChoices),
        actionCandidateScores,
        riichiGateScores,
        discardChoices);
  }

  static NDArray poolDiscardContinuation(
      DiscardBranchContexts discardBranchContexts,
      NDArray actionCandidateScores,
      NDArray riichiGateScores,
      DecisionPolicyCandidates.DiscardChoices discardChoices) {
    NDArray riichiGateLogits =
        NDArrays.paddedBatchGather(riichiGateScores, discardChoices.riichiSlots());
    NDArray riichiProbability = Activation.sigmoid(riichiGateLogits).stopGradient();
    BinaryBranchWeights weights =
        binaryBranchWeights(
            discardChoices.damaPresent(), discardChoices.riichiPresent(), riichiProbability);
    NDArray identityContexts =
        discardBranchContexts
            .dama()
            .mul(weights.baselineWeight().expandDims(2))
            .add(discardBranchContexts.riichi().mul(weights.selectedWeight().expandDims(2)));
    NDArray identityWeights =
        conditionalCandidateWeights(actionCandidateScores, discardChoices.representativeMask());
    return identityContexts.mul(identityWeights.expandDims(2)).sum(new int[] {1});
  }

  static DiscardBranchContexts gatherDiscardBranches(
      NDArray candidateEmbeddings, DecisionPolicyCandidates.DiscardChoices discardChoices) {
    return new DiscardBranchContexts(
        NDArrays.paddedBatchGather(candidateEmbeddings, discardChoices.damaSlots()),
        NDArrays.paddedBatchGather(candidateEmbeddings, discardChoices.riichiSlots()));
  }

  /** DAMA/RIICHIへ同じ物理打牌の識別情報スコアを共有する。 */
  static NDArray sharePhysicalDiscardScores(
      NDArray discardScores, DecisionPolicyCandidates.DiscardChoices discardChoices) {
    return NDArrays.paddedBatchGather(discardScores, discardChoices.representativeSlots());
  }

  static NDArray present(NDArray mask) {
    return mask.sum(new int[] {1}).reshape(-1, 1).minimum(1.0f).stopGradient();
  }

  static NDArray concatenateAlternativeAxis(NDArray[] valuesByNetworkIndex) {
    return NDArrays.stack(new NDList(valuesByNetworkIndex), 1);
  }

  private static MappedCandidateContexts mapCandidateTypes(CandidateTypeContexts byType) {
    long batch = byType.contexts().getShape().get(1);
    long width = byType.contexts().getShape().get(2);
    NDArray zeroContext =
        byType
            .contexts()
            .getManager()
            .zeros(new ai.djl.ndarray.types.Shape(batch, width), byType.contexts().getDataType());
    NDArray zeroPresence =
        byType
            .presence()
            .getManager()
            .zeros(new ai.djl.ndarray.types.Shape(batch, 1), byType.presence().getDataType());
    NDArray[] contexts = new NDArray[DecisionAlternative.NETWORK_SIZE];
    NDArray[] presence = new NDArray[DecisionAlternative.NETWORK_SIZE];
    for (DecisionAlternative alternative : DecisionAlternative.values()) {
      int networkIndex = alternative.networkIndex();
      if (!alternative.hasActionType()) {
        contexts[networkIndex] = zeroContext;
        presence[networkIndex] = zeroPresence;
        continue;
      }
      Action.Type type = alternative.actionType();
      contexts[networkIndex] = byType.context(type);
      presence[networkIndex] = byType.present(type);
    }
    NDArray passContext = byType.context(Action.Type.PASS);
    NDArray passPresence = byType.present(Action.Type.PASS);
    return new MappedCandidateContexts(
        concatenateAlternativeAxis(contexts),
        concatenateAlternativeAxis(presence).reshape(batch, DecisionAlternative.NETWORK_SIZE),
        passContext.expandDims(1),
        passPresence,
        null,
        null,
        null,
        null);
  }

  private static int[] createAlternativeTypeMapping() {
    int[] mapping = new int[DecisionAlternative.NETWORK_SIZE];
    java.util.Arrays.fill(mapping, -1);
    for (DecisionAlternative alternative : DecisionAlternative.values()) {
      if (alternative.hasActionType()) {
        mapping[alternative.networkIndex()] = alternative.actionType().ordinal();
      }
    }
    return mapping;
  }

  /** 二分岐の存在量と選択した確率から、片側だけ存在する場合も質量を保つ勾配を切り離した重みを作る。 */
  private static BinaryBranchWeights binaryBranchWeights(
      NDArray baselinePresence, NDArray selectedPresence, NDArray selectedProbability) {
    NDArray jointPresence = baselinePresence.mul(selectedPresence);
    NDArray selectedMass = jointPresence.mul(selectedProbability);
    NDArray selectedWeight = selectedPresence.sub(jointPresence).add(selectedMass).stopGradient();
    NDArray baselineWeight = baselinePresence.sub(selectedMass).stopGradient();
    return new BinaryBranchWeights(baselineWeight, selectedWeight);
  }

  record DiscardBranchContexts(NDArray dama, NDArray riichi) {}

  /** 全行動種類をAction.Type 通し番号順に保持する候補文脈と存在マスク。 */
  record CandidateTypeContexts(NDArray contexts, NDArray presence) {

    NDArray context(Action.Type type) {
      return contexts.get("{},:,:", type.ordinal());
    }

    NDArray present(Action.Type type) {
      return presence.get("{},:", type.ordinal()).reshape(-1, 1);
    }
  }

  /** ネットワーク列順の分岐候補集合と、独立したPASS 集合を保持する候補文脈。 */
  public record MappedCandidateContexts(
      NDArray alternativeContexts,
      NDArray alternativePresence,
      NDArray passContexts,
      NDArray passPresence,
      NDArray ronContexts,
      NDArray kyushuContexts,
      NDArray kyushuPresence,
      NDArray tsumoContexts) {

    NDArray context(DecisionAlternative alternative) {
      NDArray direct = directContexts(alternative);
      if (direct != null) {
        return direct.get(":,0,:");
      }
      return alternativeContexts.get(":,{},:", alternative.networkIndex());
    }

    NDArray present(DecisionAlternative alternative) {
      if (alternative == DecisionAlternative.KYUSHU && kyushuPresence != null) {
        return kyushuPresence;
      }
      return alternativePresence.get(":,{}", alternative.networkIndex()).reshape(-1, 1);
    }

    NDArray passContext() {
      return passContexts.get(":,0,:");
    }

    private NDArray directContexts(DecisionAlternative alternative) {
      return switch (alternative) {
        case RON -> ronContexts;
        case KYUSHU -> kyushuContexts;
        case TSUMO -> tsumoContexts;
        default -> null;
      };
    }
  }

  private record BinaryBranchWeights(NDArray baselineWeight, NDArray selectedWeight) {}
}
