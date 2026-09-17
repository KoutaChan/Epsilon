package com.epsilon.pico.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import com.epsilon.core.Action;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;

/**
 * 未正規化スコアから、麻雀の判断の階層に従って合法手の確率分布を作る。
 *
 * <p>応答時はロン、見送り・鳴き、面子の種類、具体候補の順に分解する。自摸時はツモ和了、九種九牌、継続・槓、槓の種類、打牌、リーチ・ダマの順に分解する。鳴きと槓の総確率は専用の二択で決まり、下位の具体候補の数には依存しない。
 *
 * <p>各行動の対数確率は、そこへ至る条件付き対数確率の和である。選択行動の損失は、その行動が通る分岐だけへ勾配を流す。パディング位置の確率は0にする。
 */
public final class EpsilonDecisionPolicyGraph {

  private static final float ILLEGAL_LOGIT = -1.0e9f;

  private EpsilonDecisionPolicyGraph() {}

  /**
   * スコア計算処理出力とencoded 行動メタデータから最終末端の行動分布を作る。
   *
   * @param scores バイナリ分岐、種類、具体候補、RIICHI/DAMA 二択の判定の未正規化スコア
   * @param actionCategories {@code [batch, legalActionCapacity, ACTION_INT_STRIDE]} の行動 メタデータ
   * @param actionRoutes 同じ物理牌の打牌の代表位置と RIICHI/DAMA 位置を指す参照先の対応メタデータ
   * @return 各行の合法手候補の位置上で総和1となる末端の行動分布
   */
  public static Distribution composeDistribution(
      DecisionPolicyScores scores, NDArray actionCategories, NDArray actionRoutes) {
    Composition composition = compose(scores, actionCategories, actionRoutes);
    NDArray logProbabilities = composition.logProbabilities();
    return new Distribution(logProbabilities, logProbabilities.exp().mul(composition.legalMask()));
  }

  /**
   * 推論で不要な確率テンソルを作らず、最終末端の行動の対数確率だけを返す。
   *
   * @param scores バイナリ分岐、種類、候補、RIICHI・DAMA 二択の判定の未正規化スコア
   * @param actionCategories 行動種類・グループを含むカテゴリ値テンソル
   * @param actionRoutes 同一打牌と RIICHI・DAMA 対応を持つ参照先の対応テンソル
   * @return 合法末端の行動上で正規化した対数確率
   */
  public static NDArray composeLogProbabilities(
      DecisionPolicyScores scores, NDArray actionCategories, NDArray actionRoutes) {
    return compose(scores, actionCategories, actionRoutes).logProbabilities();
  }

  private static Composition compose(
      DecisionPolicyScores scores, NDArray actionCategories, NDArray actionRoutes) {
    Shape candidateShape = scores.actionCandidateScores().getShape();
    if (candidateShape.dimension() != 2) {
      throw new IllegalArgumentException(
          "actionCandidateScores must be [B,A], got " + candidateShape);
    }
    long rowCount = candidateShape.get(0);
    long legalActionCapacity = candidateShape.get(1);
    requireShape(
        scores.alternativeScores(),
        "alternativeScores",
        rowCount,
        DecisionAlternative.NETWORK_SIZE);
    requireShape(scores.riichiGateScores(), "riichiGateScores", rowCount, legalActionCapacity);
    requireShape(
        actionCategories,
        "actionCategories",
        rowCount,
        legalActionCapacity,
        DecisionInputSchema.ACTION_INT_STRIDE);
    requireShape(
        actionRoutes,
        "actionRoutes",
        rowCount,
        legalActionCapacity,
        DecisionInputSchema.ACTION_ROUTE_STRIDE);

    NDArray actionIds = categoricalField(actionCategories, DecisionInputSchema.ActionInt.ID);
    NDArray actionTypes = categoricalField(actionCategories, DecisionInputSchema.ActionInt.TYPE);
    NDArray actionGroups = categoricalField(actionCategories, DecisionInputSchema.ActionInt.GROUP);
    NDArray legalMask = indicator(actionIds.neq(DecisionInputSchema.PAD_ID));
    NDArray responseRow =
        present(DecisionPolicyCandidates.responseMask(actionGroups, legalMask), rowCount);
    NDArray assigned = scores.actionCandidateScores().mul(0.0f);
    NDArray zeroMass = scores.actionCandidateScores().getManager().zeros(new Shape(rowCount, 1));

    PolicyBranch response = new PolicyBranch(assigned, legalMask.mul(responseRow), zeroMass);
    response =
        splitTerminalGate(
            response, scores, actionTypes, Action.Type.RON_AGARI, DecisionAlternative.RON);
    assigned = assignResponseBranch(response, scores, actionTypes, rowCount);

    PolicyBranch turn =
        new PolicyBranch(assigned, legalMask.mul(responseRow.neg().add(1.0f)), zeroMass);
    turn =
        splitTerminalGate(
            turn, scores, actionTypes, Action.Type.TSUMO_AGARI, DecisionAlternative.TSUMO);
    turn =
        splitTerminalGate(
            turn, scores, actionTypes, Action.Type.KYUSHU_KYUHAI, DecisionAlternative.KYUSHU);
    turn = splitKanBranch(turn, scores, actionTypes, rowCount);
    assigned =
        assignDiscardBranch(
            turn.assignedLogProbabilities(),
            scores.actionCandidateScores(),
            scores.riichiGateScores(),
            actionRoutes,
            actionTypes,
            turn.unresolvedCandidateMask(),
            turn.unresolvedLogProbabilityMass());

    NDArray logProbabilities = assigned.add(legalMask.neg().add(1.0f).mul(ILLEGAL_LOGIT));
    return new Composition(logProbabilities, legalMask);
  }

  private static PolicyBranch splitTerminalGate(
      PolicyBranch branch,
      DecisionPolicyScores scores,
      NDArray actionTypes,
      Action.Type acceptedType,
      DecisionAlternative acceptedAlternative) {
    long rowCount = scores.actionCandidateScores().getShape().get(0);
    NDArray acceptedMask =
        DecisionPolicyCandidates.typeMask(
            actionTypes, branch.unresolvedCandidateMask(), acceptedType);
    NDArray declinedMask = branch.unresolvedCandidateMask().sub(acceptedMask);
    NDArray activeGate =
        present(acceptedMask, rowCount).mul(present(declinedMask, rowCount)).stopGradient();
    NDArray gateLogit = scores.alternative(acceptedAlternative).reshape(rowCount, 1);
    NDArray acceptedMass =
        branch.unresolvedLogProbabilityMass().add(logSigmoid(gateLogit).mul(activeGate));
    NDArray assigned =
        assignConditionalDistribution(
            branch.assignedLogProbabilities(),
            scores.actionCandidateScores(),
            acceptedMask,
            acceptedMass);
    NDArray declinedMass =
        branch.unresolvedLogProbabilityMass().add(logSigmoid(gateLogit.neg()).mul(activeGate));
    return new PolicyBranch(assigned, declinedMask, declinedMass);
  }

  private static NDArray assignResponseBranch(
      PolicyBranch branch, DecisionPolicyScores scores, NDArray actionTypes, long rowCount) {
    NDArray passMask =
        DecisionPolicyCandidates.typeMask(
            actionTypes, branch.unresolvedCandidateMask(), Action.Type.PASS);
    NDArray[] meldTypeMasks =
        typeMasks(actionTypes, branch.unresolvedCandidateMask(), DecisionAlternative.meldTypes());
    NDArray meldMask = union(meldTypeMasks);
    NDArray activeGate =
        present(passMask, rowCount).mul(present(meldMask, rowCount)).stopGradient();
    NDArray callLogit = scores.alternative(DecisionAlternative.CALL).reshape(rowCount, 1);

    NDArray assigned =
        assignConditionalDistribution(
            branch.assignedLogProbabilities(),
            scores.actionCandidateScores(),
            passMask,
            branch.unresolvedLogProbabilityMass().add(logSigmoid(callLogit.neg()).mul(activeGate)));
    NDArray callMass =
        branch.unresolvedLogProbabilityMass().add(logSigmoid(callLogit).mul(activeGate));
    return assignTypedCandidates(
        assigned, scores, meldTypeMasks, DecisionAlternative.meldTypes(), callMass, rowCount);
  }

  private static PolicyBranch splitKanBranch(
      PolicyBranch branch, DecisionPolicyScores scores, NDArray actionTypes, long rowCount) {
    NDArray discardMask =
        DecisionPolicyCandidates.typeMask(
                actionTypes, branch.unresolvedCandidateMask(), Action.Type.DAHAI)
            .add(
                DecisionPolicyCandidates.typeMask(
                    actionTypes, branch.unresolvedCandidateMask(), Action.Type.RIICHI_DAHAI))
            .stopGradient();
    NDArray[] kanTypeMasks =
        typeMasks(actionTypes, branch.unresolvedCandidateMask(), DecisionAlternative.kanTypes());
    NDArray kanMask = union(kanTypeMasks);
    NDArray activeGate =
        present(discardMask, rowCount).mul(present(kanMask, rowCount)).stopGradient();
    NDArray kanLogit = scores.alternative(DecisionAlternative.KAN).reshape(rowCount, 1);
    NDArray kanMass =
        branch.unresolvedLogProbabilityMass().add(logSigmoid(kanLogit).mul(activeGate));
    NDArray assigned =
        assignTypedCandidates(
            branch.assignedLogProbabilities(),
            scores,
            kanTypeMasks,
            DecisionAlternative.kanTypes(),
            kanMass,
            rowCount);
    NDArray continueMass =
        branch.unresolvedLogProbabilityMass().add(logSigmoid(kanLogit.neg()).mul(activeGate));
    return new PolicyBranch(assigned, discardMask, continueMass);
  }

  private static NDArray assignTypedCandidates(
      NDArray currentLogProbabilities,
      DecisionPolicyScores scores,
      NDArray[] typeMasks,
      Action.Type[] types,
      NDArray branchLogProbability,
      long rowCount) {
    NDList typeScores = new NDList();
    NDList typePresence = new NDList();
    for (int index = 0; index < types.length; index++) {
      NDArray score =
          scores.alternative(DecisionAlternative.forActionType(types[index])).reshape(rowCount, 1);
      NDArray present = present(typeMasks[index], rowCount);
      typeScores.add(score);
      typePresence.add(present);
    }
    NDArray stackedTypeScores = NDArrays.concat(typeScores, 1);
    NDArray logNormalizer =
        NDArrays.maskedLogSumExp(stackedTypeScores, NDArrays.concat(typePresence, 1), 1);
    NDArray assigned = currentLogProbabilities;
    for (int index = 0; index < types.length; index++) {
      NDArray typeLogProbability =
          stackedTypeScores.get(":,{}", index).reshape(rowCount, 1).sub(logNormalizer);
      assigned =
          assignConditionalDistribution(
              assigned,
              scores.actionCandidateScores(),
              typeMasks[index],
              branchLogProbability.add(typeLogProbability));
    }
    return assigned;
  }

  private static NDArray assignDiscardBranch(
      NDArray currentLogProbabilities,
      NDArray discardIdentityScores,
      NDArray riichiGateScores,
      NDArray actionRoutes,
      NDArray actionTypes,
      NDArray eligibleCandidateMask,
      NDArray branchLogProbabilityMass) {
    DecisionPolicyCandidates.DiscardChoices discardChoices =
        DecisionPolicyCandidates.discardChoices(actionRoutes, actionTypes, eligibleCandidateMask);
    NDArray identityLogNormalizer =
        NDArrays.maskedLogSumExp(discardIdentityScores, discardChoices.representativeMask(), 1);
    NDArray representativeLogProbabilityMass =
        discardIdentityScores
            .sub(identityLogNormalizer)
            .add(branchLogProbabilityMass)
            .mul(discardChoices.representativeMask());
    NDArray candidateIdentityLogProbabilityMass =
        NDArrays.paddedBatchGather(
            representativeLogProbabilityMass, discardChoices.representativeSlots());
    NDArray identityRiichiGateLogit =
        NDArrays.paddedBatchGather(riichiGateScores, discardChoices.riichiSlots());
    NDArray activeGate =
        discardChoices.damaPresent().mul(discardChoices.riichiPresent()).stopGradient();
    NDArray variantLogProbability =
        logSigmoid(identityRiichiGateLogit)
            .mul(discardChoices.riichiMask())
            .add(logSigmoid(identityRiichiGateLogit.neg()).mul(discardChoices.damaMask()))
            .mul(activeGate);
    return currentLogProbabilities.add(
        candidateIdentityLogProbabilityMass
            .add(variantLogProbability)
            .mul(discardChoices.discardMask()));
  }

  private static NDArray assignConditionalDistribution(
      NDArray currentLogProbabilities,
      NDArray candidateScores,
      NDArray selectedMask,
      NDArray branchLogProbabilityMass) {
    NDArray logNormalizer = NDArrays.maskedLogSumExp(candidateScores, selectedMask, 1);
    return currentLogProbabilities.add(
        candidateScores.sub(logNormalizer).add(branchLogProbabilityMass).mul(selectedMask));
  }

  private static NDArray[] typeMasks(
      NDArray actionTypes, NDArray candidateMask, Action.Type[] types) {
    NDArray[] masks = new NDArray[types.length];
    for (int index = 0; index < types.length; index++) {
      masks[index] = DecisionPolicyCandidates.typeMask(actionTypes, candidateMask, types[index]);
    }
    return masks;
  }

  private static NDArray union(NDArray[] masks) {
    NDArray union = masks[0];
    for (int index = 1; index < masks.length; index++) {
      union = union.add(masks[index]);
    }
    return union.stopGradient();
  }

  private static NDArray present(NDArray mask, long rowCount) {
    return indicator(mask.sum(new int[] {1}).gt(0.5f)).reshape(rowCount, 1);
  }

  private static NDArray categoricalField(
      NDArray actionCategories, DecisionInputSchema.ActionInt field) {
    return actionCategories.get("...,{}", field.ordinal()).stopGradient();
  }

  private static NDArray logSigmoid(NDArray value) {
    return Activation.softPlus(value.neg()).neg();
  }

  private static NDArray indicator(NDArray condition) {
    return condition.toType(DataType.FLOAT32, false).stopGradient();
  }

  private static void requireShape(NDArray array, String label, long... dimensions) {
    Shape expected = new Shape(dimensions);
    if (!array.getShape().equals(expected)) {
      throw new IllegalArgumentException(
          label + " shape must be " + expected + ", got " + array.getShape());
    }
  }

  private record PolicyBranch(
      NDArray assignedLogProbabilities,
      NDArray unresolvedCandidateMask,
      NDArray unresolvedLogProbabilityMass) {}

  private record Composition(NDArray logProbabilities, NDArray legalMask) {}

  /**
   * 最終的な行動の確率分布。
   *
   * @param logProbabilities {@code [batch, legalActionCapacity]}。パディングは有限な大負値
   * @param probabilities 同形状の正規化確率。パディングは0
   */
  public record Distribution(NDArray logProbabilities, NDArray probabilities) {}
}
