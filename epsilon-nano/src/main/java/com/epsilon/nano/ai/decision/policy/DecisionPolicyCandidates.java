package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import com.epsilon.core.Action;
import com.epsilon.nano.ai.decision.input.DecisionFeatureCodec;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;

/** 合法手の種類・分岐・打牌に対応するマスクを構築し、方策出力層と方策グラフで共有する。 */
public final class DecisionPolicyCandidates {

  private static final int TYPE_COUNT = Action.Type.values().length;
  private static final int GROUP_COUNT = Action.Group.values().length;
  private static final int RESPONSE_MASK_INDEX = GROUP_COUNT;
  private static final int[] TYPE_MASK_FIELDS =
      maskFields(TYPE_COUNT, DecisionInputSchema.ActionInt.TYPE);
  private static final long[] TYPE_CATEGORY_SETS = typeCategorySets();
  private static final int[] GROUP_MASK_FIELDS =
      maskFields(GROUP_COUNT + 1, DecisionInputSchema.ActionInt.GROUP);
  private static final long[] GROUP_CATEGORY_SETS = groupCategorySets();

  private DecisionPolicyCandidates() {}

  /**
   * 一つの順伝播で共有する行動種類/グループマスクをまとめて構築する。
   *
   * <p>種類とグループ・応答所属を二つの構造化した演算で直接生成する。種類マスクはFusion入力へそのまま渡すため、 独立した連続テンソルとして保持する。返す値は従来の{@link
   * #typeMask}・{@link #groupMask}・{@link #responseMask}と同一で、参照先の対応メタデータへ勾配は流さない。
   */
  static CandidateMasks candidateMasks(NDArray actionCategories, NDArray eligibleMask) {
    return new CandidateMasks(
        NDArrays.categoricalMasks(
                actionCategories, eligibleMask, TYPE_MASK_FIELDS, TYPE_CATEGORY_SETS)
            .stopGradient(),
        NDArrays.categoricalMasks(
                actionCategories, eligibleMask, GROUP_MASK_FIELDS, GROUP_CATEGORY_SETS)
            .stopGradient());
  }

  /** 指定した行動種類に属する合法候補を1とするマスクを返す。 */
  static NDArray typeMask(NDArray actionTypes, NDArray eligibleMask, Action.Type type) {
    return actionTypes
        .eq(DecisionFeatureCodec.actionType(type))
        .toType(eligibleMask.getDataType(), false)
        .mul(eligibleMask)
        .stopGradient();
  }

  /** 指定した行動グループに属する合法候補を1とするマスクを返す。 */
  static NDArray groupMask(NDArray actionGroups, NDArray eligibleMask, Action.Group group) {
    return actionGroups
        .eq(DecisionFeatureCodec.actionGroup(group))
        .toType(eligibleMask.getDataType(), false)
        .mul(eligibleMask)
        .stopGradient();
  }

  /** 他家の打牌・加槓に対する合法応答候補を1とするマスクを返す。 */
  static NDArray responseMask(NDArray actionGroups, NDArray eligibleMask) {
    NDArray responseMask = eligibleMask.mul(0.0f);
    for (Action.Group group : Action.Group.values()) {
      if (group.isResponse()) {
        responseMask = responseMask.add(groupMask(actionGroups, eligibleMask, group));
      }
    }
    return responseMask.stopGradient();
  }

  /**
   * CONTINUE後の「物理打牌識別情報 → DAMA/RIICHI」階層を構成するマスクを返す。
   *
   * <p>同一識別情報にDAMAがあればDAMA 枠を識別情報代表とし、RIICHIしか合法でない場合だけRIICHI 枠を代表にする。
   */
  static DiscardChoices discardChoices(
      NDArray actionRoutes, NDArray actionTypes, NDArray eligibleMask) {
    NDArray damaMask = typeMask(actionTypes, eligibleMask, Action.Type.DAHAI);
    NDArray riichiMask = typeMask(actionTypes, eligibleMask, Action.Type.RIICHI_DAHAI);
    return buildDiscardChoices(actionRoutes, damaMask, riichiMask);
  }

  static DiscardChoices discardChoices(NDArray actionRoutes, CandidateMasks masks) {
    NDArray damaMask = masks.type(Action.Type.DAHAI);
    NDArray riichiMask = masks.type(Action.Type.RIICHI_DAHAI);
    return buildDiscardChoices(actionRoutes, damaMask, riichiMask);
  }

  private static DiscardChoices buildDiscardChoices(
      NDArray actionRoutes, NDArray damaMask, NDArray riichiMask) {
    NDArray masks =
        NDArrays.binaryChoiceMasks(
                actionRoutes,
                damaMask,
                riichiMask,
                DecisionInputSchema.ActionRoute.DISCARD_IDENTITY_REPRESENTATIVE.ordinal(),
                DecisionInputSchema.ActionRoute.DAMA_SLOT.ordinal(),
                DecisionInputSchema.ActionRoute.RIICHI_SLOT.ordinal(),
                DecisionInputSchema.PAD_ID)
            .stopGradient();
    NDArray discardMask = masks.get("...,0");
    NDArray representativeSlots =
        route(actionRoutes, DecisionInputSchema.ActionRoute.DISCARD_REPRESENTATIVE_SLOT);
    NDArray damaSlots = route(actionRoutes, DecisionInputSchema.ActionRoute.DAMA_SLOT);
    NDArray riichiSlots = route(actionRoutes, DecisionInputSchema.ActionRoute.RIICHI_SLOT);
    return new DiscardChoices(
        damaMask,
        riichiMask,
        discardMask,
        representativeSlots,
        damaSlots,
        riichiSlots,
        masks.get("...,1"),
        masks.get("...,2"),
        masks.get("...,3"));
  }

  private static NDArray route(NDArray actionRoutes, DecisionInputSchema.ActionRoute route) {
    return actionRoutes.get("...,{}", route.ordinal()).stopGradient();
  }

  private static int[] maskFields(int count, DecisionInputSchema.ActionInt field) {
    int[] fields = new int[count];
    for (int index = 0; index < count; index++) {
      fields[index] = field.ordinal();
    }
    return fields;
  }

  private static long[] typeCategorySets() {
    long[] categorySets = new long[TYPE_COUNT];
    for (Action.Type type : Action.Type.values()) {
      categorySets[type.ordinal()] = 1L << DecisionFeatureCodec.actionType(type);
    }
    return categorySets;
  }

  private static long[] groupCategorySets() {
    long[] categorySets = new long[GROUP_COUNT + 1];
    long responseCategories = 0L;
    for (Action.Group group : Action.Group.values()) {
      long category = 1L << DecisionFeatureCodec.actionGroup(group);
      categorySets[group.ordinal()] = category;
      if (group.isResponse()) {
        responseCategories |= category;
      }
    }
    categorySets[RESPONSE_MASK_INDEX] = responseCategories;
    return categorySets;
  }

  /** 同一順伝播で共有する種類/グループ別合法候補マスク。 */
  public record CandidateMasks(NDArray typeMasks, NDArray groupAndResponseMasks) {

    NDArray type(Action.Type type) {
      return typeMasks.get("...,{}", type.ordinal());
    }

    NDArray group(Action.Group group) {
      return groupAndResponseMasks.get("...,{}", group.ordinal());
    }

    NDArray response() {
      return groupAndResponseMasks.get("...,{}", RESPONSE_MASK_INDEX);
    }
  }

  /**
   * 物理打牌識別情報とDAMA/RIICHI分岐を表す不変マスク群。
   *
   * @param damaMask {@code [batch, action]}のDAMA候補マスク
   * @param riichiMask {@code [batch, action]}のRIICHI候補マスク
   * @param discardMask DAMAとRIICHIの和
   * @param representativeSlots 各行動と同一識別情報の代表枠 + 1
   * @param damaSlots 各行動と同一識別情報のDAMA 枠 + 1。存在しなければ0
   * @param riichiSlots 各行動と同一識別情報のRIICHI 枠 + 1。存在しなければ0
   * @param damaPresent 各行動識別情報にDAMA候補が存在する場合1
   * @param riichiPresent 各行動識別情報にRIICHI候補が存在する場合1
   * @param representativeMask 各物理打牌識別情報につき一つだけ立つ代表枠マスク
   */
  record DiscardChoices(
      NDArray damaMask,
      NDArray riichiMask,
      NDArray discardMask,
      NDArray representativeSlots,
      NDArray damaSlots,
      NDArray riichiSlots,
      NDArray damaPresent,
      NDArray riichiPresent,
      NDArray representativeMask) {}
}
