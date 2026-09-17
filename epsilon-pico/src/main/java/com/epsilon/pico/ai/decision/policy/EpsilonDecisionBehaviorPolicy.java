package com.epsilon.pico.ai.decision.policy;

import com.epsilon.config.settings.DecisionFullSupportSettings;
import com.epsilon.core.Action;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionDataException;
import java.util.Arrays;
import java.util.List;

/**
 * モデルの合法手確率に、条件付き階層の各分岐での探索と、各合法手の確率の下限を適用する。
 *
 * <p>上位分岐に割り当てる探索確率は、下位の具体候補の数に依存しない。最大40個の合法手を1つのビットマスクで扱い、推論ごとのコレクション生成を避ける。正規化は学習時の方策と同じ階層に従う。
 */
public final class EpsilonDecisionBehaviorPolicy {

  private static final float POLICY_EPSILON = 1.0e-8f;
  private static final float POLICY_SUM_TOLERANCE = 1.0e-5f;

  private EpsilonDecisionBehaviorPolicy() {}

  /**
   * 探索を加えず、丸め誤差による0だけを正値化した標準形式の方策を無作為抽出する。
   *
   * @param policyProbability モデルが返した正規化済み末端の行動確率
   * @return 入力を複製した探索適用後の・探索前の方策
   */
  public static Mixture direct(float[] policyProbability) {
    return directInPlace(policyProbability.clone());
  }

  /**
   * 呼び出し側が所有権を渡した配列をそのまま標準形式の方策として使う。
   *
   * <p>配列は正値化・正規化され、返す探索適用後のと対局生成は同一の不変配列を共有する。
   *
   * @param ownedPolicyProbability この呼び出しへ所有権を移す末端の行動確率配列
   * @return 同一配列を探索適用後の・対局生成に持つmixture
   */
  public static Mixture directInPlace(float[] ownedPolicyProbability) {
    normalizeCanonicalPolicy(ownedPolicyProbability);
    return new Mixture(ownedPolicyProbability, ownedPolicyProbability);
  }

  /**
   * 合法行動と節点別探索率に従って末端の行動実際の行動選択に使う方策を構築する。
   *
   * @param policyProbability モデルが返した正規化済み末端の行動確率
   * @param legalActions 確率位置と同順の合法行動
   * @param config 方策グラフ節点別の探索確率の割り当て
   * @return 探索後探索適用後のと探索前対局生成のmixture
   */
  public static Mixture distribution(
      float[] policyProbability, List<Action> legalActions, DecisionFullSupportSettings config) {
    return distribution(policyProbability, legalActions, config, 1.0f);
  }

  /**
   * 合法行動と節点別探索率へ判断固有の倍率を掛けて末端の行動実際の行動選択に使う方策を構築する。
   *
   * <p>倍率は節点探索確率の割り当てだけへ適用し、末端の行動確率下限は固定する。
   */
  public static Mixture distribution(
      float[] policyProbability,
      List<Action> legalActions,
      DecisionFullSupportSettings config,
      float explorationMultiplier) {
    return distributionInPlace(
        policyProbability.clone(), legalActions, config, explorationMultiplier);
  }

  /**
   * 呼び出し側が所有権を渡した標準形式の方策へFULL_SUPPORTを適用する。
   *
   * <p>入力配列を正値化・正規化して探索前の方策として再利用し、判断ごとの余分な配列複製を作らない。
   *
   * @param ownedPolicyProbability この呼び出しへ所有権を移す末端の行動確率配列
   * @param legalActions 確率位置と同順の合法行動
   * @param config 方策グラフ節点別の探索確率の割り当て
   * @return 新規探索適用後の配列と所有権移譲された対局生成配列のmixture
   */
  public static Mixture distributionInPlace(
      float[] ownedPolicyProbability,
      List<Action> legalActions,
      DecisionFullSupportSettings config) {
    return distributionInPlace(ownedPolicyProbability, legalActions, config, 1.0f);
  }

  /**
   * 呼び出し側が所有権を渡した標準形式の方策へ、判断固有倍率付きFULL_SUPPORTを適用する。
   *
   * @param explorationMultiplier 全方策グラフ節点の探索確率の割り当てへ掛ける有限の非負倍率
   */
  public static Mixture distributionInPlace(
      float[] ownedPolicyProbability,
      List<Action> legalActions,
      DecisionFullSupportSettings config,
      float explorationMultiplier) {
    int actionCount = legalActions.size();
    if (ownedPolicyProbability.length != actionCount
        || actionCount == 0
        || actionCount > Long.SIZE) {
      throw new IllegalArgumentException("policy/legal action size mismatch");
    }
    if (!Float.isFinite(explorationMultiplier) || explorationMultiplier < 0.0f) {
      throw new IllegalArgumentException("explorationMultiplier must be finite and non-negative");
    }
    normalizeCanonicalPolicy(ownedPolicyProbability);
    float[] behavior = new float[actionCount];
    long all = actionCount == Long.SIZE ? -1L : (1L << actionCount) - 1L;
    boolean response = requireConsistentActionContext(legalActions);
    if (response) {
      assignResponse(
          behavior, ownedPolicyProbability, legalActions, all, config, explorationMultiplier);
    } else {
      assignTurn(
          behavior, ownedPolicyProbability, legalActions, all, config, explorationMultiplier);
    }
    normalizeAssignedPolicy(behavior);
    applyMinimumLeafProbability(behavior, config.minimumLeafProbability());
    return new Mixture(behavior, ownedPolicyProbability);
  }

  /** 階層節点の条件付き確率が掛け合わさっても、各合法末端の行動を有限頻度で選択できる下限を付与する。 */
  private static void applyMinimumLeafProbability(float[] behavior, float minimum) {
    if (minimum == 0.0f) {
      return;
    }
    float retainedPolicyMass = 1.0f - minimum * behavior.length;
    if (!(retainedPolicyMass > 0.0f)) {
      throw new IllegalArgumentException(
          "minimumLeafProbability leaves no policy mass for " + behavior.length + " actions");
    }
    for (int index = 0; index < behavior.length; index++) {
      behavior[index] = minimum + retainedPolicyMass * behavior[index];
    }
  }

  private static void assignResponse(
      float[] behavior,
      float[] canonical,
      List<Action> actions,
      long all,
      DecisionFullSupportSettings config,
      float explorationMultiplier) {
    long ron = slotsOfType(actions, all, Action.Type.RON_AGARI);
    long declined = all & ~ron;
    long pass = slotsOfType(actions, declined, Action.Type.PASS);
    long meld = declined & ~pass;
    long chi = slotsOfType(actions, meld, Action.Type.CHI);
    long pon = slotsOfType(actions, meld, Action.Type.PON);
    long daiminkan = slotsOfType(actions, meld, Action.Type.DAIMINKAN);

    float ronAccepted =
        binaryAccepted(
            canonical,
            ron,
            declined,
            scaled(config.terminalGateExplorationMass(), explorationMultiplier));
    float ronDeclined = 1.0f - ronAccepted;
    assignVariants(behavior, canonical, ron, ronAccepted, 0.0f);

    float callAccepted =
        binaryAccepted(
            canonical, meld, pass, scaled(config.callGateExplorationMass(), explorationMultiplier));
    assignVariants(behavior, canonical, pass, ronDeclined * (1.0f - callAccepted), 0.0f);

    float typeMass = mass(canonical, meld);
    int typeCount = presentCount(chi, pon, daiminkan);
    float callBranch = ronDeclined * callAccepted;
    assignVariants(
        behavior,
        canonical,
        chi,
        callBranch
            * semanticProbability(
                canonical,
                chi,
                typeMass,
                typeCount,
                scaled(config.meldTypeExplorationMass(), explorationMultiplier)),
        scaled(config.meldCandidateExplorationMass(), explorationMultiplier));
    assignVariants(
        behavior,
        canonical,
        pon,
        callBranch
            * semanticProbability(
                canonical,
                pon,
                typeMass,
                typeCount,
                scaled(config.meldTypeExplorationMass(), explorationMultiplier)),
        scaled(config.meldCandidateExplorationMass(), explorationMultiplier));
    assignVariants(
        behavior,
        canonical,
        daiminkan,
        callBranch
            * semanticProbability(
                canonical,
                daiminkan,
                typeMass,
                typeCount,
                scaled(config.meldTypeExplorationMass(), explorationMultiplier)),
        scaled(config.meldCandidateExplorationMass(), explorationMultiplier));
  }

  private static void assignTurn(
      float[] behavior,
      float[] canonical,
      List<Action> actions,
      long all,
      DecisionFullSupportSettings config,
      float explorationMultiplier) {
    long tsumo = slotsOfType(actions, all, Action.Type.TSUMO_AGARI);
    long afterTsumo = all & ~tsumo;
    float tsumoAccepted =
        binaryAccepted(
            canonical,
            tsumo,
            afterTsumo,
            scaled(config.terminalGateExplorationMass(), explorationMultiplier));
    assignVariants(behavior, canonical, tsumo, tsumoAccepted, 0.0f);

    long kyushu = slotsOfType(actions, afterTsumo, Action.Type.KYUSHU_KYUHAI);
    long afterKyushu = afterTsumo & ~kyushu;
    float kyushuAccepted =
        binaryAccepted(
            canonical,
            kyushu,
            afterKyushu,
            scaled(config.terminalGateExplorationMass(), explorationMultiplier));
    assignVariants(behavior, canonical, kyushu, (1.0f - tsumoAccepted) * kyushuAccepted, 0.0f);

    long discards = slotsOfDiscard(actions, afterKyushu);
    long ankan = slotsOfType(actions, afterKyushu, Action.Type.ANKAN);
    long kakan = slotsOfType(actions, afterKyushu, Action.Type.KAKAN);
    long kan = ankan | kakan;
    float kanAccepted =
        binaryAccepted(
            canonical,
            kan,
            discards,
            scaled(config.kanGateExplorationMass(), explorationMultiplier));
    float kanTypeMass = mass(canonical, kan);
    int kanTypeCount = presentCount(ankan, kakan);
    float turnPath = (1.0f - tsumoAccepted) * (1.0f - kyushuAccepted);
    assignVariants(
        behavior,
        canonical,
        ankan,
        turnPath
            * kanAccepted
            * semanticProbability(
                canonical,
                ankan,
                kanTypeMass,
                kanTypeCount,
                scaled(config.kanTypeExplorationMass(), explorationMultiplier)),
        scaled(config.kanCandidateExplorationMass(), explorationMultiplier));
    assignVariants(
        behavior,
        canonical,
        kakan,
        turnPath
            * kanAccepted
            * semanticProbability(
                canonical,
                kakan,
                kanTypeMass,
                kanTypeCount,
                scaled(config.kanTypeExplorationMass(), explorationMultiplier)),
        scaled(config.kanCandidateExplorationMass(), explorationMultiplier));
    assignDiscards(
        behavior,
        canonical,
        actions,
        discards,
        turnPath * (1.0f - kanAccepted),
        config,
        explorationMultiplier);
  }

  private static void assignDiscards(
      float[] behavior,
      float[] canonical,
      List<Action> actions,
      long discardSlots,
      float branchProbability,
      DecisionFullSupportSettings config,
      float explorationMultiplier) {
    int identityCount = countDiscardIdentities(actions, discardSlots);
    float identityMass = mass(canonical, discardSlots);
    long remaining = discardSlots;
    while (remaining != 0L) {
      long slots = discardIdentitySlots(actions, remaining);
      long dama = slotsOfType(actions, slots, Action.Type.DAHAI);
      long riichi = slotsOfType(actions, slots, Action.Type.RIICHI_DAHAI);
      float identityProbability =
          semanticProbability(
              canonical,
              slots,
              identityMass,
              identityCount,
              scaled(config.discardIdentityExplorationMass(), explorationMultiplier));
      float riichiAccepted =
          binaryAccepted(
              canonical,
              riichi,
              dama,
              scaled(config.riichiGateExplorationMass(), explorationMultiplier));
      assignVariants(
          behavior,
          canonical,
          riichi,
          branchProbability * identityProbability * riichiAccepted,
          0.0f);
      assignVariants(
          behavior,
          canonical,
          dama,
          branchProbability * identityProbability * (1.0f - riichiAccepted),
          0.0f);
      remaining &= ~slots;
    }
  }

  /** 受諾側の条件付き確率。存在しない側へ確率の割り当てを割り当てない。 */
  private static float binaryAccepted(
      float[] canonical, long accepted, long declined, float explorationMass) {
    if (accepted == 0L) {
      return 0.0f;
    }
    if (declined == 0L) {
      return 1.0f;
    }
    float acceptedMass = mass(canonical, accepted);
    float total = acceptedMass + mass(canonical, declined);
    return mix(acceptedMass / total, 0.5f, explorationMass);
  }

  private static float semanticProbability(
      float[] canonical, long slots, float totalMass, int alternativeCount, float explorationMass) {
    if (slots == 0L) {
      return 0.0f;
    }
    return mix(mass(canonical, slots) / totalMass, 1.0f / alternativeCount, explorationMass);
  }

  private static void assignVariants(
      float[] behavior,
      float[] canonical,
      long slots,
      float branchProbability,
      float explorationMass) {
    if (slots == 0L) {
      return;
    }
    float canonicalMass = mass(canonical, slots);
    float probe = 1.0f / Long.bitCount(slots);
    for (long remaining = slots; remaining != 0L; remaining &= remaining - 1L) {
      int slot = Long.numberOfTrailingZeros(remaining);
      float conditional = canonical[slot] / canonicalMass;
      behavior[slot] = branchProbability * mix(conditional, probe, explorationMass);
    }
  }

  private static float mix(float policy, float probe, float explorationMass) {
    return policy * (1.0f - explorationMass) + probe * explorationMass;
  }

  private static float scaled(float explorationMass, float multiplier) {
    float result = explorationMass * multiplier;
    if (!Float.isFinite(result) || result < 0.0f || result >= 1.0f) {
      throw new IllegalArgumentException(
          "scaled exploration mass must be finite and in [0, 1): " + result);
    }
    return result;
  }

  private static void normalizeCanonicalPolicy(float[] values) {
    if (values.length == 0) {
      throw new IllegalArgumentException("policyProbability must not be empty");
    }
    double originalSum = 0.0;
    for (float value : values) {
      if (!Float.isFinite(value) || value < 0.0f) {
        throw new EpsilonDecisionDataException(
            "policy contains an invalid probability: " + Arrays.toString(values));
      }
      originalSum += value;
    }
    if (Math.abs(originalSum - 1.0) > POLICY_SUM_TOLERANCE) {
      throw new EpsilonDecisionDataException("canonical policy sum must be 1: " + originalSum);
    }
    double positiveSum = 0.0;
    for (int index = 0; index < values.length; index++) {
      values[index] = Math.max(POLICY_EPSILON, values[index]);
      positiveSum += values[index];
    }
    for (int index = 0; index < values.length; index++) {
      values[index] /= (float) positiveSum;
    }
  }

  private static void normalizeAssignedPolicy(float[] values) {
    double sum = 0.0;
    for (float value : values) {
      if (!(value > 0.0f) || !Float.isFinite(value)) {
        throw new EpsilonDecisionDataException(
            "Policy Graph left an invalid leaf probability: " + Arrays.toString(values));
      }
      sum += value;
    }
    if (Math.abs(sum - 1.0) > POLICY_SUM_TOLERANCE) {
      throw new EpsilonDecisionDataException(
          "Policy Graph leaf probability mass must be 1: " + sum);
    }
    for (int index = 0; index < values.length; index++) {
      values[index] /= (float) sum;
    }
  }

  private static float mass(float[] probability, long slots) {
    float result = 0.0f;
    for (long remaining = slots; remaining != 0L; remaining &= remaining - 1L) {
      result += probability[Long.numberOfTrailingZeros(remaining)];
    }
    return result;
  }

  private static long slotsOfType(List<Action> actions, long eligible, Action.Type type) {
    long result = 0L;
    for (long remaining = eligible; remaining != 0L; remaining &= remaining - 1L) {
      int slot = Long.numberOfTrailingZeros(remaining);
      if (actions.get(slot).type() == type) {
        result |= 1L << slot;
      }
    }
    return result;
  }

  private static long slotsOfDiscard(List<Action> actions, long eligible) {
    long result = 0L;
    for (long remaining = eligible; remaining != 0L; remaining &= remaining - 1L) {
      int slot = Long.numberOfTrailingZeros(remaining);
      Action.Type type = actions.get(slot).type();
      if (type == Action.Type.DAHAI || type == Action.Type.RIICHI_DAHAI) {
        result |= 1L << slot;
      }
    }
    return result;
  }

  private static int presentCount(long first, long second) {
    return (first != 0L ? 1 : 0) + (second != 0L ? 1 : 0);
  }

  private static int presentCount(long first, long second, long third) {
    return presentCount(first, second) + (third != 0L ? 1 : 0);
  }

  private static int countDiscardIdentities(List<Action> actions, long discardSlots) {
    int count = 0;
    for (long remaining = discardSlots; remaining != 0L; count++) {
      remaining &= ~discardIdentitySlots(actions, remaining);
    }
    return count;
  }

  private static long discardIdentitySlots(List<Action> actions, long eligible) {
    int firstSlot = Long.numberOfTrailingZeros(eligible);
    int identity = actions.get(firstSlot).discardIdentityIndex();
    long result = 0L;
    for (long remaining = eligible; remaining != 0L; remaining &= remaining - 1L) {
      int slot = Long.numberOfTrailingZeros(remaining);
      if (actions.get(slot).discardIdentityIndex() == identity) {
        result |= 1L << slot;
      }
    }
    return result;
  }

  private static boolean requireConsistentActionContext(List<Action> legalActions) {
    boolean response = legalActions.getFirst().type().isResponse();
    for (int slot = 1; slot < legalActions.size(); slot++) {
      if (legalActions.get(slot).type().isResponse() != response) {
        throw new IllegalArgumentException(
            "turn and response actions cannot share a policy: slot=" + slot);
      }
    }
    return response;
  }

  /**
   * 無作為抽出用探索適用後のと探索前探索前の方策の組。
   *
   * @param behaviorPolicy FULL_SUPPORT を適用した実無作為抽出分布
   * @param rolloutPolicy 探索を加える前のモデル末端の行動分布
   */
  public record Mixture(float[] behaviorPolicy, float[] rolloutPolicy) {

    /** 探索適用後のと対局生成の枠数が一致することを検証する。 */
    public Mixture {
      if (behaviorPolicy.length != rolloutPolicy.length) {
        throw new IllegalArgumentException("policy mixture shape mismatch");
      }
    }
  }
}
