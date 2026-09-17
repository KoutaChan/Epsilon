package com.epsilon.pico.ai.decision.training;

/** 判断の種類ごとに、選択行動の方策更新量へ掛ける有限の非負係数。 */
public record DecisionPolicySignalMultipliers(float dahai, float riichi, float reaction) {

  public DecisionPolicySignalMultipliers {
    requireNonNegativeFinite(dahai, "dahai");
    requireNonNegativeFinite(riichi, "riichi");
    requireNonNegativeFinite(reaction, "reaction");
    if (!(dahai > 0.0f || riichi > 0.0f || reaction > 0.0f)) {
      throw new IllegalArgumentException("at least one policy signal multiplier must be positive");
    }
  }

  public static DecisionPolicySignalMultipliers allCategories() {
    return new DecisionPolicySignalMultipliers(1.0f, 1.0f, 1.0f);
  }

  boolean uniform() {
    return Float.compare(dahai, riichi) == 0 && Float.compare(dahai, reaction) == 0;
  }

  public float forKind(EpsilonDecisionPointKind kind) {
    return switch (kind) {
      case DAHAI -> dahai;
      case RIICHI -> riichi;
      case REACTION -> reaction;
    };
  }

  private static void requireNonNegativeFinite(float value, String name) {
    if (!Float.isFinite(value) || value < 0.0f) {
      throw new IllegalArgumentException(
          "policy signal multiplier " + name + " must be finite and non-negative");
    }
  }
}
