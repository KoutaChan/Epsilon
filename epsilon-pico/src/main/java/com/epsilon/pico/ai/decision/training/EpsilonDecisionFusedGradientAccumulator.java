package com.epsilon.pico.ai.decision.training;

import com.epsilon.pico.ai.decision.data.EpsilonDecisionDataException;

/** 同じ逆伝播で得た勾配を、方策と価値関数の独立した最適化処理へ渡すために、学習重みと勾配の倍率を管理する。 */
final class EpsilonDecisionFusedGradientAccumulator {

  private final int configuredSamples;
  private int samples;
  private double actorMass;
  private double valueMass;

  EpsilonDecisionFusedGradientAccumulator(int configuredSamples) {
    if (configuredSamples <= 0) {
      throw new IllegalArgumentException("configuredSamples must be positive");
    }
    this.configuredSamples = configuredSamples;
  }

  int configuredSamples() {
    return configuredSamples;
  }

  void observeMicroBatch(int sampleCount, double observedActorMass, double observedValueMass) {
    if (sampleCount <= 0) {
      throw new EpsilonDecisionDataException(
          "fused Actor/Value micro-batch sample count must be positive: " + sampleCount);
    }
    requireNonNegativeFiniteMass("Actor", observedActorMass);
    requireNonNegativeFiniteMass("Value", observedValueMass);
    int nextSamples;
    try {
      nextSamples = Math.addExact(samples, sampleCount);
    } catch (ArithmeticException overflow) {
      throw new EpsilonDecisionDataException(
          "fused Actor/Value sample count overflow: current=" + samples + " added=" + sampleCount,
          overflow);
    }
    if (nextSamples > configuredSamples) {
      throw new EpsilonDecisionDataException(
          "fused Actor/Value sample count exceeds configured count: actual="
              + nextSamples
              + " configured="
              + configuredSamples);
    }
    double nextActorMass = actorMass + observedActorMass;
    double nextValueMass = valueMass + observedValueMass;
    requireNonNegativeFiniteMass("accumulated Actor", nextActorMass);
    requireNonNegativeFiniteMass("accumulated Value", nextValueMass);
    samples = nextSamples;
    actorMass = nextActorMass;
    valueMass = nextValueMass;
  }

  void requireCompleteAndValid() {
    if (samples != configuredSamples) {
      throw new EpsilonDecisionDataException(
          "fused Actor/Value sample count mismatch: actual="
              + samples
              + " configured="
              + configuredSamples);
    }
    requirePositiveFiniteMass("Actor", actorMass);
    requirePositiveFiniteMass("Value", valueMass);
  }

  double actorMass() {
    return actorMass;
  }

  double valueMass() {
    return valueMass;
  }

  double actorGradientScale() {
    return gradientScale("Actor", actorMass);
  }

  double valueGradientScale() {
    return gradientScale("Value", valueMass);
  }

  private double gradientScale(String owner, double mass) {
    requirePositiveFiniteMass(owner, mass);
    double scale = configuredSamples / mass;
    if (!Double.isFinite(scale) || !(scale > 0.0)) {
      throw new EpsilonDecisionDataException(
          owner + " gradient scale must be positive and finite: " + scale);
    }
    return scale;
  }

  private static void requireNonNegativeFiniteMass(String label, double mass) {
    if (!Double.isFinite(mass) || mass < 0.0) {
      throw new EpsilonDecisionDataException(
          label + " mass must be non-negative and finite: " + mass);
    }
  }

  private static void requirePositiveFiniteMass(String label, double mass) {
    if (!Double.isFinite(mass) || !(mass > 0.0)) {
      throw new EpsilonDecisionDataException(label + " mass must be positive and finite: " + mass);
    }
  }
}
