package com.epsilon.prism.hip;

/**
 * 学習で DJL の自動微分の外に置く背骨の演算の組: 関係の偏りつきの注意と RMSNorm。GPU は {@link HipAttention}。
 */
public interface TrainingKernels extends RelationAttention, RmsNormKernels {}
