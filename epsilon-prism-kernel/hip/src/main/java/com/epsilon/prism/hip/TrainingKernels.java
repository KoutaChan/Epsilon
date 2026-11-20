package com.epsilon.prism.hip;

/**
 * 学習のとき DJL の自動微分の外で計算するバックボーンの演算。関係のバイアスつきの Attention と RMSNorm をまとめる。GPU の実装は
 * {@link HipAttention}。
 */
public interface TrainingKernels extends RelationAttention, RmsNormKernels {
  /** 候補の演算があれば返す。対応しない実装では eager の候補層を使う。 */
  default CandidateKernels candidateKernels() {
    return null;
  }
}
