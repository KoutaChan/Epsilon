package com.epsilon.prism.kernel;

/**
 * 学習のとき DJL の自動微分の外で計算するバックボーンの演算。関係のバイアスつきの Attention と RMSNorm をまとめる。実装が作業領域を
 * 持つときは、使い終わったら {@link #close} で解放する。
 */
public interface TrainingKernels extends RelationAttention, RmsNormKernels, AutoCloseable {
  /** 候補の演算があれば返す。対応しない実装では eager の候補層を使う。 */
  default CandidateKernels candidateKernels() {
    return null;
  }

  /** 実装が持つ作業領域を解放する。提出した演算が終わった後に呼ぶ。 */
  @Override
  default void close() {}
}
