package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/** 同じ候補軸を持つ特徴量の先頭部分を連結し、推論用の射影計算へ渡す。 */
public abstract class DecisionPolicyCandidatePrefixExecution implements AutoCloseable {

  /** 一つの方策順伝播に対応する実行枠を開始する。 */
  public abstract Forward beginForward(NDManager workingManager);

  /** 完了を証明できないデバイス処理を保持しているなら{@code true}を返す。 */
  public abstract boolean hasIncompleteWork();

  /** 実行コンテキストを利用不能にした最初の失敗を返す。 */
  public abstract Throwable failure();

  /** 同じストリーム末尾の完了確認後、失敗順伝播が保持した利用権を解放する。 */
  public abstract void releaseFailedForwardAfterCompletion();

  /** 一つの方策順伝播で使う候補先頭部分出力を所有する。 */
  public abstract static class Forward implements AutoCloseable {

    /** 先頭構成要素を連結して既存射影を一回だけ実行する。 */
    public abstract NDArray project(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray actionFeatures,
        NDArray primaryTiles,
        NDArray transitions,
        NDArray playerContexts,
        NDArray stateContext);

    /** 出力枠を最終D2H完了まで再利用不能にする依存オブジェクトを返す。 */
    public abstract AutoCloseable seal();

    /** 最終出力へ到達しなかった失敗を記録する。 */
    public abstract void poison(Throwable failure);
  }
}
