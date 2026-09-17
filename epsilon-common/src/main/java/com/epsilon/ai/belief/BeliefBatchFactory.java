package com.epsilon.ai.belief;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import java.util.List;

/** 系列の状態入力をBeliefの同期実行へ接続する。 */
public interface BeliefBatchFactory<I> {

  /** 入力を順番に連結し、指定メモリ管理オブジェクトが所有する状態テンソル二つへ転送する。 */
  NDList transfer(NDManager manager, List<I> inputs);

  /** 二つの入力を同じテンソル形状で連結できるかを返す。 */
  boolean sameBucket(I first, I second);
}
