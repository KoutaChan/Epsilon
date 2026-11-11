package com.epsilon.grp.train;

import ai.djl.engine.Engine;
import ai.djl.pytorch.jni.JniUtils;
import java.util.function.Supplier;

/**
 * 学習 1 回の間だけ、torch の演算スレッドの数を替える。スレッドの数はプロセス全体に効くので、学習が終われば（例外で終わっても）元の
 * 数に戻す。同じプロセスで 2 つの学習を同時に回すと、互いの数を上書きする。
 */
final class TorchThreads {

  private TorchThreads() {}

  /**
   * スレッドの数を替えて学習を回す。
   *
   * @param threads torch の演算スレッドの数（1 以上）
   * @param training 学習を行う処理
   * @return 学習の結果
   * @throws IllegalArgumentException スレッドの数が 1 未満のとき
   */
  static <T> T with(int threads, Supplier<T> training) {
    if (threads < 1) {
      throw new IllegalArgumentException("Training needs at least one thread: " + threads);
    }
    // スレッドの数を読む前に、torch のネイティブライブラリを読み込む
    Engine.getEngine("PyTorch");
    int previous = JniUtils.getNumThreads();
    JniUtils.setNumThreads(threads);
    try {
      return training.get();
    } finally {
      JniUtils.setNumThreads(previous);
    }
  }
}
