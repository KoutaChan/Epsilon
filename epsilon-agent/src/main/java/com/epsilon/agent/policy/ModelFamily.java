package com.epsilon.agent.policy;

import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ServiceLoader;

/**
 * 方策を開く入口。系列ごとに 1 つある。実装は META-INF/services/com.epsilon.agent.policy.ModelFamily に登録し、{@link #of} が
 * ServiceLoader で見つける。epsilon-agent は系列のモジュールに依存しないので、PRISM・classic の実装は実行時のクラスパスから読む。
 */
public interface ModelFamily {

  Series series();

  /**
   * チェックポイントの方策を、指定したルールで開く。ルールかフィンガープリントがチェックポイントと合わなければ、IllegalStateException を
   * 投げる。
   *
   * @param checkpoint 開くチェックポイント（チェックポイントを取らない系列は null）
   * @param variant 打つルール
   * @param options 方策を開く設定
   * @param runtime 呼び出し側が持つ推論のランタイム。GPU の方策は、このランタイムの推論のパイプライン（{@link NetworkRuntime#lease}）と
   *     受付（{@link NetworkRuntime#dispatcher}）を使う。null なら方策が自分のランタイムを持つ。渡したランタイムの所有者は呼び出し側で、
   *     方策を閉じた後に呼び出し側が閉じる
   * @return 開いた方策
   */
  Policy open(Path checkpoint, Variant variant, PolicyOptions options, NetworkRuntime runtime) throws IOException;

  /**
   * 推論のランタイムを方策が自分で持つ形で開く。CPU の方策や、1 判断ずつ評価する呼び出し側が使う。
   *
   * @param checkpoint 開くチェックポイント（チェックポイントを取らない系列は null）
   * @param variant 打つルール
   * @param options 方策を開く設定
   * @return 開いた方策
   */
  default Policy open(Path checkpoint, Variant variant, PolicyOptions options) throws IOException {
    return open(checkpoint, variant, options, null);
  }

  /** 系列の実装をクラスパスから探す。見つからないか、2 つ以上あれば IllegalStateException を投げる。 */
  static ModelFamily of(Series series) {
    ModelFamily found = null;
    for (ModelFamily family : ServiceLoader.load(ModelFamily.class)) {
      if (family.series() != series) {
        continue;
      }
      if (found != null) {
        throw new IllegalStateException(
            "Two model families serve " + series + ": " + found.getClass().getName() + " and "
                + family.getClass().getName());
      }
      found = family;
    }
    if (found == null) {
      throw new IllegalStateException("No model family serves " + series + " on the classpath");
    }
    return found;
  }
}
