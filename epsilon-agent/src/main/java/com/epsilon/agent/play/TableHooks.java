package com.epsilon.agent.play;

/**
 * 系列が卓に差し込む処理。卓は半荘の主経路を作るたびに、経路の処理（{@link RunHooks}）を 1 つ作らせる。写しと分岐の処理は、親の
 * 経路の処理が作って {@link Run#fork} に渡す。
 *
 * @param <Q> 問い合わせの型
 */
public interface TableHooks<Q extends Query> {

  /**
   * 半荘の主経路の処理を作る。主経路を作った進行のスレッドで、最初の判断の時点へ進める前に 1 度呼ぶ。
   *
   * @param run 主経路（半荘だけが決まっている）
   */
  RunHooks<Q> main(Run<Q> run);

  /** 系列の受け取り先の書き込みが遅れていて、新しい半荘を始めないなら true。走っている半荘があるときだけ効く。ドライバのスレッドで呼ぶ。 */
  default boolean backlogged() {
    return false;
  }

  /**
   * 卓を打ち切るなら true。一度 true を返したら、その後も true を返す。打ち切った後は新しい半荘を始めず、走っている半荘が全て終われば卓を
   * 終える。ドライバのスレッドで呼ぶ。
   */
  default boolean stopped() {
    return false;
  }

  /**
   * 始めてよい半荘の数（計画の並びの先頭から数える）。並びのそれより後の半荘は、数が増えるまで始めない。ドライバのスレッドで、半荘を
   * 始める前に毎回呼ぶ。数は減らさない。走っている半荘が全て終わるまでに数を増やすか打ち切らないと、卓は止まったとみなして失敗する。
   */
  default int startable() {
    return Integer.MAX_VALUE;
  }

  /** 系列の処理を持たない卓（評価・牌譜）の処理。問い合わせは {@link Query} のままで、記録も分岐も作らない。 */
  static TableHooks<Query> plain() {
    return PlainHooks.HOOKS;
  }
}
