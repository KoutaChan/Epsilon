package com.epsilon.agent.run;

/**
 * 段の宣言。run の種類は段を enum で宣言し、宣言の順に回す。各定数が自分の段の中身を作る。
 *
 * @param <I> 反復
 */
public interface StepKind<I extends Iteration> {

  /** この反復の段を作る。ファイルには触らない。 */
  Step<I> create(I iteration);

  /** 段の報告に載せる数値の型。状態・秒・起動の行・jar と同じ名前のコンポーネントを持たない record にする。 */
  Class<? extends Record> metrics();

  /**
   * この段が済んだ反復を、固定した jar でしか続けられないか。結果が jar に依らない段（同じ入力から同じ答えを出し直せる段）は false に
   * する。そういう段だけが済んだ反復は、新しい jar で固定し直してよい。
   */
  default boolean jarBound() {
    return true;
  }
}
