package com.epsilon.agent.run;

import java.io.IOException;
import java.util.Optional;

/**
 * 反復の中の 1 つの段の中身。段はアトミックに置く完了マーカーを 1 つ持ち、済んだ段は二度と回さない。段が書くのは自分の出力（作業
 * ディレクトリ、コホート、バージョン）だけで、報告のディレクトリには書かない。後の段が使う値は、報告からでなく state の持ち越しから読む。
 *
 * @param <I> 反復
 */
public interface Step<I extends Iteration> {

  /** 反復に固定した設定で、この段を行うか。行わない段の報告は SKIPPED になる。 */
  default boolean enabled(I iteration) {
    return true;
  }

  /**
   * 完了マーカーがあれば、それと一緒に保存した報告を返す。完了マーカーを置いた後の仕上げが残っていれば、ここで済ませる（何度呼んでも
   * 同じ結果になる）。
   */
  Optional<StepReport> done(I iteration) throws IOException;

  /**
   * 書きかけの出力（{@code .part}）を消してから段を行い、報告を出力の中に書いて、完了マーカーをアトミックに置く
   * （{@link DurableFiles#seal}）。
   *
   * @return 段の報告（{@link StepReport#ran}）
   * @throws Halted 段の中で待っている間に止まると決まった（完了マーカーを置かずに抜ける）
   */
  StepReport run(I iteration) throws Exception;
}
