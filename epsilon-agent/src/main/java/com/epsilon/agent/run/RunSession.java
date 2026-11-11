package com.epsilon.agent.run;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * run のリソースと、段を回す側（{@link Runner}）が呼ぶ操作。JVM が動いている間 1 つだけ作る。閉じるとリソースを閉じる。
 *
 * @param <R> run 全体の設定
 * @param <S> state.json
 * @param <I> 反復
 */
public interface RunSession<R extends Record, S extends Record & Progress, I extends Iteration> extends AutoCloseable {

  /**
   * state.json が無い run の始めの state を作る（次の反復は 1）。バージョン 1 を置く。途中で落ちてもう一度呼ばれても、同じ結果にする。
   *
   * @param settings 最新の設定の行の値
   */
  S begin(R settings) throws Exception;

  /** 反復を作る。ファイルには触らない（同じ反復を何度作ってもよい）。 */
  I iteration(IterationFile fixed, R settings, S state);

  /**
   * 反復を閉じるための値を作る（次の state、報告に書く持ち越し、ほかの報告の行）。ファイルには書かない。
   *
   * @param steps 段の名前ごとの報告（宣言の順）
   */
  Closing<S> close(I iteration, S state, Map<String, StepReport> steps) throws Exception;

  /** ジョブの中身を回すもの。 */
  JobRunner jobRunner();

  /** 残す範囲より古い出力（古いコホートなど）を消す。消し損ねても、次の起動でもう一度呼ばれる。 */
  void clean(S state) throws IOException;

  /** GPU ごとの VRAM の使用量（MiB、デバイスの順）。GPU を使わなければ空。 */
  default List<Long> vramMiB() {
    return List.of();
  }

  /** 反復ごとの要約の 1 行に足す文（人が読む）。 */
  default String summary(IterationReport report) {
    return "";
  }
}
