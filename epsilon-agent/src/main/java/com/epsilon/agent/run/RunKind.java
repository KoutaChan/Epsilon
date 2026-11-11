package com.epsilon.agent.run;

import java.util.List;

/**
 * 段を順に回す run の種類（PRISM の系譜、classic の campaign）。run で使う型と、公開したバージョンのジョブの決め方と、JVM が動いている間
 * 1 つだけ持つリソースの開き方を決める。
 *
 * @param <R> run 全体の設定（設定の行の中身）
 * @param <S> state.json
 * @param <I> 反復
 * @param <K> 段の宣言
 */
public interface RunKind<
    R extends Record, S extends Record & Progress, I extends Iteration, K extends Enum<K> & StepKind<I>> {

  Class<R> settings();

  Class<S> state();

  Class<K> steps();

  /** 反復の報告に書く、次の反復への持ち越しの型。 */
  Class<? extends Record> carry();

  /** 最後の反復の番号。最新の設定の行の値で決める。 */
  int iterations(R settings);

  /**
   * 公開したバージョンを予定で測るジョブ（卓の順）。リソースを開かずに決まるので、{@code settings change} も変更の前と後で比べられる。
   *
   * @param version 公開したバージョン
   * @param settings ジョブを作る設定（そのバージョンを公開した反復に固定した行の値）
   */
  List<JobSpec> jobs(int version, R settings);

  /**
   * リソース（推論のリソース、学習器のストリーム、裏で動く処理）を開く。リソースは JVM が動いている間 1 つだけで、段やジョブのたびには
   * 開き直さない。
   *
   * @param launch 起動のときに決まるもの（次の起動から効くキーは、起動のときの行の値を使う）
   */
  RunSession<R, S, I> open(Launch<R> launch) throws Exception;
}
