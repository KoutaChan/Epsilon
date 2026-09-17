package com.epsilon.workflow;

/**
 * Epsilon CLI の全コマンドに共通する実行手順を定めるインターフェース。
 *
 * @param <R> 処理が生成する型付き結果
 */
public interface CommandWorkflow<R> {

  /** コマンド名、分類、引数個数、使用方法を返す。 */
  WorkflowDefinition definition();

  /** 検証済みのコマンド引数を型付き入力へ変換し、対象処理を実行する。 */
  R execute(WorkflowArguments arguments) throws Exception;

  /** 共通完了ログへ追加する短い結果要約を返す。 */
  String summarize(R result);

  /** CLI の終了成否を返す。検証結果や学習完了状態を終了コードへ反映する場合だけ上書きする。 */
  default boolean successful(R result) {
    return true;
  }
}
