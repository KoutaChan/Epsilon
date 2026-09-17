/**
 * Epsilon CLI の全コマンドを共通ライフサイクルで実行するワークフロー層。
 *
 * <p>ワークフローはコマンド定義、引数検証、型変換、資源有効期間、完了要約を所有し、麻雀・学習・推論の計算は各 domain クラスへ委譲する。
 */
package com.epsilon.pico.workflow;
