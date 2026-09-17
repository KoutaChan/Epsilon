/**
 * KL・LR制御、選択行動の方策勾配学習の実行、事前学習、勾配更新、継続学習チェックポイントを提供する。
 *
 * <p>学習器はモデル・AdamW・制御状態を一組として保存・復元する。TrainerとPretrainerが共有する
 * DataParallel、入力利用権、KL観測、収集・更新単位採否の内部契約はこのパッケージに閉じる。
 * 対局評価の実行は対戦評価、推論とGPU資源は実行時、永続サンプルの読み出しはデータへ委譲する。
 */
package com.epsilon.major.ai.decision.training;
