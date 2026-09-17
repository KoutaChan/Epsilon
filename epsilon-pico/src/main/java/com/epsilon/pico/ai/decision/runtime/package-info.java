/**
 * モデル推論、バッチ実行、GPUストリームの所有権、計測を提供する。
 *
 * <p>ワークフローがExecutionContextを所有し、推論・学習・評価が同じGPU資源を借用する。 推論パイプラインの利用権・位置・投入 規約はパッケージ内部に保持する。
 */
package com.epsilon.pico.ai.decision.runtime;
