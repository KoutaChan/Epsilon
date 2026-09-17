/**
 * 凍結済みDecision 方策をDJL Fusion 実行計画として実行する実装を提供する。
 *
 * <p>このパッケージは、学習ブロックが所有するパラメーターを参照して実行計画、セッション、永続出力利用権を構築し、推論パイプラインの
 * 実行枠寿命に合わせてそれらを管理する。学習可能なブロック、EAGER経路の数式、および方策グラフの意味構造は {@code
 * com.epsilon.major.ai.decision.policy} に残し、Fusion固有の実行資源だけをここへ隔離する。
 */
package com.epsilon.major.ai.decision.policy.fusion;
