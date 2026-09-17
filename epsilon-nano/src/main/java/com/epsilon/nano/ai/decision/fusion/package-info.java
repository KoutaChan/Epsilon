/**
 * Decision推論のホスト転送境界で利用するFusion実装です。
 *
 * <p>このパッケージはdjl-rocmのFusion 実行計画、セッション、出力利用権の寿命を所有します。学習対象のブロックや通常の
 * DJL演算は親パッケージに置き、推論時だけ必要なネイティブ実行資源と分離します。
 */
package com.epsilon.nano.ai.decision.fusion;
