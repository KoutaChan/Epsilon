/**
 * 各系列で共有するBeliefモデルの教師データ、出力層、損失、学習、推論、較正の処理。
 *
 * <p>入力の連結と転送は{@link
 * com.epsilon.ai.belief.BeliefBatchFactory}の実装に委ねる。転送先のメモリ管理オブジェクトがテンソルを所有する。全席の手牌は教師生成にだけ使い、オンライン推論には公開観測だけを渡す。
 */
package com.epsilon.ai.belief;
