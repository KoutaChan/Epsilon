/**
 * 共通の牌譜再生から、この系列のDecision・Belief特徴量と学習教師を生成する。
 *
 * <p>天鳳XML・mjai/mjsonの符号化・復号処理、合法手検証、対局進行はepsilon-commonが所有する。
 * 系列側は判断時点の状態をコールバック中だけ借用し、独立した入力バッチと終局後の教師を構築する。
 * オフラインのBelief教師生成では非公開牌配置も利用する。対戦用推論へ他家手牌を公開する入口ではない。
 */
package com.epsilon.nano.training.replay;
