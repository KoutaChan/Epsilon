package com.epsilon.reviewer.engine.review;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.mahjong.rules.Variant;

/**
 * 判断を評価するモデル。model パッケージの {@code ModelLease} が実装する。このインターフェースを挟むので、engine パッケージは
 * model パッケージに依存しない。
 */
public interface ReviewModel {
  /** 1 つの判断を評価する。渡す O は呼び出しの間だけ有効。 */
  Evaluation evaluate(Observation observation);

  /** 評価が手の価値を持つか。方策を呼ばずに候補を作るとき、値を損失にするか確率にするかもこれで決まる。 */
  boolean valued();

  /** 価値から表示する確率を作るときの温度（正の値）。 */
  double temperature();

  /** モデルが対応する牌譜のルール（四人打ちか三人打ち）。 */
  Variant variant();
}
