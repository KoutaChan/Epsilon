package com.epsilon.reviewer.engine.review;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.spi.ReviewEvaluation;
import java.util.List;

/** 判断を評価するモデル。model の貸し出しが実装する（engine は model に依らない）。 */
public interface ReviewModel {
  /** 1 判断を評価する。観測と合法手は呼出中だけ貸す。 */
  ReviewEvaluation evaluate(PublicObservation observation, List<Action> legal);

  /** 価値の評価から表示の確率を作る温度（正）。 */
  double temperature();
}
