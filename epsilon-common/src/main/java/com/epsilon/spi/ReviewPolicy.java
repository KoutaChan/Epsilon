package com.epsilon.spi;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import java.util.List;

/** 牌譜の公開観測から 1 判断を同期で評価する（全合法手の表示用の確率と、価値を推定するモデルなら手の価値）。 */
public interface ReviewPolicy extends AutoCloseable {
  /** 観測と合法手は呼出中だけ借用する。 */
  ReviewEvaluation evaluate(PublicObservation observation, List<Action> legalActions);

  @Override
  void close();
}
