package com.epsilon.spi;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import java.util.List;

/** 牌譜の公開観測から全合法候補の方策確率を同期で計算する。 */
public interface ReviewPolicy extends AutoCloseable {
  /** 観測と合法手は呼出中だけ借用し、返却配列の所有権を呼び出し元へ渡す。 */
  float[] probabilities(PublicObservation observation, List<Action> legalActions);

  @Override
  void close();
}
