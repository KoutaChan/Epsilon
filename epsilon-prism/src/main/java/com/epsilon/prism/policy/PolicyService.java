package com.epsilon.prism.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.play.TableSeat;

/**
 * あるバージョンの方策で卓の席に座る打ち手。どの方策も、問い合わせに入った O だけを読む。同じバージョンなら、呼び出しの順や同時実行に
 * 依らず、同じ入力に同じ σ を返す。
 */
public interface PolicyService extends TableSeat<PolicyRequest> {
  int version();

  /**
   * 既定では、入力の大きさで分けた受付の区分（{@link Capacity}）を返す。このため、チーかポンが合法な判断は別のバッチになる。
   */
  @Override
  default int admission(Observation observation) {
    return Capacity.of(observation).ordinal();
  }
}
