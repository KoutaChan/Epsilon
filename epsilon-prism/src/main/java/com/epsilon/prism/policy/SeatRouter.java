package com.epsilon.prism.policy;

import com.epsilon.prism.eval.HybridSeat;

/**
 * 席に置くと、判断ごとに実際に問い合わせる方策を選ぶ打ち手（評価の混成の席だけ。{@link HybridSeat}）。対局は {@link
 * #route} が返した方策に問い合わせ、この打ち手そのものには問い合わせない。
 */
public interface SeatRouter {
  /** この判断を打つ方策。 */
  PolicyService route(PolicyRequest request);
}
