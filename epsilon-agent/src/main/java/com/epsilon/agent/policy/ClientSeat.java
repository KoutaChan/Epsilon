package com.epsilon.agent.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.client.Decider;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.SeatView;

/**
 * クライアントで打つ対局で、1 つの席の手を決める。判断ごとに記録 O を作り、方策の評価から貪欲に選んだ手を返す。O を作る作業領域を
 * 持つので、スレッドセーフではない。接続ごとに 1 つ作る。
 */
public final class ClientSeat implements Decider {

  private static final int INITIAL_LONGS = 1_024;

  private final Policy policy;
  private final ObservationWriter writer;
  private final ObservationBatch batch;
  private final Observation observation;

  public ClientSeat(Policy policy) {
    this.policy = policy;
    writer = new ObservationWriter(policy.variant());
    batch = new ObservationBatch(policy.variant(), INITIAL_LONGS);
    observation = new Observation(policy.variant());
  }

  @Override
  public int decide(SeatView view, Decisions decisions, int slot) {
    batch.clear();
    int index = writer.capture(view, decisions, slot, batch);
    return policy.evaluate(observation.bind(batch, index)).greedy();
  }
}
