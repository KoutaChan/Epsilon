package com.epsilon.agent.play;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.SeatView;

/** 系列の処理を持たない卓の処理（{@link TableHooks#plain}）。状態を持たないので、全ての経路で同じインスタンスを使う。 */
final class PlainHooks implements TableHooks<Query>, RunHooks<Query> {

  static final PlainHooks HOOKS = new PlainHooks();

  private PlainHooks() {}

  @Override
  public RunHooks<Query> main(Run<Query> run) {
    return this;
  }

  @Override
  public Query query(
      Run<Query> run,
      Decisions decisions,
      int slot,
      SeatView view,
      int admission,
      ObservationBatch records,
      int record,
      Observation observation) {
    return new Query(run, run.game().wall(), slot, admission, records, record, observation);
  }
}
