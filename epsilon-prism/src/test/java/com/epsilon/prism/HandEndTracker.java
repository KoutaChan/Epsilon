package com.epsilon.prism;

import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.EventKind;
import com.epsilon.mahjong.engine.EventLog;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.HandEndAnalyzer;
import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.TenpaiReach;

/**
 * テストの卓で打牌ごとに聴牌への到達を記録し、局末の事実を作る。判断の時点と局末に {@link #scan} を呼ぶと、前回呼んだ後に増えた打牌を
 * イベントの順に到達の記録へ渡す。打牌した席はまだ次のツモをしていないので、手牌は打牌した後の形のままになっている。
 */
public final class HandEndTracker {

  private final TenpaiReach reach;
  private int seen;

  public HandEndTracker(Variant variant) {
    reach = new TenpaiReach(variant);
  }

  /** この局で聴牌に届いたかの記録を返す。呼び出し元は書き換えない。 */
  public TenpaiReach reach() {
    return reach;
  }

  /**
   * 前回呼んだ後に増えた打牌ごとに、聴牌への到達の記録を進める。
   *
   * @param full 判断の時点か局末の局面
   */
  public void scan(FullView full) {
    EventLog events = full.seat(0).events();
    for (; seen < events.size(); seen++) {
      if (events.kind(seen) == EventKind.DISCARD) {
        reach.discarded(full, events.seat(seen), seen);
      }
    }
  }

  /**
   * 局末の事実を作り、次の局に備えて到達の記録を消す。
   *
   * @param engine 局末のエンジン
   * @return 局末の事実
   */
  public HandEndFacts end(Engine engine) {
    scan(engine.full());
    HandEndFacts facts = HandEndAnalyzer.analyze(engine.full(), engine.roundEnd(), reach);
    reach.reset();
    seen = 0;
    return facts;
  }
}
