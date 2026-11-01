package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.metrics.StageTimer;

/**
 * 1 つの経路（{@link GameRunner}）が prepare で数える判断の件数。経路を今進めている 1 本のスレッドだけが書く（経路は境目で別の進行
 * スレッドへ移るが、同時に書くのはいつも 1 本）。{@link GameRunner#advance} の終わりに、段の時間（{@link
 * StageTimer}）と一緒に {@link #flush} で {@link CollectionMetrics} へまとめて足す。判断ごとに共有の
 * LongAdder を触らないため。
 *
 * <p>役割は経路の生涯で変わらないので作るときに束ねる。類ごとの件数は主経路だけが metrics に持つ（mainDecisions）ので、類ごとの配列は
 * 主経路だけが持つ。写し・分岐は合計（copyDecisions・forkDecisions）と強制手だけを数え、生まれるたびに配列を作らない。
 */
final class DecisionCounts {
  private final Role role;

  /** 主経路の類ごとの NN 推論判断（{@link DecisionClass} の番号で引く）。写し・分岐は null。 */
  private final long[] mainByClass;

  /** NN 推論判断（合法手 2 つ以上）の合計。 */
  private long model;

  /** 合法手 1 つの強制手。 */
  private long forced;

  DecisionCounts(Role role) {
    this.role = role;
    mainByClass = role == Role.MAIN ? new long[DecisionClass.COUNT] : null;
  }

  void countForced() {
    forced++;
  }

  void countModel(DecisionClass decisionClass) {
    model++;
    if (mainByClass != null) mainByClass[decisionClass.ordinal()]++;
  }

  /** 数えた件数を metrics に足して 0 に戻す。 */
  void flush(CollectionMetrics metrics) {
    if (forced != 0) {
      metrics.forcedDecisions.add(forced);
      forced = 0;
    }
    if (model == 0) return;
    metrics.modelDecisions.add(model);
    switch (role) {
      case MAIN -> flushMainByClass(metrics);
      case COPY -> metrics.copyDecisions.add(model);
      case FORK -> metrics.forkDecisions.add(model);
    }
    model = 0;
  }

  private void flushMainByClass(CollectionMetrics metrics) {
    for (int c = 0; c < mainByClass.length; c++) {
      long count = mainByClass[c];
      if (count == 0) continue;
      metrics.mainDecisions[c].add(count);
      mainByClass[c] = 0;
    }
  }
}
